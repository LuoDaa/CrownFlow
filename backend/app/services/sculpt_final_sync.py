"""Propagate Step-5 outer-surface edits into the already complete crown.

Crown 1.0 has one product flow: generation creates outer + intaglio + cervical
stitch before sculpting; sculpting must therefore not invalidate or regenerate
the intaglio.  This module transfers the accepted displacement field from the
editable outer mesh onto the external vertices of the already-watertight final
mesh while keeping the intaglio and cervical seam fixed.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import trimesh
from trimesh.proximity import closest_point

from ..geometry.io import load_mesh
from ..geometry.nonplanar_crown import MarginCurtain
from ..geometry.crown_outer.deform_gate import deformation_gate


class FinalSurfaceSyncError(ValueError):
    pass


def _closest(mesh, points):
    cps=[]; distances=[]; faces=[]
    for chunk in np.array_split(np.asarray(points,float), max(1, len(points)//1024+1)):
        cp,d,fid=closest_point(mesh,chunk)
        cps.append(cp);distances.append(d);faces.append(fid)
    return np.vstack(cps),np.concatenate(distances),np.concatenate(faces).astype(np.int64)


def _transfer(source_before, source_after, points):
    """Barycentrically transfer a topology-preserving outer displacement field."""
    if (len(source_before.vertices)!=len(source_after.vertices)
            or not np.array_equal(np.asarray(source_before.faces),np.asarray(source_after.faces))):
        raise FinalSurfaceSyncError('微调前后外冠拓扑不一致，不能直接同步完整冠')
    cp,distance,fids=_closest(source_before,points)
    faces=np.asarray(source_before.faces,np.int64)[fids]
    triangles=np.asarray(source_before.vertices,float)[faces]
    bary=trimesh.triangles.points_to_barycentric(triangles,cp)
    bary=np.clip(bary,0.0,1.0)
    denom=np.maximum(bary.sum(axis=1,keepdims=True),1e-12)
    bary/=denom
    delta=np.asarray(source_after.vertices,float)-np.asarray(source_before.vertices,float)
    displacement=np.sum(delta[faces]*bary[:,:,None],axis=1)
    return displacement,np.asarray(distance,float)


def _artifact_mesh(design, name):
    qc=json.loads(design.get('qc_json') or '{}')
    path=(qc.get('artifact_paths') or {}).get(name)
    if not path or not Path(path).is_file():
        raise FinalSurfaceSyncError(f'完整冠缺少 {name} 参考面；请重新一键生成')
    return load_mesh(Path(path)),Path(path)


def _external_mask(final_mesh, outer_surface, intaglio_surface):
    """Select external final vertices; shared cervical/seam vertices stay fixed."""
    points=np.asarray(final_mesh.vertices,float)
    _co,do,_fo=_closest(outer_surface,points)
    _ci,di,_fi=_closest(intaglio_surface,points)
    # STL round-trip error is normally microns.  A generous 0.02-mm surface
    # tolerance keeps the true external skin, while a 0.03-mm inner separation
    # explicitly removes shared seam/knife-edge vertices from the editable set.
    external=(do<=0.02)&((di-do)>=0.03)
    return external,do,di


def _cervical_taper(points, margin, frame, transition_mm):
    height=MarginCurtain(margin,frame.z).axial_above(points)
    lock_top=max(float(transition_mm)+0.30,0.65)
    free_top=lock_top+0.85
    t=np.clip((height-lock_top)/max(free_top-lock_top,1e-6),0.0,1.0)
    return t*t*(3.0-2.0*t),height,lock_top,free_top


def sync_complete_final(design, before_outer, after_outer, frame):
    """Return updated final + audit while preserving intaglio and cervical seam.

    No Boolean, blockout, cement-gap, or stitch algorithm is rerun here.  The
    generated cutter/intaglio remains byte-identical.  Only external final-surface
    vertices that can be proven to belong to the old outer skin receive the same
    displacement field as the Step-5 edit, with a smooth zero displacement band
    above the cervical stitch.
    """
    final_path=design.get('final_path');cutter_path=design.get('cutter_path')
    if not final_path or not cutter_path or not Path(final_path).is_file() or not Path(cutter_path).is_file():
        raise FinalSurfaceSyncError('微调前必须已有完整 Final Crown 与内冠')
    final_mesh=load_mesh(Path(final_path))
    outer_surface,outer_surface_path=_artifact_mesh(design,'outer_surface')
    intaglio_surface,_intaglio_path=_artifact_mesh(design,'intaglio_surface')
    external,outer_distance,inner_distance=_external_mask(final_mesh,outer_surface,intaglio_surface)
    if int(external.sum())<64:
        raise FinalSurfaceSyncError('无法可靠识别完整冠外表面；请重新一键生成')

    points=np.asarray(final_mesh.vertices,float)
    displacement,source_distance=_transfer(before_outer,after_outer,points)
    parameters=json.loads(design.get('parameters_json') or '{}')
    margin=np.asarray(json.loads(design.get('margin_json') or 'null'),float)
    transition=float(parameters.get('margin_band_mm') or 0.40)
    taper,height,lock_top,free_top=_cervical_taper(points,margin,frame,transition)
    editable=external&(source_distance<=0.35)&(taper>0.0)
    applied=displacement*taper[:,None]
    applied[~editable]=0.0

    before=np.asarray(final_mesh.vertices,float).copy()
    after=before+applied
    moved=np.linalg.norm(applied,axis=1)
    locked=np.flatnonzero(~editable)
    focus=np.flatnonzero(np.any((moved>1e-9)[np.asarray(final_mesh.faces,np.int64)],axis=1))
    gate=deformation_gate(before,after,np.asarray(final_mesh.faces,np.int64),scale_mm=.1,
                          locked_ids=locked,locked_tol=1e-10,focus_faces=focus)
    trial=final_mesh.copy();trial.vertices=after
    flips=int((np.einsum('ij,ij->i',final_mesh.face_normals,trial.face_normals)<=0.0).sum())
    if (not gate.ok or flips or np.any(np.asarray(trial.area_faces)<1e-10)
            or not trial.is_watertight or not trial.is_winding_consistent):
        raise FinalSurfaceSyncError('微调会破坏完整冠闭合/拓扑，已回滚')

    # Keep an updated external audit surface in sync as well.  It is not used to
    # rebuild the crown; it remains a provenance/QC surface for the exported final.
    os_points=np.asarray(outer_surface.vertices,float)
    os_disp,os_source_distance=_transfer(before_outer,after_outer,os_points)
    os_taper,_h,_a,_b=_cervical_taper(os_points,margin,frame,transition)
    os_apply=os_disp*os_taper[:,None]
    os_apply[os_source_distance>0.35]=0.0
    updated_outer_surface=outer_surface.copy()
    updated_outer_surface.vertices=os_points+os_apply

    return trial,updated_outer_surface,{
        'status':'FINAL_EXTERNAL_SURFACE_SYNCED',
        'method':'BARYCENTRIC_OUTER_DISPLACEMENT_ON_WATERTIGHT_FINAL',
        'final_vertex_count':int(len(points)),
        'external_candidate_vertices':int(external.sum()),
        'edited_final_vertices':int(np.count_nonzero(moved>1e-9)),
        'max_final_displacement_mm':float(moved.max(initial=0.0)),
        'p95_final_displacement_mm':float(np.quantile(moved,.95)) if len(moved) else 0.0,
        'cervical_lock_top_mm':float(lock_top),
        'cervical_free_top_mm':float(free_top),
        'outer_surface_reference':str(outer_surface_path),
        'outer_surface_distance_p95_mm':float(np.quantile(outer_distance[external],.95)),
        'inner_surface_distance_min_external_mm':float(inner_distance[external].min()),
        'intaglio_unchanged':True,
        'cervical_stitch_topology_unchanged':True,
        'watertight':bool(trial.is_watertight),
        'winding_consistent':bool(trial.is_winding_consistent),
        'deformation_gate':gate.as_dict(),
        'flipped_faces':flips,
    }
