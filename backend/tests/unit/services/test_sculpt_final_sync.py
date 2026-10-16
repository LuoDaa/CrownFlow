import json
from pathlib import Path

import numpy as np
import trimesh

from app.geometry.io import load_mesh,save_mesh
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.services.sculpt_final_sync import sync_complete_final


def _frame():
    return confirmed_patient_frame([0,0,0],[1,0,0],[0,1,0],[0,0,1])


def _margin(z=-3.0, r=3.0, n=96):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.c_[r*np.cos(t),r*np.sin(t),np.full(n,z)]


def test_step5_outer_edit_keeps_complete_final_watertight_and_intaglio_unchanged(tmp_path):
    final=trimesh.creation.cylinder(radius=3.0,height=6.0,sections=96)
    outer_surface=final.copy()
    intaglio=trimesh.creation.cylinder(radius=2.0,height=5.0,sections=96)
    before=final.copy()
    after=before.copy()
    vertices=np.asarray(after.vertices,float).copy()
    editable=(vertices[:,0]>0.5)&(vertices[:,2]>0.8)
    vertices[editable,0]+=0.08
    after.vertices=vertices

    final_path=tmp_path/'final.stl';cutter_path=tmp_path/'cutter.stl'
    outer_path=tmp_path/'outer_surface.stl';inner_path=tmp_path/'intaglio_surface.stl'
    save_mesh(final,final_path);save_mesh(intaglio,cutter_path)
    save_mesh(outer_surface,outer_path);save_mesh(intaglio,inner_path)
    # The production helper reads the persisted STL.  STL round-trip may reorder
    # vertices/faces, so topology preservation must be compared against that exact
    # persisted baseline rather than the pre-export trimesh indexing.
    persisted_final=load_mesh(final_path)
    cutter_before=cutter_path.read_bytes()
    design={
        'final_path':str(final_path),'cutter_path':str(cutter_path),
        'margin_json':json.dumps(_margin().tolist()),
        'parameters_json':json.dumps({'margin_band_mm':0.40}),
        'qc_json':json.dumps({'artifact_paths':{
            'outer_surface':str(outer_path),'intaglio_surface':str(inner_path)}}),
    }

    synced,updated_outer,report=sync_complete_final(design,before,after,_frame())

    assert report['status']=='FINAL_EXTERNAL_SURFACE_SYNCED',report
    assert report['intaglio_unchanged'] is True
    assert report['cervical_stitch_topology_unchanged'] is True
    assert synced.is_watertight and synced.is_winding_consistent
    assert np.array_equal(np.asarray(synced.faces),np.asarray(persisted_final.faces))
    assert cutter_path.read_bytes()==cutter_before
    assert report['edited_final_vertices']>0
    # Cervical band is immutable; the outer displacement only propagates above it.
    baseline=np.asarray(persisted_final.vertices,float)
    low=baseline[:,2]<=-1.5
    assert np.allclose(np.asarray(synced.vertices)[low],baseline[low],atol=1e-10)
    assert np.max(np.linalg.norm(np.asarray(synced.vertices)-baseline,axis=1))<=0.081
    assert np.isfinite(np.asarray(updated_outer.vertices)).all()
