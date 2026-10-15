import numpy as np
import pytest
import trimesh
from app.geometry.nonplanar_crown import build_nonplanar_crown, MarginCurtain, MarginBuildError


@pytest.mark.parametrize('amplitude',[0,.6])
def test_boolean_crown_uses_3d_margin_and_preserves_body(amplitude,tmp_path):
    theta=np.arange(48)*2*np.pi/48
    margin=np.c_[3*np.cos(theta),3*np.sin(theta),amplitude*np.sin(2*theta)]
    support=trimesh.creation.cylinder(radius=3,height=5.5,sections=48).apply_translation([0,0,.75])
    cutter=trimesh.creation.cylinder(radius=3.08,height=7,sections=48).apply_translation([0,0,.04])
    outer=trimesh.creation.cylinder(radius=4.2,height=7,sections=64).apply_translation([0,0,1.5])
    result=build_nonplanar_crown(outer,cutter,support,margin,[0,0,1],transition_mm=.8)
    assert result.final.is_watertight
    assert result.report['seam']['margin_to_seam_max_mm']<1e-12
    assert result.report['seam']['seam_to_margin_max_mm']<1e-12
    from trimesh.proximity import closest_point
    vertices=np.asarray(result.outer_surface.vertices)
    body=vertices[vertices[:,2]>1.41]
    assert closest_point(outer,body)[1].max()<1e-6
    assert len(result.outer_surface.faces)>0 and len(result.intaglio_surface.faces)>0
    from app.geometry.margin_seam import audit_margin_seam
    from scipy.spatial import cKDTree
    path=tmp_path/'final.stl';result.final.export(path)
    persisted=trimesh.load(path,force='mesh')
    ids=cKDTree(persisted.vertices).query(margin)[1]
    report=audit_margin_seam(persisted.vertices,persisted.faces,ids,margin)
    assert report['shared_seam_edges_incidence_two']
    assert report['margin_to_seam_max_mm']<1e-6
    assert report['seam_to_margin_max_mm']<1e-6


def test_actual_thin_wall_is_detected_independent_of_watertightness():
    from types import SimpleNamespace
    from app.geometry.manufacturing_qc import audit_manufacturing
    theta=np.arange(48)*2*np.pi/48
    margin=np.c_[3*np.cos(theta),3*np.sin(theta),np.zeros(48)]
    support=trimesh.creation.cylinder(radius=3,height=5,sections=48).apply_translation([0,0,.5])
    cutter=trimesh.creation.cylinder(radius=3.08,height=7,sections=48)
    outer=trimesh.creation.cylinder(radius=3.18,height=7,sections=48).apply_translation([0,0,1])
    result=build_nonplanar_crown(outer,cutter,support,margin,[0,0,1],transition_mm=.8)
    qc,heatmap=audit_manufacturing(result.final,result.outer_surface,result.intaglio_surface,support,
        SimpleNamespace(blockout_mesh=support,diagnostics={'resolution_mm':.1}),margin,[0,0,1],
        {'minimum_thickness_mm':.6,'cement_gap_mm':.08,'cement_gap_start_mm':.5},result.report['seam']['geometry'])
    assert result.final.is_watertight
    assert qc['FinalCrownThicknessAudit']['thin_face_sample_area_mm2']>1
    assert heatmap['thin_vertex_ids']
    assert qc['FinalCrownThicknessAudit']['conservative']['p05_mm']<.11
    assert qc['production_status']=='NOT_PRODUCTION_APPROVED'


def test_nonplanar_edit_locks_high_cervical_region():
    from app.api.designer import Edit,edit_mesh
    theta=np.arange(48)*2*np.pi/48
    margin=np.c_[3*np.cos(theta),3*np.sin(theta),.6*np.sin(2*theta)]
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=4).apply_translation([0,0,3])
    locked=MarginCurtain(margin,[0,0,1]).axial_above(mesh.vertices)<=.4
    result=edit_mesh(mesh,margin,[0,0,1],Edit(operation='transform',scale=[1.01,1,1],base_outer_path='fixture'))
    assert np.array_equal(mesh.vertices[locked],result.vertices[locked])


def test_clip_exact_level_has_no_duplicate_vertices():
    from app.geometry.crown_outer.collar_body_outer import _clip_by_scalar
    mesh=trimesh.creation.icosphere(subdivisions=1)
    vertices,faces,_=_clip_by_scalar(mesh.vertices,mesh.faces,-mesh.vertices[:,2])
    assert np.all(np.diff(np.sort(faces,axis=1),axis=1)>0)
    assert len(np.unique(vertices,axis=0))==len(vertices)


def test_crossed_margin_is_rejected():
    with pytest.raises(MarginBuildError,match='自交'):
        MarginCurtain([[-1,-1,0],[1,1,0],[-1,1,0],[1,-1,0]],[0,0,1])


def test_scan_gap_is_not_automatically_bridged():
    theta=np.arange(48)*2*np.pi/48
    margin=np.c_[3*np.cos(theta),3*np.sin(theta),np.zeros(48)]
    support=trimesh.creation.cylinder(radius=3,height=5.5,sections=48).apply_translation([0,0,.75])
    keep=np.asarray(support.triangles_center)[:,0]<2.8
    damaged=support.submesh([np.flatnonzero(keep)],append=True,repair=False)
    cutter=trimesh.creation.cylinder(radius=3.08,height=7,sections=48)
    outer=trimesh.creation.cylinder(radius=4.2,height=7,sections=64)
    with pytest.raises(MarginBuildError,match='缺口'):
        build_nonplanar_crown(outer,cutter,damaged,margin,[0,0,1],transition_mm=.8)
