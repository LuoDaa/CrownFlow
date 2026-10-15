import numpy as np
import pytest
import trimesh
from app.geometry.nonplanar_crown import build_nonplanar_crown, MarginBuildError
from app.geometry.minimum_thickness import enforce_body_thickness
from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from .test_existing_abutment import api_client


def test_batched_normal_rays_preserve_first_hit_distances():
    from app.geometry.manufacturing_qc import _normal_ray_distances
    mesh=trimesh.creation.box(extents=[4,4,4])
    points=np.zeros((257,3));directions=np.tile([0,0,1.],(257,1))
    values=_normal_ray_distances(mesh,points,directions)
    hits,ids,_=mesh.ray.intersects_location(points+directions*1e-7,directions,multiple_hits=False)
    expected=np.full(len(points),np.nan);expected[ids]=np.linalg.norm(hits-points[ids],axis=1)
    np.testing.assert_allclose(values,expected,equal_nan=True)
    assert np.isnan(_normal_ray_distances(mesh,points+[0,0,3],directions)).all()


def body(radius):
    mesh=trimesh.creation.cylinder(radius=radius,height=4,sections=32)
    mesh.apply_translation([0,0,3])
    return mesh.submesh([np.flatnonzero(mesh.face_normals[:,2]>-.5)],append=True,repair=False)


def test_outward_thickness_preserves_inner_and_locked_contact():
    inside=body(3);inside.invert();outside=body(3.4)
    outside.vertices[outside.vertices[:,2]>3,2]+=.4
    original=inside.vertices.copy()
    repaired,report=enforce_body_thickness(outside,inside,.8)
    assert report['after']['p05_mm']>report['before']['p05_mm']+.1
    assert np.array_equal(original,inside.vertices)
    assert report['max_displacement_mm']>0
    obstacle=FitSurface(trimesh.creation.box(extents=[2,10,10]).apply_translation([4.5,0,3]))
    fixed=outside.vertices[obstacle.query(outside.vertices)[2]<=.2]
    repaired,report=enforce_body_thickness(outside,inside,.8,protected_surfaces=[(obstacle,0)])
    assert report['protected_vertex_count']>0
    for point in fixed:assert np.min(np.linalg.norm(repaired.vertices-point,axis=1))<1e-12
    assert report['status']=='REVIEW_REQUIRED'


def test_parametric_border_closed_preserves_confirmed_margin():
    t=np.arange(48)*2*np.pi/48
    margin=np.c_[3*np.cos(t),3*np.sin(t),.05*np.sin(2*t)]
    support=trimesh.creation.cylinder(radius=3,height=6,sections=48)
    cutter=trimesh.creation.cylinder(radius=3.06,height=7,sections=64).apply_translation([0,0,.4])
    outer=trimesh.creation.cylinder(radius=4.2,height=8,sections=64).apply_translation([0,0,1])
    params={'border_horizontal_mm':.25,'border_angled_mm':.6,'border_angle_deg':45,'border_vertical_mm':.1,
        'enforce_minimum_thickness':True,'minimum_thickness_mm':.8}
    result=build_nonplanar_crown(outer,cutter,support,margin,[0,0,1],transition_mm=1.2,parameters=params)
    assert result.final.is_watertight
    assert result.report['crown_border']['horizontal_land_min_mm']>=.25
    assert result.report['seam']['shared_seam_edges_incidence_two']
    assert result.report['seam']['margin_to_seam_max_mm']<1e-10
    with pytest.raises(MarginBuildError,match='截面高度'):
        build_nonplanar_crown(outer,cutter,support,margin,[0,0,1],transition_mm=.2,parameters=params)


def test_real_abutment_manufacturing_border(api_client,tmp_path):
    from .test_existing_abutment import run_real_abutment
    run_real_abutment(api_client,tmp_path,manufacturing_border=True)


def test_profile_rejects_impossible_border_and_marks_final_stale(api_client):
    from .test_existing_abutment import create_confirmed_design,fixture_support
    mesh,margin=fixture_support();client=api_client;design=create_confirmed_design(client,mesh,margin)
    base=f'/api/designs/{design}';before=client.get(base+'/status').json()['input_digest']
    request={'material':'engineering fixture','coverage':'ENGINEERING_ONLY','confirmed':True,'bite_confirmed':True,
        'parameters':{'border_horizontal_mm':.25,'border_angled_mm':1.9,'margin_band_mm':.8}}
    assert client.put(base+'/profile',json=request).status_code==422
    assert client.get(base+'/status').json()['input_digest']==before
    request['parameters'].update(border_angled_mm=.6,enforce_minimum_thickness=True)
    assert client.put(base+'/profile',json=request).status_code==200
    assert client.get(base+'/status').json()['input_digest']!=before
