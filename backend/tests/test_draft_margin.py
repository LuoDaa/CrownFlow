import numpy as np
import pytest
import trimesh
from app.geometry.draft_margin import draft_margin

def test_draft_section_and_rigid_transform():
    mesh=trimesh.creation.cylinder(radius=4,height=8,sections=96)
    ring,axis=draft_margin(mesh,[0,0,4],[0,0,1],7)
    assert len(ring)==64
    assert np.allclose(np.asarray(ring)[:,2],1)
    assert np.max(np.abs(np.linalg.norm(np.asarray(ring)[:,:2],axis=1)-4))<.02
    rotation=trimesh.transformations.rotation_matrix(.8,[1,2,3]);rotation[:3,3]=[17,-4,11]
    mesh.apply_transform(rotation)
    center=trimesh.transform_points([[0,0,4]],rotation)[0]
    ring2,axis2=draft_margin(mesh,center,rotation[:3,:3]@[0,0,1],7)
    local=trimesh.transform_points(ring2,np.linalg.inv(rotation))
    assert np.allclose(local[:,2],1)
    assert np.max(np.abs(np.linalg.norm(local[:,:2],axis=1)-4))<.02

def test_does_not_use_whole_jaw_or_missing_section():
    mesh=trimesh.creation.cylinder(radius=20,height=8,sections=32)
    with pytest.raises(ValueError,match='单牙闭环'):draft_margin(mesh,[0,0,4],[0,0,1],7)
    with pytest.raises(ValueError):draft_margin(mesh,[0,0,40],[0,0,1],7)
    with pytest.raises(ValueError):draft_margin(mesh,[0,0,4],[0,0,0],7)
