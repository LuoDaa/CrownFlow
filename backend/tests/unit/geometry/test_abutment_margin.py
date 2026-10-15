import numpy as np
import pytest
import trimesh
from app.geometry.abutment_margin import extract_abutment_margin


def flared_support():
    return trimesh.creation.revolve(np.array([[0,-4],[1.3,-4],[3,0],[2.2,0],[2.2,4],[0,4]]), sections=128)


def test_outer_shoulder_not_cap_or_inner_corner_and_mesh_unchanged():
    mesh = flared_support(); before = mesh.vertices.copy()
    points, quality = extract_abutment_margin(mesh, [0,0,1])
    points = np.asarray(points)
    assert len(points) == 128
    assert np.max(np.abs(points[:,2])) < 1e-8
    assert np.linalg.norm(points[:,:2],axis=1).min() > 2.99
    assert quality['status'] == 'REVIEW_REQUIRED'
    assert np.array_equal(mesh.vertices,before)


def test_no_shoulder_is_rejected_instead_of_generating_plane():
    with pytest.raises(ValueError,match='肩部外缘'):
        extract_abutment_margin(trimesh.creation.cylinder(radius=2,height=8,sections=128),[0,0,1])


def test_rotated_translated_support_keeps_world_coordinates():
    mesh=flared_support()
    transform=trimesh.transformations.rotation_matrix(.7,[1,2,3]);transform[:3,3]=[15,-7,20]
    mesh.apply_transform(transform)
    points,_=extract_abutment_margin(mesh,transform[:3,:3]@[0,0,1])
    local=trimesh.transform_points(np.asarray(points),np.linalg.inv(transform))
    assert np.abs(local[:,2]).max()<1e-8
    assert np.linalg.norm(local[:,:2],axis=1).min()>2.99


def test_shoulder_anchor_on_rotated_ring_is_written_only_once():
    from app.geometry.margin_edit import retrace_with_anchors
    mesh=flared_support();points,_=extract_abutment_margin(mesh,[0,0,1])
    anchor=np.array([2.6,0,0])
    result=retrace_with_anchors(mesh,points,[anchor],axis=[0,0,1])
    edited=np.asarray(result['points'])
    assert np.count_nonzero(np.linalg.norm(edited-anchor,axis=1)<1e-6)==1
    assert np.linalg.norm(edited[0]-np.asarray(points)[0])<1e-8
