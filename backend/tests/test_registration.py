import numpy as np
import trimesh
from app.geometry.registration import kabsch, initialize_from_margin

def test_rigid_translation():
    a=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    b=a+np.array([5.,-2.,3.])
    m=kabsch(a,b)
    out=(m@np.c_[a,np.ones(len(a))].T).T[:,:3]
    assert np.allclose(out,b,atol=1e-7)

def test_initialize_from_margin_places_cervical_extent_on_plane():
    template=trimesh.creation.box(extents=[8,8,6])
    margin=[[20,10,4],[24,10,4],[24,14,4],[20,14,4]]
    outer=initialize_from_margin(template,margin,[0,0,1])
    assert np.allclose(outer.bounds[0,2],4)
    assert np.allclose(outer.bounds.mean(axis=0)[:2],[22,12])
