import trimesh
import pytest
from app.geometry.io import load_mesh
from app.geometry.margin import suggest_margin_from_open_boundary

def test_open_boundary():
    box=trimesh.creation.box([5,5,5])
    # remove top faces to create an open boundary
    z=box.triangles_center[:,2]
    keep=z < z.max()-1e-6
    m=box.submesh([keep.nonzero()[0]],append=True,repair=False)
    pts=suggest_margin_from_open_boundary(m)
    assert len(pts)>=4

def test_closed_stl_has_no_margin(tmp_path):
    path=tmp_path/"closed.stl"
    trimesh.creation.box([5,5,5]).export(path)
    mesh=load_mesh(path)
    assert mesh.is_watertight
    assert len(mesh.split(only_watertight=False))==1
    with pytest.raises(ValueError, match="没有可检测的开放边界"):
        suggest_margin_from_open_boundary(mesh)
