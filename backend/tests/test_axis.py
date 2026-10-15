import numpy as np, trimesh
from app.geometry.axis import suggest_insertion_axis

def test_axis_returns_unit():
    m=trimesh.creation.cylinder(radius=2,height=8)
    a=suggest_insertion_axis(m)
    assert abs(np.linalg.norm(a)-1)<1e-7
