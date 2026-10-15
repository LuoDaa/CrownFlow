import numpy as np

from app.geometry.crown_shape_quality import deformation_shape_report, shape_gate, triangle_quality


def test_triangle_quality_equilateral_is_one():
    v=np.array([[0.,0.,0.],[1.,0.,0.],[.5,np.sqrt(3.)/2.,0.]])
    q=triangle_quality(v,np.array([[0,1,2]]))
    assert np.isclose(q[0],1.0,atol=1e-12)


def test_shape_gate_rejects_obvious_long_spike():
    v0=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=np.array([[0,1,2],[1,3,2]])
    v1=v0.copy(); v1[3]=[7.,1.,0.]
    report=deformation_shape_report(v0,v1,faces)
    result=shape_gate(report)
    assert not result['ok']
    assert 'EDGE_STRETCH_MAX' in result['reasons'] or 'LOCAL_AREA_STRETCH' in result['reasons']


def test_shape_gate_accepts_small_smooth_edit():
    v0=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    faces=np.array([[0,1,2],[1,3,2]])
    v1=v0.copy(); v1[:,2]=[0.,.03,.02,.04]
    report=deformation_shape_report(v0,v1,faces)
    assert shape_gate(report)['ok']
