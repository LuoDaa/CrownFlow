"""Editable engineering outline from a designer's tooth center and guide axis."""
import numpy as np
from .margin import fit_margin_plane

def draft_margin(mesh, center, axis, radius_mm):
    center=np.asarray(center,float);axis=np.asarray(axis,float)
    if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([center,axis]).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('定位点和表面方向必须是有限三维向量')
    if not 2<=radius_mm<=12: raise ValueError('预备体范围必须在 2–12 mm 内')
    axis=axis/np.linalg.norm(axis)
    # shortcut: plane 3 mm below the picked crown surface, replace with validated anatomical detection later.
    origin=center-3*axis
    sections=mesh.section(plane_origin=origin,plane_normal=axis)
    if sections is None: raise ValueError('定位点下方未找到截面，请重新点击预备体顶部，或在局部模型上手动描线')
    x=np.cross(axis,[0,1,0])
    if np.linalg.norm(x)<.1:x=np.cross(axis,[1,0,0])
    x/=np.linalg.norm(x);y=np.cross(axis,x)
    choices=[]
    for ring in sections.discrete:
        if len(ring)<4 or np.linalg.norm(ring[0]-ring[-1])>.01:continue
        local=np.column_stack(((ring-origin)@x,(ring-origin)@y))
        a=local[:-1];b=local[1:];crosses=(a[:,1]>0)!=(b[:,1]>0)
        ids=np.flatnonzero(crosses)
        intercept=a[ids,0]-a[ids,1]*(b[ids,0]-a[ids,0])/(b[ids,1]-a[ids,1])
        if np.count_nonzero(intercept>0)%2!=1:continue
        if np.max(np.linalg.norm(local,axis=1))>radius_mm:continue
        choices.append(ring)
    if not choices:raise ValueError('未找到定位点对应的单牙闭环，请把定位点移到预备体中央')
    ring=min(choices,key=lambda p:np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ring,axis=0),axis=1))]
    points=np.column_stack([np.interp(np.linspace(0,lengths[-1],64,endpoint=False),lengths,ring[:,i]) for i in range(3)])
    fit_margin_plane(points)
    return points.tolist(),axis.tolist()
