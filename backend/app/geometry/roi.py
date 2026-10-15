import numpy as np
import trimesh

def spherical_roi(mesh, center, radius_mm):
    center=np.asarray(center,float)
    fc=np.asarray(mesh.triangles_center)
    mask=np.linalg.norm(fc-center,axis=1) <= float(radius_mm)
    if mask.sum() < 20:
        raise ValueError("ROI 内三角面过少，请扩大半径或重新选择中心")
    sub=mesh.submesh([np.nonzero(mask)[0]], append=True, repair=False)
    sub.remove_unreferenced_vertices()
    return sub
