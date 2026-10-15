import numpy as np
from trimesh.proximity import closest_point
from scipy.spatial import cKDTree
from .io import mesh_info

def _sample(v,max_n=60000):
    a=np.asarray(v,float)
    if len(a)<=max_n: return a
    idx=np.linspace(0,len(a)-1,max_n,dtype=int)
    return a[idx]

def nearest_stats(mesh_a, mesh_b):
    a=_sample(mesh_a.vertices)
    # Distance to triangles, independent of how densely the reference mesh is tessellated.
    d=np.concatenate([closest_point(mesh_b,a[i:i+1024])[1] for i in range(0,len(a),1024)])
    return {
        "min_mm":float(d.min()),
        "p01_mm":float(np.percentile(d,1)),
        "p05_mm":float(np.percentile(d,5)),
        "median_mm":float(np.median(d)),
        "p95_mm":float(np.percentile(d,95)),
        "max_mm":float(d.max()),
        "samples":int(len(a)),
        "method":"sampled_vertices_to_exact_triangles_unsigned",
    }

def shell_clearance_stats(outer,cutter):
    return nearest_stats(outer,cutter)

def filter_support_from_same_jaw(same_jaw,support,exclusion_mm=0.35):
    sv=np.asarray(support.vertices)
    tree=cKDTree(sv)
    face_centers=np.asarray(same_jaw.triangles_center)
    d,_=tree.query(face_centers,k=1,workers=-1)
    keep=d>float(exclusion_mm)
    if keep.sum()<20:
        return same_jaw
    return same_jaw.submesh([np.nonzero(keep)[0]],append=True,repair=False)

def build_qc(final_crown, outer, cutter, support=None, same_jaw=None, antagonist=None):
    out={"final_mesh":mesh_info(final_crown)}
    out["shell_clearance_estimate"]=shell_clearance_stats(outer,cutter)
    out["support_clearance"]=nearest_stats(final_crown,support) if support is not None else None
    if same_jaw is not None:
        env=filter_support_from_same_jaw(same_jaw,support) if support is not None else same_jaw
        out["neighbor_environment_distance"]=nearest_stats(final_crown,env)
    else:
        out["neighbor_environment_distance"]=None
    out["antagonist_distance"]=nearest_stats(final_crown,antagonist) if antagonist is not None else None
    return out
