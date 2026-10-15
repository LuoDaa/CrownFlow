import numpy as np
import trimesh
from scipy.spatial import cKDTree
from .margin import fit_margin_plane

def orient_margin_normal(margin_points, insertion_axis):
    c,n=fit_margin_plane(margin_points)
    axis=np.asarray(insertion_axis,float); axis/=np.linalg.norm(axis)
    if np.dot(n,axis) < 0: n=-n
    return c,n

def trim_outer_at_margin(mesh, margin_points, insertion_axis, extension_mm=0.15):
    """按 Margin 平面裁切外冠，并**断言结果水密**。

    裁切盖片需要三角化引擎（mapbox-earcut）。缺失时 ``slice_mesh_plane(cap=True)``
    会抛异常，旧实现静默退化到 ``cap=False`` 并留下开放边界 —— 这正是实测中
    最终牙冠只有真值 13% 体积的直接原因。因此这里不再吞掉失败。
    """
    from .boolean import assert_watertight
    from .margin_seam import margin_plane_diagnostics, MarginReconstructionRequired

    diagnostics = margin_plane_diagnostics(margin_points)
    if not diagnostics["numerically_planar"]:
        diagnostics["status"] = "MARGIN_RECONSTRUCTION_REQUIRED"
        diagnostics["missing_adapters"] = [
            "reviewed_outer_cervical_open_surface",
            "direction_blockout_gap_intaglio_with_prescribed_margin_boundary",
            "shared_seam_fit_thickness_audit",
        ]
        raise MarginReconstructionRequired(diagnostics)

    center, normal = orient_margin_normal(margin_points, insertion_axis)
    origin = center - normal * float(extension_mm)
    try:
        out = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=normal, plane_origin=origin, cap=True)
    except Exception as exc:
        raise ValueError(
            f"Margin 平面裁切失败：{exc}\n"
            "  cap=True 需要三角化引擎（mapbox-earcut）。缺失时裁切面无法生成，"
            "继续执行会产出非水密牙冠并最终得到碎冠。\n"
            "  安装: pip install -r backend/requirements.txt"
        ) from exc

    if out is None or len(out.faces) == 0:
        raise ValueError("牙冠模板与 Margin 平面裁切失败：结果为空")

    assert_watertight(out, "Margin 裁切后的外冠")
    return out

def enforce_clearance(outer, cutter, min_thickness_mm, margin_points=None,
                      margin_protect_mm=0.6, iterations=3):
    result=outer.copy()
    if min_thickness_mm <= 0: return result
    from trimesh.proximity import closest_point
    margin_tree=cKDTree(np.asarray(margin_points)) if margin_points is not None and len(margin_points)>0 else None

    for _ in range(int(iterations)):
        v=np.asarray(result.vertices)
        d=np.concatenate([closest_point(cutter,v[i:i+1024])[1] for i in range(0,len(v),1024)])
        deficit=float(min_thickness_mm)-d
        move_mask=deficit>0
        if margin_tree is not None:
            md,_=margin_tree.query(v,k=1,workers=-1)
            move_mask &= md > margin_protect_mm
        if not move_mask.any(): break
        normals=np.asarray(result.vertex_normals)
        v2=v.copy()
        v2[move_mask]+=normals[move_mask]*(deficit[move_mask,None]*0.72)
        result.vertices=v2
        result.remove_unreferenced_vertices()
    return result
