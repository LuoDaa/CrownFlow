"""Final Crown GT 的外表面提取与回归指标（方案 §11，acceptance-only）。

**本模块只服务于验收，不进入生产生成链。**

Final Crown STL 是闭合修复体 = Outer（外表面）+ Intaglio（内表面）。
v0 只生成外表面，所以 GT 必须先分离。分离用"法向 × 轴线径向"判据：

* 以 Margin 质心沿 Insertion Axis 的直线为参考轴；
* 外表面法向背离该轴（颊舌近远中壁），内表面（intaglio）法向朝向该轴；
* 咬合顶面外表面法向 +Z、内表面顶面法向 −Z —— 用轴向判据并入外侧；
* 从"离轴最远的面"（必是颊侧外表面）做连通扩张，取连通分量 ——
  避免散碎误判面混入。

提取后人工确认（渲染三视图），不做静默 fallback。
"""
from __future__ import annotations

import numpy as np


class GtExtractError(ValueError):
    pass


def extract_outer_surface(mesh, insertion_axis: np.ndarray,
                          margin_centroid: np.ndarray, *,
                          axial_threshold: float = 0.25) -> np.ndarray:
    """从闭合冠网格提取外表面面片下标。返回 face indices。

    ``mesh``：trimesh.Trimesh（闭合修复体）。
    """
    fn = np.asarray(mesh.face_normals, dtype=np.float64)
    fc = mesh.triangles_center
    z = np.asarray(insertion_axis, dtype=np.float64)
    z = z / np.linalg.norm(z)

    radial = fc - margin_centroid
    radial = radial - np.outer(radial @ z, z)          # 去轴向分量
    rn = np.linalg.norm(radial, axis=1)
    rn[rn < 1e-9] = 1e-9
    radial = radial / rn[:, None]

    lateral_out = np.einsum("ij,ij->i", fn, radial)    # >0 背离轴线 = 外壁
    axial_out = fn @ z                                  # >0 朝咬合 = 外咬合面

    is_outer = (lateral_out > 0) | (axial_out > axial_threshold)

    # 从离轴最远的面（必是颊侧外壁）连通扩张
    adj = _face_adjacency(np.asarray(mesh.faces, dtype=np.int64))
    seed = int(np.argmax(rn))
    if not is_outer[seed]:
        raise GtExtractError("离轴最远的面不满足外表面判据 —— 判据或轴有误")
    comp = _flood(is_outer, adj, seed)
    if len(comp) < len(mesh.faces) * 0.2:
        raise GtExtractError(
            "外表面连通分量只有 %d/%d 面 —— 分离失败，需人工检查"
            % (len(comp), len(mesh.faces)))
    return np.asarray(sorted(comp), dtype=np.int64)


def _face_adjacency(faces: np.ndarray) -> dict[int, set[int]]:
    from collections import defaultdict
    edge_map: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map[(min(u, v), max(u, v))].append(fi)
    adj: dict[int, set[int]] = defaultdict(set)
    for fl in edge_map.values():
        for i in fl:
            for j in fl:
                if i != j:
                    adj[i].add(j)
    return adj


def _flood(allowed: np.ndarray, adj: dict[int, set[int]],
           seed: int) -> set[int]:
    seen = {seed}
    stack = [seed]
    while stack:
        u = stack.pop()
        for w in adj.get(u, ()):
            if allowed[w] and w not in seen:
                seen.add(w)
                stack.append(w)
    return seen


# ---------------------------------------------------------------- 指标

def surface_distance_metrics(pred_pts: np.ndarray, gt_pts: np.ndarray,
                             pred_normals: np.ndarray | None = None,
                             gt_normals: np.ndarray | None = None
                             ) -> dict:
    """双向表面距离 + Chamfer + Hausdorff + 法向角误差。

    采样点已含面积加权的职责交给调用方；这里给的是点集指标。
    """
    from scipy.spatial import cKDTree
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    gt_pts = np.asarray(gt_pts, dtype=np.float64)
    t_pred = cKDTree(pred_pts)
    t_gt = cKDTree(gt_pts)

    d_pg, _ = t_gt.query(pred_pts, k=1)     # pred → gt
    d_gp, _ = t_pred.query(gt_pts, k=1)     # gt → pred

    def _stats(x):
        return {"median": round(float(np.median(x)), 4),
                "p95": round(float(np.percentile(x, 95)), 4),
                "max": round(float(np.max(x)), 4),
                "mean": round(float(np.mean(x)), 4)}

    out = {
        "pred_to_gt_mm": _stats(d_pg),
        "gt_to_pred_mm": _stats(d_gp),
        "chamfer_mean_mm": round(float(0.5 * (d_pg.mean() + d_gp.mean())), 4),
        "hausdorff_mm": round(float(max(d_pg.max(), d_gp.max())), 4),
    }
    if pred_normals is not None and gt_normals is not None:
        _, idx = t_gt.query(pred_pts, k=1)
        gn = np.asarray(gt_normals, dtype=np.float64)[idx]
        pn = np.asarray(pred_normals, dtype=np.float64)
        denom = (np.linalg.norm(pn, axis=1) * np.linalg.norm(gn, axis=1))
        ok = denom > 1e-12
        cosang = np.clip(np.einsum("ij,ij->i", pn[ok], gn[ok]) / denom[ok],
                         -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        out["normal_angle_deg"] = _stats(ang)
    return out


def sample_surface_points(mesh, n: int = 20000, seed: int = 7
                          ) -> tuple[np.ndarray, np.ndarray]:
    """面积加权采样点与法向。"""
    pts, face_idx = mesh.sample(n, return_index=True, seed=seed)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)[face_idx]
    return np.asarray(pts, dtype=np.float64), normals


# ---------------------------------------------------------------- 带符号 + 分区

def signed_surface_error(pred_pts: np.ndarray, gt_pts: np.ndarray,
                         gt_normals: np.ndarray) -> np.ndarray:
    """带符号表面误差（方案 H 轮要求）。

    ``signed_i = (pred_i − gt_nearest) · gt_normal``（GT 法向朝外）：

    * **负 = pred 在 GT 体内 → 我们的冠偏瘦 / 欠膨胀**；
    * **正 = pred 在 GT 体外 → 我们的冠偏胖 / 过膨胀**。

    GT 法向必须朝外 —— ``extract_outer_surface`` 提取的外表面满足这一点
    （外壁法向背离轴线、咬合面法向朝咬合）。
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(gt_pts)
    dist, idx = tree.query(pred_pts, k=1)
    gn = np.asarray(gt_normals, dtype=np.float64)[idx]
    d = np.asarray(pred_pts, dtype=np.float64) - gt_pts[idx]
    return np.einsum("ij,ij->i", d, gn / np.maximum(
        np.linalg.norm(gn, axis=1), 1e-12)[:, None])


def anatomical_regions(points: np.ndarray, margin_centroid: np.ndarray,
                       insertion_axis: np.ndarray,
                       axis_mesial: np.ndarray | None,
                       axis_buccal: np.ndarray | None, *,
                       occlusal_fraction: float = 0.65,
                       cervical_band_mm: float = 1.5) -> dict[str, np.ndarray]:
    """把冠面采样点分成解剖区域（方案 H 轮的误差归因要求）。

    方向语义**来自患者自己的 constructionInfo 解剖轴**（AxisMesial /
    AxisBuccal 对 Z 正交化），不猜：

    * occlusal：沿轴高出颈缘平面 > ``occlusal_fraction`` × 冠最大轴高；
    * cervical：距颈缘平面 < ``cervical_band_mm``；
    * 近中/远中/颊/舌：其余侧壁点按 (mesial, buccal) 平面坐标的主导象限。

    ``axis_buccal`` 缺失时颊/舌并入 ``lateral_unknown``（如实缺省，不硬分）。
    """
    pts = np.asarray(points, dtype=np.float64)
    z = np.asarray(insertion_axis, dtype=np.float64)
    z = z / np.linalg.norm(z)
    rel = pts - np.asarray(margin_centroid, dtype=np.float64)
    axial = rel @ z
    lateral = rel - np.outer(axial, z)
    lat_norm = np.linalg.norm(lateral, axis=1)

    z_top = axial.max()
    regions: dict[str, np.ndarray] = {}
    regions["occlusal"] = axial > occlusal_fraction * z_top
    regions["cervical"] = (-cervical_band_mm < axial) & ~regions["occlusal"] \
        & (axial < cervical_band_mm)
    side = ~regions["occlusal"] & ~regions["cervical"]

    m_dir = b_dir = None
    if axis_mesial is not None:
        m = np.asarray(axis_mesial, dtype=np.float64)
        m = m - (m @ z) * z
        nm = np.linalg.norm(m)
        if nm > 1e-9:
            m_dir = m / nm
    if axis_buccal is not None:
        b = np.asarray(axis_buccal, dtype=np.float64)
        b = b - (b @ z) * z                    # 只需对 Z 正交化
        nb = np.linalg.norm(b)
        if nb > 1e-9:
            b_dir = b / nb
    if m_dir is None or b_dir is None:
        regions["lateral_unknown"] = side
        return regions

    u = lateral @ m_dir          # >0 偏近中
    v = lateral @ b_dir          # >0 偏颊侧
    regions["mesial"] = side & (u >= 0) & (np.abs(u) >= np.abs(v))
    regions["distal"] = side & (u < 0) & (np.abs(u) >= np.abs(v))
    regions["buccal"] = side & (v >= 0) & (np.abs(v) > np.abs(u))
    regions["lingual"] = side & (v < 0) & (np.abs(v) > np.abs(u))
    return regions


def region_signed_stats(points: np.ndarray, signed: np.ndarray,
                        regions: dict[str, np.ndarray]) -> dict:
    """分区域的带符号误差统计（median/p95/mean + 点数）。"""
    out = {}
    for name, mask in regions.items():
        if mask is None or not mask.any():
            out[name] = {"n": 0, "signed_median_mm": None,
                         "signed_p95_mm": None, "signed_mean_mm": None}
            continue
        s = np.abs(signed[mask])
        sm = signed[mask]
        out[name] = {
            "n": int(mask.sum()),
            "signed_median_mm": round(float(np.median(sm)), 4),
            "signed_p95_mm": round(float(np.percentile(s, 95)), 4),
            "signed_mean_mm": round(float(np.mean(sm)), 4),
        }
    return out
