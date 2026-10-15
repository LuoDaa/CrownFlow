"""CrownOuter v0 的质量控制（方案 §10）。

三类 QC 全部独立计算，**不以求解器内部残差代替**：

* Margin Fit —— 变形后颈环与患者 Margin 的双向曲线距离（median/p95/max）；
* Anatomy Distortion —— 每顶点相对摆位模板的位移，按 cervical/axial/occlusal
  分区统计（理想：颈部明显动、咬合面明显少动）；
* Mesh Quality —— 连通分量、退化三角形、法向翻转、边界环一致性。

区域定义（方案 §5.3，PROVISIONAL）：
* cervical：距颈环顶点 < ``cervical_band_mm`` 的冠部顶点；
* occlusal：冠部 z 高于 ``occlusal_anchor_percentile`` 分位；
* axial：其余。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .models import CrownOuterConfig
from .template import AXIS_Z


# ---------------------------------------------------------------- 区域

def region_masks(vertices: np.ndarray, crown_mask: np.ndarray,
                 margin_loop: np.ndarray, config: CrownOuterConfig
                 ) -> dict[str, np.ndarray]:
    """cervical / axial / occlusal 三区 mask（都定义在冠部顶点上）。"""
    v = vertices[crown_mask]
    tree = cKDTree(margin_loop)
    d, _ = tree.query(v, k=1)
    z_thr = np.percentile(v[:, AXIS_Z], config.occlusal_anchor_percentile)
    cervical = d < config.cervical_band_mm
    occlusal = v[:, AXIS_Z] >= z_thr
    out = {"cervical": cervical, "occlusal": occlusal}
    out["axial"] = ~(cervical | occlusal)
    # 映射回全网格下标
    crown_idx = np.flatnonzero(crown_mask)
    return {k: crown_idx[m] for k, m in out.items()}


# ---------------------------------------------------------------- Margin Fit

def _polyline_query(points: np.ndarray, polyline: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """点到折线的最近距离与最近点（含段上插值）。"""
    seg_a = polyline
    seg_b = np.roll(polyline, -1, axis=0)
    d = seg_b - seg_a                                    # (S,3)
    denom = np.einsum("ij,ij->i", d, d)
    denom[denom < 1e-18] = 1e-18
    n_pts = len(points)
    best_d = np.full(n_pts, np.inf)
    best_p = np.zeros_like(points)
    chunk = 256
    for s in range(0, n_pts, chunk):
        p = points[s:s + chunk]                          # (C,3)
        t = np.einsum("cij,ij->ci", p[:, None, :] - seg_a[None, :, :], d)
        t = np.clip(t / denom[None, :], 0.0, 1.0)
        proj = seg_a[None, :, :] + t[:, :, None] * d[None, :, :]
        dist = np.linalg.norm(p[:, None, :] - proj, axis=2)
        k = np.argmin(dist, axis=1)
        best_d[s:s + chunk] = dist[np.arange(len(p)), k]
        best_p[s:s + chunk] = proj[np.arange(len(p)), k]
    return best_d, best_p


def margin_fit(loop_deformed: np.ndarray, patient_margin: np.ndarray
               ) -> dict:
    """双向曲线距离：变形后颈环 ↔ 患者 Margin 折线。"""
    d1, _ = _polyline_query(loop_deformed, patient_margin)
    d2, _ = _polyline_query(patient_margin, loop_deformed)
    def _stats(x):
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
                "max": float(np.max(x))}
    return {"loop_to_margin_mm": _stats(d1),
            "margin_to_loop_mm": _stats(d2),
            "symmetric": _stats(np.concatenate([d1, d2]))}


# ---------------------------------------------------------------- 形变统计

def displacement_by_region(vertices_before: np.ndarray,
                           vertices_after: np.ndarray,
                           regions: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, idx in regions.items():
        if len(idx) == 0:
            out[name] = {"n": 0, "median_mm": None, "p95_mm": None,
                         "max_mm": None}
            continue
        d = np.linalg.norm(vertices_after[idx] - vertices_before[idx], axis=1)
        out[name] = {"n": int(len(idx)),
                     "median_mm": round(float(np.median(d)), 4),
                     "p95_mm": round(float(np.percentile(d, 95)), 4),
                     "max_mm": round(float(np.max(d)), 4)}
    return out


# ---------------------------------------------------------------- 网格 QC

def mesh_quality(vertices: np.ndarray, faces: np.ndarray,
                 expected_boundary_count: int,
                 vertices_reference: np.ndarray | None = None) -> dict:
    """网格质量 QC。

    **``n_flipped_faces`` 的统一定义（批量回归前钉死，18 例共用）**：
    在**输出面片集合**（生成器实际写出的冠部+颈环面）上，逐面比较
    「变形后法向」与「**摆位后模板**（同面、同顶点序）的法向」，点积 < 0
    记为翻转。参考系取摆位后模板而非原始模板 —— 摆位是刚体+均匀缩放、
    不产生翻转，用它可以把"形变造成的翻转"从坐标系变换里干净分离。

    历史口径说明：单例调试时出现过的"0% 翻转"读数是在**错误的输出面集合**
    （不含颈环圈的冠部面）上算的，已废弃；管线正式口径就是本定义，
    第一颗病例的正式数字是 0.13%。
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    # 连通分量（并查集 via union-find on edges）
    parent = np.arange(len(v))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, c in f:
        for u, w in ((a, b), (b, c), (c, a)):
            ru, rw = find(u), find(w)
            if ru != rw:
                parent[ru] = rw
    used = np.unique(f)
    comps = len({find(int(u)) for u in used})

    # 退化三角形
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    degenerate = int(np.sum(area2 < 1e-10))

    # 边界边（只出现一次）
    from collections import Counter
    cnt = Counter()
    for a, b, c3 in f:
        for u, w2 in ((a, b), (b, c3), (c3, a)):
            cnt[(min(u, w2), max(u, w2))] += 1
    boundary = sum(1 for x in cnt.values() if x == 1)

    # 法向翻转：与参考（摆位模板）对比
    flipped = None
    if vertices_reference is not None:
        vr = np.asarray(vertices_reference, dtype=np.float64)
        n_ref = np.cross(vr[f[:, 1]] - vr[f[:, 0]], vr[f[:, 2]] - vr[f[:, 0]])
        n_new = np.cross(e1, e2)
        denom = (np.linalg.norm(n_ref, axis=1)
                 * np.linalg.norm(n_new, axis=1))
        ok = denom > 1e-18
        dots = np.zeros(len(f))
        dots[ok] = np.einsum("ij,ij->i", n_ref[ok], n_new[ok]) / denom[ok]
        flipped = int(np.sum(dots < 0.0))

    return {"n_components": comps, "n_degenerate_faces": degenerate,
            "n_boundary_edges": boundary,
            "expected_boundary_edges": expected_boundary_count,
            "boundary_consistent": boundary == expected_boundary_count,
            "n_flipped_faces": flipped,
            "flipped_ratio": (round(flipped / len(f), 5)
                              if flipped is not None else None)}


# ---------------------------------------------------------------- 状态裁决

def decide_status(margin_stats: dict, disp: dict, qc: dict,
                  config: CrownOuterConfig) -> tuple[str, list[str]]:
    """按 QC 阈值裁决 OK / REVIEW_REQUIRED / FAILED（fail-loud）。"""
    notes: list[str] = []
    sym = margin_stats["symmetric"]
    if sym["p95"] > config.qc_margin_p95_mm:
        notes.append("margin_p95=%.3f 超阈值 %.3f"
                     % (sym["p95"], config.qc_margin_p95_mm))
    if not qc["boundary_consistent"]:
        notes.append("边界环边数 %d != 预期 %d"
                     % (qc["n_boundary_edges"], qc["expected_boundary_edges"]))
    if qc["n_components"] != 1:
        notes.append("网格连通分量 %d != 1" % qc["n_components"])
    if qc["n_flipped_faces"] is not None:
        if qc["flipped_ratio"] > 0.05:
            return "FAILED", notes + [
                "法向翻转面占比 %.3f > 0.05，网格破坏" % qc["flipped_ratio"]]
        if qc["flipped_ratio"] > config.qc_flipped_faces_ratio:
            notes.append("法向翻转面占比 %.4f 超阈值 %.4f"
                         % (qc["flipped_ratio"],
                            config.qc_flipped_faces_ratio))
    occ = disp.get("occlusal", {})
    cer = disp.get("cervical", {})
    if occ.get("p95_mm") and cer.get("p95_mm"):
        ratio = occ["p95_mm"] / max(cer["p95_mm"], 1e-9)
        if ratio > config.qc_occlusal_motion_ratio:
            notes.append("咬合区位移 p95 / 颈区 p95 = %.3f 超阈值 %.3f "
                         "（咬合面被拉动了）"
                         % (ratio, config.qc_occlusal_motion_ratio))
    return ("REVIEW_REQUIRED" if notes else "OK"), notes
