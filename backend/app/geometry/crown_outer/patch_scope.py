"""邻牙/对颌 patch **作用域规则**（Patch Scope）—— 用户 2026-10-09 锁定的第一任务。

## 问题（已被 GT control 证实）

旧流程用"距冠最近的牙弓三角形 + 半径 9mm 区域生长"（`grow_neighbor_patch.py`）
长 patch，结果把**目标牙自身位置、颈缘/牙龈、颊舌远区**都包了进去。于是：
一个**临床已交付**的设计冠（GT）在这些 patch 上也读出穿透 p95 ≈ 1.0–1.1mm
（真正的近接大陆其实是零穿透、间隙 1.0–1.8mm）。

## 规则（全部相对"永远可得"的参考形状，不依赖 GT）

参考 = **代型/内表面**（`prep/prop_die`，闭合）与**颈缘线**（margin 点列）。
候选 = 源牙弓/对颌 mesh 的面。逐面判：

1. **距离带**：面心到参考表面的距离 ∈ [d_min, d_max]
   —— 近接壁就在代型外 0.05–3mm；>3mm 是远区/牙龈/别的牙。
2. **颈缘以上**：面心在颈缘平面之上 ≥ h_margin
   —— 颈缘/牙龈/牙乳头都在颈缘线附近或以下。
3. **侧向**：`sign·u ≥ sign·u_ref_extreme + side_margin`
   —— 本侧的近接壁必然在**代型本侧极值之外**（u 是 MD 轴坐标）。
4. **朝向**：`dot(n̂, axis_pt − c) > facing_cos`（axis_pt = 该高度就位轴上的点）
   —— 只留"朝着目标牙"的面；颊舌外表面/咬合面/背向面都掉。
5. **冠体内排除**：面心在**冠体（GT 或首过冠）**内部的丢掉
   —— 目标牙自身残留/临时冠/被冠覆盖的区域一律不是接触参照。

保留的种子面再做**邻接连通生长**（同一条牙面上的连续区域），避免碎点。

## 验收（前置 Gate，与用户锁定一致）

`verify_patch()`：
* 有 GT 冠时：GT 在 patch 上的**接触口径穿透 p95 ≤ allow_mm**（默认 0.2）且
  有效点不空；否则 `PATCH_SCOPE_CONTAMINATED`；
* **覆盖性**：GT 冠近接壁（|lat|≤1.5 的近接大陆）必须能在 patch 上找到有效最近点，
  否则 `PATCH_TRUNCATED`（patch 裁过头，接触区没盖住）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .thickness import winding_number


@dataclass
class PatchScopeParams:
    d_min_mm: float = 0.05          # 距参考表面下限（参考自身表面/贴面掉）
    d_max_mm: float = 3.0           # 距参考表面上限（远区/牙龈/别牙掉）
    margin_above_mm: float = 0.5    # 颈缘平面以上多少才算
    side_margin_mm: float = 0.3     # 本侧极值外多少才算
    facing_cos: float = 0.10        # 朝向阈值（dot(n̂, 指向轴线方向)）
    grow_d_max_mm: float = 6.0      # 连通生长：距种子集合的最远距离


def _frame_axes(frame):
    x = np.asarray(frame.x, float)
    z = np.asarray(frame.z, float)
    origin = np.asarray(frame.origin, float)
    return x, z, origin


def scope_patch(cand_v, cand_f, ref_v, ref_f, frame, margin_pts, side, *,
                crown_v=None, crown_f=None,
                params: PatchScopeParams | None = None) -> dict:
    """按作用域规则从候选网格里取 patch 面。

    ``ref_v/ref_f``：参考形状（代型/内表面，闭合）；
    ``crown_v/crown_f``：可选冠体（GT 或首过冠），用于"冠体内排除"。
    ``side``：``"mesial"`` / ``"distal"``（沿 frame.x 分侧）。
    → dict(keep_faces=..., stats=..., excluded=...)
    """
    p = params or PatchScopeParams()
    cv = np.asarray(cand_v, float)
    cf = np.asarray(cand_f, np.int64)
    rv = np.asarray(ref_v, float)
    x, z, origin = _frame_axes(frame)
    sign = 1.0 if side == "mesial" else -1.0
    cen = cv[cf].mean(axis=1)
    n = np.cross(cv[cf[:, 1]] - cv[cf[:, 0]], cv[cf[:, 2]] - cv[cf[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-12
    n = n / np.maximum(ln, 1e-12)[:, None]
    margin = np.asarray(margin_pts, float)
    o_margin = margin.mean(axis=0)
    u_ref = (rv - o_margin) @ x
    u_ext = float(u_ref.max()) if sign > 0 else float(u_ref.min())

    ref_tree = cKDTree(rv)
    d_ref = ref_tree.query(cen, k=1)[0]
    u = (cen - o_margin) @ x
    zz = (cen - o_margin) @ z
    axis_pt = origin[None, :] + ((cen - origin[None, :]) @ z)[:, None] * z[None, :]
    toward = axis_pt - cen
    toward_n = toward / np.maximum(np.linalg.norm(toward, axis=1, keepdims=True),
                                   1e-12)
    facing = np.einsum("ij,ij->i", n, toward_n)
    inside = np.zeros(len(cf), bool)
    if crown_v is not None and crown_f is not None:
        tri = np.asarray(crown_v, float)[np.asarray(crown_f, np.int64)]
        inside = winding_number(cen, tri) > 0.5

    m = {
        "degenerate": ~ok,
        "dist_band": (d_ref < p.d_min_mm) | (d_ref > p.d_max_mm),
        "below_margin": zz < p.margin_above_mm,
        "wrong_side": sign * u < sign * u_ext + p.side_margin_mm,
        "not_facing": facing <= p.facing_cos,
        "inside_crown": inside,
    }
    keep_mask = ~np.logical_or.reduce(list(m.values()))
    stats = {k: int(v.sum()) for k, v in m.items()}
    stats["n_candidate_faces"] = int(len(cf))
    stats["n_kept"] = int(keep_mask.sum())
    keep = np.flatnonzero(keep_mask)

    # 连通生长：从保留面 BFS，只走过保留面（去掉孤立碎块）
    if len(keep) and p.grow_d_max_mm > 0:
        from collections import defaultdict, deque
        adj = defaultdict(set)
        for i in keep:
            a, b, c = cf[i]
            for uu, vv in ((a, b), (b, c), (c, a)):
                adj[(min(int(uu), int(vv)), max(int(uu), int(vv)))].add(int(i))
        keep_set = set(int(i) for i in keep)
        comp_id = {}
        comps = []
        for i0 in keep:
            i0 = int(i0)
            if i0 in comp_id:
                continue
            cid = len(comps)
            q = deque([i0])
            comp_id[i0] = cid
            members = [i0]
            while q:
                f = q.popleft()
                a, b, c = cf[f]
                for uu, vv in ((a, b), (b, c), (c, a)):
                    for j in adj[(min(int(uu), int(vv)), max(int(uu), int(vv)))]:
                        if j != f and j in keep_set and j not in comp_id:
                            comp_id[j] = cid
                            q.append(j)
                            members.append(j)
            comps.append(members)
        if comps:
            main = max(comps, key=len)
            keep = np.asarray(sorted(main), dtype=np.int64)
            stats["n_components_all"] = len(comps)
            stats["component_sizes"] = [len(c) for c in comps[:8]]
            stats["n_kept_main"] = int(len(keep))
    return {"keep_faces": keep, "stats": stats, "excluded": m,
            "u_extreme_ref": u_ext,
            "n_candidate_faces": int(len(cf))}


def verify_patch(patch_v, patch_f, crown_v, crown_f, frame, *, side=None,
                 allow_p95_mm: float = 0.2, land_lat_mm: float = 1.5,
                 land_depth_mm: float = 1.2, min_land_points: int = 20) -> dict:
    """**前置 Gate**：用 GT（或首过）冠做对照验证 patch 可否作为接触参照。

    ``side``：``"mesial"/"distal"``（单侧 patch 只查该侧覆盖性）；
    ``None`` = 双侧都查（合并 patch 用）。
    """
    from .contact import NeighborPatch
    x, _z, origin = _frame_axes(frame)
    patch = NeighborPatch(patch_v, patch_f)
    # 定向：冠质心一侧
    fc = patch.vertices[patch.faces].mean(axis=1)
    w = np.einsum("ij,ij->i", np.asarray(crown_v, float).mean(axis=0) - fc,
                  patch.normals)
    patch.normal_sign = 1.0 if w.sum() > 0 else -1.0
    sc, valid, cp, fids, _parts = patch.clearance_detail(np.asarray(crown_v, float))
    out = {"n_valid": int(valid.sum()), "n_queried": int(len(sc))}
    if valid.any():
        pen = np.maximum(-sc[valid], 0.0)
        out.update({"penetration_p50_mm": float(np.median(pen)),
                    "penetration_p95_mm": float(np.quantile(pen, .95)),
                    "penetration_max_mm": float(pen.max()),
                    "min_clearance_mm": float(sc[valid].min())})
    else:
        out.update({"penetration_p50_mm": None, "penetration_p95_mm": None,
                    "penetration_max_mm": None, "min_clearance_mm": None})
    # 覆盖性：近接大陆（|lat|≤1.5，距本侧极值 land_depth 内）能否找到有效点
    cvv = np.asarray(crown_v, float)
    u = (cvv - origin) @ x
    lat = (cvv - origin) @ np.asarray(frame.y, float)
    land_cov = {}
    pv_tree = cKDTree(np.asarray(patch_v, float))
    for side_, sign in (("mesial", 1.0), ("distal", -1.0)):
        if side is not None and side_ != side:
            continue
        m0 = np.abs(lat) <= land_lat_mm
        if not m0.any():
            land_cov[side_] = {"n": 0, "n_valid": 0, "ok": False}
            continue
        thr = sign * u[m0].max() - land_depth_mm
        sel = np.flatnonzero(m0 & (sign * u >= thr))
        if not len(sel):
            land_cov[side_] = {"n": 0, "n_valid": 0, "ok": False}
            continue
        sc2, valid2, _c, _f, _p = patch.clearance_detail(cvv[sel])
        # 覆盖性用**顶点距离**（不受开放 patch 边界安全带影响：边界安全带是
        # "这点的读数可不可信"的问题，不是"patch 盖没盖住这里"的问题）。
        d_patch = pv_tree.query(cvv[sel], k=1)[0]
        covered = d_patch <= (land_lat_mm + 0.5)
        land_cov[side_] = {
            "n": int(len(sel)),
            "n_valid": int(valid2.sum()),
            "n_covered_1_5mm": int(covered.sum()),
            "p95_dist_mm": float(np.quantile(d_patch, .95)),
            "ok": bool(covered.sum() >= min_land_points)}
    out["land_coverage"] = land_cov
    p95 = out["penetration_p95_mm"]
    if p95 is None:
        out["status"] = "PATCH_UNUSABLE"
    elif p95 > allow_p95_mm:
        out["status"] = "PATCH_SCOPE_CONTAMINATED"
    elif not all(v["ok"] for v in land_cov.values()):
        out["status"] = "PATCH_TRUNCATED"
    else:
        out["status"] = "PATCH_OK"
    out["allow_p95_mm"] = float(allow_p95_mm)
    return out
