"""Margin 的「磁力」特征吸附（对应 exocad Supragingival / Subgingival）。

exocad 官方 `Detect / edit margins` 原文：

    "If you activate this button, the magnetic line will follow a concave
     (curved inwards) curve instead of a convex (curved outwards) curve."
                                        —— Subgingival 按钮的说明

本模块把这个行为实现为**局部后处理**，不改任何候选生成 / 路径优化：

    SUPRA（龈上，默认）→ 每个边缘点在 radius 内吸到最**凸**的棱（肩台外缘）
    SUB  （龈下）      → 吸到最**凹**的沟底（finish line 内角）

特征强度用**有符号二面角（度）**，与仓库既有的 ``sharp_deg`` 同口径：

    signed(n) = sign( dot(n_f, c_g − c_f) ) · angle(n_f, n_g)
    > 0 → 邻面在本面法向正侧 → 表面向内折 → **凹（沟）**
    < 0 → 表面向外折         → **凸（棱）**

用角度而不是质心偏移量做强度是刻意的：质心偏移会被网格细分噪声主导
（96 段的圆柱上"平坦"区域也会测出非零偏移），而二面角对细分不敏感
（细分只会降低角度），真实台阶/肩台角在几十度量级。

吸附只做三件事：挑特征面 → 把点贴到该面上最近的位置 → 沿闭环平滑一次。
**不猜**：找不到特征面的点原样保留并如实计数；特征面占比过低时返回
``REVIEW_REQUIRED``，不输出一条"看起来已经吸附过"的曲线。
"""

from __future__ import annotations

import math

import numpy as np

MODE_SUPRA = "SUPRA"   # 龈上：凸（crest）
MODE_SUB = "SUB"       # 龈下：凹（valley）
MODES = (MODE_SUPRA, MODE_SUB)

STATUS_APPLIED = "APPLIED"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"

#: 视为"特征"的最小 |有符号二面角|（度）。低于此值按平面处理（不吸附）。
#: 96 段圆柱的细分噪声约 3.75°，故 8° 可把它们挡在外面。
FEATURE_MIN_DEG = 8.0
#: 特征面占比低于该值即 REVIEW_REQUIRED（不假装吸附成功）。
FEATURE_MIN_RATIO = 0.6


def _as_tris(mesh_or_tris):
    """接受 trimesh.Trimesh 或 (F,3,3) 的三角形数组/列表。"""
    if hasattr(mesh_or_tris, "triangles"):
        return np.asarray(mesh_or_tris.triangles, float)
    arr = np.asarray(mesh_or_tris, float)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError("需要 (面数, 3, 3) 的三角形数组")
    return arr


def face_concavity_deg(tris, ref_point=None) -> np.ndarray:
    """逐面的**有符号二面角**（度）。>0 凹（沟），<0 凸（棱），0 = 无共边邻面。

    强度取共边邻面的最大 |角度|（与 features.face_stats 的 ``sharp_deg`` 同源），
    符号取该邻面相对本面法向的位置。

    **绕向无关**：给 ``ref_point``（通常是边缘线质心）时，先把每个面的法向按
    ``sign(dot(n_f, c_f − ref))`` 统一朝外修正，再算符号。这样输入网格的法向
    朝内/朝外不会把凸凹判据翻过来（实测夹具翻转绕向后符号整体反相）。
    仅当该投影足够大（|cos| > 0.2）时才修正，避免顶面这种法向本来就垂直于
    径向的表面被瞎改。
    """
    T = _as_tris(tris)
    F = len(T)
    cents = T.mean(axis=1)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1.0
    n = n / ln[:, None]

    if ref_point is not None:
        ref = np.asarray(ref_point, float)
        radial = cents - ref
        rl = np.linalg.norm(radial, axis=1)
        rl[rl == 0] = 1.0
        cos = np.einsum("ij,ij->i", n, radial) / rl
        flip = (np.abs(cos) > 0.2) & (cos < 0)
        n = np.where(flip[:, None], -n, n)

    key = np.round(T.reshape(-1, 3), 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(F, 3)
    edges = {}
    for f in range(F):
        a, b, c = inv[f]
        for u, v in ((a, b), (b, c), (c, a)):
            k = (u, v) if u < v else (v, u)
            edges.setdefault(k, []).append(f)

    out = np.zeros(F)
    for faces in edges.values():
        if len(faces) != 2:            # 边界边或非流形：不参与判据
            continue
        f, g = faces
        d = float(np.clip(np.dot(n[f], n[g]), -1.0, 1.0))
        ang = math.degrees(math.acos(d))
        sf = float(np.dot(n[f], cents[g] - cents[f]))
        sg = float(np.dot(n[g], cents[f] - cents[g]))
        vf = ang if sf > 0 else -ang
        vg = ang if sg > 0 else -ang
        if abs(vf) > abs(out[f]):
            out[f] = vf
        if abs(vg) > abs(out[g]):
            out[g] = vg
    return out


def _closest_on_triangle(p, tri):
    """点到三角形的最近点（Ericson, Real-Time Collision Detection §5.1.5）。"""
    a, b, c = tri
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 <= d1 and d3 <= d1:
        v = d1 / (d1 - d3)
        return a + v * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 <= d2 and d6 <= d2:
        w = d2 / (d2 - d6)
        return a + w * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)
    denom = va + vb + vc
    if denom == 0:
        return a
    v, w = vb / denom, vc / denom
    return a + ab * v + ac * w


def _local_faces(cents, p, radius_mm):
    d = np.linalg.norm(cents - np.asarray(p, float), axis=1)
    return np.flatnonzero(d <= radius_mm)


def snap_to_feature(tris, points, axis=None, mode=MODE_SUPRA, radius_mm=1.0,
                    smooth_passes=1, smooth_lambda=0.5,
                    feature_min_deg=FEATURE_MIN_DEG):
    """把闭环吸附到局部凸棱（SUPRA）或凹沟（SUB）。

    返回 dict：``points`` / ``status`` / ``reason`` / ``n_moved`` / ``n_kept`` /
    ``feature_ratio`` / ``mean_move_mm`` / ``max_move_mm`` / ``mode`` / ``radius_mm``。
    ``axis`` 不参与判据（二面角与方向无关），保留参数以固定接口。
    """
    if mode not in MODES:
        raise ValueError("mode 必须是 %s 之一，收到 %r" % ("/".join(MODES), mode))
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError("至少需要 3 个有序边缘点")
    radius_mm = float(radius_mm)
    if not (0.1 <= radius_mm <= 5.0):
        raise ValueError("radius_mm 必须在 0.1–5.0 mm")

    T = _as_tris(tris)
    if len(T) == 0:
        raise ValueError("网格为空")
    cents = T.mean(axis=1)
    conc = face_concavity_deg(T)
    target_sign = 1.0 if mode == MODE_SUB else -1.0

    moved = np.array(P, float)
    kept, moves = 0, []
    for i in range(len(P)):
        ids = _local_faces(cents, P[i], radius_mm)
        if len(ids) == 0:
            kept += 1
            continue
        sign_ok = ids[(np.sign(conc[ids]) == target_sign)
                      & (np.abs(conc[ids]) >= feature_min_deg)]
        if len(sign_ok) == 0:
            kept += 1
            continue
        strength = np.abs(conc[sign_ok])
        dist = np.linalg.norm(cents[sign_ok] - P[i], axis=1)
        order = np.lexsort((dist, -strength))       # 先强后近
        fid = int(sign_ok[order[0]])
        q = _closest_on_triangle(P[i], T[fid])
        d = float(np.linalg.norm(q - P[i]))
        if d > radius_mm:                           # 移动超界：判为未吸附
            kept += 1
            continue
        moved[i] = q
        moves.append(d)

    n_pts = len(P)
    feature_ratio = 1.0 - kept / float(n_pts)
    if smooth_passes and feature_ratio > 0:
        idx = np.arange(n_pts)
        prev, nxt = (idx - 1) % n_pts, (idx + 1) % n_pts
        for _ in range(int(smooth_passes)):
            moved = moved + smooth_lambda * ((moved[prev] + moved[nxt]) / 2.0 - moved)
        for i in range(n_pts):                      # 平滑后贴回网格
            ids = _local_faces(cents, moved[i], radius_mm * 1.5)
            if len(ids) == 0:
                continue
            dist = np.linalg.norm(cents[ids] - moved[i], axis=1)
            moved[i] = _closest_on_triangle(moved[i], T[int(ids[int(np.argmin(dist))])])

    mv = np.asarray(moves, float)
    status = STATUS_APPLIED if feature_ratio >= FEATURE_MIN_RATIO else STATUS_REVIEW_REQUIRED
    reason = "" if status == STATUS_APPLIED else (
        "仅有 %.0f%% 的边缘点在 %.1f mm 内找到%s特征（要求 ≥%.0f%%，阈值 %.0f°）；"
        "未吸附点已原样保留，请人工复核或改用另一种模式"
        % (100 * feature_ratio, radius_mm, "凹" if mode == MODE_SUB else "凸",
           100 * FEATURE_MIN_RATIO, feature_min_deg))
    return {
        "points": [[float(v) for v in q] for q in moved],
        "status": status,
        "reason": reason,
        "mode": mode,
        "radius_mm": radius_mm,
        "n_points": int(n_pts),
        "n_moved": int(len(moves)),
        "n_kept": int(kept),
        "feature_ratio": round(float(feature_ratio), 4),
        "mean_move_mm": round(float(mv.mean()), 4) if len(mv) else 0.0,
        "max_move_mm": round(float(mv.max()), 4) if len(mv) else 0.0,
        "feature_min_deg": float(feature_min_deg),
    }
