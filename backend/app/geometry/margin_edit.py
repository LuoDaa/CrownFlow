"""Margin 局部编辑的**小几何助手**（Margin 步骤专用；不碰 Contact/Intaglio/Bite/Material/AI）。

设计原则：**不重写 M1 求解器**。这里只提供"用户纠偏"需要的五种局部操作，
全部复用已有 primitive（``margin_feature_snap.snap_to_feature`` / ``_closest_on_triangle``）：
投影、锚点局部重描、磁性段替换、沿局部表面移动、保锁平滑。

语义约定：
- 所有函数只处理**点列**（``(N,3)``），不碰 revision/DB；
- "局部"一律用**角度窗**或**测地半径**界定，且调用方能拿到"哪些点被改动"，
  便于断言"远端点逐位不变"；
- 移动方向沿**局部表面切平面内的插入轴投影**（不是 world-Z），移完再投影回网格。
"""

from __future__ import annotations

import numpy as np

from .margin_feature_snap import _closest_on_triangle, _local_faces, snap_to_feature


def _as_tris(tris_or_mesh) -> np.ndarray:
    if hasattr(tris_or_mesh, "triangles"):
        return np.asarray(tris_or_mesh.triangles, float)
    return np.asarray(tris_or_mesh, float)


#: Margin 校验器要求"点唯一 + 相邻间距 ≥ 0.02mm"（见 `routes._validated_margin_points`）；
#: 编辑操作（移动/平滑/锚点）可能把相邻点并到一起，这里用略大的阈值统一收拾。
MIN_MARGIN_SPACING_MM = 0.03


def dedupe_ring(points, *, keep_index=()) -> dict:
    """合并过近的相邻点（保留 ``keep_index`` 指定的硬约束点）。返回新点列与合并信息。"""
    pts = np.asarray(points, float).reshape(-1, 3)
    keep = {int(i) for i in keep_index}
    out = []
    dropped = []
    for i, point in enumerate(pts):
        if i in keep:
            out.append(point)
            continue
        if out and float(np.linalg.norm(point - out[-1])) < MIN_MARGIN_SPACING_MM:
            dropped.append(i)
            continue
        out.append(point)
    return {"points": np.asarray(out, float), "dropped_index": dropped,
            "min_spacing_mm": MIN_MARGIN_SPACING_MM}


def project_to_surface(tris_or_mesh, points) -> np.ndarray:
    """把点列投影到网格表面（逐点最近点）。返回与输入同形的点列。"""
    tris = _as_tris(tris_or_mesh)
    out = np.asarray(points, float).reshape(-1, 3).copy()
    for i, p in enumerate(out):
        j = int(np.argmin(np.linalg.norm(tris.mean(axis=1) - p, axis=1)))
        c = _closest_on_triangle(p, tris[j])
        out[i] = c
    return out


def ring_param(points, *, axis=None, center=None) -> np.ndarray:
    """闭环的角参数（0..2π），用于把"一段"定义成角度窗。"""
    pts = np.asarray(points, float).reshape(-1, 3)
    origin = np.asarray(center, float) if center is not None else pts.mean(axis=0)
    z = np.asarray(axis, float) if axis is not None else np.array([0.0, 0.0, 1.0])
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(helper @ z)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    x = helper - z * float(helper @ z)
    x /= max(float(np.linalg.norm(x)), 1e-12)
    y = np.cross(z, x)
    rel = pts - origin
    return np.arctan2(rel @ y, rel @ x)


def arc_indices(theta: np.ndarray, theta_lo: float, theta_hi: float) -> np.ndarray:
    """环上落在 ``[theta_lo, theta_hi]`` 角度窗内的下标（自动处理跨 0 的环绕）。"""
    theta = np.asarray(theta, float) % (2 * np.pi)
    lo = float(theta_lo) % (2 * np.pi)
    hi = float(theta_hi) % (2 * np.pi)
    if lo <= hi:
        return np.flatnonzero((theta >= lo) & (theta <= hi))
    return np.flatnonzero((theta >= lo) | (theta <= hi))


def _resample_arc(arc, n: int) -> np.ndarray:
    """把一段折线按弧长重采样成 ``n`` 个点（含两端）。"""
    arc = np.asarray(arc, float).reshape(-1, 3)
    if len(arc) < 2 or n < 2:
        return arc
    seg = np.linalg.norm(np.diff(arc, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(seg)]
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(arc[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    return np.c_[np.interp(targets, cumulative, arc[:, 0]),
                 np.interp(targets, cumulative, arc[:, 1]),
                 np.interp(targets, cumulative, arc[:, 2])]


def retrace_with_anchors(tris_or_mesh, points, anchors, *, axis=None, mode="SUPRA",
                         radius_mm=1.0, smooth_passes=1) -> dict:
    """**锚点辅助重描**：锚点作为**硬约束**，只重描锚点之间的弧，其余点逐位不变。

    规则（可审计）：
    1. 锚点投影到网格表面，成为该弧上的**固定点**（算法不得忽略）；
    2. 两锚点之间的点先按特征吸附（``snap_to_feature``）再重采样到与原来同样的点数；
    3. 平滑**只在弧内**做，弧外点原样返回。
    """
    base = np.asarray(points, float).reshape(-1, 3).copy()
    if len(base) < 8:
        return {"status": "REVIEW_REQUIRED", "reason": "MARGIN_TOO_FEW_POINTS",
                "points": base.tolist(), "moved_index": []}
    anchor_array = np.asarray(anchors, float).reshape(-1, 3)
    if not len(anchor_array):
        return {"status": "APPLIED", "reason": "NO_ANCHORS", "points": base.tolist(),
                "moved_index": []}
    tris = _as_tris(tris_or_mesh)
    projected = project_to_surface(tris, anchor_array)
    theta = ring_param(base, axis=axis)
    # 锚点按角向排序，相邻锚点（含首尾环绕）之间的扇区就是被重描的弧
    anchor_theta = ring_param(projected, axis=axis, center=base.mean(axis=0))
    anchor_order = np.argsort(anchor_theta % (2 * np.pi))
    fixed = {}
    anchor_write = {}
    moved = set()
    out = base.copy()
    n = len(base)
    for slot, idx in enumerate(anchor_order):
        theta_here = float(anchor_theta[idx] % (2 * np.pi))
        # 找环上最靠近该角度的点下标，把它替换成锚点投影（硬约束）
        nearest = int(np.argmin(np.abs(np.angle(np.exp(1j * (theta - theta_here))))))
        out[nearest] = projected[idx]
        anchor_write[nearest] = projected[idx]
        fixed[nearest] = True
        moved.add(nearest)
    # 弧按**环上索引序**（= 角向序）成对划分；锚点写入放在**最后**，保证不会被吸附覆盖
    fixed_idx = sorted(fixed)
    for slot in range(len(fixed_idx)):
        i0 = fixed_idx[slot]
        i1 = fixed_idx[(slot + 1) % len(fixed_idx)]
        span = (i1 - i0) % n
        if span <= 2:
            continue
        inner = [(i0 + k) % n for k in range(1, span)]
        arc = np.vstack([out[i0], base[inner], out[i1]])
        snapped = snap_to_feature(tris, arc.tolist(), axis, mode, radius_mm,
                                  smooth_passes=smooth_passes)
        arc_new = np.asarray(snapped.get("points") or arc, float)
        arc_new = _resample_arc(arc_new, len(arc))
        for k, point in enumerate(arc_new[1:-1], start=1):
            out[inner[k - 1]] = point
            moved.add(inner[k - 1])
    # 硬约束**最后写入**：锚点不会被吸附/重采样覆盖
    for index, point in anchor_write.items():
        out[index] = point
    # 去重：Margin 校验器要求"点唯一 + 相邻间距 ≥ 0.02mm"（见 `_validated_margin_points`），
    # 阈取 0.03mm 留余量；**锚点一律保留**（硬约束优先于均匀性）。
    MIN_SPACING_MM = 0.03
    keep = []
    for i in range(n):
        if i in anchor_write:
            keep.append(i)
            continue
        if keep and float(np.linalg.norm(out[i] - out[keep[-1]])) < MIN_SPACING_MM:
            moved.discard(i)
            continue
        keep.append(i)
    if len(keep) >= 12:
        out = out[keep]
    return {"status": "APPLIED", "reason": "ANCHORS_APPLIED",
            "points": out.tolist(), "moved_index": sorted(i for i in moved if i < len(out)),
            "anchors_used": int(len(anchor_array)),
            "anchor_hard_constraint": "written_after_snap_and_resample"}


def replace_arc(tris_or_mesh, points, stroke, *, axis=None, radius_mm=1.5,
                mode="SUPRA", smooth_passes=1) -> dict:
    """**磁性段替换**：只用 stroke 覆盖的区间替换 Margin，远端逐位锁死。

    覆盖判定用"stroke 的每一段到 Margin 点的距离 ≤ radius_mm"，
    再取这些下标在环上的**连续**部分（避免零散点被替换）。
    """
    base = np.asarray(points, float).reshape(-1, 3).copy()
    stroke_array = np.asarray(stroke, float).reshape(-1, 3)
    if len(stroke_array) < 2:
        return {"status": "REVIEW_REQUIRED", "reason": "STROKE_TOO_SHORT",
                "points": base.tolist(), "moved_index": []}
    tris = _as_tris(tris_or_mesh)
    # 逐段最近距离
    near = np.zeros(len(base), bool)
    for a, b in zip(stroke_array, stroke_array[1:]):
        seg = b - a
        length2 = float(seg @ seg)
        for i, p in enumerate(base):
            t = 0.0 if length2 <= 1e-12 else float(np.clip((p - a) @ seg / length2, 0.0, 1.0))
            if float(np.linalg.norm(p - (a + t * seg))) <= radius_mm:
                near[i] = True
    if not near.any():
        return {"status": "REVIEW_REQUIRED", "reason": "STROKE_MISSES_MARGIN",
                "points": base.tolist(), "moved_index": []}
    # 取环上连续段（以 near 的最大连通块为被替换区间）
    theta = ring_param(base, axis=axis)
    order = np.argsort(theta % (2 * np.pi))
    flags = near[order]
    best_lo = best_len = cur_lo = 0
    length = 0
    for k in range(2 * len(flags)):
        if flags[k % len(flags)]:
            if length == 0:
                cur_lo = k
            length += 1
            if length > best_len:
                best_len, best_lo = length, cur_lo
        else:
            length = 0
    best_len = min(best_len, len(flags))
    replaced = [int(order[(best_lo + k) % len(order)]) for k in range(best_len)]
    idx_sorted = sorted(replaced)
    i0, i1 = idx_sorted[0], idx_sorted[-1]
    # 新段 = stroke 投影到网格 + 两端与原 Margin 对齐
    new_arc = project_to_surface(tris, stroke_array)
    new_arc = _resample_arc(new_arc, len(idx_sorted))
    snapped = snap_to_feature(tris, new_arc.tolist(), axis, mode, radius_mm,
                              smooth_passes=smooth_passes)
    new_arc = np.asarray(snapped.get("points") or new_arc, float)
    out = base.copy()
    for k, index in enumerate(idx_sorted):
        out[index] = new_arc[k]
    out[i0], out[i1] = base[i0], base[i1]          # 接缝处保持连续
    moved = set(idx_sorted)
    untouched = sorted(set(range(len(base))) - moved)
    return {"status": "APPLIED", "reason": "SEGMENT_REPLACED",
            "points": out.tolist(), "moved_index": sorted(moved),
            "replaced_arc_length": int(best_len), "untouched_index": untouched}


def move_along_surface(tris_or_mesh, points, *, axis, delta_mm,
                       radius_mm=None) -> dict:
    """**整体上移/下移**：沿"局部表面切平面内的插入轴投影"移动，再投影回网格。

    刻意**不是** world-Z 平移：先算每点的表面法向，把插入轴投影到该点的切平面得到
    局部移动方向（牙颈缘是三维曲线，world-Z 会在斜坡上把曲线推离表面），
    移动 ``delta_mm`` 后重新投影回网格表面。
    """
    base = np.asarray(points, float).reshape(-1, 3).copy()
    if not len(base):
        return {"status": "REVIEW_REQUIRED", "reason": "EMPTY", "points": [],
                "moved_index": []}
    tris = _as_tris(tris_or_mesh)
    centroids = tris.mean(axis=1)
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals_len = np.linalg.norm(face_normals, axis=1)
    face_normals = face_normals / np.maximum(normals_len[:, None], 1e-12)
    z = np.asarray(axis, float)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    steps = max(1, int(np.ceil(abs(float(delta_mm)) / 0.05)))     # 每步 ≤0.05mm
    step_mm = float(delta_mm) / steps
    # 局部法向必须**局部**：半径给大一点会把棱两侧的法向平均掉，
    # 于是"沿表面移动"在倒角/台阶棱上退化成随机方向（实测环被压成 97/240 点）。
    patch_radius = min(float(radius_mm or 0.6), 0.6)
    out = base.copy()
    moved = []
    stalled = 0
    for i, p in enumerate(base):
        current = p.copy()
        ids = _local_faces(centroids, current, patch_radius)
        if not len(ids):
            continue
        for _ in range(steps):
            # 局部法向 → 把插入轴投影到该点的切平面 = 沿表面的移动方向
            near_ids = _local_faces(centroids, current, patch_radius)
            if not len(near_ids):
                near_ids = ids
            # 局部法向：**面积加权的主簇方向**（颈缘本来就常落在颈缘棱上，
            # 单纯逐面平均会被两侧法向互相抵消；取主簇 + 符号统一后再平均更稳）。
            normals_here = face_normals[near_ids]
            areas_here = np.linalg.norm(
                np.cross(tris[near_ids][:, 1] - tris[near_ids][:, 0],
                         tris[near_ids][:, 2] - tris[near_ids][:, 0]), axis=1)
            weights = np.maximum(areas_here, 1e-12)
            reference = normals_here[int(np.argmax(weights))]
            aligned = normals_here * np.sign(normals_here @ reference)[:, None]
            normal = (aligned * weights[:, None]).sum(axis=0)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            # 只在**真正退化**时拒绝：主导方向占比过低（没有主簇）
            dominant = float(weights[np.sign(normals_here @ normal) > 0].sum() / weights.sum())
            if dominant < 0.6:
                out = base.copy()
                return {"status": "REVIEW_REQUIRED",
                        "reason": "EDGE_OR_CREASE_LOCAL_NORMAL_AMBIGUOUS",
                        "points": out.tolist(), "moved_index": [],
                        "direction": "not_applied_ambiguous_local_tangent",
                        "dominant_ratio": dominant,
                        "hint": "该处曲率方向不唯一：沿表面移动没有唯一定义，请改用锚点或磁性修线"}
            tangent = z - normal * float(z @ normal)
            norm = float(np.linalg.norm(tangent))
            if norm < 1e-9:
                break
            target = current + (tangent / norm) * step_mm
            # **在表面上前进**：投影到局部最近面（而不是整张网格的最近点，
            # 否则"投影回最近面"会把刚走出的位移抹掉 —— 实测位移只剩 0.05mm/0.3mm）
            j = int(np.argmin(np.linalg.norm(centroids[near_ids] - target, axis=1)))
            current = _closest_on_triangle(target, tris[near_ids[j]])
        displacement = float(np.linalg.norm(current - p))
        if displacement < 0.2 * abs(float(delta_mm)):
            stalled += 1
        out[i] = current
        moved.append(i)
    status = "APPLIED" if stalled < 0.5 * len(base) else "REVIEW_REQUIRED"
    reason = "MOVED_ALONG_SURFACE" if status == "APPLIED" else "MOVE_LIMITED_BY_SURFACE"
    return {"status": status, "reason": reason,
            "points": out.tolist(), "moved_index": moved,
            "stalled_count": stalled,
            "direction": "local_surface_projected_insertion_axis_not_world_z"}


def smooth_ring(points, *, strength=0.5, locked_index=()) -> dict:
    """保锁平滑：只对未锁点做一轮邻域加权（闭环），锁点逐位不变。"""
    base = np.asarray(points, float).reshape(-1, 3).copy()
    if len(base) < 5:
        return {"status": "REVIEW_REQUIRED", "reason": "MARGIN_TOO_FEW_POINTS",
                "points": base.tolist(), "moved_index": []}
    locked = set(int(i) for i in locked_index)
    alpha = float(np.clip(strength, 0.0, 1.0))
    out = base.copy()
    moved = []
    for i in range(len(base)):
        if i in locked:
            continue
        prev = base[(i - 1) % len(base)]
        nxt = base[(i + 1) % len(base)]
        target = 0.5 * (prev + nxt)
        out[i] = base[i] + alpha * (target - base[i])
        moved.append(i)
    return {"status": "APPLIED", "reason": "SMOOTHED", "points": out.tolist(),
            "moved_index": moved, "locked_index": sorted(locked)}
