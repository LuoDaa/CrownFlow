"""Finalize 前的颈缘网格清理（只作用于 crown cervical 区域）。

为什么需要它（实测依据）：stitch 的输入面本身就是"未焊接 + 含 sliver"的网格
（outer 19,462 顶点里有 8,122 对 <0.08mm 的近重复点；5 处自交面的短边 0.04–0.17mm
就在面内部）。stitch 只能忠实围绕这些面造三角，两条缝合带必然互穿。
在**进入 stitch 之前**把它清干净，是唯一不触碰 stitch 内部、也不碰 margin/support/cut
和 validation gate 的做法。

只做两件事，都在**颈缘区域局部**完成（绝不全局 remesh）：
1. 顶点焊接：合并空间距离 < ``weld_mm`` 的重复顶点（更新面索引、删除重复顶点）；
2. sliver 清理：对"高长宽比 + 小面积"的三角面，折叠其最短边并重新三角化局部 patch。

安全底线：每一步之后都检查"边界环数量不变"（这些面是单颈环开口面），
一旦拓扑被改变就回退并如实报告，绝不静默改坏网格。
"""
from __future__ import annotations

import numpy as np
import trimesh

from .margin_seam import _polyline_distances

WELD_MM = 0.05
ASPECT_MAX = 10.0
AREA_MIN_MM2 = 0.5
REGION_BAND_MM = 3.0
MAX_PASSES = 3


def mesh_diagnostics(mesh, seam_curve, *, weld_mm=WELD_MM, aspect_max=ASPECT_MAX, area_min=AREA_MIN_MM2):
    """清理前后对比用的只读统计。"""
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    loops = _boundary_loop_count(faces)
    from scipy.spatial import cKDTree
    pairs = 0
    if len(vertices):
        pairs = len(cKDTree(vertices).query_pairs(float(weld_mm)))
    tri = vertices[faces] if len(faces) else np.zeros((0, 3, 3))
    if len(tri):
        edges = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                          np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                          np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)], axis=1)
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        aspect = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)
        slivers = int(((aspect > aspect_max) & (area < area_min)).sum())
        min_edge = float(edges.min())
    else:
        slivers, min_edge = 0, None
    region = cervical_region_faces(mesh, seam_curve)
    return {"vertices": int(len(vertices)), "faces": int(len(faces)), "boundary_loops": int(loops),
            "duplicate_pairs_below_%.2fmm" % weld_mm: int(pairs), "sliver_faces": slivers,
            "cervical_region_faces": int(region.sum()), "min_edge_mm": min_edge}


def _ring_vertices(faces) -> np.ndarray:
    """属于边界边的顶点（这些环要留给 stitch 拉链，清理时不许合并）。"""
    faces = np.asarray(faces, np.int64)
    mask = np.zeros(int(faces.max()) + 1 if len(faces) else 0, bool)
    if not len(faces):
        return mask
    from collections import Counter
    counter = Counter()
    for a, b, c in faces:
        for u, w in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            counter[(min(u, w), max(u, w))] += 1
    boundary = [e for e, n in counter.items() if n == 1]
    if boundary:
        ids = np.unique(np.asarray(boundary, np.int64))
        mask[ids] = True
    return mask


def _boundary_loop_count(faces) -> int:
    from .crown_outer.collar_body_outer import _boundary_loops
    return len(_boundary_loops(np.asarray(faces, np.int64)))


def cervical_region_faces(mesh, seam_curve, *, band_mm=REGION_BAND_MM):
    """颈缘区域：任一顶点离确认终止线 < band_mm 的面。"""
    vertices = np.asarray(mesh.vertices, float)
    if not len(vertices) or not len(mesh.faces):
        return np.zeros(len(mesh.faces), bool)
    distance = np.asarray(_polyline_distances(vertices, np.asarray(seam_curve, float)))
    near = distance < band_mm
    return np.any(near[np.asarray(mesh.faces, np.int64)], axis=1)


def _collapse_once(mesh, candidate_pairs, *, region_faces):
    """把候选顶点对合并（保留一个代表点），重映射面索引并丢弃因此退化的面。

    只有在"边界环数量不变"时才算成功；调用方负责回退判定。
    """
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    region_vertices = np.zeros(len(vertices), bool)
    region_vertices[np.unique(faces[region_faces].ravel())] = True
    # ⚠️ 边界环顶点一律不动：它们是 stitch 要拉链的那条环，动它会让缝合结果开壳/断连
    # （实测：允许动环顶点 → "Cervical stitch produced an open or disconnected shell"）。
    ring = _ring_vertices(faces)
    region_vertices &= ~ring
    collapsed = 0
    for a, b in candidate_pairs:
        a, b = int(a), int(b)
        if not (region_vertices[a] and region_vertices[b]):
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        collapsed += 1
    if not collapsed:
        return mesh, 0, 0
    remap = np.array([find(i) for i in range(len(vertices))], np.int64)
    new_faces = remap[faces]
    degenerate = ((new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2])
                  | (new_faces[:, 0] == new_faces[:, 2]))
    kept = new_faces[~degenerate]
    rebuilt = trimesh.Trimesh(vertices=vertices, faces=kept, process=False)
    rebuilt.remove_unreferenced_vertices()
    return rebuilt, collapsed, int(degenerate.sum())


def clean_cervical_region(mesh, seam_curve, *, weld_mm=WELD_MM, aspect_max=ASPECT_MAX,
                          area_min=AREA_MIN_MM2, band_mm=REGION_BAND_MM, max_passes=MAX_PASSES):
    """焊接 + sliver 折叠（局部）。返回 ``(cleaned_mesh, report)``；任何拓扑变化都回退。"""
    before = mesh_diagnostics(mesh, seam_curve, weld_mm=weld_mm, aspect_max=aspect_max, area_min=area_min)
    original_loops = before["boundary_loops"]
    current = mesh
    total_collapsed = 0
    total_dropped = 0
    reverted = None
    for _ in range(max_passes):
        vertices = np.asarray(current.vertices, float)
        faces = np.asarray(current.faces, np.int64)
        region = cervical_region_faces(current, seam_curve, band_mm=band_mm)
        if not region.any() or not len(faces):
            break
        from scipy.spatial import cKDTree
        pairs = [tuple(int(x) for x in pair) for pair in sorted(cKDTree(vertices).query_pairs(float(weld_mm)))] \
            if len(vertices) else []
        # sliver：折叠"高长宽比 + 小面积"面的最短边
        region_ids = np.flatnonzero(region)
        tri = vertices[faces[region_ids]]
        edges = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                          np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                          np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)], axis=1)
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        aspect = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)
        bad = (aspect > aspect_max) & (area < area_min)
        for face in faces[region_ids[bad]]:
            tri_ids = [int(i) for i in face]
            pairs.append((tri_ids[0], tri_ids[1]))
            pairs.append((tri_ids[1], tri_ids[2]))
            pairs.append((tri_ids[2], tri_ids[0]))
        if not pairs:
            break
        candidate, collapsed, dropped = _collapse_once(current, pairs, region_faces=region)
        if not collapsed:
            break
        if _boundary_loop_count(candidate.faces) != original_loops:
            reverted = {"reason": "boundary_loop_count_changed", "collapsed": collapsed,
                        "loop_count_after": int(_boundary_loop_count(candidate.faces)),
                        "loop_count_before": int(original_loops)}
            break
        current = candidate
        total_collapsed += collapsed
        total_dropped += dropped
    after = mesh_diagnostics(current, seam_curve, weld_mm=weld_mm, aspect_max=aspect_max, area_min=area_min)
    report = {"algorithm": "cervical_cleanup/weld_and_sliver_collapse/1", "before": before, "after": after,
              "collapsed_vertices": int(total_collapsed), "dropped_degenerate_faces": int(total_dropped),
              "weld_mm": weld_mm, "aspect_max": aspect_max, "area_min_mm2": area_min,
              "region_band_mm": band_mm, "compensated": {
                  "vertices_removed": int(before["vertices"] - after["vertices"]),
                  "faces_removed": int(before["faces"] - after["faces"]),
                  "duplicate_pairs_before": before["duplicate_pairs_below_%.2fmm" % weld_mm],
                  "duplicate_pairs_after": after["duplicate_pairs_below_%.2fmm" % weld_mm],
                  "sliver_before": before["sliver_faces"], "sliver_after": after["sliver_faces"]},
              "reverted": reverted, "scope": "CROWN_CERVICAL_REGION_ONLY",
              "margin_curve_untouched": True, "status": "REVERTED" if reverted else "APPLIED"}
    return current, report


# ── 干净颈环（arc-length resample）+ 局部 collar 带 ──────────────────────────────
def _ring_polyline(mesh, ring_ids):
    points = np.asarray(mesh.vertices, float)[np.asarray(ring_ids, np.int64)]
    closed = np.vstack([points, points[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    return points, arc


def build_clean_cervical_ring(mesh, *, samples=128, min_samples=24):
    """从 mesh 的开口环生成 clean cervical ring：沿原环**弧长等距**重采样固定数量点。

    只读原环（不删、不动原顶点），返回 ``(ring_ids, new_points, report)``。
    ``samples`` 会被限制在 ``<= 原环点数``，避免在弦上生成近零面积三角形。
    """
    from .crown_outer.collar_body_outer import _boundary_loops
    faces = np.asarray(mesh.faces, np.int64)
    loops = _boundary_loops(faces)
    if len(loops) != 1:
        raise ValueError("干净颈环重建要求恰好一个开口环，实际 %d 个" % len(loops))
    ring_ids = np.asarray(loops[0], np.int64)
    points, arc = _ring_polyline(mesh, ring_ids)
    total = float(arc[-1])
    count = int(max(min_samples, min(samples, len(ring_ids))))
    if total <= 1e-9 or count < 3:
        raise ValueError("颈环弧长退化（%.4g mm），无法重采样" % total)
    targets = np.linspace(0.0, total, count, endpoint=False)
    closed = np.vstack([points, points[:1]])
    segments = np.diff(closed, axis=0)
    seg_len = np.linalg.norm(segments, axis=1)
    new_points = []
    for value in targets:
        index = int(np.searchsorted(arc, value, side="right") - 1)
        index = int(np.clip(index, 0, len(segments) - 1))
        local = (value - arc[index]) / max(seg_len[index], 1e-12)
        new_points.append(closed[index] + local * segments[index])
    new_points = np.asarray(new_points, float)
    old_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    report = {"old_ring_vertex_count": int(len(ring_ids)), "new_ring_vertex_count": int(count),
              "old_min_edge_mm": round(float(old_lengths.min()), 6),
              "old_median_edge_mm": round(float(np.median(old_lengths)), 6),
              "new_uniform_spacing_mm": round(total / count, 6),
              "ring_length_mm": round(total, 4),
              "edge_uniformity_ratio": round(float(np.median(old_lengths) / max(total / count, 1e-9)), 3),
              "margin_curve_untouched": True}
    return ring_ids, new_points, report


def attach_clean_ring_collar(mesh, ring_ids, new_points):
    """用局部 collar 带把"原环 → clean ring"接起来，使 clean ring 成为新的开口环。

    主体网格一个顶点都不动（只追加新顶点 + 追加面），watertight/拓扑由
    lattice zipper 本身保证：原环边恰好被 collar 消费一次，clean ring 成为新边界。
    """
    from .crown_outer.collar_body_outer import (_align_rings, _zip_path, _tris_from_path,
                                                _boundary_loops)
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    extended = np.vstack([vertices, new_points])
    clean_ids = np.arange(len(vertices), len(vertices) + len(new_points), dtype=np.int64)
    a, b, arca, arcb = _align_rings(extended, np.asarray(ring_ids, np.int64), clean_ids)
    path = _zip_path(arca, arcb, len(a), len(b))
    tris = _tris_from_path(a, b, path)
    stitched = trimesh.Trimesh(extended, np.vstack([faces, tris]), process=False)
    stitched.remove_unreferenced_vertices()
    loops_after = len(_boundary_loops(np.asarray(stitched.faces, np.int64)))
    return stitched, {"collar_faces": int(len(tris)), "boundary_loops_after": int(loops_after),
                      "collar_new_vertices": int(len(new_points)),
                      "original_body_vertices_untouched": int(len(vertices))}
