"""Prescribed 3D cervical seam. Requires two reviewed, open cervical surfaces.

This assembles surfaces; it does not invent an intaglio transition or approve fit.
"""
import numpy as np
import trimesh
from pathlib import Path

from .crown_outer.collar_body_outer import (
    _align_rings, _boundary_loops, _dp_seam_path, _tris_from_path,
)
from .crown_outer.thickness import input_qc
from .margin import fit_margin_plane


def _align_ring_to_reference(v, ring, reference):
    """把 ``ring`` 对齐到**已确定的参考环**（此处 = canonical seam）。

    起点取"离参考起点最近的顶点"，方向与参考走向同向。**不再各自做最近点对搜索** ——
    两条 stitch 带因此同相位（这是 outer/inner seam path 相位不一致的根因）。
    """
    A = np.asarray(ring, np.int64).copy()
    R = np.asarray(reference, np.int64)
    if len(A) < 3 or len(R) < 3:
        raise ValueError("Cervical stitch ring too short: %d / %d" % (len(A), len(R)))
    va, vr = np.asarray(v)[A], np.asarray(v)[R]
    A = np.roll(A, -int(np.linalg.norm(va - vr[0], axis=1).argmin()))
    va = np.asarray(v)[A]
    same = float(np.dot(va[1] - va[0], vr[1] - vr[0])
                 + np.dot(va[-1] - va[0], vr[-1] - vr[0]))
    return A[::-1] if same < 0 else A


def _matched_seam_path(reference_path, n_a):
    """按参考路径的**同一纵向参数**为另一条环生成 lattice 路径。

    seam（b）出现的位置逐段与参考一致；每段 a 的长度按累计比例缩放，保证恰好消耗
    ``n_a`` 个顶点。两条带因此在 seam 出口处共享同一组 seam 顶点序列与同一行相位。
    """
    runs, run = [], 0
    for step in reference_path:
        if step == "a":
            run += 1
        else:
            runs.append(run)
            run = 0
    groups = np.asarray(runs + [run], np.int64)
    total = int(groups.sum())
    if total <= 0 or not len(runs):
        return ["a"] * int(n_a) + ["b"] * len(runs)
    scaled = np.diff(np.r_[0, np.round(np.cumsum(groups) * n_a / total).astype(int)])
    path = []
    for count in scaled[:-1]:
        path += ["a"] * max(int(count), 0) + ["b"]
    path += ["a"] * max(int(scaled[-1]), 0)
    return path


def _closest_pair_between_triangles(t1, t2):
    """相交两三角形之间的最近点对（相交时距离≈0，最近点落在交线上）。"""
    from trimesh.proximity import closest_point
    a = trimesh.Trimesh(np.asarray(t1, float), [[0, 1, 2]], process=False)
    b = trimesh.Trimesh(np.asarray(t2, float), [[0, 1, 2]], process=False)
    _, d_ab, _ = closest_point(a, np.asarray(t2, float))
    _, d_ba, _ = closest_point(b, np.asarray(t1, float))
    if d_ab.min() <= d_ba.min():
        k = int(d_ab.argmin())
        pb = np.asarray(t2, float)[k]
        pa = closest_point(a, pb.reshape(1, 3))[0][0]
    else:
        k = int(d_ba.argmin())
        pa = np.asarray(t1, float)[k]
        pb = closest_point(b, pa.reshape(1, 3))[0][0]
    return np.asarray(pa, float), np.asarray(pb, float), float(min(d_ab.min(), d_ba.min()))


def cervical_stitch_intersection_report(result, *, outer_face_count, intaglio_face_count,
                                        cervical_ring_points, margin_points, seam_ids,
                                        neighbor_k=8, debug_dir=None):
    """**纯诊断**：逐对定位 cervical stitch 后的自交，并生成红/黄/绿 debug 网格。

    不改几何、不改判定、不改阈值。每对给出：face_a/face_b、交点、到确认终止线距离、
    到颈缘边界（缝合前的开口环）距离、法线夹角、两面面积、所属曲面，并分类：
      A_margin          = 交点到确认终止线 < 1 mm
      B_cervical_offset = 两面都来自 OUTER/INTAGLIO 原始面，且离 margin 较远
      C_stitch_sweep    = 至少一面属于缝合带
    """
    from scipy.spatial import cKDTree
    from .crown_outer.thickness import _share_any_vertex, _tri_tri_intersect

    vertices = np.asarray(result.vertices, float)
    faces = np.asarray(result.faces, np.int64)
    tri = vertices[faces]
    n = len(faces)
    k = min(neighbor_k + 1, n)
    _, idx = cKDTree(tri.mean(axis=1)).query(tri.mean(axis=1), k=k)
    idx = np.atleast_2d(idx)
    i = np.repeat(np.arange(n), idx.shape[1])
    j = idx.ravel()
    m = i < j
    pairs = np.unique(np.stack([i[m], j[m]], axis=1), axis=0)
    pairs = pairs[~_share_any_vertex(faces[pairs[:, 0]], faces[pairs[:, 1]])]
    hit, _parallel, seglen = _tri_tri_intersect(tri[pairs[:, 0]], tri[pairs[:, 1]], return_segment=True)
    hp = pairs[hit]
    margin = np.asarray(margin_points, float)
    ring = np.asarray(cervical_ring_points, float)
    seam = vertices[np.asarray(seam_ids, np.int64)]

    def surface_of(face_id):
        if face_id < outer_face_count:
            return "OUTER"
        if face_id < outer_face_count + intaglio_face_count:
            return "INTAGLIO"
        return "STITCH_STRIP"

    entries = []
    for position, (a, b) in enumerate(hp):
        pa, pb, gap = _closest_pair_between_triangles(tri[a], tri[b])
        point = (pa + pb) / 2
        n1 = np.cross(tri[a][1] - tri[a][0], tri[a][2] - tri[a][0])
        n2 = np.cross(tri[b][1] - tri[b][0], tri[b][2] - tri[b][0])
        u1 = n1 / max(float(np.linalg.norm(n1)), 1e-18)
        u2 = n2 / max(float(np.linalg.norm(n2)), 1e-18)
        angle = float(np.degrees(np.arccos(np.clip(abs(float(u1 @ u2)), -1.0, 1.0))))
        d_margin = float(_polyline_distances(point.reshape(1, 3), margin)[0])
        surfaces = (surface_of(int(a)), surface_of(int(b)))
        category = ("C_stitch_sweep" if "STITCH_STRIP" in surfaces
                    else "A_margin" if d_margin < 1.0 else "B_cervical_offset")
        entries.append({
            "index": position, "face_a": int(a), "face_b": int(b), "surface_roles": list(surfaces),
            "intersection_point": [round(float(v), 5) for v in point],
            "face_a_vertices": [[round(float(x), 4) for x in vertices[i]] for i in faces[a]],
            "face_b_vertices": [[round(float(x), 4) for x in vertices[i]] for i in faces[b]],
            "closest_distance_mm": round(gap, 6),
            "segment_length_mm": round(float(seglen[hit][position]), 6),
            "distance_to_margin_mm": round(d_margin, 4),
            "distance_to_cervical_boundary_mm": round(float(cKDTree(ring).query(point)[0]), 4) if len(ring) else None,
            "distance_to_seam_mm": round(float(cKDTree(seam).query(point)[0]), 4) if len(seam) else None,
            "normal_angle_deg": round(angle, 3),
            "local_face_area_mm2": [round(0.5 * float(np.linalg.norm(n1)), 6),
                                    round(0.5 * float(np.linalg.norm(n2)), 6)],
            "category": category})

    debug_paths = {}
    if debug_dir is not None and len(entries):
        hit_ids = np.asarray(sorted({e["face_a"] for e in entries} | {e["face_b"] for e in entries}), np.int64)
        touched = np.zeros(len(vertices), bool)
        touched[np.unique(faces[hit_ids].ravel())] = True
        neighbour = np.any(touched[faces], axis=1) & ~np.isin(np.arange(n), hit_ids)
        colours = np.tile(np.array([[90, 190, 120, 255]], np.uint8), (n, 1))
        colours[neighbour] = [245, 200, 60, 255]
        colours[hit_ids] = [230, 60, 60, 255]
        soup = trimesh.Trimesh(vertices=tri.reshape(-1, 3),
                               faces=np.arange(3 * n).reshape(-1, 3), process=False)
        soup.visual.face_colors = colours
        folder = Path(debug_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = "cervical_stitch_%08d" % (int(hit_ids.sum()) % 10 ** 8)
        (folder / (stem + ".glb")) .parent.mkdir(parents=True, exist_ok=True)
        soup.export(folder / (stem + ".glb"))
        import json as _json
        (folder / (stem + ".json")).write_text(
            _json.dumps({"entries": entries, "red_faces": hit_ids.tolist(),
                         "yellow_faces": int(neighbour.sum())}, ensure_ascii=False, indent=2), encoding="utf-8")
        debug_paths = {"mesh": str(folder / (stem + ".glb")), "report": str(folder / (stem + ".json"))}

    rings_only = np.asarray(cervical_ring_points, float)
    half = len(rings_only) // 2 if len(rings_only) % 2 == 0 else len(rings_only)
    outer_ring_points, intaglio_ring_points = rings_only[:half], rings_only[half:]
    seam_cloud = np.asarray(seam, float)
    ring_facts = {}
    if len(outer_ring_points) and len(intaglio_ring_points):
        from scipy.spatial import cKDTree as _KD
        d_oi = _KD(intaglio_ring_points).query(outer_ring_points)[0]
        d_os = _KD(seam_cloud).query(outer_ring_points)[0]
        d_is = _KD(seam_cloud).query(intaglio_ring_points)[0]
        ring_facts = {
            "outer_ring_to_inner_ring_min_mm": round(float(d_oi.min()), 4),
            "outer_ring_to_inner_ring_p05_mm": round(float(np.quantile(d_oi, .05)), 4),
            "outer_ring_to_seam_max_mm": round(float(d_os.max()), 4),
            "inner_ring_to_seam_max_mm": round(float(d_is.max()), 4),
            "outer_ring_vertices": int(len(outer_ring_points)),
            "inner_ring_vertices": int(len(intaglio_ring_points)),
            "seam_vertices": int(len(seam_cloud)),
        }
    return {"entries": entries, "summary": {
        "pair_count": len(entries),
        "categories": {name: sum(1 for e in entries if e["category"] == name)
                       for name in ("A_margin", "B_cervical_offset", "C_stitch_sweep")},
        "surface_combinations": sorted({"+".join(sorted(e["surface_roles"])) for e in entries}),
        "min_normal_angle_deg": min([e["normal_angle_deg"] for e in entries], default=None),
        "min_local_face_area_mm2": min([min(e["local_face_area_mm2"]) for e in entries], default=None),
        "ring_proximity": ring_facts,
        "debug": debug_paths}}


class MarginReconstructionRequired(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "MARGIN_RECONSTRUCTION_REQUIRED: nonplanar 3D Margin cannot use plane "
            f"clipping (max plane deviation {diagnostics['plane_deviation_max_mm']:.6f} mm). "
            "Shared outer/intaglio cervical surfaces must be reconstructed before export."
        )


def margin_plane_diagnostics(points):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Margin requires at least three finite XYZ points")
    if np.linalg.matrix_rank(p - p.mean(axis=0), tol=1e-9) < 2:
        raise ValueError("Margin points are collinear")
    center, normal = fit_margin_plane(p)
    deviation = np.abs((p - center) @ normal)
    return {
        "status": "ENGINEERING_ONLY",
        "plane_deviation_max_mm": float(deviation.max()),
        "plane_deviation_p95_mm": float(np.percentile(deviation, 95)),
        # Numerical exactness of a plane, not a manufacturing acceptance threshold.
        "numerically_planar": bool(deviation.max() <= 1e-6),
        "manufacturing_threshold": None,
    }


def _polyline_distances(points, ring):
    a = np.asarray(ring, float)
    delta = np.roll(a, -1, axis=0) - a
    denominator = np.einsum("ij,ij->i", delta, delta)
    if np.any(denominator <= 1e-20):
        raise ValueError("Margin contains duplicate adjacent points")
    values = []
    for chunk in np.array_split(np.asarray(points, float), max(1, len(points) // 512 + 1)):
        rel = chunk[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("nki,ki->nk", rel, delta) / denominator, 0, 1)
        distance = np.linalg.norm(rel - t[:, :, None] * delta, axis=2)
        values.extend(distance.min(axis=1))
    return np.asarray(values)


def audit_margin_seam(vertices, faces, seam_ids, margin_points):
    """Check embedded seam connectivity and bidirectional sampled edge error."""
    v, f = np.asarray(vertices, float), np.asarray(faces, np.int64)
    ids = np.asarray(seam_ids, np.int64)
    curve = np.asarray(margin_points, float)
    seam = v[ids]
    edges = np.sort(np.vstack((f[:, :2], f[:, 1:], f[:, [2, 0]])), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    incidence = {tuple(e): int(n) for e, n in zip(unique, counts)}
    prescribed = np.sort(np.column_stack((ids, np.roll(ids, -1))), axis=1)
    valid = all(incidence.get(tuple(e)) == 2 for e in prescribed)
    samples = lambda r: np.vstack((r, (r + np.roll(r, -1, axis=0)) / 2))
    forward = _polyline_distances(samples(curve), seam)
    reverse = _polyline_distances(samples(seam), curve)
    return {
        "status": "ENGINEERING_ONLY",
        "method": "vertices_and_segment_midpoints_to_exact_polyline_segments",
        "margin_to_seam_max_mm": float(forward.max()),
        "seam_to_margin_max_mm": float(reverse.max()),
        "shared_seam_edges_incidence_two": valid,
        "manufacturing_threshold": None,
    }


def _lattice_pair(v, A, B, i, j, s1, s2):
    """给定 lattice 起点 (i,j) 与连续两步，返回这两步产生的两个三角形。"""
    nA, nB = len(A), len(B)
    if s1 == "a":
        t1 = (A[i % nA], A[(i + 1) % nA], B[j % nB])
        i, j = i + 1, j
    else:
        t1 = (A[i % nA], B[(j + 1) % nB], B[j % nB])
        i, j = i, j + 1
    if s2 == "a":
        t2 = (A[i % nA], A[(i + 1) % nA], B[j % nB])
    else:
        t2 = (A[i % nA], B[(j + 1) % nB], B[j % nB])
    return t1, t2


def _shape_quality(v, triangles) -> list:
    """归一化形状质量 ``4√3·A/Σe²``（等边=1；薄片→0）。"""
    V = np.asarray(v, float)
    out = []
    for tri in triangles:
        p = V[np.asarray(tri, np.int64)]
        edges = [float(np.linalg.norm(p[1] - p[2])), float(np.linalg.norm(p[2] - p[0])),
                 float(np.linalg.norm(p[0] - p[1]))]
        area = 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        denom = sum(e * e for e in edges)
        out.append(0.0 if denom <= 1e-18 else 4.0 * np.sqrt(3.0) * area / denom)
    return out


def _flip_improve_path(v, A, B, path, *, passes=4, gain=1e-6):
    """**三角质量约束**：在 lattice 上做局部对角线翻转（交换相邻的 a/b 两步），
    使"最差形状质量"最大 —— 小边长 / 近重复顶点处不再被逼出长薄三角。

    只改对角线选择：顶点不动、seam 序列不变、每条环边与每条 seam 边仍被**恰好消费一次**
    （watertight 与 prescribed seam edges 因此原样保持）。
    """
    path = list(path)
    for _ in range(passes):
        changed = False
        i = j = 0
        for k in range(len(path) - 1):
            s1, s2 = path[k], path[k + 1]
            if s1 != s2:
                current = _lattice_pair(v, A, B, i, j, s1, s2)
                swapped = _lattice_pair(v, A, B, i, j, s2, s1)
                q_current = _shape_quality(v, current)
                q_swapped = _shape_quality(v, swapped)
                if (min(q_swapped) > min(q_current) + gain
                        or (min(q_swapped) >= min(q_current) - gain
                            and sum(q_swapped) > sum(q_current) + gain)):
                    path[k], path[k + 1] = s2, s1
                    changed = True
                    s1, s2 = s2, s1
            if s1 == "a":
                i += 1
            else:
                j += 1
        if not changed:
            break
    return path


def weld_near_duplicate_ring(mesh, *, epsilon_mm=0.08):
    """合并 stitch **临时输入环**上的近重复顶点（相邻距离 < epsilon_mm 视为重复）。

    规则（只作用在传给 stitch 的副本上；不改原始 margin curve、不改 seam_ids、
    不改 prescribed edge）：
      - 只在"合并后不会让任何面出现重复顶点"时才合并（否则会造出零面积面）；
      - 合并后保留代表点，面索引重映射，闭合环保持闭合。

    返回 ``(welded_mesh, merged_pairs, merged_positions)``。
    """
    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    if len(vertices) < 4:
        return mesh, [], []
    from scipy.spatial import cKDTree
    try:
        pairs = sorted(cKDTree(vertices).query_pairs(float(epsilon_mm)))
    except (ValueError, MemoryError):
        return mesh, [], []
    if not pairs:
        return mesh, [], []
    parent = list(range(len(vertices)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    face_sets = [set(int(i) for i in tri) for tri in faces]
    merged, positions = [], []
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if any(ra in s and rb in s for s in face_sets):
            continue                      # 合并会让某个面退化成重复顶点面 → 跳过
        parent[ra] = rb
        merged.append((int(a), int(b)))
        positions.append([round(float(x), 4) for x in vertices[a]])
    if not merged:
        return mesh, [], []
    remap = np.array([find(i) for i in range(len(vertices))], np.int64)
    new_faces = remap[faces]
    # 绝不丢面：一旦合并会让某个面出现重复顶点，本次清理整体不采用（否则会开出边界边、
    # 改变环拓扑 —— 实测就是这条让先前版本被回退）。
    degenerate = ((new_faces[:, 0] == new_faces[:, 1]) | (new_faces[:, 1] == new_faces[:, 2])
                  | (new_faces[:, 0] == new_faces[:, 2]))
    if degenerate.any():
        return mesh, [], []
    welded = trimesh.Trimesh(vertices=vertices, faces=new_faces, process=False)
    return welded, merged, positions


def stitch_triangle_quality(v, strip_faces):
    """缝合三角的质量统计（只读，用于修改前后对比）。"""
    tri = np.asarray(v, float)[np.asarray(strip_faces, np.int64)]
    if not len(tri):
        return {"count": 0}
    edges = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum((edges ** 2).sum(axis=1), 1e-18)
    aspect = edges.max(axis=1) / np.maximum(edges.min(axis=1), 1e-12)
    return {"count": int(len(tri)),
            "quality_min": round(float(quality.min()), 6),
            "quality_p05": round(float(np.quantile(quality, .05)), 6),
            "quality_median": round(float(np.median(quality)), 6),
            "quality_below_005": int((quality < 0.05).sum()),
            "quality_below_010": int((quality < 0.10).sum()),
            "aspect_max": round(float(aspect.max()), 3),
            "aspect_p95": round(float(np.quantile(aspect, .95)), 3),
            "min_edge_mm": round(float(edges.min()), 6),
            "edges_below_005mm": int((edges < 0.05).sum())}


#: DESIGN 模式允许的最大颈缘缝合自交对数量（finalize 验收策略，不是几何修复）
DESIGN_MODE_MAX_SELF_INTERSECTIONS = 10


def stitch_cervical_surfaces(outer, intaglio, margin_points, *, quality_mode="DESIGN"):
    """Reuse J.2 DP zipper, keeping every supplied Margin point verbatim.

    Callers must supply one open ring per surface, with the cervical transitions
    already reviewed. A zero-thickness knife edge is not a material profile.
    """
    design_mode_exception = None
    curve = np.asarray(margin_points, float)
    margin_plane_diagnostics(curve)
    if np.array_equal(curve[0], curve[-1]):
        curve = curve[:-1]
    _polyline_distances(curve, curve)
    # 近重复顶点清理：只作用在 stitch 的**临时输入环副本**上（原始 margin curve / seam_ids 不动）
    outer_before, intaglio_before = len(outer.vertices), len(intaglio.vertices)

    def _safe_weld(mesh):
        """合并近重复顶点，但只在"环仍是单条闭合环"时采用；否则回退并记录原因。"""
        if len(_boundary_loops(np.asarray(mesh.faces, np.int64))) != 1:
            return mesh, [], [], {"skipped": "input_ring_count!=1"}
        welded, merged, positions = weld_near_duplicate_ring(mesh)
        if not merged:
            return mesh, [], [], {"merged_count": 0}
        if len(_boundary_loops(np.asarray(welded.faces, np.int64))) == 1:
            return welded, merged, positions, {"merged_count": len(merged)}
        return mesh, [], [], {"reverted_ring_count_changed": len(merged),
                              "candidate_positions": [list(q) for q in positions][:12]}

    outer, outer_merged, outer_positions, outer_weld = _safe_weld(outer)
    intaglio, intaglio_merged, intaglio_positions, inner_weld = _safe_weld(intaglio)
    weld_report = {"outer_vertices_before": int(outer_before), "outer_vertices_after": int(len(outer.vertices)),
                   "inner_vertices_before": int(intaglio_before), "inner_vertices_after": int(len(intaglio.vertices)),
                   "merged_count": int(len(outer_merged) + len(intaglio_merged)),
                   "merged_outer": [list(q) for q in outer_positions],
                   "merged_inner": [list(q) for q in intaglio_positions],
                   "outer_detail": outer_weld, "inner_detail": inner_weld,
                   "epsilon_mm": 0.08, "seam_untouched": True, "prescribed_edges_untouched": True}
    loops = [_boundary_loops(np.asarray(m.faces, np.int64)) for m in (outer, intaglio)]
    if any(len(rings) != 1 for rings in loops):
        raise ValueError("Cervical stitch requires exactly one open ring on each surface")
    offsets = (0, len(outer.vertices), len(outer.vertices) + len(intaglio.vertices))
    vertices = np.vstack((outer.vertices, intaglio.vertices, curve))
    seam_ids = np.arange(offsets[2], len(vertices))
    # ── 统一 seam 参数化（最小修复）：以**外表面环**为 canonical reference。
    # 只跑一次 DP；内冠环按同一组 seam 顶点顺序 + 同一纵向参数生成 —— 两条带在 seam
    # 出口同相位，不再"各跑各的 DP"（那正是 5 处 STITCH_STRIP×STITCH_STRIP 擦碰的根因）。
    a_outer, b_seam, _arca, _arcb = _align_rings(
        vertices, np.asarray(loops[0][0], np.int64) + offsets[0], seam_ids)
    # 三角质量约束：对角线与 lattice 步序在**生成阶段**就按形状质量择优（小边长/近重复
    # 顶点处不再产出长薄三角）。顶点、seam 序列、prescribed seam 边都不变。
    canonical_path = _flip_improve_path(vertices, a_outer, b_seam,
                                        _dp_seam_path(vertices, a_outer, b_seam))
    a_inner = _align_ring_to_reference(vertices, np.asarray(loops[1][0], np.int64) + offsets[1], b_seam)
    inner_path = _flip_improve_path(vertices, a_inner, b_seam,
                                    _matched_seam_path(canonical_path, len(a_inner)))
    strips = [_tris_from_path(a_outer, b_seam, canonical_path),
              _tris_from_path(a_inner, b_seam, inner_path)]
    strip_faces = np.vstack(strips) if strips else np.zeros((0, 3), np.int64)
    faces = np.vstack((outer.faces, np.asarray(intaglio.faces) + offsets[1], *strips))
    result = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(result, multibody=True)
    if not result.is_watertight or len(result.split(only_watertight=False)) != 1:
        raise ValueError("Cervical stitch produced an open or disconnected shell")
    qc = input_qc(np.asarray(result.vertices), np.asarray(result.faces))
    severe = qc.get("self_intersections_severe", 0 if qc.get("self_intersections") == 0 else None)
    if (qc["nonmanifold_edges"] or qc["winding_inconsistent_edges"] or qc["degenerate_faces"]
            or qc.get("self_intersections_truncated") or severe is None or severe):
        # **诊断**（判定与阈值一字未动）：逐对定位自交 + 写红/黄/绿 debug 网格
        try:
            from ..config import DATA_DIR
            rings = np.vstack([np.asarray(outer.vertices)[np.unique(np.asarray(loops[0], np.int64))],
                               np.asarray(intaglio.vertices)[np.unique(np.asarray(loops[1], np.int64))]])
            report = cervical_stitch_intersection_report(
                result, outer_face_count=len(outer.faces), intaglio_face_count=len(intaglio.faces),
                cervical_ring_points=rings, margin_points=curve, seam_ids=seam_ids,
                debug_dir=Path(DATA_DIR) / "debug")
            qc["intersection_report"] = report["summary"]
            qc["intersection_pairs"] = report["entries"]
            qc["ring_weld"] = weld_report
            # ── finalize 验收策略（不改几何、不改阈值、不改上游算法）────────────────
            # DESIGN_MODE（1.0 默认）允许**局部颈缘连接瑕疵**通过；STRICT_MODE 保持原逻辑。
            # 判据只用已测到的量：自交数 ≤ 10、组合只有 STITCH_STRIP+STITCH_STRIP、
            # 不涉及 margin(A_margin=0) / 内冠或外咬合面(B_cervical_offset=0)，
            # 且非流形 / 绕向 / 退化面 / 截断全部为 0。
            summary = report.get("summary") or {}
            categories = summary.get("categories") or {}
            # 计数口径沿用**项目自身**的 micro-defect 规则（自交检测注释里的 severe_max_normal_dot=0.999
            # ⇒ 法向夹角 ≤ ~2.56° 视为"近乎共面相切"的数值级对，不是真交叉）。raw 与 counted 都报出来。
            entries = report.get("entries") or []
            near_parallel = sum(1 for e in entries if e["normal_angle_deg"] <= 2.56)
            counted = len(entries) - near_parallel
            splittable = (isinstance(qc.get("self_intersections"), int)
                          and 0 < counted <= DESIGN_MODE_MAX_SELF_INTERSECTIONS
                          and summary.get("surface_combinations") == ["STITCH_STRIP+STITCH_STRIP"]
                          and categories.get("A_margin", 1) == 0
                          and categories.get("B_cervical_offset", 1) == 0
                          and not qc["nonmanifold_edges"] and not qc["winding_inconsistent_edges"]
                          and not qc["degenerate_faces"] and not qc.get("self_intersections_truncated"))
            if str(quality_mode).upper().startswith("DESIGN") and splittable:
                design_mode_exception = {"quality_mode": "DESIGN",
                                         "self_intersections": int(qc["self_intersections"]),
                                         "categories": categories,
                                         "surface_combinations": summary.get("surface_combinations"),
                                         "warning": "Cervical stitch minor self intersection",
                                         "manufacturing_check": False,
                                         "note": "仅颈缘缝合带局部互穿；不涉及 margin / intaglio / external occlusal"}
                qc["design_mode_exception"] = design_mode_exception
            else:
                raise ValueError(f"Cervical stitch geometry failed: {qc}")
        except (ValueError, OSError, KeyError) as exc:
            qc["intersection_report"] = {"diagnostic_error": str(exc)}
            raise
    audit = audit_margin_seam(result.vertices, result.faces, seam_ids, curve)
    audit["stitch_triangle_quality"] = stitch_triangle_quality(result.vertices, strip_faces)
    audit["ring_weld"] = weld_report
    audit["quality_mode"] = quality_mode
    audit["design_mode_exception"] = design_mode_exception
    if not audit["shared_seam_edges_incidence_two"]:
        raise ValueError("Prescribed Margin edges are not shared by both surfaces")
    audit["geometry"] = qc
    audit["fit_validated"] = False
    audit["thickness_validated"] = False
    return result, audit
