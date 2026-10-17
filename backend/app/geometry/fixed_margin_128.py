"""Robust fixed-128 cervical assembly for CrownFlow 1.0.

Clinical contracts:
- accepted outer body is immutable;
- PreparedSupport owns the intaglio surface;
- ConfirmedMargin owns the final cervical seam;
- every actual source boundary edge is consumed exactly once.

The previous V2 joined a dense source loop (for example 1000+ vertices) directly
onto Margin[128].  Although topologically watertight, that compression produced
extremely thin triangles and self-intersections.  V3 keeps the complete source
boundary and reduces cardinality progressively through generated collar rings.
No source-body vertex is moved.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh

from .canonical_intaglio import _resample_closed_polyline, _triangle_quality
from .crown_outer.collar_body_outer import _boundary_loops
from .crown_outer.thickness import input_qc
from .margin_seam import _polyline_distances


FIXED_MARGIN_SAMPLES = 128
_EPS = 1e-10
_MAX_RING_RATIO = 1.85


def _one_boundary(mesh, role: str):
    loops = _boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(loops) != 1:
        raise ValueError(
            f"FIXED_128_{role}_BOUNDARY_INVALID: expected one loop, got {len(loops)}"
        )
    loop = np.asarray(loops[0], np.int64)
    if len(loop) < 3:
        raise ValueError(f"FIXED_128_{role}_BOUNDARY_TOO_SHORT")
    points = np.asarray(mesh.vertices, float)[loop]
    lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(lengths <= _EPS):
        raise ValueError(f"FIXED_128_{role}_BOUNDARY_HAS_ZERO_EDGE")
    return loop


def _edge_incidence(faces):
    f = np.asarray(faces, np.int64)
    edges = np.sort(
        np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1
    )
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.sum(counts == 1)),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "incidence_two_fraction": float(np.mean(counts == 2)) if len(counts) else 0.0,
    }


def _canonical_margin_128(margin_points):
    margin = np.asarray(margin_points, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3:
        raise ValueError("FIXED_128_MARGIN_INVALID")
    if len(margin) > 1 and np.linalg.norm(margin[0] - margin[-1]) <= _EPS:
        margin = margin[:-1]
    if len(margin) < 3:
        raise ValueError("FIXED_128_MARGIN_TOO_SHORT")

    exact = len(margin) == FIXED_MARGIN_SAMPLES
    if exact:
        margin128 = margin.copy()
        mode = "EXACT_CONFIRMED_MARGIN_128"
    else:
        margin128 = _resample_closed_polyline(margin, FIXED_MARGIN_SAMPLES)
        mode = "ARCLENGTH_RESAMPLED_TO_128"

    edge = np.linalg.norm(np.roll(margin128, -1, axis=0) - margin128, axis=1)
    if len(margin128) != FIXED_MARGIN_SAMPLES or np.any(edge <= _EPS):
        raise ValueError("FIXED_128_MARGIN_RESAMPLE_FAILED")
    return margin, margin128, mode, exact


def _normalized_arclength(points):
    p = np.asarray(points, float)
    edge = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    if np.any(edge <= _EPS):
        raise ValueError("CERVICAL_RING_HAS_ZERO_EDGE")
    total = float(edge.sum())
    if not np.isfinite(total) or total <= _EPS:
        raise ValueError("CERVICAL_RING_HAS_ZERO_PERIMETER")
    return np.r_[0.0, np.cumsum(edge)] / total


def _best_margin_order(source_points, margin128):
    """Find global orientation and cyclic phase without changing source topology."""
    source128 = _resample_closed_polyline(
        np.asarray(source_points, float), FIXED_MARGIN_SAMPLES
    )
    margin128 = np.asarray(margin128, float)
    base = np.arange(FIXED_MARGIN_SAMPLES, dtype=np.int64)
    best = None
    for reverse in (False, True):
        ids0 = base[::-1] if reverse else base
        for shift in range(FIXED_MARGIN_SAMPLES):
            ids = np.roll(ids0, -shift)
            delta = source128 - margin128[ids]
            mse = float(np.mean(np.einsum("ij,ij->i", delta, delta)))
            maxd = float(np.linalg.norm(delta, axis=1).max())
            key = (mse, maxd)
            if best is None or key < best[0]:
                best = (key, ids.copy(), reverse, shift)
    return best[1], {
        "mse_mm2": float(best[0][0]),
        "max_mm": float(best[0][1]),
        "reversed": bool(best[2]),
        "shift": int(best[3]),
        "phase_search": "EXHAUSTIVE_128_X_2",
    }


def _zipper(vertices, ring_a, ring_b, *, reverse_faces=False):
    """Consume every edge of two ordered loops using normalized arclength."""
    a = np.asarray(ring_a, np.int64)
    b = np.asarray(ring_b, np.int64)
    if len(a) < 3 or len(b) < 3:
        raise ValueError("FIXED_128_ZIPPER_RING_TOO_SHORT")
    v = np.asarray(vertices, float)
    sa = _normalized_arclength(v[a])
    sb = _normalized_arclength(v[b])
    faces = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na or j < nb:
        next_a = sa[i + 1] if i < na else np.inf
        next_b = sb[j + 1] if j < nb else np.inf
        a0 = int(a[i % na])
        b0 = int(b[j % nb])
        if next_a <= next_b:
            a1 = int(a[(i + 1) % na])
            faces.append((a0, a1, b0))
            i += 1
        else:
            b1 = int(b[(j + 1) % nb])
            faces.append((a0, b1, b0))
            j += 1
    result = np.asarray(faces, np.int64)
    if len(result) != na + nb:
        raise ValueError("FIXED_128_ZIPPER_DID_NOT_CONSUME_ALL_BOUNDARY_EDGES")
    if reverse_faces:
        result = result[:, ::-1]
    return result


def _count_schedule(source_count: int):
    """Reduce a dense source boundary gradually; adjacent ratios stay bounded."""
    source_count = int(source_count)
    if source_count < 3:
        raise ValueError("CERVICAL_SOURCE_COUNT_INVALID")
    counts = [source_count]
    current = source_count
    target = FIXED_MARGIN_SAMPLES
    if current == target:
        return counts
    if current < target:
        counts.append(target)
        return counts
    while current / target > _MAX_RING_RATIO:
        nxt = max(target, int(math.ceil(current / _MAX_RING_RATIO)))
        if nxt >= current:
            break
        counts.append(nxt)
        current = nxt
    if counts[-1] != target:
        counts.append(target)
    return counts


def _append_ring(vertex_chunks, points):
    offset = int(sum(len(chunk) for chunk in vertex_chunks))
    p = np.asarray(points, float)
    vertex_chunks.append(p)
    return np.arange(offset, offset + len(p), dtype=np.int64)


def _progressive_collar(vertex_chunks, source_ids, margin128, margin_ids_base,
                        margin_order, *, reverse_faces=False):
    """Build a local multi-ring collar without moving the actual source boundary."""
    vertices_now = np.vstack(vertex_chunks)
    source_ids = np.asarray(source_ids, np.int64)
    source_points = vertices_now[source_ids]
    ordered_margin = np.asarray(margin128, float)[np.asarray(margin_order, np.int64)]
    ordered_margin_ids = np.asarray(margin_ids_base, np.int64)[np.asarray(margin_order, np.int64)]

    counts = _count_schedule(len(source_ids))
    faces = []
    ring_reports = []
    previous_ids = source_ids
    n_steps = max(1, len(counts) - 1)

    # Generated rings progressively move from the real source boundary toward Margin.
    for step, count in enumerate(counts[1:-1], start=1):
        alpha = float(step / n_steps)
        src = _resample_closed_polyline(source_points, int(count))
        dst = _resample_closed_polyline(ordered_margin, int(count))
        pts = (1.0 - alpha) * src + alpha * dst
        ids = _append_ring(vertex_chunks, pts)
        current_vertices = np.vstack(vertex_chunks)
        strip = _zipper(current_vertices, previous_ids, ids, reverse_faces=reverse_faces)
        faces.append(strip)
        ring_reports.append({
            "count": int(count),
            "alpha": alpha,
            "triangles": int(len(strip)),
        })
        previous_ids = ids

    current_vertices = np.vstack(vertex_chunks)
    final_strip = _zipper(
        current_vertices, previous_ids, ordered_margin_ids, reverse_faces=reverse_faces
    )
    faces.append(final_strip)
    ring_reports.append({
        "count": FIXED_MARGIN_SAMPLES,
        "alpha": 1.0,
        "triangles": int(len(final_strip)),
        "exact_confirmed_margin": True,
    })
    return np.vstack(faces), {
        "source_count": int(len(source_ids)),
        "count_schedule": [int(x) for x in counts],
        "generated_ring_count": int(max(0, len(counts) - 2)),
        "rings": ring_reports,
        "max_adjacent_count_ratio": float(max(
            max(a, b) / max(min(a, b), 1) for a, b in zip(counts[:-1], counts[1:])
        )) if len(counts) > 1 else 1.0,
    }


def build_fixed_128_margin_crown(outer, inner, margin_points):
    """Join complete source boundaries to exact shared Margin[128] progressively."""
    margin_original, margin128, margin_mode, exact_margin = _canonical_margin_128(
        margin_points
    )
    outer_loop = _one_boundary(outer, "OUTER")
    inner_loop = _one_boundary(inner, "INTAGLIO")

    ov = np.asarray(outer.vertices, float).copy()
    iv = np.asarray(inner.vertices, float).copy()
    outer_faces = np.asarray(outer.faces, np.int64).copy()
    inner_faces_local = np.asarray(inner.faces, np.int64).copy()

    outer_margin_order, outer_phase = _best_margin_order(ov[outer_loop], margin128)
    inner_margin_order, inner_phase = _best_margin_order(iv[inner_loop], margin128)

    inner_offset = len(ov)
    vertex_chunks = [ov, iv, margin128]
    margin_offset = len(ov) + len(iv)
    margin_ids_base = np.arange(
        margin_offset, margin_offset + FIXED_MARGIN_SAMPLES, dtype=np.int64
    )
    outer_ids = np.asarray(outer_loop, np.int64)
    inner_ids = np.asarray(inner_loop, np.int64) + inner_offset
    inner_faces = inner_faces_local + inner_offset

    outer_collar, outer_progress = _progressive_collar(
        vertex_chunks, outer_ids, margin128, margin_ids_base, outer_margin_order,
        reverse_faces=False,
    )
    inner_collar, inner_progress = _progressive_collar(
        vertex_chunks, inner_ids, margin128, margin_ids_base, inner_margin_order,
        reverse_faces=True,
    )

    vertices = np.vstack(vertex_chunks)
    collar_faces = np.vstack([outer_collar, inner_collar])
    faces = np.vstack([outer_faces, inner_faces, collar_faces])

    incidence = _edge_incidence(faces)
    if incidence["boundary_edges"] or incidence["nonmanifold_edges"]:
        raise ValueError(
            "FIXED_128_FULL_BOUNDARY_INCIDENCE_FAILED: " + str(incidence)
        )

    final = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(final, multibody=True)

    outer_surface = trimesh.Trimesh(
        vertices, np.vstack([outer_faces, outer_collar]), process=False
    )
    inner_surface = trimesh.Trimesh(
        vertices, np.vstack([inner_faces, inner_collar]), process=False
    )
    outer_surface.remove_unreferenced_vertices()
    inner_surface.remove_unreferenced_vertices()

    qc = input_qc(np.asarray(final.vertices), np.asarray(final.faces))
    topology_broken = bool(
        qc.get("nonmanifold_edges")
        or qc.get("winding_inconsistent_edges")
        or qc.get("degenerate_faces")
    )
    if (
        not final.is_watertight
        or len(final.split(only_watertight=False)) != 1
        or topology_broken
    ):
        raise ValueError(
            "FIXED_128_MARGIN_COLLAR_TOPOLOGY_FAILED: "
            + str({"geometry": qc, "edge_incidence": incidence})
        )

    self_intersections_review = bool(
        qc.get("self_intersections")
        or qc.get("self_intersections_severe")
        or qc.get("self_intersections_truncated")
    )

    report = {
        "method": "PROGRESSIVE_FULL_BOUNDARY_TO_FIXED_128_SHARED_MARGIN_V3",
        "margin_topology_samples": FIXED_MARGIN_SAMPLES,
        "margin_sampling_mode": margin_mode,
        "confirmed_margin_points_preserved_exactly": bool(exact_margin),
        "mapping": "FULL_SOURCE_BOUNDARY_PROGRESSIVE_ARCLENGTH_TO_MARGIN_128",
        "zipper_mode": "PROGRESSIVE_NORMALIZED_ARCLENGTH",
        "outer_phase": outer_phase,
        "inner_phase": inner_phase,
        "outer_progressive_collar": outer_progress,
        "inner_progressive_collar": inner_progress,
        "outer_source_boundary_vertices": int(len(outer_loop)),
        "inner_source_boundary_vertices": int(len(inner_loop)),
        "outer_source_vertices_moved": 0,
        "inner_source_vertices_moved": 0,
        "shared_margin_vertices": True,
        "sparse_source_boundary_substitution_used": False,
        "dp_zipper_used": False,
        "intermediate_collar_ring_used": bool(
            outer_progress["generated_ring_count"] or inner_progress["generated_ring_count"]
        ),
        "outer_collar_triangles": int(len(outer_collar)),
        "inner_collar_triangles": int(len(inner_collar)),
        "collar_triangle_quality": _triangle_quality(vertices, collar_faces),
        "geometry": qc,
        "edge_incidence": incidence,
        "self_intersections_review_required": self_intersections_review,
        "margin_sample_to_confirmed_max_mm": float(
            np.max(_polyline_distances(margin128, margin_original))
            if len(margin128) else 0.0
        ),
    }
    return final, outer_surface, inner_surface, report
