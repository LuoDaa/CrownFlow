"""Fixed 128-point cervical assembly for CrownFlow 1.0.

The confirmed Margin is a fixed 128-point shared seam.  The source outer and
intaglio boundaries may have any cardinality; every *actual* source boundary
edge is consumed by a deterministic normalized-arclength zipper.

This is intentionally different from sparse boundary sampling.  We never pick
128 points out of a denser source loop and pretend they are the whole boundary.
Accepted outer/intaglio body vertices are not moved.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .canonical_intaglio import _resample_closed_polyline, _triangle_quality
from .crown_outer.collar_body_outer import _boundary_loops
from .crown_outer.thickness import input_qc
from .margin_seam import _polyline_distances


FIXED_MARGIN_SAMPLES = 128
_EPS = 1e-10


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

    if len(margin128) != FIXED_MARGIN_SAMPLES:
        raise ValueError("FIXED_128_MARGIN_RESAMPLE_FAILED")
    edge = np.linalg.norm(np.roll(margin128, -1, axis=0) - margin128, axis=1)
    if np.any(edge <= _EPS):
        raise ValueError("FIXED_128_MARGIN_HAS_ZERO_EDGE")
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
    """Align the full source loop to Margin with exhaustive cyclic phase search.

    A 128-point geometric resample of the source is used only to estimate phase;
    it is never substituted for the source mesh boundary topology.
    """
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


def _normalized_arclength_zipper(
    vertices, source_ids, margin_ids, *, reverse_faces=False
):
    """Triangulate the annulus while consuming every real boundary edge once.

    The path is deterministic.  It compares the *normalized* closed arclength of
    the next source edge and next Margin edge, so unequal loop perimeters do not
    twist the correspondence.
    """
    source_ids = np.asarray(source_ids, np.int64)
    margin_ids = np.asarray(margin_ids, np.int64)
    if len(source_ids) < 3 or len(margin_ids) < 3:
        raise ValueError("FIXED_128_ZIPPER_RING_TOO_SHORT")

    v = np.asarray(vertices, float)
    sa = _normalized_arclength(v[source_ids])
    sb = _normalized_arclength(v[margin_ids])

    faces = []
    i = j = 0
    n_a, n_b = len(source_ids), len(margin_ids)
    while i < n_a or j < n_b:
        next_a = sa[i + 1] if i < n_a else np.inf
        next_b = sb[j + 1] if j < n_b else np.inf
        a0 = int(source_ids[i % n_a])
        b0 = int(margin_ids[j % n_b])
        if next_a <= next_b:
            a1 = int(source_ids[(i + 1) % n_a])
            faces.append((a0, a1, b0))
            i += 1
        else:
            b1 = int(margin_ids[(j + 1) % n_b])
            faces.append((a0, b1, b0))
            j += 1

    result = np.asarray(faces, np.int64)
    if len(result) != n_a + n_b:
        raise ValueError("FIXED_128_ZIPPER_DID_NOT_CONSUME_ALL_BOUNDARY_EDGES")
    if reverse_faces:
        result = result[:, ::-1]
    return result


def build_fixed_128_margin_crown(outer, inner, margin_points):
    """Join full source boundaries to one fixed 128-point confirmed Margin.

    No sparse source-boundary substitution is permitted.  The outer and intaglio
    meshes retain their complete actual open loops; those loops are connected to
    the shared Margin with a normalized-arclength zipper.
    """
    margin_original, margin128, margin_mode, exact_margin = _canonical_margin_128(
        margin_points
    )
    outer_loop = _one_boundary(outer, "OUTER")
    inner_loop = _one_boundary(inner, "INTAGLIO")

    ov = np.asarray(outer.vertices, float).copy()
    iv = np.asarray(inner.vertices, float).copy()
    outer_faces = np.asarray(outer.faces, np.int64).copy()
    inner_faces_local = np.asarray(inner.faces, np.int64).copy()

    outer_points = ov[outer_loop]
    inner_points = iv[inner_loop]
    outer_margin_order, outer_phase = _best_margin_order(outer_points, margin128)
    inner_margin_order, inner_phase = _best_margin_order(inner_points, margin128)

    inner_offset = len(ov)
    margin_offset = inner_offset + len(iv)
    vertices = np.vstack([ov, iv, margin128])

    outer_ids = np.asarray(outer_loop, np.int64)
    inner_ids = np.asarray(inner_loop, np.int64) + inner_offset
    margin_ids_base = np.arange(
        margin_offset, margin_offset + FIXED_MARGIN_SAMPLES, dtype=np.int64
    )
    outer_margin_ids = margin_ids_base[outer_margin_order]
    inner_margin_ids = margin_ids_base[inner_margin_order]

    inner_faces = inner_faces_local + inner_offset

    outer_collar = _normalized_arclength_zipper(
        vertices, outer_ids, outer_margin_ids, reverse_faces=False
    )
    inner_collar = _normalized_arclength_zipper(
        vertices, inner_ids, inner_margin_ids, reverse_faces=True
    )
    collar_faces = np.vstack([outer_collar, inner_collar])
    faces = np.vstack([outer_faces, inner_faces, collar_faces])

    # Cheap exact incidence check first.  If this fails, the collar does not own
    # every real source boundary edge and expensive downstream QC cannot fix it.
    incidence = _edge_incidence(faces)
    if incidence["boundary_edges"] or incidence["nonmanifold_edges"]:
        raise ValueError(
            "FIXED_128_FULL_BOUNDARY_INCIDENCE_FAILED: " + str(incidence)
        )

    final = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(final, multibody=True)

    outer_surface = trimesh.Trimesh(
        vertices,
        np.vstack([outer_faces, outer_collar]),
        process=False,
    )
    inner_surface = trimesh.Trimesh(
        vertices,
        np.vstack([inner_faces, inner_collar]),
        process=False,
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
        "method": "FULL_SOURCE_BOUNDARY_TO_FIXED_128_SHARED_MARGIN_V2",
        "margin_topology_samples": FIXED_MARGIN_SAMPLES,
        "margin_sampling_mode": margin_mode,
        "confirmed_margin_points_preserved_exactly": bool(exact_margin),
        "mapping": "FULL_SOURCE_BOUNDARY_ARCLENGTH_TO_MARGIN_128",
        "zipper_mode": "DETERMINISTIC_NORMALIZED_ARCLENGTH",
        "outer_phase": outer_phase,
        "inner_phase": inner_phase,
        "outer_source_boundary_vertices": int(len(outer_loop)),
        "inner_source_boundary_vertices": int(len(inner_loop)),
        "outer_source_vertices_moved": 0,
        "inner_source_vertices_moved": 0,
        "shared_margin_vertices": True,
        "sparse_source_boundary_substitution_used": False,
        "dp_zipper_used": False,
        "intermediate_collar_ring_used": False,
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
