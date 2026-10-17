"""Fixed 128-point cervical assembly for CrownFlow 1.0.

The final cervical topology is deliberately simple and deterministic:

    frozen outer boundary[128]  -> confirmed Margin[128]
    frozen inner boundary[128]  -> confirmed Margin[128]

Both source loops are conformingly resampled on their existing boundary edges;
no accepted body vertex is moved.  The Margin is the only shared seam.  There is
no DP zipper and no generated intermediate ring.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .canonical_intaglio import (
    _best_phase,
    _quad_strip,
    _resample_closed_polyline,
    _triangle_quality,
)
from .crown_outer.collar_body_outer import _boundary_loops, _conforming_ring
from .crown_outer.thickness import input_qc
from .margin_seam import _polyline_distances


FIXED_MARGIN_SAMPLES = 128


def _one_boundary(mesh, role: str):
    loops = _boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(loops) != 1:
        raise ValueError(
            f"FIXED_128_{role}_BOUNDARY_INVALID: expected one loop, got {len(loops)}"
        )
    if len(loops[0]) < 3:
        raise ValueError(f"FIXED_128_{role}_BOUNDARY_TOO_SHORT")
    return loops[0]


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


def build_fixed_128_margin_crown(outer, inner, margin_points):
    """Join frozen outer/intaglio bodies directly to the same 128 Margin points.

    Geometry ownership is unchanged:
    * ``outer`` is accepted anatomy and is never displaced;
    * ``inner`` is PreparedSupport-derived intaglio and is never displaced;
    * ``margin_points`` owns the exact cervical seam.

    Only conforming boundary-edge splits and the two direct 1:1 collar strips are
    generated.  This intentionally removes the previous two-layer / DP-like
    cervical freedom that could fold or leave open boundaries.
    """
    margin = np.asarray(margin_points, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3:
        raise ValueError("FIXED_128_MARGIN_INVALID")
    margin128 = _resample_closed_polyline(margin, FIXED_MARGIN_SAMPLES)
    if len(margin128) != FIXED_MARGIN_SAMPLES:
        raise ValueError("FIXED_128_MARGIN_RESAMPLE_FAILED")

    outer_loop = _one_boundary(outer, "OUTER")
    inner_loop = _one_boundary(inner, "INTAGLIO")

    # Insert samples on existing source boundary edges.  This changes topology at
    # the open edge only; source surface coordinates remain byte-for-byte fixed.
    ov, of, oring = _conforming_ring(
        outer.vertices, outer.faces, outer_loop, FIXED_MARGIN_SAMPLES
    )
    iv, inf, iring = _conforming_ring(
        inner.vertices, inner.faces, inner_loop, FIXED_MARGIN_SAMPLES
    )

    outer_order, outer_phase_cost = _best_phase(ov[oring], margin128)
    inner_order, inner_phase_cost = _best_phase(iv[iring], margin128)
    oring = np.asarray(oring, np.int64)[outer_order]
    iring = np.asarray(iring, np.int64)[inner_order]

    inner_offset = len(ov)
    margin_offset = inner_offset + len(iv)
    vertices = np.vstack([ov, iv, margin128])

    outer_ids = np.asarray(oring, np.int64)
    inner_ids = np.asarray(iring, np.int64) + inner_offset
    margin_ids = np.arange(
        margin_offset, margin_offset + FIXED_MARGIN_SAMPLES, dtype=np.int64
    )

    outer_faces = np.asarray(of, np.int64)
    inner_faces = np.asarray(inf, np.int64) + inner_offset

    # Strict point-to-point topology: O[i] -> M[i], I[i] -> M[i].
    outer_collar = _quad_strip(vertices, outer_ids, margin_ids, reverse=False)
    inner_collar = _quad_strip(vertices, inner_ids, margin_ids, reverse=True)
    collar_faces = np.vstack([outer_collar, inner_collar])
    faces = np.vstack([outer_faces, inner_faces, collar_faces])

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
    incidence = _edge_incidence(final.faces)
    topology_broken = bool(
        incidence["boundary_edges"]
        or incidence["nonmanifold_edges"]
        or qc.get("nonmanifold_edges")
        or qc.get("winding_inconsistent_edges")
        or qc.get("degenerate_faces")
        or qc.get("self_intersections")
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

    outer_ring_points = ov[oring]
    inner_ring_points = iv[iring]
    report = {
        "method": "FIXED_128_POINT_TO_POINT_SHARED_MARGIN_V1",
        "margin_topology_samples": FIXED_MARGIN_SAMPLES,
        "mapping": "OUTER_I_TO_MARGIN_I_AND_INNER_I_TO_MARGIN_I",
        "outer_phase_cost": float(outer_phase_cost),
        "inner_phase_cost": float(inner_phase_cost),
        "outer_source_vertices_moved": 0,
        "inner_source_vertices_moved": 0,
        "shared_margin_vertices": True,
        "dp_zipper_used": False,
        "intermediate_collar_ring_used": False,
        "outer_ring_to_margin_mm": {
            "median": float(np.median(np.linalg.norm(outer_ring_points - margin128, axis=1))),
            "p95": float(np.quantile(np.linalg.norm(outer_ring_points - margin128, axis=1), .95)),
        },
        "inner_ring_to_margin_mm": {
            "median": float(np.median(np.linalg.norm(inner_ring_points - margin128, axis=1))),
            "p95": float(np.quantile(np.linalg.norm(inner_ring_points - margin128, axis=1), .95)),
        },
        "collar_triangle_quality": _triangle_quality(vertices, collar_faces),
        "geometry": qc,
        "edge_incidence": incidence,
        "margin_sample_to_confirmed_max_mm": float(
            np.max(_polyline_distances(margin128, margin)) if len(margin128) else 0.0
        ),
    }
    return final, outer_surface, inner_surface, report
