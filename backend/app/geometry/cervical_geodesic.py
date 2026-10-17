"""Surface-geodesic cervical transition for CrownFlow 1.0.

`margin_band_mm` is a surface-band width, not an insertion-axis height.  This
module starts from an already open clinical surface whose only boundary is the
Margin-side opening, computes graph geodesic distance from that boundary, and
triangle-clips at a constant surface distance.  Existing body vertices are never
moved.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .crown_outer.collar_body_outer import _boundary_loops, _clip_by_scalar


def _multi_source_geodesic(mesh, seeds, *, max_seeds=128):
    seeds = np.asarray(seeds, np.int64)
    if len(seeds) > int(max_seeds):
        pick = np.linspace(0, len(seeds) - 1, int(max_seeds), dtype=int)
        seeds = seeds[pick]
    edges = np.asarray(mesh.edges_unique, np.int64)
    weights = np.asarray(mesh.edges_unique_length, float)
    n = len(mesh.vertices)
    rows = np.r_[edges[:, 0], edges[:, 1]]
    cols = np.r_[edges[:, 1], edges[:, 0]]
    data = np.r_[weights, weights]
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    values = dijkstra(graph, directed=False, indices=seeds)
    values = np.asarray(values, float)
    if values.ndim == 1:
        return values
    return np.min(values, axis=0)


def inset_open_surface_geodesic(surface, width_mm, *, role="SURFACE"):
    """Keep body points at geodesic distance >= width from the cervical boundary."""
    width = float(width_mm)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("geodesic cervical width must be positive")
    loops = _boundary_loops(np.asarray(surface.faces, np.int64))
    if len(loops) != 1:
        raise ValueError(f"{role} geodesic inset requires exactly one source loop")
    seeds = np.asarray(loops[0], np.int64)
    distance = _multi_source_geodesic(surface, seeds)
    if not np.isfinite(distance).any():
        raise ValueError(f"{role} geodesic field has no finite values")
    # keep distance >= width using the existing triangle-level scalar clipper
    vertices, faces, _ = _clip_by_scalar(
        surface.vertices, surface.faces, -distance, level=-width
    )
    clipped = trimesh.Trimesh(vertices, faces, process=False)
    clipped.remove_unreferenced_vertices()
    parts = [p for p in clipped.split(only_watertight=False) if len(p.faces) >= 3]
    candidates = []
    for part in parts:
        part.remove_unreferenced_vertices()
        part_loops = _boundary_loops(np.asarray(part.faces, np.int64))
        if len(part_loops) != 1:
            continue
        candidates.append(part)
    if not candidates:
        raise ValueError(f"{role} geodesic inset did not leave one open body surface")
    candidates.sort(key=lambda p: float(p.area), reverse=True)
    chosen = candidates[0]
    loops_after = _boundary_loops(np.asarray(chosen.faces, np.int64))
    ring = np.asarray(chosen.vertices, float)[np.asarray(loops_after[0], np.int64)]
    return chosen, {
        "status": "PASS_ENGINEERING",
        "method": "SURFACE_GRAPH_GEODESIC_TRIANGLE_CLIP_V1",
        "role": role,
        "width_mm": width,
        "source_faces": int(len(surface.faces)),
        "selected_faces": int(len(chosen.faces)),
        "component_count": int(len(parts)),
        "eligible_single_loop_components": int(len(candidates)),
        "source_boundary_vertices": int(len(seeds)),
        "transition_boundary_vertices": int(len(ring)),
        "finite_geodesic_fraction": float(np.mean(np.isfinite(distance))),
        "body_vertices_moved": 0,
    }
