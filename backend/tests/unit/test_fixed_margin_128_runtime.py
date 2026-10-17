import numpy as np
import trimesh

from app.geometry.fixed_margin_128 import build_fixed_128_margin_crown


def _open_cup(n, radius, z0, ztop, *, inward=False, phase=0.0):
    theta = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False) + float(phase)
    bottom_radius = radius * (1.0 + 0.04 * np.cos(3.0 * theta))
    bottom_z = z0 + 0.04 * np.sin(2.0 * theta)
    bottom = np.c_[
        bottom_radius * np.cos(theta),
        bottom_radius * np.sin(theta),
        bottom_z,
    ]
    top_radius = radius * 0.78 * (1.0 + 0.02 * np.cos(2.0 * theta))
    top = np.c_[
        top_radius * np.cos(theta),
        top_radius * np.sin(theta),
        np.full(int(n), float(ztop)),
    ]
    center = np.array([[0.0, 0.0, float(ztop)]])
    vertices = np.vstack([bottom, top, center])
    faces = []
    for i in range(int(n)):
        j = (i + 1) % int(n)
        faces.append((i, j, int(n) + i))
        faces.append((j, int(n) + j, int(n) + i))
        faces.append((int(n) + i, int(n) + j, 2 * int(n)))
    faces = np.asarray(faces, np.int64)
    if inward:
        faces = faces[:, ::-1]
    return trimesh.Trimesh(vertices, faces, process=False)


def test_dense_source_boundaries_reduce_progressively_to_exact_margin_128():
    # Case-63-like density mismatch: the inner boundary can exceed one thousand
    # vertices while the clinical Margin remains exactly 128 points.
    outer = _open_cup(148, 4.2, 0.55, 4.4)
    inner = _open_cup(1043, 3.1, 0.35, 3.7, inward=True, phase=0.04)
    outer_before = np.asarray(outer.vertices).copy()
    inner_before = np.asarray(inner.vertices).copy()

    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = np.c_[
        3.5 * np.cos(theta),
        3.5 * np.sin(theta),
        0.10 * np.sin(2.0 * theta),
    ]

    final, _outer_surface, _inner_surface, report = build_fixed_128_margin_crown(
        outer, inner, margin
    )

    assert final.is_watertight
    assert len(final.split(only_watertight=False)) == 1
    assert report["edge_incidence"]["boundary_edges"] == 0
    assert report["edge_incidence"]["nonmanifold_edges"] == 0
    assert report["outer_source_boundary_vertices"] == 148
    assert report["inner_source_boundary_vertices"] == 1043
    assert report["margin_topology_samples"] == 128
    assert report["confirmed_margin_points_preserved_exactly"] is True
    assert report["mapping"] == "FULL_SOURCE_BOUNDARY_PROGRESSIVE_ARCLENGTH_TO_MARGIN_128"
    assert report["sparse_source_boundary_substitution_used"] is False
    assert report["intermediate_collar_ring_used"] is True

    # Dense compression must be distributed across bounded-ratio stages rather
    # than one 1043 -> 128 fan.
    inner_progress = report["inner_progressive_collar"]
    assert inner_progress["count_schedule"][0] == 1043
    assert inner_progress["count_schedule"][-1] == 128
    assert inner_progress["generated_ring_count"] >= 2
    assert inner_progress["max_adjacent_count_ratio"] <= 1.85 + 1e-9

    # Outer accepted anatomy and support-derived inner body remain immutable.
    assert np.array_equal(np.asarray(outer.vertices), outer_before)
    assert np.array_equal(np.asarray(inner.vertices), inner_before)
