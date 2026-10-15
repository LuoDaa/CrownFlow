import numpy as np
import pytest
import trimesh

from app.geometry.margin_seam import (
    MarginReconstructionRequired, audit_margin_seam, stitch_cervical_surfaces,
)
from app.geometry.outer import trim_outer_at_margin


def cap(radius, bottom, top, count):
    angle = np.arange(count) * 2 * np.pi / count
    ring = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), np.full(count, bottom)))
    vertices = np.vstack((ring, ring + [0, 0, top - bottom], [0, 0, top]))
    faces = []
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j, j + count), (i, j + count, i + count),
                      (i + count, j + count, 2 * count)))
    return trimesh.Trimesh(vertices, faces, process=False)


def test_nonplanar_shared_seam_and_no_silent_plane_clipping():
    angle = np.arange(40) * 2 * np.pi / 40
    margin = np.column_stack((3 * np.cos(angle), 3 * np.sin(angle), .6 * np.sin(2 * angle)))
    outer, inner = cap(3.8, 1, 4, 48), cap(2.7, 1, 3.3, 36)
    shell, audit = stitch_cervical_surfaces(outer, inner, margin)
    assert shell.is_watertight
    assert audit["shared_seam_edges_incidence_two"]
    assert audit["margin_to_seam_max_mm"] < 1e-12
    assert audit["seam_to_margin_max_mm"] < 1e-12
    assert not audit["fit_validated"] and not audit["thickness_validated"]
    ids = np.arange(len(outer.vertices) + len(inner.vertices), len(shell.vertices))
    moved = shell.vertices.copy()
    moved[ids, 2] = 0
    assert audit_margin_seam(moved, shell.faces, ids, margin)["margin_to_seam_max_mm"] > .5
    with pytest.raises(MarginReconstructionRequired) as failure:
        trim_outer_at_margin(trimesh.creation.icosphere(radius=4), margin, [0, 0, 1])
    assert failure.value.diagnostics["plane_deviation_max_mm"] > .5
    transform = trimesh.transformations.rotation_matrix(.7, [1, 2, 3])
    transform[:3, 3] = [12, -7, 4]
    transformed_margin = trimesh.transform_points(margin, transform)
    transformed_shell, moved_audit = stitch_cervical_surfaces(
        outer.copy().apply_transform(transform), inner.copy().apply_transform(transform),
        transformed_margin,
    )
    assert transformed_shell.is_watertight
    assert moved_audit["margin_to_seam_max_mm"] < 1e-12


def test_planar_clip_remains_available_and_invalid_margin_is_rejected():
    angle = np.arange(16) * 2 * np.pi / 16
    margin = np.column_stack((2 * np.cos(angle), 2 * np.sin(angle), np.zeros(16)))
    mesh = trimesh.creation.icosphere(radius=3)
    assert trim_outer_at_margin(mesh, margin, [0, 0, 1]).is_watertight
    margin[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        trim_outer_at_margin(mesh, margin, [0, 0, 1])
