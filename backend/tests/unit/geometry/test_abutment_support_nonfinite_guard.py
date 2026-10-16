import numpy as np
import pytest
import trimesh

from app.geometry.abutment_support import _embed_boundary, SupportExtractionError


def _mesh_with_one_zero_length_boundary_edge():
    # Vertex 0 and 1 are distinct ids at the same position.  The boundary edge
    # (0, 1) is therefore topologically real but geometrically degenerate.
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_embed_boundary_ignores_zero_length_edge_and_stays_finite():
    mesh = _mesh_with_one_zero_length_boundary_edge()
    source_ids = np.array([10, 11], dtype=int)
    ring = [0, 1, 2, 3]
    margin = np.array([[0.5, 0.0, 0.0]])

    result, mapping, report = _embed_boundary(mesh, source_ids, ring, margin)

    assert np.isfinite(np.asarray(result.vertices)).all()
    assert np.isfinite(report["projection_distance_max_mm"])
    assert report["projection_distance_max_mm"] <= 0.05
    assert len(mapping) == len(result.faces)


def test_embed_boundary_rejects_when_only_eligible_edge_is_degenerate():
    mesh = _mesh_with_one_zero_length_boundary_edge()
    source_ids = np.array([10, 11], dtype=int)
    # Only the zero-length edge (0, 1) is eligible.
    ring = [0, 1]
    margin = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(SupportExtractionError) as exc:
        _embed_boundary(mesh, source_ids, ring, margin)

    assert exc.value.code == "DEGENERATE_MARGIN_BOUNDARY"
