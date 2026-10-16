import numpy as np
import trimesh

from app.services.template_constrained_generation import (
    _backoff_deformation,
    _deformation_valid,
)


def test_safe_template_deformation_preserves_closed_topology():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=4.0)
    before_faces = np.asarray(mesh.faces).copy()
    proposed = np.asarray(mesh.vertices, float).copy()
    proposed[:, 0] *= 0.88
    proposed[:, 1] *= 0.92

    out, alpha, audit = _backoff_deformation(mesh, proposed)

    assert alpha == 1.0
    assert audit['collapsed_face_count'] == 0
    assert audit['flipped_face_count'] == 0
    assert np.array_equal(np.asarray(out.faces), before_faces)
    assert out.is_watertight
    assert out.is_winding_consistent


def test_invalid_large_deformation_is_backed_off_instead_of_publishing_hole():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
    before = np.asarray(mesh.vertices, float)
    proposed = before.copy()
    # Fold half of the crown through the opposite half.  Publishing this directly
    # would invert faces and can look like a missing patch with back-face culling.
    proposed[proposed[:, 0] > 0, 0] *= -1.2

    ok, direct_audit = _deformation_valid(before, proposed, mesh.faces)
    assert not ok
    assert direct_audit['flipped_face_count'] > 0 or direct_audit['collapsed_face_count'] > 0

    out, alpha, audit = _backoff_deformation(mesh, proposed)
    assert alpha < 1.0
    assert np.array_equal(np.asarray(out.faces), np.asarray(mesh.faces))
    assert out.is_watertight
    assert out.is_winding_consistent
    valid, final_audit = _deformation_valid(before, out.vertices, out.faces)
    assert valid, {**audit, **final_audit}
