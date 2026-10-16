import numpy as np
import trimesh

from app.services.template_constrained_generation import (
    _backoff_deformation,
    _deformation_valid,
    _regional_weights,
    guard_published_outer,
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


def test_occlusal_anatomy_core_moves_less_than_sidewall_and_cervical_is_locked():
    u = np.array([0.0, 3.8, 0.0])
    v = np.array([0.0, 0.2, 0.0])
    h = np.array([6.5, 4.0, 0.1])
    extents = {'u_pos': 4.0, 'u_neg': 4.0, 'v_pos': 4.0, 'v_neg': 4.0}

    cervical, adapt, height_weight, core = _regional_weights(u, v, h, extents, 7.0)

    assert core[0]
    assert not core[1]
    assert adapt[0] < adapt[1]
    assert cervical[2] == 0.0
    assert adapt[2] == 0.0
    assert height_weight[2] == 0.0


def test_broken_solver_outer_rolls_back_to_complete_seed(tmp_path):
    baseline = trimesh.creation.icosphere(subdivisions=1, radius=3.0)
    broken = baseline.copy()
    broken.update_faces(np.arange(len(broken.faces) - 6))
    broken.remove_unreferenced_vertices()

    baseline_path = tmp_path / 'baseline.stl'
    broken_path = tmp_path / 'broken.stl'
    baseline.export(baseline_path)
    broken.export(broken_path)

    published, report = guard_published_outer(baseline_path, broken_path)

    assert published == str(baseline_path)
    assert report['rollback'] is True
    assert report['status'] == 'ROLLED_BACK_TO_MORPHOLOGY_DISTANCE_SEED'
