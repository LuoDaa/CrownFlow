"""Regression checks for the 1.0 cervical topology safety guard."""
import numpy as np
import trimesh


def _quad():
    return trimesh.Trimesh(
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),
        faces=np.array([[0,1,2],[0,2,3]], dtype=int),
        process=False,
    )


def test_identity_cervical_deformation_is_safe():
    from app.v1_cervical_fit_guard import _deformation_audit
    mesh = _quad()
    audit = _deformation_audit(mesh, mesh.copy())
    assert audit['safe'] is True
    assert audit['severe_flip_count'] == 0
    assert audit['collapsed_face_count'] == 0


def test_large_cervical_pull_is_not_safe_even_without_topology_change():
    from app.v1_cervical_fit_guard import _deformation_audit, CERVICAL_MAX_VERTEX_MOVE_MM
    before = _quad()
    after = before.copy()
    vertices = np.asarray(after.vertices, float).copy()
    vertices[2] += np.array([0.0, 0.0, CERVICAL_MAX_VERTEX_MOVE_MM + 0.5])
    after.vertices = vertices
    audit = _deformation_audit(before, after)
    assert audit['safe'] is False
    assert audit['max_move_mm'] > CERVICAL_MAX_VERTEX_MOVE_MM


def test_cervical_guard_is_loaded_before_autofit_router():
    import inspect
    from app import main
    source = inspect.getsource(main)
    guard = source.index('v1_cervical_fit_guard')
    autofit = source.index('from .api.auto_fit import router')
    assert guard < autofit
