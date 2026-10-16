import numpy as np

from app.geometry.crown_outer.anatomy_refinement import refine_crown_anatomy
from app.geometry.crown_outer.patient_envelope import PatientEnvelope
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.procedural_embryo import build_procedural_crown_embryo


def _frame():
    return confirmed_patient_frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])


def _margin(n=96):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[2.5 * np.cos(t), 2.0 * np.sin(t), np.zeros(n)]


def _envelope():
    return PatientEnvelope(
        center_xy=(0.0, 0.0),
        md_axis_xy=(1.0, 0.0),
        bl_axis_xy=(0.0, 1.0),
        u_min=-4.4,
        u_max=4.6,
        v_min=-3.6,
        v_max=3.8,
        margin_z=0.0,
        target_top_z=7.2,
        sources={'antagonist': {'used': True}, 'constraint_conflicts': []},
    )


def test_procedural_seed_preserves_confirmed_margin_and_generates_geometry():
    margin = _margin()
    mesh, report = build_procedural_crown_embryo(_frame(), margin, _envelope(), fdi=46)

    assert report['template_used'] is False
    assert report['margin_root_exact'] is True
    assert report['margin_vertex_count'] == len(margin)
    assert report['family'] == 'MOLAR'
    assert np.isfinite(np.asarray(mesh.vertices)).all()
    assert len(mesh.faces) > 1000
    assert np.allclose(np.asarray(mesh.vertices)[:len(margin)], margin, atol=1e-10)


def test_anatomy_refinement_keeps_cervical_root():
    frame = _frame()
    margin = _margin()
    mesh, _ = build_procedural_crown_embryo(frame, margin, _envelope(), fdi=46)
    before = np.asarray(mesh.vertices).copy()

    refined, report = refine_crown_anatomy(mesh, frame, margin, fdi=46)
    after = np.asarray(refined.vertices)

    assert report['family'] == 'MOLAR'
    assert report['locked_vertex_count'] >= len(margin)
    assert np.allclose(after[:len(margin)], before[:len(margin)], atol=1e-10)
    assert np.linalg.norm(after - before, axis=1).max() > 1e-4
