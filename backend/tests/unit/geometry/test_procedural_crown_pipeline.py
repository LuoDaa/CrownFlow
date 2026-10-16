import numpy as np

from app.geometry.crown_outer.anatomy_refinement import refine_crown_anatomy
from app.geometry.crown_outer.patient_envelope import PatientEnvelope
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.procedural_embryo import (
    ProceduralEmbryoConfig,
    build_procedural_crown_embryo,
)


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
        sources={
            'antagonist': {'used': True},
            'constraint_conflicts': [],
        },
    )


def test_procedural_seed_uses_patient_envelope_without_template():
    margin = _margin()
    mesh, report = build_procedural_crown_embryo(
        _frame(), margin, _envelope(), fdi=46,
        config=ProceduralEmbryoConfig(sections=96, body_rings=18),
    )

    assert report['algorithm'] == 'PROCEDURAL_PATIENT_ENVELOPE_EMBRYO_V2_MULTI_RING_TABLE'
    assert report['template_used'] is False
    assert report['template_size_authoritative'] is False
    assert report['margin_root_exact'] is True
    assert report['family'] == 'MOLAR'
    assert report['cap_rings'] >= 2
    assert np.isfinite(np.asarray(mesh.vertices)).all()
    assert len(mesh.faces) > 3000

    root = np.asarray(mesh.vertices)[:96]
    assert np.max(np.linalg.norm(root - margin, axis=1)) < 1e-8

    body = np.asarray(mesh.vertices)
    assert np.quantile(body[:, 0], .98) > 3.4
    assert np.quantile(body[:, 0], .02) < -3.3
    assert np.quantile(body[:, 2], .98) > 6.4

    # Posterior seed must not be a center-apex cone: the cap center is a shallow
    # fossa below the highest table vertices before detailed anatomy refinement.
    center_z = report['central_table_z_mm']
    assert center_z < float(np.quantile(body[:, 2], .99))


def test_anatomy_refinement_preserves_cervical_root_and_is_bounded():
    frame = _frame()
    margin = _margin()
    mesh, _ = build_procedural_crown_embryo(frame, margin, _envelope(), fdi=46)
    before = np.asarray(mesh.vertices).copy()

    refined, report = refine_crown_anatomy(mesh, frame, margin, fdi=46)
    after = np.asarray(refined.vertices)

    assert report['algorithm'] == 'REGION_WEIGHTED_ANATOMY_FAIRING_V1'
    assert report['family'] == 'MOLAR'
    assert report['hard_constraints_must_be_rechecked_after'] is True
    assert report['locked_vertex_count'] >= 96
    assert report['max_displacement_mm'] < 0.80
    assert np.allclose(after[:96], before[:96], atol=1e-10)
    assert np.linalg.norm(after - before, axis=1).max() > 1e-4
    assert report['normal_angle_after']['count'] > 1000
