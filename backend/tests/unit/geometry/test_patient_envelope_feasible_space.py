import numpy as np
import trimesh

from app.geometry.crown_outer.patient_envelope import (
    PatientEnvelope,
    PatientEnvelopeConfig,
    morph_embryo_to_patient_envelope,
)
from app.geometry.crown_outer.patient_frame import PatientFrame


def _frame():
    return PatientFrame(
        origin=np.array([0.0, 0.0, 0.0]),
        x=np.array([1.0, 0.0, 0.0]),
        y=np.array([0.0, 1.0, 0.0]),
        z=np.array([0.0, 0.0, 1.0]),
    )


def _margin():
    return np.array([
        [-2.0, -1.5, 0.0],
        [0.0, -1.8, 0.0],
        [2.0, -1.5, 0.0],
        [2.2, 0.0, 0.0],
        [2.0, 1.5, 0.0],
        [0.0, 1.8, 0.0],
        [-2.0, 1.5, 0.0],
        [-2.2, 0.0, 0.0],
    ])


def _crown():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    v = np.asarray(mesh.vertices, float).copy()
    v[:, 0] *= 3.2
    v[:, 1] *= 2.5
    v[:, 2] = 3.0 + 3.0 * v[:, 2]
    mesh.vertices = v
    return mesh


def _envelope(v_min=-4.0, v_max=4.0, top=8.0):
    return PatientEnvelope(
        center_xy=(0.0, 0.0),
        md_axis_xy=(1.0, 0.0),
        bl_axis_xy=(0.0, 1.0),
        u_min=-4.0,
        u_max=4.0,
        v_min=v_min,
        v_max=v_max,
        margin_z=0.0,
        target_top_z=top,
        sources={"antagonist": {"used": True}},
    )


def test_large_feasible_box_does_not_expand_template_to_fill_it():
    mesh = _crown()
    before = np.asarray(mesh.vertices, float).copy()
    out, report = morph_embryo_to_patient_envelope(
        mesh, _frame(), _margin(), _envelope(), config=PatientEnvelopeConfig()
    )
    delta = np.linalg.norm(np.asarray(out.vertices) - before, axis=1)
    assert float(delta.max()) < 1e-8
    assert report["semantics"] == "COMPRESS_VIOLATIONS_ONLY_NEVER_FILL_BOX"
    assert report["scales"]["bl_positive"] == 1.0
    assert report["scales"]["bl_negative"] == 1.0
    assert report["scales"]["height"] == 1.0


def test_buccal_limit_does_not_squash_lingual_side():
    mesh = _crown()
    before = np.asarray(mesh.vertices, float).copy()
    out, report = morph_embryo_to_patient_envelope(
        mesh, _frame(), _margin(), _envelope(v_min=-4.0, v_max=1.7),
        config=PatientEnvelopeConfig(),
    )
    after = np.asarray(out.vertices, float)
    pos = before[:, 1] > 0.8
    neg = before[:, 1] < -0.8
    assert report["scales"]["bl_positive"] < 1.0
    assert report["scales"]["bl_negative"] == 1.0
    assert np.median(after[pos, 1]) < np.median(before[pos, 1])
    assert np.max(np.abs(after[neg, 1] - before[neg, 1])) < 1e-8


def test_antagonist_is_ceiling_not_requested_height():
    mesh = _crown()
    before = np.asarray(mesh.vertices, float).copy()
    out, report = morph_embryo_to_patient_envelope(
        mesh, _frame(), _margin(), _envelope(top=9.0), config=PatientEnvelopeConfig()
    )
    assert report["scales"]["height"] == 1.0
    assert np.max(np.abs(np.asarray(out.vertices) - before)) < 1e-8
