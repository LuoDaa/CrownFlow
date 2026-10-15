import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.space_contact import fit_proximal_envelopes


def _frame():
    return confirmed_patient_frame(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )


def _margin(rx=2.35, ry=1.95, n=96):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.zeros(n)]


def _crown():
    profile = np.array([
        [0.0, 0.0],
        [1.90, 0.0],
        [2.00, 0.35],
        [2.10, 1.20],
        [2.25, 2.80],
        [2.18, 4.30],
        [1.62, 5.70],
        [0.0, 5.95],
    ])
    return trimesh.creation.revolve(profile, sections=96)


def _neighbor(center, extents=(1.0, 7.0, 8.0), angle_deg=0.0):
    mesh = trimesh.creation.box(extents=extents)
    if angle_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.deg2rad(angle_deg), [0.0, 0.0, 1.0]
        ))
    mesh.apply_translation(center)
    return FitSurface(mesh)


def test_bilateral_neighbors_define_width_and_both_contact_cores_reach_zero():
    crown = _crown()
    before = np.asarray(crown.vertices).copy()
    surfaces = {
        "mesial": _neighbor([4.55, 0.0, 3.0]),
        "distal": _neighbor([-4.55, 0.0, 3.0]),
    }

    fitted, report, domains = fit_proximal_envelopes(
        crown, _frame(), _margin(), surfaces, 0.0,
    )

    assert report["solver"] == "SPACE_DRIVEN_PROXIMAL_ENVELOPE_V3_PATIENT_VECTOR"
    assert report["status"] == "PASS_ENGINEERING"
    assert np.ptp(np.asarray(fitted.vertices)[:, 0]) > np.ptp(before[:, 0]) + 1.0
    for side in ("mesial", "distal"):
        state = report["physical_after"][side]
        assert int(domains[side].sum()) >= 48
        assert state["target_reached"] is True
        assert state["contact_vertex_count"] >= state["required_contact_vertices"]
        assert state["target_error"]["median_mm"] < 0.12


def test_negative_proximal_target_creates_small_compressive_contact_not_a_gap():
    crown = _crown()
    surfaces = {
        "mesial": _neighbor([4.35, 0.0, 3.0]),
        "distal": _neighbor([-4.35, 0.0, 3.0]),
    }

    _fitted, report, _domains = fit_proximal_envelopes(
        crown, _frame(), _margin(), surfaces, -0.02,
    )

    assert report["status"] == "PASS_ENGINEERING"
    assert report["target_mm"] == -0.02
    for side in ("mesial", "distal"):
        state = report["physical_after"][side]
        assert state["target_reached"] is True
        assert state["contact_vertex_count"] >= state["required_contact_vertices"]
        # At least part of the continuous core must be at/beyond the zero surface.
        assert state["signed_min_mm"] <= 0.01


def test_oblique_posterior_neighbors_use_actual_patient_vectors_not_global_x():
    crown = _crown()
    root2 = np.sqrt(0.5)
    direction = np.array([root2, root2, 0.0])
    center_a = direction * 6.2 + np.array([0.0, 0.0, 3.0])
    center_b = -direction * 6.2 + np.array([0.0, 0.0, 3.0])
    surfaces = {
        "mesial": _neighbor(center_a, extents=(1.0, 7.0, 8.0), angle_deg=45.0),
        "distal": _neighbor(center_b, extents=(1.0, 7.0, 8.0), angle_deg=45.0),
    }

    fitted, report, _domains = fit_proximal_envelopes(
        crown, _frame(), _margin(), surfaces, 0.0,
    )

    assert report["status"] == "PASS_ENGINEERING"
    da = np.asarray(report["domain_quality"]["mesial"]["direction_world"])
    db = np.asarray(report["domain_quality"]["distal"]["direction_world"])
    assert float(da @ direction) > 0.75
    assert float(db @ (-direction)) > 0.75
    assert report["physical_after"]["mesial"]["target_reached"] is True
    assert report["physical_after"]["distal"]["target_reached"] is True
    assert np.isfinite(np.asarray(fitted.vertices)).all()
