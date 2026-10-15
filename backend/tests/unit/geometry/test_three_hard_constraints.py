import numpy as np
import trimesh

from app.models import DesignCreate
from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_adaptation import cervical_fit_to_margin
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.space_contact import fit_proximal_envelopes
from app.geometry.crown_outer.space_morph import (
    enforce_inner_envelope_clearance,
    space_driven_embryo_prefit,
)


def _frame():
    return confirmed_patient_frame(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )


def _margin(rx=2.5, ry=2.0, n=96):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.zeros(n)]


def _crown():
    z = np.linspace(0.0, 6.0, 24)
    control_z = np.array([0.0, 0.35, 1.20, 2.80, 4.30, 5.80, 6.00])
    control_r = np.array([1.95, 2.05, 2.15, 2.30, 2.20, 1.65, 0.0])
    radius = np.interp(z, control_z, control_r)
    profile = np.c_[radius, z]
    profile = np.vstack([[0.0, 0.0], profile])
    return trimesh.creation.revolve(profile, sections=96)


def test_margin_is_full_loop_hard_boundary():
    crown, report = cervical_fit_to_margin(_crown(), _frame(), _margin())
    assert report["status"] == "PASS_ENGINEERING", report
    assert report["full_loop_coverage"]["within_0_05mm_fraction"] >= 0.98, report
    assert report["full_loop_coverage"]["uncovered_samples"] == 0, report
    assert report["after_mm"]["p95"] <= 0.05, report
    assert report["duplicate_pairs_after"] <= report["duplicate_pairs_before"], report
    assert np.isfinite(crown.vertices).all()


def test_proximal_target_accepts_light_negative_contact():
    model = DesignCreate(
        case_id="case", target_fdi=26, support_type="abutment",
        machine_axes=5, proximal_target_mm=-0.02, occlusal_target_mm=0.05,
    )
    assert model.proximal_target_mm == -0.02


def test_patient_side_growth_reaches_continuous_negative_contact_core():
    crown, _ = cervical_fit_to_margin(_crown(), _frame(), _margin())
    neighbor = trimesh.creation.box(extents=[1.0, 7.0, 8.0])
    neighbor.apply_translation([4.5, 0.0, 3.0])
    result, report, _domains = fit_proximal_envelopes(
        crown, _frame(), _margin(), {"mesial": FitSurface(neighbor)}, -0.02,
    )
    state = report["physical_after"]["mesial"]
    diag = {
        "status": report["status"], "reason": report.get("reason"),
        "reach": report["large_scale_reach"]["mesial"],
        "domain": report["domain_quality"]["mesial"],
        "physical_after": state,
        "passes": report["passes"],
        "last_trials": report["trials"][-5:],
    }
    assert report["status"] == "PASS_ENGINEERING", diag
    assert state["target_reached"] is True, diag
    assert state["contact_vertex_count"] >= state["required_contact_vertices"], diag
    assert state["largest_connected_contact_vertices"] >= state["required_connected_contact_vertices"], diag
    # The proximal wall is intentionally not flattened onto the neighbour.  Only
    # the continuous contact core must sit inside the ±35µm target band; the wall
    # median remains a positive embrasure/convexity distance by design.
    assert state["contact_fraction"] >= (
        state["required_contact_vertices"] / state["domain_vertex_count"]
    ) - 1e-9, diag
    assert state["signed_min_mm"] <= 0.015, diag
    assert np.isfinite(result.vertices).all()


def test_antagonist_height_solver_corrects_large_global_penetration():
    crown, _ = cervical_fit_to_margin(_crown(), _frame(), _margin())
    antagonist = trimesh.creation.box(extents=[12.0, 12.0, 1.0])
    antagonist.apply_translation([0.0, 0.0, 6.25])
    result, report = space_driven_embryo_prefit(
        crown, _frame(), _margin(),
        antagonist_surface=FitSurface(antagonist, [0.0, 0.0, -1.0]),
        occlusal_target_mm=0.05,
    )
    height = report["height"]
    assert height["applied"] is True, height
    assert height["mode"] == "ONE_SIDED_AXIAL_CLOSED_LOOP_ANTAGONIST_HEIGHT_V3", height
    assert height["envelope"]["method"] == "AXIAL_FIRST_HIT_FROM_CERVICAL_SIDE", height
    assert height["final"]["penetration_p95_mm"] <= 0.02 + 1e-9, height
    assert np.max(result.vertices[:, 2]) < np.max(crown.vertices[:, 2]), height


def test_inner_shell_guard_is_lateral_and_cannot_recreate_occlusal_penetration():
    crown, _ = cervical_fit_to_margin(_crown(), _frame(), _margin())
    inner = trimesh.creation.cylinder(radius=2.35, height=4.4, sections=96)
    inner.apply_translation([0.0, 0.0, 2.2])
    before_z = np.asarray(crown.vertices)[:, 2].copy()
    result, report = enforce_inner_envelope_clearance(crown, inner, _frame(), _margin(), 0.80)
    after_z = np.asarray(result.vertices)[:, 2]
    assert report["method"] == "ACTUAL_INTAGLIO_PURE_LATERAL_CLEARANCE_V3", report
    assert np.allclose(after_z, before_z, atol=1e-10), report
    assert np.isfinite(result.vertices).all()
