import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_envelope import build_patient_envelope
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.space_contact import (
    MAX_BROAD_SIDE_GROWTH_MM,
    MAX_REFERENCE_ANGLE_DEG,
    _physical_reference,
    fit_proximal_envelopes,
)
from app.geometry.crown_outer.zones import build_crown_zones


def _frame():
    return confirmed_patient_frame(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )


def _margin(rx=2.4, ry=2.0, n=128):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.zeros(n)]


def _box(center, extents=(1.0, 8.0, 7.0)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return FitSurface(mesh)


def _crown():
    profile = np.array([
        [0.0, 0.0], [1.90, 0.0], [2.00, 0.35], [2.10, 1.20],
        [2.25, 2.80], [2.18, 4.30], [1.62, 5.70], [0.0, 5.95],
    ])
    return trimesh.creation.revolve(profile, sections=96)


def test_v5_neighbor_halo_cannot_rotate_or_unboundedly_size_patient_envelope():
    frame = _frame()
    margin = _margin()
    # Deliberately huge/off-axis reference halos: these are query surfaces, not
    # permission to turn the crown into a diagonal tongue.
    surfaces = {
        "mesial": _box([7.0, 5.5, 3.0], extents=(1.0, 16.0, 7.0)),
        "distal": _box([-7.0, -5.5, 3.0], extents=(1.0, 16.0, 7.0)),
    }
    env = build_patient_envelope(
        frame=frame,
        margin=margin,
        proximal_surfaces=surfaces,
        proximal_target_mm=-0.02,
    )
    mlocal = np.asarray(margin)
    mx_lo, mx_hi = np.quantile(mlocal[:, 0], [.02, .98])
    my_lo, my_hi = np.quantile(mlocal[:, 1], [.02, .98])

    assert np.allclose(env.md_axis_xy, [1.0, 0.0], atol=1e-12)
    assert np.allclose(env.bl_axis_xy, [0.0, 1.0], atol=1e-12)
    assert env.sources["md_axis_source"] == "CONFIRMED_PATIENT_FRAME_X"
    assert env.u_max <= mx_hi + 3.01
    assert env.u_min >= mx_lo - 3.01
    assert env.v_max <= my_hi + 2.21
    assert env.v_min >= my_lo - 2.21


def test_named_proximal_reference_is_clamped_to_patient_frame_side():
    frame = _frame()
    surface = _box([5.0, 5.0, 3.0], extents=(1.0, 6.0, 6.0))
    ref = _physical_reference(surface, frame, np.array([0.0, 0.0]), side="mesial")
    direction = np.asarray(ref["direction_local_xy"])
    angle = np.degrees(np.arccos(np.clip(direction @ np.array([1.0, 0.0]), -1.0, 1.0)))

    assert angle <= MAX_REFERENCE_ANGLE_DEG + 1e-6
    assert direction[0] > 0.0
    assert abs(np.asarray(ref["direction_world"])[2]) < 1e-12


def test_proximal_solver_preserves_margin_band_occlusal_zone_and_all_axial_heights():
    frame = _frame()
    margin = _margin()
    crown = _crown()
    before = np.asarray(crown.vertices).copy()
    zones = build_crown_zones(crown, frame, margin)
    protected = zones.cervical | zones.occlusal
    surfaces = {
        "mesial": _box([4.55, 0.0, 3.0]),
        "distal": _box([-4.55, 0.0, 3.0]),
    }

    fitted, report, _domains = fit_proximal_envelopes(
        crown, frame, margin, surfaces, -0.02,
    )
    after = np.asarray(fitted.vertices)

    # GREEN is a horizontal morphology owner only: it must not rewrite the
    # already good BLUE occlusal height or the RED/Margin cervical result.
    assert np.max(np.abs(after[:, 2] - before[:, 2])) < 1e-8
    assert np.max(np.linalg.norm(after[protected] - before[protected], axis=1)) < 1e-8
    for side in ("mesial", "distal"):
        growth = report["large_scale_reach"][side]["total_growth_mm"]
        assert growth <= MAX_BROAD_SIDE_GROWTH_MM + 1e-9
    assert report["solver"] == "SPACE_DRIVEN_PROXIMAL_BOUNDED_FRAME_V5"
