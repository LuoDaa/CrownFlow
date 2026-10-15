import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.space_contact import build_proximal_domains
from app.geometry.crown_outer.space_morph import (
    audit_inner_envelope_clearance,
    space_driven_embryo_prefit,
)


def _frame():
    return confirmed_patient_frame(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )


def _margin(rx=2.5, ry=2.0, z=0.0, n=96):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.full(n, z)]


def _crown():
    profile = np.array([
        [0.0, 0.0],
        [1.95, 0.0],
        [2.05, 0.35],
        [2.15, 1.20],
        [2.30, 2.80],
        [2.20, 4.30],
        [1.65, 5.80],
        [0.0, 6.00],
    ])
    return trimesh.creation.revolve(profile, sections=96)


def test_one_sided_neighbor_expands_only_that_side_without_whole_crown_translation():
    mesh = _crown()
    frame = _frame()
    margin = _margin()
    mesial = trimesh.creation.box(extents=[1.0, 6.0, 8.0])
    mesial.apply_translation([4.5, 0.0, 3.0])

    before = np.asarray(mesh.vertices).copy()
    result, report = space_driven_embryo_prefit(
        mesh, frame, margin,
        proximal_surfaces={"mesial": FitSurface(mesial)},
        proximal_target_mm=0.05,
    )
    after = np.asarray(result.vertices)

    assert report["algorithm"] == "SPACE_DRIVEN_EMBRYO_MORPH_V2"
    assert report["template_role"] == "TOPOLOGY_AND_ANATOMY_EMBRYO_NOT_SIZE_TRUTH"
    assert report["md"]["whole_crown_translation_mm"] == 0.0
    assert report["md"]["mesial_scale"] > 1.0
    assert abs(report["md"]["distal_scale"] - 1.0) < 0.05
    assert np.quantile(after[:, 0], .98) > np.quantile(before[:, 0], .98) + 0.5
    # Margin/cervical pivot is faded; the opposite side must not simply translate.
    distal = before[:, 0] < np.quantile(before[:, 0], .15)
    assert np.median(np.abs(after[distal, 0] - before[distal, 0])) < 0.15


def test_support_fallback_and_material_envelope_can_make_embryo_larger_before_finalization():
    mesh = _crown()
    frame = _frame()
    margin = _margin()
    support = trimesh.creation.cylinder(radius=2.20, height=4.0, sections=96)
    support.apply_translation([0.0, 0.0, 2.0])

    before = np.asarray(mesh.vertices).copy()
    result, report = space_driven_embryo_prefit(
        mesh, frame, margin,
        support_mesh=support,
        minimum_thickness_mm=0.80,
        cement_gap_mm=0.05,
    )
    after = np.asarray(result.vertices)

    shell = report["inner_shell"]
    assert shell["status"] in ("APPLIED", "NO_CORRECTION_NEEDED")
    assert shell["source"] == "PREP_SUPPORT_PLUS_CEMENT_GAP_FALLBACK"
    assert abs(shell["required_outer_to_intaglio_mm"] - 0.85) < 1e-9
    assert report["md"]["mesial_scale"] > 1.0
    assert report["md"]["distal_scale"] > 1.0
    assert report["bl"]["buccal_scale"] > 1.0
    assert report["height_floor"]["applied"]
    assert np.ptp(after[:, 0]) > np.ptp(before[:, 0])
    assert np.ptp(after[:, 1]) > np.ptp(before[:, 1])
    assert np.max(after[:, 2]) > np.max(before[:, 2])
    assert np.isfinite(after).all()


def test_actual_intaglio_preview_does_not_double_count_cement_gap():
    mesh = _crown()
    frame = _frame()
    margin = _margin()
    # This represents the already-gapped cutter/intaglio preview.  Cement gap is
    # geometrically baked into it, so only minimum thickness may be added outside.
    intaglio = trimesh.creation.cylinder(radius=2.25, height=4.2, sections=96)
    intaglio.apply_translation([0.0, 0.0, 2.1])

    result, report = space_driven_embryo_prefit(
        mesh, frame, margin,
        inner_envelope_mesh=intaglio,
        minimum_thickness_mm=0.80,
        cement_gap_mm=0.05,
    )

    shell = report["inner_shell"]
    assert shell["source"] == "ACTUAL_BLOCKOUT_GAP_INTAGLIO_PREVIEW"
    assert shell["actual_intaglio_preview_used"] is True
    assert abs(shell["required_outer_to_intaglio_mm"] - 0.80) < 1e-9
    audit = audit_inner_envelope_clearance(
        result, intaglio, frame, margin, 0.80,
    )
    assert audit["status"] in ("PASS_ENGINEERING", "REVIEW_REQUIRED")
    assert audit["sample_count"] > 50
    assert np.isfinite(audit["min_clearance_mm"])
    assert np.isfinite(np.asarray(result.vertices)).all()


def test_proximal_domain_is_not_allowed_to_collapse_to_a_handful_of_vertices():
    mesh = _crown()
    frame = _frame()
    margin = _margin()
    mesial = trimesh.creation.box(extents=[1.0, 7.0, 8.0])
    mesial.apply_translation([3.7, 0.0, 3.0])
    surface = FitSurface(mesial)

    masks, _locked, quality = build_proximal_domains(
        mesh, frame, margin, {"mesial": surface},
    )

    assert quality["mesial"]["status"] == "READY"
    assert int(masks["mesial"].sum()) >= 40
    assert quality["mesial"]["valid_vertex_count"] >= 24


def test_open_curved_contact_sign_uses_local_face_orientation_not_global_dot_gate():
    # One large slanted open patch.  Its normal has |dot(outward)| ~= .34, so the
    # old > .5 global-normal rule rejected it even at the patch center.
    outward = np.array([-1.0, 0.0, 0.0])
    normal = np.array([-0.342, 0.0, 0.940])
    normal /= np.linalg.norm(normal)
    u = np.array([0.0, 1.0, 0.0])
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    center = np.array([3.0, 0.0, 2.0])
    s = 2.0
    vertices = np.vstack([
        center - s*u - s*v,
        center + s*u - s*v,
        center + s*u + s*v,
        center - s*u + s*v,
    ])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    patch = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    surface = FitSurface(patch, outward)

    point = center + 0.20 * normal
    _cp, _fid, signed, distance, valid = surface.query(point[None, :])

    assert valid[0]
    assert distance[0] > 0.15
    assert signed[0] > 0.0
