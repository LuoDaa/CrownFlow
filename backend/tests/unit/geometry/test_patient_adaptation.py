import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_adaptation import (
    cervical_fit_to_margin,
    global_patient_prefit,
)
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


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


def _duplicate_pairs(vertices, tol=1e-8):
    from scipy.spatial import cKDTree
    return len(cKDTree(np.asarray(vertices)).query_pairs(tol))


def test_cervical_fit_moves_outer_rim_toward_margin_without_collapsing_vertices():
    mesh = _crown()
    margin = _margin()
    frame = _frame()

    before = np.asarray(mesh.vertices).copy()
    before_duplicates = _duplicate_pairs(before)
    result, report = cervical_fit_to_margin(mesh, frame, margin)

    assert report["status"] == "PASS_ENGINEERING"
    assert report["rim_vertex_count"] > 0
    assert report["seam_vertex_count"] > 0
    assert report["after_mm"]["median"] < report["before_mm"]["median"]
    assert report["after_mm"]["p95"] <= 0.05
    assert report["full_loop_coverage"]["within_0_05mm_fraction"] >= 0.98
    assert report["duplicate_pairs_after"] == report["duplicate_pairs_before"]
    assert _duplicate_pairs(result.vertices) == before_duplicates
    assert np.isfinite(result.vertices).all()
    assert not np.allclose(before, result.vertices)


def test_cervical_fit_uses_continuous_margin_segments_not_discrete_sample_snap():
    mesh = _crown()
    margin = _margin(n=20)
    frame = _frame()

    result, report = cervical_fit_to_margin(mesh, frame, margin)

    assert report["projection"].startswith("continuous_closed_margin_segment")
    assert report["segment_count_used"] > 4
    assert report["full_loop_anchor_count"] == len(margin)
    assert report["full_loop_coverage"]["within_0_05mm_fraction"] >= 0.98
    assert report["duplicate_pairs_after"] == report["duplicate_pairs_before"]
    assert report["after_mm"]["p95"] <= 0.05
    assert np.isfinite(result.vertices).all()


def test_cervical_fit_covers_margin_arc_even_when_library_cervix_is_locally_high():
    mesh = _crown()
    vertices = np.asarray(mesh.vertices).copy()
    lifted = (
        (vertices[:, 0] > 0.0)
        & (vertices[:, 1] > 0.0)
        & (vertices[:, 2] < 1.25)
    )
    vertices[lifted, 2] += 0.75
    mesh.vertices = vertices

    margin = _margin(n=96)
    result, report = cervical_fit_to_margin(mesh, _frame(), margin)

    assert report["full_loop_anchor_count"] == len(margin)
    assert report["full_loop_coverage"]["covered_samples"] >= int(0.98 * len(margin))
    assert report["full_loop_coverage"]["reverse_p95_mm"] <= 0.05
    assert report["duplicate_pairs_after"] == report["duplicate_pairs_before"]
    assert np.isfinite(result.vertices).all()


def test_cervical_fit_is_idempotent_after_full_loop_is_already_on_margin():
    margin = _margin(n=96)
    frame = _frame()
    first, first_report = cervical_fit_to_margin(_crown(), frame, margin)
    before_second = np.asarray(first.vertices).copy()
    before_duplicates = _duplicate_pairs(before_second)

    second, second_report = cervical_fit_to_margin(first, frame, margin)

    assert first_report["status"] == "PASS_ENGINEERING"
    assert second_report["status"] == "PASS_ENGINEERING"
    assert second_report["idempotent_noop"] is True
    assert second_report["duplicate_pairs_before"] == before_duplicates
    assert second_report["duplicate_pairs_after"] == before_duplicates
    assert np.allclose(second.vertices, before_second)


def test_cervical_refit_after_coronal_shell_edit_never_creates_new_duplicate_pairs():
    margin = _margin(n=96)
    frame = _frame()
    fitted, _ = cervical_fit_to_margin(_crown(), frame, margin)
    edited = fitted.copy()
    vertices = np.asarray(edited.vertices).copy()
    coronal = vertices[:, 2] > 0.60
    radial = vertices[:, :2]
    norm = np.linalg.norm(radial, axis=1)
    good = coronal & (norm > 1e-8)
    vertices[good, :2] += 0.25 * radial[good] / norm[good, None]
    edited.vertices = vertices
    before_duplicates = _duplicate_pairs(vertices)

    result, report = cervical_fit_to_margin(edited, frame, margin)

    assert report["duplicate_pairs_after"] <= before_duplicates
    assert report["duplicate_guard"]["policy"] == "BACKTRACK_INSTEAD_OF_ABORT"
    assert np.isfinite(result.vertices).all()


def test_global_prefit_expands_md_width_between_two_neighbors():
    mesh = _crown()
    margin = _margin()
    frame = _frame()

    mesial = trimesh.creation.box(extents=[1.0, 5.0, 8.0])
    mesial.apply_translation([3.4, 0.0, 3.0])
    distal = trimesh.creation.box(extents=[1.0, 5.0, 8.0])
    distal.apply_translation([-3.4, 0.0, 3.0])
    surfaces = {"mesial": FitSurface(mesial), "distal": FitSurface(distal)}

    before = np.asarray(mesh.vertices).copy()
    before_x = np.ptp(before[:, 0])
    result, report = global_patient_prefit(
        mesh, frame, margin,
        proximal_surfaces=surfaces,
        proximal_target_mm=0.05,
    )
    after = np.asarray(result.vertices)
    after_x = np.ptp(after[:, 0])

    assert report["md"]["applied"]
    assert after_x > before_x
    assert report["md"]["whole_crown_translation_mm"] == 0.0
    assert np.max(np.linalg.norm(after - before, axis=1)) > 0.05
    assert np.isfinite(result.vertices).all()


def test_global_prefit_raises_height_toward_antagonist_without_moving_margin_pivot():
    mesh = _crown()
    margin = _margin()
    frame = _frame()

    antagonist = trimesh.creation.box(extents=[12.0, 12.0, 1.0])
    antagonist.apply_translation([0.0, 0.0, 8.5])
    surface = FitSurface(antagonist)

    before = np.asarray(mesh.vertices).copy()
    result, report = global_patient_prefit(
        mesh, frame, margin,
        antagonist_surface=surface,
        occlusal_target_mm=0.05,
    )
    after = np.asarray(result.vertices)

    assert report["height"]["applied"]
    assert after[:, 2].max() > before[:, 2].max()
    low = before[:, 2] <= np.quantile(before[:, 2], 0.10)
    high = before[:, 2] >= np.quantile(before[:, 2], 0.90)
    assert np.mean(np.linalg.norm(after[low] - before[low], axis=1)) < np.mean(
        np.linalg.norm(after[high] - before[high], axis=1)
    )
    assert np.isfinite(result.vertices).all()
