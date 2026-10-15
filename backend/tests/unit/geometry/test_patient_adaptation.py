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
    # A closed tapered crown-like body with a deliberately undersized cervical base.
    mesh = trimesh.creation.cone(radius=2.0, height=6.0, sections=96)
    mesh.apply_translation([0.0, 0.0, 0.0])
    return mesh


def test_cervical_fit_moves_outer_rim_toward_margin():
    mesh = _crown()
    margin = _margin()
    frame = _frame()

    before = np.asarray(mesh.vertices).copy()
    result, report = cervical_fit_to_margin(mesh, frame, margin)

    assert report["status"] == "PASS_ENGINEERING"
    assert report["rim_vertex_count"] > 0
    assert report["after_mm"]["median"] < report["before_mm"]["median"]
    assert np.isfinite(result.vertices).all()
    assert not np.allclose(before, result.vertices)


def test_global_prefit_expands_md_width_between_two_neighbors():
    mesh = _crown()
    margin = _margin()
    frame = _frame()

    # Two solid neighbor blocks leave roughly 5.8 mm between their facing surfaces.
    mesial = trimesh.creation.box(extents=[1.0, 5.0, 8.0])
    mesial.apply_translation([3.4, 0.0, 3.0])
    distal = trimesh.creation.box(extents=[1.0, 5.0, 8.0])
    distal.apply_translation([-3.4, 0.0, 3.0])
    surfaces = {"mesial": FitSurface(mesial), "distal": FitSurface(distal)}

    before_x = np.ptp(np.asarray(mesh.vertices)[:, 0])
    result, report = global_patient_prefit(
        mesh, frame, margin,
        proximal_surfaces=surfaces,
        proximal_target_mm=0.05,
    )
    after_x = np.ptp(np.asarray(result.vertices)[:, 0])

    assert report["md"]["applied"]
    assert after_x > before_x
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
    # Cervical vertices remain substantially more stable than the occlusal top.
    low = before[:, 2] <= np.quantile(before[:, 2], 0.10)
    high = before[:, 2] >= np.quantile(before[:, 2], 0.90)
    assert np.mean(np.linalg.norm(after[low] - before[low], axis=1)) < np.mean(
        np.linalg.norm(after[high] - before[high], axis=1)
    )
    assert np.isfinite(result.vertices).all()
