from __future__ import annotations

import math

import numpy as np
import trimesh

from app.api.abutment_fast import _automatic_abutment_frame, _automatic_neighbor_patch
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


def _margin(radius_x=1.5, radius_y=1.2):
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    return np.c_[radius_x * np.cos(theta), radius_y * np.sin(theta), np.zeros_like(theta)]


def _frame():
    return confirmed_patient_frame(
        origin=(0.0, 0.0, 0.0),
        mesial=(1.0, 0.0, 0.0),
        buccal=(0.0, 1.0, 0.0),
        insertion=(0.0, 0.0, 1.0),
        revision_id="abutment-fast-test",
    )


def test_automatic_abutment_frame_is_right_handed_and_radial():
    # Subdivide the synthetic jaw so the fixture satisfies the same minimum-mesh
    # guard as production IOS scans.  The old two-box fixture had only 16 vertices
    # and was testing the guard rather than the frame geometry.
    left = trimesh.creation.box(extents=[2.0, 5.0, 3.0]).subdivide()
    left.apply_translation([-4.0, 0.0, 1.5])
    right = trimesh.creation.box(extents=[2.0, 5.0, 3.0]).subdivide()
    right.apply_translation([4.0, 0.0, 1.5])
    jaw = trimesh.util.concatenate([left, right])

    x, y, z, report = _automatic_abutment_frame(
        jaw, np.array([4.0, 0.0, 0.0]), [0.0, 0.0, 1.0],
    )

    assert np.isclose(np.linalg.norm(x), 1.0)
    assert np.isclose(np.linalg.norm(y), 1.0)
    assert np.isclose(np.linalg.norm(z), 1.0)
    assert np.allclose(np.cross(z, x), y)
    assert y[0] > 0.8
    assert report["radial_length_mm"] > 1.0


def test_automatic_neighbor_patch_finds_coronal_surfaces_on_both_sides():
    mesial = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    mesial.apply_translation([4.0, 0.0, 2.5])
    distal = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    distal.apply_translation([-4.0, 0.0, 2.5])
    jaw = trimesh.util.concatenate([mesial, distal])
    frame = _frame()

    mesial_patch, mesial_report = _automatic_neighbor_patch(jaw, frame, _margin(), "mesial", 3.0)
    distal_patch, distal_report = _automatic_neighbor_patch(jaw, frame, _margin(), "distal", 3.0)

    assert mesial_patch is not None
    assert distal_patch is not None
    assert mesial_report["status"] == "CONFIRMED"
    assert distal_report["status"] == "CONFIRMED"
    assert mesial_report["method"] == "CORONAL_MD_CORE_PLUS_REFERENCE_HALO_AUTO_V4"
    assert mesial_report["reference_face_count"] >= mesial_report["core_face_count"]
    assert distal_report["reference_face_count"] >= distal_report["core_face_count"]
    assert mesial_report["reference_inner_floor_mm"] < mesial_report["half_margin_width_mm"]
    assert distal_report["reference_inner_floor_mm"] < distal_report["half_margin_width_mm"]
    assert mesial_report["reference_outward_vs_frame_dot"] > 0.0
    assert distal_report["reference_outward_vs_frame_dot"] > 0.0

    _, _, mesial_outward = mesial_patch
    _, _, distal_outward = distal_patch
    assert np.dot(mesial_outward, frame.x) < -0.9
    assert np.dot(distal_outward, frame.x) > 0.9


def test_automatic_neighbor_patch_does_not_require_reliable_face_winding():
    mesial = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    mesial.apply_translation([4.0, 0.0, 2.5])
    mesial.invert()  # deliberately reverse normals/winding

    patch, report = _automatic_neighbor_patch(mesial, _frame(), _margin(), "mesial", 3.5)

    assert patch is not None
    assert report["status"] == "CONFIRMED"
    assert report["reference_face_count"] >= report["core_face_count"]
    assert report["reference_outward_vs_frame_dot"] > 0.0


def test_automatic_neighbor_patch_classifies_empty_side_as_absent():
    mesial = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    mesial.apply_translation([4.0, 0.0, 2.5])

    patch, report = _automatic_neighbor_patch(mesial, _frame(), _margin(), "distal", 3.5)

    assert patch is None
    assert report["status"] == "ABSENT"
    assert report["reason"] == "NO_CORONAL_NEIGHBOR_WITHIN_SEARCH_RANGE"
