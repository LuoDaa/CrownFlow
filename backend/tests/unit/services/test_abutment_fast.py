from __future__ import annotations

import math

import numpy as np
import trimesh

from app.api.abutment_fast import _automatic_abutment_frame, _automatic_neighbor_patch
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


def _margin(radius_x=1.5, radius_y=1.2):
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    return np.c_[radius_x * np.cos(theta), radius_y * np.sin(theta), np.zeros_like(theta)]


def test_automatic_abutment_frame_is_right_handed_and_radial():
    # Arch-like cloud centered near the origin; target abutment sits on +X side.
    left = trimesh.creation.box(extents=[2.0, 5.0, 3.0])
    left.apply_translation([-4.0, 0.0, 1.5])
    right = trimesh.creation.box(extents=[2.0, 5.0, 3.0])
    right.apply_translation([4.0, 0.0, 1.5])
    jaw = trimesh.util.concatenate([left, right])

    x, y, z, report = _automatic_abutment_frame(jaw, np.array([4.0, 0.0, 0.0]), [0.0, 0.0, 1.0])

    assert np.isclose(np.linalg.norm(x), 1.0)
    assert np.isclose(np.linalg.norm(y), 1.0)
    assert np.isclose(np.linalg.norm(z), 1.0)
    assert np.allclose(np.cross(z, x), y)
    assert y[0] > 0.8  # buccal/radial points away from robust arch center
    assert report["radial_length_mm"] > 1.0


def test_automatic_neighbor_patch_finds_gap_facing_surfaces_on_both_sides():
    mesial = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    mesial.apply_translation([4.0, 0.0, 2.5])
    distal = trimesh.creation.box(extents=[2.0, 4.0, 4.0])
    distal.apply_translation([-4.0, 0.0, 2.5])
    jaw = trimesh.util.concatenate([mesial, distal])
    frame = confirmed_patient_frame(
        origin=(0.0, 0.0, 0.0),
        mesial=(1.0, 0.0, 0.0),
        buccal=(0.0, 1.0, 0.0),
        insertion=(0.0, 0.0, 1.0),
        revision_id="abutment-fast-test",
    )
    margin = _margin()

    mesial_patch, mesial_report = _automatic_neighbor_patch(jaw, frame, margin, "mesial", 3.0)
    distal_patch, distal_report = _automatic_neighbor_patch(jaw, frame, margin, "distal", 3.0)

    assert mesial_patch is not None
    assert distal_patch is not None
    assert mesial_report["status"] == "CONFIRMED"
    assert distal_report["status"] == "CONFIRMED"

    _, _, mesial_outward = mesial_patch
    _, _, distal_outward = distal_patch
    assert np.dot(mesial_outward, frame.x) < -0.9
    assert np.dot(distal_outward, frame.x) > 0.9
