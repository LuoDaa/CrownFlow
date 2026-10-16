"""Crown 1.0: CrownOuter must consume confirmed-frame semantics without world-Z leakage."""
import math

import numpy as np
import pytest

from app.geometry.crown_outer.generator import (
    _occlusal_anchor_indices,
    _patient_frame_for_generation,
)
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


def _rotation(axis, degrees):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    a = math.radians(degrees)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c + x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)],
    ])


def _margin():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.c_[2.5 * np.cos(t), 1.8 * np.sin(t), np.zeros_like(t)]


def test_generator_rejects_confirmed_frame_axis_mismatch():
    margin = _margin()
    frame = confirmed_patient_frame(margin.mean(axis=0), [1, 0, 0], [0, 1, 0], [0, 0, 1])
    with pytest.raises(ValueError, match="patient_frame.z"):
        _patient_frame_for_generation(frame, margin, [1, 0, 0])


def test_occlusal_anchor_selection_is_rigid_transform_equivariant():
    """Top-quartile crown anchors follow Patient +Z even when world-Z changes."""
    local = np.array([
        [-1.0, -0.5, 0.5],
        [ 1.0, -0.5, 1.0],
        [-1.0,  0.5, 2.0],
        [ 1.0,  0.5, 3.0],
        [ 0.0,  0.0, 4.0],
        [ 0.3, -0.2, 5.0],
        [-0.2,  0.4, 6.0],
        [ 0.1,  0.1, 7.0],
    ])
    ids = np.arange(len(local), dtype=np.int64)
    base = confirmed_patient_frame([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    expected = _occlusal_anchor_indices(local, ids, base, 75.0)

    # Tilt the complete case so original local +Z has a large world-X component.
    r = _rotation([1.0, 1.0, 0.2], 67.0)
    translation = np.array([12.0, -5.0, 8.0])
    moved = local @ r.T + translation
    rotated = confirmed_patient_frame(
        translation,
        r @ np.array([1.0, 0.0, 0.0]),
        r @ np.array([0.0, 1.0, 0.0]),
        r @ np.array([0.0, 0.0, 1.0]),
    )
    got = _occlusal_anchor_indices(moved, ids, rotated, 75.0)

    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got, np.array([6, 7]))
