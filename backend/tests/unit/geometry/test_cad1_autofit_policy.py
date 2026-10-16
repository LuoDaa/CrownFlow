from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.geometry.crown_outer.auto_fit_adapter import (
    OCCLUSAL_CONTACT_FRACTION,
    PROXIMAL_CONTACT_FRACTION,
    _contact_seek_mask,
    _desired_targets,
)


class _Surface:
    def __init__(self, normals, normal_sign=1.0):
        self.patch = SimpleNamespace(
            normals=np.asarray(normals, dtype=float),
            normal_sign=float(normal_sign),
        )


FRAME = SimpleNamespace(
    x=np.array([1.0, 0.0, 0.0]),
    z=np.array([0.0, 0.0, 1.0]),
)


def test_contact_seek_uses_compact_closest_patch_not_whole_wall():
    signed = np.linspace(0.20, 1.20, 20)
    valid = np.ones(20, dtype=bool)

    proximal = _contact_seek_mask(signed, valid, 0.05, PROXIMAL_CONTACT_FRACTION)
    occlusal = _contact_seek_mask(signed, valid, 0.05, OCCLUSAL_CONTACT_FRACTION)

    assert 0 < proximal.sum() < len(signed)
    assert 0 < occlusal.sum() < len(signed)
    assert signed[proximal].max() < signed[~proximal].max()
    assert signed[occlusal].max() < signed[~occlusal].max()


def test_proximal_target_motion_is_predominantly_mesiodistal():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.2, 0.1],
        [0.0, -0.2, 0.2],
        [0.0, 0.1, 0.3],
    ])
    ids = np.arange(len(reference))
    # Neighbor surface lies mainly +X, with deliberately non-zero Y/Z offsets so
    # the test can prove we do not follow closest-point vectors blindly.
    cp = reference + np.array([1.0, 0.5, 0.4])
    fids = np.zeros(len(reference), dtype=int)
    signed = np.full(len(reference), -0.10)
    distance = np.full(len(reference), 1.2)
    valid = np.ones(len(reference), dtype=bool)
    surface = _Surface([[1.0, 0.0, 0.0]])

    seed_ids, target, _ = _desired_targets(
        reference, ids, cp, fids, signed, distance, valid,
        surface, 0.05, "PROXIMAL", "mesial", FRAME, False,
    )

    delta = target - reference[seed_ids]
    assert len(seed_ids) == len(reference)
    assert np.all(np.abs(delta[:, 0]) > np.linalg.norm(delta[:, 1:], axis=1) * 3.0)


def test_underheight_occlusal_crown_is_raised_toward_antagonist():
    reference = np.array([
        [-0.5, -0.5, 4.0],
        [0.5, -0.5, 4.1],
        [-0.5, 0.5, 4.2],
        [0.5, 0.5, 4.3],
        [0.0, 0.0, 4.4],
        [0.2, 0.1, 4.5],
    ])
    ids = np.arange(len(reference))
    # Antagonist underside around z=5.5, outward normal facing the crown (-Z).
    cp = reference.copy()
    cp[:, 2] = 5.5
    fids = np.zeros(len(reference), dtype=int)
    signed = np.linspace(0.7, 1.2, len(reference))  # all farther than target
    distance = signed.copy()
    valid = np.ones(len(reference), dtype=bool)
    surface = _Surface([[0.0, 0.0, -1.0]])

    seed_ids, target, _ = _desired_targets(
        reference, ids, cp, fids, signed, distance, valid,
        surface, 0.05, "OCCLUSAL", "occlusal", FRAME, False,
    )

    # Legacy behavior returned no seeds here, so crown height never changed.
    assert 0 < len(seed_ids) < len(reference)
    delta = target - reference[seed_ids]
    assert np.all(delta[:, 2] > 0.0)
    assert np.all(np.abs(delta[:, 2]) > np.linalg.norm(delta[:, :2], axis=1) * 5.0)


def test_existing_occlusal_target_patch_does_not_attract_entire_surface():
    reference = np.zeros((5, 3), dtype=float)
    ids = np.arange(5)
    cp = reference + np.array([0.0, 0.0, 1.0])
    fids = np.zeros(5, dtype=int)
    signed = np.array([0.05, 0.06, 0.40, 0.60, 0.80])
    distance = signed.copy()
    valid = np.ones(5, dtype=bool)
    surface = _Surface([[0.0, 0.0, -1.0]])

    seed_ids, target, _ = _desired_targets(
        reference, ids, cp, fids, signed, distance, valid,
        surface, 0.05, "OCCLUSAL", "occlusal", FRAME, False,
    )

    assert len(seed_ids) == 0
    assert target.shape == (0, 3)
