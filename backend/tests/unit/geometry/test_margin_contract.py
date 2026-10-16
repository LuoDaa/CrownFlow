import numpy as np

from app.geometry.crown_outer.margin_contract import (
    assess_margin_against_support,
    sanitize_margin_loop,
)


def _ellipse(rx, ry, n=128, z=0.0):
    t = np.arange(n) * 2.0 * np.pi / n
    return np.c_[rx * np.cos(t), ry * np.sin(t), np.full(n, z)]


def _support(rx=4.0, ry=3.2, height=5.0, n=128):
    rings = []
    for z, scale in ((0.0, 1.0), (height * 0.5, 0.85), (height, 0.65)):
        rings.append(_ellipse(rx * scale, ry * scale, n=n, z=z))
    return np.vstack(rings)


def test_sanitize_margin_removes_only_consecutive_duplicates():
    ring = _ellipse(3.2, 2.5, n=64)
    dirty = []
    for i, p in enumerate(ring):
        dirty.append(p)
        if i % 3 == 0:
            dirty.extend([p.copy(), p.copy()])
    dirty.append(ring[0].copy())
    clean, report = sanitize_margin_loop(np.asarray(dirty))
    assert len(clean) == len(ring)
    assert report["removed_consecutive_duplicates"] > 0
    assert report["closing_duplicate_removed"] is True
    assert np.allclose(clean, ring)


def test_margin_contract_rejects_tiny_ring_relative_to_prep():
    support = _support()
    tiny = _ellipse(1.05, 1.0, n=96)
    report = assess_margin_against_support(tiny, support, [0, 0, 1])
    assert report["accepted"] is False
    assert report["reason"] == "MARGIN_IMPLAUSIBLE_RELATIVE_TO_PREP"
    assert report["projected_area_ratio"] < report["minimum_projected_area_ratio"]


def test_margin_contract_accepts_cervical_scale_ring():
    support = _support()
    margin = _ellipse(3.25, 2.55, n=128)
    report = assess_margin_against_support(margin, support, [0, 0, 1])
    assert report["accepted"] is True
    assert report["projected_self_intersections"] == 0


def test_margin_contract_rejects_projected_self_intersection():
    support = _support()
    # Ordered bow-tie repeated around the footprint: adequate span, invalid topology.
    bow = np.array([
        [-3.0, -2.2, 0.0], [3.0, 2.2, 0.0],
        [-3.0, 2.2, 0.0], [3.0, -2.2, 0.0],
        [3.1, -2.0, 0.0], [3.1, 2.0, 0.0],
        [-3.1, 2.0, 0.0], [-3.1, -2.0, 0.0],
        [-2.8, -2.3, 0.0], [2.8, 2.3, 0.0],
        [-2.8, 2.3, 0.0], [2.8, -2.3, 0.0],
    ])
    report = assess_margin_against_support(bow, support, [0, 0, 1])
    assert report["accepted"] is False
    assert report["projected_self_intersections"] > 0
