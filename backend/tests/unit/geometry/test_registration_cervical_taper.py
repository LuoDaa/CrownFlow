import numpy as np

from app.geometry.registration import _tapered_cervical_scale


def test_cervical_fit_does_not_scale_coronal_buccolingual_anatomy():
    # Same B/L extent sampled at the Margin, mid-emergence and coronal anatomy.
    local = np.array([
        [0.0, -4.0, 0.0], [0.0, 4.0, 0.0],
        [0.0, -4.0, 1.1], [0.0, 4.0, 1.1],
        [0.0, -4.0, 2.2], [0.0, 4.0, 2.2],
        [0.0, -4.0, 4.0], [0.0, 4.0, 4.0],
    ])
    ring = np.array([[-2.0, -4.0], [2.0, -4.0], [2.0, 4.0], [-2.0, 4.0]])

    out, weight = _tapered_cervical_scale(
        local, ring, factor=np.array([0.75, 0.50]), cervical_z=0.0, transition_mm=2.2,
    )

    # Margin owns the root footprint.
    assert np.isclose(np.ptp(out[:2, 1]), 4.0)
    assert np.allclose(weight[:2], 1.0)

    # The emergence zone transitions smoothly instead of applying a global scale.
    assert 4.0 < np.ptp(out[2:4, 1]) < 8.0

    # Coronal anatomy is untouched once outside the emergence transition.
    assert np.isclose(np.ptp(out[4:6, 1]), 8.0)
    assert np.isclose(np.ptp(out[6:8, 1]), 8.0)
    assert np.allclose(weight[4:], 0.0)


def test_cervical_fit_keeps_md_and_bl_scaling_independent():
    local = np.array([
        [-3.0, -4.0, 0.0], [3.0, 4.0, 0.0],
        [-3.0, -4.0, 3.0], [3.0, 4.0, 3.0],
    ])
    ring = np.array([[-3.0, -4.0], [3.0, -4.0], [3.0, 4.0], [-3.0, 4.0]])

    out, _ = _tapered_cervical_scale(
        local, ring, factor=np.array([0.80, 0.55]), cervical_z=0.0, transition_mm=2.2,
    )

    # Cervical M/D and B/L can adapt by different factors.
    assert np.isclose(np.ptp(out[:2, 0]), np.ptp(local[:2, 0]) * 0.80)
    assert np.isclose(np.ptp(out[:2, 1]), np.ptp(local[:2, 1]) * 0.55)

    # Coronal dimensions remain the library morphology prior.
    assert np.allclose(out[2:, :2], local[2:, :2])
