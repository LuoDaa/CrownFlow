import numpy as np

from app.geometry.abutment_margin import _repair_weak_arc


def test_repair_weak_arc_prefers_cervical_candidate_over_top_excursion():
    count = 8
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    rim = np.column_stack((2.5 * np.cos(angles), 2.5 * np.sin(angles), np.zeros(count)))
    # Simulate the DP failure seen in test case 5: a short contiguous arc climbs
    # from the cervical shoulder to the abutment top.
    rim[2:4, 2] = 2.1
    valid = np.ones(count, dtype=bool)
    valid[2:4] = False

    candidates = []
    for index, theta in enumerate(angles):
        cervical = np.array([2.5 * np.cos(theta), 2.5 * np.sin(theta), 0.0])
        top = np.array([2.6 * np.cos(theta), 2.6 * np.sin(theta), 2.1])
        candidates.append(np.vstack((top, cervical)) if index in (2, 3) else np.vstack((cervical,)))

    repaired, repaired_count = _repair_weak_arc(rim, candidates, valid)

    assert repaired_count == 2
    assert np.allclose(repaired[2:4, 2], 0.0)
    assert np.allclose(np.linalg.norm(repaired[2:4, :2], axis=1), 2.5)
