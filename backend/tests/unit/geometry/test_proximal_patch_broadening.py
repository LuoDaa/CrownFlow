"""Crown 1.0：近远中接触不能靠单顶点/针尖去碰邻牙。"""
from types import SimpleNamespace

import numpy as np

from app.geometry.crown_outer.auto_fit_adapter import _broaden_proximal_targets


def test_proximal_seed_is_broadened_over_two_same_side_rings():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 4.0, 0.0],
    ])
    desired = reference.copy()
    desired[2] += np.array([1.0, 0.0, 0.0])
    masks = {"mesial": np.ones(len(reference), dtype=bool)}
    locked = np.array([True, False, False, False, False])
    adj = [set() for _ in reference]
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        adj[a].add(b)
        adj[b].add(a)
    frame = SimpleNamespace(x=np.array([1.0, 0.0, 0.0]))

    support, report = _broaden_proximal_targets(
        reference, desired, masks, {"mesial": {2}}, locked, adj, frame)

    assert support == {1, 3, 4}
    np.testing.assert_allclose(desired[1] - reference[1], [0.65, 0.0, 0.0])
    np.testing.assert_allclose(desired[3] - reference[3], [0.65, 0.0, 0.0])
    np.testing.assert_allclose(desired[4] - reference[4], [0.35, 0.0, 0.0])
    np.testing.assert_allclose(desired[0], reference[0])
    assert report["mesial"]["core_seed_count"] == 1
    assert report["mesial"]["support_seed_count"] == 3
    assert report["mesial"]["support_rings"] == 2
