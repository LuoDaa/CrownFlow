"""Regression checks for CrownFlow 1.0 derived Margin-sliver recovery."""


def _case_metrics(**overrides):
    base = {
        "eligible": False,
        "loop_to_margin_p95_mm": 1.0e-15,
        "loop_to_margin_max_mm": 1.3e-15,
        "coverage_fraction": 0.0390625,
        "length_ratio": 0.0591129771,
        "loop_length_mm": 1.1542969915,
    }
    base.update(overrides)
    return base


def test_fdi36_margin_split_sliver_is_recoverable():
    from app.v1_intaglio_support_guard import _is_margin_split_sliver_metrics

    assert _is_margin_split_sliver_metrics(_case_metrics())


def test_real_unknown_holes_remain_blocking():
    from app.v1_intaglio_support_guard import _is_margin_split_sliver_metrics

    # A loop that is not actually on Margin is a genuine unknown hole.
    assert not _is_margin_split_sliver_metrics(
        _case_metrics(loop_to_margin_p95_mm=0.20, loop_to_margin_max_mm=0.30)
    )
    # A loop reproducing a material fraction of the full Margin cannot be treated
    # as a tiny duplicate sliver even if it happens to be close to the curve.
    assert not _is_margin_split_sliver_metrics(
        _case_metrics(coverage_fraction=0.45, length_ratio=0.40, loop_length_mm=7.0)
    )
    # A full authoritative Margin loop is never eligible for sliver recovery.
    assert not _is_margin_split_sliver_metrics(
        _case_metrics(eligible=True, coverage_fraction=1.0, length_ratio=0.99, loop_length_mm=19.5)
    )
