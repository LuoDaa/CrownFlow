from app.geometry.crown_outer.auto_fit_adapter import _active_subset_better


def _report(median, p95, maximum=1.0, count=64):
    return {
        "status": "MEASURED_ENGINEERING",
        "valid_count": count,
        "target_error": {
            "median_mm": median,
            "p95_mm": p95,
            "max_mm": maximum,
        },
    }


def test_active_occlusal_subset_accepts_clear_local_improvement():
    # Regression for CAD-1 OCCLUSAL: only a compact contact/penetration subset is
    # edited, so full-domain p95 may stay unchanged while the actual moved patch
    # improves strongly.  Acceptance must use the exact seed subset.
    before = {"occlusal": _report(0.31, 0.405, 0.405)}
    after = {"occlusal": _report(0.09, 0.14, 0.16)}
    assert _active_subset_better(before, after)


def test_active_occlusal_subset_rejects_non_improving_trial():
    before = {"occlusal": _report(0.12, 0.20, 0.25)}
    after = {"occlusal": _report(0.13, 0.23, 0.28)}
    assert not _active_subset_better(before, after)


def test_active_occlusal_subset_allows_median_improvement_when_p95_is_stable():
    before = {"occlusal": _report(0.18, 0.25000, 0.30)}
    after = {"occlusal": _report(0.10, 0.25005, 0.31)}
    assert _active_subset_better(before, after)
