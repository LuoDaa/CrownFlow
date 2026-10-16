import numpy as np

from app.geometry.abutment_support import _select_margin_component, _select_margin_loop


def _candidate(label, area, faces, p95=0.01, max_error=0.02, seed_distance=None):
    return {
        "label": label,
        "loop": [],
        "area_mm2": float(area),
        "face_count": int(faces),
        "margin_p95_mm": float(p95),
        "margin_max_mm": float(max_error),
        "margin_reverse_p95_mm": float(p95),
        "margin_coverage_fraction": 1.0,
        "margin_loop_length_ratio": 1.0,
        "seed_distance_mm": seed_distance,
    }


def test_support_region_auto_selects_dominant_margin_connected_component():
    selected, diagnostics = _select_margin_component([
        _candidate(3, 92.0, 1800),
        _candidate(7, 24.0, 420, p95=0.012),
    ])

    assert selected is not None
    assert selected["label"] == 3
    assert diagnostics["selection"] == "DOMINANT_MARGIN_CONNECTED_REGION"
    assert diagnostics["confidence"] == "engineering_high"
    assert diagnostics["candidate_count"] == 2


def test_support_region_keeps_truly_similar_candidates_for_review_with_diagnostics():
    selected, diagnostics = _select_margin_component([
        _candidate(3, 80.0, 1000, p95=0.010),
        _candidate(7, 74.0, 960, p95=0.011),
    ])

    assert selected is None
    assert diagnostics["selection"] == "AMBIGUOUS"
    assert diagnostics["confidence"] == "low"
    assert len(diagnostics["candidates"]) == 2
    assert {item["label"] for item in diagnostics["candidates"]} == {3, 7}


def test_support_region_seed_can_disambiguate_similar_components():
    selected, diagnostics = _select_margin_component([
        _candidate(3, 80.0, 1000, seed_distance=0.05),
        _candidate(7, 79.0, 995, seed_distance=0.40),
    ], seed=[0.0, 0.0, 0.0])

    assert selected is not None
    assert selected["label"] == 3
    assert diagnostics["selection"] == "SEED_DOMINANT"


def test_full_margin_loop_beats_short_sliver_even_when_sliver_is_numerically_exact():
    """Regression for the real FDI36 332-point finish line vs 15-point sliver bug."""
    t_margin = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    margin = np.c_[3.1 * np.cos(t_margin), 2.15 * np.sin(t_margin), np.zeros_like(t_margin)]

    # True loop follows the complete finish line with a tiny deterministic offset.
    t_true = np.linspace(0.0, 2.0 * np.pi, 332, endpoint=False)
    true = np.c_[3.1 * np.cos(t_true), 2.15 * np.sin(t_true), np.zeros_like(t_true)]
    true[:, 2] = 1e-6

    # A 15-point sliver lies essentially exactly on only a short arc.  The old
    # one-way selector could choose this because its loop->Margin error was a few
    # floating-point ulps lower than the real ring.
    t_sliver = np.linspace(-0.08, 0.08, 15)
    sliver = np.c_[3.1 * np.cos(t_sliver), 2.15 * np.sin(t_sliver), np.zeros_like(t_sliver)]

    vertices = np.vstack([true, sliver])
    true_ids = np.arange(len(true), dtype=int)
    sliver_ids = np.arange(len(true), len(true) + len(sliver), dtype=int)

    selected, diagnostics = _select_margin_loop(vertices, [sliver_ids, true_ids], margin)

    assert np.array_equal(selected, true_ids)
    assert diagnostics["eligible_count"] == 1
    assert diagnostics["selected"]["coverage_fraction"] >= 0.98
    sliver_report = diagnostics["candidates"][0]
    assert sliver_report["coverage_fraction"] < 0.25
    assert sliver_report["eligible"] is False
