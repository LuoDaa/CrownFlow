from app.geometry.abutment_support import _select_margin_component


def _candidate(label, area, faces, p95=0.01, max_error=0.02, seed_distance=None):
    return {
        "label": label,
        "loop": [],
        "area_mm2": float(area),
        "face_count": int(faces),
        "margin_p95_mm": float(p95),
        "margin_max_mm": float(max_error),
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
