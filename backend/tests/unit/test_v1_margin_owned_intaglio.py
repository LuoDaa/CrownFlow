from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "app"


def test_margin_owned_intaglio_policy_is_loaded_before_api_graph():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    policy = source.index("v1_margin_owned_intaglio")
    api_graph = source.index("from .api.crown")
    assert policy < api_graph


def test_final_assembly_separates_outer_inner_and_margin_ownership():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert "PREPARED_SUPPORT_OWNS_INTAGLIO_COMPONENT" in source
    assert '"outer_owner": "ACCEPTED_OUTER"' in source
    assert '"intaglio_owner": "PREPARED_SUPPORT"' in source
    assert '"cervical_boundary_owner": "CONFIRMED_MARGIN"' in source
    assert '"boolean_source_classification_used": False' in source
    assert "AUDIT_ONLY_NOT_SOURCE_SURFACE" in source


def test_intaglio_component_is_proven_by_support_before_collar():
    source = (ROOT / "geometry" / "canonical_intaglio.py").read_text(encoding="utf-8")
    ownership = source.index("def extract_support_owned_intaglio")
    coverage = source.index('"support_coverage"')
    axial = source.index('"axial_span_fraction"')
    assert ownership < coverage
    assert ownership < axial
    assert "PREPARED_SUPPORT_CORRESPONDENCE_V1" in source
    policy = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    fixed = policy.index("build_fixed_128_margin_crown(")
    assert policy.index("extract_support_owned_intaglio(") < fixed
    assert "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT" in policy


def test_surface_geodesic_band_replaces_axial_transition_semantics():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    geodesic = (ROOT / "geometry" / "cervical_geodesic.py").read_text(encoding="utf-8")
    assert "SURFACE_GEODESIC_DISTANCE_FROM_CONFIRMED_MARGIN" in source
    assert "inset_open_surface_geodesic" in source
    assert "SURFACE_GRAPH_GEODESIC_TRIANGLE_CLIP_V1" in geodesic
    assert "_clip_by_scalar" in geodesic
    assert "body_vertices_moved\": 0" in geodesic


def test_pre_stitch_gate_runs_before_fixed_128_collar():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    gate = source.index("cervical_feasibility(outside, inside")
    build = source.index("build_fixed_128_margin_crown(")
    assert gate < build
    assert "CERVICAL_ASSEMBLY_NOT_FEASIBLE" in source


def test_margin_and_both_cervical_rings_are_strictly_128_point_to_point():
    source = (ROOT / "geometry" / "fixed_margin_128.py").read_text(encoding="utf-8")
    assert "FIXED_MARGIN_SAMPLES = 128" in source
    assert "_conforming_ring" in source
    assert "margin128 = _resample_closed_polyline(margin, FIXED_MARGIN_SAMPLES)" in source
    assert "OUTER_I_TO_MARGIN_I_AND_INNER_I_TO_MARGIN_I" in source
    assert '"dp_zipper_used": False' in source
    assert '"intermediate_collar_ring_used": False' in source
    policy = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert '"cervical_topology_owner": "FIXED_128_SHARED_INDEX"' in policy
    assert '"margin_topology_samples": 128' in policy


def test_accepted_outer_is_frozen_only_during_finalization():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert "ContextVar" in source
    assert "ACCEPTED_OUTER_IMMUTABLE_DURING_FINAL_INTAGLIO" in source
    assert '"outer_modified": False' in source
    assert "_ORIGINAL_CLEARANCE_ENFORCER" in source
    assert "_FINALIZATION_OUTER_FROZEN.set(True)" in source


def test_frontend_remains_single_user_action():
    source = (ROOT.parents[1] / "frontend" / "src" / "app" / "pages" / "CasePage.jsx").read_text(encoding="utf-8")
    assert 'label:"牙冠生成"' in source
    assert "generateAbutmentOrder" in source
