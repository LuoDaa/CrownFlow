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
    collar = source.index("def build_canonical_margin_crown")
    assert ownership < coverage < collar
    assert ownership < axial < collar
    assert "PREPARED_SUPPORT_CORRESPONDENCE_V1" in source
    assert "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT" in (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")


def test_pre_stitch_gate_runs_before_any_canonical_collar():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    gate = source.index("cervical_feasibility(outside, inside")
    build = source.index("build_canonical_margin_crown(")
    assert gate < build
    assert "CERVICAL_ASSEMBLY_NOT_FEASIBLE" in source


def test_canonical_collar_uses_equal_count_margin_sampling_without_moving_body():
    source = (ROOT / "geometry" / "canonical_intaglio.py").read_text(encoding="utf-8")
    assert "_conforming_ring" in source
    assert "CANONICAL_MARGIN_ARCLENGTH_TWO_LAYER_COLLAR_V1" in source
    assert "shared_margin_vertices" in source
    assert "outer_source_vertices_unchanged" in source
    assert "inner_source_vertices_unchanged" in source
    assert "_quad_strip" in source


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
