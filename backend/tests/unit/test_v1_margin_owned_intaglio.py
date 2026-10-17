from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "app"


def test_margin_owned_intaglio_policy_is_loaded_before_api_graph():
    source = (ROOT / "main.py").read_text(encoding="utf-8")
    policy = source.index("v1_margin_owned_intaglio")
    api_graph = source.index("from .api.crown")
    assert policy < api_graph


def test_final_assembly_uses_direct_source_clip_not_boolean_source_classification():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert "DIRECT_SOURCE_CLIP_AUTHORITATIVE" in source
    assert "boolean_source_classification_used\": False" in source
    assert "AUDIT_ONLY_NOT_SOURCE_SURFACE" in source
    assert "_single_open_surface_above_margin" in source


def test_accepted_outer_is_frozen_only_during_finalization():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert "ContextVar" in source
    assert "ACCEPTED_OUTER_IMMUTABLE_DURING_FINAL_INTAGLIO" in source
    assert "outer_modified\": False" in source
    assert "_ORIGINAL_CLEARANCE_ENFORCER" in source
    assert "_FINALIZATION_OUTER_FROZEN.set(True)" in source


def test_shared_confirmed_margin_remains_final_topology_owner():
    source = (ROOT / "v1_margin_owned_intaglio.py").read_text(encoding="utf-8")
    assert "EXACT_SHARED_3D_MARGIN_COLLARS" in source
    assert "SHARED_CONFIRMED_3D_MARGIN" in source
    assert "stitch_cervical_surfaces" in source


def test_frontend_remains_single_user_action():
    source = (ROOT.parents[1] / "frontend" / "src" / "app" / "pages" / "CasePage.jsx").read_text(encoding="utf-8")
    assert 'label:"牙冠生成"' in source
    assert "generateAbutmentOrder" in source
