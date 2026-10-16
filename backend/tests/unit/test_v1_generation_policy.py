from app import v1_test_geometry_overrides as overrides
from app.geometry.crown_outer import proximal_v9
from app.services import generation_workflow


def test_buccolingual_policy_preserves_fullness():
    cfg = overrides._patient_envelope_config()
    assert cfg.bl_min_scale_per_pass == 1.0
    assert overrides.BL_MAX_EXPANSION_SCALE >= 1.15
    assert overrides.BL_PATIENT_TARGET_MARGIN_WEIGHT == 0.50
    assert overrides.BL_PATIENT_TARGET_NEIGHBOR_WEIGHT == 0.50


def test_proximal_final_kiss_remains_guarded_but_closer():
    assert proximal_v9.FINAL_KISS_TARGET_BIAS_MM >= 0.01
    assert proximal_v9.FINAL_KISS_MAX_MOVE_MM <= 0.35
    assert proximal_v9.FINAL_KISS_NON_TARGET_TOL_MM <= 0.01


def test_occlusal_generation_keeps_positive_safety_clearance():
    assert generation_workflow.CAD1_DEFAULT_OCCLUSAL_TARGET_MM >= 0.05
    assert generation_workflow.CAD1_OCCLUSAL_SAFETY_FLOOR_MM >= 0.05
