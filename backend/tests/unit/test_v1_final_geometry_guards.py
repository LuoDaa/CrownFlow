"""Regression policy for the CrownFlow 1.0 last-mile patient-space guards."""


def test_final_geometry_guard_policy_is_active():
    from app import v1_final_geometry_guards
    from app import v1_test_geometry_overrides as width
    from app.geometry.crown_outer import auto_fit_adapter as fit
    from app.geometry.crown_outer import proximal_v9 as proximal

    # B/L must be allowed to reach the patient-derived gingiva/neighbor target;
    # normal envelope compression remains disabled by the base v1 policy.
    assert width.BL_MAX_EXPANSION_SCALE >= 1.20
    assert width.BL_ENVELOPE_SAFETY_PAD_MM >= 1.0
    assert width.BL_OCCLUSAL_CORE_WEIGHT >= 0.45

    # The remaining open physical neighbor side gets a stronger, still bounded
    # last-mile attempt while the opposite established contact remains protected.
    assert proximal.FINAL_KISS_TRIGGER_P95_MM >= 1.0
    assert 0.35 <= proximal.FINAL_KISS_MAX_MOVE_MM <= 0.50
    assert proximal.FINAL_KISS_MAX_PASSES >= 3

    # Occlusal solve is routed through the antagonist-owned target wrapper and
    # the patient-space fitter that gives the primary occlusal pass 7 iterations
    # without overriding explicit short recheck budgets.
    assert fit.OCCLUSAL_CONTACT_FRACTION >= 0.30
    assert fit._desired_targets is v1_final_geometry_guards._antagonist_owned_targets
    assert fit.fit_stage is v1_final_geometry_guards._patient_space_fit_stage
