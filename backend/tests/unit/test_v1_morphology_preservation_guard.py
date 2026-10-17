"""Regression checks for the CrownFlow 1.0 broad-morph preservation policy."""


def test_patient_envelope_cannot_globally_squash_posterior_anatomy():
    from app import v1_morphology_preservation_guard as guard

    cfg = guard._preserving_envelope_config()
    assert cfg.md_min_scale_per_pass >= 0.90
    assert cfg.height_min_scale_per_pass >= 0.90
    assert cfg.bl_min_scale_per_pass >= 1.00


def test_occlusal_height_prefit_is_only_a_coarse_initializer():
    from app import v1_morphology_preservation_guard as guard

    cfg = guard._preserving_space_config()
    assert cfg.height_iterations <= 4
    assert cfg.height_max_step_mm <= 0.35
    assert cfg.height_local_blend <= 0.35
    assert cfg.height_smoothing_passes <= 2
    # Preserve the conservative shell budgets already installed by the stable
    # 1.0 override instead of replacing them.
    assert cfg.envelope_iterations <= 2
    assert cfg.envelope_max_push_per_pass_mm <= 0.25


def test_explicit_low_scale_config_is_raised_to_preservation_floor():
    from app import v1_morphology_preservation_guard as guard

    base = guard._ACTIVE_ENVELOPE_CONFIG(
        md_min_scale_per_pass=0.70,
        height_min_scale_per_pass=0.72,
        bl_min_scale_per_pass=0.80,
    )
    cfg = guard._raise_preservation_floors(base)
    assert cfg.md_min_scale_per_pass == guard.MORPH_MD_MIN_SCALE
    assert cfg.height_min_scale_per_pass == guard.MORPH_HEIGHT_MIN_SCALE
    assert cfg.bl_min_scale_per_pass == guard.MORPH_BL_MIN_SCALE


def test_guard_is_loaded_before_auto_fit_captures_space_morph_symbols():
    import inspect
    from app import main

    source = inspect.getsource(main)
    guard_import = source.index("v1_morphology_preservation_guard")
    auto_fit_import = source.index("from .api.auto_fit import router")
    assert guard_import < auto_fit_import
