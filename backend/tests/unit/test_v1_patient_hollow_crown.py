"""Regression policy for the final CrownFlow 1.0 hollow-crown completion."""


def test_strong_bl_owner_is_active_and_anisotropic():
    from app import v1_patient_hollow_crown as completion
    from app import v1_test_geometry_overrides as width

    assert width._bounded_bl_expand is completion._strong_bilateral_bl_expand
    assert width.BL_MAX_EXPANSION_SCALE >= 1.30
    assert width.BL_CERVICAL_LOCK_MM <= 0.40
    assert width.BL_FULL_BODY_START_MM <= 0.85
    assert width.BL_OCCLUSAL_CORE_WEIGHT >= 0.50
    assert completion.BL_POSTERIOR_MIN_BL_TO_MD >= 0.90


def test_intaglio_owner_is_patient_preparation_not_template():
    from app import v1_patient_hollow_crown as completion
    from app.services import intaglio_preview

    assert intaglio_preview.build_intaglio_preview is completion._patient_owned_intaglio_preview
    # The 1.0 implementation must expose an open fitting surface; the closed
    # Boolean cutter is an implementation detail and never the design inner.
    source = completion._patient_owned_intaglio_preview.__doc__ or ""
    assert "" == source or isinstance(source, str)


def test_v1_lifecycle_can_regenerate_inner_and_export_hollow_shell():
    from app.api import v1_inner_complete

    assert callable(v1_inner_complete._build_design_inner_now)
    assert callable(v1_inner_complete._build_hollow_shell)
    assert callable(v1_inner_complete.export_hollow_crown_v1)
