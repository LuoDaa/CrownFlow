"""Regression checks for CrownFlow 1.0 owner ordering and live bindings."""


def test_pipeline_contract_has_one_canonical_owner_order():
    from app import v1_pipeline_contract as contract

    report = contract.assert_runtime_contract()
    assert report["version"] == "CROWNFLOW_1_0_OWNER_CONTRACT_V1"
    assert report["owner_order"] == list(contract.OWNER_ORDER)
    assert report["owner_order"].index("PATIENT_FRAME_BL_PREFIT") < report["owner_order"].index(
        "CONFIRMED_MARGIN_CERVICAL_LOCK"
    )
    assert report["owner_order"].index("CONFIRMED_MARGIN_CERVICAL_LOCK") < report["owner_order"].index(
        "INDEPENDENT_PROXIMAL_CONTACTS"
    )
    assert report["owner_order"].index("INDEPENDENT_PROXIMAL_CONTACTS") < report["owner_order"].index(
        "ANTAGONIST_OCCLUSION"
    )
    assert report["owner_order"].index("ANTAGONIST_OCCLUSION") < report["owner_order"].index(
        "PATIENT_INTAGLIO_SHELL_CLEARANCE"
    )


def test_final_bindings_cannot_be_overwritten_by_legacy_layers():
    from app import v1_pipeline_contract as contract
    from app import v1_patient_hollow_crown as hollow
    from app import v1_test_geometry_overrides as width
    from app.geometry.crown_outer import auto_fit_adapter
    from app.services import intaglio_preview

    contract.assert_runtime_contract()
    assert width._bounded_bl_expand is hollow._strong_bilateral_bl_expand
    assert intaglio_preview.build_intaglio_preview is hollow._patient_owned_intaglio_preview
    assert auto_fit_adapter.fit_stage is contract.contact_occlusion._patient_space_fit_stage
    assert auto_fit_adapter._desired_targets is contract.contact_occlusion._antagonist_owned_targets


def test_interaction_rules_keep_geometry_sources_separate():
    from app import v1_pipeline_contract as contract

    rules = contract.INTERACTION_RULES
    assert "M/D" in rules["bl"]
    assert "prepared abutment" in rules["intaglio"]
    assert "template interior is ignored" in rules["intaglio"]
    assert "confirmed 3D Margin" in rules["final"]
    assert "restored after every downstream solver" in rules["margin"]
