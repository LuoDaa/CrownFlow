"""Canonical CrownFlow 1.0 solver activation and interaction contract.

Why this module exists
----------------------
The 1.0 geometry stack intentionally contains several narrowly-scoped runtime
patches accumulated while validating real posterior cases.  Import order therefore
matters: a late conservative patch must never silently replace the final B/L owner,
and a legacy intaglio preview must never replace the patient-owned open fitting
surface.

This module is the *only* startup entry point for those policies.  It activates
owners in one documented order, then fails fast if any live function binding no
longer matches that contract.

Geometry ownership order
------------------------
1. template anatomy        - shape prior only; never patient size/intaglio truth
2. patient B/L prefit      - PatientFrame Y only; M/D untouched
3. confirmed Margin        - cervical fit then immutable lock
4. proximal contacts       - independent real-neighbour surfaces
5. antagonist occlusion    - protected proximal contacts
6. inner-shell clearance   - patient intaglio/material floor; local repair only
7. contact/occlusion recheck when shell movement touched those domains
8. final hollow stitch     - outer + patient intaglio share confirmed 3D Margin

The ordering deliberately avoids iterative owner fights.  A lower item may not
silently redefine an earlier owner's source of truth; it may only use the explicit
protection/recheck paths already present in ``services.auto_fit``.
"""
from __future__ import annotations

# Canonical activation sequence.  Keep these imports ordered.
from . import v1_test_geometry_overrides as base_geometry
from . import v1_final_geometry_guards as contact_occlusion
from . import v1_intaglio_support_guard as support_guard
from . import v1_margin_bl_completion as margin_bl_completion
from . import v1_patient_hollow_crown as hollow_crown
from . import v1_intaglio_cache_v5 as intaglio_cache

from .geometry.crown_outer import auto_fit_adapter
from .geometry.crown_outer import proximal_v9
from .services import generation_runtime
from .services import intaglio_preview


OWNER_ORDER = (
    "TEMPLATE_ANATOMY_PRIOR",
    "PATIENT_FRAME_BL_PREFIT",
    "CONFIRMED_MARGIN_CERVICAL_LOCK",
    "INDEPENDENT_PROXIMAL_CONTACTS",
    "ANTAGONIST_OCCLUSION",
    "PATIENT_INTAGLIO_SHELL_CLEARANCE",
    "LOCAL_CONTACT_OCCLUSION_RECHECK",
    "CONFIRMED_MARGIN_HOLLOW_STITCH",
)

INTERACTION_RULES = {
    "template": "anatomy prior only; never patient dimension or intaglio source",
    "bl": "PatientFrame B/L coordinate only; M/D and insertion height are not scaled",
    "margin": "becomes immutable after cervical fit and is restored after every downstream solver",
    "proximal": "each real side is solved independently; established opposite contact is protected",
    "occlusion": "antagonist owns clearance; established proximal contact vertices are protected",
    "intaglio": "prepared abutment + insertion path + blockout + cement gap; template interior is ignored",
    "shell": "may move outer only for material clearance; contact/occlusion rerun only when touched or drifted",
    "final": "open outer and open intaglio join only at the confirmed 3D Margin",
}


def _name(obj):
    return getattr(obj, "__name__", type(obj).__name__)


def runtime_contract_report() -> dict:
    """Return the active owners so every test/debug bundle can prove bindings."""
    return {
        "version": "CROWNFLOW_1_0_OWNER_CONTRACT_V1",
        "owner_order": list(OWNER_ORDER),
        "interaction_rules": dict(INTERACTION_RULES),
        "bindings": {
            "bl_expand": _name(base_geometry._bounded_bl_expand),
            "intaglio_builder": _name(intaglio_preview.build_intaglio_preview),
            "occlusal_target_owner": _name(auto_fit_adapter._desired_targets),
            "fit_stage_owner": _name(auto_fit_adapter.fit_stage),
            "intaglio_cache_payload": _name(generation_runtime.intaglio_cache_payload),
        },
        "constants": {
            "bl_max_scale": float(base_geometry.BL_MAX_EXPANSION_SCALE),
            "bl_cervical_lock_mm": float(base_geometry.BL_CERVICAL_LOCK_MM),
            "bl_full_body_start_mm": float(base_geometry.BL_FULL_BODY_START_MM),
            "bl_occlusal_core_weight": float(base_geometry.BL_OCCLUSAL_CORE_WEIGHT),
            "proximal_final_kiss_max_move_mm": float(proximal_v9.FINAL_KISS_MAX_MOVE_MM),
            "proximal_final_kiss_passes": int(proximal_v9.FINAL_KISS_MAX_PASSES),
        },
    }


def assert_runtime_contract() -> dict:
    """Fail startup if a patch/import-order regression changes a geometry owner."""
    errors = []

    if base_geometry._bounded_bl_expand is not hollow_crown._strong_bilateral_bl_expand:
        errors.append("B/L owner is not patient-frame bilateral expansion")
    if margin_bl_completion._width._bounded_bl_expand is not hollow_crown._strong_bilateral_bl_expand:
        errors.append("margin/B-L compatibility layer captured a stale B/L owner")
    if intaglio_preview.build_intaglio_preview is not hollow_crown._patient_owned_intaglio_preview:
        errors.append("intaglio owner is not prepared-abutment patient-owned open surface")
    if margin_bl_completion._intaglio_preview.build_intaglio_preview is not hollow_crown._patient_owned_intaglio_preview:
        errors.append("margin completion layer captured a stale intaglio owner")
    if auto_fit_adapter._desired_targets is not contact_occlusion._antagonist_owned_targets:
        errors.append("occlusal targets are not antagonist-owned targets")
    if auto_fit_adapter.fit_stage is not contact_occlusion._patient_space_fit_stage:
        errors.append("fit_stage lost the patient-space pass policy")
    if generation_runtime.intaglio_cache_payload is not intaglio_cache._intaglio_cache_payload_v5:
        errors.append("intaglio cache is not v5 patient-owned/open-surface semantics")

    # Sanity bounds prevent a later tuning patch from silently returning to the
    # exact conservative settings that produced the validated real-case failures.
    if float(base_geometry.BL_MAX_EXPANSION_SCALE) < 1.30:
        errors.append("B/L max expansion regressed below 1.30")
    if float(base_geometry.BL_CERVICAL_LOCK_MM) > 0.40:
        errors.append("B/L deformation reaches too close to the confirmed Margin")
    if int(proximal_v9.FINAL_KISS_MAX_PASSES) < 3:
        errors.append("proximal final-kiss budget regressed below validated value")

    if errors:
        raise RuntimeError("CrownFlow 1.0 solver contract violation: " + "; ".join(errors))
    return runtime_contract_report()


# Validate immediately so an invalid process never serves a mixed-owner pipeline.
ACTIVE_CONTRACT = assert_runtime_contract()
