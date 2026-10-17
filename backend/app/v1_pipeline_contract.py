"""Canonical CrownFlow 1.0 solver activation and interaction contract.

This module is the single startup entry point for the 1.0 geometry policies.
The important part is not only *which* implementation is active, but also which
coordinates/surfaces every stage owns.  A later solver may satisfy its own
constraint only inside its declared mutation scope; it may not silently redefine
an earlier owner's source of truth.

Canonical geometry transaction
------------------------------
1. template anatomy        : outer morphology prior only
2. patient B/L prefit      : PatientFrame B/L body coordinate only
3. confirmed Margin        : cervical fit, then hard immutable lock
4. proximal contacts       : local M/D contact domains, independently per side
5. antagonist occlusion    : local occlusal domain; established proximal protected
6. inner-shell clearance   : move OUTER only; patient intaglio remains authoritative
7. bounded local rechecks  : only if shell movement touched/drifted solved domains
8. final hollow stitch     : outer + patient intaglio share confirmed 3D Margin

The pipeline intentionally does not alternate all solvers to convergence.  That
would let owners fight each other, cost time, and make good earlier results regress.
"""
from __future__ import annotations

# Canonical activation sequence. Keep these imports ordered.
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

# Explicit mutation matrix.  ``writes`` is the only geometry a stage may own;
# ``must_preserve`` records invariants inherited from earlier stages.
STAGE_OWNERSHIP = {
    "TEMPLATE_ANATOMY_PRIOR": {
        "writes": ["OUTER_MORPHOLOGY_SEED"],
        "must_preserve": [],
        "forbidden_sources": ["PATIENT_INTAGLIO", "FINAL_PATIENT_SIZE"],
    },
    "PATIENT_FRAME_BL_PREFIT": {
        "writes": ["OUTER_BL_BODY"],
        "must_preserve": ["OUTER_MD_COORDINATE", "INSERTION_HEIGHT", "CERVICAL_ROOT"],
        "coordinate_owner": "PATIENT_FRAME_BL_ONLY",
    },
    "CONFIRMED_MARGIN_CERVICAL_LOCK": {
        "writes": ["OUTER_CERVICAL_BAND"],
        "must_preserve": ["CONFIRMED_3D_MARGIN"],
        "becomes_immutable": True,
    },
    "INDEPENDENT_PROXIMAL_CONTACTS": {
        "writes": ["LOCAL_PROXIMAL_PATCH"],
        "must_preserve": ["CERVICAL_LOCK", "OPPOSITE_SOLVED_CONTACT"],
        "side_independent": True,
    },
    "ANTAGONIST_OCCLUSION": {
        "writes": ["LOCAL_OCCLUSAL_PATCH"],
        "must_preserve": ["CERVICAL_LOCK", "SOLVED_PROXIMAL_CONTACTS"],
        "height_owner": "REAL_ANTAGONIST_SURFACE",
    },
    "PATIENT_INTAGLIO_SHELL_CLEARANCE": {
        "writes": ["OUTER_SHELL_ONLY"],
        "must_preserve": ["PATIENT_INTAGLIO", "CONFIRMED_3D_MARGIN", "CERVICAL_LOCK"],
        "recheck_if_touched": ["PROXIMAL", "OCCLUSAL"],
    },
    "LOCAL_CONTACT_OCCLUSION_RECHECK": {
        "writes": ["TOUCHED_SOLVED_DOMAIN_ONLY"],
        "must_preserve": ["CERVICAL_LOCK", "PATIENT_INTAGLIO"],
        "bounded": True,
    },
    "CONFIRMED_MARGIN_HOLLOW_STITCH": {
        "writes": ["CERVICAL_SEAM_TRIANGLES"],
        "must_preserve": ["OUTER_BODY", "PATIENT_INTAGLIO", "CONFIRMED_3D_MARGIN"],
        "shared_boundary": "CONFIRMED_3D_MARGIN",
    },
}

INTERACTION_RULES = {
    "template": "anatomy prior only; never patient dimension or intaglio source",
    "bl": "run once before cervical/contact solve; PatientFrame B/L only, never global crown scaling",
    "margin": "becomes immutable after cervical fit and is restored after every downstream solver",
    "proximal": "solve each real side independently; protect the opposite established contact",
    "occlusion": "real antagonist owns clearance; established proximal contact vertices are protected",
    "intaglio": "prepared abutment + insertion path + blockout + cement gap; template interior is ignored",
    "shell": "intaglio is read-only; move outer only and rerun contacts only when touched/drifted",
    "final": "stitch only open outer/open intaglio at confirmed Margin; Boolean cutter is never UI inner",
    "preview": "QC/review status never hides generated geometry and never replaces it with an older mesh",
    "sculpt": "user edits persist; analysis may warn but may not roll the visible outer back",
}


def _name(obj):
    return getattr(obj, "__name__", type(obj).__name__)


def runtime_contract_report() -> dict:
    """Return active owners so every debug bundle can prove live bindings."""
    return {
        "version": "CROWNFLOW_1_0_OWNER_CONTRACT_V2",
        "owner_order": list(OWNER_ORDER),
        "stage_ownership": STAGE_OWNERSHIP,
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
    """Fail startup if import order or a later patch changes a geometry owner."""
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

    # Validated bounds from the posterior real-case series. These are not clinical
    # certification values; they only prevent a later compatibility patch from
    # silently restoring the exact conservative behavior that caused regressions.
    if float(base_geometry.BL_MAX_EXPANSION_SCALE) < 1.30:
        errors.append("B/L max expansion regressed below 1.30")
    if float(base_geometry.BL_CERVICAL_LOCK_MM) > 0.40:
        errors.append("B/L deformation reaches too close to the confirmed Margin")
    if float(base_geometry.BL_FULL_BODY_START_MM) > 0.90:
        errors.append("B/L full-body correction starts too coronally")
    if int(proximal_v9.FINAL_KISS_MAX_PASSES) < 3:
        errors.append("proximal final-kiss budget regressed below validated value")

    # Contract self-consistency: later owners must explicitly inherit the Margin
    # and patient-intaglio invariants rather than relying on comments/import order.
    downstream = (
        "INDEPENDENT_PROXIMAL_CONTACTS", "ANTAGONIST_OCCLUSION",
        "PATIENT_INTAGLIO_SHELL_CLEARANCE", "LOCAL_CONTACT_OCCLUSION_RECHECK",
    )
    for stage in downstream:
        preserved = set(STAGE_OWNERSHIP[stage].get("must_preserve", ()))
        if "CERVICAL_LOCK" not in preserved:
            errors.append(f"{stage} does not declare cervical-lock preservation")
    if "PATIENT_INTAGLIO" not in STAGE_OWNERSHIP["PATIENT_INTAGLIO_SHELL_CLEARANCE"]["must_preserve"]:
        errors.append("shell stage does not declare patient intaglio read-only")

    if errors:
        raise RuntimeError("CrownFlow 1.0 solver contract violation: " + "; ".join(errors))
    return runtime_contract_report()


# Validate immediately so an invalid process never serves a mixed-owner pipeline.
ACTIVE_CONTRACT = assert_runtime_contract()
