"""CrownFlow 1.0 morphology-preservation policy.

Real-case bundles showed the broad PatientEnvelope pass shrinking posterior
anatomy before the local proximal/occlusal solvers ran. PatientEnvelope is a
maximum feasible space, not a requested tooth-size box, so hard conflicts must
be reported and delegated to local owners instead of being solved by a
destructive whole-crown squash.

This module is intentionally layered after the established 1.0 width/contact/
occlusion overrides and before services import the geometry functions. It only
changes broad pre-fit budgets; Margin, proximal v9, local occlusion targets,
support recovery and intaglio construction remain untouched.
"""
from __future__ import annotations

from dataclasses import replace

from .geometry.crown_outer import patient_envelope as _envelope
from .geometry.crown_outer import space_morph as _space


# A broad feasibility pass may make a modest M/D correction, but it must not
# erase the selected library tooth's anatomical proportions. Vertical crown
# height is now fully preserved here: the confirmed antagonist is handled later
# by the local OCCLUSAL owner, not by a global height scale.
MORPH_MD_MIN_SCALE = 0.90
MORPH_HEIGHT_MIN_SCALE = 1.00
MORPH_BL_MIN_SCALE = 1.00

# Kept as conservative compatibility defaults. The dedicated 1.0 occlusal-owner
# guard bypasses broad antagonist height deformation entirely, so these values
# are only relevant to callers that explicitly opt out of that policy.
HEIGHT_PREFIT_MAX_STEP_MM = 0.35
HEIGHT_PREFIT_MAX_ITERATIONS = 4
HEIGHT_PREFIT_LOCAL_BLEND = 0.35
HEIGHT_PREFIT_SMOOTHING_PASSES = 2


# Capture the already-active 1.0 wrappers. In particular, the envelope factory
# below still inherits v1_test_geometry_overrides' B/L no-compression policy and
# the space factory keeps its conservative shell push limits.
_ACTIVE_ENVELOPE_CONFIG = _envelope.PatientEnvelopeConfig
_ACTIVE_SPACE_CONFIG = _space.SpaceMorphConfig
_ACTIVE_MORPH = _envelope.morph_embryo_to_patient_envelope


def _preserving_envelope_config(*args, **kwargs):
    kwargs.setdefault("md_min_scale_per_pass", MORPH_MD_MIN_SCALE)
    kwargs.setdefault("height_min_scale_per_pass", MORPH_HEIGHT_MIN_SCALE)
    kwargs.setdefault("bl_min_scale_per_pass", MORPH_BL_MIN_SCALE)
    return _ACTIVE_ENVELOPE_CONFIG(*args, **kwargs)


def _preserving_space_config(*args, **kwargs):
    kwargs.setdefault("height_iterations", HEIGHT_PREFIT_MAX_ITERATIONS)
    kwargs.setdefault("height_max_step_mm", HEIGHT_PREFIT_MAX_STEP_MM)
    kwargs.setdefault("height_local_blend", HEIGHT_PREFIT_LOCAL_BLEND)
    kwargs.setdefault("height_smoothing_passes", HEIGHT_PREFIT_SMOOTHING_PASSES)
    return _ACTIVE_SPACE_CONFIG(*args, **kwargs)


def _raise_preservation_floors(config):
    """Honor explicit configs while preventing reintroduction of destructive floors."""
    if config is None:
        return _preserving_envelope_config()
    updates = {}
    if float(getattr(config, "md_min_scale_per_pass", 1.0)) < MORPH_MD_MIN_SCALE:
        updates["md_min_scale_per_pass"] = MORPH_MD_MIN_SCALE
    if float(getattr(config, "height_min_scale_per_pass", 1.0)) < MORPH_HEIGHT_MIN_SCALE:
        updates["height_min_scale_per_pass"] = MORPH_HEIGHT_MIN_SCALE
    if float(getattr(config, "bl_min_scale_per_pass", 1.0)) < MORPH_BL_MIN_SCALE:
        updates["bl_min_scale_per_pass"] = MORPH_BL_MIN_SCALE
    return replace(config, **updates) if updates else config


def _preserving_morph(mesh, frame, margin, envelope, *, config=None):
    """Run the established morph with anatomy-preserving global scale floors."""
    cfg = _raise_preservation_floors(config)
    out, report = _ACTIVE_MORPH(mesh, frame, margin, envelope, config=cfg)
    report = dict(report or {})
    sources = dict(getattr(envelope, "sources", {}) or {})
    conflicts = list(sources.get("constraint_conflicts") or [])
    report["morphology_preservation_guard"] = {
        "policy": "MAX_FEASIBLE_ENVELOPE_NOT_GLOBAL_DIMENSION_TARGET",
        "md_min_scale_per_pass": MORPH_MD_MIN_SCALE,
        "height_min_scale_per_pass": MORPH_HEIGHT_MIN_SCALE,
        "bl_min_scale_per_pass": MORPH_BL_MIN_SCALE,
        "constraint_conflict_count": len(conflicts),
        "conflicts_deferred_to_local_solvers": bool(conflicts),
        "conflict_policy": (
            "KEEP_ANATOMY_AND_REPORT_CONFLICT_DO_NOT_GLOBAL_SQUASH"
            if conflicts else "NORMAL_FEASIBILITY_PASS"
        ),
    }
    return out, report


# Patch both module namespaces. space_morph imported the envelope function by
# value, so patching only patient_envelope would leave the live generation path
# on the old global-squash implementation.
_envelope.PatientEnvelopeConfig = _preserving_envelope_config
_space.PatientEnvelopeConfig = _preserving_envelope_config
_envelope.morph_embryo_to_patient_envelope = _preserving_morph
_space.morph_embryo_to_patient_envelope = _preserving_morph
_space.SpaceMorphConfig = _preserving_space_config
