"""Guarded micro-closure pass for CrownFlow 1.0 proximal contacts.

V7 owns the main mesial/distal solve.  V8 only gives a nearly-solved case one
small second chance.  The second result is accepted only when:

* the original result still has an unresolved side;
* the residual error is already small enough to be a micro-closure problem;
* the whole-crown displacement stays below a tight budget; and
* no previously measured side gets worse while at least one unresolved side
  improves.

This keeps the user's current morphology, Margin and occlusal relief stable while
closing the last visible proximal sliver.
"""
from __future__ import annotations

import numpy as np

from .proximal_v7 import fit_proximal_envelopes_v7


MICRO_TRIGGER_P95_MM = 0.30
MICRO_MAX_WHOLE_CROWN_MOVE_MM = 0.35
MICRO_NONWORSE_TOL_MM = 0.02
MICRO_MIN_IMPROVEMENT_MM = 0.005
MICRO_MAX_PASSES = 2


def _p95(report, side):
    state = ((report or {}).get("physical_after") or {}).get(side) or {}
    value = (state.get("target_error") or {}).get("p95_mm")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _unresolved(report, surfaces):
    per_side = (report or {}).get("per_side") or {}
    return [side for side in surfaces if (per_side.get(side) or {}).get("status") != "PASS_ENGINEERING"]


def fit_proximal_envelopes_v8(mesh, frame, margin, surfaces, target, *, protected=None,
                              transition_rings=10, max_passes=None):
    kwargs = dict(protected=protected, transition_rings=transition_rings)
    if max_passes is not None:
        kwargs["max_passes"] = max_passes
    current, report, domains = fit_proximal_envelopes_v7(
        mesh, frame, margin, surfaces, target, **kwargs,
    )
    report["solver"] = "SPACE_DRIVEN_PROXIMAL_INDEPENDENT_SIDES_V8_MICRO_CLOSURE"
    report["micro_closure"] = {"status": "NOT_NEEDED"}

    unresolved = _unresolved(report, surfaces)
    if not unresolved:
        return current, report, domains

    before_errors = {side: _p95(report, side) for side in surfaces}
    triggerable = [
        side for side in unresolved
        if before_errors.get(side) is not None and before_errors[side] <= MICRO_TRIGGER_P95_MM
    ]
    if not triggerable:
        report["micro_closure"] = {
            "status": "SKIPPED",
            "reason": "RESIDUAL_NOT_MICRO_SCALE",
            "trigger_p95_mm": MICRO_TRIGGER_P95_MM,
            "residual_p95_mm": before_errors,
        }
        return current, report, domains

    reference = np.asarray(current.vertices, float).copy()
    trial, trial_report, trial_domains = fit_proximal_envelopes_v7(
        current, frame, margin, surfaces, target,
        protected=protected,
        transition_rings=max(5, min(int(transition_rings), 8)),
        max_passes=MICRO_MAX_PASSES,
    )
    displacement = np.linalg.norm(np.asarray(trial.vertices, float) - reference, axis=1)
    max_move = float(displacement.max(initial=0.0))
    after_errors = {side: _p95(trial_report, side) for side in surfaces}

    comparable = [
        side for side in surfaces
        if before_errors.get(side) is not None and after_errors.get(side) is not None
    ]
    nonworse = bool(comparable) and all(
        after_errors[side] <= before_errors[side] + MICRO_NONWORSE_TOL_MM
        for side in comparable
    )
    improved = any(
        side in triggerable
        and before_errors.get(side) is not None
        and after_errors.get(side) is not None
        and after_errors[side] <= before_errors[side] - MICRO_MIN_IMPROVEMENT_MM
        for side in comparable
    )
    safe_move = max_move <= MICRO_MAX_WHOLE_CROWN_MOVE_MM + 1e-9
    accepted = bool(nonworse and improved and safe_move)

    micro = {
        "status": "ACCEPTED" if accepted else "REJECTED_KEEP_V7",
        "trigger_sides": triggerable,
        "before_p95_mm": before_errors,
        "after_p95_mm": after_errors,
        "max_whole_crown_move_mm": max_move,
        "move_budget_mm": MICRO_MAX_WHOLE_CROWN_MOVE_MM,
        "nonworse": nonworse,
        "improved": improved,
        "safe_move": safe_move,
    }
    if not accepted:
        report["micro_closure"] = micro
        return current, report, domains

    trial_report["solver"] = "SPACE_DRIVEN_PROXIMAL_INDEPENDENT_SIDES_V8_MICRO_CLOSURE"
    trial_report["micro_closure"] = micro
    return trial, trial_report, trial_domains
