"""Guarded one-side micro-closure pass for CrownFlow 1.0 proximal contacts.

V7 owns the main mesial/distal solve. V8 gives only the worse nearly-solved side
one small second chance. The opposite already-good contact is converted to a
protected patch before this refinement, so a final right/left-side correction
cannot pull the other side back open.
"""
from __future__ import annotations

import numpy as np

from . import proximal_v7 as v7


MICRO_TRIGGER_P95_MM = 0.40
MICRO_MAX_WHOLE_CROWN_MOVE_MM = 0.28
MICRO_NONWORSE_TOL_MM = 0.01
MICRO_MIN_IMPROVEMENT_MM = 0.003
MICRO_MAX_PASSES = 2
MICRO_TRANSITION_RINGS = 6


def _p95_from_state(state):
    value = ((state or {}).get("target_error") or {}).get("p95_mm")
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _p95(report, side):
    state = ((report or {}).get("physical_after") or {}).get(side) or {}
    return _p95_from_state(state)


def _unresolved(report, surfaces):
    per_side = (report or {}).get("per_side") or {}
    return [side for side in surfaces if (per_side.get(side) or {}).get("status") != "PASS_ENGINEERING"]


def _measure_side_p95(mesh, domain, surface, target, margin):
    if domain is None or not np.any(np.asarray(domain, bool)):
        return None, None
    crown_center = np.median(np.asarray(margin, float), axis=0)
    adjacency = v7.v6._neighbors(mesh)
    _shared, physical = v7._side_measure(
        mesh, domain, surface, float(target), crown_center, adjacency,
    )
    return _p95_from_state(physical), physical


def _protect_other_contacts(mesh, domains, surfaces, target, margin, target_side, base_protected):
    n = len(mesh.vertices)
    protected = np.zeros(n, bool) if base_protected is None else np.asarray(base_protected, bool).copy()
    if protected.shape != (n,):
        raise ValueError("保护区域与冠形顶点数不匹配")
    crown_center = np.median(np.asarray(margin, float), axis=0)
    adjacency = v7.v6._neighbors(mesh)
    details = {}
    for side, surface in surfaces.items():
        if side == target_side or side not in domains:
            continue
        mask, seeds = v7._contact_protection(
            mesh, domains[side], surface, float(target), crown_center, adjacency,
        )
        protected |= mask
        details[side] = {
            "seed_count": int(seeds),
            "protected_vertex_count": int(mask.sum()),
        }
    return protected, details


def fit_proximal_envelopes_v8(mesh, frame, margin, surfaces, target, *, protected=None,
                              transition_rings=10, max_passes=None):
    kwargs = dict(protected=protected, transition_rings=transition_rings)
    if max_passes is not None:
        kwargs["max_passes"] = max_passes
    current, report, domains = v7.fit_proximal_envelopes_v7(
        mesh, frame, margin, surfaces, target, **kwargs,
    )
    report["solver"] = "SPACE_DRIVEN_PROXIMAL_ONE_SIDE_MICRO_CLOSURE_V8"
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

    # Only the worse residual side receives the final micro-adjustment. This is
    # deliberately geometry-driven rather than hard-coded mesial/distal: the
    # screen's visual "right side" depends on jaw/camera orientation.
    target_side = max(triggerable, key=lambda side: before_errors[side])
    micro_protected, protection_detail = _protect_other_contacts(
        current, domains, surfaces, float(target), margin, target_side, protected,
    )

    reference = np.asarray(current.vertices, float).copy()
    trial, trial_report, trial_domains = v7.fit_proximal_envelopes_v7(
        current, frame, margin, {target_side: surfaces[target_side]}, target,
        protected=micro_protected,
        transition_rings=MICRO_TRANSITION_RINGS,
        max_passes=MICRO_MAX_PASSES,
    )
    displacement = np.linalg.norm(np.asarray(trial.vertices, float) - reference, axis=1)
    max_move = float(displacement.max(initial=0.0))

    after_errors = {}
    after_states = {}
    for side, surface in surfaces.items():
        domain = trial_domains.get(side, domains.get(side))
        value, state = _measure_side_p95(trial, domain, surface, float(target), margin)
        after_errors[side] = value
        after_states[side] = state

    comparable = [
        side for side in surfaces
        if before_errors.get(side) is not None and after_errors.get(side) is not None
    ]
    nonworse = all(
        after_errors[side] <= before_errors[side] + MICRO_NONWORSE_TOL_MM
        for side in comparable if side != target_side
    )
    target_improved = (
        before_errors.get(target_side) is not None
        and after_errors.get(target_side) is not None
        and after_errors[target_side] <= before_errors[target_side] - MICRO_MIN_IMPROVEMENT_MM
    )
    safe_move = max_move <= MICRO_MAX_WHOLE_CROWN_MOVE_MM + 1e-9
    accepted = bool(nonworse and target_improved and safe_move)

    micro = {
        "status": "ACCEPTED" if accepted else "REJECTED_KEEP_V7",
        "target_side": target_side,
        "selection_policy": "WORSE_UNRESOLVED_PHYSICAL_SIDE_ONLY",
        "protected_other_contacts": protection_detail,
        "before_p95_mm": before_errors,
        "after_p95_mm": after_errors,
        "max_whole_crown_move_mm": max_move,
        "move_budget_mm": MICRO_MAX_WHOLE_CROWN_MOVE_MM,
        "transition_rings": MICRO_TRANSITION_RINGS,
        "non_target_sides_nonworse": nonworse,
        "target_improved": target_improved,
        "safe_move": safe_move,
    }
    if not accepted:
        report["micro_closure"] = micro
        return current, report, domains

    # Preserve the original two-side report and replace only the side that this
    # pass owns. The other side's status/measurements remain authoritative.
    target_per_side = (trial_report.get("per_side") or {}).get(target_side) or {}
    report.setdefault("per_side", {})[target_side] = target_per_side
    report.setdefault("physical_after", {})[target_side] = after_states.get(target_side)
    report.setdefault("after", {})[target_side] = (trial_report.get("after") or {}).get(target_side)
    report.setdefault("passes", []).extend(
        {"micro": True, **p} for p in trial_report.get("passes", [])
    )
    report.setdefault("trials", []).extend(
        {"micro": True, **t} for t in trial_report.get("trials", [])
    )
    report["micro_closure"] = micro
    report["solver"] = "SPACE_DRIVEN_PROXIMAL_ONE_SIDE_MICRO_CLOSURE_V8"

    merged_domains = dict(domains)
    merged_domains.update(trial_domains)
    unresolved_after = _unresolved(report, surfaces)
    if not unresolved_after:
        report.update(status="PASS_ENGINEERING", reason=None)
    elif len(unresolved_after) < len(unresolved):
        report.update(
            status="REVIEW_REQUIRED",
            reason="PARTIAL_PROXIMAL_SIDES_SOLVED",
            sides_not_reached=unresolved_after,
        )
    return trial, report, merged_domains
