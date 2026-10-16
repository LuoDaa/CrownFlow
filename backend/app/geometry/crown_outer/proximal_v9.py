"""Final guarded one-side proximal micro adjustment for CrownFlow 1.0.

V8 already solves the larger mesial/distal problem independently. V9 is for the
remaining visible sliver: select the physical side with the larger remaining
contact error (never screen/camera "right"), freeze the opposite contact, Margin
and occlusal owners, and give that side a small bounded final contact attempt.

The trial is accepted only when it improves the original clinical target (or
adds useful contact vertices), does not worsen the opposite side, stays inside a
small whole-crown move budget, and does not create excessive penetration.
"""
from __future__ import annotations

import numpy as np

from . import proximal_v8 as v8
from . import proximal_v7 as v7


# Real-case tuning: the previous 0.32 mm trigger was too conservative for a
# clearly visible last-side gap.  This remains a local contact refinement, not a
# whole-crown M/D scale operation.
FINAL_KISS_TRIGGER_P95_MM = 0.60
FINAL_KISS_MIN_P95_MM = 0.010
FINAL_KISS_TARGET_BIAS_MM = 0.006
FINAL_KISS_MAX_MOVE_MM = 0.30
FINAL_KISS_NON_TARGET_TOL_MM = 0.008
FINAL_KISS_MIN_IMPROVEMENT_MM = 0.002
FINAL_KISS_MIN_CONTACT_GAIN = 2
FINAL_KISS_MAX_EXTRA_PENETRATION_MM = 0.07
FINAL_KISS_TRANSITION_RINGS = 5
FINAL_KISS_MAX_PASSES = 2


def _p95(state):
    value = (((state or {}).get("target_error") or {}).get("p95_mm"))
    return float(value) if value is not None and np.isfinite(value) else None


def _contact_count(state):
    value = (state or {}).get("contact_vertex_count")
    return int(value) if value is not None else 0


def _state_for(report, side):
    return ((report or {}).get("physical_after") or {}).get(side) or {}


def fit_proximal_envelopes_v9(mesh, frame, margin, surfaces, target, *, protected=None,
                              transition_rings=10, max_passes=None):
    """Run V8 then one bounded geometry-driven final closure on the worse side."""
    kwargs = dict(protected=protected, transition_rings=transition_rings)
    if max_passes is not None:
        kwargs["max_passes"] = max_passes
    current, report, domains = v8.fit_proximal_envelopes_v8(
        mesh, frame, margin, surfaces, target, **kwargs,
    )
    report["solver"] = "SPACE_DRIVEN_PROXIMAL_FINAL_KISS_V9_1"
    report["final_kiss"] = {"status": "NOT_NEEDED"}

    if not surfaces:
        return current, report, domains

    before_states = {side: _state_for(report, side) for side in surfaces}
    before_p95 = {side: _p95(state) for side, state in before_states.items()}
    candidates = [
        side for side, value in before_p95.items()
        if value is not None
        and FINAL_KISS_MIN_P95_MM <= value <= FINAL_KISS_TRIGGER_P95_MM
        and side in domains
        and np.any(np.asarray(domains[side], bool))
    ]
    if not candidates:
        report["final_kiss"] = {
            "status": "SKIPPED",
            "reason": "NO_BOUNDED_RESIDUAL_SIDE",
            "residual_p95_mm": before_p95,
            "trigger_p95_mm": FINAL_KISS_TRIGGER_P95_MM,
        }
        return current, report, domains

    # "Right" in the viewer is camera-dependent. Always refine the physical
    # side that objectively has the larger residual to the real neighbour.
    target_side = max(candidates, key=lambda side: before_p95[side])
    micro_protected, protection_detail = v8._protect_other_contacts(
        current, domains, surfaces, float(target), margin, target_side, protected,
    )

    reference = np.asarray(current.vertices, float).copy()
    biased_target = float(target) - FINAL_KISS_TARGET_BIAS_MM
    trial, trial_report, trial_domains = v7.fit_proximal_envelopes_v7(
        current,
        frame,
        margin,
        {target_side: surfaces[target_side]},
        biased_target,
        protected=micro_protected,
        transition_rings=FINAL_KISS_TRANSITION_RINGS,
        max_passes=FINAL_KISS_MAX_PASSES,
    )
    displacement = np.linalg.norm(np.asarray(trial.vertices, float) - reference, axis=1)
    max_move = float(displacement.max(initial=0.0))

    after_states = {}
    after_p95 = {}
    for side, surface in surfaces.items():
        domain = trial_domains.get(side, domains.get(side))
        value, state = v8._measure_side_p95(
            trial, domain, surface, float(target), margin,
        )
        after_p95[side] = value
        after_states[side] = state or {}

    comparable_other = [
        side for side in surfaces
        if side != target_side
        and before_p95.get(side) is not None
        and after_p95.get(side) is not None
    ]
    non_target_nonworse = all(
        after_p95[side] <= before_p95[side] + FINAL_KISS_NON_TARGET_TOL_MM
        for side in comparable_other
    )
    p95_improved = (
        after_p95.get(target_side) is not None
        and after_p95[target_side] <= before_p95[target_side] - FINAL_KISS_MIN_IMPROVEMENT_MM
    )
    contact_gain = (
        _contact_count(after_states.get(target_side))
        >= _contact_count(before_states.get(target_side)) + FINAL_KISS_MIN_CONTACT_GAIN
    )
    signed_min = (after_states.get(target_side) or {}).get("signed_min_mm")
    penetration_safe = (
        signed_min is None
        or float(signed_min) >= float(target) - FINAL_KISS_MAX_EXTRA_PENETRATION_MM
    )
    safe_move = max_move <= FINAL_KISS_MAX_MOVE_MM + 1e-9
    accepted = bool(
        non_target_nonworse
        and (p95_improved or contact_gain)
        and penetration_safe
        and safe_move
    )

    report["final_kiss"] = {
        "status": "ACCEPTED" if accepted else "REJECTED_KEEP_V8",
        "target_side": target_side,
        "selection_policy": "LARGER_REAL_NEIGHBOR_RESIDUAL_NOT_CAMERA_RIGHT",
        "clinical_target_mm": float(target),
        "trial_target_mm": biased_target,
        "target_bias_mm": FINAL_KISS_TARGET_BIAS_MM,
        "before_p95_mm": before_p95,
        "after_p95_mm": after_p95,
        "before_contact_vertices": {side: _contact_count(state) for side, state in before_states.items()},
        "after_contact_vertices": {side: _contact_count(state) for side, state in after_states.items()},
        "protected_other_contacts": protection_detail,
        "max_whole_crown_move_mm": max_move,
        "move_budget_mm": FINAL_KISS_MAX_MOVE_MM,
        "passes_budget": FINAL_KISS_MAX_PASSES,
        "non_target_sides_nonworse": non_target_nonworse,
        "target_p95_improved": p95_improved,
        "target_contact_area_grew": contact_gain,
        "penetration_safe": penetration_safe,
        "signed_min_mm": signed_min,
        "transition_rings": FINAL_KISS_TRANSITION_RINGS,
    }
    if not accepted:
        return current, report, domains

    report.setdefault("physical_after", {})[target_side] = after_states[target_side]
    report.setdefault("passes", []).extend(
        {"final_kiss": True, **p} for p in trial_report.get("passes", [])
    )
    report.setdefault("trials", []).extend(
        {"final_kiss": True, **t} for t in trial_report.get("trials", [])
    )
    target_reached = bool((after_states.get(target_side) or {}).get("target_reached"))
    if target_reached:
        existing = dict((report.get("per_side") or {}).get(target_side) or {})
        existing.update(status="PASS_ENGINEERING", reason=None, physical_after=after_states[target_side])
        report.setdefault("per_side", {})[target_side] = existing

    merged_domains = dict(domains)
    merged_domains.update(trial_domains)
    report["solver"] = "SPACE_DRIVEN_PROXIMAL_FINAL_KISS_V9_1"
    return trial, report, merged_domains
