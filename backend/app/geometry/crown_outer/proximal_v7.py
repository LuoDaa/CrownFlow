"""Independent mesial/distal proximal solver for CrownFlow 1.0.

The previous V6 solver built a good physical neighbor domain, but solved both
proximal sides in one Laplacian system.  That made one already-good contact part
of the free transition band while the opposite side was still being corrected.
For 1.0 we make the ownership rule explicit:

* mesial and distal are solved independently;
* a solved/partially-established contact becomes a hard protected patch before
  the opposite side is touched;
* broad reach can run a second bounded round when a real patient gap is larger
  than the original 1.8 mm engineering budget;
* Margin/cervical and occlusal Z are never owned by this solver;
* one invalid/missing side never prevents the other valid side from improving.

This module intentionally reuses V6's patient-frame surface queries, domain
construction and deformation gates so there is still one geometric truth source.
"""
from __future__ import annotations

import numpy as np

from .deform_gate import deformation_gate
from .laplacian import solve_local_band
from .zones import ZONE_MODEL_VERSION, build_crown_zones
from . import space_contact as v6


CONTACT_PROTECT_TOL_MM = 0.15
CONTACT_PROTECT_RINGS = 2
BROAD_REACH_ROUNDS = 2


def _expand_mask(mask, adjacency, rings):
    out = np.asarray(mask, bool).copy()
    front = set(map(int, np.flatnonzero(out)))
    seen = set(front)
    for _ in range(max(0, int(rings))):
        nxt = {j for i in front for j in adjacency[i]} - seen
        if not nxt:
            break
        out[np.fromiter(nxt, dtype=int)] = True
        seen |= nxt
        front = nxt
    return out


def _contact_protection(mesh, domain, surface, target, crown_center, adjacency):
    """Freeze the actual contact island, not the whole proximal wall."""
    ids = np.flatnonzero(np.asarray(domain, bool))
    q = v6._physical_contact_query(mesh.vertices, ids, surface, target, crown_center)
    mask = np.zeros(len(mesh.vertices), bool)
    if q is None or not np.any(q["finite"]):
        return mask, 0
    finite_ids = ids[q["finite"]]
    signed = q["signed"][q["finite"]]
    near = np.abs(signed - float(target)) <= CONTACT_PROTECT_TOL_MM
    if not np.any(near):
        return mask, 0
    mask[finite_ids[near]] = True
    seed_count = int(mask.sum())
    return _expand_mask(mask, adjacency, CONTACT_PROTECT_RINGS), seed_count


def _side_measure(mesh, domain, surface, target, crown_center, adjacency):
    shared = v6.measure(
        mesh.vertices, domain, surface, target, kind="PROXIMAL"
    )[0]
    physical = v6._contact_state(
        mesh, domain, surface, target, crown_center, adjacency
    )
    return shared, physical


def _solve_one_side(current, frame, margin, side, surface, target, *,
                    base_protected, transition_rings, max_passes):
    """Solve one physical proximal side while every prior owner stays frozen."""
    zones = build_crown_zones(current, frame, margin)
    zones.validate(len(current.vertices))
    user_protected = np.asarray(base_protected, bool).copy()
    base_locked = np.asarray(zones.cervical | zones.occlusal | user_protected, bool)

    reach_rounds = []
    # V6 broad reach is already bounded by the actual patient neighbor face.  A
    # second round only extends the engineering displacement budget; it cannot
    # grow past that patient-owned bound.
    for round_no in range(1, BROAD_REACH_ROUNDS + 1):
        before = np.asarray(current.vertices, float).copy()
        current, reach = v6._large_scale_side_reach(
            current, frame, margin, {side: surface}, float(target), base_locked,
        )
        moved = np.linalg.norm(np.asarray(current.vertices, float) - before, axis=1)
        detail = (reach or {}).get(side, {})
        reach_rounds.append({"round": round_no, **detail})
        if float(moved.max(initial=0.0)) <= 1e-6:
            break

    domains, locked, domain_quality = v6.build_proximal_domains(
        current, frame, margin, {side: surface}, protected=user_protected,
    )
    domain = domains.get(side, np.zeros(len(current.vertices), bool))
    quality = domain_quality.get(side) or {}
    crown_center = np.median(np.asarray(margin, float), axis=0)
    adjacency = v6._neighbors(current)

    shared_before, physical_before = _side_measure(
        current, domain, surface, float(target), crown_center, adjacency,
    )
    result = {
        "side": side,
        "status": "REVIEW_REQUIRED",
        "reason": None,
        "domain_quality": quality,
        "broad_reach": reach_rounds,
        "before": shared_before,
        "physical_before": physical_before,
        "after": shared_before,
        "physical_after": physical_before,
        "passes": [],
        "trials": [],
    }

    if quality.get("status") != "READY" or int(np.sum(domain)) < v6.MIN_DOMAIN_VERTICES:
        result.update(
            status="NOT_ASSESSED",
            reason=quality.get("reason") or "INSUFFICIENT_PROXIMAL_DOMAIN",
        )
        return current, result, domain, np.zeros(len(current.vertices), bool)

    faces = np.asarray(current.faces, int).copy()
    frame_z = np.asarray(frame.z, float)
    frame_z /= max(float(np.linalg.norm(frame_z)), 1e-12)
    ever_changed = False
    unreachable = False

    for pass_no in range(1, max(1, int(max_passes)) + 1):
        reference = np.asarray(current.vertices, float).copy()
        side_ids, points, no_reachable = v6._desired_contact(
            reference, domain, surface, float(target), crown_center, adjacency,
        )
        unreachable |= bool(no_reachable)

        if not len(side_ids):
            state = v6._contact_state(
                current, domain, surface, float(target), crown_center, adjacency,
            )
            result["passes"].append({
                "pass": pass_no,
                "seed_count": 0,
                "status": "NO_CORRECTION_NEEDED" if state.get("target_reached") else "TARGET_NOT_REACHED",
                "physical_contact": state,
            })
            break

        desired = reference.copy()
        delta = np.asarray(points, float) - reference[side_ids]
        # Proximal owns X/Y only.  It must not flatten cusps or undo occlusal
        # height that has already been established.
        delta -= (delta @ frame_z)[:, None] * frame_z[None, :]
        desired[side_ids] = reference[side_ids] + delta

        seed_ids = np.unique(np.asarray(side_ids, int))
        free_ids = v6._free_band(adjacency, seed_ids, locked, transition_rings)
        fixed = np.setdiff1d(np.arange(len(reference)), free_ids)
        active_before = v6._active_subset_error(
            reference, seed_ids, surface, float(target), crown_center,
        )
        accepted = None

        for alpha in v6.ALPHAS:
            xt = reference.copy()
            step = float(alpha) * (desired[seed_ids] - reference[seed_ids])
            norm = np.linalg.norm(step, axis=1)
            large = norm > v6.PER_PASS_MAX_MOVE_MM
            if np.any(large):
                step[large] *= (v6.PER_PASS_MAX_MOVE_MM / norm[large])[:, None]
            xt[seed_ids] = reference[seed_ids] + step

            vertices = (
                solve_local_band(reference, faces, free_ids, fixed, xt)[0]
                if len(free_ids) else xt
            )
            # Exact axial invariance for every changed vertex.
            axial = (vertices - reference) @ frame_z
            vertices -= axial[:, None] * frame_z[None, :]
            moved = np.linalg.norm(vertices - reference, axis=1)

            candidate = current.copy()
            candidate.vertices = vertices
            gate = deformation_gate(
                reference,
                vertices,
                faces,
                scale_mm=.1,
                locked_ids=np.flatnonzero(locked),
                focus_faces=np.flatnonzero(np.any(moved[faces] > 1e-9, axis=1)),
            )
            flips = int((np.einsum(
                "ij,ij->i", current.face_normals, candidate.face_normals
            ) <= 0).sum())
            safe = (
                gate.ok
                and flips == 0
                and not (candidate.area_faces < 1e-10).any()
                and float(moved.max(initial=0.0)) <= v6.PER_PASS_MAX_MOVE_MM + 1e-6
                and float(moved[locked].max(initial=0.0)) <= 1e-5
            )
            active_after = v6._active_subset_error(
                vertices, seed_ids, surface, float(target), crown_center,
            )
            better = v6._subset_better(
                {side: active_before}, {side: active_after}
            )
            result["trials"].append({
                "pass": pass_no,
                "alpha": float(alpha),
                "seed_count": int(len(seed_ids)),
                "free_count": int(len(free_ids)),
                "gate": gate.as_dict(),
                "all_flips": flips,
                "maximum_displacement_mm": float(moved.max(initial=0.0)),
                "improved": bool(better),
                "accepted": bool(safe and better),
                "acceptance_scope": "INDEPENDENT_SINGLE_SIDE_CONTACT_CORE_XY_ONLY",
                "active_before": active_before,
                "active_after": active_after,
            })
            if safe and better:
                accepted = candidate
                break

        if accepted is None:
            result["passes"].append({
                "pass": pass_no,
                "seed_count": int(len(seed_ids)),
                "status": "CONSTRAINT_CONFLICT",
            })
            break

        current = accepted
        adjacency = v6._neighbors(current)
        ever_changed = True
        result["passes"].append({
            "pass": pass_no,
            "seed_count": int(len(seed_ids)),
            "status": "PASS_ENGINEERING",
        })

    shared_after, physical_after = _side_measure(
        current, domain, surface, float(target), crown_center, adjacency,
    )
    result["after"] = shared_after
    result["physical_after"] = physical_after

    contact_protection, contact_seed_count = _contact_protection(
        current, domain, surface, float(target), crown_center, adjacency,
    )
    result["contact_protection_seed_count"] = int(contact_seed_count)
    result["contact_protection_vertex_count"] = int(contact_protection.sum())
    result["contact_protection_rings"] = int(CONTACT_PROTECT_RINGS)

    if physical_after.get("target_reached"):
        result.update(
            status="PASS_ENGINEERING",
            reason=None if ever_changed else "NO_CORRECTION_NEEDED",
        )
    elif unreachable and not ever_changed:
        result.update(status="CONSTRAINT_CONFLICT", reason="TARGET_OUTSIDE_LOCAL_REACH")
    else:
        result.update(status="CONSTRAINT_CONFLICT", reason="PROXIMAL_TARGET_NOT_REACHED")
    return current, result, domain, contact_protection


def fit_proximal_envelopes_v7(mesh, frame, margin, surfaces, target, *, protected=None,
                              transition_rings=10, max_passes=v6.MAX_PASSES):
    """Solve mesial/distal independently and freeze every achieved contact.

    A missing or invalid side is reported per-side and never rolls back the other
    side.  This is deliberately fail-open for 1.0 geometry preview while keeping
    all diagnostics visible to the caller.
    """
    current = mesh.copy()
    n = len(current.vertices)
    original_protected = np.zeros(n, bool)
    if protected is not None:
        original_protected = np.asarray(protected, bool).copy()
        if original_protected.shape != (n,):
            raise ValueError("保护区域与冠形顶点数不匹配")

    zones = build_crown_zones(current, frame, margin)
    base_locked = np.asarray(zones.cervical | zones.occlusal | original_protected, bool)
    solved_protection = np.zeros(n, bool)
    domains = {}
    side_results = {}
    report = {
        "stage": "PROXIMAL",
        "status": "REVIEW_REQUIRED",
        "reason": None,
        "before": {},
        "physical_before": {},
        "after": {},
        "physical_after": {},
        "large_scale_reach": {},
        "passes": [],
        "trials": [],
        "per_side": {},
        "side_order": [],
        "zone_model": ZONE_MODEL_VERSION,
        "locked_vertex_count": int(base_locked.sum()),
        "transition_rings": int(transition_rings),
        "solver": "SPACE_DRIVEN_PROXIMAL_INDEPENDENT_SIDES_V7",
        "target_mm": float(target),
        "axial_policy": "PROXIMAL_SOLVER_PRESERVES_PATIENT_FRAME_Z",
        "ownership_policy": "SOLVED_SIDE_CONTACT_BECOMES_HARD_PROTECTED_BEFORE_NEXT_SIDE",
    }

    # Deterministic clinical order, while still accepting a single available side.
    ordered = [s for s in ("mesial", "distal") if s in surfaces]
    ordered += [s for s in surfaces if s not in ordered]

    for side in ordered:
        report["side_order"].append(side)
        side_protected = original_protected | solved_protection
        current, side_report, side_domain, new_protection = _solve_one_side(
            current,
            frame,
            margin,
            side,
            surfaces[side],
            float(target),
            base_protected=side_protected,
            transition_rings=transition_rings,
            max_passes=max_passes,
        )
        domains[side] = side_domain
        side_results[side] = side_report
        report["before"][side] = side_report.get("before")
        report["physical_before"][side] = side_report.get("physical_before")
        report["after"][side] = side_report.get("after")
        report["physical_after"][side] = side_report.get("physical_after")
        report["large_scale_reach"][side] = side_report.get("broad_reach")
        report["passes"].extend({"side": side, **p} for p in side_report.get("passes", []))
        report["trials"].extend({"side": side, **t} for t in side_report.get("trials", []))
        report["per_side"][side] = {
            "status": side_report.get("status"),
            "reason": side_report.get("reason"),
            "domain_quality": side_report.get("domain_quality"),
            "physical_after": side_report.get("physical_after"),
            "contact_protection_vertex_count": side_report.get("contact_protection_vertex_count", 0),
        }
        # Even a partial near-target island is worth preserving: the opposite side
        # has no authority to undo progress already made here.
        solved_protection |= new_protection

    if not ordered:
        report.update(status="NOT_ASSESSED", reason="NO_CONFIRMED_NEIGHBOR_SURFACES")
        return current, report, domains

    passed = [s for s, r in side_results.items() if r.get("status") == "PASS_ENGINEERING"]
    unresolved = [s for s, r in side_results.items() if r.get("status") != "PASS_ENGINEERING"]
    report["protected_contact_vertex_count"] = int(solved_protection.sum())
    if not unresolved:
        report.update(status="PASS_ENGINEERING", reason=None)
    elif passed:
        report.update(
            status="REVIEW_REQUIRED",
            reason="PARTIAL_PROXIMAL_SIDES_SOLVED",
            sides_solved=passed,
            sides_not_reached=unresolved,
        )
    else:
        report.update(
            status="CONSTRAINT_CONFLICT",
            reason="PROXIMAL_TARGET_NOT_REACHED",
            sides_not_reached=unresolved,
        )
    return current, report, domains
