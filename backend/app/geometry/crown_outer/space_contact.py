"""Patient-space driven proximal fitting for CAD-1.

The library crown is an embryo.  Adjacent teeth define the actual mesial/distal
space.  This module therefore does two different jobs, in this order:

1. **large-scale side reach** -- grow each physical side of the embryo from the
   fixed Margin root until the side is near the corresponding adjacent surface;
2. **local contact shaping** -- create a compact continuous contact core at the
   requested signed target (zero or a small negative value are both supported),
   while Laplacian transition preserves the remaining anatomy.

No clinical label is trusted to mean +X/-X.  The physical direction is inferred
from the confirmed neighbor patch itself in the patient frame.  This matters on
posterior teeth where the dental arch is curved and a single global X direction
is not an adequate contact direction.
"""
from __future__ import annotations

import numpy as np

from ..nonplanar_crown import MarginCurtain
from .deform_gate import deformation_gate
from .laplacian import solve_local_band
from .zones import ZONE_MODEL_VERSION, build_crown_zones
from .auto_fit_adapter import measure

TARGET_TOL_MM = 0.03
REFERENCE_BAND_MM = 4.0
CONTACT_FRACTION = 0.45
MAX_REACH_MM = 8.0
MAX_PASSES = 6
PER_PASS_MAX_MOVE_MM = 1.35
PREFIT_MAX_SIDE_GROWTH_MM = 4.5
ALPHAS = (1.0, 0.65, 0.4, 0.25, 0.125)
MIN_DOMAIN_VERTICES = 48
MIN_VALID_VERTICES = 24
MIN_DOMAIN_FRACTION = 0.07
MIN_CONTACT_VERTICES = 18
MIN_CONTACT_FRACTION = 0.045
BODY_TRANSITION_MM = 1.45


def _frame_basis(frame):
    origin = np.asarray(frame.origin, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 无效")
    if not np.isfinite(origin).all() or not np.isfinite(matrix).all():
        raise ValueError("patient frame 含 NaN/Inf")
    return origin, matrix


def _to_local(points, frame):
    origin, matrix = _frame_basis(frame)
    return (np.asarray(points, dtype=float) - origin) @ matrix


def _to_world_vector(local_vectors, frame):
    _origin, matrix = _frame_basis(frame)
    return np.asarray(local_vectors, dtype=float) @ matrix.T


def _smoothstep01(value):
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _neighbors(mesh):
    adj = [set() for _ in mesh.vertices]
    for a, b in np.asarray(mesh.edges_unique, int):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _free_band(adj, seeds, locked, rings=9):
    locked_ids = set(map(int, np.flatnonzero(locked)))
    seen = set(seeds) | locked_ids
    front = set(seeds)
    free = set()
    for _ in range(max(1, min(int(rings), 14))):
        nxt = {v for u in front for v in adj[u]} - seen
        if not nxt:
            break
        seen |= nxt
        free |= nxt
        front = nxt
    return np.asarray(sorted(free), int)


def _margin_center(margin, frame):
    local = _to_local(margin, frame)
    return np.median(local[:, :2], axis=0), float(np.median(local[:, 2]))


def _physical_reference(surface, frame, center_xy):
    """Return the patient-space direction from Margin centre to a neighbor.

    The nearest radial part of the patch is used, not the patch centroid.  A V4
    halo can be much wider than the true facing wall; using the whole centroid
    biases the direction and was one reason posterior contacts stayed open.
    """
    local = _to_local(surface.mesh.vertices, frame)
    finite = np.isfinite(local).all(axis=1)
    local = local[finite]
    if len(local) < 3:
        return None
    radial = local[:, :2] - np.asarray(center_xy, float)
    radius = np.linalg.norm(radial, axis=1)
    order = np.argsort(radius)
    take = max(12, min(len(order), int(np.ceil(len(order) * 0.16))))
    near = radial[order[:take]]
    direction = np.median(near, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.mean(radial, axis=0)
        norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        return None
    direction /= norm
    projection = radial @ direction
    positive = projection[projection > 0.0]
    if len(positive) < 8:
        positive = projection
    # Crown-facing surface = low projection along the physical side direction.
    face_s = float(np.quantile(positive, 0.06))
    world = _to_world_vector(np.r_[direction, 0.0][None, :], frame)[0]
    world /= max(float(np.linalg.norm(world)), 1e-12)
    return {
        "direction_local_xy": direction,
        "direction_world": world,
        "face_projection_mm": face_s,
        "patch_nearest_radius_mm": float(np.quantile(radius, 0.02)),
        "patch_median_radius_mm": float(np.median(radius)),
    }


def _side_geometry(mesh, frame, margin, ref, locked):
    vertices = np.asarray(mesh.vertices, float)
    local = _to_local(vertices, frame)
    center_xy, _margin_z = _margin_center(margin, frame)
    direction = np.asarray(ref["direction_local_xy"], float)
    projection = (local[:, :2] - center_xy) @ direction
    height = MarginCurtain(margin, frame.z).axial_above(vertices)
    editable = (~locked) & np.isfinite(projection) & np.isfinite(height) & (height > 0.20)
    positive = projection[editable & (projection > 0.0)]
    if not len(positive):
        return projection, height, editable, 0.0
    extent = float(np.quantile(positive, 0.98))
    return projection, height, editable, extent


def _large_scale_side_reach(mesh, frame, margin, surfaces, target, locked):
    """Grow the embryo to the patient MD envelope before local contact fitting."""
    current = mesh.copy()
    center_xy, _margin_z = _margin_center(margin, frame)
    details = {}
    for label, surface in surfaces.items():
        ref = _physical_reference(surface, frame, center_xy)
        if ref is None:
            details[label] = {"status": "NOT_ASSESSED", "reason": "INVALID_NEIGHBOR_REFERENCE"}
            continue
        projection, height, editable, current_extent = _side_geometry(current, frame, margin, ref, locked)
        desired_extent = float(ref["face_projection_mm"]) - float(target)
        requested = desired_extent - current_extent
        if requested <= TARGET_TOL_MM:
            details[label] = {
                "status": "NO_CORRECTION_NEEDED",
                "requested_growth_mm": float(requested),
                "current_extent_mm": current_extent,
                "desired_extent_mm": desired_extent,
                "direction_world": ref["direction_world"].tolist(),
            }
            continue
        growth = float(np.clip(requested, 0.0, PREFIT_MAX_SIDE_GROWTH_MM))
        # Side weight: Margin remains fixed, centre/body anatomy transitions smoothly,
        # the actual proximal flank reaches the patient-defined envelope.
        denom = max(current_extent, 0.5)
        side_weight = _smoothstep01((projection - 0.12 * denom) / max(0.88 * denom, 1e-6))
        height_weight = _smoothstep01(height / BODY_TRANSITION_MM)
        weight = side_weight * height_weight * editable.astype(float)
        displacement = weight[:, None] * growth * np.asarray(ref["direction_world"], float)[None, :]
        vertices = np.asarray(current.vertices, float).copy()
        vertices += displacement
        current.vertices = vertices
        details[label] = {
            "status": "APPLIED",
            "requested_growth_mm": float(requested),
            "applied_growth_mm": growth,
            "current_extent_mm": current_extent,
            "desired_extent_mm": desired_extent,
            "max_vertex_displacement_mm": float(np.linalg.norm(displacement, axis=1).max(initial=0.0)),
            "direction_world": ref["direction_world"].tolist(),
            "reference": {k: v for k, v in ref.items() if k not in ("direction_world", "direction_local_xy")},
        }
    return current, details


def build_proximal_domains(mesh, frame, margin, surfaces, *, protected=None, zones=None):
    """Build broad domains on the *physical* side facing each confirmed neighbor."""
    vertices = np.asarray(mesh.vertices, float)
    if zones is None:
        zones = build_crown_zones(mesh, frame, margin)
    zones.validate(len(vertices))
    # Margin is the hard root.  Occlusal vertices are not globally locked here:
    # posterior proximal contact often reaches the marginal ridge; movement remains
    # lateral and local so cusp/fossa anatomy is still preserved by the transition.
    locked = np.asarray(zones.cervical, bool).copy()
    if protected is not None:
        p = np.asarray(protected, bool)
        if p.shape != (len(vertices),):
            raise ValueError("保护区域与冠形顶点数不匹配")
        locked |= p

    center_xy, _margin_z = _margin_center(margin, frame)
    masks = {}
    diagnostics = {}
    refs = {}

    for label, surface in surfaces.items():
        ref = _physical_reference(surface, frame, center_xy)
        mask = np.zeros(len(vertices), bool)
        if ref is None:
            diagnostics[label] = {
                "status": "NOT_ASSESSED", "reason": "INVALID_NEIGHBOR_REFERENCE",
                "domain_vertex_count": 0, "valid_vertex_count": 0,
            }
            masks[label] = mask
            continue
        refs[label] = ref
        projection, height, editable, extent = _side_geometry(mesh, frame, margin, ref, locked)
        side_candidate = editable & (projection > max(0.15, 0.22 * max(extent, 0.5)))
        ids = np.flatnonzero(side_candidate)
        diag = {
            "clinical_label": label,
            "selection": "PATIENT_VECTOR_SIDE_PLUS_UNSIGNED_REFERENCE_BAND",
            "direction_world": ref["direction_world"].tolist(),
            "physical_extent_mm": extent,
            "candidate_vertex_count": int(len(ids)),
            "domain_vertex_count": 0,
            "valid_vertex_count": 0,
            "signed_valid_vertex_count": 0,
            "minimum_domain_vertices": int(MIN_DOMAIN_VERTICES),
            "minimum_valid_vertices": int(MIN_VALID_VERTICES),
        }
        if not len(ids):
            diagnostics[label] = {**diag, "status": "NOT_ASSESSED", "reason": "EMPTY_PHYSICAL_SIDE"}
            masks[label] = mask
            continue
        cp, _fids, _signed, distance, sign_valid = surface.query(vertices[ids])
        distance = np.asarray(distance, float)
        cp = np.asarray(cp, float)
        finite = np.isfinite(distance) & np.isfinite(cp).all(axis=1)
        sign_valid = np.asarray(sign_valid, bool) & finite
        diag["valid_vertex_count"] = int(finite.sum())
        diag["signed_valid_vertex_count"] = int(sign_valid.sum())
        if int(finite.sum()) < min(MIN_VALID_VERTICES, len(ids)):
            diagnostics[label] = {**diag, "status": "NOT_ASSESSED", "reason": "NO_VALID_SURFACE_SAMPLES"}
            masks[label] = mask
            continue
        valid_dist = distance[finite]
        minimum = float(np.min(valid_dist))
        keep = finite & (distance <= minimum + REFERENCE_BAND_MM)
        required = max(MIN_DOMAIN_VERTICES, int(np.ceil(len(ids) * MIN_DOMAIN_FRACTION)))
        required = min(required, int(finite.sum()))
        if int(keep.sum()) < required:
            order = np.sort(valid_dist)
            threshold = float(order[min(required - 1, len(order) - 1)])
            keep = finite & (distance <= threshold + 1e-9)
        chosen = ids[keep]
        mask[chosen] = True
        diag.update({
            "domain_vertex_count": int(len(chosen)),
            "minimum_distance_mm": minimum,
            "reference_band_mm": float(REFERENCE_BAND_MM),
            "status": "READY" if len(chosen) >= min(MIN_DOMAIN_VERTICES, int(finite.sum())) else "NOT_ASSESSED",
        })
        if diag["status"] != "READY":
            diag["reason"] = "INSUFFICIENT_PROXIMAL_DOMAIN"
        diagnostics[label] = diag
        masks[label] = mask

    # Two references may be clinically named differently but still land on the
    # same physical side.  Detect using their actual patient-space directions.
    labels = list(refs)
    if len(labels) > 1:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                da = np.asarray(refs[a]["direction_local_xy"], float)
                db = np.asarray(refs[b]["direction_local_xy"], float)
                if float(da @ db) > 0.55:
                    for label in (a, b):
                        diagnostics[label]["status"] = "NOT_ASSESSED"
                        diagnostics[label]["reason"] = "NEIGHBOR_PHYSICAL_SIDE_CONFLICT"
                        masks[label][:] = False
    return masks, locked, diagnostics


def _normals_toward_crown(surface, fids, cp, crown_center):
    fids = np.asarray(fids, int)
    try:
        normals = np.asarray(surface.source.oriented_normals(fids), float)
    except Exception:
        normals = np.asarray(surface.mesh.face_normals[fids], float)
    norm = np.linalg.norm(normals, axis=1)
    good = norm > 1e-10
    normals[good] /= norm[good, None]
    normals[~good] = np.array([1.0, 0.0, 0.0])
    toward = np.asarray(crown_center, float)[None, :] - np.asarray(cp, float)
    flip = np.einsum("ij,ij->i", normals, toward) < 0.0
    normals[flip] *= -1.0
    return normals


def _physical_contact_query(vertices, ids, surface, target, crown_center):
    ids = np.asarray(ids, int)
    if not len(ids):
        return None
    cp, fids, _signed, distance, _sign_valid = surface.query(np.asarray(vertices)[ids])
    cp = np.asarray(cp, float)
    fids = np.asarray(fids, int)
    distance = np.asarray(distance, float)
    finite = np.isfinite(distance) & np.isfinite(cp).all(axis=1)
    normals = _normals_toward_crown(surface, fids, cp, crown_center)
    signed_local = np.einsum("ij,ij->i", np.asarray(vertices)[ids] - cp, normals)
    finite &= np.isfinite(signed_local)
    target_points = cp + normals * float(target)
    error = np.linalg.norm(np.asarray(vertices)[ids] - target_points, axis=1)
    return {
        "ids": ids,
        "cp": cp,
        "fids": fids,
        "distance": distance,
        "finite": finite,
        "normals": normals,
        "signed": signed_local,
        "target_points": target_points,
        "error": error,
    }


def _required_contact_count(domain_count):
    return max(MIN_CONTACT_VERTICES, min(80, int(np.ceil(domain_count * MIN_CONTACT_FRACTION))))


def _contact_state(vertices, mask, surface, target, crown_center):
    ids = np.flatnonzero(mask)
    q = _physical_contact_query(vertices, ids, surface, target, crown_center)
    if q is None or not np.any(q["finite"]):
        return {"status": "NOT_ASSESSED", "valid_count": 0, "contact_vertex_count": 0}
    finite = q["finite"]
    values = q["signed"][finite]
    errors = np.abs(values - float(target))
    required = _required_contact_count(int(len(ids)))
    on = errors <= TARGET_TOL_MM
    return {
        "status": "MEASURED_ENGINEERING",
        "valid_count": int(finite.sum()),
        "domain_vertex_count": int(len(ids)),
        "required_contact_vertices": int(required),
        "contact_vertex_count": int(on.sum()),
        "contact_fraction": float(on.mean()) if len(on) else 0.0,
        "signed_min_mm": float(np.min(values)),
        "signed_median_mm": float(np.median(values)),
        "target_error": {
            "median_mm": float(np.median(errors)),
            "p95_mm": float(np.quantile(errors, 0.95)),
            "max_mm": float(np.max(errors)),
        },
        "target_reached": bool(int(on.sum()) >= required),
        "sign_method": "LOCAL_PATCH_NORMAL_REORIENTED_TOWARD_CROWN",
    }


def _active_subset_error(vertices, ids, surface, target, crown_center):
    q = _physical_contact_query(vertices, np.unique(np.asarray(ids, int)), surface, target, crown_center)
    if q is None or not np.any(q["finite"]):
        return {"status": "NOT_ASSESSED", "valid_count": 0}
    err = q["error"][q["finite"]]
    return {
        "status": "MEASURED_ENGINEERING",
        "valid_count": int(len(err)),
        "target_error": {
            "median_mm": float(np.median(err)),
            "p95_mm": float(np.quantile(err, .95)),
            "max_mm": float(np.max(err)),
        },
    }


def _subset_better(before, after):
    compared = False
    improved = False
    for side, old in before.items():
        new = after.get(side)
        if not old.get("valid_count") or not isinstance(new, dict) or not new.get("valid_count"):
            continue
        a = old.get("target_error") or {}
        b = new.get("target_error") or {}
        vals = (a.get("p95_mm"), b.get("p95_mm"), a.get("median_mm"), b.get("median_mm"))
        if not all(v is not None and np.isfinite(v) for v in vals):
            continue
        compared = True
        if b["p95_mm"] < a["p95_mm"] - 1e-4:
            improved = True
            continue
        if b["median_mm"] < a["median_mm"] - 1e-4 and b["p95_mm"] <= a["p95_mm"] + 1e-4:
            improved = True
            continue
        return False
    return compared and improved


def _desired_contact(reference, mask, surface, target, crown_center):
    ids = np.flatnonzero(mask)
    q = _physical_contact_query(reference, ids, surface, target, crown_center)
    if q is None:
        return np.zeros(0, int), np.zeros((0, 3), float), False
    usable = q["finite"] & (q["distance"] <= MAX_REACH_MM)
    if not np.any(usable):
        return np.zeros(0, int), np.zeros((0, 3), float), bool(np.any(q["finite"]))
    signed = q["signed"]
    target_error = np.abs(signed - float(target))
    required = _required_contact_count(int(len(ids)))
    on_target = usable & (target_error <= TARGET_TOL_MM)
    too_deep = usable & (signed < float(target) - TARGET_TOL_MM)
    if np.any(too_deep):
        active = too_deep
    elif int(on_target.sum()) >= required:
        active = np.zeros_like(usable)
    else:
        candidates = np.flatnonzero(usable & ~on_target)
        if not len(candidates):
            return np.zeros(0, int), np.zeros((0, 3), float), False
        # Reach the closest continuous part first, but never stop because only a
        # handful of vertices happen to touch.
        rank = candidates[np.argsort(q["distance"][candidates])]
        take = max(required - int(on_target.sum()), int(np.ceil(len(rank) * CONTACT_FRACTION)))
        take = min(len(rank), max(1, take))
        active = np.zeros_like(usable)
        active[rank[:take]] = True
    if not np.any(active):
        return np.zeros(0, int), np.zeros((0, 3), float), False
    return ids[active], q["target_points"][active], False


def fit_proximal_envelopes(mesh, frame, margin, surfaces, target, *, protected=None,
                           transition_rings=9, max_passes=MAX_PASSES):
    """Grow both physical sides to the patient space, then establish contact cores."""
    current = mesh.copy()
    faces = np.asarray(current.faces, int).copy()
    zones = build_crown_zones(current, frame, margin)
    locked = np.asarray(zones.cervical, bool).copy()
    if protected is not None:
        p = np.asarray(protected, bool)
        if p.shape != (len(current.vertices),):
            raise ValueError("保护区域与冠形顶点数不匹配")
        locked |= p

    current, reach_report = _large_scale_side_reach(
        current, frame, margin, surfaces, float(target), locked,
    )
    domains, locked, domain_quality = build_proximal_domains(
        current, frame, margin, surfaces, protected=protected,
    )
    crown_center = np.median(np.asarray(margin, float), axis=0)
    before_shared = {
        side: measure(current.vertices, domains[side], surface, target, kind="PROXIMAL", side=side)[0]
        for side, surface in surfaces.items()
    }
    before_physical = {
        side: _contact_state(current.vertices, domains[side], surface, target, crown_center)
        for side, surface in surfaces.items()
    }
    report = {
        "stage": "PROXIMAL",
        "status": "REVIEW_REQUIRED",
        "reason": None,
        "before": before_shared,
        "physical_before": before_physical,
        "after": before_shared,
        "physical_after": before_physical,
        "large_scale_reach": reach_report,
        "passes": [],
        "trials": [],
        "zone_model": ZONE_MODEL_VERSION,
        "domain_quality": domain_quality,
        "locked_vertex_count": int(locked.sum()),
        "transition_rings": int(transition_rings),
        "solver": "SPACE_DRIVEN_PROXIMAL_ENVELOPE_V3_PATIENT_VECTOR",
        "target_mm": float(target),
    }

    for side in surfaces:
        q = domain_quality.get(side) or {}
        if q.get("status") != "READY" or int(domains[side].sum()) < MIN_DOMAIN_VERTICES:
            report.update(status="NOT_ASSESSED", reason=q.get("reason") or "INSUFFICIENT_PROXIMAL_DOMAIN")
            return current, report, domains

    adj = _neighbors(current)
    ever_changed = False
    unreachable = False

    for pass_no in range(1, max(1, int(max_passes)) + 1):
        reference = np.asarray(current.vertices, float).copy()
        desired = reference.copy()
        seeds = set()
        seed_sets = {}
        for side, surface in surfaces.items():
            side_ids, points, no_reachable = _desired_contact(
                reference, domains[side], surface, float(target), crown_center,
            )
            unreachable |= no_reachable
            if len(side_ids):
                desired[side_ids] = points
                seed_sets[side] = side_ids
                seeds.update(map(int, side_ids))

        if not seeds:
            states = {
                side: _contact_state(reference, domains[side], surface, target, crown_center)
                for side, surface in surfaces.items()
            }
            done = all(state.get("target_reached") for state in states.values())
            report["passes"].append({
                "pass": pass_no,
                "seed_count": 0,
                "status": "NO_CORRECTION_NEEDED" if done else "TARGET_NOT_REACHED",
                "physical_contact": states,
            })
            break

        seed_ids = np.asarray(sorted(seeds), int)
        free_ids = _free_band(adj, seeds, locked, transition_rings)
        fixed = np.setdiff1d(np.arange(len(reference)), free_ids)
        active_before = {
            side: _active_subset_error(reference, ids, surfaces[side], target, crown_center)
            for side, ids in seed_sets.items()
        }
        accepted = None
        for alpha in ALPHAS:
            xt = reference.copy()
            step = float(alpha) * (desired[seed_ids] - reference[seed_ids])
            norm = np.linalg.norm(step, axis=1)
            large = norm > PER_PASS_MAX_MOVE_MM
            if np.any(large):
                step[large] *= (PER_PASS_MAX_MOVE_MM / norm[large])[:, None]
            xt[seed_ids] = reference[seed_ids] + step
            vertices = solve_local_band(reference, faces, free_ids, fixed, xt)[0] if len(free_ids) else xt
            moved = np.linalg.norm(vertices - reference, axis=1)
            candidate = current.copy()
            candidate.vertices = vertices
            gate = deformation_gate(
                reference, vertices, faces, scale_mm=.1,
                locked_ids=np.flatnonzero(locked),
                focus_faces=np.flatnonzero(np.any(moved[faces] > 1e-9, axis=1)),
            )
            flips = int((np.einsum("ij,ij->i", current.face_normals, candidate.face_normals) <= 0).sum())
            safe = (
                gate.ok and flips == 0 and not (candidate.area_faces < 1e-10).any()
                and float(moved.max(initial=0.0)) <= PER_PASS_MAX_MOVE_MM + 1e-6
                and float(moved[locked].max(initial=0.0)) <= 1e-5
            )
            active_after = {
                side: _active_subset_error(vertices, ids, surfaces[side], target, crown_center)
                for side, ids in seed_sets.items()
            }
            better = _subset_better(active_before, active_after)
            report["trials"].append({
                "pass": pass_no,
                "alpha": float(alpha),
                "seed_count": int(len(seed_ids)),
                "free_count": int(len(free_ids)),
                "gate": gate.as_dict(),
                "all_flips": flips,
                "maximum_displacement_mm": float(moved.max(initial=0.0)),
                "improved": bool(better),
                "accepted": bool(safe and better),
                "acceptance_scope": "ACTIVE_PATIENT_CONTACT_CORE",
                "active_before": active_before,
                "active_after": active_after,
            })
            if safe and better:
                accepted = candidate
                break
        if accepted is None:
            report["passes"].append({"pass": pass_no, "seed_count": int(len(seed_ids)), "status": "CONSTRAINT_CONFLICT"})
            break
        current = accepted
        ever_changed = True
        report["passes"].append({"pass": pass_no, "seed_count": int(len(seed_ids)), "status": "PASS_ENGINEERING"})

    final_shared = {
        side: measure(current.vertices, domains[side], surface, target, kind="PROXIMAL", side=side)[0]
        for side, surface in surfaces.items()
    }
    final_physical = {
        side: _contact_state(current.vertices, domains[side], surface, target, crown_center)
        for side, surface in surfaces.items()
    }
    report["after"] = final_shared
    report["physical_after"] = final_physical
    missing = [side for side, state in final_physical.items() if not state.get("target_reached")]
    if missing:
        report.update(
            status="CONSTRAINT_CONFLICT",
            reason="PROXIMAL_TARGET_NOT_REACHED",
            sides_not_reached=missing,
        )
    elif unreachable and not ever_changed:
        report.update(status="CONSTRAINT_CONFLICT", reason="TARGET_OUTSIDE_LOCAL_REACH")
    else:
        report.update(status="PASS_ENGINEERING", reason=None if ever_changed else "NO_CORRECTION_NEEDED")
    return current, report, domains
