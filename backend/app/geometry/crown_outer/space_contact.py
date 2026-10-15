"""Space-driven proximal contact refinement for CAD-1.

The previous path could report PASS from only a handful of vertices because a
signed-distance band and a global patch-normal gate collapsed the domain.  This
module treats the template as an embryo and the neighbor surface as the spatial
constraint:

1. start from the full mesial/distal semantic wall;
2. select a broad boundary-safe Euclidean-distance band (sign does not choose the domain);
3. require a minimum real sample population before the stage may pass;
4. create a compact contact core and move it toward the locally oriented neighbor;
5. diffuse the edit through a Laplacian transition band.
"""
from __future__ import annotations

import numpy as np

from .deform_gate import deformation_gate
from .laplacian import solve_local_band
from .zones import ZONE_MODEL_VERSION, build_crown_zones
from .auto_fit_adapter import measure

TARGET_TOL_MM = 0.03
REFERENCE_BAND_MM = 2.25
CONTACT_FRACTION = 0.35
MAX_REACH_MM = 4.5
MAX_PASSES = 5
PER_PASS_MAX_MOVE_MM = 1.50
ALPHAS = (1.0, 0.65, 0.4, 0.25, 0.125)
MIN_DOMAIN_VERTICES = 40
MIN_VALID_VERTICES = 24
MIN_DOMAIN_FRACTION_OF_SEMANTIC = 0.04


def _neighbors(mesh):
    adj = [set() for _ in mesh.vertices]
    for a, b in np.asarray(mesh.edges_unique, int):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _free_band(adj, seeds, locked, rings=8):
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


def build_proximal_domains(mesh, frame, margin, surfaces, *, protected=None, zones=None):
    """Return broad, trustworthy mesial/distal fitting domains.

    Domain selection is intentionally **unsigned**: Euclidean distance determines
    which part of the semantic wall faces the neighbor.  Signed distance is used
    later only for contact direction/penetration after the local normal has been
    established.
    """
    vertices = np.asarray(mesh.vertices, float)
    if zones is None:
        zones = build_crown_zones(mesh, frame, margin)
    zones.validate(len(vertices))

    locked = np.asarray(zones.cervical | zones.occlusal, bool).copy()
    if protected is not None:
        protected = np.asarray(protected, bool)
        if protected.shape != (len(vertices),):
            raise ValueError("保护区域与冠形顶点数不匹配")
        locked |= protected

    masks = {}
    diagnostics = {}
    for side, surface in surfaces.items():
        semantic = np.asarray(zones.mask_for_side(side), bool) & ~locked
        ids = np.flatnonzero(semantic)
        diag = {
            "semantic_vertex_count": int(len(ids)),
            "domain_vertex_count": 0,
            "valid_vertex_count": 0,
            "minimum_domain_vertices": int(MIN_DOMAIN_VERTICES),
            "minimum_valid_vertices": int(MIN_VALID_VERTICES),
            "selection": "BOUNDARY_SAFE_UNSIGNED_DISTANCE_BAND_WITH_ADAPTIVE_EXPANSION",
        }
        mask = np.zeros(len(vertices), bool)
        if not len(ids):
            diagnostics[side] = {**diag, "status": "NOT_ASSESSED", "reason": "EMPTY_SEMANTIC_SIDE"}
            masks[side] = mask
            continue

        _cp, _fids, _signed, distance, valid = surface.query(vertices[ids])
        distance = np.asarray(distance, float)
        valid = np.asarray(valid, bool) & np.isfinite(distance)
        diag["valid_vertex_count"] = int(valid.sum())
        if not np.any(valid):
            diagnostics[side] = {**diag, "status": "NOT_ASSESSED", "reason": "NO_VALID_SURFACE_SAMPLES"}
            masks[side] = mask
            continue

        valid_dist = distance[valid]
        minimum = float(np.min(valid_dist))
        keep = valid & (distance <= minimum + REFERENCE_BAND_MM)

        # Do not allow a valid but tiny sliver (e.g. 5/6021 vertices) to become a
        # green PASS.  Expand to a deterministic nearest subset when needed.
        semantic_floor = int(np.ceil(len(ids) * MIN_DOMAIN_FRACTION_OF_SEMANTIC))
        required_domain = min(int(valid.sum()), max(MIN_DOMAIN_VERTICES, semantic_floor))
        if int(keep.sum()) < required_domain:
            order = np.sort(valid_dist)
            threshold = float(order[min(required_domain - 1, len(order) - 1)])
            keep = valid & (distance <= threshold + 1e-9)

        chosen = ids[keep]
        mask[chosen] = True
        diag.update({
            "domain_vertex_count": int(len(chosen)),
            "minimum_distance_mm": minimum,
            "reference_band_mm": float(REFERENCE_BAND_MM),
            "status": "READY" if len(chosen) >= min(MIN_DOMAIN_VERTICES, int(valid.sum())) else "NOT_ASSESSED",
        })
        if diag["status"] != "READY":
            diag["reason"] = "INSUFFICIENT_PROXIMAL_DOMAIN"
        diagnostics[side] = diag
        masks[side] = mask
    return masks, locked, diagnostics


def _active_subset_error(vertices, ids, surface, target):
    ids = np.unique(np.asarray(ids, int))
    if not len(ids):
        return {"status": "NOT_ASSESSED", "valid_count": 0}
    _cp, _fids, signed, _distance, valid = surface.query(np.asarray(vertices)[ids])
    signed = np.asarray(signed, float)
    valid = np.asarray(valid, bool) & np.isfinite(signed)
    if not np.any(valid):
        return {"status": "NOT_ASSESSED", "valid_count": 0}
    values = signed[valid]
    err = np.abs(values - float(target))
    return {
        "status": "MEASURED_ENGINEERING",
        "valid_count": int(valid.sum()),
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


def _desired_contact(reference, mask, surface, target, side, frame):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return np.zeros(0, int), np.zeros((0, 3), float), False
    cp, fids, signed, distance, valid = surface.query(reference[ids])
    signed = np.asarray(signed, float)
    distance = np.asarray(distance, float)
    valid = np.asarray(valid, bool)
    usable = valid & np.isfinite(signed) & np.isfinite(distance) & (distance <= MAX_REACH_MM)
    if not np.any(usable):
        return np.zeros(0, int), np.zeros((0, 3), float), bool(np.any(valid))

    too_close = usable & (signed < float(target) - TARGET_TOL_MM)
    on_target = usable & (np.abs(signed - float(target)) <= TARGET_TOL_MM)
    if np.any(too_close):
        active = too_close
    elif np.sum(on_target) >= 6:
        active = np.zeros_like(usable)
    else:
        vals = distance[usable]
        q = float(np.quantile(vals, CONTACT_FRACTION))
        active = usable & (distance <= q + 1e-9) & (signed > float(target) + TARGET_TOL_MM)

    if not np.any(active):
        return np.zeros(0, int), np.zeros((0, 3), float), False

    normals = surface.source.oriented_normals(np.asarray(fids)[active])
    raw = np.asarray(cp)[active] + normals * float(target)
    src = reference[ids[active]]
    delta = raw - src

    direction = np.asarray(frame.x, float) * (1.0 if side == "mesial" else -1.0)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    axial = (delta @ direction)[:, None] * direction[None, :]
    # The template is an embryo, not the final side-wall truth.  Let 35% of the
    # local normal residual shape contact convexity/embrasure while retaining a
    # strong patient mesio-distal bias.
    target_points = src + axial + 0.35 * (delta - axial)
    return ids[active], target_points, False


def fit_proximal_envelopes(mesh, frame, margin, surfaces, target, *, protected=None,
                           transition_rings=8, max_passes=MAX_PASSES):
    current = mesh.copy()
    faces = np.asarray(current.faces).copy()
    domains, locked, domain_quality = build_proximal_domains(
        current, frame, margin, surfaces, protected=protected,
    )
    before_pairs = {
        side: measure(current.vertices, domains[side], surface, target, kind="PROXIMAL", side=side)
        for side, surface in surfaces.items()
    }
    before = {side: pair[0] for side, pair in before_pairs.items()}
    report = {
        "stage": "PROXIMAL",
        "status": "REVIEW_REQUIRED",
        "reason": None,
        "before": before,
        "after": before,
        "passes": [],
        "trials": [],
        "zone_model": ZONE_MODEL_VERSION,
        "domain_quality": domain_quality,
        "locked_vertex_count": int(locked.sum()),
        "transition_rings": int(transition_rings),
        "solver": "SPACE_DRIVEN_PROXIMAL_ENVELOPE_V1",
    }

    for side in surfaces:
        mask_count = int(domains[side].sum())
        valid_count = int((before.get(side) or {}).get("valid_count", 0))
        if mask_count < MIN_DOMAIN_VERTICES or valid_count < MIN_VALID_VERTICES:
            report.update(status="NOT_ASSESSED", reason="INSUFFICIENT_PROXIMAL_DOMAIN")
            return current, report, domains

    adj = _neighbors(current)
    ever_changed = False
    unreachable = False

    for pass_no in range(1, max(1, int(max_passes)) + 1):
        reference = np.asarray(current.vertices).copy()
        desired = reference.copy()
        seeds = set()
        seed_sets = {}
        pass_before = {}

        for side, surface in surfaces.items():
            pass_before[side] = measure(
                reference, domains[side], surface, target, kind="PROXIMAL", side=side,
            )[0]
            side_ids, points, no_reachable = _desired_contact(
                reference, domains[side], surface, target, side, frame,
            )
            unreachable |= no_reachable
            if len(side_ids):
                desired[side_ids] = points
                seed_sets[side] = side_ids
                seeds.update(map(int, side_ids))

        if not seeds:
            report["passes"].append({
                "pass": pass_no,
                "seed_count": 0,
                "status": "NO_CORRECTION_NEEDED" if not unreachable else "TARGET_OUTSIDE_LOCAL_REACH",
            })
            break

        seed_ids = np.asarray(sorted(seeds), int)
        free_ids = _free_band(adj, seeds, locked, transition_rings)
        fixed = np.setdiff1d(np.arange(len(reference)), free_ids)
        active_before = {
            side: _active_subset_error(reference, ids, surfaces[side], target)
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
                side: _active_subset_error(vertices, ids, surfaces[side], target)
                for side, ids in seed_sets.items()
            }
            better = _subset_better(active_before, active_after)
            trial = {
                "pass": pass_no,
                "alpha": float(alpha),
                "seed_count": int(len(seed_ids)),
                "free_count": int(len(free_ids)),
                "gate": gate.as_dict(),
                "all_flips": flips,
                "maximum_displacement_mm": float(moved.max(initial=0.0)),
                "improved": bool(better),
                "accepted": bool(safe and better),
                "acceptance_scope": "ACTIVE_CONTACT_CORE",
                "active_before": active_before,
                "active_after": active_after,
            }
            report["trials"].append(trial)
            if safe and better:
                accepted = candidate
                break

        if accepted is None:
            report["passes"].append({
                "pass": pass_no,
                "seed_count": int(len(seed_ids)),
                "status": "CONSTRAINT_CONFLICT",
            })
            break

        current = accepted
        ever_changed = True
        report["passes"].append({
            "pass": pass_no,
            "seed_count": int(len(seed_ids)),
            "status": "PASS_ENGINEERING",
        })

    final = {
        side: measure(current.vertices, domains[side], surface, target, kind="PROXIMAL", side=side)[0]
        for side, surface in surfaces.items()
    }
    report["after"] = final
    if any(p.get("status") == "CONSTRAINT_CONFLICT" for p in report["passes"]):
        report.update(status="CONSTRAINT_CONFLICT", reason="CONTACT_CONSTRAINT_CONFLICT")
    elif unreachable and not ever_changed:
        report.update(status="CONSTRAINT_CONFLICT", reason="TARGET_OUTSIDE_LOCAL_REACH")
    else:
        report.update(status="PASS_ENGINEERING", reason=None if ever_changed else "NO_CORRECTION_NEEDED")
    return current, report, domains
