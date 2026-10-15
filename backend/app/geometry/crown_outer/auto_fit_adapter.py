"""CAD-1 automatic crown surface fitting.

The template is the anatomy prior. Neighbor and antagonist meshes are constraints,
not replacement surfaces:

* proximal: create a small contact patch and deform mainly along the confirmed
  mesial/distal axis;
* occlusal: first establish crown height from the closest antagonist envelope,
  then remove local clearance deficits mainly along the insertion axis;
* cervical vertices are never moved by AutoFit;
* already established proximal contacts can be protected during occlusal fitting.

All deformation is vertex-only local Laplacian editing, so frozen semantic-zone
vertex ids remain valid through the whole CAD-1 outer-crown pass.
"""
from dataclasses import dataclass

import numpy as np

from .deform_gate import deformation_gate
from .laplacian import solve_local_band
from .zones import ZONE_MODEL_VERSION, build_crown_zones


# CAD-1 engineering limits.  They are intentionally conservative per iteration;
# larger case-level changes are reached by repeated accepted passes rather than
# one violent deformation.
TARGET_TOL_MM = 0.03
PROXIMAL_REFERENCE_BAND_MM = 1.0
PROXIMAL_CONTACT_FRACTION = 0.35
OCCLUSAL_CONTACT_FRACTION = 0.22
PROXIMAL_MAX_REACH_MM = 3.5
OCCLUSAL_MAX_REACH_MM = 4.5
PER_PASS_MAX_MOVE_MM = 1.50
# Height-seeking may require several safe partial passes.  Five passes lets the
# solver accept a non-flipping partial raise (for example alpha=.65) and then
# continue toward the antagonist instead of demanding one violent move.
MAX_PASSES = 5
ALPHAS = (1.0, 0.65, 0.4, 0.25, 0.125)


@dataclass
class FitSurface:
    """Signed closest-surface query used by both proximal and occlusal fitting."""

    mesh: object
    outward: object = None

    def __post_init__(self):
        from ..measurement_core import SIGN_UNSIGNED_ONLY, SignSource
        self.source = SignSource.for_mesh(self.mesh, self.outward)
        if self.source.method == SIGN_UNSIGNED_ONLY:
            raise ValueError("开放表面需要人工确认外向法向")
        self.patch = self.source.patch
        self.solid = self.source.method == "SOLID_CONTAINMENT"
        self.outward = self.source.outward
        self.sign_method = self.source.method

    def query(self, points):
        return self.source.query(points)


def measure(vertices, mask, surface, target, *, kind=None, side=None, crown=None):
    """Single measurement truth source shared with reports/QC."""
    from ...services.contact_measurement import measure_surface
    return measure_surface(
        surface, vertices, mask, target,
        kind=kind or "CONTACT", side=side, crown=crown,
    )


def surface_crossings(mesh, surface, mask):
    """Triangle crossing audit in the requested crown domain."""
    from .thickness import _tri_tri_intersect

    face_ids = np.flatnonzero(np.any(np.asarray(mask)[mesh.faces], axis=1))
    pairs = []
    tree = surface.mesh.triangles_tree
    for face in face_ids:
        tri = mesh.triangles[face]
        pairs.extend(
            (int(face), int(other))
            for other in tree.intersection(np.r_[tri.min(0) - 1e-8, tri.max(0) + 1e-8])
        )
        if len(pairs) > 200000:
            return {
                "status": "NOT_ASSESSED",
                "reason": "PAIR_BUDGET_EXCEEDED",
                "candidate_pairs": len(pairs),
            }
    if not pairs:
        return {
            "status": "MEASURED_ENGINEERING",
            "crossing_pairs": 0,
            "near_parallel_candidates": 0,
            "candidate_pairs": 0,
            "intersecting_crown_area_mm2": 0,
        }
    pairs = np.asarray(pairs, int)
    hit, parallel, length = _tri_tri_intersect(
        mesh.triangles[pairs[:, 0]], surface.mesh.triangles[pairs[:, 1]], return_segment=True,
    )
    faces = np.unique(pairs[hit, 0])
    return {
        "status": "REVIEW_REQUIRED" if parallel else "MEASURED_ENGINEERING",
        "crossing_pairs": int(hit.sum()),
        "near_parallel_candidates": int(parallel),
        "candidate_pairs": len(pairs),
        "intersecting_crown_area_mm2": float(mesh.area_faces[faces].sum()),
        "intersection_segment_total_mm": float(length[hit].sum()),
        "method": "overlapping_triangle_AABB_and_existing_triangle_intersection",
        "coverage_note": "Crown domain only; coplanar overlaps require review.",
    }


def fit_domains(mesh, frame, margin, surfaces, stage, protected=None, zones=None):
    """Build one stable set of CAD-1 deformation domains.

    Geometry may move, topology may not.  Callers can therefore compute zones on
    the baseline template crown and reuse the same vertex ids through all stages.
    """
    reference = np.asarray(mesh.vertices)
    if zones is None:
        zones = build_crown_zones(mesh, frame, margin)
    zones.validate(len(reference))

    if stage == "PROXIMAL":
        masks = {side: np.asarray(zones.mask_for_side(side), bool).copy() for side in surfaces}
        # Contact design is not allowed to change margin or occlusal anatomy.
        locked = np.asarray(zones.cervical | zones.occlusal, bool).copy()
    elif stage == "OCCLUSAL":
        masks = {"occlusal": np.asarray(zones.occlusal, bool).copy()}
        locked = np.asarray(zones.cervical, bool).copy()
    else:
        raise ValueError(f"未知 AutoFit stage: {stage}")

    if protected is not None:
        protected = np.asarray(protected, bool)
        if protected.shape != (len(reference),):
            raise ValueError("保护区域与冠形顶点数不匹配")
        locked |= protected

    masks = {side: mask & ~locked for side, mask in masks.items()}

    if stage == "PROXIMAL":
        # Semantic mesial/distal zones are intentionally broad.  Keep only the
        # portion geometrically facing the actual neighbor so the side wall is
        # not pulled as one flat plate.
        for side, surface in surfaces.items():
            ids = np.flatnonzero(masks[side])
            if not len(ids):
                continue
            _, _, signed, _, valid = surface.query(reference[ids])
            if valid.any():
                closest = float(np.min(signed[valid]))
                masks[side][ids] &= valid & (signed <= closest + PROXIMAL_REFERENCE_BAND_MM)
    return masks, locked


def _neighbors(mesh):
    adj = [set() for _ in mesh.vertices]
    for a, b in np.asarray(mesh.edges_unique):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _free_band(adj, seeds, locked, rings):
    """Return transition vertices around seeds; seeds themselves stay target rows."""
    locked_ids = set(map(int, np.flatnonzero(locked)))
    seen = set(seeds) | locked_ids
    front = set(seeds)
    free = set()
    for _ in range(max(1, min(int(rings), 12))):
        nxt = {v for u in front for v in adj[u]} - seen
        if not nxt:
            break
        seen |= nxt
        free |= nxt
        front = nxt
    return np.asarray(sorted(free), int)


def _contact_seek_mask(signed, valid, target, fraction):
    """Select a compact closest-point patch when the whole zone is too far away."""
    ok = valid & np.isfinite(signed)
    if not np.any(ok):
        return np.zeros_like(valid, dtype=bool)
    vals = signed[ok]
    q = float(np.quantile(vals, min(max(fraction, 0.05), 0.8)))
    return ok & (signed <= q + 1e-9) & (signed > target + TARGET_TOL_MM)


def _desired_targets(reference, ids, cp, fids, signed, distance, valid,
                     surface, target, stage, side, frame, cut_only):
    """Return local seed ids + CAD target points for one reference surface."""
    target = float(target)
    reach = PROXIMAL_MAX_REACH_MM if stage == "PROXIMAL" else OCCLUSAL_MAX_REACH_MM
    usable = valid & np.isfinite(signed) & np.isfinite(distance) & (distance <= reach)
    if not np.any(usable):
        return np.zeros(0, int), np.zeros((0, 3), float), bool(np.any(valid))

    if cut_only:
        active = usable & (signed < -TARGET_TOL_MM)
    elif stage == "PROXIMAL":
        too_close = usable & (signed < target - TARGET_TOL_MM)
        on_target = usable & (np.abs(signed - target) <= TARGET_TOL_MM)
        if np.any(too_close):
            active = too_close
        elif np.any(on_target):
            # We already own a contact patch.  Do not attract the rest of the wall.
            active = np.zeros_like(usable)
        else:
            active = _contact_seek_mask(
                signed, usable, target, PROXIMAL_CONTACT_FRACTION,
            )
    else:
        # Occlusion is asymmetric once contact exists: collision/clearance deficit
        # is relieved everywhere it occurs, but an under-height crown is raised
        # only through a small closest antagonist patch.  This sets height without
        # copying the opposing surface into every cusp/fossa vertex.
        too_close = usable & (signed < target - TARGET_TOL_MM)
        on_target = usable & (np.abs(signed - target) <= TARGET_TOL_MM)
        if np.any(too_close):
            active = too_close
        elif np.any(on_target):
            active = np.zeros_like(usable)
        else:
            active = _contact_seek_mask(
                signed, usable, target, OCCLUSAL_CONTACT_FRACTION,
            )

    if not np.any(active):
        return np.zeros(0, int), np.zeros((0, 3), float), False

    surface_normals = surface.patch.normals[fids[active]] * surface.patch.normal_sign
    raw = cp[active] + surface_normals * target
    src = reference[ids[active]]
    delta = raw - src

    if stage == "PROXIMAL" and side in ("mesial", "distal"):
        direction = np.asarray(frame.x, float) * (1.0 if side == "mesial" else -1.0)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        axial = (delta @ direction)[:, None] * direction[None, :]
        # Small normal residual lets the local wall follow neighbor convexity while
        # preventing nearest-point collapse/spikes.
        raw = src + axial + 0.12 * (delta - axial)
    elif stage == "OCCLUSAL":
        direction = np.asarray(frame.z, float)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        axial = (delta @ direction)[:, None] * direction[None, :]
        # Opposing anatomy controls height/relief, but template cusp-fossa anatomy
        # remains the dominant prior.
        raw = src + axial + 0.08 * (delta - axial)

    return ids[active], raw, False


def _metric_value(report, stage, which):
    """Metric used to decide whether a trial moves the crown toward its target."""
    block = report.get("target_error") if isinstance(report, dict) else None
    if not block:
        block = report.get("deficit") if isinstance(report, dict) else None
    if not block:
        return None
    value = block.get(which)
    return float(value) if value is not None and np.isfinite(value) else None


def _is_better(before, after, stage):
    """Accept a pass only when no measured side worsens and at least one improves."""
    improved = False
    compared = False
    for side, old in before.items():
        new = after.get(side)
        if not isinstance(new, dict):
            continue
        old_max, new_max = _metric_value(old, stage, "max_mm"), _metric_value(new, stage, "max_mm")
        old_p95, new_p95 = _metric_value(old, stage, "p95_mm"), _metric_value(new, stage, "p95_mm")
        if old_max is None or new_max is None:
            continue
        compared = True
        if new_max > old_max + 1e-6:
            return False
        if old_p95 is not None and new_p95 is not None and new_p95 < old_p95 - 1e-4:
            improved = True
        elif new_max < old_max - 1e-4:
            improved = True
    return compared and improved


def _active_subset_error(vertices, ids, surface, target):
    """Measure target error only on the vertices selected for *this* correction pass.

    A stage-wide p95/max cannot judge a local edit: OCCLUSAL intentionally edits a
    compact contact patch (about 22% during height seek, or the penetrating set
    during relief), so the untouched majority can keep the full-domain aggregate
    almost unchanged even when every moved vertex improved substantially.
    """
    ids = np.unique(np.asarray(ids, dtype=int))
    if not len(ids):
        return {"status": "NOT_ASSESSED", "valid_count": 0}
    _cp, _faces, signed, _distance, valid = surface.query(np.asarray(vertices)[ids])
    signed = np.asarray(signed, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(signed)
    if not np.any(valid):
        return {"status": "NOT_ASSESSED", "valid_count": 0}
    values = signed[valid]
    error = np.abs(values - float(target))
    penetration = np.maximum(0.0, float(target) - values)
    return {
        "status": "MEASURED_ENGINEERING",
        "valid_count": int(valid.sum()),
        "target_error": {
            "median_mm": float(np.median(error)),
            "p95_mm": float(np.quantile(error, 0.95)),
            "max_mm": float(np.max(error)),
        },
        "signed_mm": {
            "min_mm": float(np.min(values)),
            "median_mm": float(np.median(values)),
            "max_mm": float(np.max(values)),
        },
        "penetration": {
            "p95_mm": float(np.quantile(penetration, 0.95)),
            "max_mm": float(np.max(penetration)),
        },
    }


def _active_subset_better(before, after):
    """Acceptance criterion for local OCCLUSAL edits.

    Compare the exact seed set that requested a correction.  p95 is the primary
    criterion, median is the secondary criterion.  The full-domain measurement is
    still recorded for QC/reporting, but it no longer vetoes a geometrically useful
    local move merely because most occlusal vertices were intentionally untouched.
    """
    compared = False
    improved = False
    for side, old in before.items():
        new = after.get(side)
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        if not old.get("valid_count") or not new.get("valid_count"):
            continue
        old_err = old.get("target_error") or {}
        new_err = new.get("target_error") or {}
        old_p95, new_p95 = old_err.get("p95_mm"), new_err.get("p95_mm")
        old_med, new_med = old_err.get("median_mm"), new_err.get("median_mm")
        if not all(np.isfinite(v) for v in (old_p95, new_p95, old_med, new_med)):
            continue
        compared = True
        # Local edit must move its actual target patch closer overall.  A tiny
        # median wobble is tolerated when p95 clearly improves; this is important
        # for curved antagonist surfaces and Laplacian transition bands.
        if new_p95 < old_p95 - 1e-4:
            improved = True
            continue
        if new_med < old_med - 1e-4 and new_p95 <= old_p95 + 1e-4:
            improved = True
            continue
        return False
    return compared and improved


def fit_stage(mesh, frame, margin, surfaces, targets, stage, *, protected=None,
              domain_mesh=None, domain_masks=None, cut_only=False,
              transition_rings=None, max_passes=MAX_PASSES):
    """Iteratively fit one functional crown region while preserving template anatomy."""
    current = mesh.copy()
    faces = np.asarray(mesh.faces).copy()
    domain_source = mesh if domain_mesh is None else domain_mesh
    frozen_masks, locked = fit_domains(domain_source, frame, margin, surfaces, stage, protected)
    if domain_masks is not None:
        for side, mask in domain_masks.items():
            mask = np.asarray(mask, bool)
            if mask.shape != (len(current.vertices),):
                raise ValueError("冻结适配区域与冠形顶点不匹配")
            frozen_masks[side] = mask & ~locked

    before_pairs = {
        side: measure(current.vertices, frozen_masks[side], surface, targets[side], kind=stage, side=side)
        for side, surface in surfaces.items()
    }
    before = {side: value[0] for side, value in before_pairs.items()}
    base = {
        "stage": stage,
        "before": before,
        "after": before,
        "passes": [],
        "trials": [],
        "zone_model": ZONE_MODEL_VERSION,
        "locked_vertex_count": int(locked.sum()),
        "transition_rings": int(transition_rings if transition_rings is not None else (6 if stage == "PROXIMAL" else 4)),
        "status": "REVIEW_REQUIRED",
        "reason": None,
    }
    if not any(r.get("valid_count", 0) for r in before.values()):
        base.update(status="NOT_ASSESSED", reason="NO_VALID_SURFACE_SAMPLES")
        return current, base, frozen_masks

    adj = _neighbors(mesh)
    rings = base["transition_rings"]
    ever_changed = False
    unreachable = False

    for pass_no in range(1, max(1, int(max_passes)) + 1):
        reference = np.asarray(current.vertices).copy()
        desired = reference.copy()
        seeds = set()
        seed_sets = {}

        pass_before = {}
        for side, surface in surfaces.items():
            report, query = measure(
                reference, frozen_masks[side], surface, targets[side], kind=stage, side=side,
            )
            pass_before[side] = report
            if query is None:
                continue
            ids, cp, fids, signed, distance, valid = query
            side_seed_ids, target_points, no_reachable = _desired_targets(
                reference, ids, cp, fids, signed, distance, valid,
                surface, targets[side], stage, side, frame, cut_only,
            )
            unreachable |= no_reachable
            if len(side_seed_ids):
                desired[side_seed_ids] = target_points
                seed_sets[side] = np.asarray(side_seed_ids, dtype=int)
                seeds.update(map(int, side_seed_ids))

        if not seeds:
            base["passes"].append({
                "pass": pass_no,
                "seed_count": 0,
                "status": "NO_CORRECTION_NEEDED" if not unreachable else "TARGET_OUTSIDE_LOCAL_REACH",
                "before": pass_before,
            })
            break

        free_ids = _free_band(adj, seeds, locked, rings)
        fixed = np.setdiff1d(np.arange(len(reference)), free_ids)
        seed_ids = np.asarray(sorted(seeds), int)
        active_before = {
            side: _active_subset_error(reference, ids, surfaces[side], targets[side])
            for side, ids in seed_sets.items()
        }
        accepted = None

        for alpha in ALPHAS:
            xt = reference.copy()
            step = alpha * (desired[seed_ids] - reference[seed_ids])
            norm = np.linalg.norm(step, axis=1)
            too_big = norm > PER_PASS_MAX_MOVE_MM
            if np.any(too_big):
                step[too_big] *= (PER_PASS_MAX_MOVE_MM / norm[too_big])[:, None]
            xt[seed_ids] = reference[seed_ids] + step

            vertices = solve_local_band(reference, faces, free_ids, fixed, xt)[0] if len(free_ids) else xt
            moved = np.linalg.norm(vertices - reference, axis=1)
            gate = deformation_gate(
                reference, vertices, faces, scale_mm=.1,
                locked_ids=np.flatnonzero(locked),
                focus_faces=np.flatnonzero(np.any(moved[faces] > 1e-9, axis=1)),
            )
            candidate = current.copy()
            candidate.vertices = vertices
            measured_pairs = {
                side: measure(vertices, frozen_masks[side], surface, targets[side], kind=stage, side=side)
                for side, surface in surfaces.items()
            }
            after = {side: value[0] for side, value in measured_pairs.items()}
            active_after = {
                side: _active_subset_error(vertices, ids, surfaces[side], targets[side])
                for side, ids in seed_sets.items()
            }
            flips = int((np.einsum("ij,ij->i", current.face_normals, candidate.face_normals) <= 0).sum())
            safe = (
                gate.ok
                and flips == 0
                and not (candidate.area_faces < 1e-10).any()
                and float(moved.max(initial=0.0)) <= PER_PASS_MAX_MOVE_MM + 1e-6
                and float(moved[locked].max(initial=0.0)) <= 1e-5
            )
            better = (
                _active_subset_better(active_before, active_after)
                if stage == "OCCLUSAL"
                else _is_better(pass_before, after, stage)
            )
            trial = {
                "pass": pass_no,
                "alpha": float(alpha),
                "seed_count": int(len(seed_ids)),
                "free_count": int(len(free_ids)),
                "gate": gate.as_dict(),
                "all_flips": flips,
                "maximum_displacement_mm": float(moved.max(initial=0.0)),
                "improved": bool(better),
                "acceptance_scope": "ACTIVE_SEED_SUBSET" if stage == "OCCLUSAL" else "FROZEN_STAGE_DOMAIN",
                "active_before": active_before if stage == "OCCLUSAL" else None,
                "active_after": active_after if stage == "OCCLUSAL" else None,
                "after": after,
                "accepted": bool(safe and better),
            }
            base["trials"].append(trial)
            if safe and better:
                accepted = (candidate, after, trial)
                break

        if accepted is None:
            base["passes"].append({
                "pass": pass_no,
                "seed_count": int(len(seed_ids)),
                "status": "CONSTRAINT_CONFLICT",
                "acceptance_scope": "ACTIVE_SEED_SUBSET" if stage == "OCCLUSAL" else "FROZEN_STAGE_DOMAIN",
            })
            break

        current, pass_after, trial = accepted
        ever_changed = True
        base["passes"].append({
            "pass": pass_no,
            "seed_count": int(len(seed_ids)),
            "status": "PASS_ENGINEERING",
            "accepted_alpha": trial["alpha"],
            "acceptance_scope": trial["acceptance_scope"],
            "after": pass_after,
        })

    final_pairs = {
        side: measure(current.vertices, frozen_masks[side], surface, targets[side], kind=stage, side=side)
        for side, surface in surfaces.items()
    }
    final = {side: value[0] for side, value in final_pairs.items()}
    base["after"] = final

    # A stage may legitimately need no geometry change because it is already in
    # target.  If there was a correction request but every candidate failed, keep
    # the conflict visible instead of silently passing the original template.
    if any(p.get("status") == "CONSTRAINT_CONFLICT" for p in base["passes"]):
        base.update(
            status="CONSTRAINT_CONFLICT",
            reason="CONTACT_CONSTRAINT_CONFLICT" if stage == "PROXIMAL" else "OCCLUSAL_CONSTRAINT_CONFLICT",
        )
    elif unreachable and not ever_changed:
        base.update(status="CONSTRAINT_CONFLICT", reason="TARGET_OUTSIDE_LOCAL_REACH")
    else:
        base.update(
            status="PASS_ENGINEERING",
            reason=None if ever_changed else "NO_CORRECTION_NEEDED",
            locked_displacement_mm=float(np.linalg.norm(
                np.asarray(current.vertices)[locked] - np.asarray(mesh.vertices)[locked], axis=1,
            ).max(initial=0.0)) if locked.any() else 0.0,
        )
    return current, base, frozen_masks
