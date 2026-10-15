"""Space-driven deterministic crown morphing.

The library tooth is only an embryo. Patient geometry owns dimensions:

* Margin = immutable cervical boundary;
* actual intaglio preview + minimum thickness = inner no-go shell;
* adjacent teeth = handled by ``space_contact.fit_proximal_envelopes`` using
  their real patient-space directions (never a global-X approximation here);
* antagonist = coronal height/clearance envelope;
* library = topology/anatomical character only.

This module deliberately does *not* size the crown from neighbor patches.  Older
versions did that by projecting both neighbors onto patient-frame X, which can
severely under-size posterior crowns when the arch is oblique.  The global prefit
now owns only the support/intaglio material envelope and antagonist height.  The
proximal solver owns MD reach and contact shape end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from trimesh.proximity import closest_point

from ..nonplanar_crown import MarginCurtain
from .zones import build_crown_zones


@dataclass(frozen=True)
class SpaceMorphConfig:
    body_transition_mm: float = 1.45
    md_scale_min: float = 0.55
    md_scale_max: float = 2.00
    bl_scale_min: float = 0.70
    bl_scale_max: float = 1.90
    height_scale_min: float = 0.70
    height_scale_max: float = 1.65
    max_height_delta_mm: float = 3.5
    contact_quantile: float = 0.18
    height_iterations: int = 5
    height_max_step_mm: float = 1.25
    height_target_band_mm: float = 0.03
    envelope_iterations: int = 4
    envelope_max_push_per_pass_mm: float = 0.90
    shell_audit_sample_limit: int = 4096


def _basis(frame):
    origin = np.asarray(frame.origin, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 必须包含三维 origin 与 3x3 matrix")
    if not np.isfinite(origin).all() or not np.isfinite(matrix).all():
        raise ValueError("patient frame 含非有限数")
    return origin, matrix


def _local(points, frame):
    origin, matrix = _basis(frame)
    return (np.asarray(points, dtype=float) - origin) @ matrix


def _world(local, frame):
    origin, matrix = _basis(frame)
    return np.asarray(local, dtype=float) @ matrix.T + origin


def _smoothstep01(value):
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _finite_mesh(mesh, label="冠形"):
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"{label}没有有效顶点")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label}含 NaN/Inf")
    if not len(mesh.faces):
        raise ValueError(f"{label}没有三角面")


def _body_weight(mesh, frame, margin, transition_mm):
    height = MarginCurtain(margin, frame.z).axial_above(mesh.vertices)
    return _smoothstep01(height / max(float(transition_mm), 1e-6)), height


def _side_scale(current_extent, desired_extent, center, positive, lo, hi):
    if positive:
        current = max(float(current_extent - center), 1e-6)
        desired = max(float(desired_extent - center), 1e-6)
    else:
        current = max(float(center - current_extent), 1e-6)
        desired = max(float(center - desired_extent), 1e-6)
    return float(np.clip(desired / current, lo, hi))


def _apply_asymmetric_axis_warp(local, axis, center, negative_scale, positive_scale, weight):
    values = local[:, axis].copy()
    delta = values - float(center)
    target = values.copy()
    positive = delta >= 0.0
    target[positive] = float(center) + delta[positive] * float(positive_scale)
    target[~positive] = float(center) + delta[~positive] * float(negative_scale)
    local[:, axis] += np.asarray(weight, float) * (target - values)


def _envelope_window_local(envelope_mesh, frame, margin_z, crown_top):
    local = _local(envelope_mesh.vertices, frame)
    local = local[np.isfinite(local).all(axis=1)]
    if not len(local):
        return local
    mask = (local[:, 2] >= margin_z - 0.50) & (local[:, 2] <= crown_top + 1.50)
    return local[mask] if np.any(mask) else local


def _neighbor_reference(surface, frame, center_x):
    """Legacy diagnostic only.

    Kept for report/backward compatibility.  It is intentionally no longer used
    for crown sizing; posterior neighbors are not reliably represented by X.
    """
    local = _local(surface.mesh.vertices, frame)
    x = local[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    median = float(np.median(x))
    sign = 1.0 if median >= center_x else -1.0
    if sign > 0:
        side_values = x[x > center_x]
        if len(side_values) < 8:
            side_values = x
        face_x = float(np.quantile(side_values, 0.06))
        axis_side = "POSITIVE_X"
    else:
        side_values = x[x < center_x]
        if len(side_values) < 8:
            side_values = x
        face_x = float(np.quantile(side_values, 0.94))
        axis_side = "NEGATIVE_X"
    return {"axis_sign": sign, "axis_side": axis_side, "face_x_mm": face_x, "median_x_mm": median}


def audit_inner_envelope_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm,
                                   *, config: SpaceMorphConfig | None = None):
    """Read-only outer-vs-intaglio body clearance audit."""
    cfg = config or SpaceMorphConfig()
    if inner_envelope_mesh is None or required_mm is None or float(required_mm) <= 0:
        return {"status": "NOT_ASSESSED", "reason": "INNER_ENVELOPE_OR_THICKNESS_MISSING"}
    weight, height = _body_weight(mesh, frame, margin, cfg.body_transition_mm)
    ids = np.flatnonzero((weight > 0.05) & (height > 0.20))
    if not len(ids):
        return {"status": "NOT_ASSESSED", "reason": "NO_BODY_VERTICES"}
    vertices = np.asarray(mesh.vertices)[ids]
    _cp, dist, _ = closest_point(inner_envelope_mesh, vertices)
    dist = np.asarray(dist, float)
    values = dist[np.isfinite(dist)]
    if not len(values):
        return {"status": "NOT_ASSESSED", "reason": "NO_FINITE_CLEARANCE"}
    deficit = np.maximum(float(required_mm) - values, 0.0)
    inside_fraction = None
    try:
        inner_vertices = np.asarray(inner_envelope_mesh.vertices, float)
        if len(inner_vertices) > cfg.shell_audit_sample_limit:
            sample = np.linspace(0, len(inner_vertices) - 1, cfg.shell_audit_sample_limit, dtype=int)
            inner_vertices = inner_vertices[sample]
        if bool(mesh.is_watertight and mesh.is_winding_consistent):
            inside_fraction = float(np.mean(mesh.contains(inner_vertices)))
    except Exception:
        inside_fraction = None
    return {
        "status": "PASS_ENGINEERING" if float(np.quantile(deficit, .95)) <= 0.02 else "REVIEW_REQUIRED",
        "method": "OUTER_VERTEX_TO_ACTUAL_INTAGLIO_DISTANCE",
        "required_mm": float(required_mm),
        "sample_count": int(len(values)),
        "min_clearance_mm": float(np.min(values)),
        "p05_clearance_mm": float(np.quantile(values, .05)),
        "median_clearance_mm": float(np.median(values)),
        "deficit_p95_mm": float(np.quantile(deficit, .95)),
        "deficit_max_mm": float(np.max(deficit)),
        "inner_vertex_containment_fraction": inside_fraction,
    }


def enforce_inner_envelope_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm,
                                     *, config: SpaceMorphConfig | None = None):
    """Expand the outer body laterally away from the authoritative intaglio.

    The shell guard must never be a hidden crown-height solver.  Earlier versions
    added a positive patient-Z component while expanding thickness; when this ran
    *after* occlusal fitting it could re-create millimetre antagonist penetration.
    Coronal height is owned exclusively by the antagonist solver.  Here we keep
    positive-Z displacement out of the shell correction.  If a central point has
    no safe lateral direction, it is left for the final shell audit instead of
    corrupting the occlusion.
    """
    cfg = config or SpaceMorphConfig()
    if inner_envelope_mesh is None or required_mm is None or float(required_mm) <= 0:
        return mesh.copy(), {"status": "NOT_ASSESSED", "reason": "INNER_ENVELOPE_OR_THICKNESS_MISSING"}

    current = mesh.copy()
    margin_local = _local(margin, frame)
    center_xy = np.median(margin_local[:, :2], axis=0)
    origin, matrix = _basis(frame)
    total_moved = set()
    max_push = 0.0
    unresolved_direction = 0
    passes = []

    for iteration in range(max(1, int(cfg.envelope_iterations))):
        weight, height = _body_weight(current, frame, margin, cfg.body_transition_mm)
        vertices = np.asarray(current.vertices, float).copy()
        active = np.flatnonzero((weight > 0.02) & (height > 0.12))
        if not len(active):
            break
        cp, dist, _ = closest_point(inner_envelope_mesh, vertices[active])
        dist = np.asarray(dist, float)
        finite = np.isfinite(dist) & np.isfinite(cp).all(axis=1)
        deficit = np.maximum(float(required_mm) - dist, 0.0)
        bad = finite & (deficit > 1e-5)
        if not np.any(bad):
            passes.append({
                "iteration": iteration + 1,
                "candidate_vertices": int(len(active)),
                "moved_vertices": 0,
                "min_clearance_mm": float(np.min(dist[finite])) if np.any(finite) else None,
            })
            break

        chosen_all = active[bad]
        local_vertices = (vertices[chosen_all] - origin) @ matrix
        local_delta = (vertices[chosen_all] - cp[bad]) @ matrix

        radial = np.c_[local_vertices[:, :2] - center_xy, np.zeros(len(chosen_all))]
        radial_norm = np.linalg.norm(radial[:, :2], axis=1)
        radial_good = radial_norm > 1e-8
        radial[radial_good] /= radial_norm[radial_good, None]
        radial[~radial_good] = 0.0

        dnorm = np.linalg.norm(local_delta, axis=1)
        sep = np.zeros_like(local_delta)
        sep_good = dnorm > 1e-8
        sep[sep_good] = local_delta[sep_good] / dnorm[sep_good, None]

        desired = 0.65 * sep + 0.35 * radial
        # Never increase crown height in a material-space correction.
        desired[:, 2] = np.minimum(desired[:, 2], 0.0)
        norm = np.linalg.norm(desired, axis=1)
        safe_dir = norm > 1e-8
        unresolved_direction += int((~safe_dir).sum())
        if not np.any(safe_dir):
            passes.append({
                "iteration": iteration + 1,
                "candidate_vertices": int(len(active)),
                "moved_vertices": 0,
                "unresolved_direction_vertices": int(len(chosen_all)),
            })
            break

        chosen = chosen_all[safe_dir]
        desired = desired[safe_dir] / norm[safe_dir, None]
        push = deficit[bad][safe_dir] * np.clip(weight[chosen], 0.20, 1.0)
        push = np.minimum(push, float(cfg.envelope_max_push_per_pass_mm))
        vertices[chosen] += (desired @ matrix.T) * push[:, None]
        current.vertices = vertices
        total_moved.update(map(int, chosen.tolist()))
        max_push = max(max_push, float(push.max(initial=0.0)))
        passes.append({
            "iteration": iteration + 1,
            "candidate_vertices": int(len(active)),
            "moved_vertices": int(len(chosen)),
            "unresolved_direction_vertices": int((~safe_dir).sum()),
            "min_before_mm": float(np.min(dist[bad])),
            "max_push_mm": float(push.max(initial=0.0)),
        })

    audit = audit_inner_envelope_clearance(current, inner_envelope_mesh, frame, margin, required_mm, config=cfg)
    return current, {
        "status": "APPLIED" if total_moved else "NO_CORRECTION_NEEDED",
        "method": "ACTUAL_INTAGLIO_LATERAL_CLEARANCE_OCCLUSION_PRESERVING_V2",
        "required_outer_to_intaglio_mm": float(required_mm),
        "moved_vertices": int(len(total_moved)),
        "unresolved_direction_vertices": int(unresolved_direction),
        "max_push_per_pass_mm": float(max_push),
        "passes": passes,
        "audit_after": audit,
    }


def _occlusal_contact_metric(values, target, quantile):
    values = np.asarray(values, float)
    err = values - float(target)
    positive = np.maximum(err, 0.0)
    penetration = np.maximum(-err, 0.0)
    return {
        "gap_quantile_mm": float(np.quantile(positive, min(max(float(quantile), .05), .5))),
        "penetration_p95_mm": float(np.quantile(penetration, .95)),
        "penetration_max_mm": float(np.max(penetration)),
        "signed_min_mm": float(np.min(values)),
        "signed_median_mm": float(np.median(values)),
    }


def _fit_antagonist_height(current, frame, margin, antagonist_surface, target, cfg):
    """Monotone closed-loop crown-height seek along patient +Z."""
    target = float(target)
    passes = []
    ever = False
    for iteration in range(max(1, int(cfg.height_iterations))):
        zones = build_crown_zones(current, frame, margin)
        ids = np.flatnonzero(zones.occlusal)
        if not len(ids):
            return current, {"applied": ever, "reason": "NO_OCCLUSAL_ZONE", "passes": passes}
        _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(current.vertices)[ids])
        signed = np.asarray(signed, float)
        valid = np.asarray(valid, bool) & np.isfinite(signed)
        if not np.any(valid):
            return current, {"applied": ever, "reason": "NO_VALID_ANTAGONIST_SAMPLES", "passes": passes}
        values = signed[valid]
        before = _occlusal_contact_metric(values, target, cfg.contact_quantile)
        if before["penetration_p95_mm"] > 0.02:
            return current, {
                "applied": ever,
                "reason": "PENETRATION_PRESENT_LOCAL_REFINEMENT_REQUIRED",
                "passes": passes,
                "final": before,
            }
        if before["gap_quantile_mm"] <= cfg.height_target_band_mm:
            return current, {"applied": ever, "reason": "HEIGHT_ON_TARGET", "passes": passes, "final": before}

        errors = np.sort((values - target)[values > target + cfg.height_target_band_mm])
        if not len(errors):
            return current, {"applied": ever, "reason": "HEIGHT_ON_TARGET", "passes": passes, "final": before}
        take = max(1, int(np.ceil(len(errors) * float(cfg.contact_quantile))))
        requested = float(np.median(errors[:take]))
        requested = float(np.clip(requested, 0.0, cfg.height_max_step_mm))
        if requested <= 1e-5:
            break

        weight, _height = _body_weight(current, frame, margin, cfg.body_transition_mm)
        local0 = _local(current.vertices, frame)
        accepted = None
        for alpha in (1.0, 0.65, 0.40, 0.25, 0.125):
            trial_local = local0.copy()
            trial_local[:, 2] += weight * requested * float(alpha)
            trial = current.copy()
            trial.vertices = _world(trial_local, frame)
            _finite_mesh(trial)
            _cp2, _f2, signed2, _d2, valid2 = antagonist_surface.query(np.asarray(trial.vertices)[ids])
            signed2 = np.asarray(signed2, float)
            valid2 = np.asarray(valid2, bool) & np.isfinite(signed2)
            if not np.any(valid2):
                continue
            after = _occlusal_contact_metric(signed2[valid2], target, cfg.contact_quantile)
            improves = after["gap_quantile_mm"] < before["gap_quantile_mm"] - 1e-4
            safe_pen = after["penetration_p95_mm"] <= max(0.02, before["penetration_p95_mm"] + 0.003)
            passes.append({
                "iteration": iteration + 1,
                "alpha": float(alpha),
                "requested_step_mm": requested,
                "applied_step_mm": requested * float(alpha),
                "before": before,
                "after": after,
                "accepted": bool(improves and safe_pen),
            })
            if improves and safe_pen:
                accepted = trial
                break
        if accepted is None:
            return current, {"applied": ever, "reason": "HEIGHT_LINE_SEARCH_CONFLICT", "passes": passes, "final": before}
        current = accepted
        ever = True

    zones = build_crown_zones(current, frame, margin)
    ids = np.flatnonzero(zones.occlusal)
    final = None
    if len(ids):
        _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(current.vertices)[ids])
        signed = np.asarray(signed, float)
        valid = np.asarray(valid, bool) & np.isfinite(signed)
        if np.any(valid):
            final = _occlusal_contact_metric(signed[valid], target, cfg.contact_quantile)
    return current, {
        "applied": ever,
        "mode": "MONOTONE_CLOSED_LOOP_ANTAGONIST_HEIGHT_SEEK",
        "passes": passes,
        "final": final,
        "target_clearance_mm": target,
    }


def space_driven_embryo_prefit(
    mesh,
    frame,
    margin,
    *,
    support_mesh=None,
    inner_envelope_mesh=None,
    minimum_thickness_mm=None,
    cement_gap_mm=None,
    proximal_surfaces=None,
    proximal_target_mm=None,
    antagonist_surface=None,
    occlusal_target_mm=None,
    config: SpaceMorphConfig | None = None,
):
    """Fit the embryo to hard patient space before local contact refinement.

    Neighbor patches are accepted only for provenance/reporting here.  Their real
    geometry is consumed later by ``fit_proximal_envelopes``.  This prevents two
    independent MD solvers from fighting each other.
    """
    cfg = config or SpaceMorphConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    local = _local(current.vertices, frame)
    margin_local = _local(margin, frame)
    center_x = float(np.median(margin_local[:, 0]))
    center_y = float(np.median(margin_local[:, 1]))
    margin_z = float(np.median(margin_local[:, 2]))

    body_weight, height_above = _body_weight(current, frame, margin, cfg.body_transition_mm)
    body = height_above > max(0.30, float(cfg.body_transition_mm) * 0.30)
    if not np.any(body):
        body = np.ones(len(local), bool)

    report = {
        "stage": "GLOBAL_PREFIT",
        "status": "PASS_ENGINEERING",
        "algorithm": "SPACE_DRIVEN_EMBRYO_MORPH_V4",
        "template_role": "TOPOLOGY_AND_ANATOMY_EMBRYO_NOT_SIZE_TRUTH",
        "md": {
            "applied": False,
            "mode": "INNER_SHELL_FLOOR_ONLY_PROXIMAL_DELEGATED",
            "neighbor_constraints_delegated": True,
            "neighbor_labels": sorted((proximal_surfaces or {}).keys()),
            "requested_proximal_target_mm": None if proximal_target_mm is None else float(proximal_target_mm),
            "whole_crown_translation_mm": 0.0,
        },
        "bl": {"applied": False},
        "inner_shell": {"status": "NOT_ASSESSED"},
        "height_floor": {"applied": False},
        "height": {"applied": False},
    }

    body_x = local[body, 0]
    body_y = local[body, 1]
    current_min_x = float(np.quantile(body_x, 0.02))
    current_max_x = float(np.quantile(body_x, 0.98))
    current_min_y = float(np.quantile(body_y, 0.02))
    current_max_y = float(np.quantile(body_y, 0.98))

    min_thickness = None
    if minimum_thickness_mm is not None and np.isfinite(minimum_thickness_mm) and float(minimum_thickness_mm) > 0:
        min_thickness = float(minimum_thickness_mm)

    if inner_envelope_mesh is not None:
        envelope_mesh = inner_envelope_mesh
        shell_required = min_thickness
        envelope_source = "ACTUAL_BLOCKOUT_GAP_INTAGLIO_PREVIEW"
    else:
        envelope_mesh = support_mesh
        shell_required = min_thickness
        if shell_required is not None and cement_gap_mm is not None and np.isfinite(cement_gap_mm) and float(cement_gap_mm) > 0:
            shell_required += float(cement_gap_mm)
        envelope_source = "PREP_SUPPORT_PLUS_CEMENT_GAP_FALLBACK"

    envelope_local = np.zeros((0, 3), float)
    if envelope_mesh is not None:
        crown_top = float(np.quantile(local[:, 2], 0.98))
        envelope_local = _envelope_window_local(envelope_mesh, frame, margin_z, crown_top)

    env_min_x = env_max_x = env_min_y = env_max_y = env_top = None
    if len(envelope_local):
        env_min_x = float(np.quantile(envelope_local[:, 0], 0.02))
        env_max_x = float(np.quantile(envelope_local[:, 0], 0.98))
        env_min_y = float(np.quantile(envelope_local[:, 1], 0.02))
        env_max_y = float(np.quantile(envelope_local[:, 1], 0.98))
        env_top = float(np.quantile(envelope_local[:, 2], 0.98))

    # Global MD is only a *material floor*.  Adjacent teeth are handled later in
    # real patient vectors by space_contact; no global-X neighbor sizing here.
    floor_max_x = None if env_max_x is None or shell_required is None else env_max_x + shell_required
    floor_min_x = None if env_min_x is None or shell_required is None else env_min_x - shell_required
    desired_max_x = max(current_max_x, floor_max_x) if floor_max_x is not None else current_max_x
    desired_min_x = min(current_min_x, floor_min_x) if floor_min_x is not None else current_min_x
    if desired_min_x < desired_max_x - 0.20:
        positive_scale = _side_scale(current_max_x, desired_max_x, center_x, True, cfg.md_scale_min, cfg.md_scale_max)
        negative_scale = _side_scale(current_min_x, desired_min_x, center_x, False, cfg.md_scale_min, cfg.md_scale_max)
        _apply_asymmetric_axis_warp(local, 0, center_x, negative_scale, positive_scale, body_weight)
        report["md"].update({
            "applied": bool(abs(positive_scale - 1.0) > 1e-6 or abs(negative_scale - 1.0) > 1e-6),
            "positive_x_scale": positive_scale,
            "negative_x_scale": negative_scale,
            "current_min_x_mm": current_min_x,
            "current_max_x_mm": current_max_x,
            "desired_min_x_mm": float(desired_min_x),
            "desired_max_x_mm": float(desired_max_x),
            "inner_floor_min_x_mm": floor_min_x,
            "inner_floor_max_x_mm": floor_max_x,
        })
    else:
        report["md"].update({
            "status": "CONSTRAINT_CONFLICT",
            "reason": "INNER_SHELL_GEOMETRY_INVALID",
            "desired_min_x_mm": float(desired_min_x),
            "desired_max_x_mm": float(desired_max_x),
        })
        report["status"] = "REVIEW_REQUIRED"

    desired_min_y = current_min_y
    desired_max_y = current_max_y
    if shell_required is not None and env_min_y is not None and env_max_y is not None:
        desired_min_y = min(desired_min_y, env_min_y - shell_required)
        desired_max_y = max(desired_max_y, env_max_y + shell_required)
    buccal_scale = _side_scale(current_max_y, desired_max_y, center_y, True, cfg.bl_scale_min, cfg.bl_scale_max)
    lingual_scale = _side_scale(current_min_y, desired_min_y, center_y, False, cfg.bl_scale_min, cfg.bl_scale_max)
    _apply_asymmetric_axis_warp(local, 1, center_y, lingual_scale, buccal_scale, body_weight)
    report["bl"] = {
        "applied": bool(abs(buccal_scale - 1.0) > 1e-6 or abs(lingual_scale - 1.0) > 1e-6),
        "mode": "INNER_SHELL_MINIMUM_WIDTH",
        "buccal_scale": buccal_scale,
        "lingual_scale": lingual_scale,
        "desired_min_y_mm": float(desired_min_y),
        "desired_max_y_mm": float(desired_max_y),
    }

    current_top = float(np.quantile(local[body, 2], .98))
    desired_top = current_top
    if env_top is not None and shell_required is not None:
        desired_top = max(current_top, env_top + shell_required)
    if desired_top > current_top + 1e-6:
        height_now = max(current_top - margin_z, 0.5)
        height_desired = max(desired_top - margin_z, 0.5)
        top_scale = float(np.clip(height_desired / height_now, cfg.height_scale_min, cfg.height_scale_max))
        full_z = margin_z + (local[:, 2] - margin_z) * top_scale
        local[:, 2] += body_weight * (full_z - local[:, 2])
        report["height_floor"] = {
            "applied": True,
            "mode": "ACTUAL_INTAGLIO_TOP_PLUS_MINIMUM_THICKNESS",
            "scale": top_scale,
            "current_top_mm": current_top,
            "desired_top_mm": desired_top,
            "inner_top_mm": env_top,
            "minimum_thickness_mm": shell_required,
        }

    current.vertices = _world(local, frame)
    _finite_mesh(current)
    current, shell_report = enforce_inner_envelope_clearance(
        current, envelope_mesh, frame, margin, shell_required, config=cfg,
    )
    report["inner_shell"] = {
        **shell_report,
        "source": envelope_source,
        "minimum_thickness_mm": min_thickness,
        "cement_gap_mm": None if cement_gap_mm is None else float(cement_gap_mm),
        "actual_intaglio_preview_used": bool(inner_envelope_mesh is not None),
    }

    if antagonist_surface is not None and occlusal_target_mm is not None:
        current, height_report = _fit_antagonist_height(
            current, frame, margin, antagonist_surface, occlusal_target_mm, cfg,
        )
        report["height"] = height_report

    _finite_mesh(current)
    return current, report
