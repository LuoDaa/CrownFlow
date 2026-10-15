"""Space-driven deterministic crown morphing for CAD-1.

The library tooth is an embryo only.  Patient geometry owns size:
Margin owns the cervical root, the actual intaglio owns the material floor,
adjacent teeth own proximal reach, and the antagonist owns coronal height.
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
    height_iterations: int = 7
    height_max_step_mm: float = 1.10
    height_target_band_mm: float = 0.025
    penetration_p95_limit_mm: float = 0.020
    envelope_iterations: int = 5
    envelope_max_push_per_pass_mm: float = 0.85
    shell_audit_sample_limit: int = 4096


def _basis(frame):
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 必须包含三维 origin 与 3x3 matrix")
    if not np.isfinite(origin).all() or not np.isfinite(matrix).all():
        raise ValueError("patient frame 含非有限数")
    return origin, matrix


def _local(points, frame):
    origin, matrix = _basis(frame)
    return (np.asarray(points, float) - origin) @ matrix


def _world(local, frame):
    origin, matrix = _basis(frame)
    return np.asarray(local, float) @ matrix.T + origin


def _smoothstep01(value):
    x = np.clip(np.asarray(value, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _finite_mesh(mesh, label="冠形"):
    vertices = np.asarray(mesh.vertices, float)
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
    pos = delta >= 0.0
    target[pos] = float(center) + delta[pos] * float(positive_scale)
    target[~pos] = float(center) + delta[~pos] * float(negative_scale)
    local[:, axis] += np.asarray(weight, float) * (target - values)


def _envelope_window_local(envelope_mesh, frame, margin_z, crown_top):
    local = _local(envelope_mesh.vertices, frame)
    local = local[np.isfinite(local).all(axis=1)]
    if not len(local):
        return local
    mask = (local[:, 2] >= margin_z - 0.50) & (local[:, 2] <= crown_top + 1.50)
    return local[mask] if np.any(mask) else local


def audit_inner_envelope_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm,
                                   *, config: SpaceMorphConfig | None = None):
    cfg = config or SpaceMorphConfig()
    if inner_envelope_mesh is None or required_mm is None or float(required_mm) <= 0:
        return {"status": "NOT_ASSESSED", "reason": "INNER_ENVELOPE_OR_THICKNESS_MISSING"}
    weight, height = _body_weight(mesh, frame, margin, cfg.body_transition_mm)
    ids = np.flatnonzero((weight > 0.05) & (height > 0.20))
    if not len(ids):
        return {"status": "NOT_ASSESSED", "reason": "NO_BODY_VERTICES"}
    _cp, dist, _ = closest_point(inner_envelope_mesh, np.asarray(mesh.vertices)[ids])
    dist = np.asarray(dist, float)
    values = dist[np.isfinite(dist)]
    if not len(values):
        return {"status": "NOT_ASSESSED", "reason": "NO_FINITE_CLEARANCE"}
    deficit = np.maximum(float(required_mm) - values, 0.0)
    inside_fraction = None
    try:
        inner_vertices = np.asarray(inner_envelope_mesh.vertices, float)
        if len(inner_vertices) > cfg.shell_audit_sample_limit:
            pick = np.linspace(0, len(inner_vertices) - 1, cfg.shell_audit_sample_limit, dtype=int)
            inner_vertices = inner_vertices[pick]
        if bool(mesh.is_watertight and mesh.is_winding_consistent):
            inside_fraction = float(np.mean(mesh.contains(inner_vertices)))
    except Exception:
        inside_fraction = None
    p95 = float(np.quantile(deficit, .95))
    return {
        "status": "PASS_ENGINEERING" if p95 <= 0.02 else "REVIEW_REQUIRED",
        "method": "OUTER_VERTEX_TO_ACTUAL_INTAGLIO_DISTANCE",
        "required_mm": float(required_mm),
        "sample_count": int(len(values)),
        "min_clearance_mm": float(np.min(values)),
        "p05_clearance_mm": float(np.quantile(values, .05)),
        "median_clearance_mm": float(np.median(values)),
        "deficit_p95_mm": p95,
        "deficit_max_mm": float(np.max(deficit)),
        "inner_vertex_containment_fraction": inside_fraction,
    }


def enforce_inner_envelope_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm,
                                     *, config: SpaceMorphConfig | None = None):
    """Expand the outer body around the real intaglio without changing crown height.

    Height is owned exclusively by the antagonist solver.  This guard therefore
    projects every material correction into the patient XY plane.  It cannot
    recreate an occlusal penetration after occlusal fitting and it fades to zero
    at the Margin root.
    """
    cfg = config or SpaceMorphConfig()
    if inner_envelope_mesh is None or required_mm is None or float(required_mm) <= 0:
        return mesh.copy(), {"status": "NOT_ASSESSED", "reason": "INNER_ENVELOPE_OR_THICKNESS_MISSING"}
    current = mesh.copy()
    margin_local = _local(margin, frame)
    center_xy = np.median(margin_local[:, :2], axis=0)
    origin, matrix = _basis(frame)
    moved_all, passes = set(), []
    unresolved = 0
    max_push = 0.0
    for iteration in range(max(1, int(cfg.envelope_iterations))):
        weight, height = _body_weight(current, frame, margin, cfg.body_transition_mm)
        vertices = np.asarray(current.vertices, float).copy()
        active = np.flatnonzero((weight > 0.02) & (height > 0.15))
        if not len(active):
            break
        cp, dist, _ = closest_point(inner_envelope_mesh, vertices[active])
        dist = np.asarray(dist, float)
        finite = np.isfinite(dist) & np.isfinite(cp).all(axis=1)
        deficit = np.maximum(float(required_mm) - dist, 0.0)
        bad = finite & (deficit > 1e-5)
        if not np.any(bad):
            passes.append({"iteration": iteration + 1, "moved_vertices": 0,
                           "min_clearance_mm": float(np.min(dist[finite])) if np.any(finite) else None})
            break
        chosen = active[bad]
        local_v = (vertices[chosen] - origin) @ matrix
        local_cp = (cp[bad] - origin) @ matrix
        sep_xy = local_v[:, :2] - local_cp[:, :2]
        radial = local_v[:, :2] - center_xy
        sep_n = np.linalg.norm(sep_xy, axis=1)
        rad_n = np.linalg.norm(radial, axis=1)
        sep_dir = np.zeros_like(sep_xy); rad_dir = np.zeros_like(radial)
        sg = sep_n > 1e-8; rg = rad_n > 1e-8
        sep_dir[sg] = sep_xy[sg] / sep_n[sg, None]
        rad_dir[rg] = radial[rg] / rad_n[rg, None]
        desired_xy = 0.70 * sep_dir + 0.30 * rad_dir
        dn = np.linalg.norm(desired_xy, axis=1)
        good = dn > 1e-8
        unresolved += int((~good).sum())
        if not np.any(good):
            break
        chosen = chosen[good]
        desired_xy = desired_xy[good] / dn[good, None]
        local_dir = np.c_[desired_xy, np.zeros(len(chosen))]
        world_dir = local_dir @ matrix.T
        push = deficit[bad][good] * np.clip(weight[chosen], 0.20, 1.0)
        push = np.minimum(push, float(cfg.envelope_max_push_per_pass_mm))
        vertices[chosen] += world_dir * push[:, None]
        current.vertices = vertices
        moved_all.update(map(int, chosen.tolist()))
        max_push = max(max_push, float(push.max(initial=0.0)))
        passes.append({"iteration": iteration + 1, "moved_vertices": int(len(chosen)),
                       "max_push_mm": float(push.max(initial=0.0)),
                       "min_before_mm": float(np.min(dist[bad]))})
    audit = audit_inner_envelope_clearance(current, inner_envelope_mesh, frame, margin, required_mm, config=cfg)
    return current, {
        "status": "APPLIED" if moved_all else "NO_CORRECTION_NEEDED",
        "method": "ACTUAL_INTAGLIO_PURE_LATERAL_CLEARANCE_V3",
        "required_outer_to_intaglio_mm": float(required_mm),
        "moved_vertices": int(len(moved_all)),
        "unresolved_direction_vertices": int(unresolved),
        "max_push_per_pass_mm": float(max_push),
        "passes": passes,
        "audit_after": audit,
    }


def _occlusal_metrics(values, target):
    values = np.asarray(values, float)
    error = values - float(target)
    penetration = np.maximum(-error, 0.0)
    gap = np.maximum(error, 0.0)
    positive = gap[gap > 0.0]
    gap_contact = float(np.quantile(positive, 0.18)) if len(positive) else 0.0
    return {
        "gap_quantile_mm": gap_contact,
        "penetration_p95_mm": float(np.quantile(penetration, .95)),
        "penetration_max_mm": float(np.max(penetration)),
        "signed_min_mm": float(np.min(values)),
        "signed_median_mm": float(np.median(values)),
        "signed_p05_mm": float(np.quantile(values, .05)),
    }


def _occlusal_objective(metrics, cfg):
    return (4.0 * max(metrics["penetration_p95_mm"] - cfg.penetration_p95_limit_mm, 0.0)
            + metrics["gap_quantile_mm"])


def _fit_antagonist_height(current, frame, margin, antagonist_surface, target, cfg):
    """Bidirectional closed-loop crown-height fit.

    A too-short embryo grows coronally; a globally over-high/penetrating embryo is
    lowered.  Local cusp/fossa relief remains the following OCCLUSAL stage's job.
    """
    target = float(target)
    passes, ever = [], False
    for iteration in range(max(1, int(cfg.height_iterations))):
        zones = build_crown_zones(current, frame, margin)
        ids = np.flatnonzero(zones.occlusal)
        if not len(ids):
            return current, {"applied": ever, "reason": "NO_OCCLUSAL_ZONE", "passes": passes}
        _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(current.vertices)[ids])
        signed = np.asarray(signed, float)
        valid = np.asarray(valid, bool) & np.isfinite(signed)
        if int(valid.sum()) < 12:
            return current, {"applied": ever, "reason": "NO_VALID_ANTAGONIST_SAMPLES", "passes": passes}
        values = signed[valid]
        before = _occlusal_metrics(values, target)
        if (before["penetration_p95_mm"] <= cfg.penetration_p95_limit_mm
                and before["gap_quantile_mm"] <= cfg.height_target_band_mm):
            return current, {"applied": ever, "reason": "HEIGHT_ON_TARGET", "passes": passes, "final": before,
                             "target_clearance_mm": target}
        penetration = np.maximum(target - values, 0.0)
        if float(np.quantile(penetration, .95)) > cfg.penetration_p95_limit_mm:
            p = penetration[penetration > cfg.penetration_p95_limit_mm]
            requested = -float(np.quantile(p, 0.70)) if len(p) else 0.0
            direction = "LOWER_TO_RELIEVE_PENETRATION"
        else:
            gap = values - target
            g = np.sort(gap[gap > cfg.height_target_band_mm])
            if not len(g):
                break
            take = max(1, int(np.ceil(len(g) * 0.22)))
            requested = float(np.median(g[:take]))
            direction = "RAISE_TO_CLOSE_GAP"
        requested = float(np.clip(requested, -cfg.height_max_step_mm, cfg.height_max_step_mm))
        if abs(requested) <= 1e-5:
            break
        weight, _ = _body_weight(current, frame, margin, cfg.body_transition_mm)
        local0 = _local(current.vertices, frame)
        accepted = None
        for alpha in (1.0, 0.7, 0.5, 0.3, 0.15):
            trial_local = local0.copy()
            trial_local[:, 2] += weight * requested * float(alpha)
            trial = current.copy(); trial.vertices = _world(trial_local, frame)
            _finite_mesh(trial)
            _cp2, _f2, signed2, _d2, valid2 = antagonist_surface.query(np.asarray(trial.vertices)[ids])
            signed2 = np.asarray(signed2, float)
            valid2 = np.asarray(valid2, bool) & np.isfinite(signed2)
            if int(valid2.sum()) < 12:
                continue
            after = _occlusal_metrics(signed2[valid2], target)
            improved = _occlusal_objective(after, cfg) < _occlusal_objective(before, cfg) - 1e-4
            passes.append({"iteration": iteration + 1, "direction": direction, "alpha": float(alpha),
                           "requested_step_mm": requested, "applied_step_mm": requested * float(alpha),
                           "before": before, "after": after, "accepted": bool(improved)})
            if improved:
                accepted = trial
                break
        if accepted is None:
            return current, {"applied": ever, "reason": "HEIGHT_LINE_SEARCH_CONFLICT", "passes": passes,
                             "final": before, "target_clearance_mm": target}
        current = accepted
        ever = True
    zones = build_crown_zones(current, frame, margin)
    ids = np.flatnonzero(zones.occlusal)
    final = None
    if len(ids):
        _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(current.vertices)[ids])
        signed = np.asarray(signed, float); valid = np.asarray(valid, bool) & np.isfinite(signed)
        if np.any(valid):
            final = _occlusal_metrics(signed[valid], target)
    return current, {"applied": ever, "mode": "BIDIRECTIONAL_CLOSED_LOOP_ANTAGONIST_HEIGHT_V2",
                     "passes": passes, "final": final, "target_clearance_mm": target}


def space_driven_embryo_prefit(mesh, frame, margin, *, support_mesh=None,
                               inner_envelope_mesh=None, minimum_thickness_mm=None,
                               cement_gap_mm=None, proximal_surfaces=None,
                               proximal_target_mm=None, antagonist_surface=None,
                               occlusal_target_mm=None, config: SpaceMorphConfig | None = None):
    """Fit embryo size floors and antagonist height before local refinements."""
    cfg = config or SpaceMorphConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    local = _local(current.vertices, frame)
    margin_local = _local(margin, frame)
    center_x = float(np.median(margin_local[:, 0])); center_y = float(np.median(margin_local[:, 1]))
    margin_z = float(np.median(margin_local[:, 2]))
    body_weight, height_above = _body_weight(current, frame, margin, cfg.body_transition_mm)
    body = height_above > max(0.30, cfg.body_transition_mm * 0.30)
    if not np.any(body):
        body = np.ones(len(local), bool)
    report = {
        "stage": "GLOBAL_PREFIT", "status": "PASS_ENGINEERING",
        "algorithm": "SPACE_DRIVEN_EMBRYO_MORPH_V4",
        "template_role": "TOPOLOGY_AND_ANATOMY_EMBRYO_NOT_SIZE_TRUTH",
        "md": {"applied": False, "mode": "INNER_SHELL_FLOOR_ONLY_PROXIMAL_DELEGATED",
               "neighbor_constraints_delegated": True,
               "neighbor_labels": sorted((proximal_surfaces or {}).keys()),
               "requested_proximal_target_mm": None if proximal_target_mm is None else float(proximal_target_mm),
               "whole_crown_translation_mm": 0.0},
        "bl": {"applied": False}, "inner_shell": {"status": "NOT_ASSESSED"},
        "height_floor": {"applied": False}, "height": {"applied": False},
    }
    current_min_x, current_max_x = map(float, (np.quantile(local[body, 0], .02), np.quantile(local[body, 0], .98)))
    current_min_y, current_max_y = map(float, (np.quantile(local[body, 1], .02), np.quantile(local[body, 1], .98)))
    min_thickness = float(minimum_thickness_mm) if minimum_thickness_mm is not None and np.isfinite(minimum_thickness_mm) and float(minimum_thickness_mm) > 0 else None
    if inner_envelope_mesh is not None:
        envelope_mesh = inner_envelope_mesh; shell_required = min_thickness
        envelope_source = "ACTUAL_BLOCKOUT_GAP_INTAGLIO_PREVIEW"
    else:
        envelope_mesh = support_mesh; shell_required = min_thickness
        if shell_required is not None and cement_gap_mm is not None and np.isfinite(cement_gap_mm) and float(cement_gap_mm) > 0:
            shell_required += float(cement_gap_mm)
        envelope_source = "PREP_SUPPORT_PLUS_CEMENT_GAP_FALLBACK"
    env = np.zeros((0, 3), float)
    if envelope_mesh is not None:
        env = _envelope_window_local(envelope_mesh, frame, margin_z, float(np.quantile(local[:, 2], .98)))
    env_min_x = env_max_x = env_min_y = env_max_y = env_top = None
    if len(env):
        env_min_x, env_max_x = float(np.quantile(env[:, 0], .02)), float(np.quantile(env[:, 0], .98))
        env_min_y, env_max_y = float(np.quantile(env[:, 1], .02)), float(np.quantile(env[:, 1], .98))
        env_top = float(np.quantile(env[:, 2], .98))
    floor_max_x = None if env_max_x is None or shell_required is None else env_max_x + shell_required
    floor_min_x = None if env_min_x is None or shell_required is None else env_min_x - shell_required
    desired_max_x = max(current_max_x, floor_max_x) if floor_max_x is not None else current_max_x
    desired_min_x = min(current_min_x, floor_min_x) if floor_min_x is not None else current_min_x
    pos_scale = _side_scale(current_max_x, desired_max_x, center_x, True, cfg.md_scale_min, cfg.md_scale_max)
    neg_scale = _side_scale(current_min_x, desired_min_x, center_x, False, cfg.md_scale_min, cfg.md_scale_max)
    _apply_asymmetric_axis_warp(local, 0, center_x, neg_scale, pos_scale, body_weight)
    report["md"].update({"applied": bool(abs(pos_scale - 1) > 1e-6 or abs(neg_scale - 1) > 1e-6),
                         "positive_x_scale": pos_scale, "negative_x_scale": neg_scale,
                         "desired_min_x_mm": desired_min_x, "desired_max_x_mm": desired_max_x})
    floor_max_y = None if env_max_y is None or shell_required is None else env_max_y + shell_required
    floor_min_y = None if env_min_y is None or shell_required is None else env_min_y - shell_required
    desired_max_y = max(current_max_y, floor_max_y) if floor_max_y is not None else current_max_y
    desired_min_y = min(current_min_y, floor_min_y) if floor_min_y is not None else current_min_y
    buc_scale = _side_scale(current_max_y, desired_max_y, center_y, True, cfg.bl_scale_min, cfg.bl_scale_max)
    lin_scale = _side_scale(current_min_y, desired_min_y, center_y, False, cfg.bl_scale_min, cfg.bl_scale_max)
    _apply_asymmetric_axis_warp(local, 1, center_y, lin_scale, buc_scale, body_weight)
    report["bl"] = {"applied": bool(abs(buc_scale - 1) > 1e-6 or abs(lin_scale - 1) > 1e-6),
                    "mode": "INNER_SHELL_MINIMUM_WIDTH", "buccal_scale": buc_scale,
                    "lingual_scale": lin_scale, "desired_min_y_mm": desired_min_y, "desired_max_y_mm": desired_max_y}
    current_top = float(np.quantile(local[body, 2], .98)); desired_top = current_top
    if env_top is not None and shell_required is not None:
        desired_top = max(current_top, env_top + shell_required)
    if desired_top > current_top + 1e-6:
        now_h = max(current_top - margin_z, .5); want_h = max(desired_top - margin_z, .5)
        scale = float(np.clip(want_h / now_h, cfg.height_scale_min, cfg.height_scale_max))
        full_z = margin_z + (local[:, 2] - margin_z) * scale
        local[:, 2] += body_weight * (full_z - local[:, 2])
        report["height_floor"] = {"applied": True, "mode": "ACTUAL_INTAGLIO_TOP_PLUS_MINIMUM_THICKNESS",
                                  "scale": scale, "current_top_mm": current_top, "desired_top_mm": desired_top,
                                  "inner_top_mm": env_top, "minimum_thickness_mm": shell_required}
    current.vertices = _world(local, frame)
    _finite_mesh(current)
    current, shell_report = enforce_inner_envelope_clearance(current, envelope_mesh, frame, margin, shell_required, config=cfg)
    report["inner_shell"] = {**shell_report, "source": envelope_source, "minimum_thickness_mm": min_thickness,
                             "cement_gap_mm": None if cement_gap_mm is None else float(cement_gap_mm),
                             "actual_intaglio_preview_used": bool(inner_envelope_mesh is not None)}
    if antagonist_surface is not None and occlusal_target_mm is not None:
        current, height_report = _fit_antagonist_height(current, frame, margin, antagonist_surface, occlusal_target_mm, cfg)
        report["height"] = height_report
    _finite_mesh(current)
    return current, report
