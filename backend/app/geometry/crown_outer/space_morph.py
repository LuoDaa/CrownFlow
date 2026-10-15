"""Space-driven deterministic crown morphing.

The library tooth is an *embryo*: it supplies topology and local anatomy, but not
patient dimensions.  Patient geometry owns the usable space:

* Margin is the cervical boundary (handled by ``cervical_fit_to_margin``);
* mesial/distal neighbors define independent side envelopes;
* prepared support + cement gap + minimum thickness define an inner no-go envelope;
* antagonist defines the coronal height envelope;
* local Laplacian/ARAP-like stages refine contacts after this global space fit.

This follows the same deterministic idea used by clinical CAD systems and the
literature on deformable crown fitting: solve positional/spatial constraints first,
then use a shape-preserving regularizer.  No ML model is required for CAD-1.
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
    bl_scale_min: float = 0.75
    bl_scale_max: float = 1.80
    height_scale_min: float = 0.70
    height_scale_max: float = 1.55
    max_height_delta_mm: float = 3.5
    contact_quantile: float = 0.18
    support_iterations: int = 3
    support_max_push_per_pass_mm: float = 0.85
    support_query_cervical_exclusion_mm: float = 0.30


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


def _neighbor_contact_x(surface, frame, side, center_x):
    """Estimate the neighbor face that actually points into the edentulous space."""
    local = _local(surface.mesh.vertices, frame)
    x = local[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    if side == "mesial":
        values = x[x > center_x]
        if len(values) < 8:
            values = x
        return float(np.quantile(values, 0.06))
    if side == "distal":
        values = x[x < center_x]
        if len(values) < 8:
            values = x
        return float(np.quantile(values, 0.94))
    raise ValueError(f"未知邻接侧: {side}")


def _side_scale(current_extent, desired_extent, center, positive, lo, hi):
    if positive:
        current = max(float(current_extent - center), 1e-6)
        desired = max(float(desired_extent - center), 1e-6)
    else:
        current = max(float(center - current_extent), 1e-6)
        desired = max(float(center - desired_extent), 1e-6)
    return float(np.clip(desired / current, lo, hi))


def _apply_asymmetric_axis_warp(local, axis, center, negative_scale, positive_scale, weight):
    values = local[:, axis]
    delta = values - float(center)
    target = values.copy()
    positive = delta >= 0.0
    target[positive] = float(center) + delta[positive] * float(positive_scale)
    target[~positive] = float(center) + delta[~positive] * float(negative_scale)
    local[:, axis] += np.asarray(weight, float) * (target - values)


def _support_window_local(support_mesh, frame, margin_z, crown_top):
    local = _local(support_mesh.vertices, frame)
    finite = np.isfinite(local).all(axis=1)
    local = local[finite]
    if not len(local):
        return local
    mask = (local[:, 2] >= margin_z - 0.50) & (local[:, 2] <= crown_top + 1.0)
    return local[mask] if np.any(mask) else local


def _support_clearance_push(mesh, support_mesh, frame, margin, required_mm, body_weight, cfg):
    """Push outer body out of the support/intaglio minimum-material envelope.

    The prepared support is authoritative.  We move only the *outer* embryo and
    never the support.  Closest-point direction is blended with the radial patient
    direction so a point that accidentally starts inside the support is still
    pushed outward rather than deeper through it.
    """
    if support_mesh is None or required_mm is None or required_mm <= 0:
        return mesh.copy(), {
            "status": "NOT_ASSESSED",
            "reason": "SUPPORT_OR_MINIMUM_THICKNESS_MISSING",
        }

    current = mesh.copy()
    margin_local = _local(margin, frame)
    center_xy = np.median(margin_local[:, :2], axis=0)
    origin, matrix = _basis(frame)
    total_moved = set()
    max_push = 0.0
    passes = []

    body_weight = np.asarray(body_weight, dtype=float)
    for iteration in range(max(1, int(cfg.support_iterations))):
        vertices = np.asarray(current.vertices, dtype=float).copy()
        active = np.flatnonzero(body_weight > 0.02)
        if not len(active):
            break
        cp, dist, _ = closest_point(support_mesh, vertices[active])
        dist = np.asarray(dist, dtype=float)
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

        chosen = active[bad]
        world_delta = vertices[chosen] - cp[bad]
        local_delta = world_delta @ matrix
        local_vertices = (vertices[chosen] - origin) @ matrix

        radial = np.c_[local_vertices[:, :2] - center_xy, np.zeros(len(chosen))]
        radial_norm = np.linalg.norm(radial, axis=1)
        near_axis = radial_norm < 1e-8
        if np.any(near_axis):
            radial[near_axis] = np.array([0.0, 0.0, 1.0])
            radial_norm[near_axis] = 1.0
        radial /= np.maximum(radial_norm[:, None], 1e-12)

        dnorm = np.linalg.norm(local_delta, axis=1)
        local_dir = np.zeros_like(local_delta)
        ok = dnorm > 1e-8
        local_dir[ok] = local_delta[ok] / dnorm[ok, None]
        local_dir[~ok] = radial[~ok]
        blended = 0.65 * local_dir + 0.35 * radial
        # Always keep the correction on the material/outward side of the patient
        # radial field.  This is the key guard when the embryo starts *inside* prep.
        inward = np.einsum("ij,ij->i", blended, radial) < 0.0
        blended[inward] *= -1.0
        bnorm = np.linalg.norm(blended, axis=1)
        blended /= np.maximum(bnorm[:, None], 1e-12)

        push = deficit[bad] * np.clip(body_weight[chosen], 0.15, 1.0)
        push = np.minimum(push, float(cfg.support_max_push_per_pass_mm))
        vertices[chosen] += (blended @ matrix.T) * push[:, None]
        current.vertices = vertices
        total_moved.update(map(int, chosen.tolist()))
        max_push = max(max_push, float(push.max(initial=0.0)))
        passes.append({
            "iteration": iteration + 1,
            "candidate_vertices": int(len(active)),
            "moved_vertices": int(len(chosen)),
            "min_before_mm": float(np.min(dist[bad])),
            "max_push_mm": float(push.max(initial=0.0)),
        })

    active = np.flatnonzero(body_weight > 0.02)
    if len(active):
        _cp, after_dist, _ = closest_point(support_mesh, np.asarray(current.vertices)[active])
        after_dist = np.asarray(after_dist, dtype=float)
        finite = np.isfinite(after_dist)
        min_after = float(np.min(after_dist[finite])) if np.any(finite) else None
        p05_after = float(np.quantile(after_dist[finite], 0.05)) if np.any(finite) else None
    else:
        min_after = p05_after = None

    return current, {
        "status": "APPLIED" if total_moved else "NO_CORRECTION_NEEDED",
        "method": "SUPPORT_CLOSEST_POINT_PLUS_PATIENT_RADIAL_ENVELOPE",
        "required_outer_to_support_mm": float(required_mm),
        "moved_vertices": int(len(total_moved)),
        "max_push_per_pass_mm": float(max_push),
        "min_after_mm": min_after,
        "p05_after_mm": p05_after,
        "passes": passes,
    }


def space_driven_embryo_prefit(
    mesh,
    frame,
    margin,
    *,
    support_mesh=None,
    minimum_thickness_mm=None,
    cement_gap_mm=None,
    proximal_surfaces=None,
    proximal_target_mm=None,
    antagonist_surface=None,
    occlusal_target_mm=None,
    config: SpaceMorphConfig | None = None,
):
    """Fit the embryo crown to the *available patient space* before local editing.

    Unlike a rigid/translation proposal this warp has independent mesial and distal
    scales around the Margin center.  A one-sided neighbor therefore grows only the
    relevant side; the entire crown is never translated merely to create contact.
    """
    cfg = config or SpaceMorphConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    local = _local(current.vertices, frame)
    margin_local = _local(margin, frame)
    center_x = float(np.median(margin_local[:, 0]))
    center_y = float(np.median(margin_local[:, 1]))
    margin_z = float(np.median(margin_local[:, 2]))

    height_above = MarginCurtain(margin, frame.z).axial_above(current.vertices)
    body_weight = _smoothstep01(height_above / max(float(cfg.body_transition_mm), 1e-6))
    body = height_above > max(0.30, float(cfg.body_transition_mm) * 0.30)
    if not np.any(body):
        body = np.ones(len(local), bool)

    report = {
        "stage": "GLOBAL_PREFIT",
        "status": "PASS_ENGINEERING",
        "algorithm": "SPACE_DRIVEN_EMBRYO_MORPH_V1",
        "template_role": "TOPOLOGY_AND_ANATOMY_EMBRYO_NOT_SIZE_TRUTH",
        "md": {"applied": False},
        "bl": {"applied": False},
        "support_envelope": {"status": "NOT_ASSESSED"},
        "height": {"applied": False},
    }

    body_x = local[body, 0]
    body_y = local[body, 1]
    current_min_x = float(np.quantile(body_x, 0.02))
    current_max_x = float(np.quantile(body_x, 0.98))
    current_min_y = float(np.quantile(body_y, 0.02))
    current_max_y = float(np.quantile(body_y, 0.98))

    shell_required = None
    if minimum_thickness_mm is not None and np.isfinite(minimum_thickness_mm) and float(minimum_thickness_mm) > 0:
        shell_required = float(minimum_thickness_mm)
        if cement_gap_mm is not None and np.isfinite(cement_gap_mm) and float(cement_gap_mm) > 0:
            shell_required += float(cement_gap_mm)

    support_local = np.zeros((0, 3), dtype=float)
    if support_mesh is not None:
        crown_top = float(np.quantile(local[:, 2], 0.98))
        support_local = _support_window_local(support_mesh, frame, margin_z, crown_top)

    support_min_x = support_max_x = support_min_y = support_max_y = None
    if len(support_local):
        support_min_x = float(np.quantile(support_local[:, 0], 0.02))
        support_max_x = float(np.quantile(support_local[:, 0], 0.98))
        support_min_y = float(np.quantile(support_local[:, 1], 0.02))
        support_max_y = float(np.quantile(support_local[:, 1], 0.98))

    surfaces = proximal_surfaces or {}
    target = None if proximal_target_mm is None else float(proximal_target_mm)
    mesial_face = _neighbor_contact_x(surfaces["mesial"], frame, "mesial", center_x) if "mesial" in surfaces else None
    distal_face = _neighbor_contact_x(surfaces["distal"], frame, "distal", center_x) if "distal" in surfaces else None
    mesial_contact = (mesial_face - target) if mesial_face is not None and target is not None else None
    distal_contact = (distal_face + target) if distal_face is not None and target is not None else None

    floor_max_x = None if support_max_x is None or shell_required is None else support_max_x + shell_required
    floor_min_x = None if support_min_x is None or shell_required is None else support_min_x - shell_required

    desired_max_x = mesial_contact if mesial_contact is not None else current_max_x
    desired_min_x = distal_contact if distal_contact is not None else current_min_x
    if floor_max_x is not None:
        desired_max_x = max(desired_max_x, floor_max_x)
    if floor_min_x is not None:
        desired_min_x = min(desired_min_x, floor_min_x)

    constraint_conflict = bool(desired_min_x >= desired_max_x - 0.20)
    if not constraint_conflict:
        mesial_scale = _side_scale(current_max_x, desired_max_x, center_x, True,
                                    cfg.md_scale_min, cfg.md_scale_max)
        distal_scale = _side_scale(current_min_x, desired_min_x, center_x, False,
                                    cfg.md_scale_min, cfg.md_scale_max)
        _apply_asymmetric_axis_warp(local, 0, center_x, distal_scale, mesial_scale, body_weight)
        report["md"] = {
            "applied": bool(abs(mesial_scale - 1.0) > 1e-6 or abs(distal_scale - 1.0) > 1e-6),
            "mode": "INDEPENDENT_MESIAL_DISTAL_SCALE_ABOUT_MARGIN_CENTER",
            "center_x_mm": center_x,
            "mesial_scale": mesial_scale,
            "distal_scale": distal_scale,
            "current_min_x_mm": current_min_x,
            "current_max_x_mm": current_max_x,
            "desired_min_x_mm": float(desired_min_x),
            "desired_max_x_mm": float(desired_max_x),
            "mesial_neighbor_face_x_mm": mesial_face,
            "distal_neighbor_face_x_mm": distal_face,
            "support_floor_min_x_mm": floor_min_x,
            "support_floor_max_x_mm": floor_max_x,
            "whole_crown_translation_mm": 0.0,
        }
    else:
        report["md"] = {
            "applied": False,
            "status": "CONSTRAINT_CONFLICT",
            "reason": "SUPPORT_ENVELOPE_EXCEEDS_PROXIMAL_SPACE",
            "desired_min_x_mm": float(desired_min_x),
            "desired_max_x_mm": float(desired_max_x),
        }
        report["status"] = "REVIEW_REQUIRED"

    # Buccal/lingual have no direct adjacent-tooth contact target in CAD-1.  Use
    # the preparation/material envelope as a *minimum* width, never shrink the
    # embryo merely because its library BL width is smaller/larger.
    desired_min_y = current_min_y
    desired_max_y = current_max_y
    if shell_required is not None and support_min_y is not None and support_max_y is not None:
        desired_min_y = min(desired_min_y, support_min_y - shell_required)
        desired_max_y = max(desired_max_y, support_max_y + shell_required)
    buccal_scale = _side_scale(current_max_y, desired_max_y, center_y, True,
                                cfg.bl_scale_min, cfg.bl_scale_max)
    lingual_scale = _side_scale(current_min_y, desired_min_y, center_y, False,
                                 cfg.bl_scale_min, cfg.bl_scale_max)
    _apply_asymmetric_axis_warp(local, 1, center_y, lingual_scale, buccal_scale, body_weight)
    report["bl"] = {
        "applied": bool(abs(buccal_scale - 1.0) > 1e-6 or abs(lingual_scale - 1.0) > 1e-6),
        "mode": "SUPPORT_MATERIAL_ENVELOPE_MINIMUM_WIDTH",
        "buccal_scale": buccal_scale,
        "lingual_scale": lingual_scale,
        "desired_min_y_mm": float(desired_min_y),
        "desired_max_y_mm": float(desired_max_y),
    }

    current.vertices = _world(local, frame)
    _finite_mesh(current)

    current, support_report = _support_clearance_push(
        current, support_mesh, frame, margin, shell_required, body_weight, cfg,
    )
    report["support_envelope"] = {
        **support_report,
        "minimum_thickness_mm": None if minimum_thickness_mm is None else float(minimum_thickness_mm),
        "cement_gap_mm": None if cement_gap_mm is None else float(cement_gap_mm),
        "cement_gap_included": bool(cement_gap_mm is not None and np.isfinite(cement_gap_mm) and float(cement_gap_mm) > 0),
    }

    if antagonist_surface is not None and occlusal_target_mm is not None:
        local = _local(current.vertices, frame)
        zones = build_crown_zones(current, frame, margin)
        ids = np.flatnonzero(zones.occlusal)
        if len(ids):
            _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(current.vertices)[ids])
            signed = np.asarray(signed, float)
            valid = np.asarray(valid, bool) & np.isfinite(signed)
            if np.any(valid):
                errors = signed[valid] - float(occlusal_target_mm)
                ordered = np.sort(errors)
                take = max(1, int(np.ceil(len(ordered) * float(cfg.contact_quantile))))
                delta = float(np.median(ordered[:take]))
                delta = float(np.clip(delta, -cfg.max_height_delta_mm, cfg.max_height_delta_mm))
                occ_z = local[ids, 2]
                top = float(np.quantile(occ_z, 0.90))
                crown_height = max(top - margin_z, 0.5)
                scale = float(np.clip((crown_height + delta) / crown_height,
                                      cfg.height_scale_min, cfg.height_scale_max))
                height_above = MarginCurtain(margin, frame.z).axial_above(current.vertices)
                weight = _smoothstep01(height_above / max(float(cfg.body_transition_mm), 1e-6))
                full_z = margin_z + (local[:, 2] - margin_z) * scale
                local[:, 2] += weight * (full_z - local[:, 2])
                current.vertices = _world(local, frame)
                _finite_mesh(current)
                report["height"] = {
                    "applied": bool(abs(scale - 1.0) > 1e-6),
                    "mode": "ANTAGONIST_CONTACT_QUANTILE_HEIGHT_ENVELOPE",
                    "scale": scale,
                    "requested_delta_mm": delta,
                    "reference_height_mm": crown_height,
                    "sample_count": int(valid.sum()),
                    "target_clearance_mm": float(occlusal_target_mm),
                }
            else:
                report["height"] = {"applied": False, "reason": "NO_VALID_ANTAGONIST_SAMPLES"}
        else:
            report["height"] = {"applied": False, "reason": "NO_OCCLUSAL_ZONE"}

    _finite_mesh(current)
    return current, report
