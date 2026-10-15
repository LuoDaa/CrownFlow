"""Space-driven deterministic crown morphing.

The library tooth is only an *embryo*: topology/anatomical character may come from
it, but patient dimensions never do.  The outer crown is first expanded into the
feasible patient space, then local Laplacian stages create proximal/occlusal detail.

Spatial truth, in order of authority:

* confirmed Margin = cervical boundary;
* actual intaglio preview (blockout + gap) + minimum thickness = inner no-go shell;
* mesial/distal neighbor surfaces = independent side contact envelopes;
* antagonist = coronal height/clearance envelope;
* library embryo = soft shape prior only.
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


def _envelope_window_local(envelope_mesh, frame, margin_z, crown_top):
    local = _local(envelope_mesh.vertices, frame)
    finite = np.isfinite(local).all(axis=1)
    local = local[finite]
    if not len(local):
        return local
    mask = (local[:, 2] >= margin_z - 0.50) & (local[:, 2] <= crown_top + 1.50)
    return local[mask] if np.any(mask) else local


def _body_weight(mesh, frame, margin, transition_mm):
    h = MarginCurtain(margin, frame.z).axial_above(mesh.vertices)
    return _smoothstep01(h / max(float(transition_mm), 1e-6)), h


def audit_inner_envelope_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm,
                                   *, config: SpaceMorphConfig | None = None):
    """Read-only outer-vs-intaglio body clearance audit.

    This is an engineering shell metric, not a substitute for final ray thickness.
    It exists to stop the old failure mode where the intaglio visibly protruded
    through an undersized embryo before finalization.
    """
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
    finite = np.isfinite(dist)
    values = dist[finite]
    if not len(values):
        return {"status": "NOT_ASSESSED", "reason": "NO_FINITE_CLEARANCE"}
    deficit = np.maximum(float(required_mm) - values, 0.0)

    inside_fraction = None
    try:
        inner_vertices = np.asarray(inner_envelope_mesh.vertices, float)
        if len(inner_vertices) > cfg.shell_audit_sample_limit:
            sample = np.linspace(0, len(inner_vertices) - 1,
                                 cfg.shell_audit_sample_limit, dtype=int)
            inner_vertices = inner_vertices[sample]
        if bool(mesh.is_watertight and mesh.is_winding_consistent):
            inside_fraction = float(np.mean(mesh.contains(inner_vertices)))
    except Exception:  # noqa: BLE001 - audit must never crash generation
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
    """Expand only the outer body until it clears the authoritative intaglio shell.

    Correction direction is a blend of closest-point separation and patient-radial
    expansion.  The radial component prevents an embryo that starts *inside* the
    abutment/intaglio from being pushed through the wrong side.  Near the insertion
    axis the fallback direction is coronal (+Z), which also protects occlusal shell
    thickness above a broad abutment top.
    """
    cfg = config or SpaceMorphConfig()
    if inner_envelope_mesh is None or required_mm is None or float(required_mm) <= 0:
        return mesh.copy(), {
            "status": "NOT_ASSESSED",
            "reason": "INNER_ENVELOPE_OR_THICKNESS_MISSING",
        }

    current = mesh.copy()
    margin_local = _local(margin, frame)
    center_xy = np.median(margin_local[:, :2], axis=0)
    origin, matrix = _basis(frame)
    total_moved = set()
    max_push = 0.0
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

        chosen = active[bad]
        local_vertices = (vertices[chosen] - origin) @ matrix
        local_delta = (vertices[chosen] - cp[bad]) @ matrix

        radial = np.c_[local_vertices[:, :2] - center_xy, np.zeros(len(chosen))]
        radial_norm = np.linalg.norm(radial, axis=1)
        near_axis = radial_norm < 0.35
        radial[~near_axis] /= np.maximum(radial_norm[~near_axis, None], 1e-12)
        radial[near_axis] = np.array([0.0, 0.0, 1.0])

        dnorm = np.linalg.norm(local_delta, axis=1)
        sep = np.zeros_like(local_delta)
        good = dnorm > 1e-8
        sep[good] = local_delta[good] / dnorm[good, None]
        sep[~good] = radial[~good]

        # On the occlusal half of the crown we allow an additional +Z component;
        # on side walls radial growth dominates.  This is deterministic and keeps
        # the actual intaglio immutable.
        coronal = np.c_[np.zeros(len(chosen)), np.zeros(len(chosen)), np.ones(len(chosen))]
        z_fraction = np.clip(height[chosen] / max(float(np.quantile(height[height > 0], .9))
                                                  if np.any(height > 0) else 1.0, 1e-6), 0.0, 1.0)
        desired = 0.55 * sep + 0.35 * radial + 0.10 * z_fraction[:, None] * coronal
        inward = np.einsum("ij,ij->i", desired, radial) < -0.05
        desired[inward] -= 2.0 * np.einsum("ij,ij->i", desired[inward], radial[inward])[:, None] * radial[inward]
        norm = np.linalg.norm(desired, axis=1)
        desired /= np.maximum(norm[:, None], 1e-12)

        push = deficit[bad] * np.clip(weight[chosen], 0.20, 1.0)
        push = np.minimum(push, float(cfg.envelope_max_push_per_pass_mm))
        vertices[chosen] += (desired @ matrix.T) * push[:, None]
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

    audit = audit_inner_envelope_clearance(
        current, inner_envelope_mesh, frame, margin, required_mm, config=cfg,
    )
    return current, {
        "status": "APPLIED" if total_moved else "NO_CORRECTION_NEEDED",
        "method": "ACTUAL_INTAGLIO_CLOSEST_POINT_PLUS_PATIENT_RADIAL_CORONAL_ENVELOPE",
        "required_outer_to_intaglio_mm": float(required_mm),
        "moved_vertices": int(len(total_moved)),
        "max_push_per_pass_mm": float(max_push),
        "passes": passes,
        "audit_after": audit,
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
    """Fit the embryo to patient space before local contact/occlusal refinement.

    A one-sided neighbor grows only that side; the whole crown is never translated
    merely to make contact.  If the actual intaglio preview is available, it is the
    authoritative inner shell.  Otherwise support + cement gap is a conservative
    fallback lower bound.
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
        "algorithm": "SPACE_DRIVEN_EMBRYO_MORPH_V2",
        "template_role": "TOPOLOGY_AND_ANATOMY_EMBRYO_NOT_SIZE_TRUTH",
        "md": {"applied": False},
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

    surfaces = proximal_surfaces or {}
    target = None if proximal_target_mm is None else float(proximal_target_mm)
    mesial_face = _neighbor_contact_x(surfaces["mesial"], frame, "mesial", center_x) if "mesial" in surfaces else None
    distal_face = _neighbor_contact_x(surfaces["distal"], frame, "distal", center_x) if "distal" in surfaces else None
    mesial_contact = (mesial_face - target) if mesial_face is not None and target is not None else None
    distal_contact = (distal_face + target) if distal_face is not None and target is not None else None

    floor_max_x = None if env_max_x is None or shell_required is None else env_max_x + shell_required
    floor_min_x = None if env_min_x is None or shell_required is None else env_min_x - shell_required

    desired_max_x = mesial_contact if mesial_contact is not None else current_max_x
    desired_min_x = distal_contact if distal_contact is not None else current_min_x
    if floor_max_x is not None:
        desired_max_x = max(desired_max_x, floor_max_x)
    if floor_min_x is not None:
        desired_min_x = min(desired_min_x, floor_min_x)

    if desired_min_x < desired_max_x - 0.20:
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
            "inner_floor_min_x_mm": floor_min_x,
            "inner_floor_max_x_mm": floor_max_x,
            "whole_crown_translation_mm": 0.0,
        }
    else:
        report["md"] = {
            "applied": False,
            "status": "CONSTRAINT_CONFLICT",
            "reason": "INNER_SHELL_EXCEEDS_PROXIMAL_SPACE",
            "desired_min_x_mm": float(desired_min_x),
            "desired_max_x_mm": float(desired_max_x),
        }
        report["status"] = "REVIEW_REQUIRED"

    desired_min_y = current_min_y
    desired_max_y = current_max_y
    if shell_required is not None and env_min_y is not None and env_max_y is not None:
        desired_min_y = min(desired_min_y, env_min_y - shell_required)
        desired_max_y = max(desired_max_y, env_max_y + shell_required)
    buccal_scale = _side_scale(current_max_y, desired_max_y, center_y, True,
                                cfg.bl_scale_min, cfg.bl_scale_max)
    lingual_scale = _side_scale(current_min_y, desired_min_y, center_y, False,
                                 cfg.bl_scale_min, cfg.bl_scale_max)
    _apply_asymmetric_axis_warp(local, 1, center_y, lingual_scale, buccal_scale, body_weight)
    report["bl"] = {
        "applied": bool(abs(buccal_scale - 1.0) > 1e-6 or abs(lingual_scale - 1.0) > 1e-6),
        "mode": "INNER_SHELL_MINIMUM_WIDTH",
        "buccal_scale": buccal_scale,
        "lingual_scale": lingual_scale,
        "desired_min_y_mm": float(desired_min_y),
        "desired_max_y_mm": float(desired_max_y),
    }

    # Ensure enough vertical material above the actual intaglio before antagonist
    # fitting.  This was missing in the old path and produced the characteristic
    # OUTER×INTAGLIO ring around the abutment top.
    current_top = float(np.quantile(local[body, 2], .98))
    desired_top = current_top
    if env_top is not None and shell_required is not None:
        desired_top = max(current_top, env_top + shell_required)
    if desired_top > current_top + 1e-6:
        height_now = max(current_top - margin_z, 0.5)
        height_desired = max(desired_top - margin_z, 0.5)
        top_scale = float(np.clip(height_desired / height_now,
                                  cfg.height_scale_min, cfg.height_scale_max))
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
                weight, _height = _body_weight(current, frame, margin, cfg.body_transition_mm)
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
