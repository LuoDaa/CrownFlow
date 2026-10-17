"""CrownFlow 1.0 final completion for B/L fullness and intaglio Margin ownership.

This layer deliberately leaves the proven occlusal/proximal/anatomy owners alone.
It fixes two real-case defects shown by the FDI46 bundle:

* B/L fullness was measured across the whole crown, so broad occlusal cusps made a
  visibly narrow axial body report ``NOT_NEEDED``.  Measure the cervical/mid-body
  band instead and expand only when that patient-derived body is narrow.
* The Margin-clipped intaglio preview still ended ~0.13-0.16 mm away from the
  confirmed Margin because the offset cutter intersected the Margin curtain away
  from the authoritative curve.  Snap only the preview boundary ring onto the
  confirmed 3D Margin polyline; the fitting surface above it remains unchanged.
"""
from __future__ import annotations

import numpy as np

from . import v1_test_geometry_overrides as _width
from .services import intaglio_preview as _intaglio_preview
from .geometry.crown_outer.collar_body_outer import _boundary_loops
from .geometry.nonplanar_crown import MarginCurtain


_ORIGINAL_BL_EXPAND = _width._bounded_bl_expand
_ORIGINAL_INTAGLIO_PREVIEW = _intaglio_preview.build_intaglio_preview

# Mid-body is the clinically visible axial fullness owner.  The occlusal cusp
# envelope must not be allowed to hide a narrow buccal/lingual wall.
BL_BODY_MIN_HEIGHT_MM = 0.80
BL_BODY_MAX_HEIGHT_MM = 3.40
BL_BODY_TOP_FRACTION = 0.58
BL_BODY_Q_LO = 0.05
BL_BODY_Q_HI = 0.95


def _midbody_width(mesh, frame, margin, envelope):
    vertices = np.asarray(mesh.vertices, float)
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    local = (vertices - origin) @ matrix
    center = np.asarray(envelope.center_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    v = (local[:, :2] - center) @ bl
    height = MarginCurtain(margin, frame.z).axial_above(vertices)
    positive = height > max(float(_width.BL_CERVICAL_LOCK_MM), 0.30)
    if int(positive.sum()) < 32:
        return None
    top = max(float(np.quantile(height[positive], 0.98)), 1.0)
    upper = min(BL_BODY_MAX_HEIGHT_MM, max(1.60, top * BL_BODY_TOP_FRACTION))
    band = (height >= BL_BODY_MIN_HEIGHT_MM) & (height <= upper)
    if int(band.sum()) < 32:
        return None
    lo = float(np.quantile(v[band], BL_BODY_Q_LO))
    hi = float(np.quantile(v[band], BL_BODY_Q_HI))
    return {
        "width_mm": hi - lo,
        "vertex_count": int(band.sum()),
        "height_min_mm": BL_BODY_MIN_HEIGHT_MM,
        "height_max_mm": float(upper),
        "quantiles": [BL_BODY_Q_LO, BL_BODY_Q_HI],
    }


def _midbody_owned_bl_expand(mesh, frame, margin, envelope, cfg):
    """Reuse the proven deformation path, but decide need from axial-body width."""
    measurement = _midbody_width(mesh, frame, margin, envelope)
    if measurement is None:
        return _ORIGINAL_BL_EXPAND(mesh, frame, margin, envelope, cfg)

    target_width, target_report = _width._bl_patient_target_width(envelope, cfg)
    current_width = float(measurement["width_mm"])
    if target_width is None or target_width <= current_width + _width.BL_MIN_EXPANSION_MM:
        out = mesh.copy()
        return out, {
            "status": "NOT_NEEDED",
            "current_width_mm": current_width,
            "measurement_scope": "CERVICAL_MIDBODY_AXIAL_BAND",
            "midbody_measurement": measurement,
            **target_report,
        }

    # The established routine has all geometry/backoff/Margin safeguards.  Feed it
    # a temporary body-only scale decision by increasing the target just enough
    # that the whole-crown quantile cannot veto the correction, then restore the
    # patient target in the public report.  The actual displacement still uses the
    # same B/L direction, cervical lock, occlusal-core attenuation and line search.
    out = mesh.copy()
    vertices = np.asarray(out.vertices, float)
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    local = (vertices - origin) @ matrix
    center = np.asarray(envelope.center_xy, float)
    md = np.asarray(envelope.md_axis_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    xy = local[:, :2] - center
    u = xy @ md
    v = xy @ bl
    height = MarginCurtain(margin, frame.z).axial_above(vertices)

    requested_scale = float(target_width / max(current_width, 1e-6))
    scale = min(requested_scale, float(_width.BL_MAX_EXPANSION_SCALE))
    top = max(float(np.quantile(np.maximum(height[height > _width.BL_CERVICAL_LOCK_MM], 0.0), 0.98)), 0.5)
    h_frac = np.clip(np.maximum(height, 0.0) / top, 0.0, 1.2)
    radial_u = np.abs(u) / max(float(np.quantile(np.abs(u[height > _width.BL_CERVICAL_LOCK_MM]), 0.98)), 0.5)
    radial_v = np.abs(v) / max(float(np.quantile(np.abs(v[height > _width.BL_CERVICAL_LOCK_MM]), 0.98)), 0.5)
    radial = np.sqrt(radial_u * radial_u + radial_v * radial_v)
    occlusal_core = (h_frac >= 0.62) & (radial <= 0.72)

    weight = _width._smoothstep01(
        (height - _width.BL_CERVICAL_LOCK_MM) /
        max(_width.BL_FULL_BODY_START_MM - _width.BL_CERVICAL_LOCK_MM, 1e-6)
    )
    weight[height <= _width.BL_CERVICAL_LOCK_MM] = 0.0
    weight[occlusal_core] *= _width.BL_OCCLUSAL_CORE_WEIGHT

    proposed_v = v * scale
    proposed_v = np.clip(
        proposed_v,
        float(envelope.v_min) - _width.BL_ENVELOPE_SAFETY_PAD_MM,
        float(envelope.v_max) + _width.BL_ENVELOPE_SAFETY_PAD_MM,
    )
    v_new = v + weight * (proposed_v - v)
    local[:, :2] = center + u[:, None] * md + v_new[:, None] * bl
    proposed_world = local @ matrix.T + origin
    locked = height <= _width.BL_CERVICAL_LOCK_MM
    proposed_world[locked] = vertices[locked]

    faces = np.asarray(out.faces, int)
    before_vec = _width._face_vectors(vertices, faces)
    before_norm = np.linalg.norm(before_vec, axis=1)
    accepted = None
    accepted_alpha = 0.0
    for alpha in (1.0, 0.75, 0.50, 0.25):
        trial = vertices + alpha * (proposed_world - vertices)
        after_vec = _width._face_vectors(trial, faces)
        after_norm = np.linalg.norm(after_vec, axis=1)
        valid = (before_norm > 1e-10) & (after_norm > 1e-10)
        collapsed = int(np.sum(after_norm <= np.maximum(before_norm * 0.12, 1e-10)))
        flipped = 0
        if np.any(valid):
            cosine = np.einsum("ij,ij->i", before_vec[valid], after_vec[valid]) / (
                before_norm[valid] * after_norm[valid]
            )
            flipped = int(np.sum(cosine <= 0.0))
        if not collapsed and not flipped and np.isfinite(trial).all():
            accepted = trial
            accepted_alpha = float(alpha)
            break
    if accepted is None:
        return out, {
            "status": "REJECTED_KEEP_ORIGINAL",
            "reason": "BL_MIDBODY_EXPANSION_GEOMETRY_GATE",
            "current_width_mm": current_width,
            "measurement_scope": "CERVICAL_MIDBODY_AXIAL_BAND",
            "midbody_measurement": measurement,
            "requested_scale": requested_scale,
            **target_report,
        }

    out.vertices = accepted
    final_measurement = _midbody_width(out, frame, margin, envelope)
    return out, {
        "status": "APPLIED",
        "current_width_mm": current_width,
        "final_width_mm": None if final_measurement is None else float(final_measurement["width_mm"]),
        "measurement_scope": "CERVICAL_MIDBODY_AXIAL_BAND",
        "midbody_measurement": measurement,
        "final_midbody_measurement": final_measurement,
        "requested_scale": requested_scale,
        "applied_scale": float(1.0 + accepted_alpha * (scale - 1.0)),
        "backoff_alpha": accepted_alpha,
        "max_expansion_scale": float(_width.BL_MAX_EXPANSION_SCALE),
        "cervical_lock_mm": float(_width.BL_CERVICAL_LOCK_MM),
        "occlusal_core_weight": float(_width.BL_OCCLUSAL_CORE_WEIGHT),
        **target_report,
    }


def _nearest_points_on_closed_polyline(points, curve):
    points = np.asarray(points, float)
    curve = np.asarray(curve, float)
    a = curve
    b = np.roll(curve, -1, axis=0)
    delta = b - a
    denom = np.einsum("ij,ij->i", delta, delta)
    denom = np.maximum(denom, 1e-18)
    result = np.empty_like(points)
    distance = np.empty(len(points), float)
    for i, point in enumerate(points):
        t = np.clip(((point - a) * delta).sum(axis=1) / denom, 0.0, 1.0)
        projected = a + t[:, None] * delta
        d = np.linalg.norm(projected - point, axis=1)
        j = int(np.argmin(d))
        result[i] = projected[j]
        distance[i] = float(d[j])
    return result, distance


def _snap_preview_boundary_to_margin(preview, margin):
    loops = _boundary_loops(np.asarray(preview.faces, np.int64))
    if len(loops) != 1:
        return preview, {"status": "SKIPPED", "reason": "PREVIEW_NOT_SINGLE_BOUNDARY", "loop_count": len(loops)}
    ids = np.asarray(loops[0], np.int64)
    if len(ids) < 8:
        return preview, {"status": "SKIPPED", "reason": "BOUNDARY_TOO_SHORT", "boundary_vertices": int(len(ids))}
    vertices = np.asarray(preview.vertices, float).copy()
    projected, before = _nearest_points_on_closed_polyline(vertices[ids], margin)
    p95 = float(np.quantile(before, 0.95))
    max_d = float(np.max(before))
    if max_d > 0.50:
        return preview, {"status": "SKIPPED", "reason": "BOUNDARY_TOO_FAR_FROM_CONFIRMED_MARGIN",
                         "before_p95_mm": p95, "before_max_mm": max_d}

    trial = preview.copy()
    vertices[ids] = projected
    trial.vertices = vertices
    area = np.asarray(trial.area_faces, float)
    if not np.isfinite(vertices).all() or np.any(area < 1e-10):
        return preview, {"status": "REJECTED_KEEP_ORIGINAL", "reason": "MARGIN_SNAP_DEGENERATE_FACE",
                         "before_p95_mm": p95, "before_max_mm": max_d}
    _projected_again, after = _nearest_points_on_closed_polyline(np.asarray(trial.vertices)[ids], margin)
    return trial, {
        "status": "SNAPPED_TO_CONFIRMED_MARGIN",
        "boundary_vertices": int(len(ids)),
        "before_median_mm": float(np.median(before)),
        "before_p95_mm": p95,
        "before_max_mm": max_d,
        "after_p95_mm": float(np.quantile(after, 0.95)),
        "after_max_mm": float(np.max(after)),
        "source": "CONFIRMED_3D_MARGIN_POLYLINE",
        "interior_fitting_surface_unchanged": True,
    }


def _margin_owned_intaglio_preview(fitting_support, margin_points, insertion_axis, params: dict):
    preview, blockout, report = _ORIGINAL_INTAGLIO_PREVIEW(
        fitting_support, margin_points, insertion_axis, params
    )
    snapped, snap_report = _snap_preview_boundary_to_margin(preview, np.asarray(margin_points, float))
    report = dict(report or {})
    report["margin_boundary_snap"] = snap_report
    if snap_report.get("status") == "SNAPPED_TO_CONFIRMED_MARGIN":
        report["margin_owned_cervical_boundary"] = True
        report["method"] = "DIRECTIONAL_BLOCKOUT_PLUS_GAP_FIELD_EXACT_CONFIRMED_MARGIN_BOUNDARY"
    return snapped, blockout, report


# Existing runtime call sites resolve these globals dynamically.
_width._bounded_bl_expand = _midbody_owned_bl_expand
_intaglio_preview.build_intaglio_preview = _margin_owned_intaglio_preview
