"""CrownFlow 1.0 patient-owned outer/inner completion.

This patch intentionally keeps the already-working proximal and occlusal solvers
unchanged.  It addresses two last-mile real-case failures seen on posterior crowns:

1. The crown can remain visually too narrow in the patient B/L direction even when
   the template itself is anatomically broad.  The correction below is strictly
   anisotropic in PatientFrame B/L: M/D coordinates are byte-for-byte unchanged,
   the confirmed cervical root is locked, the middle axial body receives the full
   bilateral expansion, and the occlusal core follows only partially.
2. The generated intaglio must be an OPEN fitting surface derived from the current
   prepared abutment, insertion-direction blockout and cement-gap field.  The
   artificial Boolean cutter bottom is never returned as the design inner.  The
   cervical boundary is finally snapped onto the confirmed 3D Margin so outer and
   inner can share one authoritative seam.

The implementation follows the same separation of concerns used by dental CAD:
outer anatomy is a deformable template prior; the intaglio is patient-owned by the
preparation, insertion path and spacing parameters.
"""
from __future__ import annotations

import numpy as np

from . import v1_margin_bl_completion as _completion
from . import v1_test_geometry_overrides as _width
from .geometry.crown_outer.collar_body_outer import _boundary_loops
from .geometry.intaglio import BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout
from .geometry.nonplanar_crown import MarginCurtain, _single_open_surface_above_margin
from .services import intaglio_preview as _intaglio_preview


# ------------------------------ stronger patient B/L owner -------------------
# These values are intentionally larger than the earlier conservative preview
# tuning.  They apply only on the B/L coordinate, never M/D or insertion height.
BL_STRONG_MAX_SCALE = 1.35
BL_STRONG_CERVICAL_LOCK_MM = 0.35
BL_STRONG_FULL_BODY_START_MM = 0.80
BL_STRONG_OCCLUSAL_CORE_WEIGHT = 0.55
BL_STRONG_ENVELOPE_PAD_MM = 1.25
BL_POSTERIOR_MD_TRIGGER_MM = 8.0
BL_POSTERIOR_MIN_BL_TO_MD = 0.95
BL_PATIENT_TARGET_BONUS_MM = 0.55


def _body_metrics(mesh, frame, margin, envelope):
    vertices = np.asarray(mesh.vertices, float)
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
    positive = height > BL_STRONG_CERVICAL_LOCK_MM
    if int(positive.sum()) < 32:
        return None
    top = max(float(np.quantile(height[positive], 0.98)), 1.0)
    upper = min(3.8, max(1.8, top * 0.62))
    band = (height >= 0.70) & (height <= upper)
    if int(band.sum()) < 32:
        return None
    return {
        "bl_width_mm": float(np.quantile(v[band], .95) - np.quantile(v[band], .05)),
        "md_width_mm": float(np.quantile(u[band], .95) - np.quantile(u[band], .05)),
        "band": band,
        "height": height,
        "u": u,
        "v": v,
        "local": local,
        "origin": origin,
        "matrix": matrix,
        "center": center,
        "md": md,
        "bl": bl,
        "upper_mm": upper,
        "vertex_count": int(band.sum()),
    }


def _strong_bilateral_bl_expand(mesh, frame, margin, envelope, cfg):
    """Expand only PatientFrame B/L, symmetrically about the confirmed centre.

    This is deliberately not a global crown scale.  It is the geometric analogue
    of pushing the buccal and lingual walls apart while leaving M/D contact and the
    insertion axis untouched.
    """
    metrics = _body_metrics(mesh, frame, margin, envelope)
    if metrics is None:
        return _completion._midbody_owned_bl_expand(mesh, frame, margin, envelope, cfg)

    patient_target, target_report = _width._bl_patient_target_width(envelope, cfg)
    current_bl = float(metrics["bl_width_mm"])
    md_width = float(metrics["md_width_mm"])

    candidates = []
    if patient_target is not None:
        candidates.append(float(patient_target) + BL_PATIENT_TARGET_BONUS_MM)
    posterior_ratio_floor = None
    # Broad M/D posterior bodies need comparable B/L fullness.  The trigger avoids
    # applying a molar-like ratio floor to narrow anterior/premolar geometries.
    if md_width >= BL_POSTERIOR_MD_TRIGGER_MM:
        posterior_ratio_floor = md_width * BL_POSTERIOR_MIN_BL_TO_MD
        candidates.append(posterior_ratio_floor)
    if not candidates:
        return _completion._midbody_owned_bl_expand(mesh, frame, margin, envelope, cfg)

    target_bl = max(candidates)
    if target_bl <= current_bl + 0.08:
        out = mesh.copy()
        return out, {
            "status": "NOT_NEEDED",
            "mode": "PATIENT_FRAME_BL_ONLY_BILATERAL_V2",
            "current_width_mm": current_bl,
            "md_width_mm": md_width,
            "target_width_mm": target_bl,
            "posterior_ratio_floor_mm": posterior_ratio_floor,
            "patient_target_bonus_mm": BL_PATIENT_TARGET_BONUS_MM,
            "measurement_scope": "CERVICAL_MIDBODY_AXIAL_BAND",
            "md_coordinate_changed": False,
            **target_report,
        }

    requested_scale = target_bl / max(current_bl, 1e-6)
    scale = min(requested_scale, BL_STRONG_MAX_SCALE)

    out = mesh.copy()
    vertices = np.asarray(out.vertices, float)
    local = np.asarray(metrics["local"], float).copy()
    u = np.asarray(metrics["u"], float)
    v = np.asarray(metrics["v"], float)
    height = np.asarray(metrics["height"], float)
    center = np.asarray(metrics["center"], float)
    md = np.asarray(metrics["md"], float)
    bl = np.asarray(metrics["bl"], float)

    body = height > BL_STRONG_CERVICAL_LOCK_MM
    top = max(float(np.quantile(np.maximum(height[body], 0.0), 0.98)), 0.5)
    h_frac = np.clip(np.maximum(height, 0.0) / top, 0.0, 1.2)
    radial_u = np.abs(u) / max(float(np.quantile(np.abs(u[body]), .98)), .5)
    radial_v = np.abs(v) / max(float(np.quantile(np.abs(v[body]), .98)), .5)
    radial = np.sqrt(radial_u * radial_u + radial_v * radial_v)
    occlusal_core = (h_frac >= .62) & (radial <= .72)

    weight = _width._smoothstep01(
        (height - BL_STRONG_CERVICAL_LOCK_MM) /
        max(BL_STRONG_FULL_BODY_START_MM - BL_STRONG_CERVICAL_LOCK_MM, 1e-6)
    )
    weight[height <= BL_STRONG_CERVICAL_LOCK_MM] = 0.0
    weight[occlusal_core] *= BL_STRONG_OCCLUSAL_CORE_WEIGHT

    # Pure bilateral B/L deformation: u is never modified.
    proposed_v = v * scale
    proposed_v = np.clip(
        proposed_v,
        float(envelope.v_min) - BL_STRONG_ENVELOPE_PAD_MM,
        float(envelope.v_max) + BL_STRONG_ENVELOPE_PAD_MM,
    )
    v_new = v + weight * (proposed_v - v)
    local[:, :2] = center + u[:, None] * md + v_new[:, None] * bl
    proposed_world = local @ metrics["matrix"].T + metrics["origin"]
    proposed_world[height <= BL_STRONG_CERVICAL_LOCK_MM] = vertices[height <= BL_STRONG_CERVICAL_LOCK_MM]

    faces = np.asarray(out.faces, int)
    before_vec = _width._face_vectors(vertices, faces)
    before_norm = np.linalg.norm(before_vec, axis=1)
    accepted = None
    accepted_alpha = 0.0
    for alpha in (1.0, .80, .60, .40, .25):
        trial = vertices + alpha * (proposed_world - vertices)
        after_vec = _width._face_vectors(trial, faces)
        after_norm = np.linalg.norm(after_vec, axis=1)
        valid = (before_norm > 1e-10) & (after_norm > 1e-10)
        collapsed = int(np.sum(after_norm <= np.maximum(before_norm * .10, 1e-10)))
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
            "reason": "STRONG_BL_BILATERAL_GEOMETRY_GATE",
            "mode": "PATIENT_FRAME_BL_ONLY_BILATERAL_V2",
            "current_width_mm": current_bl,
            "md_width_mm": md_width,
            "target_width_mm": target_bl,
            "requested_scale": requested_scale,
            "md_coordinate_changed": False,
            **target_report,
        }

    out.vertices = accepted
    final_metrics = _body_metrics(out, frame, margin, envelope)
    return out, {
        "status": "APPLIED",
        "mode": "PATIENT_FRAME_BL_ONLY_BILATERAL_V2",
        "current_width_mm": current_bl,
        "final_width_mm": None if final_metrics is None else float(final_metrics["bl_width_mm"]),
        "md_width_mm": md_width,
        "target_width_mm": target_bl,
        "posterior_ratio_floor_mm": posterior_ratio_floor,
        "requested_scale": requested_scale,
        "applied_scale": float(1.0 + accepted_alpha * (scale - 1.0)),
        "max_expansion_scale": BL_STRONG_MAX_SCALE,
        "cervical_lock_mm": BL_STRONG_CERVICAL_LOCK_MM,
        "full_body_start_mm": BL_STRONG_FULL_BODY_START_MM,
        "occlusal_core_weight": BL_STRONG_OCCLUSAL_CORE_WEIGHT,
        "patient_target_bonus_mm": BL_PATIENT_TARGET_BONUS_MM,
        "measurement_scope": "CERVICAL_MIDBODY_AXIAL_BAND",
        "md_coordinate_changed": False,
        "proximal_contact_axis_preserved": True,
        **target_report,
    }


# ------------------------------ true open patient intaglio -------------------
def _exact_margin_boundary(surface, margin):
    loops = _boundary_loops(np.asarray(surface.faces, np.int64))
    if len(loops) != 1:
        return None, {"status": "FAILED", "reason": "INTAGLIO_NOT_SINGLE_OPEN_BOUNDARY", "loop_count": len(loops)}
    ids = np.asarray(loops[0], np.int64)
    if len(ids) < 8:
        return None, {"status": "FAILED", "reason": "INTAGLIO_BOUNDARY_TOO_SHORT", "boundary_vertices": int(len(ids))}
    vertices = np.asarray(surface.vertices, float).copy()
    projected, before = _completion._nearest_points_on_closed_polyline(vertices[ids], margin)
    vertices[ids] = projected
    out = surface.copy()
    out.vertices = vertices
    area = np.asarray(out.area_faces, float)
    if not np.isfinite(vertices).all() or np.any(area <= 1e-12):
        return None, {"status": "FAILED", "reason": "MARGIN_SNAP_DEGENERATE"}
    _p, after = _completion._nearest_points_on_closed_polyline(np.asarray(out.vertices)[ids], margin)
    return out, {
        "status": "EXACT_CONFIRMED_MARGIN",
        "boundary_vertices": int(len(ids)),
        "before_p95_mm": float(np.quantile(before, .95)),
        "before_max_mm": float(np.max(before)),
        "after_p95_mm": float(np.quantile(after, .95)),
        "after_max_mm": float(np.max(after)),
    }


def _fallback_open_intaglio_from_blockout(blockout, fitting_support, margin, axis, params):
    """Build an open fitting surface without ever exposing the cutter bottom.

    Primary generation uses the full variable-gap cutter.  This fallback only runs
    when that cutter cannot be cleanly clipped.  It starts from the insertion-path
    blockout surface, applies a smooth patient gap away from the Margin, and keeps
    the boundary itself at zero gap.
    """
    curtain = MarginCurtain(margin, axis)
    open_blockout, extraction = _single_open_surface_above_margin(
        blockout.blockout_mesh, curtain, 0.0, "INTAGLIO_BLOCKOUT_FALLBACK", invert=False
    )
    open_blockout = open_blockout.copy()
    vertices = np.asarray(open_blockout.vertices, float).copy()
    loops = _boundary_loops(np.asarray(open_blockout.faces, np.int64))
    if len(loops) != 1:
        raise ValueError("blockout fallback 未得到唯一 Margin 开口")
    boundary_ids = np.asarray(loops[0], np.int64)
    _nearest, distance = _completion._nearest_points_on_closed_polyline(vertices, margin)

    explicit_start = params.get("cement_gap_start_mm")
    if explicit_start is not None and np.isfinite(float(explicit_start)) and float(explicit_start) > 1e-6:
        transition = float(explicit_start)
        transition_source = "EXPLICIT_CEMENT_GAP_START"
    else:
        edges = np.asarray(open_blockout.edges_unique_length, float)
        edges = edges[np.isfinite(edges) & (edges > 1e-8)]
        transition = float(np.clip((np.median(edges) if len(edges) else .08) * 4.0, .18, .45))
        transition_source = "MESH_RESOLUTION_DERIVED_MARGIN_BLEND"

    gap = float(params.get("cement_gap_mm") or 0.0)
    weight = _width._smoothstep01(distance / max(transition, 1e-6))
    weight[boundary_ids] = 0.0
    normals = np.asarray(open_blockout.vertex_normals, float)
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        raise ValueError("blockout fallback 顶点法向无效")
    vertices = vertices + normals * (gap * weight)[:, None]
    open_blockout.vertices = vertices
    open_blockout.invert()  # material-facing orientation for intaglio
    exact, boundary = _exact_margin_boundary(open_blockout, margin)
    if exact is None:
        raise ValueError(f"blockout fallback Margin 对齐失败: {boundary}")
    return exact, {
        "status": "READY_FALLBACK_OPEN_SURFACE",
        "method": "INSERTION_BLOCKOUT_OPEN_SURFACE_PLUS_SMOOTH_GAP_EXACT_MARGIN",
        "gap_mm": gap,
        "margin_blend_mm": transition,
        "margin_blend_source": transition_source,
        "surface_extraction": extraction,
        "margin_boundary": boundary,
        "artificial_bottom_closure_rendered": False,
    }


def _patient_owned_intaglio_preview(fitting_support, margin_points, insertion_axis, params: dict):
    margin = np.asarray(margin_points, float)
    axis = np.asarray(insertion_axis, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 8:
        raise ValueError("Intaglio 需要完整 confirmed 3D Margin")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("Intaglio 需要有效插入方向")
    if params.get("cement_gap_mm") is None:
        raise ValueError("Intaglio 缺少 cement_gap_mm")

    policy = BlockoutPolicy(
        machine_axes=int(params.get("machine_axes") or 0),
        additional_blockout_after_gap=None,
        untouchable_zone_mm=float(params.get("untouchable_zone_mm") or 0.0),
        provenance={"source": "confirmed_design_profile", "purpose": "patient_owned_intaglio"},
    )
    blockout = build_blockout(fitting_support, margin, axis, policy)
    gap_params = GapFieldParams(
        cement_gap_mm=float(params["cement_gap_mm"]),
        cement_gap_start_mm=params.get("cement_gap_start_mm"),
        cement_gap_top_mm=params.get("cement_gap_top_mm"),
        additional_spacing_xyz_mm=None,
        additional_spacing_z_on_top_mm=None,
        provenance={"source": "confirmed_design_profile", "purpose": "patient_owned_intaglio"},
    )
    closed_cutter = apply_gap_field(blockout, fitting_support, margin, axis, gap_params)
    curtain = MarginCurtain(margin, axis)

    try:
        preview, extraction = _single_open_surface_above_margin(
            closed_cutter, curtain, 0.0, "INTAGLIO", invert=True
        )
        exact, boundary = _exact_margin_boundary(preview, margin)
        if exact is None:
            raise ValueError(f"Margin boundary snap failed: {boundary}")
        report = {
            "status": "READY",
            "method": "DIRECTIONAL_BLOCKOUT_VARIABLE_GAP_OPEN_INTAGLIO_EXACT_MARGIN",
            "blockout": blockout.as_dict(),
            "cement_gap_mm": float(params["cement_gap_mm"]),
            "cement_gap_start_mm": params.get("cement_gap_start_mm"),
            "cement_gap_top_mm": params.get("cement_gap_top_mm"),
            "surface_extraction": extraction,
            "margin_boundary": boundary,
            "margin_owned_cervical_boundary": True,
            "generated_from_prepared_abutment": True,
            "artificial_bottom_closure_rendered": False,
            "hollow_crown_inner_surface": True,
        }
        return exact, blockout, report
    except Exception as primary_exc:
        fallback, fallback_report = _fallback_open_intaglio_from_blockout(
            blockout, fitting_support, margin, axis, params
        )
        fallback_report.update({
            "primary_gap_cutter_clip_error": str(primary_exc),
            "blockout": blockout.as_dict(),
            "generated_from_prepared_abutment": True,
            "margin_owned_cervical_boundary": True,
            "hollow_crown_inner_surface": True,
        })
        return fallback, blockout, fallback_report


# Activate after the prior conservative patches.  Existing runtime call sites
# resolve these module globals dynamically, so no pipeline rewrite is required.
_width.BL_MAX_EXPANSION_SCALE = BL_STRONG_MAX_SCALE
_width.BL_CERVICAL_LOCK_MM = BL_STRONG_CERVICAL_LOCK_MM
_width.BL_FULL_BODY_START_MM = BL_STRONG_FULL_BODY_START_MM
_width.BL_OCCLUSAL_CORE_WEIGHT = BL_STRONG_OCCLUSAL_CORE_WEIGHT
_width.BL_ENVELOPE_SAFETY_PAD_MM = BL_STRONG_ENVELOPE_PAD_MM
_width._bounded_bl_expand = _strong_bilateral_bl_expand
_completion._width._bounded_bl_expand = _strong_bilateral_bl_expand
_intaglio_preview.build_intaglio_preview = _patient_owned_intaglio_preview
_completion._intaglio_preview.build_intaglio_preview = _patient_owned_intaglio_preview
