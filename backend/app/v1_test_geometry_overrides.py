"""CrownFlow 1.0 test-stage geometry overrides.

Keep generation visible and conservative while the patient-space route is being
validated on real cases.  These overrides intentionally tighten M/D expansion,
add a bounded patient-derived B/L body-width correction, and prevent anatomy
fairing from publishing locally inverted/collapsed faces.
"""
from __future__ import annotations

import numpy as np

from .geometry.crown_outer import anatomy_refinement as _anatomy
from .geometry.crown_outer import patient_envelope as _envelope
from .geometry.crown_outer import space_morph as _space
from .geometry.nonplanar_crown import MarginCurtain


_EnvelopeConfig = _envelope.PatientEnvelopeConfig
_SpaceMorphConfig = _space.SpaceMorphConfig
_AnatomyConfig = _anatomy.AnatomyRefineConfig
_refine = _anatomy.refine_crown_anatomy
_morph = _envelope.morph_embryo_to_patient_envelope


# 1.0 patient-width tuning.  The envelope remains a maximum feasible space, but
# posterior crowns that are visibly too slim may expand toward a patient-derived
# B/L target built from the gingival/emergence width and confirmed neighbour
# widths.  This is deliberately modest and never stretches to fill the box.
BL_PATIENT_TARGET_MARGIN_WEIGHT = 0.50
BL_PATIENT_TARGET_NEIGHBOR_WEIGHT = 0.50
BL_MAX_EXPANSION_SCALE = 1.12
BL_MIN_EXPANSION_MM = 0.10
BL_CERVICAL_LOCK_MM = 0.50
BL_FULL_BODY_START_MM = 1.25
BL_OCCLUSAL_CORE_WEIGHT = 0.38


def _patient_envelope_config(*args, **kwargs):
    # M/D belongs to the available interproximal space.  The old 3 mm halo cap
    # was intentionally generous and can create an over-wide molar on broad
    # automatic neighbour patches.  Keep enough room for a molar but stop the
    # halo from becoming crown-size truth.
    kwargs.setdefault("neighbor_md_extra_from_margin_mm", 2.10)
    kwargs.setdefault("neighbor_inner_quantile", 0.08)
    kwargs.setdefault("max_half_span_mm", 6.25)
    return _EnvelopeConfig(*args, **kwargs)


def _space_morph_config(*args, **kwargs):
    # 1.0 test stage: material-floor correction may improve clearance, but it
    # must not perform multi-millimetre broad shell deformation.
    kwargs.setdefault("envelope_iterations", 2)
    kwargs.setdefault("envelope_max_push_per_pass_mm", 0.25)
    return _SpaceMorphConfig(*args, **kwargs)


def _safe_anatomy_config(*args, **kwargs):
    # Anatomy should refine an already valid patient-space crown, not reshape it.
    kwargs.setdefault("anatomy_amplitude_mm", 0.12)
    kwargs.setdefault("fairing_iterations", 2)
    kwargs.setdefault("fairing_lambda", 0.14)
    kwargs.setdefault("body_weight", 0.30)
    kwargs.setdefault("occlusal_weight", 0.10)
    kwargs.setdefault("max_vertex_move_mm", 0.08)
    return _AnatomyConfig(*args, **kwargs)


def _face_vectors(vertices, faces):
    tri = np.asarray(vertices, float)[np.asarray(faces, int)]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def _smoothstep01(value):
    x = np.clip(np.asarray(value, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bl_patient_target_width(envelope, cfg):
    """Patient-derived B/L target = mean(gingival/emergence width, neighbour width)."""
    sources = dict(getattr(envelope, "sources", {}) or {})
    margin_bounds = dict(sources.get("margin_bounds") or {})
    neighbor_bounds = dict(sources.get("neighbor_bl_capped_bounds") or {})
    try:
        margin_width = float(margin_bounds["v_max"]) - float(margin_bounds["v_min"])
    except (KeyError, TypeError, ValueError):
        return None, {"reason": "MARGIN_BL_WIDTH_UNAVAILABLE"}
    # A bare finish-line width is not the desired crown-body width.  Add the
    # emergence allowance on both sides before blending it with neighbouring
    # clinical crown width.
    gingival_body_width = margin_width + 2.0 * float(getattr(cfg, "bl_margin_emergence_mm", 1.60))
    try:
        nv_lo = neighbor_bounds.get("v_min")
        nv_hi = neighbor_bounds.get("v_max")
        neighbor_width = None if nv_lo is None or nv_hi is None else float(nv_hi) - float(nv_lo)
    except (TypeError, ValueError):
        neighbor_width = None
    if neighbor_width is not None and neighbor_width > 0.5:
        target = (
            BL_PATIENT_TARGET_MARGIN_WEIGHT * gingival_body_width
            + BL_PATIENT_TARGET_NEIGHBOR_WEIGHT * neighbor_width
        )
        source = "MEAN_GINGIVAL_EMERGENCE_AND_CONFIRMED_NEIGHBOR_BL"
    else:
        target = gingival_body_width
        source = "GINGIVAL_EMERGENCE_ONLY"
    # The six-face envelope remains the hard maximum.  This target only asks for
    # a more natural body width inside that feasible space.
    target = min(float(target), float(envelope.bl_width_mm))
    return target, {
        "source": source,
        "margin_width_mm": float(margin_width),
        "gingival_emergence_width_mm": float(gingival_body_width),
        "neighbor_width_mm": None if neighbor_width is None else float(neighbor_width),
        "target_width_mm": float(target),
        "margin_weight": BL_PATIENT_TARGET_MARGIN_WEIGHT,
        "neighbor_weight": BL_PATIENT_TARGET_NEIGHBOR_WEIGHT,
    }


def _bounded_bl_expand(mesh, frame, margin, envelope, cfg):
    """Expand a slim B/L body toward the patient target without touching Margin."""
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
    body = height > max(BL_CERVICAL_LOCK_MM, 0.30)
    if int(body.sum()) < 16:
        return out, {"status": "SKIPPED", "reason": "INSUFFICIENT_BODY_VERTICES"}

    current_lo = float(np.quantile(v[body], 0.02))
    current_hi = float(np.quantile(v[body], 0.98))
    current_width = current_hi - current_lo
    target_width, target_report = _bl_patient_target_width(envelope, cfg)
    if target_width is None or target_width <= current_width + BL_MIN_EXPANSION_MM:
        return out, {
            "status": "NOT_NEEDED",
            "current_width_mm": float(current_width),
            **target_report,
        }

    requested_scale = float(target_width / max(current_width, 1e-6))
    scale = min(requested_scale, BL_MAX_EXPANSION_SCALE)

    top = max(float(np.quantile(np.maximum(height[body], 0.0), 0.98)), 0.5)
    h_frac = np.clip(np.maximum(height, 0.0) / top, 0.0, 1.2)
    radial_u = np.abs(u) / max(float(np.quantile(np.abs(u[body]), 0.98)), 0.5)
    radial_v = np.abs(v) / max(float(np.quantile(np.abs(v[body]), 0.98)), 0.5)
    radial = np.sqrt(radial_u * radial_u + radial_v * radial_v)
    occlusal_core = (h_frac >= 0.62) & (radial <= 0.72)

    # Root/Margin stays immutable.  Axial walls receive the full correction;
    # the occlusal morphology core receives only a small fraction so cusp/fossa
    # anatomy is not stretched flat.
    weight = _smoothstep01((height - BL_CERVICAL_LOCK_MM) /
                           max(BL_FULL_BODY_START_MM - BL_CERVICAL_LOCK_MM, 1e-6))
    weight[height <= BL_CERVICAL_LOCK_MM] = 0.0
    weight[occlusal_core] *= BL_OCCLUSAL_CORE_WEIGHT

    proposed_v = v * scale
    # Never cross patient-owned B/L maximum bounds.
    proposed_v = np.clip(proposed_v, float(envelope.v_min), float(envelope.v_max))
    v_new = v + weight * (proposed_v - v)
    local[:, :2] = center + u[:, None] * md + v_new[:, None] * bl
    proposed_world = local @ matrix.T + origin
    proposed_world[height <= BL_CERVICAL_LOCK_MM] = vertices[height <= BL_CERVICAL_LOCK_MM]

    faces = np.asarray(out.faces, int)
    before_vec = _face_vectors(vertices, faces)
    before_norm = np.linalg.norm(before_vec, axis=1)
    accepted = None
    accepted_alpha = 0.0
    for alpha in (1.0, 0.75, 0.50, 0.25):
        trial = vertices + alpha * (proposed_world - vertices)
        after_vec = _face_vectors(trial, faces)
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
            accepted_alpha = alpha
            break
    if accepted is None:
        return out, {
            "status": "REJECTED_KEEP_ORIGINAL",
            "reason": "BL_EXPANSION_GEOMETRY_GATE",
            "current_width_mm": float(current_width),
            "requested_scale": requested_scale,
            **target_report,
        }

    out.vertices = accepted
    final_local = (np.asarray(out.vertices, float) - origin) @ matrix
    final_v = (final_local[:, :2] - center) @ bl
    final_width = float(np.quantile(final_v[body], 0.98) - np.quantile(final_v[body], 0.02))
    return out, {
        "status": "APPLIED",
        "current_width_mm": float(current_width),
        "final_width_mm": final_width,
        "requested_scale": requested_scale,
        "applied_scale": float(1.0 + accepted_alpha * (scale - 1.0)),
        "backoff_alpha": float(accepted_alpha),
        "max_expansion_scale": BL_MAX_EXPANSION_SCALE,
        "cervical_lock_mm": BL_CERVICAL_LOCK_MM,
        "occlusal_core_weight": BL_OCCLUSAL_CORE_WEIGHT,
        **target_report,
    }


def _patient_width_morph(mesh, frame, margin, envelope, *, config=None):
    """Original max-envelope morph plus bounded patient-derived B/L expansion."""
    cfg = config or _patient_envelope_config()
    base, report = _morph(mesh, frame, margin, envelope, config=cfg)
    expanded, bl_report = _bounded_bl_expand(base, frame, margin, envelope, cfg)
    merged = dict(report or {})
    merged["bl_patient_target_expansion"] = bl_report
    return expanded, merged


def _safe_refine_crown_anatomy(mesh, frame, margin, *, fdi: int, config=None):
    cfg = config or _safe_anatomy_config()
    refined, report = _refine(mesh, frame, margin, fdi=fdi, config=cfg)

    before_v = np.asarray(mesh.vertices, float)
    after_v = np.asarray(refined.vertices, float)
    faces = np.asarray(mesh.faces, int)
    if before_v.shape != after_v.shape or np.asarray(refined.faces).shape != faces.shape:
        fallback = mesh.copy()
        return fallback, {
            **dict(report or {}),
            "status": "REVIEW_REQUIRED",
            "reason": "ANATOMY_TOPOLOGY_CHANGED_ROLLBACK",
            "published_geometry": "PRE_ANATOMY",
        }

    b = _face_vectors(before_v, faces)
    a = _face_vectors(after_v, faces)
    bn = np.linalg.norm(b, axis=1)
    an = np.linalg.norm(a, axis=1)
    valid = (bn > 1e-10) & (an > 1e-10)
    flip_count = 0
    collapsed_count = int(np.sum(an <= np.maximum(bn * 0.12, 1e-10)))
    if np.any(valid):
        cosine = np.einsum("ij,ij->i", b[valid], a[valid]) / (bn[valid] * an[valid])
        flip_count = int(np.sum(cosine <= 0.0))

    # Any inverted face can appear as a visible hole with back-face culling.
    # For the test phase, preserve the pre-anatomy crown rather than publishing
    # a damaged-looking refinement.
    if flip_count or collapsed_count:
        fallback = mesh.copy()
        return fallback, {
            **dict(report or {}),
            "status": "REVIEW_REQUIRED",
            "reason": "ANATOMY_FACE_INVERSION_ROLLBACK",
            "face_flip_count": flip_count,
            "collapsed_face_count": collapsed_count,
            "published_geometry": "PRE_ANATOMY",
        }

    report = dict(report or {})
    report.update({
        "algorithm": "REGION_WEIGHTED_ANATOMY_FAIRING_V1_SAFE",
        "face_flip_count": 0,
        "collapsed_face_count": 0,
        "published_geometry": "REFINED",
    })
    return refined, report


# Patch the module globals before API/service modules import these symbols.
_envelope.PatientEnvelopeConfig = _patient_envelope_config
_envelope.morph_embryo_to_patient_envelope = _patient_width_morph
_space.PatientEnvelopeConfig = _patient_envelope_config
_space.SpaceMorphConfig = _space_morph_config
_anatomy.AnatomyRefineConfig = _safe_anatomy_config
_anatomy.refine_crown_anatomy = _safe_refine_crown_anatomy
