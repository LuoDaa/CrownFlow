"""Patient-owned feasible crown envelope.

The envelope is a *maximum feasible space*, not a requested final tooth size.
Margin, adjacent teeth, the inner shell and antagonist define where a crown may
exist.  Library anatomy stays unchanged while it is already inside that space;
only dimensions that exceed a patient-owned bound are compressed.

This distinction is important: fitting every template exactly to the six bounds
turns the envelope into a box-shaped morphology target and can flatten posterior
crowns.  Local proximal and occlusal solvers own contact details after this broad
feasibility pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..nonplanar_crown import MarginCurtain


@dataclass(frozen=True)
class PatientEnvelopeConfig:
    emergence_transition_mm: float = 1.60
    cervical_immutable_mm: float = 0.50
    absent_side_emergence_mm: float = 1.80
    bl_margin_emergence_mm: float = 1.60
    neighbor_inner_quantile: float = 0.10
    neighbor_width_low_quantile: float = 0.12
    neighbor_width_high_quantile: float = 0.88
    neighbor_md_extra_from_margin_mm: float = 3.00
    neighbor_bl_extra_from_margin_mm: float = 2.20
    antagonist_low_quantile: float = 0.08
    antagonist_xy_pad_mm: float = 0.80
    max_half_span_mm: float = 7.00
    min_half_span_mm: float = 1.20
    min_coronal_height_mm: float = 1.00
    # Broad feasibility deformation must preserve tooth morphology.  Large case
    # mismatches are left for the local surface solvers/review rather than being
    # solved by one destructive global squash.
    md_min_scale_per_pass: float = 0.70
    bl_min_scale_per_pass: float = 0.80
    height_min_scale_per_pass: float = 0.72
    bl_core_preservation: float = 0.38


@dataclass(frozen=True)
class PatientEnvelope:
    center_xy: tuple[float, float]
    md_axis_xy: tuple[float, float]
    bl_axis_xy: tuple[float, float]
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    margin_z: float
    target_top_z: float | None
    sources: dict

    @property
    def md_width_mm(self) -> float:
        return float(self.u_max - self.u_min)

    @property
    def bl_width_mm(self) -> float:
        return float(self.v_max - self.v_min)

    @property
    def height_mm(self) -> float | None:
        return None if self.target_top_z is None else float(self.target_top_z - self.margin_z)

    def as_report(self) -> dict:
        return {
            "mode": "PATIENT_6_FACE_MAX_FEASIBLE_ENVELOPE_V2",
            "semantics": "MAX_FEASIBLE_BOUNDS_NOT_TARGET_TOOTH_DIMENSIONS",
            "global_dimensions_source": "PATIENT_GEOMETRY_ONLY",
            "template_size_authoritative": False,
            "center_xy": list(self.center_xy),
            "md_axis_xy": list(self.md_axis_xy),
            "bl_axis_xy": list(self.bl_axis_xy),
            "u_min": float(self.u_min), "u_max": float(self.u_max),
            "v_min": float(self.v_min), "v_max": float(self.v_max),
            "md_width_mm": self.md_width_mm, "bl_width_mm": self.bl_width_mm,
            "margin_z": float(self.margin_z),
            "target_top_z": None if self.target_top_z is None else float(self.target_top_z),
            "height_mm": self.height_mm,
            "sources": dict(self.sources),
        }


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


def _world(points, frame):
    origin, matrix = _basis(frame)
    return np.asarray(points, float) @ matrix.T + origin


def _smoothstep01(value):
    x = np.clip(np.asarray(value, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _surface_local(surface, frame):
    mesh = getattr(surface, "mesh", surface)
    if mesh is None or not hasattr(mesh, "vertices"):
        return np.zeros((0, 3), float)
    pts = _local(mesh.vertices, frame)
    return pts[np.isfinite(pts).all(axis=1)]


def _near_direction(local, center_xy):
    if len(local) < 3:
        return None
    radial = local[:, :2] - np.asarray(center_xy, float)
    radius = np.linalg.norm(radial, axis=1)
    order = np.argsort(radius)
    take = max(12, min(len(order), int(np.ceil(len(order) * .20))))
    direction = np.median(radial[order[:take]], axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        direction = np.mean(radial[order[:take]], axis=0)
        norm = float(np.linalg.norm(direction))
    return None if norm <= 1e-8 else direction / norm


def _md_axis(proximal_surfaces, frame, center_xy):
    refs = []
    for label, surface in (proximal_surfaces or {}).items():
        local = _surface_local(surface, frame)
        direction = _near_direction(local, center_xy)
        if direction is not None:
            refs.append((str(label), direction, local))
    # Confirmed PatientFrame owns anatomy axes.  Broad neighbour halos may own
    # bounds, but never rotate the crown coordinate system.
    return np.array([1.0, 0.0]), refs, "CONFIRMED_PATIENT_FRAME_X"


def _project_uv(local, center_xy, md_axis, bl_axis):
    xy = np.asarray(local, float)[:, :2] - np.asarray(center_xy, float)
    return xy @ np.asarray(md_axis, float), xy @ np.asarray(bl_axis, float)


def _clamp_side(value, sign, cfg):
    if sign > 0:
        return float(np.clip(value, cfg.min_half_span_mm, cfg.max_half_span_mm))
    return float(np.clip(value, -cfg.max_half_span_mm, -cfg.min_half_span_mm))


def _physical_side_sign(pu, direction, md_axis):
    values = np.asarray(pu, float)
    finite = values[np.isfinite(values)]
    center_u = float(np.median(finite)) if len(finite) else 0.0
    direction_u = float(np.asarray(direction, float) @ np.asarray(md_axis, float))
    if abs(center_u) > 0.10:
        return (1 if center_u > 0.0 else -1), center_u, direction_u, "PATCH_POSITION"
    if abs(direction_u) > 1e-6:
        return (1 if direction_u > 0.0 else -1), center_u, direction_u, "NEAR_PATCH_DIRECTION"
    return 1, center_u, direction_u, "DEGENERATE_TIE_BREAK_POSITIVE"


def build_patient_envelope(*, frame, margin, inner_envelope_mesh=None,
                           minimum_thickness_mm=None, proximal_surfaces=None,
                           proximal_target_mm=0.0, antagonist_surface=None,
                           occlusal_target_mm=None,
                           config: PatientEnvelopeConfig | None = None):
    """Build patient-owned maximum feasible bounds for a crown."""
    cfg = config or PatientEnvelopeConfig()
    margin_local = _local(margin, frame)
    margin_local = margin_local[np.isfinite(margin_local).all(axis=1)]
    if len(margin_local) < 8:
        raise ValueError("PatientEnvelope 需要有效的完整 Margin")

    center_xy = np.median(margin_local[:, :2], axis=0)
    margin_z = float(np.median(margin_local[:, 2]))
    md_axis, refs, axis_source = _md_axis(proximal_surfaces, frame, center_xy)
    md_axis = np.asarray(md_axis, float)
    md_axis /= max(float(np.linalg.norm(md_axis)), 1e-12)
    bl_axis = np.array([-md_axis[1], md_axis[0]], float)
    mu, mv = _project_uv(margin_local, center_xy, md_axis, bl_axis)
    margin_u_lo, margin_u_hi = float(np.quantile(mu, .02)), float(np.quantile(mu, .98))
    margin_v_lo, margin_v_hi = float(np.quantile(mv, .02)), float(np.quantile(mv, .98))

    thickness = (float(minimum_thickness_mm)
                 if minimum_thickness_mm is not None and np.isfinite(minimum_thickness_mm)
                 and float(minimum_thickness_mm) > 0 else 0.0)
    shell_u_lo = shell_u_hi = shell_v_lo = shell_v_hi = shell_top = None
    if inner_envelope_mesh is not None:
        env = _surface_local(inner_envelope_mesh, frame)
        if len(env):
            eu, ev = _project_uv(env, center_xy, md_axis, bl_axis)
            shell_u_lo = float(np.quantile(eu, .02) - thickness)
            shell_u_hi = float(np.quantile(eu, .98) + thickness)
            shell_v_lo = float(np.quantile(ev, .02) - thickness)
            shell_v_hi = float(np.quantile(ev, .98) + thickness)
            shell_top = float(np.quantile(env[:, 2], .98) + thickness)

    # Margin + emergence is the minimum sensible body envelope.  Confirmed
    # neighbours can then replace M/D sides with the actual available space.
    u_lo = margin_u_lo - float(cfg.absent_side_emergence_mm)
    u_hi = margin_u_hi + float(cfg.absent_side_emergence_mm)
    if shell_u_lo is not None:
        u_lo = min(u_lo, shell_u_lo)
    if shell_u_hi is not None:
        u_hi = max(u_hi, shell_u_hi)

    target = float(proximal_target_mm or 0.0)
    side_reports = []
    positive_found = negative_found = False
    for label, direction, local in refs:
        pu, pv = _project_uv(local, center_xy, md_axis, bl_axis)
        radial = np.sqrt(pu * pu + pv * pv)
        if len(radial) > 20:
            cutoff = float(np.quantile(radial, .35))
            near = radial <= cutoff
            if int(near.sum()) >= 10:
                pu, pv = pu[near], pv[near]
        sign, physical_center_u, direction_u, sign_source = _physical_side_sign(
            pu, direction, md_axis
        )
        clinical_sign = 1 if label == "mesial" else -1 if label == "distal" else None
        if sign > 0:
            inner = float(np.quantile(pu, cfg.neighbor_inner_quantile))
            raw_bound = inner - target
            patient_cap = margin_u_hi + float(cfg.neighbor_md_extra_from_margin_mm)
            bound = _clamp_side(min(raw_bound, patient_cap), 1, cfg)
            u_hi = bound
            positive_found = True
        else:
            inner = float(np.quantile(pu, 1.0 - cfg.neighbor_inner_quantile))
            raw_bound = inner + target
            patient_cap = margin_u_lo - float(cfg.neighbor_md_extra_from_margin_mm)
            bound = _clamp_side(max(raw_bound, patient_cap), -1, cfg)
            u_lo = bound
            negative_found = True
        side_reports.append({
            "label": label,
            "clinical_expected_sign": clinical_sign,
            "clinical_label_matches_physical_side": clinical_sign is None or clinical_sign == sign,
            "physical_sign": sign,
            "physical_sign_source": sign_source,
            "physical_center_u_mm": physical_center_u,
            "near_direction_u": direction_u,
            "neighbor_inner_u_mm": inner,
            "raw_target_bound_u_mm": raw_bound,
            "target_bound_u_mm": bound,
            "margin_relative_cap_u_mm": patient_cap,
            "target_signed_contact_mm": target,
        })

    shell_conflicts = []
    if shell_u_lo is not None and shell_u_lo < u_lo:
        shell_conflicts.append({"side": "negative", "required_u": shell_u_lo, "patient_limit_u": u_lo})
    if shell_u_hi is not None and shell_u_hi > u_hi:
        shell_conflicts.append({"side": "positive", "required_u": shell_u_hi, "patient_limit_u": u_hi})

    # B/L is intentionally not inferred as a target tooth width from neighbour
    # halos.  Margin/emergence and the inner shell establish a safe envelope;
    # neighbouring teeth only provide a conservative prior that may enlarge that
    # envelope.  They never globally squeeze the buccal or lingual anatomy.
    v_lo = margin_v_lo - float(cfg.bl_margin_emergence_mm)
    v_hi = margin_v_hi + float(cfg.bl_margin_emergence_mm)
    if shell_v_lo is not None:
        v_lo = min(v_lo, shell_v_lo)
    if shell_v_hi is not None:
        v_hi = max(v_hi, shell_v_hi)
    neigh_v_lo, neigh_v_hi = [], []
    for _label, _direction, local in refs:
        _pu, pv = _project_uv(local, center_xy, md_axis, bl_axis)
        if len(pv) >= 8:
            neigh_v_lo.append(float(np.quantile(pv, cfg.neighbor_width_low_quantile)))
            neigh_v_hi.append(float(np.quantile(pv, cfg.neighbor_width_high_quantile)))
    neighbor_v_lo = neighbor_v_hi = None
    if neigh_v_lo:
        neighbor_v_lo = max(float(np.median(neigh_v_lo)),
                            margin_v_lo - float(cfg.neighbor_bl_extra_from_margin_mm))
        v_lo = min(v_lo, neighbor_v_lo)
    if neigh_v_hi:
        neighbor_v_hi = min(float(np.median(neigh_v_hi)),
                            margin_v_hi + float(cfg.neighbor_bl_extra_from_margin_mm))
        v_hi = max(v_hi, neighbor_v_hi)
    v_lo = _clamp_side(v_lo, -1, cfg)
    v_hi = _clamp_side(v_hi, 1, cfg)

    # Inner shell top is a material floor. Antagonist is the only coronal ceiling.
    target_top = shell_top
    antagonist_report = {"used": False}
    if antagonist_surface is not None and occlusal_target_mm is not None:
        ant = _surface_local(antagonist_surface, frame)
        if len(ant):
            au, av = _project_uv(ant, center_xy, md_axis, bl_axis)
            pad = float(cfg.antagonist_xy_pad_mm)
            inside = ((au >= u_lo - pad) & (au <= u_hi + pad)
                      & (av >= v_lo - pad) & (av <= v_hi + pad)
                      & (ant[:, 2] > margin_z + cfg.min_coronal_height_mm))
            zvals = ant[inside, 2]
            if len(zvals) >= 4:
                ceiling = float(np.quantile(zvals, cfg.antagonist_low_quantile))
                candidate = ceiling - float(occlusal_target_mm)
                if shell_top is None or candidate >= shell_top - 1e-6:
                    target_top = candidate
                else:
                    target_top = shell_top
                    shell_conflicts.append({"side": "occlusal", "required_top_z": shell_top,
                                            "antagonist_limit_z": candidate})
                antagonist_report = {
                    "used": True,
                    "ceiling_z": ceiling,
                    "target_clearance_mm": float(occlusal_target_mm),
                    "sample_count": int(len(zvals)),
                    "target_top_z": float(target_top),
                }

    sources = {
        "semantics": "MAX_FEASIBLE_BOUNDS_NOT_TARGET_TOOTH_DIMENSIONS",
        "md_axis_source": axis_source,
        "confirmed_neighbor_count": len(refs),
        "positive_neighbor": positive_found,
        "negative_neighbor": negative_found,
        "proximal_sides": side_reports,
        "margin_bounds": {"u_min": margin_u_lo, "u_max": margin_u_hi,
                          "v_min": margin_v_lo, "v_max": margin_v_hi},
        "inner_shell_bounds": {"u_min": shell_u_lo, "u_max": shell_u_hi,
                               "v_min": shell_v_lo, "v_max": shell_v_hi,
                               "top_z": shell_top},
        "neighbor_bl_prior_count": len(neigh_v_lo),
        "neighbor_bl_capped_bounds": {"v_min": neighbor_v_lo, "v_max": neighbor_v_hi,
                                      "extra_from_margin_mm": float(cfg.neighbor_bl_extra_from_margin_mm)},
        "antagonist": antagonist_report,
        "constraint_conflicts": shell_conflicts,
    }
    return PatientEnvelope(tuple(map(float, center_xy)), tuple(map(float, md_axis)),
                           tuple(map(float, bl_axis)), float(u_lo), float(u_hi),
                           float(v_lo), float(v_hi), margin_z,
                           None if target_top is None else float(target_top), sources)


def _side_scale(extent, limit, floor):
    extent = max(float(extent), 1e-8)
    limit = max(float(limit), 1e-8)
    if extent <= limit + 1e-6:
        return 1.0
    return float(max(limit / extent, float(floor)))


def morph_embryo_to_patient_envelope(mesh, frame, margin, envelope: PatientEnvelope,
                                     *, config: PatientEnvelopeConfig | None = None):
    """Compress only envelope violations; never expand anatomy to fill a box.

    This is deliberately one-sided.  M/D contact reach is solved later against
    real neighbour surfaces.  B/L morphology is kept from the library unless it
    genuinely exceeds the broad patient envelope.  Antagonist height acts as a
    ceiling, never as a requested crown height.
    """
    cfg = config or PatientEnvelopeConfig()
    out = mesh.copy()
    original_world = np.asarray(out.vertices, float).copy()
    local = _local(original_world, frame)
    center = np.asarray(envelope.center_xy, float)
    md = np.asarray(envelope.md_axis_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    xy = local[:, :2] - center
    u0 = xy @ md
    v0 = xy @ bl
    height = MarginCurtain(margin, frame.z).axial_above(original_world)
    base_weight = _smoothstep01(height / max(float(cfg.emergence_transition_mm), 1e-6))
    base_weight[height <= float(cfg.cervical_immutable_mm)] = 0.0
    body = height > max(.30, cfg.emergence_transition_mm * .30)
    if int(body.sum()) < 16:
        body = height > 0.05
    if int(body.sum()) < 8:
        body = np.ones(len(local), bool)

    u_neg = max(abs(float(np.quantile(u0[body], .02))), .25)
    u_pos = max(abs(float(np.quantile(u0[body], .98))), .25)
    v_neg = max(abs(float(np.quantile(v0[body], .02))), .25)
    v_pos = max(abs(float(np.quantile(v0[body], .98))), .25)

    su_pos = _side_scale(u_pos, envelope.u_max, cfg.md_min_scale_per_pass)
    su_neg = _side_scale(u_neg, -envelope.u_min, cfg.md_min_scale_per_pass)
    sv_pos = _side_scale(v_pos, envelope.v_max, cfg.bl_min_scale_per_pass)
    sv_neg = _side_scale(v_neg, -envelope.v_min, cfg.bl_min_scale_per_pass)

    # M/D may adapt broadly, but the B/L occlusal core retains more anatomy so a
    # narrow margin cannot turn the whole crown into a flat slab.
    radial_u = np.abs(u0) / max(u_pos, u_neg, .25)
    radial_v = np.abs(v0) / max(v_pos, v_neg, .25)
    radial = np.sqrt(radial_u * radial_u + radial_v * radial_v)
    top_h = max(float(np.quantile(np.maximum(height[body], 0.0), .98)), .25)
    h_frac = np.clip(np.maximum(height, 0.0) / top_h, 0.0, 1.2)
    occlusal_core = (h_frac >= .58) & (radial <= .72)
    bl_weight = base_weight.copy()
    bl_weight[occlusal_core] *= float(cfg.bl_core_preservation)

    u_factor = np.where(u0 >= 0.0, su_pos, su_neg)
    v_factor = np.where(v0 >= 0.0, sv_pos, sv_neg)
    u = u0 + base_weight * (u0 * u_factor - u0)
    v = v0 + bl_weight * (v0 * v_factor - v0)
    local[:, :2] = center + u[:, None] * md + v[:, None] * bl

    z_before = local[:, 2].copy()
    top_before = float(np.quantile(z_before[body], .98))
    antagonist_used = bool((envelope.sources.get("antagonist") or {}).get("used"))
    height_scale = 1.0
    if envelope.target_top_z is not None and antagonist_used and top_before > float(envelope.target_top_z) + 1e-6:
        current_h = max(top_before - envelope.margin_z, .25)
        target_h = max(float(envelope.target_top_z) - envelope.margin_z, .25)
        height_scale = float(max(target_h / current_h, cfg.height_min_scale_per_pass))
        positive = np.maximum(z_before - envelope.margin_z, 0.0)
        z_weight = base_weight * _smoothstep01((h_frac - .25) / .55)
        target_z = envelope.margin_z + positive * height_scale
        local[:, 2] = z_before + z_weight * (target_z - z_before)

    out.vertices = _world(local, frame)
    # Exact hard root lock after all broad deformation.
    locked = height <= float(cfg.cervical_immutable_mm)
    out.vertices[locked] = original_world[locked]
    if not np.isfinite(np.asarray(out.vertices, float)).all():
        raise ValueError("PatientEnvelope morph 产生 NaN/Inf")

    after = _local(out.vertices, frame)
    axy = after[:, :2] - center
    au, av = axy @ md, axy @ bl
    report = {
        "mode": "ONE_SIDED_MAX_FEASIBLE_ENVELOPE_V4",
        "semantics": "COMPRESS_VIOLATIONS_ONLY_NEVER_FILL_BOX",
        "template_size_authoritative": False,
        "root_lock": "HARD_CERVICAL_ROOT_PLUS_SMOOTH_EMERGENCE",
        "cervical_immutable_mm": float(cfg.cervical_immutable_mm),
        "transition_mm": float(cfg.emergence_transition_mm),
        "height_owner": "ANTAGONIST_CEILING_ONLY" if antagonist_used else "UNCHANGED_EXCEPT_MATERIAL_FLOOR",
        "scales": {
            "md_positive": su_pos, "md_negative": su_neg,
            "bl_positive": sv_pos, "bl_negative": sv_neg,
            "height": height_scale,
        },
        "occlusal_bl_core_vertices": int(np.sum(occlusal_core)),
        "before": {
            "u_p02": float(np.quantile(u0[body], .02)), "u_p98": float(np.quantile(u0[body], .98)),
            "v_p02": float(np.quantile(v0[body], .02)), "v_p98": float(np.quantile(v0[body], .98)),
            "top_p98": top_before,
        },
        "after": {
            "u_p02": float(np.quantile(au[body], .02)), "u_p98": float(np.quantile(au[body], .98)),
            "v_p02": float(np.quantile(av[body], .02)), "v_p98": float(np.quantile(av[body], .98)),
            "top_p98": float(np.quantile(after[body, 2], .98)),
        },
        "max_cervical_weight": float(np.max(base_weight[height <= .20])) if np.any(height <= .20) else None,
    }
    return out, report
