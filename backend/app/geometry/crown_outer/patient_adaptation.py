"""Patient-driven CAD-1 outer-crown adaptation.

This module owns the *global* patient-space adaptation that must happen before
local contact refinement:

1. keep the Margin/cervical region authoritative;
2. fit mesiodistal body width to available proximal space;
3. fit overall crown height to the antagonist while preserving template relief;
4. explicitly pull the cervical rim onto the patient Margin with a smooth band.

The tooth library remains the anatomy prior.  Surrounding scans provide envelopes
and dimensions, not replacement surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..nonplanar_crown import MarginCurtain
from .zones import build_crown_zones


@dataclass(frozen=True)
class AdaptationConfig:
    cervical_band_mm: float = 1.20
    cervical_hard_band_mm: float = 0.22
    # Keep a tiny source contribution so coincident projections cannot collapse
    # distinct topology.  0.995 leaves only 0.5% of the original seam error.
    cervical_hard_weight: float = 0.995
    body_transition_mm: float = 1.60
    md_scale_min: float = 0.72
    md_scale_max: float = 1.32
    height_scale_min: float = 0.72
    height_scale_max: float = 1.45
    max_height_delta_mm: float = 3.0
    contact_quantile: float = 0.18


def _basis(frame) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(frame.origin, dtype=float)
    matrix = np.asarray(frame.matrix(), dtype=float)
    if origin.shape != (3,) or matrix.shape != (3, 3):
        raise ValueError("patient frame 必须包含三维 origin 与 3x3 matrix")
    if not np.isfinite(origin).all() or not np.isfinite(matrix).all():
        raise ValueError("patient frame 含非有限数")
    return origin, matrix


def _local(points, frame) -> np.ndarray:
    origin, matrix = _basis(frame)
    return (np.asarray(points, dtype=float) - origin) @ matrix


def _world(local, frame) -> np.ndarray:
    origin, matrix = _basis(frame)
    return np.asarray(local, dtype=float) @ matrix.T + origin


def _smoothstep01(value) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _finite_mesh(mesh) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("冠形网格没有有效顶点")
    if not np.isfinite(vertices).all():
        raise ValueError("冠形适配产生 NaN/Inf，已中止")


def _duplicate_pair_count(points, tolerance=1e-8) -> int:
    points = np.asarray(points, dtype=float)
    if not len(points):
        return 0
    return int(len(cKDTree(points).query_pairs(float(tolerance))))


def _polyline_distances_3d(points, loop) -> np.ndarray:
    """Exact point-to-closed-polyline distance in 3D.

    Margin is a curve, not a cloud of samples.  Measuring with a KD-tree over
    Margin vertices over-reports error whenever the confirmed Margin is sparse:
    a point can lie exactly on a segment yet be far from both sampled endpoints.
    """
    points = np.asarray(points, dtype=float)
    loop = np.asarray(loop, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Margin 距离测量点必须是 Nx3")
    if loop.ndim != 2 or loop.shape[1] != 3 or len(loop) < 3:
        raise ValueError("Margin 必须至少包含 3 个三维点")
    if not np.isfinite(points).all() or not np.isfinite(loop).all():
        raise ValueError("Margin 距离测量输入含 NaN/Inf")
    a = loop
    delta = np.roll(loop, -1, axis=0) - a
    denom = np.einsum("ij,ij->i", delta, delta)
    valid = denom > 1e-20
    if not np.any(valid):
        raise ValueError("Margin 全部由退化线段组成")
    safe = np.where(valid, denom, 1.0)
    out = np.empty(len(points), dtype=float)
    for start in range(0, len(points), 1024):
        stop = min(start + 1024, len(points))
        p = points[start:stop]
        rel = p[:, None, :] - a[None, :, :]
        t = np.einsum("nsi,si->ns", rel, delta) / safe[None, :]
        t = np.clip(t, 0.0, 1.0)
        nearest = a[None, :, :] + t[:, :, None] * delta[None, :, :]
        d2 = np.sum((p[:, None, :] - nearest) ** 2, axis=2)
        d2[:, ~valid] = np.inf
        out[start:stop] = np.sqrt(np.min(d2, axis=1))
    return out


def _closest_points_on_margin_curve(points_local, margin_local):
    """Continuous closest points on the *segments* of the closed Margin loop.

    The old implementation snapped every cervical vertex to the nearest sampled
    Margin vertex.  Several crown vertices could therefore receive exactly the
    same target and collapse into duplicate vertices.  Segment projection gives a
    continuous target (segment id + interpolation parameter) and preserves the 3D
    Margin height by interpolating the full XYZ segment, while proximity is chosen
    in the patient-frame XY plane.
    """
    points = np.asarray(points_local, dtype=float)
    loop = np.asarray(margin_local, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("颈缘待适配点必须是 Nx3")
    if loop.ndim != 2 or loop.shape[1] != 3 or len(loop) < 3:
        raise ValueError("Margin 必须至少包含 3 个三维点")
    if not np.isfinite(points).all() or not np.isfinite(loop).all():
        raise ValueError("颈缘或 Margin 含 NaN/Inf")

    a = loop
    b = np.roll(loop, -1, axis=0)
    delta = b - a
    delta_xy = delta[:, :2]
    denom = np.einsum("ij,ij->i", delta_xy, delta_xy)
    valid_segments = denom > 1e-18
    if not np.any(valid_segments):
        raise ValueError("Margin 全部由退化线段组成")

    safe = np.where(valid_segments, denom, 1.0)
    targets = np.empty_like(points)
    segment_ids = np.empty(len(points), dtype=int)
    parameters = np.empty(len(points), dtype=float)

    # Chunking avoids a large temporary tensor on dense library meshes.
    for start in range(0, len(points), 1024):
        stop = min(start + 1024, len(points))
        p = points[start:stop]
        rel = p[:, None, :2] - a[None, :, :2]
        t = np.einsum("nsi,si->ns", rel, delta_xy) / safe[None, :]
        t = np.clip(t, 0.0, 1.0)
        projected_xy = a[None, :, :2] + t[:, :, None] * delta_xy[None, :, :]
        d2 = np.sum((p[:, None, :2] - projected_xy) ** 2, axis=2)
        d2[:, ~valid_segments] = np.inf
        ids = np.argmin(d2, axis=1)
        rows = np.arange(len(p))
        chosen_t = t[rows, ids]
        targets[start:stop] = a[ids] + chosen_t[:, None] * delta[ids]
        segment_ids[start:stop] = ids
        parameters[start:stop] = chosen_t

    if not np.isfinite(targets).all():
        raise ValueError("Margin 曲线投影产生非有限目标")
    return targets, segment_ids, parameters


def _neighbor_face_x(surface, frame, side: str) -> float | None:
    """Return the neighbor X coordinate facing the target tooth.

    Auto-neighbor patches are already side-labelled.  In the confirmed patient
    frame the mesial patch should be on +X and distal on -X.  The quantile nearest
    the crown is deliberately robust to patch edges/outliers.
    """
    local = _local(surface.mesh.vertices, frame)
    x = local[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    if side == "mesial":
        positive = x[x > 0]
        values = positive if len(positive) >= max(8, len(x) // 10) else x
        return float(np.quantile(values, 0.08))
    if side == "distal":
        negative = x[x < 0]
        values = negative if len(negative) >= max(8, len(x) // 10) else x
        return float(np.quantile(values, 0.92))
    raise ValueError(f"未知邻接侧: {side}")


def global_patient_prefit(mesh, frame, margin, *, proximal_surfaces=None,
                          proximal_target_mm=None, antagonist_surface=None,
                          occlusal_target_mm=None, config: AdaptationConfig | None = None):
    """Fit whole-crown width/height before local Laplacian contact refinement.

    The cervical band is blended out, so the Margin remains the pivot of the
    transform.  X is scaled/translated from mesial/distal available space.  Z is
    scaled from antagonist clearance error; cusp/fossa relief is not flattened.
    """
    cfg = config or AdaptationConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    vertices = np.asarray(current.vertices, dtype=float)
    local = _local(vertices, frame)
    margin_local = _local(margin, frame)
    margin_z = float(np.median(margin_local[:, 2]))
    height_above = MarginCurtain(margin, frame.z).axial_above(vertices)
    body_weight = _smoothstep01(height_above / max(cfg.body_transition_mm, 1e-6))

    report = {
        "stage": "GLOBAL_PREFIT",
        "status": "PASS_ENGINEERING",
        "md": {"applied": False},
        "height": {"applied": False},
    }

    # ---- Mesiodistal available-space fit -------------------------------------------------
    surfaces = proximal_surfaces or {}
    target = None if proximal_target_mm is None else float(proximal_target_mm)
    if surfaces and target is not None:
        crown_non_cervical = height_above > cfg.cervical_hard_band_mm
        crown_x = local[crown_non_cervical, 0] if np.any(crown_non_cervical) else local[:, 0]
        current_min = float(np.quantile(crown_x, 0.03))
        current_max = float(np.quantile(crown_x, 0.97))
        current_center = 0.5 * (current_min + current_max)
        current_span = max(current_max - current_min, 1e-6)

        mesial_x = _neighbor_face_x(surfaces["mesial"], frame, "mesial") if "mesial" in surfaces else None
        distal_x = _neighbor_face_x(surfaces["distal"], frame, "distal") if "distal" in surfaces else None
        desired_max = mesial_x - target if mesial_x is not None else None
        desired_min = distal_x + target if distal_x is not None else None

        if desired_min is not None and desired_max is not None and desired_max > desired_min + 0.5:
            desired_center = 0.5 * (desired_min + desired_max)
            desired_span = desired_max - desired_min
            scale = float(np.clip(desired_span / current_span, cfg.md_scale_min, cfg.md_scale_max))
            translation = desired_center - current_center
        elif desired_max is not None:
            scale = 1.0
            translation = desired_max - current_max
        elif desired_min is not None:
            scale = 1.0
            translation = desired_min - current_min
        else:
            scale = 1.0
            translation = 0.0

        full_x = (local[:, 0] - current_center) * scale + current_center + translation
        local[:, 0] += body_weight * (full_x - local[:, 0])
        report["md"] = {
            "applied": bool(abs(scale - 1.0) > 1e-6 or abs(translation) > 1e-6),
            "scale": scale,
            "translation_mm": float(translation),
            "current_span_mm": current_span,
            "desired_min_x_mm": desired_min,
            "desired_max_x_mm": desired_max,
            "mesial_face_x_mm": mesial_x,
            "distal_face_x_mm": distal_x,
        }

    # ---- Overall height fit ---------------------------------------------------------------
    if antagonist_surface is not None and occlusal_target_mm is not None:
        temp = current.copy()
        temp.vertices = _world(local, frame)
        zones = build_crown_zones(temp, frame, margin)
        ids = np.flatnonzero(zones.occlusal)
        if len(ids):
            _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(temp.vertices)[ids])
            valid = np.asarray(valid, bool) & np.isfinite(signed)
            if np.any(valid):
                errors = np.asarray(signed[valid], float) - float(occlusal_target_mm)
                # Use the closest fraction of the antagonist envelope to establish
                # crown height.  This prevents deep fossae from forcing a global
                # raise while still allowing the whole crown to reach occlusion.
                ordered = np.sort(errors)
                take = max(1, int(np.ceil(len(ordered) * cfg.contact_quantile)))
                delta = float(np.median(ordered[:take]))
                delta = float(np.clip(delta, -cfg.max_height_delta_mm, cfg.max_height_delta_mm))
                occ_z = local[ids, 2]
                top = float(np.quantile(occ_z, 0.90))
                crown_height = max(top - margin_z, 0.5)
                scale = float(np.clip((crown_height + delta) / crown_height,
                                      cfg.height_scale_min, cfg.height_scale_max))
                full_z = margin_z + (local[:, 2] - margin_z) * scale
                local[:, 2] += body_weight * (full_z - local[:, 2])
                report["height"] = {
                    "applied": bool(abs(scale - 1.0) > 1e-6),
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

    current.vertices = _world(local, frame)
    _finite_mesh(current)
    return current, report


def cervical_fit_to_margin(mesh, frame, margin, *, config: AdaptationConfig | None = None):
    """Pull the template cervical seam onto the confirmed patient Margin.

    Targets are continuous projections onto Margin *segments*, never discrete
    Margin samples.  The bottom seam is fitted almost completely while a wider
    emergence band blends the displacement out.  Keeping a tiny fraction of the
    source position in the hard band avoids many-to-one collapse on pathological
    vertical columns while still bringing the seam to clinical-scale proximity.
    """
    cfg = config or AdaptationConfig()
    if not (0.0 < cfg.cervical_hard_weight < 1.0):
        raise ValueError("cervical_hard_weight 必须位于 (0,1)")
    _finite_mesh(mesh)
    current = mesh.copy()
    vertices = np.asarray(current.vertices, dtype=float)
    local = _local(vertices, frame)
    margin_local = _local(margin, frame)
    center = np.mean(margin_local[:, :2], axis=0)

    height = MarginCurtain(margin, frame.z).axial_above(vertices)
    band = np.asarray((height >= -0.15) & (height <= cfg.cervical_band_mm), bool)
    vxy = local[:, :2] - center
    vradius = np.linalg.norm(vxy, axis=1)

    target_local, segment_ids, segment_t = _closest_points_on_margin_curve(local, margin_local)
    target_radius = np.linalg.norm(target_local[:, :2] - center, axis=1)
    rim = band & (vradius >= np.maximum(0.40 * target_radius, target_radius - 2.0))
    if not np.any(rim):
        return current, {"stage": "CERVICAL_FIT", "status": "REVIEW_REQUIRED", "reason": "NO_CERVICAL_RIM"}

    seam = rim & (height <= cfg.cervical_hard_band_mm)
    if int(seam.sum()) < 8:
        rim_ids = np.flatnonzero(rim)
        order = rim_ids[np.argsort(height[rim_ids])]
        seam[order[:max(8, int(np.ceil(len(order) * 0.20)))]] = True

    # Transition band: strong near the Margin, smoothly vanishing coronally.
    h = np.clip(height / max(cfg.cervical_band_mm, 1e-6), 0.0, 1.0)
    weight = (1.0 - _smoothstep01(h)) ** 1.25
    weight *= rim.astype(float)
    # Near-full rather than exact 1.0 snap: distinct source vertices stay distinct
    # even if their continuous curve projections coincide.
    weight[seam] = cfg.cervical_hard_weight

    margin_world = np.asarray(margin, dtype=float)
    before_seam = _polyline_distances_3d(vertices[seam], margin_world)
    before_band = _polyline_distances_3d(vertices[rim], margin_world)
    duplicates_before = _duplicate_pair_count(vertices)

    local += weight[:, None] * (target_local - local)
    current.vertices = _world(local, frame)
    _finite_mesh(current)

    duplicates_after = _duplicate_pair_count(current.vertices)
    if duplicates_after > duplicates_before:
        raise ValueError(
            f"CERVICAL_FIT 产生新的重合顶点（{duplicates_before} → {duplicates_after}），已中止"
        )

    after_vertices = np.asarray(current.vertices)
    after_seam = _polyline_distances_3d(after_vertices[seam], margin_world)
    after_band = _polyline_distances_3d(after_vertices[rim], margin_world)
    report = {
        "stage": "CERVICAL_FIT",
        "status": "PASS_ENGINEERING",
        "rim_vertex_count": int(rim.sum()),
        "seam_vertex_count": int(seam.sum()),
        "duplicate_pairs_before": duplicates_before,
        "duplicate_pairs_after": duplicates_after,
        "projection": "continuous_closed_margin_segment_xy_full_xyz_interpolation",
        "distance_metric": "exact_point_to_closed_margin_polyline_3d",
        "segment_count_used": int(len(np.unique(segment_ids[seam]))),
        "segment_t_range": [float(np.min(segment_t[seam])), float(np.max(segment_t[seam]))],
        # `before_mm/after_mm` intentionally describe the actual cervical seam.
        "before_mm": {
            "median": float(np.median(before_seam)),
            "p95": float(np.quantile(before_seam, 0.95)),
            "max": float(np.max(before_seam)),
        },
        "after_mm": {
            "median": float(np.median(after_seam)),
            "p95": float(np.quantile(after_seam, 0.95)),
            "max": float(np.max(after_seam)),
        },
        "transition_band_mm": {
            "before_median": float(np.median(before_band)),
            "after_median": float(np.median(after_band)),
            "after_p95": float(np.quantile(after_band, 0.95)),
        },
    }
    return current, report
