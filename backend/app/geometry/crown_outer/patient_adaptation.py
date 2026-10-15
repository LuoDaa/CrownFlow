"""Patient-driven CAD-1 outer-crown adaptation.

The library tooth is an anatomy prior, not patient-size truth.  This module keeps
legacy/global pre-fit helpers plus the authoritative full-loop cervical fit used
throughout the current CAD-1 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from ..nonplanar_crown import MarginCurtain
from .zones import build_crown_zones


@dataclass(frozen=True)
class AdaptationConfig:
    cervical_band_mm: float = 1.20
    cervical_hard_band_mm: float = 0.22
    cervical_hard_weight: float = 0.995
    cervical_noop_p95_mm: float = 0.020
    cervical_line_search: tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.125, 0.0625)
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
    """Exact point-to-closed-polyline distance in 3D."""
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
    """Continuous closest points on segments of the closed Margin loop."""
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


def _full_loop_margin_anchors(local, margin_local, rim, height):
    """Choose unique rim vertices that cover the whole confirmed Margin loop."""
    rim_ids = np.flatnonzero(np.asarray(rim, bool))
    loop = np.asarray(margin_local, float)
    if len(rim_ids) < 8 or len(loop) < 3:
        return np.zeros(0, int), np.zeros(0, int)
    count = min(len(loop), len(rim_ids))
    margin_ids = (
        np.arange(len(loop), dtype=int)
        if count == len(loop)
        else np.floor(np.linspace(0, len(loop), count, endpoint=False)).astype(int)
    )
    candidates = np.asarray(local, float)[rim_ids]
    targets = loop[margin_ids]
    dxy = np.linalg.norm(targets[:, None, :2] - candidates[None, :, :2], axis=2)
    dz = np.abs(targets[:, None, 2] - candidates[None, :, 2])
    center = np.mean(loop[:, :2], axis=0)
    ta = np.arctan2(targets[:, 1] - center[1], targets[:, 0] - center[0])
    ca = np.arctan2(candidates[:, 1] - center[1], candidates[:, 0] - center[0])
    da = ta[:, None] - ca[None, :]
    angular = np.abs(np.arctan2(np.sin(da), np.cos(da)))
    radius_scale = max(float(np.median(np.linalg.norm(loop[:, :2] - center, axis=1))), 0.5)
    hpenalty = np.maximum(np.asarray(height, float)[rim_ids], 0.0)[None, :]
    cost = dxy + 0.25 * dz + 0.35 * radius_scale * angular + 0.15 * hpenalty
    rows, cols = linear_sum_assignment(cost)
    anchor_ids = rim_ids[cols]
    matched_margin_ids = margin_ids[rows]
    order = np.argsort(matched_margin_ids)
    return anchor_ids[order], matched_margin_ids[order]


def _anchor_transition_weights(mesh, anchor_ids, rim):
    anchor_ids = np.asarray(anchor_ids, int)
    rim = np.asarray(rim, bool)
    influence = np.zeros(len(mesh.vertices), float)
    if not len(anchor_ids):
        return influence
    influence[anchor_ids] = 1.0
    adjacency = [set() for _ in mesh.vertices]
    for a, b in np.asarray(mesh.edges_unique, int):
        adjacency[int(a)].add(int(b))
        adjacency[int(b)].add(int(a))
    seen = set(map(int, anchor_ids))
    front = set(seen)
    for value in (0.80, 0.55, 0.30):
        nxt = {v for u in front for v in adjacency[u] if v not in seen and rim[v]}
        if not nxt:
            break
        ids = np.asarray(sorted(nxt), int)
        influence[ids] = np.maximum(influence[ids], value)
        seen |= nxt
        front = nxt
    return influence


def _surface_axis_reference(surface, frame, center_x: float):
    """Return the physical +/-X side and the face coordinate nearest the space.

    Clinical mesial/distal labels are metadata.  Geometry must follow where the
    actual neighbor patch sits in the patient frame, otherwise a mirrored frame
    can make a one-sided case grow on the wrong side.
    """
    local = _local(surface.mesh.vertices, frame)
    x = local[:, 0]
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    median = float(np.median(x))
    sign = 1.0 if median >= center_x else -1.0
    if sign > 0:
        values = x[x > center_x]
        if len(values) < 8:
            values = x
        face_x = float(np.quantile(values, 0.06))
    else:
        values = x[x < center_x]
        if len(values) < 8:
            values = x
        face_x = float(np.quantile(values, 0.94))
    return {"axis_sign": sign, "face_x_mm": face_x, "median_x_mm": median}


def global_patient_prefit(mesh, frame, margin, *, proximal_surfaces=None,
                          proximal_target_mm=None, antagonist_surface=None,
                          occlusal_target_mm=None, config: AdaptationConfig | None = None):
    """Legacy/global fit retained for tests and natural-prep compatibility.

    One-sided proximal space expands only that physical side around the Margin
    center; the whole crown is never translated merely to make contact.
    """
    cfg = config or AdaptationConfig()
    _finite_mesh(mesh)
    current = mesh.copy()
    vertices = np.asarray(current.vertices, dtype=float)
    local = _local(vertices, frame)
    margin_local = _local(margin, frame)
    center_x = float(np.median(margin_local[:, 0]))
    margin_z = float(np.median(margin_local[:, 2]))
    height_above = MarginCurtain(margin, frame.z).axial_above(vertices)
    body_weight = _smoothstep01(height_above / max(cfg.body_transition_mm, 1e-6))
    report = {"stage": "GLOBAL_PREFIT", "status": "PASS_ENGINEERING", "md": {"applied": False}, "height": {"applied": False}}

    surfaces = proximal_surfaces or {}
    target = None if proximal_target_mm is None else float(proximal_target_mm)
    if surfaces and target is not None:
        non_cervical = height_above > cfg.cervical_hard_band_mm
        crown_x = local[non_cervical, 0] if np.any(non_cervical) else local[:, 0]
        current_min = float(np.quantile(crown_x, 0.03))
        current_max = float(np.quantile(crown_x, 0.97))
        refs = {name: _surface_axis_reference(surface, frame, center_x) for name, surface in surfaces.items()}
        refs = {name: ref for name, ref in refs.items() if ref is not None}
        pos_faces = [ref["face_x_mm"] for ref in refs.values() if ref["axis_sign"] > 0]
        neg_faces = [ref["face_x_mm"] for ref in refs.values() if ref["axis_sign"] < 0]
        desired_max = min(pos_faces) - target if pos_faces else current_max
        desired_min = max(neg_faces) + target if neg_faces else current_min

        pos_now = max(current_max - center_x, 1e-6)
        neg_now = max(center_x - current_min, 1e-6)
        pos_want = max(desired_max - center_x, 1e-6)
        neg_want = max(center_x - desired_min, 1e-6)
        pos_scale = float(np.clip(pos_want / pos_now, cfg.md_scale_min, cfg.md_scale_max))
        neg_scale = float(np.clip(neg_want / neg_now, cfg.md_scale_min, cfg.md_scale_max))
        x = local[:, 0].copy()
        delta = x - center_x
        target_x = x.copy()
        positive = delta >= 0
        target_x[positive] = center_x + delta[positive] * pos_scale
        target_x[~positive] = center_x + delta[~positive] * neg_scale
        local[:, 0] += body_weight * (target_x - x)
        report["md"] = {
            "applied": bool(abs(pos_scale - 1.0) > 1e-6 or abs(neg_scale - 1.0) > 1e-6),
            "mode": "PHYSICAL_SIDE_ASYMMETRIC_SCALE_ABOUT_MARGIN_CENTER",
            "positive_scale": pos_scale,
            "negative_scale": neg_scale,
            "whole_crown_translation_mm": 0.0,
            "desired_min_x_mm": desired_min,
            "desired_max_x_mm": desired_max,
            "reference_mapping": refs,
        }

    if antagonist_surface is not None and occlusal_target_mm is not None:
        for iteration in range(4):
            temp = current.copy()
            temp.vertices = _world(local, frame)
            zones = build_crown_zones(temp, frame, margin)
            ids = np.flatnonzero(zones.occlusal)
            if not len(ids):
                report["height"] = {"applied": False, "reason": "NO_OCCLUSAL_ZONE"}
                break
            _cp, _faces, signed, _distance, valid = antagonist_surface.query(np.asarray(temp.vertices)[ids])
            signed = np.asarray(signed, float)
            valid = np.asarray(valid, bool) & np.isfinite(signed)
            if not np.any(valid):
                report["height"] = {"applied": False, "reason": "NO_VALID_ANTAGONIST_SAMPLES"}
                break
            values = signed[valid]
            target_clearance = float(occlusal_target_mm)
            penetration = np.maximum(target_clearance - values, 0.0)
            if float(np.quantile(penetration, 0.95)) > 0.02:
                report["height"] = {"applied": iteration > 0, "reason": "PENETRATION_PRESENT_LOCAL_REFINEMENT_REQUIRED", "iterations": iteration}
                break
            positive_error = values - target_clearance
            positive_error = positive_error[positive_error > 0.03]
            if not len(positive_error):
                report["height"] = {"applied": iteration > 0, "reason": "HEIGHT_ON_TARGET", "iterations": iteration}
                break
            ordered = np.sort(positive_error)
            take = max(1, int(np.ceil(len(ordered) * cfg.contact_quantile)))
            step = float(np.clip(np.median(ordered[:take]), 0.0, min(cfg.max_height_delta_mm, 1.25)))
            if step <= 1e-5:
                break
            local[:, 2] += body_weight * step
            report["height"] = {
                "applied": True,
                "mode": "ITERATIVE_ANTAGONIST_AXIAL_HEIGHT_SEEK",
                "last_step_mm": step,
                "iterations": iteration + 1,
                "target_clearance_mm": target_clearance,
            }

    current.vertices = _world(local, frame)
    _finite_mesh(current)
    return current, report


def _coverage_from_anchors(vertices, anchor_ids, margin_world):
    if len(anchor_ids):
        reverse = cKDTree(np.asarray(vertices)[anchor_ids]).query(margin_world, k=1)[0]
        return reverse, float(np.mean(reverse <= 0.05)), float(np.quantile(reverse, 0.95)), float(np.max(reverse))
    reverse = np.full(len(margin_world), np.inf)
    return reverse, 0.0, float("inf"), float("inf")


def cervical_fit_to_margin(mesh, frame, margin, *, config: AdaptationConfig | None = None):
    """Pull the cervical region onto the entire Margin without collapsing topology.

    The function is deliberately idempotent: once the full loop is within tolerance
    a later shell/contact recheck leaves it untouched.  If a full-strength refit
    would create coincident vertices, a deterministic line search backs off rather
    than aborting the whole automatic design.
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

    near_seam = rim & (height <= cfg.cervical_hard_band_mm)
    anchor_ids, anchor_margin_ids = _full_loop_margin_anchors(local, margin_local, rim, height)
    seam = near_seam.copy()
    if len(anchor_ids):
        seam[anchor_ids] = True
        target_local[anchor_ids] = margin_local[anchor_margin_ids]
    if int(seam.sum()) < 8:
        rim_ids = np.flatnonzero(rim)
        order = rim_ids[np.argsort(height[rim_ids])]
        seam[order[:max(8, int(np.ceil(len(order) * 0.20)))]] = True

    h = np.clip(height / max(cfg.cervical_band_mm, 1e-6), 0.0, 1.0)
    weight = (1.0 - _smoothstep01(h)) ** 1.25
    weight *= rim.astype(float)
    influence = _anchor_transition_weights(current, anchor_ids, rim)
    weight = np.maximum(weight, influence * cfg.cervical_hard_weight)
    weight[seam] = cfg.cervical_hard_weight

    margin_world = np.asarray(margin, dtype=float)
    before_seam = _polyline_distances_3d(vertices[seam], margin_world)
    before_band = _polyline_distances_3d(vertices[rim], margin_world)
    duplicates_before = _duplicate_pair_count(vertices)
    reverse_before, coverage_before, reverse_before_p95, _reverse_before_max = _coverage_from_anchors(vertices, anchor_ids, margin_world)
    before_p95 = float(np.quantile(before_seam, 0.95))

    # Once the full loop is good, repeated calls must be a no-op.  This is crucial
    # after shell/contact passes: Margin is a hard boundary, not something to snap
    # repeatedly until vertices become numerically coincident.
    noop = before_p95 <= cfg.cervical_noop_p95_mm and coverage_before >= 0.98
    selected_scale = 0.0
    tried = []
    accepted_vertices = vertices.copy()
    duplicates_after = duplicates_before

    if not noop:
        base_delta = weight[:, None] * (target_local - local)
        for scale in cfg.cervical_line_search:
            trial_local = local + float(scale) * base_delta
            trial_vertices = _world(trial_local, frame)
            if not np.isfinite(trial_vertices).all():
                tried.append({"scale": float(scale), "finite": False})
                continue
            duplicate_count = _duplicate_pair_count(trial_vertices)
            tried.append({"scale": float(scale), "finite": True, "duplicate_pairs": duplicate_count})
            if duplicate_count <= duplicates_before:
                accepted_vertices = trial_vertices
                selected_scale = float(scale)
                duplicates_after = duplicate_count
                break

    current.vertices = accepted_vertices
    _finite_mesh(current)
    after_vertices = np.asarray(current.vertices)
    after_seam = _polyline_distances_3d(after_vertices[seam], margin_world)
    after_band = _polyline_distances_3d(after_vertices[rim], margin_world)
    reverse, coverage_fraction, reverse_p95, reverse_max = _coverage_from_anchors(after_vertices, anchor_ids, margin_world)
    after_p95 = float(np.quantile(after_seam, 0.95))

    used_segments = np.unique(np.r_[segment_ids[near_seam], anchor_margin_ids]).astype(int)
    status = "PASS_ENGINEERING" if after_p95 <= 0.05 and coverage_fraction >= 0.98 else "REVIEW_REQUIRED"
    report = {
        "stage": "CERVICAL_FIT",
        "status": status,
        "rim_vertex_count": int(rim.sum()),
        "seam_vertex_count": int(seam.sum()),
        "full_loop_anchor_count": int(len(anchor_ids)),
        "margin_sample_count": int(len(margin_world)),
        "duplicate_pairs_before": duplicates_before,
        "duplicate_pairs_after": duplicates_after,
        "projection": "continuous_closed_margin_segment_plus_full_loop_unique_anchor_assignment",
        "distance_metric": "exact_point_to_closed_margin_polyline_3d",
        "segment_count_used": int(len(used_segments)),
        "segment_t_range": [float(np.min(segment_t[seam])), float(np.max(segment_t[seam]))],
        "idempotent_noop": bool(noop),
        "duplicate_guard": {
            "policy": "BACKTRACK_INSTEAD_OF_ABORT",
            "selected_scale": float(selected_scale),
            "backed_off": bool(selected_scale not in (0.0, 1.0)),
            "trials": tried,
        },
        "before_mm": {"median": float(np.median(before_seam)), "p95": before_p95, "max": float(np.max(before_seam))},
        "after_mm": {"median": float(np.median(after_seam)), "p95": after_p95, "max": float(np.max(after_seam))},
        "full_loop_coverage_before": {
            "within_0_05mm_fraction": coverage_before,
            "reverse_p95_mm": reverse_before_p95,
            "covered_samples": int(np.sum(reverse_before <= 0.05)),
        },
        "full_loop_coverage": {
            "within_0_05mm_fraction": coverage_fraction,
            "reverse_p95_mm": reverse_p95,
            "reverse_max_mm": reverse_max,
            "covered_samples": int(np.sum(reverse <= 0.05)),
            "uncovered_samples": int(np.sum(reverse > 0.05)),
        },
        "transition_band_mm": {
            "before_median": float(np.median(before_band)),
            "after_median": float(np.median(after_band)),
            "after_p95": float(np.quantile(after_band, 0.95)),
        },
    }
    if not noop and selected_scale == 0.0 and status != "PASS_ENGINEERING":
        report["reason"] = "DUPLICATE_SAFE_MARGIN_BACKTRACK_EXHAUSTED"
    return current, report
