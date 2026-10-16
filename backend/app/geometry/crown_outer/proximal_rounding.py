"""Crown 1.0 proximal-contact rounding pass.

The Sectional RED/GREEN/BLUE builder already owns the hard Margin, minimum-wall
and antagonist-ceiling constraints.  This module deliberately touches only the
*middle GREEN side rings* after the baseline is built:

* Margin ring stays byte-for-byte geometrically unchanged.
* Occlusal rim and every BLUE cap vertex stay unchanged.
* The intaglio/finalize path is not involved.
* Confirmed mesial/distal neighbour geometry is used as a radial upper bound.

The goal is practical Crown 1.0 morphology: a broad, rounded proximal contact
patch instead of a thin ray/spike, while preserving the existing inner-crown and
occlusal behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .patient_frame import PatientFrame, validate_patient_frame
from .sectional_v1 import SectionalOuterError, SectionalV1Config


@dataclass(frozen=True)
class ProximalRoundingConfig:
    """Soft morphology controls; all hard collision ceilings remain upstream."""

    vertical_sigma: float = 0.20
    angular_power: float = 0.42
    strength: float = 0.96
    smoothing_passes: int = 3
    smoothing_alpha: float = 0.18
    contact_tolerance_mm: float = 0.20
    minimum_contact_rays: int = 5
    minimum_contiguous_rays: int = 3

    def validate(self) -> None:
        if self.vertical_sigma <= 0:
            raise SectionalOuterError("proximal vertical_sigma must be > 0")
        if not (0.0 < self.angular_power <= 1.0):
            raise SectionalOuterError("proximal angular_power must be in (0,1]")
        if not (0.0 <= self.strength <= 1.0):
            raise SectionalOuterError("proximal strength must be in [0,1]")
        if self.smoothing_passes < 0:
            raise SectionalOuterError("proximal smoothing_passes must be >= 0")
        if not (0.0 <= self.smoothing_alpha <= 0.49):
            raise SectionalOuterError("proximal smoothing_alpha must be in [0,0.49]")
        if self.contact_tolerance_mm <= 0:
            raise SectionalOuterError("proximal contact_tolerance_mm must be > 0")


def _local(points, frame: PatientFrame) -> np.ndarray:
    return (np.asarray(points, dtype=float) - np.asarray(frame.origin, float)) @ frame.matrix()


def _world(points, frame: PatientFrame) -> np.ndarray:
    return np.asarray(points, dtype=float) @ frame.matrix().T + np.asarray(frame.origin, float)


def _mesh_points(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    centers = np.asarray(mesh.triangles_center, dtype=float)
    return np.vstack((vertices, centers))


def _margin(points) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 12:
        raise SectionalOuterError("proximal rounding requires an ordered Nx3 Margin")
    if np.linalg.norm(value[0] - value[-1]) < 1e-8:
        value = value[:-1]
    return value


def _smoothstep(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _neighbor_limits(points: np.ndarray, dirs: np.ndarray, margin_r: np.ndarray,
                     side: str, section: SectionalV1Config) -> np.ndarray:
    limits = np.full(len(dirs), np.inf, dtype=float)
    sign = 1.0 if side == "mesial" else -1.0
    xy = np.asarray(points, dtype=float)[:, :2]
    for i, direction in enumerate(dirs):
        if float(direction[0]) * sign < float(section.neighbor_sector_cos):
            continue
        projection = xy @ direction
        perpendicular = np.linalg.norm(
            xy - projection[:, None] * direction[None, :], axis=1)
        candidate = (
            (projection > margin_r[i] + 0.10)
            & (perpendicular <= float(section.neighbor_ray_band_mm))
        )
        if candidate.any():
            limits[i] = float(
                np.quantile(projection[candidate], 0.05)
                - float(section.proximal_gap_mm)
            )
    return limits


def _masked_circular_smooth(values: np.ndarray, mask: np.ndarray, *,
                            passes: int, alpha: float) -> np.ndarray:
    """Smooth without inventing displacement on rays with no neighbour evidence."""
    result = np.asarray(values, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    for _ in range(int(passes)):
        previous = result.copy()
        left = np.roll(previous, 1)
        right = np.roll(previous, -1)
        left_ok = np.roll(mask, 1)
        right_ok = np.roll(mask, -1)
        left = np.where(left_ok, left, previous)
        right = np.where(right_ok, right, previous)
        result[mask] = (
            (1.0 - 2.0 * alpha) * previous[mask]
            + alpha * left[mask]
            + alpha * right[mask]
        )
        result[~mask] = 0.0
    return result


def _longest_circular_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    if mask.all():
        return int(len(mask))
    doubled = np.r_[mask, mask]
    best = current = 0
    for value in doubled:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(min(best, len(mask)))


def round_proximal_contacts(
    mesh,
    margin_points,
    frame: PatientFrame,
    *,
    neighbors: Mapping[str, object | None] | None,
    sectional_config: SectionalV1Config,
    config: ProximalRoundingConfig | None = None,
):
    """Broaden/round mesial and distal contacts without touching RED/BLUE truth.

    The sectional mesh has deterministic topology: ``len(side_levels)`` side
    rings followed by BLUE cap rings and the central fossa vertex.  Only side
    rings ``1 .. -2`` are allowed to move.  This keeps the exact Margin (ring 0)
    and the BLUE occlusal rim/cap (last side ring and all later vertices) fixed.
    """
    config = config or ProximalRoundingConfig()
    config.validate()
    sectional_config.validate()
    validate_patient_frame(frame)

    margin = _margin(margin_points)
    count = int(len(margin))
    level_count = int(len(sectional_config.side_levels))
    protected_start = level_count * count
    if len(mesh.vertices) <= protected_start:
        raise SectionalOuterError("sectional topology too small for proximal rounding")

    local = _local(np.asarray(mesh.vertices, dtype=float), frame)
    original_local = local.copy()
    local_margin = _local(margin, frame)
    center_xy = local_margin[:, :2].mean(axis=0)
    radial = local_margin[:, :2] - center_xy
    margin_r = np.linalg.norm(radial, axis=1)
    if float(np.min(margin_r)) < 1e-8:
        raise SectionalOuterError("Margin radial frame is degenerate")
    dirs = radial / margin_r[:, None]

    neighbors = dict(neighbors or {})
    levels = np.asarray(sectional_config.side_levels, dtype=float)
    side_reports: dict[str, dict] = {}

    for side in ("mesial", "distal"):
        neighbor = neighbors.get(side)
        if neighbor is None:
            side_reports[side] = {
                "status": "NOT_ASSESSED",
                "reason": "NO_CONFIRMED_NEIGHBOR",
                "mechanism": "no_contact_is_invented_for_absent_side",
            }
            continue

        points = _local(_mesh_points(neighbor), frame)
        sign = 1.0 if side == "mesial" else -1.0
        points = points[(points[:, 0] * sign) > 0]
        if len(points) < 8:
            side_reports[side] = {
                "status": "REVIEW_REQUIRED",
                "reason": "NEIGHBOR_PATCH_TOO_SMALL",
            }
            continue

        limit = _neighbor_limits(points, dirs, margin_r, side, sectional_config)
        finite = np.isfinite(limit)
        direction_score = dirs[:, 0] * sign
        sector = np.clip(
            (direction_score - float(sectional_config.neighbor_sector_cos))
            / (1.0 - float(sectional_config.neighbor_sector_cos)),
            0.0,
            1.0,
        )
        # Power < 1 flattens the angular peak into a broad plateau instead of a horn.
        angular = _smoothstep(sector) ** float(config.angular_power)

        moved_vertices = 0
        maximum_move = 0.0
        total_move = 0.0
        for ring_index in range(1, level_count - 1):
            s = float(levels[ring_index])
            vertical = float(np.exp(
                -0.5 * ((s - float(sectional_config.contact_height_ratio))
                        / float(config.vertical_sigma)) ** 2))
            if vertical < 0.02:
                continue

            ids = np.arange(ring_index * count, (ring_index + 1) * count, dtype=np.int64)
            ring_xy = local[ids, :2] - center_xy
            radius = np.sum(ring_xy * dirs, axis=1)
            allowed = np.minimum(
                limit,
                margin_r + float(sectional_config.max_proximal_extension_mm),
            )
            outward_room = np.maximum(allowed - radius, 0.0)
            evidence = finite & (angular > 0.0) & (outward_room > 1e-6)
            raw_delta = np.zeros(count, dtype=float)
            raw_delta[evidence] = (
                outward_room[evidence]
                * angular[evidence]
                * vertical
                * float(config.strength)
            )
            delta = _masked_circular_smooth(
                raw_delta,
                evidence,
                passes=int(config.smoothing_passes),
                alpha=float(config.smoothing_alpha),
            )
            # Never cross the measured facing surface after smoothing.
            delta = np.minimum(delta, outward_room)
            delta[~evidence] = 0.0
            local[ids, :2] += delta[:, None] * dirs

            moved_vertices += int(np.sum(delta > 1e-5))
            if len(delta):
                maximum_move = max(maximum_move, float(delta.max()))
                total_move += float(delta.sum())

        contact_ring_index = int(np.argmin(np.abs(
            levels - float(sectional_config.contact_height_ratio))))
        contact_ring_index = int(np.clip(contact_ring_index, 1, level_count - 2))
        contact_ids = np.arange(
            contact_ring_index * count,
            (contact_ring_index + 1) * count,
            dtype=np.int64,
        )
        contact_radius = np.sum(
            (local[contact_ids, :2] - center_xy) * dirs, axis=1)
        gap = limit - contact_radius
        contact_like = (
            finite
            & (angular >= 0.20)
            & (gap >= -1e-6)
            & (gap <= float(config.contact_tolerance_mm))
        )
        contact_count = int(contact_like.sum())
        contiguous = _longest_circular_run(contact_like)

        # A compact roundness diagnostic: second angular difference of the contact
        # ring radius.  It is not a clinical curvature metric, but it catches the
        # exact single-ray horn/faceted profile Crown 1.0 is trying to avoid.
        second = np.abs(
            np.roll(contact_radius, 1)
            - 2.0 * contact_radius
            + np.roll(contact_radius, -1))
        sector_mask = finite & (angular > 0.05)
        roundness_p95 = (
            float(np.quantile(second[sector_mask], 0.95))
            if sector_mask.any() else None
        )

        passed = (
            contact_count >= int(config.minimum_contact_rays)
            and contiguous >= int(config.minimum_contiguous_rays)
        )
        side_reports[side] = {
            "status": "PASS_ENGINEERING" if passed else "REVIEW_REQUIRED",
            "neighbor_point_count": int(len(points)),
            "ray_coverage": int(finite.sum()),
            "rounded_contact_ray_count": contact_count,
            "longest_contiguous_contact_run": contiguous,
            "contact_ring_level": float(levels[contact_ring_index]),
            "moved_side_vertex_count": moved_vertices,
            "maximum_outward_move_mm": maximum_move,
            "mean_outward_move_mm": (
                total_move / moved_vertices if moved_vertices else 0.0),
            "angular_roundness_second_diff_p95_mm": roundness_p95,
            "mechanism": "broad_angular_plateau_plus_vertical_gaussian_and_circular_smoothing",
        }

    # Hard invariants: RED bottom boundary and BLUE/topology are untouched.
    margin_unchanged = bool(np.allclose(local[:count], original_local[:count], atol=1e-12))
    protected_unchanged = bool(np.allclose(
        local[(level_count - 1) * count:],
        original_local[(level_count - 1) * count:],
        atol=1e-12,
    ))
    if not margin_unchanged:
        raise SectionalOuterError("proximal rounding modified the hard Margin boundary")
    if not protected_unchanged:
        raise SectionalOuterError("proximal rounding modified the BLUE occlusal rim/cap")

    import trimesh
    rounded = trimesh.Trimesh(
        vertices=_world(local, frame),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
    )
    if not rounded.is_winding_consistent:
        raise SectionalOuterError("proximal rounding broke face winding")
    if np.any(rounded.area_faces < 1e-10):
        raise SectionalOuterError("proximal rounding created degenerate faces")

    assessed = [r["status"] for r in side_reports.values()
                if r.get("status") != "NOT_ASSESSED"]
    overall = (
        "PASS_ENGINEERING"
        if assessed and all(v == "PASS_ENGINEERING" for v in assessed)
        else "REVIEW_REQUIRED" if assessed else "NOT_ASSESSED"
    )
    report = {
        "implementation": "geometry.crown_outer.proximal_rounding/1",
        "status": overall,
        "margin_preserved": margin_unchanged,
        "blue_occlusal_vertices_preserved": protected_unchanged,
        "side_levels": [float(v) for v in levels],
        "per_side": side_reports,
        "surface_goal": "rounded_emergence_to_broad_contact_without_contact_horns",
    }
    return rounded, report
