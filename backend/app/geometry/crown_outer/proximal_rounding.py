"""Crown 1.0 neighbour-facing proximal surface pass.

The Sectional RED/GREEN/BLUE builder owns the hard truths that are already working:

* RED: exact confirmed Margin + prep/minimum-wall envelope.
* BLUE: antagonist ceiling, occlusal height and cap relief.
* W3: prep-driven blockout/gap/intaglio boolean.

This module changes only the *middle GREEN side rings*.  Instead of extending a
few radial rays until they happen to touch a neighbour, it treats each confirmed
neighbour patch as a real 3-D target surface:

1. build a robust bucco-lingual contact band from the neighbour patch;
2. query the inward-facing neighbour surface in Y/Z and move the crown toward a
   target gap over a broad band (not a point/horn);
3. use the neighbours' bucco-lingual width as a soft GREEN side-wall envelope;
4. fair the displacement field around the ring while keeping Margin and BLUE cap
   bit-for-bit geometrically unchanged.

This is intentionally an outer-surface morphology step.  It never touches the
intaglio pipeline and never changes any vertex height along PatientFrame +Z.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .patient_frame import PatientFrame, validate_patient_frame
from .sectional_v1 import SectionalOuterError, SectionalV1Config


@dataclass(frozen=True)
class ProximalRoundingConfig:
    """Soft GREEN morphology controls; RED/BLUE hard constraints stay upstream."""

    # Broad contact band around the anatomical contact height.
    vertical_sigma: float = 0.19
    contact_strength: float = 1.00
    proximal_sector_start: float = 0.12
    contact_tolerance_mm: float = 0.20

    # Query the confirmed neighbour patch as a true Y/Z -> facing-X surface.
    facing_query_radius_mm: float = 1.55
    facing_k: int = 28
    facing_quantile: float = 0.10
    bl_quantile_low: float = 0.08
    bl_quantile_high: float = 0.92
    contact_band_pad_mm: float = 0.22

    # GREEN bucco-lingual envelope.  Only outward movement is allowed.
    bl_width_scale: float = 0.94
    bl_strength: float = 0.72
    bl_max_expansion_mm: float = 1.10
    bl_sector_start: float = 0.22
    emergence_extra_mm: float = 0.22

    # Fair the displacement field, not the hard boundary/cap geometry.
    smoothing_passes: int = 4
    smoothing_alpha: float = 0.16

    minimum_contact_rays: int = 5
    minimum_contiguous_rays: int = 3

    def validate(self) -> None:
        if self.vertical_sigma <= 0:
            raise SectionalOuterError("proximal vertical_sigma must be > 0")
        if not (0.0 < self.contact_strength <= 1.0):
            raise SectionalOuterError("proximal contact_strength must be in (0,1]")
        if not (0.0 <= self.proximal_sector_start < 0.95):
            raise SectionalOuterError("proximal_sector_start outside supported range")
        if self.contact_tolerance_mm <= 0:
            raise SectionalOuterError("proximal contact_tolerance_mm must be > 0")
        if self.facing_query_radius_mm <= 0 or self.facing_k < 3:
            raise SectionalOuterError("invalid proximal facing-surface query")
        if not (0.0 < self.facing_quantile < 0.5):
            raise SectionalOuterError("facing_quantile must be in (0,0.5)")
        if not (0.0 <= self.bl_quantile_low < self.bl_quantile_high <= 1.0):
            raise SectionalOuterError("invalid bucco-lingual quantiles")
        if not (0.5 <= self.bl_width_scale <= 1.25):
            raise SectionalOuterError("bl_width_scale outside supported range")
        if not (0.0 <= self.bl_strength <= 1.0):
            raise SectionalOuterError("bl_strength must be in [0,1]")
        if self.bl_max_expansion_mm < 0:
            raise SectionalOuterError("bl_max_expansion_mm must be >= 0")
        if self.smoothing_passes < 0:
            raise SectionalOuterError("proximal smoothing_passes must be >= 0")
        if not (0.0 <= self.smoothing_alpha <= 0.49):
            raise SectionalOuterError("proximal smoothing_alpha must be in [0,0.49]")


@dataclass
class _FacingPatch:
    side: str
    sign: float
    points: np.ndarray
    tree_yz: cKDTree
    y_low: float
    y_high: float
    z_low: float
    z_high: float
    y_mid: float
    bl_width: float


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
        raise SectionalOuterError("proximal surface pass requires an ordered Nx3 Margin")
    if np.linalg.norm(value[0] - value[-1]) < 1e-8:
        value = value[:-1]
    if not np.isfinite(value).all():
        raise SectionalOuterError("proximal surface pass received non-finite Margin")
    return value


def _smoothstep(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _masked_circular_smooth(values: np.ndarray, mask: np.ndarray, *,
                            passes: int, alpha: float) -> np.ndarray:
    """Smooth a ring displacement without inventing support outside ``mask``."""
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


def _build_patch(mesh, frame: PatientFrame, side: str,
                 config: ProximalRoundingConfig) -> _FacingPatch | None:
    if mesh is None:
        return None
    points = _local(_mesh_points(mesh), frame)
    sign = 1.0 if side == "mesial" else -1.0
    points = points[(points[:, 0] * sign) > 0]
    if len(points) < 8:
        return None

    qlo = float(config.bl_quantile_low)
    qhi = float(config.bl_quantile_high)
    y_low, y_high = np.quantile(points[:, 1], [qlo, qhi])
    z_low, z_high = np.quantile(points[:, 2], [0.08, 0.92])
    if not np.isfinite([y_low, y_high, z_low, z_high]).all():
        return None
    if float(y_high - y_low) < 0.35:
        # A click patch can be narrow, but a degenerate strip is not enough to
        # define a clinically useful broad contact surface.
        return None

    return _FacingPatch(
        side=side,
        sign=sign,
        points=points,
        tree_yz=cKDTree(points[:, 1:3]),
        y_low=float(y_low),
        y_high=float(y_high),
        z_low=float(z_low),
        z_high=float(z_high),
        y_mid=float(0.5 * (y_low + y_high)),
        bl_width=float(y_high - y_low),
    )


def _query_facing_x(model: _FacingPatch, yz: np.ndarray,
                    config: ProximalRoundingConfig) -> np.ndarray:
    """Return the inward neighbour surface X at crown Y/Z query positions."""
    yz = np.asarray(yz, dtype=float)
    k = max(3, min(int(config.facing_k), len(model.points)))
    distance, index = model.tree_yz.query(
        yz,
        k=k,
        distance_upper_bound=float(config.facing_query_radius_mm),
    )
    if k == 1:
        distance = distance[:, None]
        index = index[:, None]
    out = np.full(len(yz), np.nan, dtype=float)
    for i, (dist_i, idx_i) in enumerate(zip(distance, index)):
        valid = np.isfinite(dist_i) & (idx_i < len(model.points))
        if int(valid.sum()) < 3:
            continue
        x = model.points[idx_i[valid], 0]
        # Mesial neighbour is +X: its inward/facing surface is the low-X side.
        # Distal neighbour is -X: its inward/facing surface is the high-X side.
        q = float(config.facing_quantile if model.sign > 0
                  else 1.0 - config.facing_quantile)
        out[i] = float(np.quantile(x, q))
    return out


def _green_bl_reference(models: Mapping[str, _FacingPatch | None],
                        local_margin: np.ndarray,
                        config: ProximalRoundingConfig) -> dict:
    """Build a soft bucco-lingual crown envelope from available neighbours."""
    usable = [m for m in models.values() if m is not None]
    margin_low, margin_high = np.quantile(local_margin[:, 1], [0.04, 0.96])
    margin_mid = float(0.5 * (margin_low + margin_high))

    if usable:
        mean_mid = float(np.mean([m.y_mid for m in usable]))
        mean_width = float(np.mean([m.bl_width for m in usable]))
        center = 0.72 * margin_mid + 0.28 * mean_mid
        half = 0.5 * mean_width * float(config.bl_width_scale)
        target_low = center - half
        target_high = center + half
        source = "neighbor_patch_robust_bl_width"
    else:
        mean_width = float(margin_high - margin_low)
        target_low = float(margin_low)
        target_high = float(margin_high)
        source = "margin_only_no_neighbor"

    # The crown must never be narrower than the cervical envelope plus a small
    # natural emergence allowance.  Movement is still outward-only downstream.
    target_low = min(float(target_low), float(margin_low - config.emergence_extra_mm))
    target_high = max(float(target_high), float(margin_high + config.emergence_extra_mm))
    return {
        "target_low_y_mm": float(target_low),
        "target_high_y_mm": float(target_high),
        "target_width_mm": float(target_high - target_low),
        "neighbor_mean_bl_width_mm": mean_width,
        "source": source,
    }


def round_proximal_contacts(
    mesh,
    margin_points,
    frame: PatientFrame,
    *,
    neighbors: Mapping[str, object | None] | None,
    sectional_config: SectionalV1Config,
    config: ProximalRoundingConfig | None = None,
):
    """Generate broad neighbour-facing RED bands + smooth GREEN side-wall lofts.

    Despite the historical function name, this is no longer a ray-only rounding
    tweak.  The confirmed neighbour triangle patch is the contact truth source.
    Only local X/Y coordinates of middle GREEN rings can move:

    * side ring 0 (Margin) is immutable;
    * last side ring + every BLUE cap vertex are immutable;
    * every PatientFrame-Z coordinate is immutable;
    * missing/ABSENT neighbour sides never get an invented contact.
    """
    config = config or ProximalRoundingConfig()
    config.validate()
    sectional_config.validate()
    validate_patient_frame(frame)

    margin = _margin(margin_points)
    count = int(len(margin))
    level_count = int(len(sectional_config.side_levels))
    if len(mesh.vertices) <= level_count * count:
        raise SectionalOuterError("sectional topology too small for proximal surface pass")

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
    models = {
        side: _build_patch(neighbors.get(side), frame, side, config)
        for side in ("mesial", "distal")
    }
    levels = np.asarray(sectional_config.side_levels, dtype=float)
    bl_reference = _green_bl_reference(models, local_margin, config)

    movement = {
        side: {"vertices": 0, "maximum_mm": 0.0, "sum_mm": 0.0}
        for side in ("mesial", "distal")
    }
    bl_moved_vertices = 0
    bl_max_move = 0.0

    margin_x_low = float(np.min(local_margin[:, 0]))
    margin_x_high = float(np.max(local_margin[:, 0]))

    # Build the GREEN loft ring-by-ring.  The weights are smooth in both angular
    # and vertical directions, so the resulting wall is rounded rather than a
    # set of independent radial columns.
    for ring_index in range(1, level_count - 1):
        s = float(levels[ring_index])
        ids = np.arange(ring_index * count, (ring_index + 1) * count, dtype=np.int64)
        current = local[ids].copy()
        delta_xy = np.zeros((count, 2), dtype=float)

        # GREEN bucco-lingual width: follow average neighbour width while keeping
        # a cervical-to-occlusal taper.  Only expand; never thin the safe baseline.
        vertical_bl = float(np.sin(np.pi * s) ** 1.25)
        abs_y_dir = np.abs(dirs[:, 1])
        bl_sector = _smoothstep(
            (abs_y_dir - float(config.bl_sector_start))
            / max(1e-6, 1.0 - float(config.bl_sector_start)))
        bl_weight = vertical_bl * bl_sector * float(config.bl_strength)
        positive = dirs[:, 1] >= 0.0
        room_hi = np.maximum(float(bl_reference["target_high_y_mm"]) - current[:, 1], 0.0)
        room_lo = np.maximum(current[:, 1] - float(bl_reference["target_low_y_mm"]), 0.0)
        room_hi = np.minimum(room_hi, float(config.bl_max_expansion_mm))
        room_lo = np.minimum(room_lo, float(config.bl_max_expansion_mm))
        dy = np.where(positive, room_hi, -room_lo) * bl_weight
        delta_xy[:, 1] += dy
        moved_bl = np.abs(dy) > 1e-5
        bl_moved_vertices += int(moved_bl.sum())
        if moved_bl.any():
            bl_max_move = max(bl_max_move, float(np.max(np.abs(dy[moved_bl]))))

        # RED contact band per confirmed side: query the actual neighbour inward
        # surface at each Y/Z and drive a broad part of the crown to target gap.
        for side in ("mesial", "distal"):
            model = models[side]
            if model is None:
                continue
            sign = float(model.sign)
            probe_y = current[:, 1] + delta_xy[:, 1]
            probe_yz = np.c_[probe_y, current[:, 2]]
            face_x = _query_facing_x(model, probe_yz, config)
            finite = np.isfinite(face_x)

            y_band = (
                (probe_y >= model.y_low - float(config.contact_band_pad_mm))
                & (probe_y <= model.y_high + float(config.contact_band_pad_mm))
            )
            direction_score = dirs[:, 0] * sign
            angular = _smoothstep(
                (direction_score - float(config.proximal_sector_start))
                / max(1e-6, 1.0 - float(config.proximal_sector_start)))
            vertical = float(np.exp(
                -0.5 * ((s - float(sectional_config.contact_height_ratio))
                        / float(config.vertical_sigma)) ** 2))

            target_x = face_x - sign * float(sectional_config.proximal_gap_mm)
            outward_room = sign * (target_x - current[:, 0])
            # Hard Crown 1.0 growth limit measured from the confirmed Margin.
            cap_x = (margin_x_high + float(sectional_config.max_proximal_extension_mm)
                     if sign > 0 else
                     margin_x_low - float(sectional_config.max_proximal_extension_mm))
            cap_room = sign * (cap_x - current[:, 0])
            room = np.minimum(np.maximum(outward_room, 0.0), np.maximum(cap_room, 0.0))
            evidence = finite & y_band & (angular > 0.0) & (room > 1e-6)

            raw = np.zeros(count, dtype=float)
            raw[evidence] = (
                room[evidence]
                * angular[evidence]
                * vertical
                * float(config.contact_strength)
            )
            smoothed = _masked_circular_smooth(
                raw,
                evidence,
                passes=int(config.smoothing_passes),
                alpha=float(config.smoothing_alpha),
            )
            smoothed = np.minimum(smoothed, room)
            smoothed[~evidence] = 0.0
            delta_xy[:, 0] += sign * smoothed

            moved = smoothed > 1e-5
            movement[side]["vertices"] += int(moved.sum())
            if moved.any():
                movement[side]["maximum_mm"] = max(
                    movement[side]["maximum_mm"], float(np.max(smoothed[moved])))
                movement[side]["sum_mm"] += float(np.sum(smoothed[moved]))

        local[ids, :2] += delta_xy

    # Evaluate the real contact band on the ring closest to anatomical contact
    # height.  This is a surface-to-surface gap test, not a radial-ray proxy.
    contact_ring_index = int(np.argmin(np.abs(
        levels - float(sectional_config.contact_height_ratio))))
    contact_ring_index = int(np.clip(contact_ring_index, 1, level_count - 2))
    contact_ids = np.arange(
        contact_ring_index * count,
        (contact_ring_index + 1) * count,
        dtype=np.int64,
    )
    contact_ring = local[contact_ids]
    side_reports: dict[str, dict] = {}

    for side in ("mesial", "distal"):
        model = models[side]
        if neighbors.get(side) is None:
            side_reports[side] = {
                "status": "NOT_ASSESSED",
                "reason": "NO_CONFIRMED_NEIGHBOR",
                "mechanism": "no_contact_is_invented_for_absent_side",
            }
            continue
        if model is None:
            side_reports[side] = {
                "status": "REVIEW_REQUIRED",
                "reason": "NEIGHBOR_PATCH_TOO_SMALL_OR_NARROW",
                "mechanism": "confirmed_patch_could_not_define_broad_facing_band",
            }
            continue

        face_x = _query_facing_x(model, contact_ring[:, 1:3], config)
        finite = np.isfinite(face_x)
        gap = model.sign * (face_x - contact_ring[:, 0])
        in_bl_band = (
            (contact_ring[:, 1] >= model.y_low - float(config.contact_band_pad_mm))
            & (contact_ring[:, 1] <= model.y_high + float(config.contact_band_pad_mm))
        )
        side_sector = dirs[:, 0] * model.sign > float(config.proximal_sector_start)
        contact_like = (
            finite & in_bl_band & side_sector
            & (gap >= -1e-4)
            & (gap <= float(sectional_config.proximal_gap_mm)
                    + float(config.contact_tolerance_mm))
        )
        contact_count = int(contact_like.sum())
        contiguous = _longest_circular_run(contact_like)
        if contact_like.any():
            contact_band_width = float(
                np.max(contact_ring[contact_like, 1])
                - np.min(contact_ring[contact_like, 1]))
            contact_gap_p50 = float(np.quantile(gap[contact_like], 0.50))
            contact_gap_p95 = float(np.quantile(gap[contact_like], 0.95))
        else:
            contact_band_width = 0.0
            contact_gap_p50 = None
            contact_gap_p95 = None

        # Roundness diagnostic on the actual XY contact-ring curve.
        xy = contact_ring[:, :2]
        second = np.linalg.norm(
            np.roll(xy, 1, axis=0) - 2.0 * xy + np.roll(xy, -1, axis=0),
            axis=1,
        )
        sector_mask = finite & side_sector
        roundness_p95 = (
            float(np.quantile(second[sector_mask], 0.95))
            if sector_mask.any() else None
        )
        passed = (
            contact_count >= int(config.minimum_contact_rays)
            and contiguous >= int(config.minimum_contiguous_rays)
        )
        moved_vertices = int(movement[side]["vertices"])
        side_reports[side] = {
            "status": "PASS_ENGINEERING" if passed else "REVIEW_REQUIRED",
            "neighbor_point_count": int(len(model.points)),
            "facing_patch_bl_width_mm": float(model.bl_width),
            "facing_patch_y_range_mm": [float(model.y_low), float(model.y_high)],
            "facing_patch_z_range_mm": [float(model.z_low), float(model.z_high)],
            "surface_query_coverage": int(finite.sum()),
            "rounded_contact_ray_count": contact_count,
            "longest_contiguous_contact_run": contiguous,
            "contact_band_width_bl_mm": contact_band_width,
            "contact_gap_p50_mm": contact_gap_p50,
            "contact_gap_p95_mm": contact_gap_p95,
            "contact_ring_level": float(levels[contact_ring_index]),
            "moved_side_vertex_count": moved_vertices,
            "maximum_outward_move_mm": float(movement[side]["maximum_mm"]),
            "mean_outward_move_mm": (
                float(movement[side]["sum_mm"]) / moved_vertices
                if moved_vertices else 0.0),
            "xy_roundness_second_diff_p95_mm": roundness_p95,
            "mechanism": "neighbor_facing_patch_surface_plus_broad_contact_band",
        }

    # Hard invariants: RED Margin, BLUE occlusal rim/cap and every axial height
    # remain untouched.  If any of these regress, fail instead of publishing.
    margin_unchanged = bool(np.allclose(local[:count], original_local[:count], atol=1e-12))
    protected_start = (level_count - 1) * count
    protected_unchanged = bool(np.allclose(
        local[protected_start:], original_local[protected_start:], atol=1e-12))
    axial_heights_unchanged = bool(np.allclose(
        local[:, 2], original_local[:, 2], atol=1e-12))
    if not margin_unchanged:
        raise SectionalOuterError("proximal surface pass modified the hard Margin boundary")
    if not protected_unchanged:
        raise SectionalOuterError("proximal surface pass modified the BLUE occlusal rim/cap")
    if not axial_heights_unchanged:
        raise SectionalOuterError("proximal surface pass modified occlusal/axial heights")

    import trimesh
    rounded = trimesh.Trimesh(
        vertices=_world(local, frame),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=False,
    )
    if not rounded.is_winding_consistent:
        raise SectionalOuterError("proximal surface pass broke face winding")
    if np.any(rounded.area_faces < 1e-10):
        raise SectionalOuterError("proximal surface pass created degenerate faces")

    assessed = [r["status"] for r in side_reports.values()
                if r.get("status") != "NOT_ASSESSED"]
    overall = (
        "PASS_ENGINEERING"
        if assessed and all(v == "PASS_ENGINEERING" for v in assessed)
        else "REVIEW_REQUIRED" if assessed else "NOT_ASSESSED"
    )
    report = {
        "implementation": "geometry.crown_outer.proximal_rounding/2",
        "status": overall,
        "margin_preserved": margin_unchanged,
        "blue_occlusal_vertices_preserved": protected_unchanged,
        "axial_heights_preserved": axial_heights_unchanged,
        "side_levels": [float(v) for v in levels],
        "buccolingual_envelope": {
            **bl_reference,
            "moved_vertex_count": int(bl_moved_vertices),
            "maximum_expansion_mm": float(bl_max_move),
        },
        "per_side": side_reports,
        "surface_goal": "neighbor-facing broad contact bands + rounded GREEN buccolingual side-wall loft",
    }
    return rounded, report
