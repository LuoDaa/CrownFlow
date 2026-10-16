"""Section-driven Crown 1.0 outer generator.

This module implements the geometry model sketched in the Crown 1.0 cross-section:

    RED   = preparation / intaglio safety envelope (hard minimum-wall floor)
    GREEN = lateral emergence + broad proximal contact envelope
    BLUE  = occlusal cap constrained by antagonist ceiling and neighbor ridge heights

The generated object is an *outer surface* with one open cervical boundary.  The
bottom ring is the confirmed patient Margin exactly; the existing finalize path is
still responsible for directional blockout, cement gap and final intaglio boolean.

Design goals
------------
* No world-axis guesses: all calculations are in the confirmed PatientFrame.
* No single-vertex "contact horn": mesial/distal growth is a broad angular sector.
* No floating short crown: the blue cap follows a local antagonist ceiling.
* Fail loud when the confirmed geometry cannot satisfy minimum wall/occlusal space.
* Tooth-library anatomy is not used to decide the envelope.  A very small analytic
  posterior relief is used only to avoid a perfectly flat cap; local antagonist
  geometry remains the hard ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .patient_frame import PatientFrame, validate_patient_frame


class SectionalOuterError(ValueError):
    """Input/context cannot produce a safe Sectional Crown 1.0 outer."""


@dataclass(frozen=True)
class SectionalV1Config:
    min_wall_mm: float = 0.50
    proximal_gap_mm: float = 0.02
    occlusal_gap_mm: float = 0.05

    # GREEN lateral wall.
    emergence_bulge_mm: float = 0.80
    contact_height_ratio: float = 0.66
    top_inset_mm: float = 0.28
    neighbor_sector_cos: float = 0.45
    neighbor_ray_band_mm: float = 1.50
    max_proximal_extension_mm: float = 2.80

    # BLUE cap.
    cap_rhos: tuple[float, ...] = (0.72, 0.45, 0.22)
    ceiling_radius_mm: float = 1.00
    ceiling_k: int = 24
    ceiling_quantile: float = 0.10
    minimum_ceiling_coverage: float = 0.60
    noncontact_clearance_mm: float = 0.12
    cusp_relief_mm: float = 0.14
    fossa_depth_mm: float = 0.22
    cap_smooth_iterations: int = 2

    # Side loft levels from Margin (0) to occlusal rim (1).
    side_levels: tuple[float, ...] = (0.0, 0.18, 0.38, 0.62, 0.82, 1.0)

    def validate(self) -> None:
        if self.min_wall_mm <= 0:
            raise SectionalOuterError("minimum wall thickness must be > 0")
        if self.proximal_gap_mm < 0 or self.occlusal_gap_mm < 0:
            raise SectionalOuterError("contact clearances must be >= 0")
        if not (0.45 <= self.contact_height_ratio <= 0.85):
            raise SectionalOuterError("contact_height_ratio outside supported range")
        if not (0.0 < self.neighbor_sector_cos < 0.95):
            raise SectionalOuterError("neighbor_sector_cos outside supported range")
        if self.max_proximal_extension_mm <= 0:
            raise SectionalOuterError("max_proximal_extension_mm must be > 0")
        if not (0.0 < self.minimum_ceiling_coverage <= 1.0):
            raise SectionalOuterError("minimum_ceiling_coverage must be in (0,1]")
        if len(self.side_levels) < 3 or self.side_levels[0] != 0 or self.side_levels[-1] != 1:
            raise SectionalOuterError("side_levels must start at 0 and end at 1")
        if any(b <= a for a, b in zip(self.side_levels[:-1], self.side_levels[1:])):
            raise SectionalOuterError("side_levels must be strictly increasing")
        if any(not (0 < r < 1) for r in self.cap_rhos):
            raise SectionalOuterError("cap_rhos must lie in (0,1)")
        if any(b >= a for a, b in zip(self.cap_rhos[:-1], self.cap_rhos[1:])):
            raise SectionalOuterError("cap_rhos must be strictly decreasing")


def _as_margin(points) -> np.ndarray:
    margin = np.asarray(points, dtype=float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 12:
        raise SectionalOuterError("Margin must contain at least 12 finite XYZ points")
    if not np.isfinite(margin).all():
        raise SectionalOuterError("Margin contains non-finite coordinates")
    if np.linalg.norm(margin[0] - margin[-1]) < 1e-8:
        margin = margin[:-1]
    if len(margin) < 12:
        raise SectionalOuterError("Margin is too short after closing-point removal")
    if np.min(np.linalg.norm(np.roll(margin, -1, axis=0) - margin, axis=1)) < 1e-6:
        raise SectionalOuterError("Margin contains duplicate/degenerate adjacent points")
    return margin


def _local(points, frame: PatientFrame) -> np.ndarray:
    return (np.asarray(points, dtype=float) - np.asarray(frame.origin, float)) @ frame.matrix()


def _world(points, frame: PatientFrame) -> np.ndarray:
    return np.asarray(points, dtype=float) @ frame.matrix().T + np.asarray(frame.origin, float)


def _ccw_margin_local(margin_local: np.ndarray) -> np.ndarray:
    xy = np.asarray(margin_local[:, :2], float)
    area2 = float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
    if abs(area2) < 1e-8:
        raise SectionalOuterError("Margin projection has near-zero signed area")
    return margin_local if area2 > 0 else margin_local[::-1].copy()


def _mesh_points(mesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    centers = np.asarray(mesh.triangles_center, dtype=float)
    return np.vstack((vertices, centers))


def _support_radial(prep_local: np.ndarray, dirs: np.ndarray, margin_r: np.ndarray) -> np.ndarray:
    """Approximate RED radial safety envelope for each Margin ray.

    The actual intaglio is constructed later by the existing blockout+gap path;
    here we only prevent the outer wall from being generated inside the prep.
    """
    xy = prep_local[:, :2]
    radial = np.linalg.norm(xy, axis=1)
    valid = radial > 1e-8
    unit = np.zeros_like(xy)
    unit[valid] = xy[valid] / radial[valid, None]
    out = np.asarray(margin_r, dtype=float).copy()
    for i, direction in enumerate(dirs):
        # Wide angular support is deliberate: scanner tessellation is irregular.
        aligned = valid & ((unit @ direction) > 0.94) & (prep_local[:, 2] >= -0.75)
        if aligned.any():
            out[i] = max(out[i], float(np.max(xy[aligned] @ direction)))
    return out


def _neighbor_local(mesh, frame: PatientFrame, side: str) -> dict | None:
    if mesh is None:
        return None
    points = _local(_mesh_points(mesh), frame)
    sign = 1.0 if side == "mesial" else -1.0
    points = points[(points[:, 0] * sign) > 0]
    if len(points) < 8:
        return None
    return {
        "points": points,
        "top_mm": float(np.quantile(points[:, 2], 0.95)),
        "count": int(len(points)),
    }


def _neighbor_limits(stats: dict | None, dirs: np.ndarray, margin_r: np.ndarray,
                     side: str, config: SectionalV1Config) -> np.ndarray:
    """Facing neighbor surface radius for each ray in the mesial/distal sector."""
    limits = np.full(len(dirs), np.inf, dtype=float)
    if stats is None:
        return limits
    points = np.asarray(stats["points"], float)
    xy = points[:, :2]
    sign = 1.0 if side == "mesial" else -1.0
    for i, direction in enumerate(dirs):
        if float(direction[0]) * sign < config.neighbor_sector_cos:
            continue
        projection = xy @ direction
        perpendicular = np.linalg.norm(xy - projection[:, None] * direction[None, :], axis=1)
        candidates = ((projection > margin_r[i] + 0.10)
                      & (perpendicular <= config.neighbor_ray_band_mm))
        if candidates.any():
            # Low quantile follows the facing surface but rejects isolated triangles.
            limits[i] = float(np.quantile(projection[candidates], 0.05)
                              - config.proximal_gap_mm)
    return limits


class _Ceiling:
    """Local antagonist ceiling queried along PatientFrame +Z."""

    def __init__(self, antagonist, frame: PatientFrame, minimum_z: float,
                 config: SectionalV1Config):
        local = _local(_mesh_points(antagonist), frame)
        local = local[local[:, 2] > minimum_z - 0.75]
        if len(local) < 16:
            raise SectionalOuterError("antagonist has no usable points above the prep")
        self.local = local
        self.tree = cKDTree(local[:, :2])
        self.config = config

    def __call__(self, xy) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.ndim == 1:
            xy = xy[None, :]
        k = max(1, min(int(self.config.ceiling_k), len(self.local)))
        distance, index = self.tree.query(
            xy, k=k, distance_upper_bound=float(self.config.ceiling_radius_mm))
        if k == 1:
            distance = distance[:, None]
            index = index[:, None]
        values = np.full(len(xy), np.nan, dtype=float)
        for i, (dist_i, idx_i) in enumerate(zip(distance, index)):
            valid = np.isfinite(dist_i) & (idx_i < len(self.local))
            if not valid.any():
                continue
            z = self.local[idx_i[valid], 2]
            q = float(np.clip(self.config.ceiling_quantile, 0.0, 1.0))
            values[i] = float(np.quantile(z, q))
        return values


def _anatomy_harmonic(fdi: int | None) -> int:
    if fdi is None:
        return 4
    digit = int(fdi) % 10
    if digit in (6, 7, 8):
        return 4
    if digit in (4, 5):
        return 2
    return 1


def _smoothstep(t):
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    from collections import Counter
    counter: Counter = Counter()
    for a, b, c in np.asarray(faces, dtype=np.int64):
        for u, v in ((a, b), (b, c), (c, a)):
            counter[(min(int(u), int(v)), max(int(u), int(v)))] += 1
    return [edge for edge, count in counter.items() if count == 1]


def build_sectional_outer(
    margin_points,
    frame: PatientFrame,
    prep_mesh,
    *,
    neighbors: Mapping[str, object | None] | None = None,
    antagonist_mesh=None,
    fdi: int | None = None,
    config: SectionalV1Config | None = None,
):
    """Generate a Crown 1.0 outer surface from the red/green/blue constraints.

    Parameters
    ----------
    margin_points:
        Confirmed ordered patient Margin.  The generated bottom boundary uses
        these exact points (possibly reversed only to make the local ring CCW).
    frame:
        Confirmed PatientFrame.  No world-axis inference is performed here.
    prep_mesh:
        Prepared tooth / conditioned abutment used for the RED safety envelope.
    neighbors:
        Optional ``{"mesial": mesh, "distal": mesh}`` confirmed tooth patches.
        Missing sides are not guessed.
    antagonist_mesh:
        Static-pose antagonist already transformed into the working-jaw frame.
        When supplied it is the hard BLUE ceiling.

    Returns
    -------
    (mesh, report)
        Open cervical outer surface plus a machine-readable geometry report.
    """
    config = config or SectionalV1Config()
    config.validate()
    validate_patient_frame(frame)
    margin = _as_margin(margin_points)
    local_margin = _ccw_margin_local(_local(margin, frame))
    count = len(local_margin)

    center_xy = local_margin[:, :2].mean(axis=0)
    radial_xy = local_margin[:, :2] - center_xy
    margin_r = np.linalg.norm(radial_xy, axis=1)
    if float(np.min(margin_r)) < 0.35:
        raise SectionalOuterError("Margin is not star-shaped around its confirmed frame origin")
    dirs = radial_xy / margin_r[:, None]

    prep_local = _local(np.asarray(prep_mesh.vertices, float), frame)
    if len(prep_local) < 16 or not np.isfinite(prep_local).all():
        raise SectionalOuterError("prep mesh is empty or invalid")
    prep_top = float(np.quantile(prep_local[:, 2], 0.995))
    minimum_top = prep_top + float(config.min_wall_mm)

    # RED: minimum radial outer envelope outside the prep.
    support_r = _support_radial(prep_local - np.r_[center_xy, 0.0], dirs, margin_r)
    minimum_body_r = support_r + float(config.min_wall_mm)
    default_body_r = np.maximum(margin_r + float(config.emergence_bulge_mm),
                                minimum_body_r + 0.10)

    # GREEN: grow a *broad sector* toward each confirmed neighbor.
    neighbors = dict(neighbors or {})
    stats = {side: _neighbor_local(neighbors.get(side), frame, side)
             for side in ("mesial", "distal")}
    limits = {side: _neighbor_limits(stats[side], dirs, margin_r, side, config)
              for side in ("mesial", "distal")}
    body_r = default_body_r.copy()
    green_report: dict[str, dict] = {}
    hard_radial_conflict = np.zeros(count, dtype=bool)

    for side in ("mesial", "distal"):
        sign = 1.0 if side == "mesial" else -1.0
        limit = limits[side]
        finite = np.isfinite(limit)
        weight = np.clip((dirs[:, 0] * sign - config.neighbor_sector_cos)
                         / (1.0 - config.neighbor_sector_cos), 0.0, 1.0)
        if stats[side] is None:
            green_report[side] = {"status": "NOT_ASSESSED", "reason": "NO_CONFIRMED_NEIGHBOR"}
            continue

        radial_conflict = finite & (limit < minimum_body_r - 1e-6)
        hard_radial_conflict |= radial_conflict
        reachable = np.minimum(limit, margin_r + float(config.max_proximal_extension_mm))
        target = np.maximum(reachable, minimum_body_r)
        use = finite & (weight > 0)
        body_r[use] = ((1.0 - weight[use]) * body_r[use]
                       + weight[use] * target[use])
        near = use & (weight >= 0.35) & (np.abs(body_r - limit) <= 0.18)
        green_report[side] = {
            "status": "PASS_ENGINEERING" if int(near.sum()) >= 3 else "REVIEW_REQUIRED",
            "neighbor_point_count": int(stats[side]["count"]),
            "neighbor_top_mm": float(stats[side]["top_mm"]),
            "ray_coverage": int(finite.sum()),
            "broad_contact_ray_count": int(near.sum()),
            "radial_conflict_ray_count": int(radial_conflict.sum()),
            "maximum_allowed_extension_mm": float(config.max_proximal_extension_mm),
            "mechanism": "broad_mesial_distal_sector_not_single_vertex_target",
        }

    # Keep the occlusal rim slightly inside the broad contact contour, creating a
    # normal embrasure/marginal-ridge transition instead of a vertical box wall.
    top_r = np.maximum(minimum_body_r,
                       np.maximum(margin_r + 0.12, body_r - float(config.top_inset_mm)))
    top_xy = center_xy + top_r[:, None] * dirs

    ceiling = None
    boundary_ceiling = np.full(count, np.nan, dtype=float)
    if antagonist_mesh is not None:
        ceiling = _Ceiling(antagonist_mesh, frame, minimum_top, config)
        boundary_ceiling = ceiling(top_xy)
        coverage = float(np.isfinite(boundary_ceiling).mean())
        if coverage < float(config.minimum_ceiling_coverage):
            raise SectionalOuterError(
                "OCCLUSAL_CEILING_COVERAGE_LOW: %.3f < %.3f"
                % (coverage, config.minimum_ceiling_coverage))
    else:
        coverage = 0.0

    allowed_boundary = boundary_ceiling - float(config.occlusal_gap_mm)
    valid_boundary = np.isfinite(allowed_boundary)
    floor_conflict = valid_boundary & (allowed_boundary < minimum_top - 1e-6)
    if int(floor_conflict.sum()) > max(2, int(0.03 * count)):
        raise SectionalOuterError(
            "INSUFFICIENT_OCCLUSAL_SPACE: antagonist ceiling enters minimum-wall floor")

    neighbor_tops = [float(value["top_mm"]) for value in stats.values() if value is not None]
    if valid_boundary.any():
        base_height = float(np.quantile(allowed_boundary[valid_boundary], 0.35))
    elif neighbor_tops:
        base_height = float(np.mean(neighbor_tops))
    else:
        base_height = minimum_top + 2.5
    base_height = max(base_height, minimum_top)

    top_z = np.full(count, base_height, dtype=float)
    for side in ("mesial", "distal"):
        if stats[side] is None:
            continue
        sign = 1.0 if side == "mesial" else -1.0
        weight = np.clip((dirs[:, 0] * sign - config.neighbor_sector_cos)
                         / (1.0 - config.neighbor_sector_cos), 0.0, 1.0)
        # Neighbor marginal ridge is a soft anchor; the antagonist remains hard.
        anchor = max(minimum_top, float(stats[side]["top_mm"]))
        soft_anchor = np.minimum(base_height + 0.9, anchor)
        top_z = (1.0 - weight) * top_z + weight * soft_anchor
    top_z = np.maximum(top_z, minimum_top)
    top_z[valid_boundary] = np.minimum(top_z[valid_boundary],
                                       allowed_boundary[valid_boundary])

    # Side loft (GREEN), starting from the exact confirmed Margin.
    vertices_local: list[list[float]] = []
    side_rings: list[np.ndarray] = []
    levels = np.asarray(config.side_levels, dtype=float)
    for s in levels:
        if s <= config.contact_height_ratio:
            t = _smoothstep(s / config.contact_height_ratio)
            radius = margin_r + (body_r - margin_r) * t
        else:
            t = _smoothstep((s - config.contact_height_ratio)
                            / (1.0 - config.contact_height_ratio))
            radius = body_r + (top_r - body_r) * t
        z = local_margin[:, 2] + s * (top_z - local_margin[:, 2])
        ring = np.c_[center_xy + radius[:, None] * dirs, z]
        ids = np.arange(len(vertices_local), len(vertices_local) + count, dtype=np.int64)
        vertices_local.extend(ring.tolist())
        side_rings.append(ids)

    # BLUE cap: antagonist-driven ceiling + neighbor rim heights + weak posterior relief.
    cap_rings: list[np.ndarray] = [side_rings[-1]]
    theta = np.arctan2(dirs[:, 1], dirs[:, 0])
    harmonic = _anatomy_harmonic(fdi)
    cap_query_records: list[tuple[np.ndarray, np.ndarray]] = []

    for rho in config.cap_rhos:
        xy = center_xy + float(rho) * top_r[:, None] * dirs
        allowed = (ceiling(xy) - float(config.occlusal_gap_mm)
                   if ceiling is not None else np.full(count, np.nan))
        # Radial interpolation keeps the neighbor marginal-ridge boundary coherent.
        boundary_component = top_z * (0.55 + 0.45 * rho)
        center_component = (base_height - 0.12) * (0.45 * (1.0 - rho))
        smooth_base = boundary_component + center_component
        if np.isfinite(allowed).any():
            target = np.where(np.isfinite(allowed),
                              0.45 * smooth_base
                              + 0.55 * (allowed - float(config.noncontact_clearance_mm)),
                              smooth_base)
        else:
            target = smooth_base
        relief = (float(config.cusp_relief_mm) * (rho ** 1.5)
                  * np.cos(float(harmonic) * theta)
                  - float(config.fossa_depth_mm) * ((1.0 - rho) ** 2))
        z = np.maximum(target + relief, minimum_top)
        valid = np.isfinite(allowed)
        if valid.any():
            if np.any(allowed[valid] < minimum_top - 1e-6):
                raise SectionalOuterError("INSUFFICIENT_OCCLUSAL_SPACE inside BLUE cap")
            z[valid] = np.minimum(z[valid], allowed[valid])
        ring = np.c_[xy, z]
        ids = np.arange(len(vertices_local), len(vertices_local) + count, dtype=np.int64)
        vertices_local.extend(ring.tolist())
        cap_rings.append(ids)
        cap_query_records.append((ids, allowed))

    vertices_local_arr = np.asarray(vertices_local, dtype=float)
    inner_ring = cap_rings[-1]
    center_cap_xy = vertices_local_arr[inner_ring, :2].mean(axis=0)
    center_z = float(np.mean(vertices_local_arr[inner_ring, 2]) - config.fossa_depth_mm)
    center_z = max(center_z, minimum_top)
    center_allowed = np.nan
    if ceiling is not None:
        center_allowed = float(ceiling(center_cap_xy[None, :])[0] - config.occlusal_gap_mm)
        if np.isfinite(center_allowed):
            if center_allowed < minimum_top - 1e-6:
                raise SectionalOuterError("INSUFFICIENT_OCCLUSAL_SPACE at central fossa")
            center_z = min(center_z, center_allowed)
    center_id = int(len(vertices_local_arr))
    vertices_local_arr = np.vstack((vertices_local_arr,
                                    [center_cap_xy[0], center_cap_xy[1], center_z]))

    # Small angular smoothing removes scanner noise without flattening the cap.
    for _ in range(int(config.cap_smooth_iterations)):
        previous = vertices_local_arr.copy()
        for ring in cap_rings[1:]:
            z = previous[ring, 2]
            vertices_local_arr[ring, 2] = 0.50 * z + 0.25 * np.roll(z, 1) + 0.25 * np.roll(z, -1)
            vertices_local_arr[ring, 2] = np.maximum(vertices_local_arr[ring, 2], minimum_top)
            if ceiling is not None:
                allowed = ceiling(vertices_local_arr[ring, :2]) - float(config.occlusal_gap_mm)
                valid = np.isfinite(allowed)
                vertices_local_arr[ring[valid], 2] = np.minimum(
                    vertices_local_arr[ring[valid], 2], allowed[valid])

    # Triangulate side rings and the cap.  The bottom Margin remains the only boundary.
    faces: list[list[int]] = []
    for lower, upper in zip(side_rings[:-1], side_rings[1:]):
        for i in range(count):
            j = (i + 1) % count
            faces.append([int(lower[i]), int(lower[j]), int(upper[i])])
            faces.append([int(lower[j]), int(upper[j]), int(upper[i])])
    for outer, inner in zip(cap_rings[:-1], cap_rings[1:]):
        for i in range(count):
            j = (i + 1) % count
            faces.append([int(outer[i]), int(outer[j]), int(inner[i])])
            faces.append([int(outer[j]), int(inner[j]), int(inner[i])])
    innermost = cap_rings[-1]
    for i in range(count):
        j = (i + 1) % count
        faces.append([int(innermost[i]), int(innermost[j]), center_id])

    faces_arr = np.asarray(faces, dtype=np.int64)
    vertices_world = _world(vertices_local_arr, frame)
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces_arr, process=False)

    boundary_edges = _boundary_edges(faces_arr)
    margin_ids = set(map(int, side_rings[0]))
    boundary_ok = (len(boundary_edges) == count
                   and all(a in margin_ids and b in margin_ids for a, b in boundary_edges))
    if not boundary_ok:
        raise SectionalOuterError("generated outer does not have exactly one Margin boundary loop")
    if not mesh.is_winding_consistent:
        raise SectionalOuterError("generated outer has inconsistent face winding")
    if np.any(mesh.area_faces < 1e-10):
        raise SectionalOuterError("generated outer contains degenerate faces")

    # Margin points are exact by construction (possibly reversed only for CCW winding).
    margin_world_generated = vertices_world[side_rings[0]]
    expected_margin_world = _world(local_margin, frame)
    margin_error = np.linalg.norm(margin_world_generated - expected_margin_world, axis=1)

    # Occlusal clearance snapshot on cap vertices.
    cap_vertex_ids = np.unique(np.concatenate(cap_rings))
    if ceiling is not None:
        cap_ceiling = ceiling(vertices_local_arr[cap_vertex_ids, :2])
        valid = np.isfinite(cap_ceiling)
        clearance = cap_ceiling[valid] - vertices_local_arr[cap_vertex_ids[valid], 2]
        blue = {
            "status": "PASS_ENGINEERING" if float(valid.mean()) >= config.minimum_ceiling_coverage else "REVIEW_REQUIRED",
            "ceiling_coverage": float(valid.mean()),
            "clearance_min_mm": float(clearance.min()) if len(clearance) else None,
            "clearance_p05_mm": float(np.quantile(clearance, 0.05)) if len(clearance) else None,
            "clearance_p50_mm": float(np.quantile(clearance, 0.50)) if len(clearance) else None,
            "contact_like_vertex_count": int(np.sum(clearance <= config.occlusal_gap_mm + 0.05)),
            "cap_relief_p95_p05_mm": float(np.quantile(vertices_local_arr[cap_vertex_ids, 2], 0.95)
                                                - np.quantile(vertices_local_arr[cap_vertex_ids, 2], 0.05)),
            "mechanism": "antagonist_local_ceiling_plus_smoothed_posterior_relief",
        }
    else:
        blue = {
            "status": "NOT_ASSESSED",
            "reason": "NO_CONFIRMED_ANTAGONIST",
            "ceiling_coverage": 0.0,
            "cap_relief_p95_p05_mm": float(np.quantile(vertices_local_arr[cap_vertex_ids, 2], 0.95)
                                                - np.quantile(vertices_local_arr[cap_vertex_ids, 2], 0.05)),
        }

    statuses = [value.get("status") for value in green_report.values() if value.get("status")]
    if antagonist_mesh is not None:
        statuses.append(blue["status"])
    overall = "PASS_ENGINEERING"
    if hard_radial_conflict.any():
        overall = "REVIEW_REQUIRED"
    if any(status == "REVIEW_REQUIRED" for status in statuses):
        overall = "REVIEW_REQUIRED"

    report = {
        "implementation": "geometry.crown_outer.sectional_v1/1",
        "status": overall,
        "concept": {
            "red": "prep/intaglio safety envelope + minimum wall",
            "green": "broad lateral emergence + neighbor contact sectors",
            "blue": "antagonist ceiling + neighbor ridge anchors + smooth relief",
        },
        "patient_frame_revision_id": getattr(frame, "revision_id", None),
        "margin": {
            "point_count": int(count),
            "max_error_mm": float(margin_error.max()),
            "rms_error_mm": float(np.sqrt(np.mean(margin_error ** 2))),
            "hard_boundary": True,
        },
        "red_inner_constraint": {
            "prep_top_mm": prep_top,
            "minimum_wall_mm": float(config.min_wall_mm),
            "minimum_occlusal_outer_height_mm": minimum_top,
            "actual_intaglio_stage": "existing_finalize_blockout_plus_gap_boolean",
        },
        "green_lateral": green_report,
        "blue_occlusal": blue,
        "radial_conflict_ray_count": int(hard_radial_conflict.sum()),
        "occlusal_floor_conflict_ray_count": int(floor_conflict.sum()),
        "mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "boundary_edges": int(len(boundary_edges)),
            "boundary_all_on_margin": bool(boundary_ok),
            "winding_consistent": bool(mesh.is_winding_consistent),
        },
    }
    return mesh, report
