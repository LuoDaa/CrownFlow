"""Margin input contract shared by Crown 1.0 regression tooling.

The sectional generator intentionally stays strict: it must never silently turn a
bad loop into a crown boundary.  This module performs only two safe operations
before generation:

1. remove consecutive duplicate / near-zero segments from an otherwise ordered
   loop;
2. reject a loop whose projected footprint is implausibly small relative to the
   prepared support, or whose projection self-intersects.

It does *not* invent missing Margin geometry and does not smooth or resnap points.
"""
from __future__ import annotations

import numpy as np


class MarginContractError(ValueError):
    pass


def _as_points(points) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3:
        raise MarginContractError("Margin must be an Nx3 array")
    if not np.isfinite(value).all():
        raise MarginContractError("Margin contains non-finite coordinates")
    return value


def sanitize_margin_loop(points, *, tolerance_mm: float = 1e-4,
                         minimum_points: int = 12):
    """Remove only consecutive duplicate points; preserve ordering and geometry."""
    pts = _as_points(points)
    if tolerance_mm <= 0:
        raise MarginContractError("tolerance_mm must be > 0")
    input_count = int(len(pts))
    if input_count == 0:
        raise MarginContractError("Margin is empty")

    # Common closed-polyline encoding repeats point 0 at the end.  Keep the open
    # ring representation expected by the geometry modules.
    closing_duplicate_removed = False
    if len(pts) > 1 and np.linalg.norm(pts[-1] - pts[0]) <= tolerance_mm:
        pts = pts[:-1]
        closing_duplicate_removed = True

    kept = []
    removed = 0
    for point in pts:
        if not kept or np.linalg.norm(point - kept[-1]) > tolerance_mm:
            kept.append(point)
        else:
            removed += 1
    clean = np.asarray(kept, dtype=float)

    # A second closing duplicate can be exposed after collapsing a repeated tail.
    if len(clean) > 1 and np.linalg.norm(clean[-1] - clean[0]) <= tolerance_mm:
        clean = clean[:-1]
        removed += 1
        closing_duplicate_removed = True

    if len(clean) < minimum_points:
        raise MarginContractError(
            "Margin has too few unique ordered points after deduplication: %d < %d"
            % (len(clean), minimum_points))
    segment = np.linalg.norm(np.roll(clean, -1, axis=0) - clean, axis=1)
    if float(segment.min()) <= tolerance_mm:
        raise MarginContractError("Margin still contains a degenerate adjacent segment")

    report = {
        "input_point_count": input_count,
        "output_point_count": int(len(clean)),
        "removed_consecutive_duplicates": int(removed),
        "closing_duplicate_removed": bool(closing_duplicate_removed),
        "tolerance_mm": float(tolerance_mm),
        "minimum_segment_mm": float(segment.min()),
        "perimeter_3d_mm": float(segment.sum()),
        "geometry_changed_beyond_duplicate_removal": False,
    }
    return clean, report


def _basis(axis):
    z = np.asarray(axis, dtype=float)
    if z.shape != (3,) or not np.isfinite(z).all() or np.linalg.norm(z) < 1e-9:
        raise MarginContractError("insertion axis is invalid")
    z = z / np.linalg.norm(z)
    seed = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    x = seed - z * float(seed @ z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    return x, y, z


def _project(points, axis):
    x, y, _ = _basis(axis)
    pts = _as_points(points)
    return np.c_[pts @ x, pts @ y]


def _polygon_area(xy: np.ndarray) -> float:
    return 0.5 * abs(float(np.sum(
        xy[:, 0] * np.roll(xy[:, 1], -1)
        - np.roll(xy[:, 0], -1) * xy[:, 1])))


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _segments_intersect(a, b, c, d, *, eps=1e-10) -> bool:
    # Proper / endpoint intersection in 2D. Adjacent ring edges are excluded by
    # the caller, so an endpoint hit here is still a real self-intersection.
    ab = b - a
    cd = d - c
    o1 = _cross2(ab, c - a)
    o2 = _cross2(ab, d - a)
    o3 = _cross2(cd, a - c)
    o4 = _cross2(cd, b - c)
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and \
       ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)):
        return True

    def on_segment(p, q, r):
        return (min(p[0], q[0]) - eps <= r[0] <= max(p[0], q[0]) + eps
                and min(p[1], q[1]) - eps <= r[1] <= max(p[1], q[1]) + eps)

    return ((abs(o1) <= eps and on_segment(a, b, c))
            or (abs(o2) <= eps and on_segment(a, b, d))
            or (abs(o3) <= eps and on_segment(c, d, a))
            or (abs(o4) <= eps and on_segment(c, d, b)))


def projected_self_intersections(points, axis) -> int:
    xy = _project(points, axis)
    n = len(xy)
    count = 0
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        for j in range(i + 1, n):
            # Same or adjacent edges share a legitimate vertex.
            if j == i or j == (i + 1) % n or i == (j + 1) % n:
                continue
            # First and last edges are also adjacent.
            if i == 0 and j == n - 1:
                continue
            c, d = xy[j], xy[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                count += 1
    return int(count)


def assess_margin_against_support(points, support_vertices, insertion_axis, *,
                                  minimum_axis_span_ratio: float = 0.35,
                                  minimum_projected_area_ratio: float = 0.16):
    """Reject tiny / self-crossing loops relative to the prep footprint.

    Ratios intentionally use the *same support* rather than hard-coded molar mm
    dimensions, so the gate is scale-aware and remains useful for other teeth.
    Support span uses 2--98 percentiles to ignore isolated scanner triangles.
    """
    margin = _as_points(points)
    support = _as_points(support_vertices)
    if len(support) < 16:
        raise MarginContractError("support has too few vertices for Margin plausibility")
    mxy = _project(margin, insertion_axis)
    sxy = _project(support, insertion_axis)
    slo = np.quantile(sxy, 0.02, axis=0)
    shi = np.quantile(sxy, 0.98, axis=0)
    support_span = np.maximum(shi - slo, 1e-6)
    margin_span = np.ptp(mxy, axis=0)
    axis_ratio = margin_span / support_span
    area = _polygon_area(mxy)
    support_bbox_area = float(np.prod(support_span))
    area_ratio = float(area / max(support_bbox_area, 1e-9))
    crossings = projected_self_intersections(margin, insertion_axis)
    accepted = bool(
        crossings == 0
        and float(np.min(axis_ratio)) >= float(minimum_axis_span_ratio)
        and area_ratio >= float(minimum_projected_area_ratio)
    )
    return {
        "status": "PASS_ENGINEERING" if accepted else "REVIEW_REQUIRED",
        "accepted": accepted,
        "projected_margin_span_mm": margin_span.tolist(),
        "robust_support_span_mm": support_span.tolist(),
        "axis_span_ratio": axis_ratio.tolist(),
        "minimum_axis_span_ratio": float(minimum_axis_span_ratio),
        "projected_margin_area_mm2": float(area),
        "support_robust_bbox_area_mm2": support_bbox_area,
        "projected_area_ratio": area_ratio,
        "minimum_projected_area_ratio": float(minimum_projected_area_ratio),
        "projected_self_intersections": int(crossings),
        "reason": None if accepted else "MARGIN_IMPLAUSIBLE_RELATIVE_TO_PREP",
    }
