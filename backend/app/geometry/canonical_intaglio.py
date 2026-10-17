"""Deterministic CrownFlow 1.0 intaglio ownership + cervical assembly.

This module deliberately does not touch accepted outer anatomy.  It solves two
separate contracts:

1. PreparedSupport owns which clipped cutter component is the clinical intaglio.
2. ConfirmedMargin owns the final cervical boundary and its topology sampling.

The source body meshes remain unchanged.  Boundary vertices may only be split
conformingly on existing edges; all new geometry is confined to the generated
cervical collar.
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.proximity import closest_point

from .crown_outer.collar_body_outer import _boundary_loops, _clip_by_scalar, _conforming_ring
from .crown_outer.thickness import input_qc
from .margin_seam import _polyline_distances


class IntaglioOwnershipError(ValueError):
    def __init__(self, code: str, report: dict):
        self.code = code
        self.report = report
        super().__init__(f"{code}: {report}")


def _sample_rows(points, limit=384):
    p = np.asarray(points, float)
    if len(p) <= limit:
        return p
    ids = np.linspace(0, len(p) - 1, int(limit), dtype=int)
    return p[ids]


def _support_regions(support, curtain):
    v = np.asarray(support.vertices, float)
    h = np.asarray(curtain.axial_above(v), float)
    finite = np.isfinite(h)
    v, h = v[finite], h[finite]
    if not len(v):
        return {"cervical": np.zeros((0, 3)), "axial": np.zeros((0, 3)), "top": np.zeros((0, 3))}, 0.0
    floor = float(np.quantile(h, 0.02))
    ceil = float(np.quantile(h, 0.98))
    span = max(ceil - floor, 1e-6)
    t = np.clip((h - floor) / span, 0.0, 1.0)
    return {
        "cervical": _sample_rows(v[t <= 0.28]),
        "axial": _sample_rows(v[(t > 0.28) & (t < 0.74)]),
        "top": _sample_rows(v[t >= 0.74]),
    }, span


def _coverage(component, points, threshold):
    p = np.asarray(points, float)
    if not len(p):
        return {"samples": 0, "coverage": None, "median_mm": None, "p95_mm": None}
    _cp, d, _ = closest_point(component, p)
    d = np.asarray(d, float)
    finite = np.isfinite(d)
    if not np.any(finite):
        return {"samples": int(len(p)), "coverage": 0.0, "median_mm": None, "p95_mm": None}
    d = d[finite]
    return {
        "samples": int(len(d)),
        "coverage": float(np.mean(d <= float(threshold))),
        "median_mm": float(np.median(d)),
        "p95_mm": float(np.quantile(d, .95)),
    }


def extract_support_owned_intaglio(cutter, support, curtain, transition_mm, params=None):
    """Extract the clinical intaglio from a clipped cutter using support evidence.

    Margin proximity is reported but never owns the decision.  Hard evidence is
    correspondence to prepared support across cervical / axial / top regions plus
    insertion-axis span.  Area is only a weak tie-breaker/sanity prior.
    """
    params = params or {}
    values = -curtain.axial_above(cutter.vertices)
    vertices, faces, _ = _clip_by_scalar(
        cutter.vertices, cutter.faces, values, level=-float(transition_mm)
    )
    clipped = trimesh.Trimesh(vertices, faces, process=False)
    clipped.remove_unreferenced_vertices()
    raw_parts = [p for p in clipped.split(only_watertight=False) if len(p.faces) >= 3]
    if not raw_parts:
        raise IntaglioOwnershipError("INTAGLIO_NO_CLIPPED_COMPONENT", {"component_count": 0})

    # Existing non-planar clipping can produce a second tiny derived opening.
    # Reuse the proven Margin-loop recovery on each candidate, but do not let its
    # Margin score decide which component is the clinical fitting surface.
    from .nonplanar_crown import _recover_single_cervical_loop

    regions, support_span = _support_regions(support, curtain)
    configured = [
        params.get("cement_gap_mm"), params.get("cement_gap_top_mm"),
        params.get("cement_gap_start_mm"),
    ]
    gaps = [float(x) for x in configured if x is not None and np.isfinite(float(x))]
    nominal_gap = max(gaps or [0.06])
    ownership_window = float(np.clip(max(0.45, nominal_gap * 4.0 + 0.15), 0.45, 0.90))

    candidates = []
    recovered_meshes = []
    total_area = float(sum(float(p.area) for p in raw_parts))
    axis = np.asarray(curtain.frame.z, float)
    for index, part in enumerate(raw_parts):
        recovered, recovery = _recover_single_cervical_loop(part, curtain)
        if recovered is None:
            candidates.append({
                "component": index, "eligible": False, "decision": "REJECTED_LOOP_RECOVERY",
                "faces": int(len(part.faces)), "area_mm2": float(part.area),
                "ring_recovery": recovery,
            })
            recovered_meshes.append(None)
            continue
        recovered.remove_unreferenced_vertices()
        loops = _boundary_loops(np.asarray(recovered.faces, np.int64))
        if len(loops) != 1:
            candidates.append({
                "component": index, "eligible": False, "decision": "REJECTED_LOOP_COUNT",
                "faces": int(len(recovered.faces)), "area_mm2": float(recovered.area),
                "boundary_loop_count": int(len(loops)), "ring_recovery": recovery,
            })
            recovered_meshes.append(None)
            continue

        pts = np.asarray(recovered.vertices, float)
        z = pts @ axis
        span = float(z.max() - z.min()) if len(z) else 0.0
        span_fraction = float(span / max(support_span, 1e-6))
        cover = {name: _coverage(recovered, sample, ownership_window)
                 for name, sample in regions.items()}
        valid_cov = [x["coverage"] for x in cover.values() if x["coverage"] is not None]
        total_samples = np.vstack([p for p in regions.values() if len(p)]) if any(len(p) for p in regions.values()) else np.zeros((0, 3))
        total = _coverage(recovered, total_samples, ownership_window)
        loop_pts = pts[np.asarray(loops[0], np.int64)]
        md = _polyline_distances(loop_pts, curtain.points)
        area = float(recovered.area)
        area_fraction = float(area / max(total_area, 1e-12))
        cervical = cover["cervical"]["coverage"] if cover["cervical"]["coverage"] is not None else 1.0
        axial = cover["axial"]["coverage"] if cover["axial"]["coverage"] is not None else 0.0
        top = cover["top"]["coverage"] if cover["top"]["coverage"] is not None else 0.0
        total_cov = total["coverage"] if total["coverage"] is not None else 0.0
        hard_valid = bool(
            total_cov >= 0.40 and axial >= 0.35 and top >= 0.20
            and span_fraction >= 0.45
        )
        # Ownership ordering: worst regional support correspondence first, then
        # total coverage/span.  Area and Margin distance are only weak priors.
        regional_floor = float(min(cervical, axial, top))
        rank = (
            1 if hard_valid else 0,
            regional_floor,
            float(total_cov),
            float(min(span_fraction, 1.5)),
            float(min(area_fraction, 1.0)),
            -float(np.median(md)) if len(md) else -1e9,
        )
        candidates.append({
            "component": index,
            "eligible": True,
            "hard_valid": hard_valid,
            "faces": int(len(recovered.faces)),
            "vertices": int(len(recovered.vertices)),
            "area_mm2": area,
            "area_fraction": area_fraction,
            "support_coverage": {**cover, "total": total},
            "ownership_window_mm": ownership_window,
            "axial_span_mm": span,
            "axial_span_fraction": span_fraction,
            "margin_distance_mm": {
                "median": float(np.median(md)) if len(md) else None,
                "p95": float(np.quantile(md, .95)) if len(md) else None,
            },
            "ring_recovery": recovery,
            "rank": [float(x) for x in rank],
            "decision": "CANDIDATE",
        })
        recovered_meshes.append(recovered)

    eligible = [(tuple(c["rank"]), i) for i, c in enumerate(candidates)
                if c.get("eligible") and recovered_meshes[i] is not None]
    if not eligible:
        raise IntaglioOwnershipError("INTAGLIO_NO_ELIGIBLE_COMPONENT", {"components": candidates})
    eligible.sort(reverse=True)
    selected_i = eligible[0][1]
    selected = recovered_meshes[selected_i]
    selected_report = candidates[selected_i]
    selected_report["decision"] = "SELECTED"
    hard_valid = bool(selected_report.get("hard_valid"))

    report = {
        "status": "PASS_ENGINEERING" if hard_valid else "REVIEW_REQUIRED",
        "method": "PREPARED_SUPPORT_CORRESPONDENCE_V1",
        "owner": "PREPARED_SUPPORT",
        "margin_role": "BOUNDARY_ONLY_NOT_COMPONENT_OWNER",
        "source_faces": int(len(cutter.faces)),
        "clipped_faces": int(len(clipped.faces)),
        "component_count": int(len(raw_parts)),
        "ownership_window_mm": ownership_window,
        "nominal_gap_mm": nominal_gap,
        "support_span_mm": float(support_span),
        "components": candidates,
        "selected_component": int(selected_report["component"]),
        "selected_area_mm2": float(selected_report["area_mm2"]),
        "selected_area_fraction": float(selected_report["area_fraction"]),
        "selected_hard_valid": hard_valid,
    }
    if not hard_valid:
        report["reason"] = "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT"
    selected = selected.copy()
    selected.invert()
    return selected, report


def cervical_feasibility(outer, inner, margin_points):
    """Read-only gate before any collar triangles are generated."""
    outer_loops = _boundary_loops(np.asarray(outer.faces, np.int64))
    inner_loops = _boundary_loops(np.asarray(inner.faces, np.int64))
    report = {
        "status": "PASS_ENGINEERING",
        "outer_loop_count": int(len(outer_loops)),
        "inner_loop_count": int(len(inner_loops)),
    }
    if len(outer_loops) != 1 or len(inner_loops) != 1:
        report.update(status="REVIEW_REQUIRED", reason="CERVICAL_LOOP_COUNT_INVALID")
        return False, report
    op = np.asarray(outer.vertices, float)[np.asarray(outer_loops[0], np.int64)]
    ip = np.asarray(inner.vertices, float)[np.asarray(inner_loops[0], np.int64)]
    if not len(op) or not len(ip):
        report.update(status="REVIEW_REQUIRED", reason="EMPTY_CERVICAL_RING")
        return False, report
    _cp, d, _ = closest_point(inner, op)
    d = np.asarray(d, float)
    report["source_separation_mm"] = {
        "min": float(d.min()), "p01": float(np.quantile(d, .01)),
        "p05": float(np.quantile(d, .05)), "median": float(np.median(d)),
        "fraction_below_0_03": float(np.mean(d < .03)),
        "fraction_below_0_05": float(np.mean(d < .05)),
        "fraction_below_0_10": float(np.mean(d < .10)),
    }
    margin = np.asarray(margin_points, float)
    od = _polyline_distances(op, margin)
    id_ = _polyline_distances(ip, margin)
    report["ring_to_margin_mm"] = {
        "outer_median": float(np.median(od)), "outer_p95": float(np.quantile(od, .95)),
        "inner_median": float(np.median(id_)), "inner_p95": float(np.quantile(id_, .95)),
    }
    collapsed = bool(np.quantile(d, .05) < .02 and np.mean(d < .03) > .20)
    too_far = bool(np.quantile(od, .95) > 3.5 or np.quantile(id_, .95) > 3.5)
    if collapsed:
        report.update(status="REVIEW_REQUIRED", reason="CERVICAL_SOURCE_COLLAPSE")
        return False, report
    if too_far:
        report.update(status="REVIEW_REQUIRED", reason="CERVICAL_RING_TOO_FAR_FROM_MARGIN")
        return False, report
    return True, report


def _resample_closed_polyline(points, n):
    p = np.asarray(points, float)
    if len(p) > 1 and np.linalg.norm(p[0] - p[-1]) <= 1e-10:
        p = p[:-1]
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    keep = seg > 1e-10
    if not np.any(keep):
        raise ValueError("Margin polyline has zero length")
    s = np.r_[0.0, np.cumsum(seg)]
    total = float(s[-1])
    target = np.linspace(0.0, total, int(n), endpoint=False)
    idx = np.clip(np.searchsorted(s, target, side="right") - 1, 0, len(seg) - 1)
    frac = (target - s[idx]) / np.maximum(seg[idx], 1e-12)
    return p[idx] + frac[:, None] * (np.roll(p, -1, axis=0)[idx] - p[idx])


def _best_phase(ring_points, margin_points):
    """Return ring index order aligned to the canonical Margin samples."""
    r = np.asarray(ring_points, float)
    m = np.asarray(margin_points, float)
    n = len(m)
    if len(r) != n:
        raise ValueError("canonical ring point count mismatch")
    best = None
    for reverse in (False, True):
        rr = r[::-1] if reverse else r
        # Candidate shifts from the nearest ring vertices to the Margin start.
        order = np.argsort(np.linalg.norm(rr - m[0], axis=1))[:min(8, n)]
        for shift in order:
            aligned = np.roll(rr, -int(shift), axis=0)
            cost = float(np.mean(np.linalg.norm(aligned - m, axis=1) ** 2))
            if best is None or cost < best[0]:
                ids = np.arange(n)[::-1] if reverse else np.arange(n)
                ids = np.roll(ids, -int(shift))
                best = (cost, ids)
    return np.asarray(best[1], np.int64), float(best[0])


def _quad_strip(a, b, reverse=False):
    """Equal-count 1:1 strip with locally quality-optimal diagonals."""
    a = np.asarray(a, np.int64); b = np.asarray(b, np.int64)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("canonical collar requires equal-count rings")
    faces = []
    for i in range(len(a)):
        j = (i + 1) % len(a)
        faces.extend([(a[i], b[i], a[j]), (a[j], b[i], b[j])])
    f = np.asarray(faces, np.int64)
    return f[:, ::-1] if reverse else f


def _triangle_quality(vertices, faces):
    tri = np.asarray(vertices, float)[np.asarray(faces, np.int64)]
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    area = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    q = 4.0 * np.sqrt(3.0) * area / np.maximum(e0**2 + e1**2 + e2**2, 1e-18)
    longest = np.maximum(np.maximum(e0, e1), e2)
    shortest = np.maximum(np.minimum(np.minimum(e0, e1), e2), 1e-12)
    return {
        "triangle_count": int(len(faces)),
        "quality_min": float(q.min(initial=1.0)),
        "quality_p05": float(np.quantile(q, .05)) if len(q) else None,
        "aspect_max": float((longest / shortest).max(initial=1.0)),
        "aspect_p95": float(np.quantile(longest / shortest, .95)) if len(q) else None,
        "min_edge_mm": float(shortest.min(initial=0.0)),
    }


def build_canonical_margin_crown(outer, inner, margin_points):
    """Assemble two frozen source bodies to one shared, resampled exact Margin.

    Source boundary edges are conformingly split to a common N (no source point
    moves).  A generated midpoint ring makes the clinical collar two-layered;
    source->mid and mid->Margin are deterministic 1:1 strips.
    """
    margin = np.asarray(margin_points, float)
    if len(margin) > 1 and np.linalg.norm(margin[0] - margin[-1]) <= 1e-10:
        margin = margin[:-1]
    perimeter = float(np.sum(np.linalg.norm(np.roll(margin, -1, axis=0) - margin, axis=1)))
    loops_o = _boundary_loops(np.asarray(outer.faces, np.int64))
    loops_i = _boundary_loops(np.asarray(inner.faces, np.int64))
    if len(loops_o) != 1 or len(loops_i) != 1:
        raise ValueError("canonical collar requires one source loop per surface")
    edge_o = np.linalg.norm(np.diff(np.vstack([
        np.asarray(outer.vertices)[loops_o[0]], np.asarray(outer.vertices)[loops_o[0]][:1]
    ]), axis=0), axis=1)
    edge_i = np.linalg.norm(np.diff(np.vstack([
        np.asarray(inner.vertices)[loops_i[0]], np.asarray(inner.vertices)[loops_i[0]][:1]
    ]), axis=0), axis=1)
    med = np.median(np.r_[edge_o[edge_o > 1e-8], edge_i[edge_i > 1e-8]])
    target_edge = float(np.clip(med if np.isfinite(med) else perimeter / 128.0, .08, .30))
    n = int(np.clip(round(perimeter / max(target_edge, 1e-6)), 64, 256))

    ov, of, oring = _conforming_ring(outer.vertices, outer.faces, loops_o[0], n)
    iv, inf, iring = _conforming_ring(inner.vertices, inner.faces, loops_i[0], n)
    margin_n = _resample_closed_polyline(margin, n)

    oi, outer_phase_cost = _best_phase(ov[oring], margin_n)
    ii, inner_phase_cost = _best_phase(iv[iring], margin_n)
    oring = np.asarray(oring, np.int64)[oi]
    iring = np.asarray(iring, np.int64)[ii]
    op = ov[oring]; ip = iv[iring]

    # One generated intermediate ring on each side.  Body vertices are untouched.
    outer_mid = .55 * op + .45 * margin_n
    inner_mid = .55 * ip + .45 * margin_n

    off_i = len(ov)
    off_om = off_i + len(iv)
    off_im = off_om + n
    off_m = off_im + n
    vertices = np.vstack([ov, iv, outer_mid, inner_mid, margin_n])
    oi_ids = oring
    ii_ids = iring + off_i
    om_ids = np.arange(off_om, off_om + n, dtype=np.int64)
    im_ids = np.arange(off_im, off_im + n, dtype=np.int64)
    m_ids = np.arange(off_m, off_m + n, dtype=np.int64)

    outer_faces = np.asarray(of, np.int64)
    inner_faces = np.asarray(inf, np.int64) + off_i
    collar_outer_1 = _quad_strip(oi_ids, om_ids, reverse=False)
    collar_outer_2 = _quad_strip(om_ids, m_ids, reverse=False)
    collar_inner_1 = _quad_strip(ii_ids, im_ids, reverse=True)
    collar_inner_2 = _quad_strip(im_ids, m_ids, reverse=True)
    collar_faces = np.vstack([collar_outer_1, collar_outer_2, collar_inner_1, collar_inner_2])
    faces = np.vstack([outer_faces, inner_faces, collar_faces])
    final = trimesh.Trimesh(vertices, faces, process=False)
    trimesh.repair.fix_normals(final, multibody=True)

    # Exportable clinical surfaces include their respective collar up to Margin.
    outer_surface = trimesh.Trimesh(
        vertices, np.vstack([outer_faces, collar_outer_1, collar_outer_2]), process=False
    )
    inner_surface = trimesh.Trimesh(
        vertices, np.vstack([inner_faces, collar_inner_1, collar_inner_2]), process=False
    )
    outer_surface.remove_unreferenced_vertices(); inner_surface.remove_unreferenced_vertices()

    qc = input_qc(np.asarray(final.vertices), np.asarray(final.faces))
    topology_broken = bool(qc.get("nonmanifold_edges") or qc.get("winding_inconsistent_edges") or qc.get("degenerate_faces"))
    if not final.is_watertight or len(final.split(only_watertight=False)) != 1 or topology_broken:
        raise ValueError(f"CANONICAL_MARGIN_COLLAR_TOPOLOGY_FAILED: {qc}")

    report = {
        "method": "CANONICAL_MARGIN_ARCLENGTH_TWO_LAYER_COLLAR_V1",
        "margin_topology_samples": int(n),
        "target_edge_mm": target_edge,
        "outer_phase_cost": outer_phase_cost,
        "inner_phase_cost": inner_phase_cost,
        "outer_source_vertices_unchanged": True,
        "inner_source_vertices_unchanged": True,
        "shared_margin_vertices": True,
        "collar_triangle_quality": _triangle_quality(vertices, collar_faces),
        "geometry": qc,
        "margin_sample_to_confirmed_max_mm": float(_polyline_distances(margin_n, margin).max(initial=0.0)),
    }
    return final, outer_surface, inner_surface, report
