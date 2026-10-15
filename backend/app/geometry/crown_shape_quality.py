"""Light-weight shape quality audit for crown deformations.

This module is deliberately *relative*: it compares a candidate to the same
mesh before a deformation step.  It is therefore suitable for Step-5 edits
and proximal coarse/fine fitting without pretending to be a manufacturing QC.
All thresholds are engineering defaults and must not be presented as clinical
acceptance criteria.
"""
from __future__ import annotations

import numpy as np


def _q05(values):
    values = np.asarray(values, float)
    return float(np.quantile(values, .05)) if values.size else None


def _q95(values):
    values = np.asarray(values, float)
    return float(np.quantile(values, .95)) if values.size else None


def triangle_quality(vertices, faces):
    """Normalized triangle quality: equilateral=1, degenerate=0."""
    v = np.asarray(vertices, float)
    f = np.asarray(faces, int)
    tri = v[f]
    e0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    e1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    e2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    denom = e0 * e0 + e1 * e1 + e2 * e2
    return np.divide(2.0 * np.sqrt(3.0) * area2, denom,
                     out=np.zeros_like(area2), where=denom > 1e-18)


def deformation_shape_report(v0, v1, faces, *, focus_faces=None):
    """Report local stretch/sliver/area changes caused by a deformation."""
    v0 = np.asarray(v0, float)
    v1 = np.asarray(v1, float)
    faces = np.asarray(faces, int)
    if focus_faces is None:
        focus = np.arange(len(faces), dtype=int)
    else:
        focus = np.unique(np.asarray(focus_faces, int))
        focus = focus[(focus >= 0) & (focus < len(faces))]
    if not len(focus):
        return {
            'focus_face_count': 0,
            'edge_stretch_p95': 1.0, 'edge_stretch_max': 1.0,
            'triangle_quality_before_p05': None, 'triangle_quality_after_p05': None,
            'area_ratio_p05': 1.0, 'area_ratio_p95': 1.0,
        }

    local_faces = faces[focus]
    edges = np.sort(np.vstack((local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]])), axis=1)
    edges = np.unique(edges, axis=0)
    l0 = np.linalg.norm(v0[edges[:, 1]] - v0[edges[:, 0]], axis=1)
    l1 = np.linalg.norm(v1[edges[:, 1]] - v1[edges[:, 0]], axis=1)
    stretch = np.divide(l1, np.maximum(l0, 1e-9))
    # Compression can create the same sort of sliver as excessive stretching;
    # make the metric symmetric around 1.
    sym_stretch = np.maximum(stretch, 1.0 / np.maximum(stretch, 1e-9))

    q0 = triangle_quality(v0, local_faces)
    q1 = triangle_quality(v1, local_faces)
    tri0 = v0[local_faces]
    tri1 = v1[local_faces]
    a0 = 0.5 * np.linalg.norm(np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0]), axis=1)
    a1 = 0.5 * np.linalg.norm(np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0]), axis=1)
    ratio = np.divide(a1, np.maximum(a0, 1e-12))

    return {
        'focus_face_count': int(len(focus)),
        'edge_stretch_p95': _q95(sym_stretch),
        'edge_stretch_max': float(np.max(sym_stretch)) if len(sym_stretch) else 1.0,
        'triangle_quality_before_p05': _q05(q0),
        'triangle_quality_after_p05': _q05(q1),
        'triangle_quality_after_min': float(np.min(q1)) if len(q1) else None,
        'area_ratio_p05': _q05(ratio),
        'area_ratio_p95': _q95(ratio),
    }


def shape_gate(report, *, max_edge_stretch=3.0, p95_edge_stretch=1.8,
               min_quality_floor=0.02, relative_quality_floor=0.35,
               min_area_ratio=0.15, max_area_ratio=5.0):
    """Engineering gate for *new* distortion caused by a local operation.

    The limits are intentionally permissive for a 1.0 draft.  Their purpose is
    to reject obvious spikes/bridges while leaving final manufacturing QC to the
    existing finalize pipeline.
    """
    reasons = []
    if report.get('edge_stretch_max', 1.0) > max_edge_stretch:
        reasons.append('EDGE_STRETCH_MAX')
    if report.get('edge_stretch_p95', 1.0) > p95_edge_stretch:
        reasons.append('EDGE_STRETCH_P95')
    qb = report.get('triangle_quality_before_p05')
    qa = report.get('triangle_quality_after_p05')
    if qa is not None:
        floor = min_quality_floor if qb is None else max(min_quality_floor, float(qb) * relative_quality_floor)
        if qa < floor:
            reasons.append('TRIANGLE_QUALITY_DEGRADED')
    if report.get('area_ratio_p05') is not None and report['area_ratio_p05'] < min_area_ratio:
        reasons.append('LOCAL_AREA_COLLAPSE')
    if report.get('area_ratio_p95') is not None and report['area_ratio_p95'] > max_area_ratio:
        reasons.append('LOCAL_AREA_STRETCH')
    return {'ok': not reasons, 'reasons': reasons, 'limits': {
        'max_edge_stretch': max_edge_stretch,
        'p95_edge_stretch': p95_edge_stretch,
        'min_quality_floor': min_quality_floor,
        'relative_quality_floor': relative_quality_floor,
        'min_area_ratio': min_area_ratio,
        'max_area_ratio': max_area_ratio,
    }}
