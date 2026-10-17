"""CrownFlow 1.0 guard for destructive cervical fitting.

The confirmed Margin owns the cervical target, but the outer anatomy must not be
sacrificed to hit that curve in one deformation.  Real FDI36/FDI26 runs showed the
legacy full-loop fitter moving cervical vertices by 2.6-3.0 mm while introducing
hundreds of severe face flips/self-intersections, yet the stage still reported
PASS because Margin distance alone improved.

This guard keeps the established outer solvers unchanged.  It wraps only the
cervical fit and backtracks the proposed deformation until topology and a bounded
per-vertex move are safe.  If no non-destructive step exists, the pre-cervical
outer is preserved and the stage is marked REVIEW_REQUIRED instead of publishing
a visibly torn crown.
"""
from __future__ import annotations

import numpy as np

from .geometry.crown_outer import patient_adaptation as _adapt


_ORIGINAL_CERVICAL_FIT = _adapt.cervical_fit_to_margin
CERVICAL_MAX_VERTEX_MOVE_MM = 0.90
CERVICAL_FACE_AREA_MIN_MM2 = 0.0025
CERVICAL_BACKTRACK_SCALES = (1.0, 0.75, 0.50, 0.35, 0.25, 0.15, 0.10, 0.0)


def _deformation_audit(before, after):
    bv = np.asarray(before.vertices, float)
    av = np.asarray(after.vertices, float)
    faces = np.asarray(before.faces, int)
    if bv.shape != av.shape or not np.array_equal(faces, np.asarray(after.faces, int)):
        return {
            'safe': False,
            'reason': 'TOPOLOGY_CHANGED_DURING_CERVICAL_FIT',
            'max_move_mm': float('inf'),
            'severe_flip_count': -1,
            'collapsed_face_count': -1,
        }
    tri0 = bv[faces]
    tri1 = av[faces]
    n0 = np.cross(tri0[:, 1] - tri0[:, 0], tri0[:, 2] - tri0[:, 0])
    n1 = np.cross(tri1[:, 1] - tri1[:, 0], tri1[:, 2] - tri1[:, 0])
    l0 = np.linalg.norm(n0, axis=1)
    l1 = np.linalg.norm(n1, axis=1)
    valid = (l0 > 1e-10) & (l1 > 1e-10)
    cosine = np.ones(len(faces), float)
    cosine[valid] = np.einsum('ij,ij->i', n0[valid], n1[valid]) / (l0[valid] * l1[valid])
    source_area = 0.5 * l0
    severe = (cosine <= 0.0) & (source_area >= CERVICAL_FACE_AREA_MIN_MM2)
    collapsed = l1 <= np.maximum(l0 * 0.10, 1e-10)
    move = np.linalg.norm(av - bv, axis=1)
    max_move = float(move.max(initial=0.0))
    return {
        'safe': bool(not np.any(severe) and not np.any(collapsed) and max_move <= CERVICAL_MAX_VERTEX_MOVE_MM),
        'max_move_mm': max_move,
        'p95_move_mm': float(np.quantile(move, 0.95)) if len(move) else 0.0,
        'severe_flip_count': int(np.sum(severe)),
        'collapsed_face_count': int(np.sum(collapsed)),
        'min_normal_dot': float(cosine.min(initial=1.0)),
    }


def cervical_fit_to_margin_guarded(mesh, frame, margin, *, config=None):
    fitted, report = _ORIGINAL_CERVICAL_FIT(mesh, frame, margin, config=config)
    before = mesh.copy()
    base_vertices = np.asarray(before.vertices, float)
    target_vertices = np.asarray(fitted.vertices, float)

    trials = []
    selected = None
    selected_scale = 0.0
    for scale in CERVICAL_BACKTRACK_SCALES:
        trial = before.copy()
        trial.vertices = base_vertices + float(scale) * (target_vertices - base_vertices)
        audit = _deformation_audit(before, trial)
        trials.append({'scale': float(scale), **audit})
        if audit['safe']:
            selected = trial
            selected_scale = float(scale)
            break

    if selected is None:
        selected = before
        selected_scale = 0.0
        selected_audit = _deformation_audit(before, selected)
    else:
        selected_audit = trials[-1]

    out_report = dict(report or {})
    out_report['cervical_geometry_guard'] = {
        'policy': 'MARGIN_TARGET_MUST_NOT_DESTROY_OUTER_ANATOMY',
        'max_vertex_move_mm': CERVICAL_MAX_VERTEX_MOVE_MM,
        'face_area_min_mm2': CERVICAL_FACE_AREA_MIN_MM2,
        'selected_scale': selected_scale,
        'full_fit_audit': trials[0] if trials else None,
        'selected_audit': selected_audit,
        'trials': trials,
    }
    if selected_scale < 1.0:
        out_report['original_fit_status'] = out_report.get('status')
        out_report['status'] = 'REVIEW_REQUIRED'
        out_report['reason'] = 'CERVICAL_FIT_BACKED_OFF_TO_PROTECT_OUTER_TOPOLOGY'
        out_report['margin_fit_partial'] = bool(selected_scale > 0.0)
    return selected, out_report


# main.py imports this before services.auto_fit captures the function symbol.
_adapt.cervical_fit_to_margin = cervical_fit_to_margin_guarded
