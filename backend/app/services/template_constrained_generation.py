"""Morphology-prior + patient feasible-space deformation for CrownFlow AUTO 1.0.

A complete library crown supplies anatomy and closed topology. Patient geometry
supplies *maximum feasible bounds*. The seed is never stretched merely to fill a
six-face box. Only envelope violations are compressed; local proximal and
occlusal solvers establish real contacts later.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR, TOOTH_LIBRARY_ROOT
from ..geometry.io import load_mesh, save_mesh
from ..geometry.crown_outer.patient_envelope import (
    PatientEnvelopeConfig,
    build_patient_envelope,
    morph_embryo_to_patient_envelope,
)
from ..importers.tooth_library import discover_libraries
from .auto_fit_templates import placed_library_crown
from .design_dependencies import file_digest
from .space_driven_generation import (
    _confirmed_antagonist_surface,
    _confirmed_neighbor_surfaces,
    _envelope_support_domain,
)


def _smoothstep01(x):
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _local(points, frame):
    return (np.asarray(points, float) - np.asarray(frame.origin, float)) @ np.asarray(frame.matrix(), float)


def _world(points, frame):
    return np.asarray(points, float) @ np.asarray(frame.matrix(), float).T + np.asarray(frame.origin, float)


def _resolve_library(d):
    selected = {}
    try:
        selected = (json.loads(d.get('landmarks_json') or '{}').get('_selected_template') or {})
    except (TypeError, ValueError):
        selected = {}
    preferred = selected.get('library')
    libraries = discover_libraries(str(TOOTH_LIBRARY_ROOT))
    if preferred:
        libraries = sorted(libraries, key=lambda lib: 0 if lib.name == preferred else 1)
    errors = []
    for library in libraries:
        try:
            library.template_path(int(d['target_fdi']))
            return library, preferred == library.name
        except Exception as exc:
            errors.append(f'{library.name}:{exc}')
    raise ValueError('没有可用于当前牙位的完整牙库模板' + (f" ({'; '.join(errors[:3])})" if errors else ''))


def _uv(local, envelope):
    center = np.asarray(envelope.center_xy, float)
    md = np.asarray(envelope.md_axis_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    xy = np.asarray(local, float)[:, :2] - center
    return xy @ md, xy @ bl


def _from_uv(local, u, v, envelope):
    out = np.asarray(local, float).copy()
    center = np.asarray(envelope.center_xy, float)
    md = np.asarray(envelope.md_axis_xy, float)
    bl = np.asarray(envelope.bl_axis_xy, float)
    out[:, :2] = center + np.asarray(u)[:, None] * md + np.asarray(v)[:, None] * bl
    return out


def _positive_extent(values):
    values = np.asarray(values, float)
    pos = values[values > 0]
    return float(np.quantile(pos, .985)) if len(pos) else 0.0


def _negative_extent(values):
    values = np.asarray(values, float)
    neg = -values[values < 0]
    return float(np.quantile(neg, .985)) if len(neg) else 0.0


def _safe_ratio(target, current):
    if current <= 1e-8 or target <= 1e-8:
        return 1.0
    return float(target / current)


def _face_vectors(vertices, faces):
    tri = np.asarray(vertices, float)[np.asarray(faces, int)]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def _deformation_valid(before, candidate, faces):
    if not np.isfinite(candidate).all():
        return False, {'reason': 'NONFINITE_VERTEX'}
    b = _face_vectors(before, faces)
    a = _face_vectors(candidate, faces)
    bn = np.linalg.norm(b, axis=1)
    an = np.linalg.norm(a, axis=1)
    valid = (bn > 1e-10) & (an > 1e-10)
    collapsed = int(np.sum(an <= np.maximum(bn * .10, 1e-10)))
    flipped = 0
    if np.any(valid):
        cosine = np.einsum('ij,ij->i', b[valid], a[valid]) / (bn[valid] * an[valid])
        flipped = int(np.sum(cosine <= 0.0))
    return not (collapsed or flipped), {
        'collapsed_face_count': collapsed,
        'flipped_face_count': flipped,
    }


def _backoff_deformation(mesh, proposed_vertices):
    before = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    last = {'reason': 'UNKNOWN'}
    for alpha in (1.0, .85, .70, .55, .40, .25, 0.0):
        candidate = before + alpha * (np.asarray(proposed_vertices, float) - before)
        ok, audit = _deformation_valid(before, candidate, faces)
        last = audit
        if ok:
            out = mesh.copy()
            out.vertices = candidate
            return out, alpha, audit
    return mesh.copy(), 0.0, last


def _build_envelope(d, frame, frame_ref, support, margin, parameters, state):
    neighbors = _confirmed_neighbor_surfaces(d, frame_ref)
    antagonist = _confirmed_antagonist_surface(d, frame, frame_ref)
    support_domain, support_report = _envelope_support_domain(support, margin, frame)

    try:
        thickness = float(parameters.get('minimum_thickness_mm'))
        if not np.isfinite(thickness) or thickness <= 0:
            thickness = None
    except (TypeError, ValueError):
        thickness = None
    try:
        cement = max(float(parameters.get('cement_gap_mm') or 0.0), 0.0)
    except (TypeError, ValueError):
        cement = 0.0
    shell_required = None if thickness is None else thickness + cement

    envelope = build_patient_envelope(
        frame=frame,
        margin=np.asarray(margin, float),
        inner_envelope_mesh=support_domain,
        minimum_thickness_mm=shell_required,
        proximal_surfaces=neighbors,
        proximal_target_mm=float(state.get('proximal_target_mm') or 0.0),
        antagonist_surface=antagonist,
        occlusal_target_mm=(float(state.get('occlusal_target_mm') or 0.0) if antagonist is not None else None),
        config=PatientEnvelopeConfig(),
    )
    return envelope, neighbors, antagonist, support_report


def _regional_weights(u, v, h, extents, current_height):
    """Compatibility helper used by tests and reports.

    The current route no longer stretches the seed to target dimensions.  These
    weights describe only where broad compression is allowed: cervical root is
    frozen, side walls are freer, and the occlusal anatomy core is protected.
    """
    h_pos = np.maximum(np.asarray(h, float), 0.0)
    h_frac = np.clip(h_pos / max(float(current_height), 1e-6), 0.0, 1.0)
    cervical = _smoothstep01((h_pos - .35) / 1.20)
    u_scale = max(float(extents['u_pos']), float(extents['u_neg']), .5)
    v_scale = max(float(extents['v_pos']), float(extents['v_neg']), .5)
    radial = np.sqrt((np.asarray(u, float) / u_scale) ** 2 + (np.asarray(v, float) / v_scale) ** 2)
    radial = np.clip(radial, 0.0, 1.4)
    side_wall = np.clip(.35 + .85 * radial, .35, 1.0)
    occlusal_core = (h_frac >= .58) & (radial <= .72)
    anatomy_keep = np.ones(len(h_pos), float)
    anatomy_keep[occlusal_core] = .28
    adaptation = cervical * side_wall * anatomy_keep
    height_weight = cervical * np.clip((h_frac - .30) / .55, 0.0, 1.0)
    height_weight[occlusal_core] *= .45
    return cervical, adaptation, height_weight, occlusal_core


def build_template_constrained_seed(d, frame, frame_ref, support, margin, parameters, state):
    """Build a closed anatomy-preserving crown seed inside patient max bounds."""
    envelope, neighbors, antagonist, support_report = _build_envelope(
        d, frame, frame_ref, support, margin, parameters, state
    )
    library, user_selected = _resolve_library(d)
    mesh, metadata = placed_library_crown(
        library, int(d['target_fdi']), np.asarray(margin, float), frame
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split(only_watertight=False)) != 1:
        raise ValueError('牙库模板放置后不是单一闭合实体')

    # One source of truth for global space adaptation.  The seed and the later
    # GLOBAL_PREFIT now share the same one-sided operation, making the second pass
    # idempotent instead of progressively flattening the tooth.
    proposed, morph_report = morph_embryo_to_patient_envelope(
        mesh, frame, np.asarray(margin, float), envelope, config=PatientEnvelopeConfig()
    )
    safe_mesh, deformation_alpha, deformation_audit = _backoff_deformation(
        mesh, np.asarray(proposed.vertices, float)
    )
    if not safe_mesh.is_watertight or not safe_mesh.is_winding_consistent:
        safe_mesh = mesh.copy()
        deformation_alpha = 0.0
        deformation_audit = {**deformation_audit, 'reason': 'WATERTIGHT_GUARD_ROLLBACK'}

    path = CASE_DIR / d['case_id'] / f"{d['id']}_morphology_distance_seed_{uuid4().hex}.stl"
    save_mesh(safe_mesh, path)
    sha = file_digest(path)
    report = {
        'stage': 'MORPHOLOGY_DISTANCE_SEED',
        'status': 'PASS_ENGINEERING' if deformation_alpha > 0 else 'REVIEW_REQUIRED',
        'route': 'MORPHOLOGY_PRIOR_IN_MAX_FEASIBLE_PATIENT_SPACE_V2',
        'template_used': True,
        'template_role': 'MORPHOLOGY_AND_CLOSED_TOPOLOGY_PRIOR',
        'metric_owner': 'PATIENT_MAX_FEASIBLE_ENVELOPE',
        'envelope_semantics': 'MAXIMUM_BOUNDS_NOT_DIMENSION_TARGETS',
        'library': library.name,
        'library_user_selected': bool(user_selected),
        'template_metadata': metadata,
        'morph': morph_report,
        'deformation_alpha': float(deformation_alpha),
        'deformation_audit': deformation_audit,
        'patient_envelope': envelope.as_report(),
        'neighbor_surface_count': int(len(neighbors)),
        'antagonist_used': bool(antagonist is not None),
        'envelope_support': support_report,
        'cervical_policy': 'PATIENT_MARGIN_OWNS_ROOT_DOWNSTREAM_CERVICAL_SOLVER',
        'boolean_used': False,
        'remesh_used': False,
        'topology_changed': False,
        'watertight': bool(safe_mesh.is_watertight),
        'winding_consistent': bool(safe_mesh.is_winding_consistent),
        'artifact': str(path),
        'sha256': sha,
    }
    return path, report


def guard_published_outer(baseline_path, candidate_path):
    """Never publish an open, non-winding or fragmented downstream outer."""
    baseline = load_mesh(Path(baseline_path))
    try:
        candidate = load_mesh(Path(candidate_path))
        valid = bool(
            len(candidate.faces) > 0
            and np.isfinite(np.asarray(candidate.vertices, float)).all()
            and candidate.is_watertight
            and candidate.is_winding_consistent
            and len(candidate.split(only_watertight=False)) == 1
            and not np.any(np.asarray(candidate.area_faces, float) < 1e-10)
        )
    except Exception as exc:
        valid = False
        reason = str(exc)
    else:
        reason = None if valid else 'DOWNSTREAM_OUTER_GEOMETRY_INVALID'
    if valid:
        return str(candidate_path), {'status': 'KEPT_SOLVER_OUTER', 'rollback': False}
    if not baseline.is_watertight or not baseline.is_winding_consistent:
        return str(candidate_path), {'status': 'NO_VALID_BASELINE_ROLLBACK', 'rollback': False, 'reason': reason}
    return str(baseline_path), {
        'status': 'ROLLED_BACK_TO_MORPHOLOGY_DISTANCE_SEED',
        'rollback': True,
        'reason': reason,
    }
