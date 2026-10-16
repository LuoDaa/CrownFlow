"""Template-constrained AUTO seed for CrownFlow 1.0.

The library crown owns anatomy/topology. Patient geometry owns metric limits.
The route deliberately avoids boolean cuts and remeshing: it starts from a
watertight placed crown and uses topology-preserving vertex deformation only.

    placed library crown
      -> body-weighted uniform overscale
      -> independent M/D and B/L soft compression into PatientEnvelope
      -> antagonist-owned height compression
      -> topology guard / deformation backoff
      -> existing cervical/contact/occlusal/intaglio solvers

The cervical band is not scaled in this stage. ``placed_library_crown`` already
registers the template to the confirmed Margin, and the downstream cervical
solver remains the final Margin owner.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR, TOOTH_LIBRARY_ROOT
from ..geometry.io import save_mesh
from ..geometry.crown_outer.patient_envelope import PatientEnvelopeConfig, build_patient_envelope
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
    collapsed = int(np.sum(an <= np.maximum(bn * .08, 1e-10)))
    flipped = 0
    if np.any(valid):
        cos = np.einsum('ij,ij->i', b[valid], a[valid]) / (bn[valid] * an[valid])
        flipped = int(np.sum(cos <= 0.0))
    return not (collapsed or flipped), {
        'collapsed_face_count': collapsed,
        'flipped_face_count': flipped,
    }


def _backoff_deformation(mesh, proposed_vertices):
    before = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, int)
    last = {'reason': 'UNKNOWN'}
    for alpha in (1.0, .80, .60, .40, .25, 0.0):
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


def build_template_constrained_seed(d, frame, frame_ref, support, margin, parameters, state):
    """Return a closed patient-sized template crown without changing topology."""
    envelope, neighbors, antagonist, support_report = _build_envelope(
        d, frame, frame_ref, support, margin, parameters, state
    )
    library, user_selected = _resolve_library(d)
    mesh, metadata = placed_library_crown(
        library, int(d['target_fdi']), np.asarray(margin, float), frame
    )
    if not mesh.is_watertight or not mesh.is_winding_consistent or len(mesh.split(only_watertight=False)) != 1:
        raise ValueError('牙库模板放置后不是单一闭合实体')

    before = np.asarray(mesh.vertices, float).copy()
    local = _local(before, frame)
    u, v = _uv(local, envelope)
    h = local[:, 2] - float(envelope.margin_z)
    body_weight = _smoothstep01((h - .25) / 1.45)

    current = {
        'u_pos': _positive_extent(u),
        'u_neg': _negative_extent(u),
        'v_pos': _positive_extent(v),
        'v_neg': _negative_extent(v),
        'height': float(np.quantile(h[h > 0], .985)) if np.any(h > 0) else 0.0,
    }
    target = {
        'u_pos': max(float(envelope.u_max), .1),
        'u_neg': max(float(-envelope.u_min), .1),
        'v_pos': max(float(envelope.v_max), .1),
        'v_neg': max(float(-envelope.v_min), .1),
        'height': (None if envelope.target_top_z is None else max(float(envelope.target_top_z - envelope.margin_z), .1)),
    }

    ratios = [
        _safe_ratio(target['u_pos'], current['u_pos']),
        _safe_ratio(target['u_neg'], current['u_neg']),
        _safe_ratio(target['v_pos'], current['v_pos']),
        _safe_ratio(target['v_neg'], current['v_neg']),
    ]
    if target['height'] is not None:
        ratios.append(_safe_ratio(target['height'], current['height']))
    # Start slightly oversized so all later operations are subtractive/compressive.
    uniform_scale = float(np.clip(max(1.0, max(ratios)) * 1.035, 1.0, 1.35))

    center = np.asarray(envelope.center_xy, float)
    scaled = local.copy()
    scaled[:, :2] = center + (local[:, :2] - center) * (1.0 + body_weight[:, None] * (uniform_scale - 1.0))
    scaled[:, 2] = float(envelope.margin_z) + h * (1.0 + body_weight * (uniform_scale - 1.0))

    su, sv = _uv(scaled, envelope)
    sh = scaled[:, 2] - float(envelope.margin_z)
    body_weight = _smoothstep01((sh - .25) / 1.45)

    ext = {
        'u_pos': _positive_extent(su), 'u_neg': _negative_extent(su),
        'v_pos': _positive_extent(sv), 'v_neg': _negative_extent(sv),
    }
    factors = {
        'u_pos': min(1.0, _safe_ratio(target['u_pos'], ext['u_pos'])),
        'u_neg': min(1.0, _safe_ratio(target['u_neg'], ext['u_neg'])),
        'v_pos': min(1.0, _safe_ratio(target['v_pos'], ext['v_pos'])),
        'v_neg': min(1.0, _safe_ratio(target['v_neg'], ext['v_neg'])),
    }

    u_factor = np.where(su >= 0, factors['u_pos'], factors['u_neg'])
    v_factor = np.where(sv >= 0, factors['v_pos'], factors['v_neg'])
    cu = su * (1.0 + body_weight * (u_factor - 1.0))
    cv = sv * (1.0 + body_weight * (v_factor - 1.0))
    compressed = _from_uv(scaled, cu, cv, envelope)

    if target['height'] is not None:
        current_height = float(np.quantile(sh[sh > 0], .985)) if np.any(sh > 0) else 0.0
        z_factor = min(1.0, _safe_ratio(target['height'], current_height))
        compressed[:, 2] = float(envelope.margin_z) + sh * (1.0 + body_weight * (z_factor - 1.0))
    else:
        z_factor = 1.0

    proposed_world = _world(compressed, frame)
    safe_mesh, deformation_alpha, deformation_audit = _backoff_deformation(mesh, proposed_world)
    if not safe_mesh.is_watertight or not safe_mesh.is_winding_consistent:
        # Topology is inherited from the closed template.  This branch is only a
        # defensive fallback for geometry libraries that report an invalid result.
        safe_mesh = mesh.copy()
        deformation_alpha = 0.0
        deformation_audit = {**deformation_audit, 'reason': 'WATERTIGHT_GUARD_ROLLBACK'}

    path = CASE_DIR / d['case_id'] / f"{d['id']}_template_constrained_seed_{uuid4().hex}.stl"
    save_mesh(safe_mesh, path)
    sha = file_digest(path)
    report = {
        'stage': 'TEMPLATE_CONSTRAINED_SEED',
        'status': 'PASS_ENGINEERING' if deformation_alpha > 0 else 'REVIEW_REQUIRED',
        'route': 'TEMPLATE_CONSTRAINED_CROWN_V1',
        'template_used': True,
        'template_role': 'ANATOMY_AND_CLOSED_TOPOLOGY_PRIOR',
        'library': library.name,
        'library_user_selected': bool(user_selected),
        'template_metadata': metadata,
        'uniform_overscale': uniform_scale,
        'soft_compression': factors,
        'height_compression_factor': z_factor,
        'deformation_alpha': float(deformation_alpha),
        'deformation_audit': deformation_audit,
        'patient_envelope': envelope.as_report(),
        'neighbor_surface_count': int(len(neighbors)),
        'antagonist_used': bool(antagonist is not None),
        'envelope_support': support_report,
        'cervical_policy': 'PLACED_MARGIN_ROOT_THEN_DOWNSTREAM_CERVICAL_SOLVER',
        'boolean_used': False,
        'remesh_used': False,
        'topology_changed': False,
        'watertight': bool(safe_mesh.is_watertight),
        'winding_consistent': bool(safe_mesh.is_winding_consistent),
        'artifact': str(path),
        'sha256': sha,
    }
    return path, report
