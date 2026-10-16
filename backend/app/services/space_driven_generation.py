"""Patient-space generation helpers used by the AUTO workflow."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR
from ..geometry.io import load_mesh, save_mesh
from ..geometry.crown_outer.auto_fit_adapter import FitSurface
from ..geometry.crown_outer.patient_envelope import (
    PatientEnvelope,
    PatientEnvelopeConfig,
    build_patient_envelope,
)
from ..geometry.crown_outer.procedural_embryo import build_procedural_crown_embryo
from ..geometry.crown_outer.anatomy_refinement import refine_crown_anatomy
from .design_dependencies import file_digest


def _asset_for_design(d, kind):
    from ..api.routes import asset
    return asset(d['case_id'], kind)


def _confirmed_neighbor_surfaces(d, frame_ref):
    neighbors = json.loads(d.get('neighbors_json') or '{}')
    surfaces = {}
    for side in ('mesial', 'distal'):
        item = neighbors.get(side) or {}
        if item.get('confirmation_status') != 'CONFIRMED':
            continue
        if item.get('frame_revision_id') != frame_ref['revision_id']:
            continue
        path = item.get('path')
        if not path or not Path(path).is_file():
            continue
        if item.get('region_sha256') and file_digest(path) != item['region_sha256']:
            continue
        surfaces[side] = FitSurface(load_mesh(Path(path)), item.get('outward_normal'))
    return surfaces


def _confirmed_antagonist_surface(d, frame, frame_ref):
    bite = json.loads(d.get('bite_json') or '{}')
    if not (
        bite.get('revision_id') and bite.get('pose_confirmed') and bite.get('transform')
        and bite.get('frame_revision_id') == frame_ref['revision_id']
    ):
        return None
    kind = 'upper' if d['target_fdi'] >= 31 else 'lower'
    ref = _asset_for_design(d, kind)
    if not ref:
        return None
    mesh = load_mesh(Path(ref['path']))
    mesh.apply_transform(np.asarray(bite['transform'], float))
    return FitSurface(mesh, (-np.asarray(frame.z, float)).tolist())


def _margin_fallback_envelope(frame, margin, reason):
    """Conservative seed envelope used only when full patient-envelope build fails.

    This keeps v1.0 test runs inspectable.  It is deliberately marked review-only
    and never claims neighbor/antagonist constraints were solved.
    """
    origin = np.asarray(frame.origin, float)
    matrix = np.asarray(frame.matrix(), float)
    local = (np.asarray(margin, float) - origin) @ matrix
    finite = local[np.isfinite(local).all(axis=1)]
    if len(finite) < 8:
        raise ValueError('Margin fallback 也无法建立：有效点不足')
    center = np.median(finite[:, :2], axis=0)
    x = finite[:, 0] - center[0]
    y = finite[:, 1] - center[1]
    pad_x = 1.8
    pad_y = 1.6
    u_min = float(np.quantile(x, .02) - pad_x)
    u_max = float(np.quantile(x, .98) + pad_x)
    v_min = float(np.quantile(y, .02) - pad_y)
    v_max = float(np.quantile(y, .98) + pad_y)
    margin_z = float(np.median(finite[:, 2]))
    return PatientEnvelope(
        center_xy=(float(center[0]), float(center[1])),
        md_axis_xy=(1.0, 0.0),
        bl_axis_xy=(0.0, 1.0),
        u_min=u_min,
        u_max=u_max,
        v_min=v_min,
        v_max=v_max,
        margin_z=margin_z,
        target_top_z=None,
        sources={
            'fallback': True,
            'fallback_mode': 'MARGIN_ONLY_CONSERVATIVE_ENVELOPE',
            'fallback_reason': str(reason),
            'antagonist': {'used': False},
            'constraint_conflicts': [],
        },
    )


def _bind_runtime_seed_as_geometry_source(d, path, sha256):
    """Keep legacy AutoFit provenance code from touching an external tooth library.

    ``propose`` still asks ``resolve_template_source`` for historical reporting.
    For the patient-space AUTO route we inject the generated seed only into this
    in-memory design copy.  Nothing is persisted; the workflow report later marks
    the template role as NOT_USED.  This compatibility shim removes runtime tooth
    library dependency without rewriting the mature contact/occlusion solver.
    """
    try:
        landmarks = json.loads(d.get('landmarks_json') or '{}')
    except (TypeError, ValueError):
        landmarks = {}
    landmarks['_selected_template'] = {
        'source_path': str(path),
        'sha256': sha256,
        'library': None,
        'proposal_id': None,
        'geometry_frame_revision_id': None,
        'runtime_only_procedural_seed': True,
    }
    d['landmarks_json'] = json.dumps(landmarks, ensure_ascii=False)


def build_space_driven_seed(d, frame, frame_ref, support, margin, parameters, state):
    """Build PatientEnvelope first, then generate a template-free embryo.

    In v1.0 test mode, failure to assemble the complete six-direction envelope
    falls back to a conservative Margin-only envelope so the run still produces
    geometry and a reviewable report.
    """
    neighbors = _confirmed_neighbor_surfaces(d, frame_ref)
    antagonist = _confirmed_antagonist_surface(d, frame, frame_ref)

    thickness = parameters.get('minimum_thickness_mm')
    try:
        thickness = float(thickness) if thickness is not None else None
    except (TypeError, ValueError):
        thickness = None
    cement = parameters.get('cement_gap_mm')
    try:
        cement = float(cement) if cement is not None else 0.0
    except (TypeError, ValueError):
        cement = 0.0

    shell_required = None
    if thickness is not None and np.isfinite(thickness) and thickness > 0:
        shell_required = thickness + max(cement, 0.0)

    fallback_reason = None
    try:
        envelope = build_patient_envelope(
            frame=frame,
            margin=np.asarray(margin, float),
            inner_envelope_mesh=support,
            minimum_thickness_mm=shell_required,
            proximal_surfaces=neighbors,
            proximal_target_mm=float(state['proximal_target_mm']),
            antagonist_surface=antagonist,
            occlusal_target_mm=float(state['occlusal_target_mm']) if antagonist is not None else None,
            config=PatientEnvelopeConfig(),
        )
    except Exception as exc:
        fallback_reason = str(exc)
        envelope = _margin_fallback_envelope(frame, margin, exc)

    mesh, seed_report = build_procedural_crown_embryo(
        frame, np.asarray(margin, float), envelope, fdi=int(d['target_fdi'])
    )
    path = CASE_DIR / d['case_id'] / f"{d['id']}_space_seed_{uuid4().hex}.stl"
    save_mesh(mesh, path)
    seed_sha = file_digest(path)
    _bind_runtime_seed_as_geometry_source(d, path, seed_sha)
    seed_report['artifact'] = str(path)
    seed_report['sha256'] = seed_sha
    seed_report['neighbor_surface_count'] = int(len(neighbors))
    seed_report['antagonist_used'] = bool(antagonist is not None and fallback_reason is None)
    seed_report['envelope_status'] = 'FALLBACK_MARGIN_ENVELOPE' if fallback_reason else 'PATIENT_ENVELOPE'
    seed_report['runtime_template_dependency'] = False
    if fallback_reason:
        seed_report['status'] = 'REVIEW_REQUIRED'
        seed_report['fallback_reason'] = fallback_reason
    return path, seed_report


def anatomy_refine_candidate(d, frame, margin, source_path):
    mesh = load_mesh(Path(source_path))
    refined, report = refine_crown_anatomy(
        mesh, frame, np.asarray(margin, float), fdi=int(d['target_fdi'])
    )
    path = CASE_DIR / d['case_id'] / f"{d['id']}_anatomy_{uuid4().hex}.stl"
    save_mesh(refined, path)
    report['artifact'] = str(path)
    report['sha256'] = file_digest(path)
    return path, report
