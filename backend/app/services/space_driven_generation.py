"""Patient-space generation helpers used by the AUTO workflow."""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR
from ..geometry.io import load_mesh, save_mesh
from ..geometry.crown_outer.auto_fit_adapter import FitSurface
from ..geometry.crown_outer.patient_envelope import PatientEnvelopeConfig, build_patient_envelope
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


def build_space_driven_seed(d, frame, frame_ref, support, margin, parameters, state):
    """Build PatientEnvelope first, then generate a template-free embryo."""
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
    mesh, seed_report = build_procedural_crown_embryo(
        frame, np.asarray(margin, float), envelope, fdi=int(d['target_fdi'])
    )
    path = CASE_DIR / d['case_id'] / f"{d['id']}_space_seed_{uuid4().hex}.stl"
    save_mesh(mesh, path)
    seed_report['artifact'] = str(path)
    seed_report['sha256'] = file_digest(path)
    seed_report['neighbor_surface_count'] = int(len(neighbors))
    seed_report['antagonist_used'] = bool(antagonist is not None)
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
