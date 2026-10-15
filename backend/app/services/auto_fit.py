"""Patient-driven CAD-1 proposals, functional remeasurement and W3 preflight.

Outer-crown geometry is now built in this order:

    library placement -> patient prefit -> cervical/Margin fit -> proximal envelope
    -> antagonist height prefit -> occlusal refinement -> intaglio/final.

Engineering QC records warnings but never silently restores the template/baseline.
Only unusable numeric/input data (missing files, NaN/Inf, impossible references)
remain hard failures.
"""
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi import HTTPException

from ..config import CASE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh
from ..geometry.nonplanar_crown import fitting_support_above_margin, MarginBuildError
from ..geometry.crown_outer.collar_body_outer import _boundary_loops
from ..geometry.crown_outer.auto_fit_adapter import FitSurface, fit_stage, fit_domains, measure, surface_crossings
from ..geometry.crown_outer.patient_adaptation import global_patient_prefit, cervical_fit_to_margin
from ..geometry.crown_outer.thickness import input_qc
from .design_dependencies import file_digest, require_current_outer
from .pipeline import finalize_crown


def _asset(d, kind):
    with db() as c:
        row = c.execute(
            'SELECT * FROM assets WHERE case_id=? AND kind=? ORDER BY created_at DESC LIMIT 1',
            (d['case_id'], kind),
        ).fetchone()
    return dict(row) if row else None


def _ensure_finite_mesh(mesh, label='mesh'):
    vertices = np.asarray(mesh.vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f'{label} 没有有效顶点')
    if not np.isfinite(vertices).all():
        raise ValueError(f'{label} 含 NaN/Inf，已中止')
    if len(mesh.faces) == 0:
        raise ValueError(f'{label} 没有三角面')


def audit_final_context(d, outer_path, final_path, qc):
    """Requery persisted W3 surfaces; proposal metrics do not replace final QC."""
    from ..api.auto_fit import frame_from
    from ..geometry.manufacturing_qc import _intersection_volume

    if not d.get('autofit_frame_json'):
        return
    frame, ref = frame_from(d)
    outer = load_mesh(Path(outer_path))
    final = load_mesh(Path(final_path))
    parameters = json.loads(d['parameters_json'])
    margin = np.asarray(json.loads(d['margin_json']))
    neighbors = json.loads(d.get('neighbors_json') or '{}')
    patches, sides = {}, {}
    for side in ('mesial', 'distal'):
        entry = neighbors.get(side, {})
        if entry.get('frame_revision_id') != ref['revision_id']:
            sides[side] = {'status': 'NOT_ASSESSED', 'reason': 'NEIGHBOR_UNVERIFIED'}
            continue
        if entry.get('confirmation_status') == 'ABSENT':
            sides[side] = {'status': 'SKIPPED_ABSENT'}
            continue
        if entry.get('confirmation_status') != 'CONFIRMED':
            sides[side] = {'status': 'NOT_ASSESSED', 'reason': 'NEIGHBOR_UNVERIFIED'}
            continue
        source = entry['source']
        current = _asset(d, source['kind'])
        if (not current or current['id'] != source['asset_id'] or file_digest(current['path']) != source['sha256']
                or file_digest(entry['path']) != entry['region_sha256']):
            raise HTTPException(409, '最终接触审核的邻牙来源已变化，请重新确认')
        patches[side] = FitSurface(load_mesh(Path(entry['path'])), entry.get('outward_normal'))

    target = parameters.get('proximal_target_mm')
    if patches and target is not None:
        domains = fit_domains(outer, frame, margin, patches, 'PROXIMAL')[0]
        for side, surface in patches.items():
            sides[side] = {
                **measure(outer.vertices, domains[side], surface, target, kind='PROXIMAL', side=side)[0],
                'neighbor_revision_id': neighbors[side]['revision_id'],
                'material_collision': _intersection_volume(final, surface.mesh),
                'triangle_crossings': surface_crossings(outer, surface, domains[side]),
            }
    else:
        for side in patches:
            sides[side] = {'status': 'NOT_ASSESSED', 'reason': 'TARGET_UNCONFIRMED'}
    qc['ContactQC'] = {
        'status': 'MEASURED_ENGINEERING' if patches and target is not None else 'NOT_ASSESSED',
        'per_side': sides,
        'scope': 'frozen_final_outer_vertex_domains_not_continuous_certification',
        'geometry_frame_revision_id': ref['revision_id'],
        'final_sha256': file_digest(final_path),
    }

    bite = json.loads(d.get('bite_json') or '{}')
    target = parameters.get('occlusal_target_mm')
    if not (bite.get('pose_confirmed') and bite.get('revision_id')
            and bite.get('frame_revision_id') == ref['revision_id'] and target is not None):
        return
    for source in bite.get('scans', []):
        current = _asset(d, source['kind'])
        if not current or current['id'] != source['asset_id'] or file_digest(current['path']) != source['sha256']:
            raise HTTPException(409, '最终咬合审核的扫描来源已变化，请重新确认')
    ant = load_mesh(Path(_asset(d, 'upper' if d['target_fdi'] >= 31 else 'lower')['path']))
    ant.apply_transform(np.asarray(bite['transform']))
    surface = FitSurface(ant, -frame.z)
    mask = fit_domains(outer, frame, margin, {'occlusal': surface}, 'OCCLUSAL')[0]['occlusal']
    qc['OcclusionQC'] = {
        **measure(outer.vertices, mask, surface, target, kind='OCCLUSAL')[0],
        'bite_revision_id': bite['revision_id'],
        'scope': 'confirmed_static_pose_sampled_outer_not_dynamic_occlusion',
        'material_collision': _intersection_volume(final, ant),
        'triangle_crossings': surface_crossings(outer, surface, mask),
        'final_sha256': file_digest(final_path),
    }


def propose(d, frame, frame_ref, mode, baseline_path=None):
    if baseline_path is None:
        require_current_outer(d)
    else:
        d = {**d, 'outer_path': str(baseline_path)}

    baseline = load_mesh(Path(d['outer_path']))
    _ensure_finite_mesh(baseline, '模板初稿')
    mesh = baseline.copy()
    if len(mesh.faces) > 100000:
        raise ValueError('AutoFit 同步工程预算为 100000 个冠面，请先简化并重新确认')

    margin = np.asarray(json.loads(d['margin_json']), dtype=float)
    axis = np.asarray(json.loads(d['axis_json']), dtype=float)
    if not np.isfinite(margin).all() or not np.isfinite(axis).all():
        raise ValueError('Margin/Axis 含 NaN/Inf')
    parameters = json.loads(d['parameters_json'])
    support_ref = _asset(d, 'support')
    if not support_ref:
        raise ValueError('缺少基台/预备体 support')
    support = load_mesh(Path(support_ref['path']))

    from .abutment_support import prepare_design_support
    support_contract, prepared_support = prepare_design_support(d)
    if prepared_support:
        support = prepared_support.mesh
    _ensure_finite_mesh(support, '基台/预备体')

    prefix = CASE_DIR / d['case_id'] / f"{d['id']}_autofit_{uuid4().hex}"
    report = {
        'mode': mode,
        'status': 'INPUT_UNVERIFIED',
        'stages': [],
        'warnings': [],
        'per_side': {},
        'solver_version': 'patient_prefit+cervical_fit+local_band/2.0',
        'material_process_status': 'NOT_ASSESSED',
        'manufacturing_status': 'NOT_PRODUCTION_APPROVED',
        'geometry_frame_revision_id': frame_ref['revision_id'],
        'neighbor_revision_ids': {},
        'bite_revision_id': None,
        'template_sha256': None,
        'baseline_outer_sha256': file_digest(d['outer_path']),
    }
    with db() as conn:
        row = conn.execute(
            "SELECT id FROM design_revisions WHERE design_id=? AND kind='outer' "
            "AND json_extract(artifacts_json,'$.outer')=? ORDER BY revision_no DESC LIMIT 1",
            (d['id'], d['outer_path']),
        ).fetchone()
    report['baseline_outer_revision_id'] = row['id'] if row else None

    from .template_source import resolve_template_source
    resolved = resolve_template_source(d)
    report['template_sha256'] = resolved.sha256
    report['template_source'] = resolved.as_report()
    if not resolved.is_user_selected:
        report['template_source_warning'] = '未找到人工确认的牙库；本次使用回退模板，验收前请确认牙库'

    diagnostics = {
        'prep_sha256': file_digest(support_ref['path']),
        'prep_components_faces': [len(part.faces) for part in support.split(only_watertight=False)],
        'prep_boundary_loops': len(_boundary_loops(support.faces)),
    }
    report['input_diagnostics'] = diagnostics
    from ..geometry.prep_margin_diagnostics import diagnose
    provenance, cropped = diagnose(support, margin, axis)
    diagnostic_path = prefix.with_name(prefix.name + '_input_diagnostics.json')
    diagnostic_path.write_text(json.dumps(provenance, ensure_ascii=False), encoding='utf-8')
    diagnostic_artifacts = {'input_diagnostics': str(diagnostic_path)}
    if cropped is not None and len(cropped.faces):
        crop_path = prefix.with_name(prefix.name + '_diagnostic_crop.stl')
        save_mesh(cropped, crop_path)
        diagnostic_artifacts['diagnostic_crop'] = str(crop_path)
    diagnostics.update(
        codes=provenance['codes'],
        cropped_components_faces=provenance.get('cropped_components_faces'),
        cropped_boundary_loops=len(provenance.get('cropped_boundary_loops_world', [])),
    )

    # Prep/Margin topology findings no longer suppress outer adaptation.  They are
    # carried as warnings and may still prevent the later intaglio/final operation.
    final_construction_risk = bool(provenance['codes'])
    try:
        _fitting, domain = fitting_support_above_margin(
            support, margin, axis, prepared=prepared_support is not None,
        )
        diagnostics['fitting_domain'] = domain
    except MarginBuildError as exc:
        reason = str(exc)
        diagnostics['failure_code'] = (
            'MARGIN_CROSSES_OPEN_BOUNDARY' if '缺口' in reason else 'PREP_MARGIN_TOPOLOGY_UNSUPPORTED'
        )
        diagnostics['reason'] = reason
        final_construction_risk = True
        report['warnings'].append({'stage': 'PREP', 'reason': reason})
        if diagnostics['failure_code'] not in provenance['codes']:
            provenance['codes'].append(diagnostics['failure_code'])
        provenance['fitting_failure_reason'] = reason
    diagnostic_path.write_text(json.dumps(provenance, ensure_ascii=False), encoding='utf-8')

    q = input_qc(mesh.vertices, mesh.faces)
    diagnostics['baseline_topology'] = q
    if q['nonmanifold_edges'] or q['degenerate_faces'] or q.get('self_intersections_truncated') or q.get('self_intersections_severe', 0):
        diagnostics['baseline_geometry_status'] = 'REVIEW_REQUIRED'
        report['warnings'].append({'stage': 'BASELINE', 'reason': 'TEMPLATE_TOPOLOGY_REVIEW_REQUIRED', 'qc': q})

    neighbors = json.loads(d.get('neighbors_json') or '{}')
    patches, unknown = {}, []
    for side in ('mesial', 'distal'):
        entry = neighbors.get(side, {})
        report['neighbor_revision_ids'][side] = entry.get('revision_id')
        if entry.get('confirmation_status') == 'ABSENT' and entry.get('frame_revision_id') == frame_ref['revision_id']:
            report['per_side'][side] = {'status': 'SKIPPED_ABSENT'}
            continue
        if entry.get('confirmation_status') != 'CONFIRMED' or entry.get('frame_revision_id') != frame_ref['revision_id']:
            report['per_side'][side] = {'status': 'INPUT_UNVERIFIED'}
            unknown.append(side)
            continue
        source = entry.get('source', {})
        current = _asset(d, source.get('kind'))
        path = entry.get('path')
        if (not current or current['id'] != source.get('asset_id')
                or file_digest(current['path']) != source.get('sha256')
                or not path or not Path(path).is_file()
                or file_digest(path) != entry.get('region_sha256')):
            raise HTTPException(409, '邻牙来源或区域已变化，请重新确认')
        patches[side] = FitSurface(load_mesh(Path(path)), entry.get('outward_normal'))
        report['per_side'][side] = {
            'status': 'CONFIRMED',
            'instance_ref': entry['instance_ref'],
            'scope': entry['scope'],
        }

    from .crown_generation_profile import SOURCE_MISSING, from_parameters
    profile = from_parameters(parameters)
    preserve_target, preserve_source = profile.value('proximal.target_distance_mm')
    proximal_target = parameters.get('proximal_target_mm')
    occlusal_target = parameters.get('occlusal_target_mm')

    # P0-A: fit body width to actual mesial/distal space before local contact work.
    mesh, prefit_report = global_patient_prefit(
        mesh, frame, margin,
        proximal_surfaces=patches,
        proximal_target_mm=proximal_target,
    )
    report['stages'].append(prefit_report)
    prefit_path = prefix.with_name(prefix.name + '_prefit.stl')
    save_mesh(mesh, prefit_path)

    # P0-B: Margin is an active geometry boundary, not merely a locked zone.
    mesh, cervical_report = cervical_fit_to_margin(mesh, frame, margin)
    report['stages'].append(cervical_report)
    cervical_path = prefix.with_name(prefix.name + '_cervical.stl')
    save_mesh(mesh, cervical_path)

    # Domains are frozen only after patient-space prefit and cervical adaptation.
    domains = fit_domains(mesh, frame, margin, patches, 'PROXIMAL')[0] if patches else {}

    def _joint(vertices, side, surface):
        if preserve_source == SOURCE_MISSING:
            return {'status': 'NOT_ASSESSED', 'reason': 'PROXIMAL_TARGET_UNCONFIRMED'}
        return measure(vertices, domains[side], surface, float(preserve_target), kind='PROXIMAL', side=side)[0]

    before_joint = {side: _joint(mesh.vertices, side, surface) for side, surface in patches.items()}
    required = []
    incomplete = False

    # P0-C: local proximal morphology/contact refinement on top of the width fit.
    if mode in ('PROXIMAL_ONLY', 'BOTH'):
        if not patches:
            report['stages'].append({'stage': 'PROXIMAL', 'status': 'NOT_ASSESSED', 'reason': 'NO_CONFIRMED_NEIGHBOR'})
            incomplete = True
        elif proximal_target is None:
            report['stages'].append({'stage': 'PROXIMAL', 'status': 'INPUT_UNVERIFIED', 'reason': 'PROXIMAL_TARGET_UNCONFIRMED'})
            incomplete = True
        else:
            targets = {side: proximal_target for side in patches}
            mesh, rep, domains = fit_stage(mesh, frame, margin, patches, targets, 'PROXIMAL', domain_mesh=mesh.copy())
            _ensure_finite_mesh(mesh, '近远中适配结果')
            before_joint = rep['before']
            report['stages'].append(rep)
            required.append(rep['status'])
            placed_path = prefix.with_name(prefix.name + '_proximal.stl')
            save_mesh(mesh, placed_path)
            report['proximal_asset'] = {'path': str(placed_path), 'sha256': file_digest(placed_path)}
        incomplete |= bool(unknown)

    occlusal = None
    occl_mask = None
    occl_before = None

    # P0-D: establish whole-crown height from antagonist, then locally refine relief.
    if mode in ('OCCLUSAL_ONLY', 'BOTH'):
        bite = json.loads(d.get('bite_json') or '{}')
        report['bite_revision_id'] = bite.get('revision_id')
        trustworthy = (
            bite.get('revision_id') and bite.get('pose_confirmed') and bite.get('transform')
            and bite.get('frame_revision_id') == frame_ref['revision_id']
        )
        if trustworthy:
            for scan in bite.get('scans', []):
                current = _asset(d, scan['kind'])
                if not current or current['id'] != scan['asset_id'] or file_digest(current['path']) != scan['sha256']:
                    raise HTTPException(409, '咬合来源扫描已变化')
        if not trustworthy or occlusal_target is None:
            report['stages'].append({
                'stage': 'OCCLUSAL',
                'status': 'NOT_ASSESSED',
                'reason': 'OCCLUSION_REGISTRATION_UNVERIFIED' if not trustworthy else 'OCCLUSAL_TARGET_UNCONFIRMED',
            })
            incomplete = True
        else:
            antagonist_kind = 'upper' if d['target_fdi'] >= 31 else 'lower'
            antagonist_asset = _asset(d, antagonist_kind)
            if not antagonist_asset:
                raise ValueError('缺少对颌扫描')
            ant = load_mesh(Path(antagonist_asset['path']))
            ant.apply_transform(np.asarray(bite['transform']))
            _ensure_finite_mesh(ant, '对颌扫描')
            if len(ant.faces) > 200000:
                raise ValueError('对颌工程预算为 200000 面，请先处理扫描并重新确认')
            occlusal = FitSurface(ant, (-frame.z).tolist())

            mesh, height_report = global_patient_prefit(
                mesh, frame, margin,
                antagonist_surface=occlusal,
                occlusal_target_mm=occlusal_target,
            )
            report['stages'].append({**height_report, 'stage': 'OCCLUSAL_HEIGHT_PREFIT'})
            # Reassert the Margin after global body-height scaling.
            mesh, cervical_after_height = cervical_fit_to_margin(mesh, frame, margin)
            report['stages'].append({**cervical_after_height, 'stage': 'CERVICAL_REFIT_AFTER_HEIGHT'})
            height_path = prefix.with_name(prefix.name + '_height_prefit.stl')
            save_mesh(mesh, height_path)

            protected = np.zeros(len(mesh.vertices), bool)
            if preserve_source == SOURCE_MISSING:
                report['occlusal_protection'] = {'status': 'NOT_ASSESSED', 'reason': 'PROXIMAL_TARGET_UNCONFIRMED'}
            else:
                for side, surface in patches.items():
                    if side not in domains:
                        continue
                    _, query = measure(
                        mesh.vertices, domains[side], surface, float(preserve_target),
                        kind='PROXIMAL', side=side,
                    )
                    if query is not None:
                        ids, _, _, signed, _, valid = query
                        protected[ids[valid & (np.abs(signed - float(preserve_target)) <= .15)]] = True

            domain_reference = mesh.copy()
            mesh, rep, masks = fit_stage(
                mesh, frame, margin,
                {'occlusal': occlusal}, {'occlusal': occlusal_target}, 'OCCLUSAL',
                protected=protected, domain_mesh=domain_reference,
            )
            _ensure_finite_mesh(mesh, '咬合适配结果')
            occl_mask = masks['occlusal']
            occl_before = measure(
                domain_reference.vertices, occl_mask, occlusal, occlusal_target, kind='OCCLUSAL'
            )[0]

            # P1 policy: preserve the actual generated geometry.  Contact/crossing
            # regressions are warnings for review, never implicit rollback to the template.
            nonworse = True
            for side, surface in patches.items():
                if side not in domains:
                    continue
                after = _joint(mesh.vertices, side, surface)
                previous = _joint(domain_reference.vertices, side, surface)
                old = previous.get('target_error')
                new = after.get('target_error')
                if old and (not new or new['max_mm'] > old['max_mm'] + 1e-9):
                    nonworse = False
            if not nonworse:
                report['warnings'].append({'stage': 'OCCLUSAL', 'reason': 'JOINT_CONTACT_PRESERVATION_FAILED'})
                rep['review_warning'] = 'JOINT_CONTACT_PRESERVATION_FAILED'
            report['stages'].append(rep)
            required.append(rep['status'])
            occlusal_path = prefix.with_name(prefix.name + '_occlusal.stl')
            save_mesh(mesh, occlusal_path)

    # P1 QC: measure, never mutate/rollback the candidate.
    from ..geometry.crown_outer.deform_gate import deformation_gate
    from ..geometry.nonplanar_crown import MarginCurtain
    audited = {side: (surface, domains[side]) for side, surface in patches.items() if side in domains}
    if occl_mask is not None:
        audited['occlusal'] = (occlusal, occl_mask)
    crossing_before = {
        side: surface_crossings(baseline, surface, mask)
        for side, (surface, mask) in audited.items()
    }
    crossing_after = {
        side: surface_crossings(mesh, surface, mask)
        for side, (surface, mask) in audited.items()
    }
    cross_ok = all(
        crossing_after[side].get('crossing_pairs') is not None
        and crossing_before[side].get('crossing_pairs') is not None
        and crossing_after[side]['crossing_pairs'] <= crossing_before[side]['crossing_pairs']
        for side in audited
    ) if audited else True
    report['joint_surface_crossings'] = {
        'before': crossing_before,
        'candidate': crossing_after,
        'nonworse': cross_ok,
    }
    if not cross_ok:
        report['joint_surface_crossings']['reason'] = 'CROSSING_WORSENED_OR_NOT_ASSESSED'
        report['warnings'].append({'stage': 'GEOMETRY_QC', 'reason': 'CROSSING_WORSENED_OR_NOT_ASSESSED'})

    locked = np.flatnonzero(MarginCurtain(margin, axis).axial_above(baseline.vertices) <= .4)
    moved = np.linalg.norm(mesh.vertices - baseline.vertices, axis=1) > 1e-9
    cumulative = deformation_gate(
        baseline.vertices, mesh.vertices, mesh.faces, scale_mm=.1, locked_ids=locked,
        focus_faces=np.flatnonzero(np.any(moved[mesh.faces], axis=1)),
    )
    flips = int((np.einsum('ij,ij->i', baseline.face_normals, mesh.face_normals) <= 0).sum())
    report['cumulative_geometry_gate'] = {**cumulative.as_dict(), 'all_flips': flips}
    if not cumulative.ok or flips:
        report['warnings'].append({
            'stage': 'GEOMETRY_QC',
            'reason': 'DEFORMATION_REVIEW_REQUIRED',
            'flips': flips,
            'gate': cumulative.as_dict(),
        })

    after_joint = {
        side: _joint(mesh.vertices, side, surface)
        for side, surface in patches.items() if side in domains
    }
    joint = {
        'before': before_joint,
        'after': after_joint,
        'occlusal_before': occl_before,
        'occlusal_after': measure(
            mesh.vertices, occl_mask, occlusal, occlusal_target, kind='OCCLUSAL'
        )[0] if occl_mask is not None else None,
    }
    report['joint_recheck'] = joint
    report['metrics_before'] = {'proximal': before_joint, 'occlusal': occl_before}
    report['metrics_after'] = {'proximal': after_joint, 'occlusal': joint['occlusal_after']}
    changed = np.linalg.norm(mesh.vertices - baseline.vertices, axis=1)
    report['displacement'] = {
        'max_mm': float(changed.max()),
        'p95_mm': float(np.quantile(changed, .95)),
        'median_mm': float(np.median(changed)),
        'moved_vertex_count': int(np.sum(changed > 1e-5)),
    }

    outer_path = prefix.with_suffix('.stl')
    save_mesh(mesh, outer_path)
    _ensure_finite_mesh(load_mesh(outer_path), '候选 STL')

    # Audit actual persisted vertex correspondence after STL quantization.
    from scipy.spatial import cKDTree
    persisted = load_mesh(outer_path)
    error, ids = cKDTree(persisted.vertices).query(mesh.vertices)
    if error.max() > 1e-5 or len(np.unique(ids)) != len(mesh.vertices):
        raise ValueError('候选 STL 重读顶点对应失效')
    ordered = persisted.vertices[ids]
    report['joint_recheck']['after'] = {
        side: _joint(ordered, side, surface)
        for side, surface in patches.items() if side in domains
    }
    report['metrics_after']['proximal'] = report['joint_recheck']['after']
    if occl_mask is not None:
        report['joint_recheck']['occlusal_after'] = measure(
            ordered, occl_mask, occlusal, occlusal_target, kind='OCCLUSAL'
        )[0]
        report['metrics_after']['occlusal'] = report['joint_recheck']['occlusal_after']
    report['candidate_readback_max_mm'] = float(error.max())
    report['candidate_outer_sha256'] = file_digest(outer_path)

    if incomplete:
        report['status'] = 'PARTIAL_REVIEW_REQUIRED'
    elif report['warnings'] or any(status not in ('PASS_ENGINEERING', 'PLACED') for status in required):
        report['status'] = 'REVIEW_REQUIRED'
    else:
        report['status'] = 'PASS_ENGINEERING'

    artifacts = {
        'outer': str(outer_path),
        'prefit': str(prefit_path),
        'cervical': str(cervical_path),
        **diagnostic_artifacts,
    }
    if report.get('proximal_asset'):
        artifacts['proximal'] = report['proximal_asset']['path']
    if 'height_path' in locals():
        artifacts['height_prefit'] = str(height_path)
    if 'occlusal_path' in locals():
        artifacts['occlusal'] = str(occlusal_path)

    # P1 intaglio/final: best effort.  Failure is visible but never replaces the
    # already generated patient-specific outer crown.
    final_path = prefix.with_name(prefix.name + '_w3_final.stl')
    cutter_path = prefix.with_name(prefix.name + '_w3_cutter.stl')
    try:
        from .contact_measurement import thickness_protection_surfaces
        protection = thickness_protection_surfaces(d) if parameters.get('enforce_minimum_thickness') else ()
        _, _, _, result = finalize_crown(
            outer_path, Path(support_ref['path']), final_path, cutter_path,
            margin, axis, parameters,
            prepared_support=prepared_support,
            support_contract=support_contract,
            protected_surfaces=protection,
        )
        qc = result['w3']['qc']
        report['w3_preflight'] = {'status': 'CONSTRUCTED_ENGINEERING', 'qc': qc}
        try:
            audit_final_context(d, result['w3']['artifact_paths']['outer_surface'], final_path, qc)
        except (ValueError, HTTPException) as exc:
            report['warnings'].append({'stage': 'FINAL_QC', 'reason': str(exc)})
        audit_path = Path(result['w3']['artifact_paths']['manufacturing_qc'])
        stored = json.loads(audit_path.read_text(encoding='utf-8'))
        stored['qc'] = qc
        audit_path.write_text(json.dumps(stored, ensure_ascii=False), encoding='utf-8')
        report['thickness_audit_status'] = qc['FinalCrownThicknessAudit']['status']
        if qc['FinalCrownThicknessAudit']['body_thin_face_sample_area_mm2'] > 0:
            report['warnings'].append({'stage': 'THICKNESS', 'reason': 'ENGINEERING_MINIMUM_THICKNESS_VIOLATED'})
            report['w3_preflight']['reason'] = 'ENGINEERING_MINIMUM_THICKNESS_VIOLATED'
            report['status'] = 'REVIEW_REQUIRED'
        artifacts.update(final=str(final_path), cutter=str(cutter_path), **result['w3']['artifact_paths'])
    except (ValueError, MarginBuildError) as exc:
        report['w3_preflight'] = {
            'status': 'BLOCKED',
            'reason': str(exc),
            'prep_margin_risk': final_construction_risk,
        }
        report['warnings'].append({'stage': 'INTAGLIO_FINAL', 'reason': str(exc)})
        if report['status'] == 'PASS_ENGINEERING':
            report['status'] = 'REVIEW_REQUIRED'

    report['artifact_sha256'] = {key: file_digest(path) for key, path in artifacts.items()}
    return mesh, report, artifacts
