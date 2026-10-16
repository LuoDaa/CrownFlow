"""Read-only eligibility and deterministic generation candidates.

AUTO v1.0 is generation-first.  Margin/frame/support are the only inputs that can
prevent a run from starting.  The primary route uses a closed library crown as
anatomy/topology prior and lets patient geometry own all metric limits:

    template -> rigid patient placement -> uniform overscale -> patient-envelope
    soft compression -> one constraint pass -> latest geometry is always kept.

The older procedural patient-space embryo is retained only as a fail-open
fallback while the template-constrained route is validated on real cases.
"""
import json
from pathlib import Path

from fastapi import HTTPException

from .design_dependencies import dependency_digest, file_digest

LATEST_AUTO_NEIGHBOR_SCOPE = 'AUTO_CAD1_CORONAL_MD_NEIGHBOR_V5'
CAD1_DEFAULT_PROXIMAL_TARGET_MM = -0.02
CAD1_DEFAULT_OCCLUSAL_TARGET_MM = 0.02


def _resolved_contact_targets(params):
    p = params.get('proximal_target_mm')
    o = params.get('occlusal_target_mm')
    return (
        CAD1_DEFAULT_PROXIMAL_TARGET_MM if p is None else float(p),
        CAD1_DEFAULT_OCCLUSAL_TARGET_MM if o is None else float(o),
        'DEFAULT_POLICY' if p is None else 'USER',
        'DEFAULT_POLICY' if o is None else 'USER',
    )


def _latest_stage(report, *names):
    wanted = set(names)
    for stage in reversed((report or {}).get('stages') or []):
        if stage.get('stage') in wanted:
            return stage
    return None


def _apply_auto_candidate_safety(d, report, artifacts):
    """Record observations without suppressing a generated outer in v1.0."""
    observations = []
    if d.get('support_type') == 'abutment' and report.get('mode') != 'BOTH':
        observations.append({'code': 'FULL_SPACE_NOT_AVAILABLE', 'mode': report.get('mode')})
    shell = _latest_stage(report, 'INNER_SHELL_FINAL_AUDIT')
    if shell and shell.get('status') not in (None, 'PASS_ENGINEERING'):
        observations.append({'code': 'SHELL_REVIEW', 'status': shell.get('status')})
    report['generation_safety'] = {
        'status': 'RECORDED',
        'policy': 'MINIMAL_AUDIT_FAIL_OPEN_TEST_V1',
        'output_preserved': bool(artifacts.get('outer')),
        'observations': observations,
    }
    report['automatic_accept_allowed'] = False
    return report, artifacts


def eligibility(d):
    from ..api.routes import asset
    from ..api.auto_fit import frame_from
    from ..api.design_inputs import scan_reference

    blockers = []
    advisories = []
    support = asset(d['case_id'], 'support')
    if not support:
        blockers.append({'stage': 'SUPPORT', 'message': '缺少 support，无法定义冠体根部空间'})

    margin = json.loads(d.get('margin_json') or 'null')
    axis = json.loads(d.get('axis_json') or 'null')
    if not margin:
        blockers.append({'stage': 'MARGIN', 'message': '缺少 Margin'})
    if not axis:
        blockers.append({'stage': 'AXIS', 'message': '缺少插入轴'})

    frame = None
    try:
        _, frame = frame_from(d)
    except (HTTPException, KeyError, TypeError, ValueError):
        pass
    if not frame:
        blockers.append({'stage': 'FRAME', 'message': '缺少患者近远中/颊舌向坐标系'})

    neighbors = json.loads(d.get('neighbors_json') or '{}')
    sides = {}
    for side in ('mesial', 'distal'):
        item = neighbors.get(side, {})
        valid = bool(frame and item.get('frame_revision_id') == frame.get('revision_id') if isinstance(frame, dict) else True)
        if item.get('confirmation_status') == 'CONFIRMED':
            try:
                valid = bool(
                    valid and Path(item['path']).is_file()
                    and file_digest(item['path']) == item['region_sha256']
                    and scan_reference(d['case_id'], item['source']['kind']) == item['source']
                )
            except (HTTPException, KeyError, OSError):
                valid = False
        sides[side] = item.get('confirmation_status', 'UNKNOWN') if valid else 'UNKNOWN'

    bite = json.loads(d.get('bite_json') or '{}')
    bite_valid = bool(frame and bite.get('revision_id') and bite.get('pose_confirmed') and bite.get('transform'))
    if not bite_valid:
        advisories.append({
            'stage': 'OCCLUSAL', 'code': 'NO_TRUSTED_BITE',
            'message': '本轮仍生成，但不把对颌作为可信高度所有者',
        })

    params = json.loads(d.get('parameters_json') or '{}')
    proximal_target, occlusal_target, prox_source, occl_source = _resolved_contact_targets(params)
    proximal = bool(any(v == 'CONFIRMED' for v in sides.values()))
    occlusal = bool(bite_valid)
    mode = (
        'BOTH' if proximal and occlusal else
        'PROXIMAL_ONLY' if proximal else
        'OCCLUSAL_ONLY' if occlusal else
        'SPACE_ONLY'
    )
    return {
        'input_digest': dependency_digest(d),
        'blockers': blockers,
        'advisories': advisories,
        'automatic_mode': mode if not blockers else None,
        'frame_confirmed': bool(frame),
        'neighbors': sides,
        'bite_confirmed': bite_valid,
        'proximal_target_mm': proximal_target,
        'occlusal_target_mm': occlusal_target,
        'proximal_target_source': prox_source,
        'occlusal_target_source': occl_source,
        'generation_policy': 'TEMPLATE_CONSTRAINED_FAIL_OPEN_TEST_V1',
        'release_status': 'NOT_PRODUCTION_APPROVED',
    }


def _fallback_report(mode, outer, exc, *, stage):
    return {
        'mode': mode,
        'status': 'REVIEW_REQUIRED',
        'stages': [{
            'stage': stage,
            'status': 'REVIEW_REQUIRED',
            'reason': str(exc),
            'output_preserved': True,
        }],
        'warnings': [{'stage': stage, 'reason': str(exc)}],
        'candidate_kind': 'AUTO_FIT_CANDIDATE',
        'w3_preflight': {'status': 'NOT_COMPLETED', 'reason': str(exc)},
        'complete_crown_ready': False,
        'candidate_outer_sha256': file_digest(outer),
    }


def generate_candidate(d, kind):
    from ..api.routes import asset, initialize_draft_outer
    from ..models import AutoOuterRequest
    from ..geometry.io import load_mesh
    from ..geometry.prep_margin_diagnostics import diagnose
    from ..api.auto_fit import frame_from
    from .auto_fit import propose

    state = eligibility(d)
    if state['blockers']:
        raise HTTPException(409, {
            'message': '缺少建立冠体几何所必需的最小输入',
            'blockers': state['blockers'],
        })

    margin = json.loads(d['margin_json'])
    axis = json.loads(d['axis_json'])
    support_ref = asset(d['case_id'], 'support')
    support = load_mesh(Path(support_ref['path']))
    diagnostics, _ = diagnose(support, margin, axis)

    if kind != 'AUTO':
        placement, baseline = initialize_draft_outer(
            d, AutoOuterRequest(margin_points=margin, insertion_axis=axis)
        )
        report = {
            'type': 'template_draft',
            'candidate_kind': 'TEMPLATE_ONLY_DRAFT',
            'status': 'TEMPLATE_REVIEW_REQUIRED',
            'stages': [{'stage': 'TEMPLATE', 'status': 'PLACED'}],
            'warnings': [],
            'w3_preflight': {'status': 'NOT_ASSESSED'},
            'complete_crown_ready': False,
        }
        artifacts = {'outer': str(baseline), 'baseline': str(baseline)}
    else:
        effective = dict(d)
        effective_params = json.loads(d.get('parameters_json') or '{}')
        effective_params.setdefault('proximal_target_mm', state['proximal_target_mm'])
        effective_params.setdefault('occlusal_target_mm', state['occlusal_target_mm'])
        effective['parameters_json'] = json.dumps(effective_params, ensure_ascii=False)
        frame, ref = frame_from(effective)

        seed_fallback = None
        try:
            from .template_constrained_generation import build_template_constrained_seed
            baseline, seed_report = build_template_constrained_seed(
                effective, frame, ref, support, margin, effective_params, state
            )
            placement = {
                'placement_path': 'TEMPLATE_CONSTRAINED_PATIENT_SPACE_V1',
                'frame_source': 'CONFIRMED_FRAME',
                'template_source': {
                    'status': 'USED_AS_ANATOMY_TOPOLOGY_PRIOR',
                    'library': seed_report.get('library'),
                    'route': 'TEMPLATE_CONSTRAINED_CROWN_V1',
                },
                'template_source_warning': None,
            }
        except Exception as exc:
            # 1.0 fail-open: a missing/bad library must not suppress a test result.
            from .space_driven_generation import build_space_driven_seed
            baseline, seed_report = build_space_driven_seed(
                effective, frame, ref, support, margin, effective_params, state
            )
            seed_fallback = str(exc)
            seed_report['template_route_failure'] = seed_fallback
            seed_report['route'] = 'PROCEDURAL_FALLBACK_AFTER_TEMPLATE_ROUTE_FAILURE'
            placement = {
                'placement_path': 'PROCEDURAL_PATIENT_ENVELOPE_FALLBACK',
                'frame_source': 'CONFIRMED_FRAME',
                'template_source': {'status': 'FALLBACK_FAILED', 'route': 'SPACE_DRIVEN_AUTO_V1'},
                'template_source_warning': seed_fallback,
            }

        attempts = []
        try:
            # Single solver pass only.  Do not feed a constrained output through a
            # second anatomy/recheck deformation cycle in 1.0 test mode.
            _, report, artifacts = propose(
                effective, frame, ref, state['automatic_mode'],
                baseline_path=baseline, run_w3_preflight=True,
            )
            attempts.append({
                'attempt': 'SINGLE_CONSTRAINT_PASS',
                'status': report.get('status'),
                'w3_status': (report.get('w3_preflight') or {}).get('status'),
                'outer': artifacts.get('outer'),
                'output_preserved': True,
            })
        except Exception as exc:
            report = _fallback_report(
                state['automatic_mode'], baseline, exc, stage='SINGLE_CONSTRAINT_PASS'
            )
            artifacts = {'outer': str(baseline)}
            attempts.append({
                'attempt': 'SINGLE_CONSTRAINT_PASS',
                'status': 'NOT_COMPLETED',
                'reason': str(exc),
                'outer': str(baseline),
                'output_preserved': True,
            })

        report.setdefault('stages', [])
        report['stages'] = [seed_report] + report['stages']
        report['generation_attempts'] = attempts
        report['candidate_kind'] = 'AUTO_FIT_CANDIDATE'
        report['baseline_source'] = (
            'TEMPLATE_CONSTRAINED_WATERTIGHT_SEED'
            if not seed_fallback else 'PATIENT_ENVELOPE_PROCEDURAL_FALLBACK'
        )
        report['generation_route'] = (
            'TEMPLATE_RIGID_PLACEMENT_TO_UNIFORM_OVERSCALE_TO_PATIENT_ENVELOPE_'
            'SOFT_COMPRESSION_TO_SINGLE_CONSTRAINT_PASS_V1'
            if not seed_fallback else
            'PROCEDURAL_FALLBACK_TO_SINGLE_CONSTRAINT_PASS_V1'
        )
        report['template_role'] = (
            'ANATOMY_AND_CLOSED_TOPOLOGY_PRIOR'
            if not seed_fallback else 'TEMPLATE_ROUTE_FAILED_FALLBACK_USED'
        )
        if seed_fallback:
            report.setdefault('warnings', []).append({
                'stage': 'TEMPLATE_CONSTRAINED_SEED',
                'reason': seed_fallback,
                'fallback': 'PROCEDURAL_PATIENT_ENVELOPE',
            })
        artifacts['baseline'] = str(baseline)
        artifacts['template_constrained_seed' if not seed_fallback else 'space_seed'] = str(baseline)
        report['complete_crown_ready'] = bool(
            (report.get('w3_preflight') or {}).get('status') == 'CONSTRUCTED_ENGINEERING'
            and artifacts.get('final') and artifacts.get('cutter')
        )
        report, artifacts = _apply_auto_candidate_safety(effective, report, artifacts)

    report.update(
        eligibility=state,
        input_diagnostics=diagnostics,
        manufacturing_status='NOT_PRODUCTION_APPROVED',
        output_policy='ALWAYS_RETURN_LATEST_GEOMETRY_TEST_V1',
        candidate_outer_sha256=file_digest(artifacts['outer']),
        artifact_sha256={
            k: file_digest(v) for k, v in artifacts.items()
            if isinstance(v, (str, Path)) and Path(v).is_file()
        },
    )
    report['template_source'] = placement.get('template_source') or report.get('template_source')
    report['frame_source'] = placement.get('frame_source') or report.get('frame_source')
    report['patient_frame'] = {
        'revision_id': (json.loads(d.get('autofit_frame_json') or 'null') or {}).get('revision_id'),
        'source': 'CONFIRMED',
    }
    report['placement_path'] = placement.get('placement_path') or report.get('placement_path')
    report['baseline_sha256'] = file_digest(artifacts['baseline'])
    return report, artifacts
