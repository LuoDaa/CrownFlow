"""Read-only eligibility and deterministic generation candidates."""
import json
from pathlib import Path
from fastapi import HTTPException
from .design_dependencies import confirmed_inputs, dependency_digest, file_digest

LATEST_AUTO_NEIGHBOR_SCOPE = 'AUTO_CAD1_CORONAL_MD_NEIGHBOR_V5'
CAD1_DEFAULT_PROXIMAL_TARGET_MM = -0.02
CAD1_DEFAULT_OCCLUSAL_TARGET_MM = 0.02


def _resolved_contact_targets(params):
    """CAD-1 always has explicit usable contact defaults.

    Existing designs created before these fields were persisted must not silently
    degrade to PROXIMAL_ONLY / no occlusion. User values still win.
    """
    p = params.get('proximal_target_mm')
    o = params.get('occlusal_target_mm')
    return (
        CAD1_DEFAULT_PROXIMAL_TARGET_MM if p is None else float(p),
        CAD1_DEFAULT_OCCLUSAL_TARGET_MM if o is None else float(o),
        'DEFAULT_POLICY' if p is None else 'USER',
        'DEFAULT_POLICY' if o is None else 'USER',
    )


def _latest_stage(report, *names):
    """Return the last matching stage report without assuming stage ordering."""
    wanted = set(names)
    for stage in reversed((report or {}).get('stages') or []):
        if stage.get('stage') in wanted:
            return stage
    return None


def _apply_auto_candidate_safety(d, report, artifacts):
    """Fail closed when a generated outer did not actually satisfy patient space.

    AUTO candidates used to stay accept-able merely because an ``outer`` artifact
    existed, even if the run had silently degraded to PROXIMAL_ONLY or the inner
    shell pass had moved a large part of the mesh and ended in REVIEW_REQUIRED.
    That is exactly the failure mode that can compound template deformation and
    feed a self-intersecting crown into final construction.

    This policy does not discard diagnostic artifacts.  It only changes the
    candidate kind/status so the existing accept endpoint cannot treat an unsafe
    AUTO result as an automatically acceptable crown.
    """
    reasons = []

    if d.get('support_type') == 'abutment':
        if report.get('mode') != 'BOTH':
            reasons.append({
                'code': 'ABUTMENT_AUTO_REQUIRES_PROXIMAL_AND_OCCLUSAL_SPACE',
                'mode': report.get('mode'),
            })
        height = _latest_stage(report, 'OCCLUSAL_HEIGHT_PREFIT')
        occlusion = _latest_stage(report, 'OCCLUSAL', 'OCCLUSAL_RECHECK_AFTER_SHELL')
        if not report.get('bite_revision_id') or height is None or occlusion is None:
            reasons.append({
                'code': 'OCCLUSAL_STAGE_MISSING',
                'bite_revision_id': report.get('bite_revision_id'),
                'height_stage_present': height is not None,
                'occlusal_stage_present': occlusion is not None,
            })

    shell_decision = _latest_stage(report, 'SHELL_REPAIR_DECISION')
    shell_final = _latest_stage(report, 'INNER_SHELL_FINAL_AUDIT')
    if shell_decision and shell_decision.get('major_change'):
        shell_ok = shell_final and shell_final.get('status') == 'PASS_ENGINEERING'
        if not shell_ok:
            reasons.append({
                'code': 'LARGE_SHELL_DEFORMATION_NOT_CERTIFIED',
                'moved_vertices': shell_decision.get('moved_vertices'),
                'moved_fraction': shell_decision.get('moved_fraction'),
                'max_move_mm': shell_decision.get('max_move_mm'),
                'shell_final_status': None if shell_final is None else shell_final.get('status'),
            })

    if reasons:
        report['candidate_kind'] = 'AUTO_FIT_BLOCKED'
        report['status'] = 'REVIEW_REQUIRED'
        report['complete_crown_ready'] = False
        report['automatic_accept_allowed'] = False
        report['generation_safety'] = {
            'status': 'BLOCKED',
            'policy': 'PATIENT_SPACE_FAIL_CLOSED_V1',
            'reasons': reasons,
        }
        report.setdefault('warnings', []).append({
            'stage': 'GENERATION_SAFETY',
            'reason': 'PATIENT_SPACE_CONSTRAINTS_NOT_CERTIFIED',
            'details': reasons,
        })
    else:
        report['automatic_accept_allowed'] = report.get('status') == 'PASS_ENGINEERING'
        report['generation_safety'] = {
            'status': 'PASS_ENGINEERING',
            'policy': 'PATIENT_SPACE_FAIL_CLOSED_V1',
            'reasons': [],
        }
    return report, artifacts


def eligibility(d):
    from ..api.routes import asset
    from ..api.auto_fit import frame_from
    from ..api.design_inputs import scan_reference
    blockers=[]
    try: confirmed_inputs(d)
    except HTTPException as exc: blockers.append({'stage':'PREP','message':exc.detail})
    support=asset(d['case_id'],'support')
    if not support: blockers.append({'stage':'PREP','message':'请定位预备体或上传独立预备体'})
    from .abutment_support import prepare_design_support
    support_contract,_=prepare_design_support(d,require_ready=False)
    if support_contract and not support_contract.ready:
        blockers.append({'stage':'SUPPORT','message':support_contract.rejection_reason or support_contract.rejection_code,'code':support_contract.rejection_code})
    frame=None
    try: _,frame=frame_from(d)
    except (HTTPException,KeyError): pass
    if not frame:
        blockers.append({'stage':'FRAME','message':'请先确认近中/颊侧方向与毫米单位（坐标确认），再生成外形'})

    neighbors=json.loads(d.get('neighbors_json') or '{}');sides={};stale_auto=[]
    for side in ('mesial','distal'):
        item=neighbors.get(side,{})
        valid=bool(frame and item.get('frame_revision_id')==frame['revision_id'])
        scope=str(item.get('scope') or '')
        if (
            d.get('support_type')=='abutment'
            and scope.startswith('AUTO_CAD1_CORONAL_MD_NEIGHBOR_')
            and scope != LATEST_AUTO_NEIGHBOR_SCOPE
        ):
            valid=False
            stale_auto.append(side)
        if valid and item.get('confirmation_status')=='CONFIRMED':
            try:
                valid=(scan_reference(d['case_id'],item['source']['kind'])==item['source']
                       and Path(item['path']).is_file() and file_digest(item['path'])==item['region_sha256'])
            except (HTTPException,KeyError,OSError): valid=False
        sides[side]=item.get('confirmation_status','UNKNOWN') if valid else 'UNKNOWN'
    if stale_auto:
        blockers.append({
            'stage':'PROXIMAL',
            'code':'AUTO_NEIGHBOR_REFERENCE_STALE',
            'message':'自动邻牙参考面算法版本已更新，请先刷新邻牙参考后再生成',
            'sides':stale_auto,
            'required_scope':LATEST_AUTO_NEIGHBOR_SCOPE,
        })

    bite=json.loads(d.get('bite_json') or '{}');bite_valid=bool(frame and bite.get('revision_id')
        and bite.get('pose_confirmed') and bite.get('frame_revision_id')==frame['revision_id'] and bite.get('transform'))
    if bite_valid:
        try: bite_valid=all(scan_reference(d['case_id'],s['kind'])==s for s in bite['scans']) and len(bite['scans'])==2
        except (HTTPException,KeyError): bite_valid=False
    if d['support_type']=='abutment' and not bite_valid:
        blockers.append({'stage':'OCCLUSAL','code':'OCCLUSION_REGISTRATION_UNVERIFIED',
                         'message':'基台冠需先确认上下颌咬合配准，才能测量对颌空间并生成初稿'})
    params=json.loads(d['parameters_json'])
    proximal_target,occlusal_target,prox_source,occl_source=_resolved_contact_targets(params)
    proximal=bool(frame and any(v=='CONFIRMED' for v in sides.values())
        and all(v in ('CONFIRMED','ABSENT') for v in sides.values()))
    occlusal=bool(bite_valid)
    mode='BOTH' if proximal and occlusal else 'PROXIMAL_ONLY' if proximal else 'OCCLUSAL_ONLY' if occlusal else None
    return {'input_digest':dependency_digest(d),'blockers':blockers,'template_allowed':not blockers,
        'support_contract':support_contract.as_report() if support_contract else None,
        'automatic_mode':mode if not blockers else None,'frame_confirmed':bool(frame),'neighbors':sides,
        'bite_confirmed':bite_valid,'proximal_target_confirmed':True,
        'occlusal_target_confirmed':True,
        'proximal_target_mm':proximal_target,'occlusal_target_mm':occlusal_target,
        'proximal_target_source':prox_source,'occlusal_target_source':occl_source,
        'neighbor_reference_version':LATEST_AUTO_NEIGHBOR_SCOPE,
        'release_status':'NOT_PRODUCTION_APPROVED'}


def generate_candidate(d,kind):
    from ..api.routes import asset, initialize_draft_outer
    from ..models import AutoOuterRequest
    from ..geometry.io import load_mesh
    from ..geometry.prep_margin_diagnostics import diagnose
    from ..geometry.nonplanar_crown import fitting_support_above_margin
    from ..api.auto_fit import frame_from
    from .auto_fit import propose
    state=eligibility(d)
    if state['blockers']: raise HTTPException(409,{'message':'请先确认关键输入','blockers':state['blockers']})
    if kind=='AUTO' and not state['automatic_mode']:
        raise HTTPException(409,{'message':'尚无可执行的自动适配。请确认方向、邻牙或咬合输入；也可明确选择仅模板初稿。','eligibility':state})
    consumed=confirmed_inputs(d);margin=consumed['margin'].payload['points'];axis=consumed['axis'].payload['axis']
    support=load_mesh(Path(asset(d['case_id'],'support')['path']))
    from .abutment_support import prepare_design_support
    support_contract,prepared=prepare_design_support(d)
    if prepared:
        support=prepared.mesh
    diagnostics,_=diagnose(support,margin,axis)
    if support_contract:diagnostics['support_contract']=support_contract.as_report()
    try: fitting_support_above_margin(support,margin,axis,prepared=prepared is not None)
    except ValueError as exc:
        raise HTTPException(422,{'message':str(exc),'geometry':diagnostics,'repair_stage':'PREP'}) from exc
    if diagnostics['codes']:
        raise HTTPException(422,{'message':'预备体组件或边界不满足构冠要求，请检查边缘或上传独立预备体',
            'geometry':diagnostics,'repair_stage':'PREP'})

    # One-click AUTO is an idempotent build from the same confirmed inputs.
    # Never use a previously accepted outer as the next automatic baseline:
    # doing so compounds proximal/occlusal deformation every time the user clicks
    # Generate.  The same generation job now also builds the intaglio + stitched
    # watertight final so Step 5 starts from a complete crown, not a solid outer.
    placement,baseline=initialize_draft_outer(
        d,AutoOuterRequest(margin_points=margin,insertion_axis=axis))
    if kind=='AUTO':
        placement['placement_path']='FRESH_TEMPLATE_INITIALIZATION'

    if kind=='AUTO':
        effective=dict(d)
        effective_params=json.loads(d['parameters_json'])
        effective_params.setdefault('proximal_target_mm',state['proximal_target_mm'])
        effective_params.setdefault('occlusal_target_mm',state['occlusal_target_mm'])
        effective['parameters_json']=json.dumps(effective_params,ensure_ascii=False)
        frame,ref=frame_from(effective);_,report,artifacts=propose(
            effective,frame,ref,state['automatic_mode'],baseline_path=baseline,run_w3_preflight=True)
        artifacts['baseline']=str(baseline);report['candidate_kind']='AUTO_FIT_CANDIDATE'
        report['baseline_source']=placement.get('placement_path') or 'TEMPLATE_INITIALIZATION'
        report['resolved_contact_targets']={
            'proximal_target_mm':state['proximal_target_mm'],
            'occlusal_target_mm':state['occlusal_target_mm'],
            'proximal_source':state['proximal_target_source'],
            'occlusal_source':state['occlusal_target_source'],
        }
        final_ready=(report.get('w3_preflight') or {}).get('status')=='CONSTRUCTED_ENGINEERING' \
            and bool(artifacts.get('final')) and bool(artifacts.get('cutter'))
        report['complete_crown_ready']=bool(final_ready)
        if not final_ready:
            report['warnings'].append({'stage':'FINAL_CONSTRUCTION','reason':
                (report.get('w3_preflight') or {}).get('reason') or 'FINAL_OR_INTAGLIO_NOT_CONSTRUCTED'})
            report['status']='REVIEW_REQUIRED'
        report,artifacts=_apply_auto_candidate_safety(effective,report,artifacts)
    else:
        report={'type':'template_draft','candidate_kind':'TEMPLATE_ONLY_DRAFT','status':'TEMPLATE_REVIEW_REQUIRED',
            'stages':[{'stage':'TEMPLATE','status':'PLACED'},{'stage':'PROXIMAL','status':'NOT_ASSESSED'},
                      {'stage':'OCCLUSAL','status':'NOT_ASSESSED'}],
            'w3_preflight':{'status':'NOT_ASSESSED','reason':'模板草稿不生成最终内冠；请使用一键自动生成完成冠体'},
            'complete_crown_ready':False}
        artifacts={'outer':str(baseline),'baseline':str(baseline)}
    report.update(eligibility=state,input_diagnostics=diagnostics,
        completed_stages=[s['stage'] for s in report['stages'] if s['status'] in ('PLACED','PASS_ENGINEERING')],
        blocked_stages=[s for s in report['stages'] if s['status'] in ('NOT_ASSESSED','INPUT_UNVERIFIED','BLOCKED')],
        review_required_stages=['CANDIDATE','FINAL_QC'],manufacturing_status='NOT_PRODUCTION_APPROVED',
        candidate_outer_sha256=file_digest(artifacts['outer']),
        artifact_sha256={k:file_digest(v) for k,v in artifacts.items()})
    report['template_source']=placement.get('template_source') or report.get('template_source')
    report['template_source_warning']=placement.get('template_source_warning') or report.get('template_source_warning')
    report['frame_source']=placement.get('frame_source') or report.get('frame_source')
    report['patient_frame']={'revision_id':(json.loads(d.get('autofit_frame_json') or 'null') or {}).get('revision_id'),
                             'source':'CONFIRMED'}
    report['placement_path']=placement.get('placement_path') or report.get('placement_path')
    report['baseline_sha256']=file_digest(artifacts['baseline'])
    from .contact_measurement import measure_confirmed_inputs, serializable_measurement
    from fastapi import HTTPException as _HTTPException
    parameters=json.loads(d['parameters_json'])
    proximal_target,occlusal_target,_ps,_os=_resolved_contact_targets(parameters)
    try:
        contact=measure_confirmed_inputs(d,json.loads(d.get('autofit_frame_json') or 'null') or {},
            artifacts['baseline'],target_proximal=proximal_target,
            target_occlusal=occlusal_target,strict=False)
        report['placement_contact']=serializable_measurement(contact)
    except (_HTTPException,ValueError,OSError) as exc:
        report['placement_contact']={'status':'NOT_ASSESSED','reason':'PLACEMENT_SNAPSHOT_FAILED: %s'%exc}
    return report,artifacts
