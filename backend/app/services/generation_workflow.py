"""Read-only eligibility and generation candidates without replacing accepted geometry."""
import json
from pathlib import Path
from fastapi import HTTPException
from .design_dependencies import confirmed_inputs, dependency_digest, file_digest


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
        # A1：坐标帧是 live 生成的**前置条件**。Crown 1.0 的红/绿/蓝截面同样只消费
        # 已确认 Patient Frame，不允许由世界轴猜近远中或咬合方向。
        blockers.append({'stage':'FRAME','message':'请先确认近中/颊侧方向与毫米单位（坐标确认），再生成外形'})
    neighbors=json.loads(d.get('neighbors_json') or '{}');sides={}
    for side in ('mesial','distal'):
        item=neighbors.get(side,{})
        valid=bool(frame and item.get('frame_revision_id')==frame['revision_id'])
        if valid and item.get('confirmation_status')=='CONFIRMED':
            try:
                valid=(scan_reference(d['case_id'],item['source']['kind'])==item['source']
                       and Path(item['path']).is_file() and file_digest(item['path'])==item['region_sha256'])
            except (HTTPException,KeyError,OSError): valid=False
        sides[side]=item.get('confirmation_status','UNKNOWN') if valid else 'UNKNOWN'
    bite=json.loads(d.get('bite_json') or '{}');bite_valid=bool(frame and bite.get('revision_id')
        and bite.get('pose_confirmed') and bite.get('frame_revision_id')==frame['revision_id'] and bite.get('transform'))
    if bite_valid:
        try: bite_valid=all(scan_reference(d['case_id'],s['kind'])==s for s in bite['scans']) and len(bite['scans'])==2
        except (HTTPException,KeyError): bite_valid=False
    if d['support_type']=='abutment' and not bite_valid:
        blockers.append({'stage':'OCCLUSAL','code':'OCCLUSION_REGISTRATION_UNVERIFIED',
                         'message':'基台冠需先确认上下颌咬合配准，才能测量对颌空间并生成初稿'})
    params=json.loads(d['parameters_json'])
    proximal=bool(frame and any(v=='CONFIRMED' for v in sides.values())
        and all(v in ('CONFIRMED','ABSENT') for v in sides.values()) and params.get('proximal_target_mm') is not None)
    occlusal=bool(bite_valid and params.get('occlusal_target_mm') is not None)
    mode='BOTH' if proximal and occlusal else 'PROXIMAL_ONLY' if proximal else 'OCCLUSAL_ONLY' if occlusal else None
    return {'input_digest':dependency_digest(d),'blockers':blockers,'template_allowed':not blockers,
        'support_contract':support_contract.as_report() if support_contract else None,
        'automatic_mode':mode if not blockers else None,'frame_confirmed':bool(frame),'neighbors':sides,
        'bite_confirmed':bite_valid,'proximal_target_confirmed':params.get('proximal_target_mm') is not None,
        'occlusal_target_confirmed':params.get('occlusal_target_mm') is not None,
        # Full red/green/blue baseline is only truthful when both lateral context and antagonist are confirmed.
        'sectional_v1_ready':bool(not blockers and mode=='BOTH'),
        'release_status':'NOT_PRODUCTION_APPROVED'}


def _template_baseline(d, margin, axis):
    """Legacy/partial-context baseline retained for TEMPLATE_ONLY and incomplete AUTO modes."""
    from ..api.routes import initialize_draft_outer
    from ..models import AutoOuterRequest
    return initialize_draft_outer(d,AutoOuterRequest(margin_points=margin,insertion_axis=axis))


def generate_candidate(d,kind):
    from ..api.routes import asset
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

    frame,ref=frame_from(d)
    sectional_report=None
    sectional_artifacts={}

    # Crown 1.0 live path：当红（Prep/内冠约束）、绿（邻牙）和蓝（对颌）证据齐全时，
    # 不再用牙库尺寸作为外形真相。先按患者空间直接生成 section-driven baseline，
    # 再复用 AutoFit 做精细 signed-distance 修正和 W3 intaglio/finalize 预检。
    if kind=='AUTO' and state['automatic_mode']=='BOTH':
        from .sectional_generation import build_sectional_baseline
        try:
            _mesh,sectional_report,sectional_artifacts=build_sectional_baseline(
                d,frame,ref,support,margin)
        except ValueError as exc:
            raise HTTPException(422,{
                'message':'Crown 1.0 红/绿/蓝空间包络无法生成，请复核邻牙、咬合位姿或基台空间',
                'detail':str(exc),'repair_stage':'SECTIONAL_ENVELOPE'}) from exc
        baseline=Path(sectional_artifacts['outer'])
        _,report,artifacts=propose(d,frame,ref,'BOTH',baseline_path=baseline)
        # Preserve both the exact coarse envelope and the final fine-fit candidate.
        artifacts['baseline']=str(baseline)
        artifacts['sectional_baseline']=str(baseline)
        artifacts['sectional_report']=sectional_artifacts['sectional_report']
        report['sectional_v1']=sectional_report
        report['candidate_kind']='AUTO_FIT_CANDIDATE'
        report['generator']='SECTIONAL_RED_GREEN_BLUE_V1+AUTOFIT_FINE'
        report['placement_path']='SECTIONAL_RED_GREEN_BLUE_V1'
        placement={
            'template_source':sectional_report.get('template_source'),
            'template_source_warning':'牙库不再决定 Crown 1.0 的宽度/高度；仅保留弱解剖先验语义',
            'frame_source':'CONFIRMED_FRAME',
            'placement_path':'SECTIONAL_RED_GREEN_BLUE_V1',
        }
    else:
        placement,baseline=_template_baseline(d,margin,axis)
        if kind=='AUTO':
            _,report,artifacts=propose(d,frame,ref,state['automatic_mode'],baseline_path=baseline)
            artifacts['baseline']=str(baseline);report['candidate_kind']='AUTO_FIT_CANDIDATE'
            report['generator']='TEMPLATE_BASELINE+AUTOFIT_PARTIAL_CONTEXT'
        else:
            report={'type':'template_draft','candidate_kind':'TEMPLATE_ONLY_DRAFT','status':'TEMPLATE_REVIEW_REQUIRED',
                'stages':[{'stage':'TEMPLATE','status':'PLACED'},{'stage':'PROXIMAL','status':'NOT_ASSESSED'},
                          {'stage':'OCCLUSAL','status':'NOT_ASSESSED'}],
                'w3_preflight':{'status':'INPUT_CHECKED','reason':'预备体输入检查通过；内冠与壁厚仍需最终构建检查'}}
            artifacts={'outer':str(baseline),'baseline':str(baseline)}
            report['generator']='TEMPLATE_ONLY_DRAFT'

    report.update(eligibility=state,input_diagnostics=diagnostics,
        completed_stages=[s['stage'] for s in report['stages'] if s['status'] in ('PLACED','PASS_ENGINEERING')],
        blocked_stages=[s for s in report['stages'] if s['status'] in ('NOT_ASSESSED','INPUT_UNVERIFIED','BLOCKED')],
        review_required_stages=['CANDIDATE','FINAL_QC'],manufacturing_status='NOT_PRODUCTION_APPROVED',
        candidate_outer_sha256=file_digest(artifacts['outer']),
        artifact_sha256={k:file_digest(v) for k,v in artifacts.items()})
    # Candidate report must self-prove which geometry truth source was consumed.
    report['template_source']=placement.get('template_source')
    report['template_source_warning']=placement.get('template_source_warning')
    report['frame_source']=placement.get('frame_source')
    report['patient_frame']={'revision_id':(json.loads(d.get('autofit_frame_json') or 'null') or {}).get('revision_id'),
                             'source':'CONFIRMED'}
    report['placement_path']=placement.get('placement_path')
    report['baseline_sha256']=file_digest(artifacts['baseline'])
    # B：放置路径给出**同一有符号口径**的接触快照（未确认的输入只标未评估，绝不猜符号）。
    # 快照不是门禁：失败时如实记下原因，而不是让候选生成整个失败。
    from .contact_measurement import measure_confirmed_inputs, serializable_measurement
    from fastapi import HTTPException as _HTTPException
    parameters=json.loads(d['parameters_json'])
    try:
        contact=measure_confirmed_inputs(d,json.loads(d.get('autofit_frame_json') or 'null') or {},
            artifacts['baseline'],target_proximal=parameters.get('proximal_target_mm'),
            target_occlusal=parameters.get('occlusal_target_mm'),strict=False)
        report['placement_contact']=serializable_measurement(contact)
    except (_HTTPException,ValueError,OSError) as exc:
        report['placement_contact']={'status':'NOT_ASSESSED','reason':'PLACEMENT_SNAPSHOT_FAILED: %s'%exc}
    return report,artifacts
