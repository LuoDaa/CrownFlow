"""Manual-first 1.0 tools. Every mesh edit is an immutable outer revision."""
from pathlib import Path
import json
from threading import RLock
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError
from ..models import DesignCreate
from .routes import design_row, asset, now, VALID_FDI
from ..config import CASE_DIR,DATA_DIR,TEMPLATE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh, mesh_info
from ..geometry.margin import fit_margin_plane
from ..revisions import DesignRevision, RevisionKind, append_revision, revision_chain

router = APIRouter(prefix='/api')
# ponytail: one process-wide edit lock; use per-design locks if simultaneous editing matters.
_edit_lock=RLock()

class Profile(BaseModel):
    material: str = Field(min_length=1,max_length=120)
    bite_confirmed: bool
    coverage: str = Field(pattern='^(FULL_ARCH|LOCAL_VERIFIED|ENGINEERING_ONLY)$')
    confirmed: bool
    parameters: dict

@router.put('/designs/{design_id}/profile')
def set_profile(design_id:str,p:Profile):
    d=design_row(design_id)
    if not p.confirmed or not p.bite_confirmed:
        raise HTTPException(400,'请确认设计参数和上下颌相对位姿')
    if p.coverage!='ENGINEERING_ONLY' and (not asset(d['case_id'],'upper') or not asset(d['case_id'],'lower')):
        raise HTTPException(400,'正式输入确认需要上下颌扫描')
    unknown=set(p.parameters)-set(DesignCreate.model_fields)
    if unknown: raise HTTPException(400,f'尚未接入参数：{sorted(unknown)}')
    try:
        values=DesignCreate(**{**json.loads(d['parameters_json']),**p.parameters,
            'case_id':d['case_id'],'target_fdi':d['target_fdi'],'support_type':d['support_type'],'generator_mode':d['generator_mode'],
            'indication':d.get('indication') or ('CROWN_ON_EXISTING_ABUTMENT' if d['support_type']=='abutment' else 'NATURAL_PREP')}).model_dump()
    except ValidationError as exc:
        raise HTTPException(422,str(exc)) from exc
    if values['machine_axes'] not in (3,5): raise HTTPException(400,'请选择加工轴数')
    if values['border_horizontal_mm']:
        border_height=values['border_angled_mm']*np.cos(np.deg2rad(values['border_angle_deg']))+values['border_vertical_mm']
        if border_height>=values['margin_band_mm']-1e-5:raise HTTPException(422,'冠边缘截面高度必须小于颈部过渡高度')
    elif values['enforce_minimum_thickness'] or values['border_angled_mm'] or values['border_vertical_mm']:
        raise HTTPException(422,'补厚或边缘截面需要非零水平边缘宽度')
    profile=p.model_dump(exclude={'parameters'});profile['status']='CONFIRMED'
    profile['material_validation']='USER_CONFIRMED_LABEL_NOT_VALIDATED_PROCESS'
    profile['order_source']=json.loads(d.get('order_source_json') or 'null')
    profile['bite_geometry_status']='NOT_RUN'
    encoded=json.dumps(values,sort_keys=True);profile_json=json.dumps(profile,sort_keys=True,ensure_ascii=False)
    if encoded==d['parameters_json'] and profile_json==d.get('profile_json') and d.get('current_profile_revision_id'): return design_row(design_id)
    append_revision(DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.PROFILE,payload=profile,parameter_set=values),make_current=True)
    return design_row(design_id)

@router.get('/designs/{design_id}/status')
def design_status(design_id:str):
    from ..services.design_dependencies import require_current_final, require_current_outer, dependency_digest
    d=design_row(design_id)
    state='NOT_READY'
    if d.get('profile_json'): state='READY'
    if d.get('outer_path'): state='REVIEW_REQUIRED'
    current=False
    if d.get('final_path'):
        try: require_current_final(d);current=True
        except HTTPException: state='STALE'
    outer_current=False
    if d.get('outer_path'):
        try: require_current_outer(d);outer_current=True
        except HTTPException: pass
    from ..services.design_dependencies import confirmed_inputs
    blockers=[];actions=[]
    if not asset(d['case_id'],'support'): blockers.append({'code':'PREP_MISSING','message':'请确认目标基牙并上传或提取独立区域'})
    if not d.get('confirmed_margin_revision_id'):
        blockers.append({'code':'MARGIN_UNCONFIRMED','message':'请在第 3 步点击“确认当前 Margin”'})
    if not d.get('current_axis_revision_id'):
        blockers.append({'code':'AXIS_UNCONFIRMED','message':'请在第 3 步确认插入轴'})
    if not d.get('current_profile_revision_id'): blockers.append({'code':'PROFILE_UNCONFIRMED','message':'请确认设计参数'})
    try: confirmed_inputs(d);inputs_confirmed=True
    except HTTPException:
        inputs_confirmed=False
        if not blockers: blockers.append({'code':'CONFIRMATION_STALE','message':'当前数据与确认修订不一致，请重新确认'})
    if inputs_confirmed and d.get('generator_mode')=='template':
        # A1：模板放置的横向轴只能来自确认坐标帧，没确认就不该显示"可以生成"
        try:
            from .auto_fit import frame_from
            frame_from(d)
        except HTTPException:
            blockers.append({'code':'FRAME_UNCONFIRMED','message':'请确认近中/颊侧方向与毫米单位（坐标确认）后再生成外形'})
    try:
        from .auto_fit import frame_from
        frame_from(d)
        frame_confirmed=True
    except (HTTPException, KeyError, ValueError, OSError):
        frame_confirmed=False
    if not blockers: actions.append('generate_outer')
    if outer_current and not blockers: actions.extend(['edit_outer','finalize'])
    if current: actions.append('engineering_export')
    return {'design_id':design_id,'state':state,'final_current':current,'outer_current':outer_current,
        'design_revision':d['design_revision'],'input_digest':dependency_digest(d),
        'confirmed_revisions':{'margin_revision_id':d.get('confirmed_margin_revision_id'),
            'axis_revision_id':d.get('current_axis_revision_id'),'profile_revision_id':d.get('current_profile_revision_id')},
        'frame_confirmed':frame_confirmed,
        'bite':json.loads(d.get('bite_json') or 'null'),'neighbors':json.loads(d.get('neighbors_json') or '{}'),
        'profile':json.loads(d.get('profile_json') or 'null'),
        'blockers':blockers,'next_actions':actions,
        'release_status':'NOT_PRODUCTION_APPROVED','unsupported':['automatic_bite_registration','dynamic_occlusion','verified_material_process','certified_manufacturing_qc']}

@router.get('/templates/candidates/{fdi}/file')
def template_candidate(fdi:int):
    from fastapi.responses import FileResponse
    if fdi not in VALID_FDI: raise HTTPException(400,'无效 FDI')
    path=DATA_DIR/'template_studio'/'candidates'/f'FDI_{fdi}'/'crown_template.stl'
    if not path.is_file(): raise HTTPException(404,'此牙位没有候选冠模板，请上传牙冠 STL')
    return FileResponse(path,media_type='model/stl')

class ApproveTemplate(BaseModel):
    confirmed:bool

@router.post('/templates/candidates/{fdi}/approve')
def approve_template(fdi:int,p:ApproveTemplate):
    if not p.confirmed or fdi not in VALID_FDI: raise HTTPException(400,'需确认候选冠形及切口')
    directory=DATA_DIR/'template_studio'/'candidates'/f'FDI_{fdi}'
    if not (directory/'crown_template.stl').is_file(): raise HTTPException(404,'候选不存在')
    mesh=load_mesh(directory/'crown_template.stl')
    frame=json.loads((directory/'local_frame.json').read_text(encoding='utf-8'))
    x=np.asarray(frame['mesial']);z=np.asarray(frame['occlusal']);y=np.cross(z,x)
    basis=np.column_stack((x,y,z))
    v=np.asarray(mesh.vertices)@basis
    v-=np.array([v[:,0].mean(),v[:,1].mean(),v[:,2].min()]);mesh.vertices=v
    if not mesh.is_watertight: raise HTTPException(400,'模板不水密')
    path=TEMPLATE_DIR/f'FDI_{fdi}.stl';save_mesh(mesh,path)
    review=json.loads((directory/'review.json').read_text(encoding='utf-8'))
    review.update(designer_confirmed=True,confirmed_at=now(),normalization='mesial_cross_occlusal_positive_z')
    path.with_suffix('.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'fdi':fdi,'mesh':mesh_info(mesh),'source':'designer_confirmed_candidate'}

class Edit(BaseModel):
    operation: str
    translation: list[float] = Field(default_factory=lambda:[0,0,0])
    rotation: list[float] = Field(default_factory=lambda:[0,0,0])
    scale: list[float] = Field(default_factory=lambda:[1,1,1])
    centers: list[list[float]] = Field(default_factory=list, max_length=100)
    radius_mm: float = Field(default=1.5, ge=.2, le=8)
    strength_mm: float = Field(default=.05, ge=.001, le=.3)
    base_outer_path: str
    frame_angle_deg: float = Field(default=0,ge=-180,le=180)
    falloff: float = Field(default=2,ge=1,le=4)
    vertex_weights: list[float] | None = Field(default=None,max_length=60000)
    region_masks: list[str] | None = Field(default=None,max_length=60000)
    semantic_region: str | None = Field(default=None,max_length=80)
    source_session_sha256: str | None = None
    expected_input_digest: str | None = None
    target_distance_mm: float | None = Field(default=None,ge=0,le=.5)

def edit_mesh(mesh, margin, axis, request, frame=None):
    from ..geometry.sculpt import sculpt_edit
    if frame is None:
        # Legacy geometry callers have no dental semantics; live edits require a confirmed frame.
        from ..geometry.crown_outer.patient_frame import build_patient_frame
        frame=build_patient_frame(np.asarray(margin,float),np.asarray(axis,float))
    return sculpt_edit(mesh,margin,axis,request,frame)[0]

@router.get('/cases/{case_id}/designs')
def list_designs(case_id:str):
    with db() as c:
        return [dict(r) for r in c.execute('SELECT * FROM designs WHERE case_id=? ORDER BY updated_at DESC',(case_id,)).fetchall()]

@router.post('/designs/{design_id}/edit-outer')
def edit_outer(design_id:str,p:Edit):
    with _edit_lock:
        return _edit_outer(design_id,p)

def _edit_outer(design_id:str,p:Edit):
    d=design_row(design_id)
    from ..services.design_dependencies import require_current_outer, dependency_digest, publish_artifacts, file_digest
    require_current_outer(d)
    expected=dependency_digest(d)
    if not d['outer_path']: raise HTTPException(400,'请先生成外形')
    if p.base_outer_path!=d['outer_path']: raise HTTPException(409,'外形已更新，请重新载入后编辑')
    from ..services.sculpt_session import read_session
    mesh,frame,ref,state=read_session(d,allow_legacy=True)
    if p.source_session_sha256 and p.source_session_sha256!=file_digest(d['outer_path']):
        raise HTTPException(409,'编辑会话已过期，请重新载入')
    if p.expected_input_digest and p.expected_input_digest!=expected:
        raise HTTPException(409,'编辑输入已改变，预览没有保存')
    try:
        from ..services.sculpt_commit import commit_edit
        return commit_edit(d,p,mesh,frame,ref,state,expected)
    except ValueError as e:
        from ..geometry.sculpt import SculptEditRejected
        if isinstance(e,SculptEditRejected):
            raise HTTPException(422,detail=e.report) from e
        raise HTTPException(422,str(e)) from e


class Restore(BaseModel):
    revision_no: int

@router.post('/designs/{design_id}/restore-outer')
def restore_outer(design_id:str,p:Restore):
    with _edit_lock:
        return _restore_outer(design_id,p)

def _restore_outer(design_id:str,p:Restore):
    d=design_row(design_id)
    from ..services.design_dependencies import dependency_digest,publish_artifacts,file_digest
    expected=dependency_digest(d)
    candidates=revision_chain(design_id)['revisions']
    r=next((r for r in candidates if r['revision_no']==p.revision_no and r['kind']=='outer'),None)
    path=(r or {}).get('artifacts',{}).get('outer')
    if not path or not Path(path).is_file(): raise HTTPException(404,'此修订没有可恢复的外形')
    metadata=r.get('payload',{})
    if metadata.get('outer_dependency_digest')!=dependency_digest(d,stage='outer') or metadata.get('output_sha256')!=file_digest(path):
        raise HTTPException(409,'历史外冠与当前基牙、Margin 或轴版本不一致，不能恢复为当前结果')
    updates={'outer_path':path,'final_path':None,'cutter_path':None,'qc_json':None,'status':'outer_ready','updated_at':now(),'sculpt_state_json':None}
    state_path=r.get('artifacts',{}).get('sculpt_state')
    if state_path:
        if file_digest(state_path)!=metadata.get('sculpt_state_sha256'):raise HTTPException(409,'历史语义文件已改变，不能恢复')
        updates['sculpt_state_json']=Path(state_path).read_text(encoding='utf-8')
    # A1 原子修订：恢复历史外冠与修订登记同事务，避免"当前几何已换、链上未记"。
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER,
        payload={'type':'restore_outer','restored_revision':p.revision_no,'sculpt_state_sha256':metadata.get('sculpt_state_sha256'),
                 'outer_dependency_digest':metadata['outer_dependency_digest'],'output_sha256':metadata['output_sha256']},
        artifacts={'outer':path,**({'sculpt_state':state_path} if state_path else {})},algorithm_version='welooai/1.0')
    publish_artifacts(d,expected,updates,stage='outer',revision=revision)
    return {'outer_path':path,'revision_no':revision.revision_no,'final_invalidated':True}

def _legacy_reference(d,kind):
    """符号不可得时的**幅值参考面**（旧口径），返回 ``(mesh, reason)``。

    - ``antagonist``：对颌扫描原样（旧行为：不施加咬合变换，位姿本身未确认）；
    - ``neighbors``：旧的球面 ROI 选区拼接（无外向法向 ⇒ 无法定符号）。
    """
    from .design_inputs import neighbor_file
    if kind=='antagonist':
        source='upper' if d['target_fdi']>=31 else 'lower'
        a=asset(d['case_id'],source)
        if not a: raise HTTPException(400,'缺少参考颌模型')
        return load_mesh(Path(a['path'])),'OCCLUSION_REFERENCE_POSE_NOT_APPLIED'
    import trimesh
    from ..services.design_dependencies import file_digest
    choices=json.loads(d.get('neighbors_json') or '{}')
    usable=[s for s in ('mesial','distal')
            if choices.get(s,{}).get('status') in ('SELECTED','CONFIRMED') and choices[s].get('path')]
    if not usable:
        if all(choices.get(s,{}).get('status')=='ABSENT' for s in ('mesial','distal')):
            raise HTTPException(409,'两侧均为 ABSENT，邻牙距离分析不适用')
        raise HTTPException(409,'请分别选择真实近中、远中邻牙区域，或标记 ABSENT')
    regions=[]
    for side in usable:
        path=Path(choices[side]['path'])
        if choices[side]['status']=='SELECTED':
            neighbor_file(d['id'],side)                  # 旧的球面 ROI：来源与区域字节校验
        elif not path.is_file() or file_digest(path)!=choices[side].get('region_sha256'):
            raise HTTPException(409,'邻牙（%s）区域已变化，请重新确认'%side)
        regions.append(load_mesh(path))
    reason='NEIGHBOR_UNVERIFIED' if any(choices[s].get('confirmation_status')!='CONFIRMED' for s in usable) \
        else 'REFERENCE_OUTWARD_NORMAL_UNCONFIRMED'
    return trimesh.util.concatenate(regions),reason


@router.get('/designs/{design_id}/outer-analysis')
def outer_analysis(design_id:str,kind:str='antagonist'):
    d=design_row(design_id)
    from ..services.design_dependencies import require_current_outer, require_current_final, dependency_digest
    from ..services.contact_measurement import (measure_confirmed_inputs, per_vertex_contact, combine_per_vertex,
        unsigned_reference_distance, serializable_measurement, IMPLEMENTATION)
    from trimesh.proximity import closest_point
    require_current_outer(d)
    expected=dependency_digest(d)
    if not d['outer_path']: raise HTTPException(400,'请先生成外形')
    # **数据流修正**：有 rebuild-cervical/finalize 产出的最终冠时，接触/邻牙/咬合/距离分析
    # 一律基于 **final 网格**（颈缘已贴合 margin 的冠体），而不是 outer 模板壳。
    final_path=d.get('final_path')
    analysis_mesh_source='final' if final_path and Path(final_path).is_file() else 'outer'
    source_path=final_path if analysis_mesh_source=='final' else d['outer_path']
    mesh=load_mesh(Path(source_path));vertices=np.asarray(mesh.vertices)
    if kind=='thickness':
        require_current_final(d)
        report=json.loads(d.get('qc_json') or '{}')
        audit_path=report.get('artifact_paths',{}).get('manufacturing_qc')
        if audit_path:
            data=json.loads(Path(audit_path).read_text(encoding='utf-8'))['thickness_heatmap']
            if dependency_digest(design_row(design_id))!=expected:
                raise HTTPException(409,'读取期间设计已变化，请重查')
            return {**data,'input_dependency_digest':expected,'result_status':'REVIEW_REQUIRED'}
        if not d['cutter_path']: raise HTTPException(400,'修改后需先重建内冠才能检查内外表面距离')
        reference=load_mesh(Path(d['cutter_path']))
        values=np.concatenate([closest_point(reference,vertices[i:i+1024])[1] for i in range(0,len(vertices),1024)])
        result={'vertices':vertices.tolist(),'faces':mesh.faces.tolist(),'distance_mm':values.tolist(),
            'min_mm':float(values.min()),'p05_mm':float(np.quantile(values,.05)),
            'method':'unsigned_exact_triangle_distance','sign_method':'UNSIGNED_ONLY',
            'note':'无符号表面距离；内外表面距离不等于完整壁厚，邻牙区域及咬合配准须人工确认。'}
    elif kind in ('antagonist','neighbors'):
        # B：与候选审核/放置/最终 QC 同一有符号口径（`services.contact_measurement`）。
        # 以前这里是无符号 `closest_point`；现在只在**符号真的不可得**时退回幅值，并写明机器可读原因。
        parameters=json.loads(d.get('parameters_json') or '{}')
        frame_ref=json.loads(d.get('autofit_frame_json') or 'null') or {}
        measured=(measure_confirmed_inputs(d,frame_ref,d['outer_path'],
            target_proximal=parameters.get('proximal_target_mm') if kind=='neighbors' else None,
            target_occlusal=parameters.get('occlusal_target_mm') if kind=='antagonist' else None,
            strict=False) if frame_ref.get('z') else
            {'status':'NOT_ASSESSED','reason':'FRAME_UNCONFIRMED','surfaces':{},'domains':{},'skipped':{},
             'implementation':IMPLEMENTATION})
        if measured['status']=='MEASURED':
            items={key:per_vertex_contact(surface,vertices)   # 全顶点（前端按 index 上色）
                   for key,surface in measured['surfaces'].items()}
            combined=combine_per_vertex(items)
            result={**serializable_measurement(measured),'vertices':vertices.tolist(),'faces':mesh.faces.tolist(),
                'signed_mm':combined['signed_mm'],'distance_mm':combined['distance_mm'],'valid':combined['valid'],
                'per_side_signed':{k:{'signed_mm':v['signed_mm'],'valid':v['valid']} for k,v in combined['per_side'].items()},
                'min_mm':float(min(combined['signed_mm'])),'p05_mm':float(np.quantile(combined['signed_mm'],.05)),
                'method':'signed_clearance_contact_measurement','combine_rule':combined['combine_rule'],
                'note':'有符号 clearance：正 = 间隙，负 = 穿进参考面；幅值见 distance_mm。'}
        else:
            reason=measured.get('reason') or next((v.get('reason') for v in measured.get('skipped',{}).values()
                                                   if v.get('reason')),'SIGNED_MEASUREMENT_UNAVAILABLE')
            reference,legacy_reason=_legacy_reference(d,kind)
            result={**unsigned_reference_distance(vertices,reference,reason=reason),
                'vertices':vertices.tolist(),'faces':mesh.faces.tolist(),
                'legacy_reference_reason':legacy_reason,
                'note':'无符号表面距离（符号不可得：%s）；内外表面距离不等于完整壁厚，邻牙区域及咬合配准须人工确认。'%reason}
    else: raise HTTPException(400,'未知检查类型')
    if dependency_digest(design_row(design_id))!=expected:
        raise HTTPException(409,'分析期间设计已变化，请重新计算')
    return {**result,'kind':kind,'input_dependency_digest':expected,'result_status':'REVIEW_REQUIRED',
        'analysis_mesh_source':analysis_mesh_source,'vertex_count':int(len(vertices)),
        'scope_status':'REVIEW_REQUIRED' if kind=='neighbors' else 'DESIGNER_CONFIRM_REFERENCE'}

@router.get('/designs/{design_id}/prep-surface-contract')
def prep_surface_contract(design_id:str):
    """只读自检：当前设计的预备体资产是否满足 crown-bottom chain 的输入契约（P1）。

    不改设计、不落库、不做任何几何修复；``readiness`` 只有三档
    （``PREP_SURFACE_READY`` / ``REVIEW_REQUIRED`` / ``UNSUPPORTED_INDICATION``），
    另外如实报 ``NO_SUPPORT_ASSET``（还没定位/上传目标基牙）。
    """
    from ..services.prep_conditioning import assess_design_prep
    d=design_row(design_id)
    support=asset(d['case_id'],'support')
    row=None;path=None
    if support:
        with db() as conn: row=dict(conn.execute('SELECT * FROM assets WHERE id=?',(support['id'],)).fetchone())
        path=support['path']
    contract=assess_design_prep(d,support_path=path,asset_row=row)
    return {**contract.as_report(),'release_status':'NOT_PRODUCTION_APPROVED',
            'note':'本契约只回答“能否进入内冠链”，不证明几何质量；不做任何修复动作'}

@router.get('/designs/{design_id}/generation-profile')
def generation_profile(design_id:str):
    """只读：参数驱动的生成 profile（每个值的来源 + 缺哪些 + 约束求解计划）。

    这是"用户填参数 → 系统自动适配"那条链的**唯一参数视图**：
    - 每个字段带 provenance（USER / DERIVED / DEFAULT_POLICY / **MISSING**），
      MISSING 一律保持 None，绝不当 0 用；
    - ``missing`` 明确列出"还没人给值"的参数（前端要拿它做输入校验，而不是自己猜默认）；
    - ``plan`` 是 `CrownConstraintOptimizer` 的**阶段顺序**（只读计划，不跑几何）；
    - 不落库、不改设计、不触发任何求解。
    """
    from ..services.crown_generation_profile import from_design
    from ..services.crown_constraint_optimizer import MAX_ITER, STAGES
    d=design_row(design_id)
    profile=from_design(d)
    area_target,area_source=profile.effective_contact_area_target_mm2()
    return {
        'profile':profile.as_report(),
        'missing':profile.missing(),
        'contact_area_target_mm2':area_target,
        'contact_area_target_source':area_source,
        'plan':{'strategy':'staged_iterative_solver','stage_order':list(STAGES),'max_iter':MAX_ITER,
                'intaglio_stage_in_optimizer':False,
                'note':'阶段顺序固定：近远中 → 静态咬合（需已确认咬合位姿）→ 最小厚度 → 复测；内冠随后按固定顺序构建'},
        'release_status':'NOT_PRODUCTION_APPROVED',
        'note':'只读参数视图：不代表任何几何已通过检查',
    }

class SupportConfirmation(BaseModel):
    expected_input_digest:str
    confirmed_loop_ids:list[str]=Field(default_factory=list,max_length=8)
    seed:list[float]|None=Field(default=None,min_length=3,max_length=3)

@router.post('/designs/{design_id}/abutment-support/confirm')
def confirm_abutment_support(design_id:str,p:SupportConfirmation):
    from ..services.abutment_support import prepare_design_support,indication_of
    from ..services.design_dependencies import dependency_digest,publish_artifacts
    d=design_row(design_id)
    if indication_of(d)!='CROWN_ON_EXISTING_ABUTMENT':raise HTTPException(422,{'status':'UNSUPPORTED_INDICATION','rejection_code':'CUSTOM_ABUTMENT_NOT_SUPPORTED'})
    if dependency_digest(d)!=p.expected_input_digest:raise HTTPException(409,'支撑或确认版本已修改，请重新检查')
    contract,prepared=prepare_design_support(d,require_ready=False)
    candidates={entry['loop_id'] for entry in contract.diagnostics.get('loops',[]) if entry['classification']=='SCREW_CHANNEL_CANDIDATE'}
    if not set(p.confirmed_loop_ids)<=candidates:raise HTTPException(422,'确认引用的孔不是当前螺丝孔候选')
    value={'basis':contract.provenance['confirmation_basis'],'confirmed_loop_ids':sorted(set(p.confirmed_loop_ids)),'seed':p.seed}
    candidate={**d,'support_conditioning_json':json.dumps(value)}
    verified,prepared=prepare_design_support(candidate)
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.SUPPORT_CONDITIONING,
        payload={**value,'support_contract':verified.as_report()},algorithm_version=prepared.report['algorithm_version'],
        operator_actions=[{'action':'CONFIRM_SCREW_CHANNEL','loop_ids':value['confirmed_loop_ids']}])
    value['revision_id']=revision.id
    publish_artifacts(d,p.expected_input_digest,{'support_conditioning_json':json.dumps(value),'final_path':None,
        'cutter_path':None,'qc_json':None,'final_dependency_digest':None,'updated_at':now()},revision=revision)
    return {'revision_id':revision.id,'support_contract':verified.as_report()}


@router.get('/designs/{design_id}/manufacturing-qc')
def manufacturing_qc(design_id:str):
    from ..services.design_dependencies import require_current_final
    d=design_row(design_id);require_current_final(d)
    report=json.loads(d['qc_json'])
    path=report.get('artifact_paths',{}).get('manufacturing_qc')
    if not path: raise HTTPException(409,'当前结果没有 W3 制造检查，请重新生成')
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    current=design_row(design_id);require_current_final(current)
    if current['design_revision']!=d['design_revision']:
        raise HTTPException(409,'设计已修改，请重新读取当前 QC')
    return result

class RoughMargin(BaseModel):
    seeds: list[list[float]] = Field(min_length=12,max_length=64)
    center_hint: list[float] | None = None
    guide_axis: list[float] | None = None

class DraftPreparation(BaseModel):
    center:list[float]|None=None
    axis:list[float]|None=None
    radius_mm:float=Field(default=7,ge=2,le=12)
    restorative_abutment_confirmed:bool=False


def _draft_implant_preparation(d,p,scan,source):
    from .routes import save_support_mesh
    from ..geometry.implant_candidate import implant_candidates,uploaded_support_candidate
    from ..geometry.abutment_margin import extract_abutment_margin, ALGORITHM_VERSION
    from ..geometry.prep_margin_diagnostics import diagnose
    from ..revisions import record_auto_margin_candidate

    jaw=load_mesh(Path(scan['path']))
    opposite=asset(d['case_id'],'upper' if source=='lower' else 'lower')
    opposing=load_mesh(Path(opposite['path'])) if opposite else None
    candidates=implant_candidates(jaw,opposing)
    prior_support=asset(d['case_id'],'support')
    supplied=prior_support if prior_support and not prior_support['original_name'].startswith('extracted_') else None
    if not candidates and supplied:
        candidates=[uploaded_support_candidate(load_mesh(Path(supplied['path'])),jaw,opposing)]
    if not candidates:
        raise HTTPException(422,'未可靠识别独立种植组件；请上传最终修复基台 STL，或人工定位预备体')
    if p.center is None:
        if len(candidates)!=1:
            raise HTTPException(422,'发现多个种植组件候选，请先确认目标牙位与候选位置')
        component,candidate=candidates[0]
    else:
        center=np.asarray(p.center,float)
        if center.shape!=(3,) or not np.isfinite(center).all():
            raise HTTPException(422,'无效的候选中心')
        component,candidate=min(candidates,key=lambda item:np.linalg.norm(center-item[1]['center']))
        if np.linalg.norm(center-candidate['center'])>max(5,2*candidate['radius_mm']):
            raise HTTPException(422,'点击位置与种植组件候选不匹配，请确认目标牙位')
    axis=candidate['axis']
    if supplied:
        support=load_mesh(Path(supplied['path']))
        if np.linalg.norm(np.asarray(support.centroid)-candidate['center'])>8:
            raise HTTPException(422,'上传基台与扫描候选不在同一位置，请检查坐标系')
        _, supplied_geometry=uploaded_support_candidate(support,jaw,opposing)
        axis=supplied_geometry['axis']
    else:
        support=component
        reused_existing=False
        if prior_support and prior_support['original_name']=='extracted_implant_component.stl':
            details=json.loads(prior_support['mesh_info_json'] or '{}').get('implant_candidate',{})
            if np.linalg.norm(np.asarray(details.get('center',[float('inf')]*3))-candidate['center'])<0.5:
                support=load_mesh(Path(prior_support['path']))
                reused_existing=True
    response={'implant_candidate':candidate,'points':[],'axis':axis,
              'status':'IMPLANT_COMPONENT_REVIEW_REQUIRED',
              'method':'isolated_implant_component_geometry',
              'note':'仅定位种植组件；尚未确认实际修复基台与修复终止线'}
    if not p.restorative_abutment_confirmed:
        if not supplied and not reused_existing:
            save_support_mesh(d['case_id'],support,'extracted_implant_component.stl',
                              {'implant_candidate':candidate,'restorative_abutment_confirmed':False})
        return response
    if not supplied and candidate['confidence']!='high':
        raise HTTPException(422,'种植组件几何定位置信度不足，不能自动推断修复基台边缘；请上传独立最终基台')
    direction=np.asarray(axis,float)
    origin=supplied_geometry['base_center'] if supplied else candidate['base_center']
    try:
        # Abutment Support First：终止线**就在支撑网格上**求（输入 mesh = support，不是颌扫描），
        # 因此"确认终止线 ↔ 支撑切边界"天然属于同一张三角网格。
        points,quality=extract_abutment_margin(support,direction,origin)
        report,_=diagnose(support,points,axis)
    except ValueError as exc:
        raise HTTPException(422,f'无法自动提取基台肩线：{exc}') from exc
    on_support=margin_on_support_distance(np.asarray(points,float),support)
    if not supplied and not reused_existing:
        save_support_mesh(d['case_id'],support,'extracted_implant_component.stl',
                          {'implant_candidate':candidate,'restorative_abutment_confirmed':True})
        prior_support=asset(d['case_id'],'support')
    if prior_support:
        info=json.loads(prior_support['mesh_info_json'] or '{}')
        info['implant_candidate']=candidate
        info['restorative_abutment_confirmed']=True
        with db() as connection:
            connection.execute('UPDATE assets SET mesh_info_json=? WHERE id=?',
                               (json.dumps(info,ensure_ascii=False),prior_support['id']))
    record_auto_margin_candidate(d['id'],points,algorithm_version=ALGORITHM_VERSION,
        note='ENGINEERING CANDIDATE; restorative_abutment_confirmed=true (user declaration); '
             'nonplanar convex shoulder + cyclic DP; clinical finish line requires review; '
             'support_topology_codes='+','.join(report['codes']))
    return {**response,'points':points,'axis':axis,'status':'DRAFT_REVIEW_REQUIRED',
            'method':ALGORITHM_VERSION,'margin_quality':quality,'support_topology_codes':report['codes'],
            'support_mesh':support_info(support),'margin_on_support':on_support,
            'margin_points_count':int(len(points)),
            'note':'已提取基台肩部闭合线（在支撑网格上求解），请旋转核对后确认；支撑拓扑另在内冠步骤检查'}


def support_info(mesh):
    return {'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces)),
            'watertight':bool(mesh.is_watertight)}


def margin_on_support_distance(points,support):
    """确认终止线到**支撑网格**的距离（mean/p95/max）——p95<0.05 才说明两者同一参考面。"""
    from trimesh.proximity import closest_point
    if not len(points):return {'mean_mm':None,'p95_mm':None,'max_mm':None}
    _,distance,_=closest_point(support,points)
    distance=np.asarray(distance,float)
    return {'mean_mm':float(distance.mean()),'p95_mm':float(np.quantile(distance,.95)),
            'max_mm':float(distance.max()),'sample_count':int(len(distance)),
            'gate_mm':0.05,'passes':bool(np.quantile(distance,.95)<0.05)}

@router.post('/designs/{design_id}/draft-preparation')
def draft_preparation(design_id:str,p:DraftPreparation):
    from .routes import save_extracted_support
    from ..models import RoiRequest
    from ..geometry.draft_margin import draft_margin
    from ..revisions import current_margin,record_auto_margin_candidate
    d=design_row(design_id)
    prior=current_margin(design_id)
    if (d.get('confirmed_margin_revision_id') or d.get('current_axis_revision_id') or d.get('outer_path')
            or (prior and prior.margin_source and prior.margin_source.is_human_authored)):
        raise HTTPException(409,'边缘已人工保存或存在后续设计，不能重新定位基牙')
    with db() as c:
        other=c.execute('SELECT 1 FROM designs WHERE case_id=? AND id<>? AND (margin_json IS NOT NULL OR axis_json IS NOT NULL OR outer_path IS NOT NULL) LIMIT 1',(d['case_id'],design_id)).fetchone()
    if other:raise HTTPException(409,'该病例还有其他设计几何，不能替换共用预备体')
    source='lower' if d['target_fdi']>=31 else 'upper'
    scan=asset(d['case_id'],source)
    if not scan:raise HTTPException(400,'请先上传目标颌')
    if d['support_type']=='abutment':
        return _draft_implant_preparation(d,p,scan,source)
    if p.center is None or p.axis is None:
        raise HTTPException(422,'请在基牙顶部选择中心与方向')
    center=np.asarray(p.center,float);direction=np.asarray(p.axis,float)
    if center.shape!=(3,) or direction.shape!=(3,) or not np.isfinite([center,direction]).all() or np.linalg.norm(direction)<1e-8:
        raise HTTPException(422,'请在预备体顶部选择有效定位点')
    jaw=load_mesh(Path(scan['path']))
    nearby=np.linalg.norm(jaw.triangles_center-center,axis=1)<1.5
    if nearby.any():
        averaged=np.average(jaw.face_normals[nearby],axis=0,weights=jaw.area_faces[nearby])
        if np.linalg.norm(averaged)>.2 and np.dot(averaged,direction)>0:direction=averaged/np.linalg.norm(averaged)
    try:points,axis=draft_margin(jaw,p.center,direction,p.radius_mm)
    except ValueError as e:
        if prior is None:save_extracted_support(d['case_id'],RoiRequest(source=source,center=p.center,radius_mm=p.radius_mm))
        raise HTTPException(422,str(e)) from e
    save_extracted_support(d['case_id'],RoiRequest(source=source,center=p.center,radius_mm=p.radius_mm))
    record_auto_margin_candidate(design_id,points,algorithm_version='welooai/draft-section/1.0',
        note='ENGINEERING DRAFT: section 3mm below designer seed, not anatomical margin')
    return {'points':points,'axis':axis,'status':'DRAFT_REVIEW_REQUIRED','method':'seeded_section_3mm','note':'粗边缘线与牙弓方向，待设计师微调'}

@router.post('/designs/{design_id}/rough-margin')
def rough_margin(design_id:str,p:RoughMargin):
    d=design_row(design_id); a=asset(d['case_id'],'support')
    if not a: raise HTTPException(400,'请先选择并提取目标基牙')
    support=load_mesh(Path(a['path']))
    pts=np.asarray(p.seeds,float)
    if pts.shape!=(len(pts),3) or not np.isfinite(pts).all(): raise HTTPException(400,'无效种子')
    try: center,normal=fit_margin_plane(pts)
    except ValueError as e: raise HTTPException(400,str(e))
    _,_,basis=np.linalg.svd(pts-center)
    xy=(pts-center)@basis[:2].T
    a,b=xy,np.roll(xy,-1,axis=0)
    area=abs(np.sum(a[:,0]*b[:,1]-b[:,0]*a[:,1]))/2
    if area<1:raise HTTPException(400,'手绘点围成的范围过小，请绕完整颈缘描绘')
    def side(p,q,r):return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    for i in range(len(a)):
        for j in range(i+2,len(a)):
            if i==0 and j==len(a)-1:continue
            if side(a[i],b[i],a[j])*side(a[i],b[i],b[j])<0 and side(a[j],b[j],a[i])*side(a[j],b[j],b[i])<0:
                raise HTTPException(400,'手绘线交叉，请按顺序沿颈缘绕一整圈')
    if p.guide_axis is not None:
        guide=np.asarray(p.guide_axis,float)
        if guide.shape!=(3,) or not np.isfinite(guide).all() or np.linalg.norm(guide)<1e-8:
            raise HTTPException(400,'无效的牙体方向')
        if abs(np.dot(normal,guide/np.linalg.norm(guide)))<.5:
            raise HTTPException(400,'描绘点集中在牙体一侧；请旋转模型，沿四面颈缘完成一整圈')
    if p.center_hint is not None:
        hint=np.asarray(p.center_hint,float)
        if hint.shape!=(3,) or not np.isfinite(hint).all():raise HTTPException(400,'无效的预备体中心')
        hx,hy=(hint-center)@basis[:2].T
        crossings=(a[:,1]>hy)!=(b[:,1]>hy)
        intersections=a[crossings,0]+(hy-a[crossings,1])*(b[crossings,0]-a[crossings,0])/(b[crossings,1]-a[crossings,1])
        if np.count_nonzero(intersections>hx)%2!=1:
            raise HTTPException(400,'手绘闭环没有围绕目标预备体，请沿完整颈缘重新描绘')
    # Candidate only: interpolate the user-guided loop and project to the actual surface.
    from scipy.interpolate import CubicSpline
    from trimesh.proximity import closest_point
    ring=np.vstack((pts,pts[0])); s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ring,axis=0),axis=1))]
    if np.any(np.diff(s)<.05): raise HTTPException(400,'种子过近或重复')
    curve=CubicSpline(s,ring,bc_type='periodic')(np.linspace(0,s[-1],64,endpoint=False))
    projected,_,_=closest_point(support,curve)
    return {'points':projected.tolist(),'status':'REVIEW_REQUIRED','method':'seeded_periodic_spline_surface_projection'}

class Locate(BaseModel):
    target_fdi:int
    center:list[float]|None=None

@router.post('/cases/{case_id}/traditional-location')
def traditional_location(case_id:str,p:Locate):
    from .routes import case_has_geometry
    if p.center is not None and case_has_geometry(case_id):
        raise HTTPException(409,'已有设计几何，不允许替换目标基牙；请新建病例')
    from ..geometry.tooth_location import locate_target_tooth,extract_support
    from uuid import uuid4
    if p.target_fdi not in VALID_FDI: raise HTTPException(400,'无效 FDI')
    source='lower' if p.target_fdi>=31 else 'upper'
    a=asset(case_id,source)
    if not a: raise HTTPException(400,'请先上传工作颌')
    try:
        jaw=load_mesh(Path(a['path']))
        if p.center is not None:
            seed=np.asarray(p.center,float)
            if seed.shape!=(3,) or not np.isfinite(seed).all(): raise ValueError('无效中心坐标')
        located=locate_target_tooth(jaw,p.target_fdi,user_seed=p.center)
        if p.center is None: return located.as_dict()
        segmented=extract_support(jaw,located,target_fdi=p.target_fdi)
        if segmented.mesh is None: return segmented.as_dict()
        path=CASE_DIR/case_id/f'support_geodesic_{uuid4().hex}.stl';save_mesh(segmented.mesh,path)
        with db() as c:
            c.execute('INSERT INTO assets(id,case_id,kind,path,original_name,mesh_info_json,created_at) VALUES(?,?,?,?,?,?,?)',
                (uuid4().hex,case_id,'support',str(path),'extracted_geodesic.stl',json.dumps(mesh_info(segmented.mesh)),now()))
        return {**segmented.as_dict(),'location':located.as_dict(),'review_required':True}
    except ValueError as e: raise HTTPException(400,str(e))
