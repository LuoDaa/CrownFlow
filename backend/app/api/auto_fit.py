"""Confirmed geometric inputs and reviewable AutoFit proposals."""
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from .routes import design_row, asset, now
from .design_inputs import scan_reference
from ..config import CASE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh
from ..geometry.crown_outer.patient_frame import PatientFrame, validate_patient_frame
from ..revisions import DesignRevision, RevisionKind, append_revision
from ..services.design_dependencies import dependency_digest, file_digest, publish_artifacts, confirmed_inputs

router=APIRouter(prefix='/api/designs')


class StrictInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    expected_input_digest: str
    confirmed: Literal[True]


class FrameInput(StrictInput):
    mesial: list[float]=Field(min_length=3,max_length=3)
    buccal: list[float]=Field(min_length=3,max_length=3)
    units: Literal['mm']


class NeighborInput(StrictInput):
    status: Literal['CONFIRMED','ABSENT','UNKNOWN']
    source: Literal['upper','lower']
    instance_name: str=Field(min_length=1,max_length=100)
    face_ids: list[int]=Field(default_factory=list,max_length=200000)
    outward_normal: list[float]|None=None
    pollution_reviewed: bool=False


class BiteInput(StrictInput):
    transform: list[list[float]]
    source_method: Literal['REGISTERED_EXPORT','MANUAL_REGISTRATION','BITE_SCAN_REGISTRATION']
    units: Literal['mm']


def checked(design_id,digest,natural_only=True):
    d=design_row(design_id)
    if natural_only and d['support_type']!='prep': raise HTTPException(409,'AutoFit 当前只支持天然基牙单冠')
    if dependency_digest(d)!=digest: raise HTTPException(409,'输入版本已变化，请重新载入')
    return d


def save_confirmation(d,column,value,kind,digest):
    revision=DesignRevision(design_id=d['id'],revision_no=0,kind=kind,payload=value)
    value={**value,'revision_id':revision.id};revision.payload=value
    publish_artifacts(d,digest,{column:json.dumps(value,ensure_ascii=False),
        'final_path':None,'cutter_path':None,'qc_json':None,'final_dependency_digest':None,'updated_at':now()},revision=revision)
    return value


def frame_from(d):
    value=json.loads(d.get('autofit_frame_json') or 'null')
    if not value or value['axis_revision_id']!=d['current_axis_revision_id']:
        raise HTTPException(409,'请确认当前轴下的近中/颊侧方向与毫米单位')
    # A1：帧的原点来自边缘线（frame.origin=边缘线质心），所以帧必须绑定当时确认的边缘线版本，
    # 否则"改了 Margin 而帧自动漂移"会让放置看起来仍然合法。
    if value.get('margin_revision_id')!=d.get('confirmed_margin_revision_id'):
        raise HTTPException(409,'边缘线已修改，请重新确认近中/颊侧方向与毫米单位')
    support=asset(d['case_id'],'support')
    if not support or value['prep_sha256']!=file_digest(support['path']): raise HTTPException(409,'基牙已变化，请重确认坐标')
    frame=PatientFrame(np.asarray(json.loads(d['margin_json'])).mean(0),np.asarray(value['z']),np.asarray(value['x']),np.asarray(value['y']),
                       revision_id=value.get('revision_id'))
    validate_patient_frame(frame)     # A0：确认帧必须右手正交，否则拒绝
    return frame,value


@router.put('/{design_id}/auto-fit/frame')
def confirm_frame(design_id:str,p:FrameInput):
    # 坐标帧对天然基牙与种植基台都要用（模板放置的横向轴来自它），
    # 所以这里不套 "AutoFit 只支持天然基牙" 的门；种植病例的扫描杆纪律由 confirmed_inputs 把关。
    d=checked(design_id,p.expected_input_digest,natural_only=False);confirmed_inputs(d,require_profile=False)
    z=np.asarray(json.loads(d['axis_json']),float);z/=np.linalg.norm(z)
    x=np.asarray(p.mesial,float);b=np.asarray(p.buccal,float)
    if not np.isfinite(np.r_[x,b]).all(): raise HTTPException(422,'方向必须为有限数值')
    x=x-z*np.dot(x,z)
    if np.linalg.norm(x)<1e-8: raise HTTPException(422,'近中方向不能与插入轴平行')
    x/=np.linalg.norm(x);y=np.cross(z,x)
    if np.linalg.norm(b)<1e-8 or abs(np.dot(b/np.linalg.norm(b),y))<.5: raise HTTPException(422,'颊側方向与近中方向不独立')
    support=asset(d['case_id'],'support')
    value={'x':x.tolist(),'y':y.tolist(),'z':z.tolist(),'buccal':(y*np.sign(np.dot(b,y))).tolist(),
        'units':'mm','axis_revision_id':d['current_axis_revision_id'],'prep_sha256':file_digest(support['path']),
        'margin_revision_id':d['confirmed_margin_revision_id'],
        'method':'HUMAN_CONFIRMED_ANATOMICAL_DIRECTIONS','authentication_status':'NOT_AUTHENTICATED'}
    return save_confirmation(d,'autofit_frame_json',value,RevisionKind.GEOMETRY_FRAME,p.expected_input_digest)


@router.put('/{design_id}/auto-fit/neighbors/{side}')
def confirm_neighbor(design_id:str,side:str,p:NeighborInput):
    if side not in ('mesial','distal'): raise HTTPException(422,'邻牙侧别无效')
    d=checked(design_id,p.expected_input_digest,natural_only=False);frame,fref=frame_from(d)
    expected='lower' if d['target_fdi']>=31 else 'upper'
    if p.source!=expected: raise HTTPException(422,'邻牙必须位于工作颌')
    value={'status':'ABSENT' if p.status=='ABSENT' else 'UNSET','confirmation_status':p.status,
        'instance_ref':uuid4().hex,'instance_name':p.instance_name,'frame_revision_id':fref['revision_id']}
    if p.status=='CONFIRMED':
        if p.outward_normal is not None and (len(p.outward_normal)!=3 or not np.isfinite(p.outward_normal).all() or np.linalg.norm(p.outward_normal)<1e-8):
            raise HTTPException(422,'外向法向必须是有限的非零三维向量')
        if not p.pollution_reviewed: raise HTTPException(422,'请检查并明确确认区域不含基牙、牙龈及其他牙')
        source=asset(d['case_id'],p.source)
        if not source: raise HTTPException(409,'缺少工作颌扫描')
        mesh=load_mesh(Path(source['path']));ids=np.asarray(sorted(set(p.face_ids)),int)
        if len(ids)<3 or np.any(ids<0) or np.any(ids>=len(mesh.faces)): raise HTTPException(422,'需要有效的原扫描面 ID 集合')
        region=mesh.submesh([ids],append=True,repair=False)
        if len(region.split(only_watertight=False))!=1 or np.any(region.area_faces<1e-10) or not region.is_winding_consistent:
            raise HTTPException(422,'邻牙区域须单连通、朝向一致且没有退化面')
        center=region.vertices.mean(0)
        if (center-frame.origin)@frame.x*(1 if side=='mesial' else -1)<=0: raise HTTPException(422,'区域位于已确认近远中方向的错误侧')
        from ..geometry.manufacturing_qc import _closest
        prep=load_mesh(Path(asset(d['case_id'],'support')['path']))
        points=region.triangles_center
        overlap=_closest(prep,points)[1]<1e-5
        if overlap.any(): raise HTTPException(422,'选区包含与基牙重合的面，请重新选择')
        from ..geometry.crown_outer.auto_fit_adapter import FitSurface
        try: FitSurface(region,p.outward_normal)
        except ValueError as exc: raise HTTPException(422,str(exc)) from exc
        path=CASE_DIR/d['case_id']/f'neighbor_{side}_{uuid4().hex}.stl';save_mesh(region,path)
        value.update(status='SELECTED',source=scan_reference(d['case_id'],p.source),path=str(path),
            region_sha256=file_digest(path),face_ids=ids.tolist(),outward_normal=p.outward_normal,
            pollution_reviewed=True,pollution_audit={'exact_prep_overlap_faces':int(overlap.sum()),
                'other_tissue_classification':'HUMAN_REVIEW_NOT_AUTOMATICALLY_PROVEN'},
            scope='HUMAN_CONFIRMED_FACE_MASK_NOT_FDI_OR_RADIUS_ORACLE')
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.NEIGHBOR_CONFIRMED,payload={'side':side,**value})
    value['revision_id']=revision.id;revision.payload={'side':side,**value}
    choices=json.loads(d.get('neighbors_json') or '{}');choices[side]=value
    publish_artifacts(d,p.expected_input_digest,{'neighbors_json':json.dumps(choices,ensure_ascii=False),
        'final_path':None,'cutter_path':None,'qc_json':None,'final_dependency_digest':None,'updated_at':now()},revision=revision)
    return value


@router.put('/{design_id}/auto-fit/bite')
def confirm_bite(design_id:str,p:BiteInput):
    d=checked(design_id,p.expected_input_digest,natural_only=False);_,fref=frame_from(d)
    transform=np.asarray(p.transform,float)
    if (transform.shape!=(4,4) or not np.isfinite(transform).all() or
        not np.allclose(transform[3],[0,0,0,1],atol=1e-8) or
        not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-6) or
        not np.isclose(np.linalg.det(transform[:3,:3]),1,atol=1e-6)):
        raise HTTPException(422,'咬合变换必须为有限、无镜像的刚体 4x4 矩阵')
    scans=[scan_reference(d['case_id'],kind) for kind in ('upper','lower')]
    value={'viewed':True,'pose_confirmed':True,'transform':transform.tolist(),'scans':scans,'units':'mm',
        'source_method':p.source_method,'frame_revision_id':fref['revision_id'],
        'geometry_status':'MANUAL_REGISTERED_STATIC_ONLY_NOT_ALGORITHM_CERTIFIED'}
    return save_confirmation(d,'bite_json',value,RevisionKind.BITE_CONFIRMED,p.expected_input_digest)


class PreviewInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode: Literal['PROXIMAL_ONLY','OCCLUSAL_ONLY','BOTH']
    expected_input_digest: str
    margin_revision_id: str
    axis_revision_id: str
    profile_revision_id: str
    dry_run: Literal[True]=True


@router.get('/{design_id}/auto-fit/templates')
def template_libraries(design_id:str):
    from ..config import TOOTH_LIBRARY_ROOT
    from ..importers.tooth_library import discover_libraries
    d=design_row(design_id)
    try: libraries=discover_libraries(str(TOOTH_LIBRARY_ROOT))
    except ValueError: return {'libraries':[],'status':'LIBRARY_UNAVAILABLE'}
    entries=[]
    for lib in libraries:
        try:
            path=lib.template_path(d['target_fdi'])
            entries.append({'name':lib.name,'fdi':d['target_fdi'],'source_sha256':file_digest(path),'status':'NORMALIZATION_REQUIRES_CHECK'})
        except ValueError as exc: entries.append({'name':lib.name,'status':'UNAVAILABLE','reason':str(exc)})
    return {'libraries':entries,'status':'REVIEW_REQUIRED'}


class TemplateInput(StrictInput):
    library: str


@router.post('/{design_id}/auto-fit/template-preview')
def template_preview(design_id:str,p:TemplateInput):
    d=checked(design_id,p.expected_input_digest,natural_only=False);consumed=confirmed_inputs(d);frame,fref=frame_from(d)
    from ..config import TOOTH_LIBRARY_ROOT
    from ..importers.tooth_library import discover_libraries
    from ..services.auto_fit_templates import placed_library_crown
    try:
        lib=next((lib for lib in discover_libraries(str(TOOTH_LIBRARY_ROOT)) if lib.name==p.library),None)
        if lib is None: raise ValueError('不存在此牙库')
        source_path=lib.template_path(d['target_fdi'])
        metadata_path=Path(lib.metadata_dir)/('lowerjaw.xml' if d['target_fdi']>=31 else 'upperjaw.xml')
        source_sha=file_digest(source_path);metadata_sha=file_digest(metadata_path)
        mesh,metadata=placed_library_crown(lib,d['target_fdi'],consumed['margin'].payload['points'],frame)
    except ValueError as exc: raise HTTPException(422,{'message':str(exc),'status':'TEMPLATE_NORMALIZATION_BLOCKED'}) from exc
    from ..geometry.crown_outer.thickness import input_qc
    geometry=input_qc(mesh.vertices,mesh.faces)
    if geometry['degenerate_faces'] or geometry['nonmanifold_edges'] or geometry.get('self_intersections_severe',0) or geometry.get('self_intersections_truncated'):
        raise HTTPException(422,{'message':'模板几何检查失败','geometry':geometry})
    path=CASE_DIR/d['case_id']/f'{design_id}_library_{uuid4().hex}.stl';save_mesh(mesh,path)
    if file_digest(source_path)!=source_sha or file_digest(metadata_path)!=metadata_sha:
        raise HTTPException(409,'模板计算期间牙库源文件变化，请重新提案')
    payload={'type':'template_placement','status':'TEMPLATE_REVIEW_REQUIRED','stages':[],
        'input_digest':p.expected_input_digest,'design_revision':d['design_revision'],'template':metadata,
        'source_template_path':source_path,'template_sha256':source_sha,
        'source_metadata_path':str(metadata_path),'metadata_sha256':metadata_sha,
        'candidate_outer_sha256':file_digest(path),'artifact_sha256':{'outer':file_digest(path)},'geometry':geometry,
        'consumed_revisions':{s+'_revision_id':r.id for s,r in consumed.items()},
        'geometry_frame_revision_id':fref['revision_id'],'w3_preflight':{'status':'NOT_ASSESSED','reason':'Template placement only; acceptance does not approve W3 construction'}}
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER_AUTOFIT_CANDIDATE,payload=payload,
        artifacts={'outer':str(path)},algorithm_version='welooai/library-placement/1.0')
    with db() as c:
        c.execute('BEGIN IMMEDIATE');current=dict(c.execute('SELECT * FROM designs WHERE id=?',(design_id,)).fetchone())
        if current['design_revision']!=d['design_revision'] or dependency_digest(current,conn=c)!=p.expected_input_digest:
            raise HTTPException(409,'模板计算期间输入变化，候选不发布')
        from ..revisions.service import _next_revision_no
        revision.revision_no=_next_revision_no(c,design_id);revision.created_at=now();row=revision.to_row()
        c.execute('INSERT INTO design_revisions('+','.join(row)+') VALUES('+','.join(':'+k for k in row)+')',row)
    return revision.as_dict()


@router.get('/{design_id}/auto-fit/inputs')
def inputs(design_id:str):
    d=design_row(design_id)
    return {'frame':json.loads(d.get('autofit_frame_json') or 'null'),
        'neighbors':json.loads(d.get('neighbors_json') or '{}'),'bite':json.loads(d.get('bite_json') or 'null'),
        'input_digest':dependency_digest(d),'release_status':'NOT_PRODUCTION_APPROVED'}


@router.post('/{design_id}/auto-fit/preview')
def preview(design_id:str,p:PreviewInput):
    d=checked(design_id,p.expected_input_digest);consumed=confirmed_inputs(d)
    if any(getattr(p,s+'_revision_id')!=r.id for s,r in consumed.items()): raise HTTPException(409,'确认修订已过期')
    frame,frame_ref=frame_from(d)
    from ..services.auto_fit import propose
    try: mesh,report,artifacts=propose(d,frame,frame_ref,p.mode)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER_AUTOFIT_CANDIDATE,
        payload={**report,'input_digest':p.expected_input_digest,'design_revision':d['design_revision'],
            'consumed_revisions':{s+'_revision_id':r.id for s,r in consumed.items()},
            'baseline_outer_sha256':file_digest(d['outer_path'])},
        artifacts=artifacts,algorithm_version='welooai/autofit/1.0',parameter_set=consumed['profile'].parameter_set)
    # A candidate is stored only while its complete input snapshot is still current.
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        current=dict(c.execute('SELECT * FROM designs WHERE id=?',(design_id,)).fetchone())
        if current['design_revision']!=d['design_revision'] or dependency_digest(current,conn=c)!=p.expected_input_digest:
            raise HTTPException(409,'候选计算期间输入已修改，本轮候选不发布')
        from ..revisions.service import _next_revision_no
        revision.revision_no=_next_revision_no(c,design_id);revision.created_at=now();row=revision.to_row()
        c.execute('INSERT INTO design_revisions('+','.join(row)+') VALUES('+','.join(':'+k for k in row)+')',row)
    return revision.as_dict()


def candidate(design_id,proposal_id):
    with db() as c: row=c.execute('SELECT * FROM design_revisions WHERE design_id=? AND id=?',(design_id,proposal_id)).fetchone()
    if row is None or row['kind']!=RevisionKind.OUTER_AUTOFIT_CANDIDATE.value: raise HTTPException(404,'没有此 AutoFit 候选')
    return DesignRevision.from_row(row)


@router.get('/{design_id}/auto-fit/{proposal_id}/report')
def report(design_id:str,proposal_id:str): return candidate(design_id,proposal_id).as_dict()


@router.get('/{design_id}/auto-fit/{proposal_id}/candidate.stl')
def candidate_file(design_id:str,proposal_id:str):
    r=candidate(design_id,proposal_id);path=r.artifacts.get('outer')
    if not path or not Path(path).is_file() or file_digest(path)!=r.payload.get('candidate_outer_sha256'):
        raise HTTPException(409,'候选文件已变化')
    return FileResponse(path,media_type='model/stl')


@router.get('/{design_id}/auto-fit/{proposal_id}/artifact/{label}')
def proposal_artifact(design_id:str,proposal_id:str,label:str):
    r=candidate(design_id,proposal_id);path=r.artifacts.get(label)
    if not path or not Path(path).is_file() or file_digest(path)!=r.payload.get('artifact_sha256',{}).get(label):
        raise HTTPException(409,'候选诊断产物不存在或已变化')
    return FileResponse(path,media_type='application/json' if Path(path).suffix=='.json' else 'model/stl')


class AcceptInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    expected_input_digest: str
    confirmed: Literal[True]
    confirmed_template_anatomy: bool=False


@router.get('/{design_id}/workflow/eligibility')
def workflow_eligibility(design_id:str):
    from ..services.generation_workflow import eligibility
    return eligibility(design_row(design_id))


class GenerationInput(StrictInput):
    kind: Literal['AUTO','TEMPLATE_ONLY']


@router.post('/{design_id}/workflow/generate')
def workflow_generate(design_id:str,p:GenerationInput):
    d=checked(design_id,p.expected_input_digest,natural_only=False)
    from ..services.generation_workflow import generate_candidate
    consumed=confirmed_inputs(d)
    try: report,artifacts=generate_candidate(d,p.kind)
    except ValueError as exc: raise HTTPException(422,str(exc)) from exc
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER_AUTOFIT_CANDIDATE,
        payload={**report,'input_digest':p.expected_input_digest,'design_revision':d['design_revision'],
            'consumed_revisions':{s+'_revision_id':r.id for s,r in consumed.items()}},
        artifacts=artifacts,algorithm_version='welooai/generation-workflow/1.0')
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        current=dict(c.execute('SELECT * FROM designs WHERE id=?',(design_id,)).fetchone())
        if current['design_revision']!=d['design_revision'] or dependency_digest(current,conn=c)!=p.expected_input_digest:
            raise HTTPException(409,'生成期间输入变化，候选不发布')
        from ..revisions.service import _next_revision_no
        revision.revision_no=_next_revision_no(c,design_id);revision.created_at=now();row=revision.to_row()
        c.execute('INSERT INTO design_revisions('+','.join(row)+') VALUES('+','.join(':'+k for k in row)+')',row)
    return revision.as_dict()


@router.post('/{design_id}/auto-fit/{proposal_id}/accept')
def accept(design_id:str,proposal_id:str,p:AcceptInput):
    d=checked(design_id,p.expected_input_digest,natural_only=False);r=candidate(design_id,proposal_id)
    if r.payload['input_digest']!=p.expected_input_digest or r.payload['design_revision']!=d['design_revision']:
        raise HTTPException(409,'候选已过期，不能采纳')
    if r.payload['status']!='PASS_ENGINEERING' and not (r.payload.get('type') in ('template_placement','template_draft') and p.confirmed_template_anatomy and r.payload['status']=='TEMPLATE_REVIEW_REQUIRED'):
        raise HTTPException(409,'部分完成、冲突或未验证候选不能自动采纳；模板初稿须明确确认冠形')
    with db() as c:
        if c.execute("SELECT 1 FROM design_revisions WHERE design_id=? AND kind=? AND json_extract(payload_json,'$.proposal_id')=?",(design_id,RevisionKind.AUTOFIT_REJECTED.value,proposal_id)).fetchone():
            raise HTTPException(409,'候选已退回')
    path=r.artifacts['outer']
    if not Path(path).is_file() or file_digest(path)!=r.payload['candidate_outer_sha256']: raise HTTPException(409,'候选文件已变化')
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.OUTER,
        payload={'type':'autofit_accepted','proposal_id':r.id,'consumed_revisions':r.payload['consumed_revisions'],
            'outer_dependency_digest':dependency_digest(d,stage='outer'),'output_sha256':file_digest(path)},
        artifacts={'outer':path},algorithm_version='welooai/autofit/1.0')
    updates={'outer_path':path,'final_path':None,'cutter_path':None,
        'qc_json':None,'final_dependency_digest':None,'status':'outer_ready','updated_at':now()}
    if r.payload.get('type')=='template_placement':
        landmarks=json.loads(d.get('landmarks_json') or '{}')
        landmarks['_selected_template']={'source_path':r.payload['source_template_path'],'sha256':r.payload['template_sha256'],
            'metadata_path':r.payload['source_metadata_path'],'metadata_sha256':r.payload['metadata_sha256'],
            'library':r.payload['template']['library'],'proposal_id':r.id,'geometry_frame_revision_id':r.payload['geometry_frame_revision_id']}
        updates['landmarks_json']=json.dumps(landmarks,ensure_ascii=False)
        revision.payload['outer_dependency_digest']=dependency_digest({**d,**updates},stage='outer')
    publish_artifacts(d,p.expected_input_digest,updates,stage='outer',revision=revision)
    return {'outer_path':path,'revision_id':revision.id,'final_invalidated':True,'release_status':'NOT_PRODUCTION_APPROVED'}


@router.post('/{design_id}/auto-fit/{proposal_id}/select-template')
def select_template(design_id:str,proposal_id:str,p:StrictInput):
    d=checked(design_id,p.expected_input_digest,natural_only=False);r=candidate(design_id,proposal_id)
    if (r.payload.get('type')!='template_placement' or r.payload.get('input_digest')!=p.expected_input_digest
            or r.payload.get('design_revision')!=d['design_revision']):
        raise HTTPException(409,'牙型候选已过期，请重新预览')
    _,frame_ref=frame_from(d)
    if r.payload.get('geometry_frame_revision_id')!=frame_ref['revision_id']:
        raise HTTPException(409,'牙型方向已变化，请重新预览')
    landmarks=json.loads(d.get('landmarks_json') or '{}')
    landmarks['_selected_template']={'source_path':r.payload['source_template_path'],'sha256':r.payload['template_sha256'],
        'metadata_path':r.payload['source_metadata_path'],'metadata_sha256':r.payload['metadata_sha256'],
        'library':r.payload['template']['library'],'proposal_id':r.id,'geometry_frame_revision_id':frame_ref['revision_id']}
    landmarks.pop('_body_dependency_digest',None)
    revision=DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.TEMPLATE_SELECTED,
        payload={'type':'template_selected','proposal_id':r.id,'library':r.payload['template']['library']})
    publish_artifacts(d,p.expected_input_digest,{'landmarks_json':json.dumps(landmarks,ensure_ascii=False),
        'outer_path':None,'outer_dependency_digest':None,'outer_output_sha256':None,
        'final_path':None,'cutter_path':None,'final_dependency_digest':None,'qc_json':None,
        'status':'template_selected','updated_at':now()},stage='template',revision=revision)
    return {'library':r.payload['template']['library'],'revision_id':revision.id,'outer_current':False}


@router.post('/{design_id}/auto-fit/{proposal_id}/reject')
def reject(design_id:str,proposal_id:str):
    candidate(design_id,proposal_id)
    r=append_revision(DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.AUTOFIT_REJECTED,payload={'proposal_id':proposal_id}))
    return r.as_dict()


@router.get('/{design_id}/auto-fit/{proposal_id}/analysis')
def candidate_analysis(design_id:str,proposal_id:str,kind:Literal['neighbors','antagonist','thickness']):
    d=design_row(design_id);r=candidate(design_id,proposal_id);expected=dependency_digest(d)
    if r.payload['input_digest']!=expected or r.payload['design_revision']!=d['design_revision']:
        raise HTTPException(409,'候选已过期，请重新生成后检查')
    path=r.artifacts.get('outer')
    if not path or file_digest(path)!=r.payload['candidate_outer_sha256']:
        raise HTTPException(409,'候选文件已变化')
    if kind=='thickness':
        audit=r.artifacts.get('manufacturing_qc')
        if not audit or file_digest(audit)!=r.payload.get('artifact_sha256',{}).get('manufacturing_qc'):
            raise HTTPException(409,'候选尚无有效内冠壁厚检查，请采纳后构建检查')
        result=json.loads(Path(audit).read_text(encoding='utf-8'))['thickness_heatmap']
    else:
        # B：候选审核与放置/AutoFit/最终 QC 共用同一有符号接触口径（审计 #5）。
        # 以前这里是无符号 `closest_point`，页面会说"不证明接触或穿透"，与求解器的 signed 结论矛盾。
        from ..services.contact_measurement import (measure_confirmed_inputs, per_vertex_contact, combine_per_vertex)
        from ..services.generation_workflow import eligibility
        loaded=load_mesh(Path(path));vertices=np.asarray(loaded.vertices)
        state=eligibility(d)
        if not state['frame_confirmed']: raise HTTPException(409,'请先确认近中/颊侧方向与毫米单位')
        frame_ref=json.loads(d['autofit_frame_json']);parameters=json.loads(d['parameters_json'])
        if kind=='neighbors':
            if not all(v in ('CONFIRMED','ABSENT') for v in state['neighbors'].values()):
                raise HTTPException(409,'请先确认当前两侧邻牙或明确缺牙')
            target=parameters.get('proximal_target_mm')
            if target is None: raise HTTPException(409,'尚未确认邻接目标间隙')
            measured=measure_confirmed_inputs(d,frame_ref,path,target_proximal=target)
            if not measured['proximal']: raise HTTPException(409,'两侧缺牙，邻接距离不适用')
            # 热力图覆盖**全部冠顶点**（前端按 index 上色）；统计量取审核接触域
            per_vertex={side:per_vertex_contact(surface,vertices)
                        for side,surface in measured['surfaces'].items()}
            result={'vertices':vertices.tolist(),'faces':loaded.faces.tolist(),
                **combine_per_vertex(per_vertex),'stats':{'per_side':measured['proximal'],
                'scope':'audited_contact_domain'},'coloring_scope':'all_crown_vertices','target_mm':target,
                'measurement_status':measured['status'],'skipped':measured['skipped']}
        else:
            if not state['bite_confirmed']: raise HTTPException(409,'请先确认当前静态咬合位姿')
            target=parameters.get('occlusal_target_mm')
            if target is None: raise HTTPException(409,'尚未确认咬合目标间隙')
            measured=measure_confirmed_inputs(d,frame_ref,path,target_occlusal=target)
            if measured['occlusal'] is None: raise HTTPException(409,'静态咬合尚未确认，无法测咬合接触')
            result={'vertices':vertices.tolist(),'faces':loaded.faces.tolist(),
                **combine_per_vertex({'occlusal':per_vertex_contact(measured['surfaces']['occlusal'],vertices)}),
                'stats':{'occlusal':measured['occlusal'],'scope':'audited_occlusal_domain'},
                'coloring_scope':'all_crown_vertices','target_mm':target,
                'measurement_status':measured['status'],'skipped':measured['skipped']}
        result['note']='有符号 clearance：正 = 与参考面的间隙，负 = 穿进参考面；幅值见 distance_mm（热力图用它）。'
        result['candidate_outer_sha256']=r.payload['candidate_outer_sha256']
        result['implementation']=measured['implementation']
        result['sign_semantics']=measured['sign_semantics']
    if dependency_digest(design_row(design_id))!=expected or file_digest(path)!=r.payload['candidate_outer_sha256']:
        raise HTTPException(409,'检查期间输入或候选已变化，请重查')
    return {**result,'kind':kind,'input_dependency_digest':expected,'result_status':'REVIEW_REQUIRED'}
