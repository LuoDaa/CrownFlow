"""Human-confirmed scan references; no manual action grants geometry PASS."""
from pathlib import Path
from hashlib import sha256
from uuid import uuid4
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from .routes import design_row, asset, now
from ..db import db
from ..config import CASE_DIR
from ..geometry.io import load_mesh, save_mesh
from ..geometry.roi import spherical_roi
from ..revisions import DesignRevision, RevisionKind, append_revision

router=APIRouter(prefix='/api/designs')

def scan_reference(case_id, kind):
    source=asset(case_id,kind)
    if not source: raise HTTPException(400,f'缺少 {kind} 扫描')
    return {'asset_id':source['id'],'kind':kind,'sha256':sha256(Path(source['path']).read_bytes()).hexdigest()}

def save_input(design_id,column,value,expected):
    encoded=json.dumps(value,ensure_ascii=False,sort_keys=True)
    with db() as c:
        updated=c.execute(f"UPDATE designs SET {column}=?,final_path=NULL,cutter_path=NULL,qc_json=NULL,final_dependency_digest=NULL,updated_at=? WHERE id=? AND {column} IS ?",(encoded,now(),design_id,expected))
        if updated.rowcount!=1: raise HTTPException(409,'输入已被其他操作修改，请刷新后重试')
    append_revision(DesignRevision(design_id=design_id,revision_no=0,kind=RevisionKind.PROFILE,payload={column:value}))
    return design_row(design_id)

class Bite(BaseModel):
    viewed:bool
    pose_confirmed:bool

@router.put('/{design_id}/bite')
def set_bite(design_id:str,p:Bite):
    d=design_row(design_id)
    if p.pose_confirmed and not p.viewed: raise HTTPException(400,'先查看两颌，再确认相对位姿')
    scans=[scan_reference(d['case_id'],kind) for kind in ('upper','lower')]
    return save_input(design_id,'bite_json',{'viewed':p.viewed,'pose_confirmed':p.pose_confirmed,
        'geometry_status':'NOT_RUN','method':'MANUAL_REVIEW_NO_REGISTRATION','scans':scans,'updated_at':now()},d.get('bite_json'))

class Neighbor(BaseModel):
    status:str=Field(pattern='^(SELECTED|ABSENT|UNSET)$')
    source:str=Field(default='upper',pattern='^(upper|lower)$')
    center:list[float]|None=None
    radius_mm:float=Field(default=4,ge=1,le=12)

@router.put('/{design_id}/neighbors/{side}')
def set_neighbor(design_id:str,side:str,p:Neighbor):
    if side not in ('mesial','distal'): raise HTTPException(400,'side 必须为 mesial/distal')
    d=design_row(design_id)
    choices=json.loads(d.get('neighbors_json') or '{}')
    value={'status':p.status,'updated_at':now()}
    if p.status=='SELECTED':
        expected='lower' if d['target_fdi']>=31 else 'upper'
        if p.source!=expected: raise HTTPException(400,'邻牙必须从目标牙所在颌选择')
        center=np.asarray(p.center,dtype=float)
        if center.shape!=(3,) or not np.isfinite(center).all(): raise HTTPException(400,'需要有限三维中心坐标')
        ref=scan_reference(d['case_id'],p.source)
        source=asset(d['case_id'],p.source)
        try: region=spherical_roi(load_mesh(Path(source['path'])),center,p.radius_mm)
        except ValueError as exc: raise HTTPException(400,str(exc)) from exc
        path=CASE_DIR/d['case_id']/f'neighbor_{side}_{uuid4().hex}.stl'
        save_mesh(region,path)
        value.update({'source':ref,'center':center.tolist(),'radius_mm':p.radius_mm,'path':str(path),
                      'region_sha256':sha256(path.read_bytes()).hexdigest(),'scope':'HUMAN_SELECTED_SPHERICAL_ROI_NOT_TOOTH_SEGMENTATION'})
    choices[side]=value
    return save_input(design_id,'neighbors_json',choices,d.get('neighbors_json'))

@router.get('/{design_id}/neighbors/{side}/file')
def neighbor_file(design_id:str,side:str):
    from fastapi.responses import FileResponse
    d=design_row(design_id);value=json.loads(d.get('neighbors_json') or '{}').get(side,{})
    if value.get('status')!='SELECTED': raise HTTPException(404,'尚未选择邻牙区域')
    current=scan_reference(d['case_id'],value['source']['kind'])
    if current!=value['source']: raise HTTPException(409,'邻牙来源扫描已变更，请重新选择')
    path=Path(value['path'])
    if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=value['region_sha256']: raise HTTPException(409,'邻牙区域已变更')
    return FileResponse(path,media_type='model/stl')
