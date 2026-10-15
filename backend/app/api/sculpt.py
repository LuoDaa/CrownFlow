"""Sculpt preview is read-only; Apply reuses edit-outer atomic revisions."""
import json
import numpy as np
from fastapi import APIRouter,HTTPException
from pydantic import BaseModel,Field,conlist
from .routes import design_row
from ..services.sculpt_session import session_report,read_session,canonical_mapping,annotate_region,measure_contacts
from ..services.design_dependencies import dependency_digest

router=APIRouter(prefix='/api/designs')


class Preview(BaseModel):
    expected_input_digest:str
    source_session_sha256:str
    vertices:list[conlist(float,min_length=3,max_length=3)]=Field(min_length=3,max_length=60000)
    sample_ids:list[int]|None=Field(default=None,max_length=512)


def checked(design_id,request):
    design=design_row(design_id);mesh,frame,ref,state=read_session(design)
    if request.expected_input_digest!=dependency_digest(design) or request.source_session_sha256!=state['outer_sha256']:
        raise HTTPException(409,'编辑会话已过期；预览不会提交')
    return design,mesh,frame,ref,state


@router.get('/{design_id}/sculpt-session')
def get_session(design_id:str,regions:bool=False):
    """``regions=1`` 时额外下发逐顶点语义权重（前端开发态叠加显示用）。"""
    return session_report(design_row(design_id),include_regions=regions)


@router.post('/{design_id}/sculpt-contact-preview')
def contact_preview(design_id:str,p:Preview):
    d,mesh,frame,ref,state=checked(design_id,p)
    vertices=np.asarray(p.vertices,float)
    if vertices.shape!=mesh.vertices.shape or not np.isfinite(vertices).all():raise HTTPException(422,'预览顶点与当前冠形不对应')
    if np.linalg.norm(vertices-mesh.vertices,axis=1).max()>5:raise HTTPException(422,'预览位移超出工程范围')
    sample=np.asarray(p.sample_ids if p.sample_ids is not None else np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int),int)
    if not len(sample) or np.any(sample<0) or np.any(sample>=len(vertices)):raise HTTPException(422,'采样索引无效')
    mesh.vertices=vertices
    result=measure_contacts(d,mesh,frame,ref,sample_ids=sample)
    if dependency_digest(design_row(design_id))!=p.expected_input_digest:raise HTTPException(409,'预览期间输入改变')
    return {**result,'scope':'PREVIEW_ASYNC_SAMPLED','source_session_sha256':p.source_session_sha256}


class Annotation(BaseModel):
    expected_input_digest:str
    source_session_sha256:str
    region:str
    weights:list[float]=Field(min_length=3,max_length=60000)
    mapping_confirmed:bool=False


@router.post('/{design_id}/sculpt-thickness-preview')
def thickness_preview(design_id:str,p:Preview):
    from ..services.sculpt_thickness import preview_thickness
    d,mesh,frame,ref,state=checked(design_id,p);vertices=np.asarray(p.vertices,float)
    if vertices.shape!=mesh.vertices.shape or not np.isfinite(vertices).all():raise HTTPException(422,'无效预览顶点')
    if np.linalg.norm(vertices-mesh.vertices,axis=1).max()>5:raise HTTPException(422,'预览位移超出工程范围')
    sample=np.asarray(p.sample_ids if p.sample_ids is not None else np.linspace(0,len(vertices)-1,min(512,len(vertices)),dtype=int),int)
    if not len(sample) or np.any(sample<0) or np.any(sample>=len(vertices)):raise HTTPException(422,'采样索引无效')
    mesh.vertices=vertices
    result=preview_thickness(d,mesh,state,sample)
    if dependency_digest(design_row(design_id))!=p.expected_input_digest:raise HTTPException(409,'预览期间输入改变')
    return result


@router.put('/{design_id}/sculpt-anatomy')
def save_anatomy(design_id:str,p:Annotation):
    from .designer import Edit,_edit_lock
    from ..services.sculpt_commit import commit_edit
    with _edit_lock:
        d,mesh,frame,ref,state=checked(design_id,p)
        state=canonical_mapping(d,mesh,frame,state)
        state=annotate_region(state,p.region,p.weights,confirmed=p.mapping_confirmed)
        request=Edit(operation='annotate',base_outer_path=d['outer_path'],semantic_region=p.region)
        return commit_edit(d,request,mesh,frame,ref,state,p.expected_input_digest,annotation_state=state)
