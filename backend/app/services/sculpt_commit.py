"""All sculpt commits use the existing atomic outer revision publication."""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from ..geometry.io import load_mesh,save_mesh,mesh_info
from ..geometry.sculpt import sculpt_edit,cervical_weights
from ..revisions import DesignRevision,RevisionKind
from .sculpt_session import reorder_state,measure_contacts
from .design_dependencies import file_digest,dependency_digest,publish_artifacts


#: 提交时的前后接触是**审计记录、不是门**：全量逐顶点（6021 点）要 ≈6s，而每笔微调要跑
#: before/after 两次 ⇒ ≈12s，就是"每笔微调卡一下"的根源。这里按固定间隔采样测量，
#: 并在返回/payload 里写明**采样口径与覆盖率**（完整逐顶点图仍由 `outer-analysis` 端点按需提供）。
CONTACT_AUDIT_SAMPLES=1024


def contact_audit(design,mesh,frame,ref):
    total=int(len(mesh.vertices))
    ids=None if total<=CONTACT_AUDIT_SAMPLES else np.linspace(0,total-1,CONTACT_AUDIT_SAMPLES,dtype=int)
    measured=measure_contacts(design,mesh,frame,ref,sample_ids=ids)
    measured['audit_sampled']=ids is not None
    measured['vertex_count']=total
    measured['audit_sample_count']=int(len(measured.get('sample_ids') or []))
    measured['audit_note']=('采样审计测量（每 %d 点取 1）'%max(total//CONTACT_AUDIT_SAMPLES,1)) if ids is not None else '全量逐顶点'
    return measured


def region_masks(mesh,weights,lock,*,fixed_weights=None):
    moving=(np.asarray(weights)>.5)&(lock>0)
    masks=np.full(len(mesh.vertices),'FIXED',dtype='<U7');masks[moving]='MOVING'
    adj=[set() for _ in mesh.vertices]
    for a,b in mesh.edges_unique:adj[a].add(int(b));adj[b].add(int(a))
    seen=set(np.flatnonzero(moving));front=seen.copy()
    for _ in range(3):
        front={j for i in front for j in adj[i]}-seen;seen|=front
        masks[list(front)]='ELASTIC'
    masks[lock==0]='FIXED'
    if fixed_weights is not None:masks[np.asarray(fixed_weights)>.5]='FIXED'
    return masks.tolist()


def resolve_region_weights(design,mesh,frame,ref,state,name):
    """语义权重**唯一取用口**：人工标注优先，其次自动几何（CrownRegionMap）。

    返回**列表**（与人工标注路径同一形状）：这些权重会进 `Edit.parameter_set` 与修订 payload，
    ndarray 会让 json 序列化失败（实测踩到）。
    自动区域只覆盖外表面语义（MESIAL/DISTAL/BUCCAL/LINGUAL/OCCLUSAL）；
    解剖区域名（cusp_* / central_fissure / marginal_ridge）仍必须人工标注模板映射。
    """
    manual=(state.get('regions') or {}).get(name)
    if manual is not None:return np.asarray(manual,float).tolist(),'MANUAL_CONFIRMED'
    from ..geometry.crown_regions import OUTER_REGIONS,build_region_map,region_weights
    if name not in OUTER_REGIONS:
        raise ValueError('该解剖区域只能通过人工标注模板映射得到：'+str(name))
    neighbors=None
    try:
        from .contact_measurement import confirmed_neighbor_surfaces
        neighbors,_=confirmed_neighbor_surfaces(design,ref,strict=False)
    except (ValueError,KeyError,OSError):neighbors=None
    region_map=build_region_map(design,mesh,frame,ref,neighbors=neighbors)
    weights,_source=region_weights(state,name,region_map)
    return np.asarray(weights,float).tolist(),'AUTO_GEOMETRY'


def commit_edit(d,p,mesh,frame,ref,state,expected,*,annotation_state=None):
    from ..config import CASE_DIR
    margin=json.loads(d['margin_json']);before_sha=file_digest(d['outer_path'])
    before=contact_audit(d,mesh,frame,ref)
    metadata={};state={**state};region_source=None
    if annotation_state is not None:
        edited=mesh.copy();state=annotation_state;metadata={'affected_area_mm2':0,'max_displacement_mm':0,'p95_displacement_mm':0}
    else:
        if p.semantic_region and p.semantic_region!='entire_tooth':
            weights,region_source=resolve_region_weights(d,mesh,frame,ref,state,p.semantic_region)
            lock=cervical_weights(mesh,margin,frame.z)
            # Fissure edits hold all manually identified cusp tips fixed.
            fixed=np.maximum.reduce([w for k,w in state['regions'].items() if k.startswith('cusp_')]) if p.semantic_region=='central_fissure' and any(k.startswith('cusp_') for k in state['regions']) else None
            p=p.model_copy(update={'vertex_weights':weights,'region_masks':region_masks(mesh,weights,lock,fixed_weights=fixed)})
        if p.operation in ('adapt_mesial','adapt_distal','cut_intersections','adapt_antagonist'):
            from .sculpt_adapt import adapt_sculpt
            edited,metadata=adapt_sculpt(d,mesh,frame,ref,state,p)
        else:edited,metadata=sculpt_edit(mesh,margin,frame.z,p,frame)
        if p.region_masks:state['region_masks']=p.region_masks
    if metadata.get('status')=='NO_APPLICABLE_SIDE':
        # 请求适配的侧都不可用（例如该侧缺牙 / 自动区域为空）：**不写修订、不动几何**，
        # 如实把原因返回给前端（缺牙不是算法错误，也不该在修订链里留一条空改动）。
        return {'outer_path':d['outer_path'],'revision_no':None,'mesh':mesh_info(mesh),
            'final_invalidated':False,'contacts':before,'edit':metadata,'thickness_audit':None,
            'status':'NO_APPLICABLE_SIDE'}
    path=CASE_DIR/d['case_id']/f"{d['id']}_sculpt_{uuid4().hex}.stl";save_mesh(edited,path)
    persisted=load_mesh(path);state=reorder_state(state,edited,persisted)
    from .sculpt_thickness import formal_thickness_gate
    try:
        thickness=formal_thickness_gate(d,mesh,persisted,state)
    except ValueError as exc:
        from ..geometry.sculpt import SculptEditRejected
        displacement=np.linalg.norm(np.asarray(persisted.vertices)-np.asarray(mesh.vertices),axis=1)
        raise SculptEditRejected({
            'status':'REJECTED_AND_ROLLED_BACK','reason':'THICKNESS_GATE','message':str(exc),
            'operation':p.operation,'affected_vertex_count':int(np.count_nonzero(displacement>1e-9)),
            'max_displacement_mm':float(displacement.max()) if len(displacement) else 0.0,
            'p95_displacement_mm':float(np.quantile(displacement,.95)) if len(displacement) else 0.0,
            'locked_displacement_max_mm':0.0,'severe_flip_count':0,'new_self_intersection_count':0,
            'baseline_self_intersection_count':0,'after_self_intersection_count':0,'degenerate_face_count':0,
            'failure_face_ids':[],'failure_region':'THICKNESS_PROTECTED',
            'suggested_action':'减小减材强度/半径，或先重建内冠与厚度参考后再编辑',
        }) from exc
    after=contact_audit(d,persisted,frame,ref)
    state.update(outer_sha256=file_digest(path),frame_revision_id=ref['revision_id'])
    state_path=path.with_suffix('.sculpt.json');state_path.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
    updates={'outer_path':str(path),'final_path':None,'cutter_path':None,'qc_json':None,'status':'outer_ready',
        'sculpt_state_json':json.dumps(state,ensure_ascii=False)}
    report=lambda value:{k:v for k,v in value.items() if k not in ('map','sample_ids')}
    revision=DesignRevision(design_id=d['id'],revision_no=0,kind=RevisionKind.OUTER,
        payload={'type':'sculpt_annotation' if annotation_state is not None else 'manual_outer_edit',
            'tool':p.operation,'operation':p.operation,'semantic_region':p.semantic_region,
            'semantic_region_source':region_source,'previous_outer':d['outer_path'],
            'before_sha256':before_sha,'after_sha256':file_digest(path),'output_sha256':file_digest(path),
            'sculpt_state_sha256':file_digest(state_path),'contact_before':report(before),'contact_after':report(after),
            'thickness_audit':{k:v for k,v in thickness.items() if k!='heatmap'},
            'outer_dependency_digest':dependency_digest({**d,**updates},stage='outer'),**metadata},
        algorithm_version='crown_sculpt/1',parameter_set=p.model_dump(),
        artifacts={'outer':str(path),'sculpt_state':str(state_path)})
    publish_artifacts(d,expected,updates,stage='outer',revision=revision)
    return {'outer_path':str(path),'revision_no':revision.revision_no,'mesh':mesh_info(persisted),
        'final_invalidated':True,'contacts':after,'edit':revision.payload,'thickness_audit':thickness,
        'status':metadata.get('status','APPLIED')}
