"""Atomic Step-5 sculpt commits for an already complete Crown 1.0 result.

Generation owns intaglio creation and cervical stitching.  Step 5 edits only the
external crown morphology; a successful commit propagates that outer displacement
onto the existing watertight final mesh while keeping the generated intaglio and
cervical seam fixed.  Therefore a sculpt commit remains directly exportable and
never returns the design to an outer-only workflow.
"""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from ..geometry.io import load_mesh,save_mesh,mesh_info
from ..geometry.sculpt import sculpt_edit,cervical_weights
from ..revisions import DesignRevision,RevisionKind
from .sculpt_session import reorder_state,measure_contacts
from .design_dependencies import file_digest,dependency_digest,publish_artifacts,require_current_final


CONTACT_AUDIT_SAMPLES=1024


def contact_audit(design,mesh,frame,ref):
    """Cheap provenance audit; full contact maps remain on the analysis endpoint."""
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
    """语义权重唯一取用口：人工标注优先，其次自动几何。"""
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


def _body_digests(d,updates):
    """Publish outer/body currentness together with the complete final revision."""
    merged={**d,**updates}
    updates['outer_dependency_digest']=dependency_digest(merged,stage='outer')
    updates['outer_output_sha256']=file_digest(merged['outer_path'])
    landmarks=json.loads(merged.get('landmarks_json') or '{}')
    landmarks['_body_dependency_digest']=dependency_digest(merged,stage='body')
    updates['landmarks_json']=json.dumps(landmarks,ensure_ascii=False)
    return updates


def _qc_after_sculpt(d,final_path,cutter_path,outer_surface_path,sync_report,metadata):
    qc=json.loads(d.get('qc_json') or '{}')
    paths=dict(qc.get('artifact_paths') or {})
    if outer_surface_path is not None:
        paths['outer_surface']=str(outer_surface_path)
    qc['artifact_paths']=paths
    qc['release_status']='ENGINEERING_ONLY'
    qc['sculpt_sync']={
        **(sync_report or {'status':'ANNOTATION_ONLY_GEOMETRY_UNCHANGED'}),
        'final_path':str(final_path),
        'cutter_path':str(cutter_path),
        'intaglio_regenerated':False,
        'cervical_stitch_regenerated':False,
        'step5_directly_exportable':True,
        'edit_status':metadata.get('status','APPLIED'),
    }
    return qc


def commit_edit(d,p,mesh,frame,ref,state,expected,*,annotation_state=None):
    from ..config import CASE_DIR
    # Product invariant: Step 5 only exists after one-click generation has already
    # produced a current, closed final crown and intaglio.
    require_current_final(d)
    margin=json.loads(d['margin_json']);before_sha=file_digest(d['outer_path'])
    before=contact_audit(d,mesh,frame,ref)
    metadata={};state={**state};region_source=None

    if annotation_state is not None:
        # Semantic annotation changes no geometry.  Keep outer/final/cutter bytes
        # exactly as generated and only advance state/provenance atomically.
        state=annotation_state
        metadata={'affected_area_mm2':0,'max_displacement_mm':0,'p95_displacement_mm':0,
                  'status':'ANNOTATION_ONLY_GEOMETRY_UNCHANGED'}
        state.update(outer_sha256=before_sha,frame_revision_id=ref['revision_id'])
        state_path=CASE_DIR/d['case_id']/f"{d['id']}_sculpt_{uuid4().hex}.sculpt.json"
        state_path.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
        qc=_qc_after_sculpt(d,d['final_path'],d['cutter_path'],None,None,metadata)
        updates={'outer_path':d['outer_path'],'final_path':d['final_path'],'cutter_path':d['cutter_path'],
                 'qc_json':json.dumps(qc,ensure_ascii=False),'status':'final_ready',
                 'sculpt_state_json':json.dumps(state,ensure_ascii=False)}
        _body_digests(d,updates)
        revision=DesignRevision(design_id=d['id'],revision_no=0,kind=RevisionKind.OUTER,
            payload={'type':'sculpt_annotation','tool':p.operation,'operation':p.operation,
                     'semantic_region':p.semantic_region,'semantic_region_source':region_source,
                     'previous_outer':d['outer_path'],'before_sha256':before_sha,
                     'after_sha256':before_sha,'output_sha256':before_sha,
                     'sculpt_state_sha256':file_digest(state_path),
                     'complete_final_preserved':True,'intaglio_preserved':True,
                     'outer_dependency_digest':updates['outer_dependency_digest'],**metadata},
            algorithm_version='crown_sculpt/2-complete-final',parameter_set=p.model_dump(),
            artifacts={'outer':d['outer_path'],'final':d['final_path'],'cutter':d['cutter_path'],
                       'sculpt_state':str(state_path)})
        publish_artifacts(d,expected,updates,stage='final',revision=revision)
        return {'outer_path':d['outer_path'],'final_path':d['final_path'],'cutter_path':d['cutter_path'],
                'revision_no':revision.revision_no,'mesh':mesh_info(mesh),'final_invalidated':False,
                'complete_crown_ready':True,'contacts':before,'edit':revision.payload,
                'thickness_audit':None,'status':'ANNOTATION_ONLY_GEOMETRY_UNCHANGED'}

    if p.semantic_region and p.semantic_region!='entire_tooth':
        weights,region_source=resolve_region_weights(d,mesh,frame,ref,state,p.semantic_region)
        lock=cervical_weights(mesh,margin,frame.z)
        fixed=np.maximum.reduce([w for k,w in state['regions'].items() if k.startswith('cusp_')]) if p.semantic_region=='central_fissure' and any(k.startswith('cusp_') for k in state['regions']) else None
        p=p.model_copy(update={'vertex_weights':weights,'region_masks':region_masks(mesh,weights,lock,fixed_weights=fixed)})
    if p.operation in ('adapt_mesial','adapt_distal','cut_intersections','adapt_antagonist'):
        from .sculpt_adapt import adapt_sculpt
        edited,metadata=adapt_sculpt(d,mesh,frame,ref,state,p)
    else:
        edited,metadata=sculpt_edit(mesh,margin,frame.z,p,frame)
    if p.region_masks:state['region_masks']=p.region_masks
    if metadata.get('status')=='NO_APPLICABLE_SIDE':
        return {'outer_path':d['outer_path'],'final_path':d['final_path'],'cutter_path':d['cutter_path'],
            'revision_no':None,'mesh':mesh_info(mesh),'final_invalidated':False,
            'complete_crown_ready':True,'contacts':before,'edit':metadata,'thickness_audit':None,
            'status':'NO_APPLICABLE_SIDE'}

    outer_path=CASE_DIR/d['case_id']/f"{d['id']}_sculpt_{uuid4().hex}.stl"
    save_mesh(edited,outer_path)
    persisted=load_mesh(outer_path);state=reorder_state(state,edited,persisted)
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
            'suggested_action':'减小减材强度/半径；内冠保持生成阶段结果，不需要重新生成',
        }) from exc

    # Transfer the already accepted external displacement into the generated
    # watertight final.  Intaglio and cervical seam vertices are immutable here.
    from .sculpt_final_sync import sync_complete_final,FinalSurfaceSyncError
    try:
        final_mesh,outer_surface_mesh,sync_report=sync_complete_final(d,mesh,edited,frame)
    except FinalSurfaceSyncError as exc:
        from ..geometry.sculpt import SculptEditRejected
        displacement=np.linalg.norm(np.asarray(edited.vertices)-np.asarray(mesh.vertices),axis=1)
        raise SculptEditRejected({
            'status':'REJECTED_AND_ROLLED_BACK','reason':'FINAL_SURFACE_SYNC','message':str(exc),
            'operation':p.operation,'affected_vertex_count':int(np.count_nonzero(displacement>1e-9)),
            'max_displacement_mm':float(displacement.max(initial=0.0)),
            'p95_displacement_mm':float(np.quantile(displacement,.95)) if len(displacement) else 0.0,
            'locked_displacement_max_mm':0.0,'severe_flip_count':0,'new_self_intersection_count':0,
            'baseline_self_intersection_count':0,'after_self_intersection_count':0,'degenerate_face_count':0,
            'failure_face_ids':[],'failure_region':'FINAL_EXTERNAL_SURFACE',
            'suggested_action':'减小微调强度/半径；完整冠、内冠和 Margin 已保留并回滚本次修改',
        }) from exc

    final_path=CASE_DIR/d['case_id']/f"{d['id']}_final_sculpt_{uuid4().hex}.stl"
    outer_surface_path=CASE_DIR/d['case_id']/f"{d['id']}_outer_surface_sculpt_{uuid4().hex}.stl"
    save_mesh(final_mesh,final_path);save_mesh(outer_surface_mesh,outer_surface_path)
    after=contact_audit(d,persisted,frame,ref)
    state.update(outer_sha256=file_digest(outer_path),frame_revision_id=ref['revision_id'])
    state_path=outer_path.with_suffix('.sculpt.json')
    state_path.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
    qc=_qc_after_sculpt(d,final_path,d['cutter_path'],outer_surface_path,sync_report,metadata)
    updates={'outer_path':str(outer_path),'final_path':str(final_path),'cutter_path':d['cutter_path'],
             'qc_json':json.dumps(qc,ensure_ascii=False),'status':'final_ready',
             'sculpt_state_json':json.dumps(state,ensure_ascii=False)}
    _body_digests(d,updates)
    report=lambda value:{k:v for k,v in value.items() if k not in ('map','sample_ids')}
    revision=DesignRevision(design_id=d['id'],revision_no=0,kind=RevisionKind.OUTER,
        payload={'type':'manual_complete_crown_edit','tool':p.operation,'operation':p.operation,
            'semantic_region':p.semantic_region,'semantic_region_source':region_source,
            'previous_outer':d['outer_path'],'previous_final':d['final_path'],
            'before_sha256':before_sha,'after_sha256':file_digest(outer_path),
            'output_sha256':file_digest(outer_path),'final_sha256':file_digest(final_path),
            'cutter_sha256':file_digest(d['cutter_path']),'sculpt_state_sha256':file_digest(state_path),
            'contact_before':report(before),'contact_after':report(after),
            'thickness_audit':{k:v for k,v in thickness.items() if k!='heatmap'},
            'final_surface_sync':sync_report,'complete_final_preserved':True,
            'intaglio_preserved':True,'cervical_stitch_topology_preserved':True,
            'outer_dependency_digest':updates['outer_dependency_digest'],**metadata},
        algorithm_version='crown_sculpt/2-complete-final',parameter_set=p.model_dump(),
        artifacts={'outer':str(outer_path),'final':str(final_path),'cutter':d['cutter_path'],
                   'outer_surface':str(outer_surface_path),'sculpt_state':str(state_path)})
    publish_artifacts(d,expected,updates,stage='final',revision=revision)
    return {'outer_path':str(outer_path),'final_path':str(final_path),'cutter_path':d['cutter_path'],
        'revision_no':revision.revision_no,'mesh':mesh_info(persisted),'final_invalidated':False,
        'complete_crown_ready':True,'contacts':after,'edit':revision.payload,'thickness_audit':thickness,
        'status':metadata.get('status','APPLIED')}
