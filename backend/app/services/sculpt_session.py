"""Immutable sculpt state and canonical-template semantic weights."""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from fastapi import HTTPException
from ..geometry.io import load_mesh
from ..geometry.sculpt import cervical_weights
from ..geometry.crown_regions import OUTER_REGIONS
from .design_dependencies import file_digest,require_current_outer,dependency_digest

REGIONS=('mesial','distal','buccal','lingual','occlusal','cusp_1','cusp_2','cusp_3','cusp_4',
    'central_fissure','mesial_marginal_ridge','distal_marginal_ridge')


def read_session(design,*,allow_legacy=False):
    require_current_outer(design)
    from ..api.auto_fit import frame_from
    if allow_legacy and not design.get('autofit_frame_json') and json.loads(design.get('landmarks_json') or '{}').get('legacy_landmarks'):
        from ..geometry.crown_outer.patient_frame import build_patient_frame
        frame=build_patient_frame(np.asarray(json.loads(design['margin_json'])),np.asarray(json.loads(design['axis_json'])))
        ref={'revision_id':None,'status':'LEGACY_RESEARCH_NO_DENTAL_SEMANTICS'}
    else:frame,ref=frame_from(design)
    mesh=load_mesh(Path(design['outer_path']))
    if len(mesh.vertices)>60000:raise HTTPException(422,'微调当前工程预算为 60000 顶点，请先简化')
    sha=file_digest(design['outer_path']);state=json.loads(design.get('sculpt_state_json') or 'null')
    if state:
        if state['outer_sha256']!=sha or state['frame_revision_id']!=ref['revision_id']:
            state=None
    if state is None:state={'outer_sha256':sha,'frame_revision_id':ref['revision_id'],'regions':{},'region_masks':None,'template':None}
    from .sculpt_thickness import attach_reference
    state=attach_reference(design,state)
    return mesh,frame,ref,state


def canonical_mapping(design,mesh,frame,state):
    from .template_source import resolve_template_source
    source=resolve_template_source(design)
    if state.get('template'):
        if state['template']['template_sha256']!=source.sha256:raise HTTPException(409,'模板已改变，语义标注过期')
        return state
    margin=np.asarray(json.loads(design['margin_json']))
    if source.library_obj:
        from .auto_fit_templates import canonical_library_crown,placed_library_crown
        canonical,_=canonical_library_crown(source.library_obj,int(design['target_fdi']))
        placed,_=placed_library_crown(source.library_obj,int(design['target_fdi']),margin,frame)
    else:
        from ..geometry.registration import initialize_from_margin
        canonical=load_mesh(source.path)
        placed=initialize_from_margin(canonical,margin,frame.z,patient_frame=frame)
    from ..geometry.manufacturing_qc import _closest
    points,distance,faces=_closest(placed,np.asarray(mesh.vertices))
    if distance.max()>1.5:raise HTTPException(422,'当前冠形与规范模板偏离超过 1.5 mm，不能可靠绑定语义；请先恢复模板冠形')
    bary=trimesh.triangles.points_to_barycentric(placed.triangles[faces],points)
    bary=np.clip(bary,0,1);bary/=bary.sum(axis=1)[:,None]
    return {**state,'template':{'schema':'TemplateAnatomyMetadata/1','template_sha256':source.sha256,
        'fdi':int(design['target_fdi']),'regions':{},'canonical_vertex_count':len(canonical.vertices),
        'canonical_faces':np.asarray(canonical.faces).tolist(),'mapping_review_required':True},
        'canonical_face_ids':faces.tolist(),'canonical_barycentric':bary.tolist()}


def annotate_region(state,name,weights,*,confirmed=False):
    if name not in REGIONS:raise HTTPException(422,'不支持该解剖区域名称')
    if not confirmed:raise HTTPException(422,'请人工确认模板映射及所标注的解剖区域')
    template=state['template'];faces=np.asarray(template['canonical_faces'])[state['canonical_face_ids']]
    bary=np.asarray(state['canonical_barycentric']);w=np.asarray(weights,float)
    if w.shape!=(len(faces),) or not np.isfinite(w).all() or np.any((w<0)|(w>1)) or not np.any(w>.5):
        raise HTTPException(422,'语义标注需要与当前冠形对应的有效权重')
    values=np.zeros(template['canonical_vertex_count']);counts=np.zeros(len(values))
    for corner in range(3):
        np.add.at(values,faces[:,corner],w*bary[:,corner]);np.add.at(counts,faces[:,corner],bary[:,corner])
    canonical=np.divide(values,counts,out=np.zeros_like(values),where=counts>1e-12)
    template={**template,'mapping_review_required':False,'regions':{**template['regions'],name:{'weights':canonical.tolist(),'source':'MANUAL_CONFIRMED'}}}
    propagated=np.clip(np.sum(canonical[faces]*bary,axis=1),0,1)
    return {**state,'template':template,'regions':{**state['regions'],name:propagated.tolist()}}


def reorder_state(state,edited,persisted):
    distances,ids=cKDTree(edited.vertices).query(persisted.vertices)
    if len(ids)!=len(edited.vertices) or len(np.unique(ids))!=len(ids) or distances.max()>1e-5:
        raise ValueError('STL 顶点对应已变化，不能静默丢失语义映射')
    result={**state,'regions':{name:np.asarray(weights)[ids].tolist() for name,weights in state['regions'].items()}}
    for key in ('region_masks','canonical_face_ids','canonical_barycentric'):
        if state.get(key) is not None:result[key]=np.asarray(state[key])[ids].tolist()
    return result


def measure_contacts(design,mesh,frame,ref,*,sample_ids=None):
    if ref.get('status')=='LEGACY_RESEARCH_NO_DENTAL_SEMANTICS':
        return {'status':'NOT_AVAILABLE','reports':{},'map':None,'skipped':{'frame':'LEGACY_RESEARCH_NO_DENTAL_SEMANTICS'}}
    from .contact_measurement import confirmed_neighbor_surfaces,confirmed_antagonist_surface,measure_surface,per_vertex_contact,combine_per_vertex
    parameters=json.loads(design['parameters_json']);neighbors,skipped=confirmed_neighbor_surfaces(design,ref,strict=False)
    # A declared reference losing its truth contract blocks use; an unconfirmed one is NOT_AVAILABLE.
    entries=json.loads(design.get('neighbors_json') or '{}')
    if any(entries.get(side,{}).get('confirmation_status')=='CONFIRMED' for side in skipped):raise HTTPException(409,'邻牙来源已过期，请重新确认')
    ant,bite,error=confirmed_antagonist_surface(design,ref,strict=False)
    if error and bite.get('pose_confirmed'):raise HTTPException(409,'咬合来源已过期，请重新确认')
    surfaces={side:surface for side,(surface,_) in neighbors.items()}
    if ant:surfaces['antagonist']=ant
    reports={};items={};ids=np.arange(len(mesh.vertices)) if sample_ids is None else np.asarray(sample_ids,int)
    for side,surface in surfaces.items():
        target=parameters.get('occlusal_target_mm' if side=='antagonist' else 'proximal_target_mm')
        if target is None:skipped[side]={'status':'NOT_AVAILABLE','reason':'TARGET_UNCONFIRMED'};continue
        mask=np.zeros(len(mesh.vertices),bool);mask[ids]=True
        reports[side]=measure_surface(surface,mesh.vertices,mask,target,kind='OCCLUSAL' if side=='antagonist' else 'PROXIMAL',side=side)[0]
        item=per_vertex_contact(surface,np.asarray(mesh.vertices)[ids]);items[side]=item
    combined=combine_per_vertex(items)
    if combined:
        targets=[]
        for i,value in enumerate(combined['signed_mm']):
            available=[key for key,item in items.items() if item['valid'][i]]
            side=min(available,key=lambda k:items[k]['signed_mm'][i]) if available else None
            targets.append(reports[side]['target_mm'] if side else None)
        combined['target_mm']=targets
    return {'status':'MEASURED' if reports else 'NOT_AVAILABLE','reports':reports,'skipped':skipped,
        'sample_ids':ids.tolist(),'map':combined,'target_contact_area_status':'NOT_AVAILABLE_NO_VALIDATED_AREA_DEFINITION',
        'implementation':'ContactMeasurement','frame_revision_id':ref['revision_id']}


def region_map_or_error(design,mesh,frame,ref):
    """自动语义区域：**永不抛**，拿不到就给出可读原因（fail-loud 但不阻断会话）。

    适配与自由塑形各自走同一条构建路径；这里只是给前端一份可显示的状态与计数。
    """
    from ..geometry.crown_regions import RegionMapError,OUTER_REGIONS,build_region_map
    from .contact_measurement import confirmed_neighbor_surfaces
    neighbors=None;neighbor_reports=None
    try:
        neighbors,neighbor_reports=confirmed_neighbor_surfaces(design,ref,strict=False)
    except (ValueError,KeyError,OSError):
        neighbors=None
    try:
        return build_region_map(design,mesh,frame,ref,neighbors=neighbors,neighbor_reports=neighbor_reports),None
    except RegionMapError as exc:
        return None,{'status':'NOT_AVAILABLE','code':getattr(exc,'code',str(exc)),'message':str(exc),
                     'automatic_region_names':list(OUTER_REGIONS)}
    except (ValueError,KeyError,OSError) as exc:
        return None,{'status':'NOT_AVAILABLE','code':'REGION_BUILD_FAILED','message':str(exc),
                     'automatic_region_names':list(OUTER_REGIONS)}


def session_report(design,*,include_regions=False):
    mesh,frame,ref,state=read_session(design);lock=cervical_weights(mesh,json.loads(design['margin_json']),frame.z)
    region_map,region_error=region_map_or_error(design,mesh,frame,ref)
    from .contact_measurement import confirmed_neighbor_surfaces, neighbor_patch_diagnostics
    neighbors,_=confirmed_neighbor_surfaces(design,ref,strict=False)
    patch_reports=neighbor_patch_diagnostics(mesh,region_map,neighbors) if region_map is not None else {}
    contacts=measure_contacts(design,mesh,frame,ref,sample_ids=np.linspace(0,len(mesh.vertices)-1,min(512,len(mesh.vertices)),dtype=int))
    thickness_ready=bool(state.get('thickness_reference'))
    from ..geometry.crown_outer.deform_gate import nonmanifold_edges
    baseline_geometry={'vertices':int(len(mesh.vertices)),'faces':int(len(mesh.faces)),
        'watertight':bool(mesh.is_watertight),'nonmanifold_edges':int(nonmanifold_edges(mesh.faces)),
        'degenerate_faces':int(np.count_nonzero(np.asarray(mesh.area_faces)<1e-10)),
        'note':'Step5 baseline; full self-intersection audit remains in deformation/finalize gates'}
    return {'vertices':mesh.vertices.tolist(),'faces':mesh.faces.tolist(),'normals':mesh.vertex_normals.tolist(),'configured_minimum_mm':json.loads(design['parameters_json'])['minimum_thickness_mm'],
        'cervical_weights':lock.tolist(),'frame':{'origin':frame.origin.tolist(),'x':frame.x.tolist(),'y':frame.y.tolist(),
        'z':frame.z.tolist(),'revision_id':ref['revision_id'],'buccal':ref['buccal']},'state':state,
        'source_session_sha256':state['outer_sha256'],'input_digest':dependency_digest(design),
        'baseline_geometry':baseline_geometry,
        'edit_session':{'status':'EDITABLE','outer_sha256':state['outer_sha256'],'frame_revision_id':ref['revision_id'],
                        'contact_status':contacts.get('status'),'thickness_reference_valid':thickness_ready},
        'capabilities':{'add':True,'smooth':True,'flatten':True,'transform':True,'remove':thickness_ready,
                        'remove_reason':None if thickness_ready else 'REBUILD_REQUIRED'},
        'contacts':contacts,
        # 自动语义区域：report 恒有状态；逐顶点权重只在明确要求时下发（开发态叠加用）。
        'region_map':region_map.report() if region_map is not None else region_error,
        'neighbor_patches':patch_reports,
        'region_weights':region_map.weights_report() if (include_regions and region_map is not None) else None,
        'region_protection':region_map.protection.tolist() if (include_regions and region_map is not None) else None,
        'semantic_region_names':REGIONS,'manual_region_names':sorted((state.get('regions') or {}).keys()),
        'automatic_region_names':list(OUTER_REGIONS),'manual_annotation_required':False,
        'thickness_status':'FINAL_REFERENCE_AVAILABLE' if thickness_ready else 'REBUILD_REQUIRED'}
