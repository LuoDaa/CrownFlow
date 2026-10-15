"""Sequential proposals, frozen domains, joint remeasurement and W3 preflight."""
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from fastapi import HTTPException
from ..config import CASE_DIR
from ..db import db
from ..geometry.io import load_mesh, save_mesh
from ..geometry.nonplanar_crown import fitting_support_above_margin, MarginBuildError
from ..geometry.crown_outer.collar_body_outer import _boundary_loops
from ..geometry.crown_outer.auto_fit_adapter import FitSurface, fit_stage, fit_domains, measure, surface_crossings
from ..geometry.crown_outer.thickness import input_qc
from .design_dependencies import file_digest, require_current_outer
from .pipeline import finalize_crown


def _asset(d,kind):
    with db() as c: row=c.execute('SELECT * FROM assets WHERE case_id=? AND kind=? ORDER BY created_at DESC LIMIT 1',(d['case_id'],kind)).fetchone()
    return dict(row) if row else None


def audit_final_context(d,outer_path,final_path,qc):
    """Requery the persisted W3 surfaces; proposal metrics cannot stand in for final QC."""
    from ..api.auto_fit import frame_from
    from ..geometry.manufacturing_qc import _intersection_volume
    if not d.get('autofit_frame_json'): return
    frame,ref=frame_from(d);outer=load_mesh(Path(outer_path));final=load_mesh(Path(final_path))
    parameters=json.loads(d['parameters_json']);margin=np.asarray(json.loads(d['margin_json']))
    neighbors=json.loads(d.get('neighbors_json') or '{}');patches={};sides={}
    for side in ('mesial','distal'):
        entry=neighbors.get(side,{})
        if entry.get('frame_revision_id')!=ref['revision_id']:
            sides[side]={'status':'NOT_ASSESSED','reason':'NEIGHBOR_UNVERIFIED'};continue
        if entry.get('confirmation_status')=='ABSENT':
            sides[side]={'status':'SKIPPED_ABSENT'};continue
        if entry.get('confirmation_status')!='CONFIRMED':
            sides[side]={'status':'NOT_ASSESSED','reason':'NEIGHBOR_UNVERIFIED'};continue
        source=entry['source'];current=_asset(d,source['kind'])
        if (not current or current['id']!=source['asset_id'] or file_digest(current['path'])!=source['sha256']
            or file_digest(entry['path'])!=entry['region_sha256']):
            raise HTTPException(409,'最终接触审核的邻牙来源已变化，请重新确认')
        patches[side]=FitSurface(load_mesh(Path(entry['path'])),entry.get('outward_normal'))
    target=parameters.get('proximal_target_mm')
    if patches and target is not None:
        domains=fit_domains(outer,frame,margin,patches,'PROXIMAL')[0]
        for side,surface in patches.items():
            sides[side]={**measure(outer.vertices,domains[side],surface,target,kind='PROXIMAL',side=side)[0],
                'neighbor_revision_id':neighbors[side]['revision_id'],
                'material_collision':_intersection_volume(final,surface.mesh),
                'triangle_crossings':surface_crossings(outer,surface,domains[side])}
    else:
        for side in patches: sides[side]={'status':'NOT_ASSESSED','reason':'TARGET_UNCONFIRMED'}
    qc['ContactQC']={'status':'MEASURED_ENGINEERING' if patches and target is not None else 'NOT_ASSESSED',
        'per_side':sides,'scope':'frozen_final_outer_vertex_domains_not_continuous_certification',
        'geometry_frame_revision_id':ref['revision_id'],'final_sha256':file_digest(final_path)}
    bite=json.loads(d.get('bite_json') or '{}');target=parameters.get('occlusal_target_mm')
    if not (bite.get('pose_confirmed') and bite.get('revision_id') and bite.get('frame_revision_id')==ref['revision_id'] and target is not None): return
    for source in bite.get('scans',[]):
        current=_asset(d,source['kind'])
        if not current or current['id']!=source['asset_id'] or file_digest(current['path'])!=source['sha256']:
            raise HTTPException(409,'最终咬合审核的扫描来源已变化，请重新确认')
    ant=load_mesh(Path(_asset(d,'upper' if d['target_fdi']>=31 else 'lower')['path']))
    ant.apply_transform(np.asarray(bite['transform']));surface=FitSurface(ant,-frame.z)
    mask=fit_domains(outer,frame,margin,{'occlusal':surface},'OCCLUSAL')[0]['occlusal']
    qc['OcclusionQC']={**measure(outer.vertices,mask,surface,target,kind='OCCLUSAL')[0],
        'bite_revision_id':bite['revision_id'],'scope':'confirmed_static_pose_sampled_outer_not_dynamic_occlusion',
        'material_collision':_intersection_volume(final,ant),'triangle_crossings':surface_crossings(outer,surface,mask),
        'final_sha256':file_digest(final_path)}


def propose(d,frame,frame_ref,mode,baseline_path=None):
    if baseline_path is None: require_current_outer(d)
    else: d={**d,'outer_path':str(baseline_path)}
    baseline=load_mesh(Path(d['outer_path']));mesh=baseline.copy()
    if len(mesh.faces)>100000: raise ValueError('AutoFit 同步工程预算为 100000 个冠面，请先简化并重新确认')
    margin=np.asarray(json.loads(d['margin_json']));axis=np.asarray(json.loads(d['axis_json']))
    parameters=json.loads(d['parameters_json']);support_ref=_asset(d,'support');support=load_mesh(Path(support_ref['path']))
    from .abutment_support import prepare_design_support
    support_contract,prepared_support=prepare_design_support(d)
    if prepared_support:support=prepared_support.mesh
    prefix=CASE_DIR/d['case_id']/f"{d['id']}_autofit_{uuid4().hex}"
    report={'mode':mode,'status':'INPUT_UNVERIFIED','stages':[],'per_side':{},'solver_version':'local_band_exact_query_alpha/1.0',
        'material_process_status':'NOT_ASSESSED','manufacturing_status':'NOT_PRODUCTION_APPROVED',
        'geometry_frame_revision_id':frame_ref['revision_id'],'neighbor_revision_ids':{},'bite_revision_id':None,
        'template_sha256':None,'baseline_outer_sha256':file_digest(d['outer_path'])}
    with db() as conn:
        row=conn.execute("SELECT id FROM design_revisions WHERE design_id=? AND kind='outer' AND json_extract(artifacts_json,'$.outer')=? ORDER BY revision_no DESC LIMIT 1",(d['id'],d['outer_path'])).fetchone()
    report['baseline_outer_revision_id']=row['id'] if row else None
    # A0：模板来源**只有一个真相** —— 报告里的 sha256 必须就是 draft/placement 真正消费的那个源
    from .template_source import resolve_template_source
    resolved=resolve_template_source(d)
    report['template_sha256']=resolved.sha256
    report['template_source']=resolved.as_report()
    if not resolved.is_user_selected:
        report['template_source_warning']='未找到人工确认的牙库；本次使用回退模板，验收前请确认牙库'
    diagnostics={'prep_sha256':file_digest(support_ref['path']),'prep_components_faces':[len(part.faces) for part in support.split(only_watertight=False)],
        'prep_boundary_loops':len(_boundary_loops(support.faces))}
    report['input_diagnostics']=diagnostics
    from ..geometry.prep_margin_diagnostics import diagnose
    provenance,cropped=diagnose(support,margin,axis)
    diagnostic_path=prefix.with_name(prefix.name+'_input_diagnostics.json')
    diagnostic_path.write_text(json.dumps(provenance,ensure_ascii=False),encoding='utf-8')
    diagnostic_artifacts={'input_diagnostics':str(diagnostic_path)}
    if cropped is not None and len(cropped.faces):
        crop_path=prefix.with_name(prefix.name+'_diagnostic_crop.stl');save_mesh(cropped,crop_path)
        diagnostic_artifacts['diagnostic_crop']=str(crop_path)
    diagnostics.update(codes=provenance['codes'],cropped_components_faces=provenance.get('cropped_components_faces'),
        cropped_boundary_loops=len(provenance.get('cropped_boundary_loops_world',[])))
    blocked=bool(provenance['codes'])
    try:
        fitting,domain=fitting_support_above_margin(support,margin,axis,prepared=prepared_support is not None);diagnostics['fitting_domain']=domain
    except MarginBuildError as exc:
        reason=str(exc)
        diagnostics['failure_code']='MARGIN_CROSSES_OPEN_BOUNDARY' if '缺口' in reason else 'PREP_MARGIN_TOPOLOGY_UNSUPPORTED'
        diagnostics['reason']=reason;blocked=True
        if diagnostics['failure_code'] not in provenance['codes']: provenance['codes'].append(diagnostics['failure_code'])
        provenance['fitting_failure_reason']=reason
    diagnostic_path.write_text(json.dumps(provenance,ensure_ascii=False),encoding='utf-8')
    q=input_qc(mesh.vertices,mesh.faces)
    diagnostics['baseline_topology']=q
    if q['nonmanifold_edges'] or q['degenerate_faces'] or q.get('self_intersections_truncated') or q.get('self_intersections_severe',0):
        diagnostics['baseline_geometry_status']='HARD_FAIL';blocked=True
    neighbors=json.loads(d.get('neighbors_json') or '{}');patches={};unknown=[]
    for side in ('mesial','distal'):
        entry=neighbors.get(side,{})
        report['neighbor_revision_ids'][side]=entry.get('revision_id')
        if entry.get('confirmation_status')=='ABSENT' and entry.get('frame_revision_id')==frame_ref['revision_id']:
            report['per_side'][side]={'status':'SKIPPED_ABSENT'};continue
        if entry.get('confirmation_status')!='CONFIRMED' or entry.get('frame_revision_id')!=frame_ref['revision_id']:
            report['per_side'][side]={'status':'INPUT_UNVERIFIED'};unknown.append(side);continue
        source=entry.get('source',{});current=_asset(d,source.get('kind'))
        path=entry.get('path')
        if (not current or current['id']!=source.get('asset_id') or file_digest(current['path'])!=source.get('sha256')
            or not path or not Path(path).is_file() or file_digest(path)!=entry.get('region_sha256')):
            raise HTTPException(409,'邻牙来源或区域已变化，请重新确认')
        patches[side]=FitSurface(load_mesh(Path(path)),entry.get('outward_normal'))
        report['per_side'][side]={'status':'CONFIRMED','instance_ref':entry['instance_ref'],'scope':entry['scope']}
    # 参数一律经 profile 解析：**不再 `parameters.get(...) or 0`**。
    # 那会让"没填"与"填了 0"不可区分，而且 0 会当成真实目标距离去量几何（本批修掉的反例）。
    from .crown_generation_profile import SOURCE_MISSING, from_parameters
    profile = from_parameters(parameters)
    preserve_target, preserve_source = profile.value('proximal.target_distance_mm')
    domains=fit_domains(baseline,frame,margin,patches,'PROXIMAL')[0] if patches else {}
    def _joint(vertices,side,surface):
        """近远中复测唯一入口：目标 MISSING ⇒ NOT_ASSESSED（绝不用 0 顶替）。"""
        if preserve_source==SOURCE_MISSING:
            return {'status':'NOT_ASSESSED','reason':'PROXIMAL_TARGET_UNCONFIRMED'}
        return measure(vertices,domains[side],surface,float(preserve_target),kind='PROXIMAL',side=side)[0]
    before_joint = {}
    for side, surface in patches.items():
        before_joint[side] = _joint(baseline.vertices, side, surface)
    required=[];incomplete=False
    if mode in ('PROXIMAL_ONLY','BOTH'):
        target=parameters.get('proximal_target_mm')
        if blocked: report['stages'].append({'stage':'PROXIMAL','status':'INPUT_UNVERIFIED','reason':'PREP_MARGIN_BLOCKED'});incomplete=True
        elif not patches: report['stages'].append({'stage':'PROXIMAL','status':'NOT_ASSESSED','reason':'NO_CONFIRMED_NEIGHBOR'});incomplete=True
        elif target is None: report['stages'].append({'stage':'PROXIMAL','status':'INPUT_UNVERIFIED','reason':'PROXIMAL_TARGET_UNCONFIRMED'});incomplete=True
        else:
            targets={s:target for s in patches};mesh,rep,domains=fit_stage(mesh,frame,margin,patches,targets,'PROXIMAL')
            before_joint=rep['before'];report['stages'].append(rep);required.append(rep['status'])
            placed_path=prefix.with_name(prefix.name+'_proximal.stl');save_mesh(mesh,placed_path)
            report['proximal_asset']={'path':str(placed_path),'sha256':file_digest(placed_path)}
        incomplete |= bool(unknown)
    last_verified=mesh.copy()
    occlusal=None;occl_mask=None;occl_before=None
    if mode in ('OCCLUSAL_ONLY','BOTH'):
        bite=json.loads(d.get('bite_json') or '{}');report['bite_revision_id']=bite.get('revision_id')
        trustworthy=bite.get('revision_id') and bite.get('pose_confirmed') and bite.get('transform') and bite.get('frame_revision_id')==frame_ref['revision_id']
        if trustworthy:
            for scan in bite.get('scans',[]):
                current=_asset(d,scan['kind'])
                if not current or current['id']!=scan['asset_id'] or file_digest(current['path'])!=scan['sha256']: raise HTTPException(409,'咬合来源扫描已变化')
        target=parameters.get('occlusal_target_mm')
        if blocked or not trustworthy or target is None:
            report['stages'].append({'stage':'OCCLUSAL','status':'NOT_ASSESSED','reason':'PREP_MARGIN_BLOCKED' if blocked else 'OCCLUSION_REGISTRATION_UNVERIFIED' if not trustworthy else 'OCCLUSAL_TARGET_UNCONFIRMED'});incomplete=True
        else:
            antagonist_kind='upper' if d['target_fdi']>=31 else 'lower'
            ant=load_mesh(Path(_asset(d,antagonist_kind)['path']));ant.apply_transform(np.asarray(bite['transform']))
            if len(ant.faces)>200000: raise ValueError('对颌工程预算为 200000 面，请先处理扫描并重新确认')
            # Axis-facing surface; open signs are only reviewed local references.
            occlusal=FitSurface(ant,(-frame.z).tolist())
            protected=np.zeros(len(mesh.vertices),bool)
            if preserve_source==SOURCE_MISSING:
                # 近远中目标没确认 ⇒ 无法定义"哪个距离算接触带"，保护带留空并如实登记，
                # 而不是拿 0 当目标去圈一块"接触区"（那会保护错地方）。
                report['occlusal_protection']={'status':'NOT_ASSESSED','reason':'PROXIMAL_TARGET_UNCONFIRMED'}
            else:
                for side,surface in patches.items():
                    if side not in domains: continue
                    _,query=measure(mesh.vertices,domains[side],surface,float(preserve_target),kind='PROXIMAL',side=side)
                    if query is not None:
                        ids,_,_,signed,_,valid=query
                        protected[ids[valid & (np.abs(signed-float(preserve_target))<=.15)]]=True
            saved=mesh.copy()
            mesh,rep,masks=fit_stage(mesh,frame,margin,{'occlusal':occlusal},{'occlusal':target},'OCCLUSAL',protected=protected,domain_mesh=baseline)
            occl_mask=masks['occlusal'];occl_before=measure(baseline.vertices,occl_mask,occlusal,target,kind='OCCLUSAL')[0]
            nonworse=True
            for side,surface in patches.items():
                if side not in domains: continue
                after=_joint(mesh.vertices,side,surface);previous=_joint(saved.vertices,side,surface)
                if previous.get('target_error') and (not after.get('target_error') or after['target_error']['max_mm']>previous['target_error']['max_mm']+1e-9): nonworse=False
            if not nonworse: mesh=saved;rep.update(status='CONSTRAINT_CONFLICT',reason='JOINT_CONTACT_PRESERVATION_FAILED')
            report['stages'].append(rep);required.append(rep['status'])
    from ..geometry.crown_outer.deform_gate import deformation_gate
    from ..geometry.nonplanar_crown import MarginCurtain
    audited={s:(surface,domains[s]) for s,surface in patches.items() if s in domains}
    if occl_mask is not None: audited['occlusal']=(occlusal,occl_mask)
    crossing_before={s:surface_crossings(baseline,surface,mask) for s,(surface,mask) in audited.items()}
    crossing_after={s:surface_crossings(mesh,surface,mask) for s,(surface,mask) in audited.items()}
    cross_ok=all(crossing_after[s].get('crossing_pairs') is not None and crossing_before[s].get('crossing_pairs') is not None
        and crossing_after[s]['crossing_pairs']<=crossing_before[s]['crossing_pairs'] for s in audited)
    report['joint_surface_crossings']={'before':crossing_before,'candidate':crossing_after,'nonworse':cross_ok}
    if not cross_ok:
        mesh=baseline.copy();required.append('CONSTRAINT_CONFLICT')
        report['joint_surface_crossings']['reason']='CROSSING_WORSENED_OR_NOT_ASSESSED'
    locked=np.flatnonzero(MarginCurtain(margin,axis).axial_above(baseline.vertices)<=.4)
    moved=np.linalg.norm(mesh.vertices-baseline.vertices,axis=1)>1e-9
    cumulative=deformation_gate(baseline.vertices,mesh.vertices,mesh.faces,scale_mm=.1,locked_ids=locked,
        focus_faces=np.flatnonzero(np.any(moved[mesh.faces],axis=1)))
    flips=int((np.einsum('ij,ij->i',baseline.face_normals,mesh.face_normals)<=0).sum())
    report['cumulative_geometry_gate']={**cumulative.as_dict(),'all_flips':flips}
    if not cumulative.ok or flips:
        mesh=last_verified;required.append('CONSTRAINT_CONFLICT')
    after_joint={s:_joint(mesh.vertices,s,surface) for s,surface in patches.items() if s in domains}
    joint={'before':before_joint,'after':after_joint,'occlusal_before':occl_before,
        'occlusal_after':measure(mesh.vertices,occl_mask,occlusal,parameters['occlusal_target_mm'],kind='OCCLUSAL')[0] if occl_mask is not None else None}
    report['joint_recheck']=joint
    report['metrics_before']={'proximal':before_joint,'occlusal':occl_before};report['metrics_after']={'proximal':after_joint,'occlusal':joint['occlusal_after']}
    changed=np.linalg.norm(mesh.vertices-baseline.vertices,axis=1)
    report['displacement']={'max_mm':float(changed.max()),'p95_mm':float(np.quantile(changed,.95))}
    outer_path=prefix.with_suffix('.stl');save_mesh(mesh,outer_path)
    # Audit the actual candidate file. In-memory data may differ after STL quantization.
    from scipy.spatial import cKDTree
    persisted=load_mesh(outer_path)
    error,ids=cKDTree(persisted.vertices).query(mesh.vertices)
    if error.max()>1e-5 or len(np.unique(ids))!=len(mesh.vertices): raise ValueError('候选 STL 重读顶点对应失效')
    ordered=persisted.vertices[ids]
    report['joint_recheck']['after']={s:_joint(ordered,s,surface) for s,surface in patches.items() if s in domains}
    report['metrics_after']['proximal']=report['joint_recheck']['after']
    if occl_mask is not None:
        report['joint_recheck']['occlusal_after']=measure(ordered,occl_mask,occlusal,parameters['occlusal_target_mm'],kind='OCCLUSAL')[0]
        report['metrics_after']['occlusal']=report['joint_recheck']['occlusal_after']
    report['candidate_readback_max_mm']=float(error.max())
    report['candidate_outer_sha256']=file_digest(outer_path)
    report['status']='CONSTRAINT_CONFLICT' if 'CONSTRAINT_CONFLICT' in required else 'PARTIAL_REVIEW_REQUIRED' if incomplete else 'PASS_ENGINEERING' if required and all(s=='PASS_ENGINEERING' for s in required) else 'REVIEW_REQUIRED'
    artifacts={'outer':str(outer_path),**diagnostic_artifacts}
    if report.get('proximal_asset'): artifacts['proximal']=report['proximal_asset']['path']
    if not blocked:
        final_path=prefix.with_name(prefix.name+'_w3_final.stl');cutter_path=prefix.with_name(prefix.name+'_w3_cutter.stl')
        try:
            from .contact_measurement import thickness_protection_surfaces
            protection=thickness_protection_surfaces(d) if parameters.get('enforce_minimum_thickness') else ()
            _,_,_,result=finalize_crown(outer_path,Path(support_ref['path']),final_path,cutter_path,margin,axis,parameters,prepared_support=prepared_support,support_contract=support_contract,protected_surfaces=protection)
            qc=result['w3']['qc'];report['w3_preflight']={'status':'CONSTRUCTED_ENGINEERING','qc':qc}
            audit_final_context(d,result['w3']['artifact_paths']['outer_surface'],final_path,qc)
            audit_path=Path(result['w3']['artifact_paths']['manufacturing_qc'])
            stored=json.loads(audit_path.read_text(encoding='utf-8'));stored['qc']=qc
            audit_path.write_text(json.dumps(stored,ensure_ascii=False),encoding='utf-8')
            report['thickness_audit_status']=qc['FinalCrownThicknessAudit']['status']
            if qc['FinalCrownThicknessAudit']['body_thin_face_sample_area_mm2']>0:
                report['status']='CONSTRAINT_CONFLICT';report['w3_preflight']['reason']='ENGINEERING_MINIMUM_THICKNESS_VIOLATED'
            artifacts.update(final=str(final_path),cutter=str(cutter_path),**result['w3']['artifact_paths'])
        except ValueError as exc:
            report['status']='CONSTRAINT_CONFLICT';report['w3_preflight']={'status':'BLOCKED','reason':str(exc)}
    else: report['w3_preflight']={'status':'BLOCKED','reason':'PREP_MARGIN_TOPOLOGY_UNSUPPORTED'}
    report['artifact_sha256']={key:file_digest(path) for key,path in artifacts.items()}
    return mesh,report,artifacts
