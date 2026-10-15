import json
from pathlib import Path
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from app.geometry.crown_outer.auto_fit_adapter import FitSurface,fit_stage
from app.geometry.crown_outer.patient_frame import PatientFrame


def geometry():
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=3).apply_translation([0,0,3])
    theta=np.arange(32)*np.pi/16
    margin=np.c_[2*np.cos(theta),2*np.sin(theta),np.zeros(32)]
    frame=PatientFrame(np.zeros(3),np.array([0,0,1.]),np.array([1.,0,0]),np.array([0,1.,0]))
    neighbor=trimesh.creation.icosphere(subdivisions=2,radius=2).apply_translation([4.6,0,3])
    return mesh,margin,frame,neighbor


def test_single_side_local_fit_is_immutable_and_locks_margin():
    mesh,margin,frame,neighbor=geometry();reference=mesh.vertices.copy()
    candidate,report,masks=fit_stage(mesh,frame,margin,{'mesial':FitSurface(neighbor)},{'mesial':.05},'PROXIMAL')
    assert np.array_equal(mesh.vertices,reference)
    assert report['status']=='PASS_ENGINEERING',report
    assert report['after']['mesial']['target_error']['p95_mm']<report['before']['mesial']['target_error']['p95_mm']
    assert report['locked_displacement_mm']==0
    assert candidate.is_watertight


def test_open_edge_is_ambiguous_not_fake_penetration():
    mesh=trimesh.Trimesh([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],[[0,1,2],[0,2,3]],process=False)
    surface=FitSurface(mesh,[0,0,1])
    _,_,_,_,valid=surface.query(np.array([[1,1,-.1],[0,0,-.1]]))
    assert not valid[0] and valid[1]
    with pytest.raises(ValueError): FitSurface(mesh)


def test_static_occlusal_fit_reduces_deficit_and_preserves_neck():
    mesh,margin,frame,_=geometry()
    antagonist=trimesh.creation.box(extents=[20,20,2]).apply_translation([0,0,6.5])
    reference=mesh.vertices.copy()
    candidate,report,_=fit_stage(mesh,frame,margin,{'occlusal':FitSurface(antagonist)},{'occlusal':.05},'OCCLUSAL')
    assert report['status']=='PASS_ENGINEERING',report
    assert report['after']['occlusal']['deficit']['p95_mm']<report['before']['occlusal']['deficit']['p95_mm']
    assert np.array_equal(mesh.vertices,reference)
    assert report['locked_displacement_mm']==0


@pytest.fixture
def ready(isolated_db,monkeypatch):
    from app.api import routes,auto_fit
    from app.services import auto_fit as service,design_dependencies
    for module in (routes,auto_fit,service): monkeypatch.setattr(module,'CASE_DIR',isolated_db/'cases')
    monkeypatch.setattr(routes,'TEMPLATE_DIR',isolated_db/'templates')
    monkeypatch.setattr(design_dependencies,'TEMPLATE_DIR',isolated_db/'templates')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'synthetic-autofit'}).json()['id']
    prep=trimesh.creation.cone(radius=2,height=4,sections=32)
    neighbors=trimesh.util.concatenate([trimesh.creation.icosphere(subdivisions=1,radius=2).apply_translation([5,0,2]),trimesh.creation.icosphere(subdivisions=1,radius=2).apply_translation([-5,0,2])])
    antagonist=trimesh.creation.box(extents=[20,20,2]).apply_translation([0,0,7])
    for kind,mesh in [('support',prep),('lower',neighbors),('upper',antagonist)]:
        assert client.post(f'/api/cases/{case}/assets/{kind}',files={'file':(kind+'.stl',mesh.export(file_type='stl'),'model/stl')}).status_code==200
    d=client.post('/api/designs',json={'draft':True,'case_id':case,'target_fdi':46,'support_type':'prep'}).json()['id'];path=f'/api/designs/{d}'
    theta=np.arange(32)*np.pi/16;margin=np.c_[2*np.cos(theta),2*np.sin(theta),np.zeros(32)].tolist()
    assert client.post(path+'/margin',json={'points':margin}).status_code==200
    assert client.post(path+'/axis',json={'axis':[0,0,1]}).status_code==200
    assert client.put(path+'/profile',json={'material':'unvalidated fixture','coverage':'ENGINEERING_ONLY','bite_confirmed':True,'confirmed':True,
        'parameters':{'machine_axes':5,'minimum_thickness_mm':.2,'proximal_target_mm':.05,'occlusal_target_mm':.05}}).status_code==200
    outer=trimesh.creation.cylinder(radius=3,height=6,sections=32).apply_translation([0,0,2])
    outer.export(isolated_db/'templates'/'FDI_46.stl')
    # A1：live 初稿顺序与 exocad 一致 —— 先确认坐标帧，再做模板放置；
    # 没有确认帧时 /outer/auto 会 409，不再用世界轴猜横向方向。
    assert confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'}).status_code==200
    assert client.post(path+'/outer/auto',json={'margin_points':margin,'insertion_axis':[0,0,1]}).status_code==200
    return client,path,d,neighbors


def confirm(client,path,subpath,data):
    request={**data,'expected_input_digest':client.get(path+'/status').json()['input_digest'],'confirmed':True}
    return client.put(path+'/auto-fit/'+subpath,json=request)


def prepare(client,path,neighbors):
    r=confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'});assert r.status_code==200,r.text
    r=confirm(client,path,'neighbors/mesial',{'status':'CONFIRMED','source':'lower','instance_name':'actual face mask',
        'face_ids':list(range(len(neighbors.faces)//2)),'pollution_reviewed':True});assert r.status_code==200,r.text
    r=confirm(client,path,'neighbors/distal',{'status':'ABSENT','source':'lower','instance_name':'missing neighbor'});assert r.status_code==200,r.text


def request(client,path,mode='PROXIMAL_ONLY'):
    status=client.get(path+'/status').json()
    return {'mode':mode,'expected_input_digest':status['input_digest'],**status['confirmed_revisions'],'dry_run':True}


def fake_propose(d,frame,ref,mode):
    from app.services.design_dependencies import file_digest
    path=Path(d['outer_path']).with_name('candidate.stl');path.write_bytes(Path(d['outer_path']).read_bytes())
    return trimesh.load(path,force='mesh'),{'status':'PASS_ENGINEERING','candidate_outer_sha256':file_digest(path),'artifact_sha256':{'outer':file_digest(path)}},{'outer':str(path)}


def test_proposal_then_atomic_accept_and_reject_semantics(ready,monkeypatch):
    client,path,d,neighbors=ready;prepare(client,path,neighbors)
    from app.services import auto_fit as service
    monkeypatch.setattr(service,'propose',fake_propose)
    before=client.get(path).json()
    r=client.post(path+'/auto-fit/preview',json=request(client,path));assert r.status_code==200,r.text
    candidate=r.json();assert client.get(path).json()==before
    accepted=client.post(path+f"/auto-fit/{candidate['id']}/accept",json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']})
    assert accepted.status_code==200,accepted.text
    after=client.get(path).json();assert after['design_revision']==before['design_revision']+1
    assert after['outer_path']!=before['outer_path'] and after['final_path'] is None
    revisions=client.get(path+'/revisions').json()['revisions']
    assert revisions[-1]['kind']=='outer' and revisions[-1]['payload']['proposal_id']==candidate['id']
    assert client.post(path+f"/auto-fit/{candidate['id']}/reject").status_code==409


def test_stale_tampered_and_rejected_candidates_cannot_be_accepted(ready,monkeypatch):
    client,path,d,neighbors=ready;prepare(client,path,neighbors)
    from app.services import auto_fit as service
    monkeypatch.setattr(service,'propose',fake_propose)
    r=client.post(path+'/auto-fit/preview',json=request(client,path)).json();prefix=path+f"/auto-fit/{r['id']}"
    assert client.post(prefix+'/reject').status_code==200
    assert client.post(prefix+'/accept',json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']}).status_code==409
    r=client.post(path+'/auto-fit/preview',json=request(client,path)).json();prefix=path+f"/auto-fit/{r['id']}"
    Path(r['artifacts']['outer']).write_bytes(b'tampered')
    assert client.get(prefix+'/candidate.stl').status_code==409
    assert client.post(prefix+'/accept',json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']}).status_code==409
    confirm(client,path,'neighbors/distal',{'status':'UNKNOWN','source':'lower','instance_name':'withdrawn'})
    assert client.post(prefix+'/accept',json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']}).status_code==409


def test_old_radius_selection_and_manual_view_do_not_grant_inputs(ready,monkeypatch):
    client,path,d,neighbors=ready
    assert confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'}).status_code==200
    from app.services import auto_fit as service
    called=[]
    def noop_final(outer,support,final,cutter,*args,**kwargs):
        called.append(True)
        final.write_bytes(Path(outer).read_bytes());cutter.write_bytes(Path(support).read_bytes())
        audit=final.with_suffix('.json');audit.write_text(json.dumps({'qc':{}}))
        return None,None,None,{'w3':{'qc':{'FinalCrownThicknessAudit':{'status':'REVIEW_REQUIRED','thin_face_sample_area_mm2':0}},'artifact_paths':{'outer_surface':str(outer),'manufacturing_qc':str(audit)}}}
    # Isolate input-state behavior, not W3 construction; full W3 has separate tests.
    monkeypatch.setattr(service,'finalize_crown',noop_final)
    from app.db import db
    with db() as c: c.execute('UPDATE designs SET bite_json=? WHERE id=?',(json.dumps({'viewed':True,'pose_confirmed':True}),d))
    r=client.post(path+'/auto-fit/preview',json=request(client,path,'BOTH'))
    assert r.status_code==200,r.text
    assert r.json()['payload']['status']=='PARTIAL_REVIEW_REQUIRED'
    assert r.json()['payload']['stages'][1]['reason']=='OCCLUSION_REGISTRATION_UNVERIFIED'
    assert client.post(path+f"/auto-fit/{r.json()['id']}/accept",json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']}).status_code==409


def test_bad_frame_bite_and_neighbor_side_are_rejected(ready):
    client,path,d,neighbors=ready
    assert confirm(client,path,'frame',{'mesial':[0,0,1],'buccal':[0,1,0],'units':'mm'}).status_code==422
    prepare(client,path,neighbors)
    assert confirm(client,path,'neighbors/distal',{'status':'CONFIRMED','source':'lower','instance_name':'wrong side',
        'face_ids':list(range(len(neighbors.faces)//2)),'pollution_reviewed':True}).status_code==422
    assert confirm(client,path,'bite',{'transform':np.diag([-1,1,1,1]).tolist(),'source_method':'REGISTERED_EXPORT','units':'mm'}).status_code==422
    assert confirm(client,path,'bite',{'transform':np.eye(4).tolist(),'source_method':'REGISTERED_EXPORT','units':'mm'}).status_code==200


def test_actual_auto_fit_uses_w3_and_keeps_accepted_outer(ready):
    client,path,d,neighbors=ready;prepare(client,path,neighbors)
    assert confirm(client,path,'bite',{'transform':np.eye(4).tolist(),'source_method':'REGISTERED_EXPORT','units':'mm'}).status_code==200
    before=client.get(path).json()
    result=client.post(path+'/auto-fit/preview',json=request(client,path,'BOTH'))
    assert result.status_code==200,result.text
    r=result.json();report=r['payload']
    assert client.get(path).json()==before
    assert report['per_side']['distal']['status']=='SKIPPED_ABSENT'
    assert len(report['stages'])==2
    assert report['w3_preflight']['status']=='CONSTRUCTED_ENGINEERING',report
    assert report['w3_preflight']['qc']['NonplanarMarginBuild']['exported_stl_seam']['margin_to_seam_max_mm']<1e-5
    assert report['w3_preflight']['qc']['MaterialProcessQC']['status']=='NOT_ASSESSED'
    assert report['w3_preflight']['qc']['ContactQC']['per_side']['mesial']['valid_count']>0
    assert report['w3_preflight']['qc']['OcclusionQC']['bite_revision_id']
    diagnostic=client.get(path+f"/auto-fit/{r['id']}/artifact/input_diagnostics")
    assert diagnostic.status_code==200
    sources=diagnostic.json()['cropped_face_source_ids']
    assert sources and min(sources)>=0
    if report['status']!='PASS_ENGINEERING':
        assert client.post(path+f"/auto-fit/{r['id']}/accept",json={'confirmed':True,'expected_input_digest':request(client,path)['expected_input_digest']}).status_code==409


def test_proposal_publication_rejects_concurrent_aba(ready,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from app.db import db
    from app.services import auto_fit as service
    client,path,d,neighbors=ready;prepare(client,path,neighbors)
    started=Event();resume=Event()
    def paused(*args):
        started.set();assert resume.wait(10);return fake_propose(*args)
    monkeypatch.setattr(service,'propose',paused)
    baseline=client.get(path).json();req=request(client,path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(client.post,path+'/auto-fit/preview',json=req)
        assert started.wait(10)
        with db() as c: c.execute('UPDATE designs SET neighbors_json=? WHERE id=?',('{}',d))
        with db() as c: c.execute('UPDATE designs SET neighbors_json=? WHERE id=?',(baseline['neighbors_json'],d))
        resume.set();r=future.result(timeout=10)
    assert r.status_code==409
    assert not any(r['kind']=='outer_autofit_candidate' for r in client.get(path+'/revisions').json()['revisions'])


@pytest.mark.parametrize('rotation',[0,1.3])
def test_local_fit_is_rotation_equivariant(rotation):
    mesh,margin,frame,neighbor=geometry()
    reference,_,_=fit_stage(mesh,frame,margin,{'mesial':FitSurface(neighbor)},{'mesial':.05},'PROXIMAL')
    transform=trimesh.transformations.rotation_matrix(rotation,[.2,.4,1])
    transform[:3,3]=[5,-2,1]
    m=mesh.copy().apply_transform(transform);n=neighbor.copy().apply_transform(transform)
    rotated=PatientFrame(transform[:3,3],transform[:3,:3]@frame.z,transform[:3,:3]@frame.x,transform[:3,:3]@frame.y)
    cand,rep,_=fit_stage(m,rotated,trimesh.transform_points(margin,transform),{'mesial':FitSurface(n)},{'mesial':.05},'PROXIMAL')
    assert rep['status']=='PASS_ENGINEERING'
    assert np.allclose(cand.vertices,trimesh.transform_points(reference.vertices,transform),atol=1e-6)


def test_explicit_nonplanar_template_cuff_avoids_lowest_vertex_sliver():
    from app.geometry.registration import initialize_from_margin
    mesh=trimesh.creation.cylinder(radius=3,height=5,sections=32)
    t=np.arange(32)*np.pi/16;cuff=np.c_[3*np.cos(t),3*np.sin(t),.5*np.sin(2*t)]
    crown=initialize_from_margin(mesh,cuff,[0,0,1],template_cervical_points=cuff)
    assert crown.is_watertight


@pytest.mark.parametrize('transform',[np.diag([-1.,1,1,1]),np.diag([1.,-1,-1,1])])
def test_mirror_and_reversed_jaw_axis_preserve_local_result(transform):
    mesh,margin,frame,neighbor=geometry()
    reference,_,_=fit_stage(mesh,frame,margin,{'mesial':FitSurface(neighbor)},{'mesial':.05},'PROXIMAL')
    m=mesh.copy().apply_transform(transform);n=neighbor.copy().apply_transform(transform)
    rotated=PatientFrame(np.zeros(3),transform[:3,:3]@frame.z,transform[:3,:3]@frame.x,transform[:3,:3]@frame.y)
    candidate,report,_=fit_stage(m,rotated,trimesh.transform_points(margin,transform),{'mesial':FitSurface(n)},{'mesial':.05},'PROXIMAL')
    assert report['status']=='PASS_ENGINEERING'
    assert np.allclose(candidate.vertices,trimesh.transform_points(reference.vertices,transform),atol=1e-6)


def test_far_neighbor_does_not_claim_no_correction_needed():
    mesh,margin,frame,neighbor=geometry();neighbor.apply_translation([20,0,0])
    candidate,report,_=fit_stage(mesh,frame,margin,{'mesial':FitSurface(neighbor)},{'mesial':.05},'PROXIMAL')
    assert report['status']=='CONSTRAINT_CONFLICT'
    assert report['reason']=='TARGET_OUTSIDE_LOCAL_REACH'
    assert np.array_equal(candidate.vertices,mesh.vertices)


def test_crossing_audit_counts_faces_without_using_vertex_distance():
    from app.geometry.crown_outer.auto_fit_adapter import surface_crossings
    a=trimesh.Trimesh([[-1,-1,0],[1,-1,0],[0,1,0]],[[0,1,2]],process=False)
    b=trimesh.Trimesh([[0,-.5,-1],[0,-.5,1],[0,.5,0]],[[0,1,2]],process=False)
    report=surface_crossings(a,FitSurface(b,[1,0,0]),np.ones(3,bool))
    assert report['crossing_pairs']==1 and report['intersection_segment_total_mm']>0
    b.apply_translation([10,0,0])
    assert surface_crossings(a,FitSurface(b,[1,0,0]),np.ones(3,bool))['crossing_pairs']==0


def test_clip_source_ids_remain_aligned_after_nonplanar_cut():
    from app.geometry.prep_margin_diagnostics import diagnose
    mesh=trimesh.creation.cylinder(radius=2,height=4,sections=32)
    theta=np.arange(32)*np.pi/16;margin=np.c_[2*np.cos(theta),2*np.sin(theta),.4*np.sin(2*theta)]
    report,crop=diagnose(mesh,margin,[0,0,1])
    ids=np.asarray(report['cropped_face_source_ids'])
    assert len(ids)==len(crop.faces) and ids.min()>=0 and ids.max()<len(mesh.faces)
    assert report['codes']==[]


def test_real_library_template_candidate_requires_anatomy_review(ready):
    from app.config import TOOTH_LIBRARY_ROOT
    if not TOOTH_LIBRARY_ROOT.is_dir(): pytest.skip('external professional tooth library unavailable')
    client,path,d,neighbors=ready
    theta=np.arange(32)*np.pi/16
    margin=np.c_[4*np.cos(theta),4*np.sin(theta),.3*np.sin(2*theta)]
    assert client.post(path+'/margin',json={'points':margin.tolist()}).status_code==200
    assert confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'}).status_code==200
    before=client.get(path).json()
    response=client.post(path+'/auto-fit/template-preview',json={'library':'洋溢-长夜同在',
        'confirmed':True,'expected_input_digest':client.get(path+'/status').json()['input_digest']})
    assert response.status_code==200,response.text
    result=response.json();assert client.get(path).json()==before
    assert result['payload']['metadata_sha256']
    accept={'confirmed':True,'expected_input_digest':client.get(path+'/status').json()['input_digest']}
    assert client.post(path+f"/auto-fit/{result['id']}/accept",json=accept).status_code==409
    accept['confirmed_template_anatomy']=True
    assert client.post(path+f"/auto-fit/{result['id']}/accept",json=accept).status_code==200
    selected=json.loads(client.get(path).json()['landmarks_json'])['_selected_template']
    assert selected['metadata_sha256']==result['payload']['metadata_sha256']


def test_selecting_reference_tooth_does_not_publish_outer(ready):
    from app.config import TOOTH_LIBRARY_ROOT
    if not TOOTH_LIBRARY_ROOT.is_dir(): pytest.skip('external professional tooth library unavailable')
    client,path,_,_=ready
    theta=np.arange(32)*np.pi/16
    margin=np.c_[4*np.cos(theta),4*np.sin(theta),.3*np.sin(2*theta)]
    assert client.post(path+'/margin',json={'points':margin.tolist()}).status_code==200
    assert confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'}).status_code==200
    digest=client.get(path+'/status').json()['input_digest']
    preview=client.post(path+'/auto-fit/template-preview',json={'library':'洋溢-长夜同在',
        'confirmed':True,'expected_input_digest':digest})
    assert preview.status_code==200,preview.text
    response=client.post(path+f"/auto-fit/{preview.json()['id']}/select-template",
        json={'confirmed':True,'expected_input_digest':digest})
    assert response.status_code==200,response.text
    design=client.get(path).json()
    assert design['outer_path'] is None
    assert not client.get(path+'/status').json()['outer_current']
    assert json.loads(design['landmarks_json'])['_selected_template']['library']=='洋溢-长夜同在'
