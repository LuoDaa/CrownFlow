"""Existing abutment crowns reuse the crown-bottom chain; never modify source assets."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np
import pytest
import trimesh
from app.geometry.abutment_support import extract_abutment_support,SupportExtractionError
from app.services.abutment_support import assess_abutment_surface

def fixture_support(hole=False):
    n=48;theta=np.arange(n)*2*np.pi/n
    margin=np.c_[3*np.cos(theta),3*np.sin(theta),np.zeros(n)]
    if not hole:return trimesh.creation.cylinder(radius=3,height=5,sections=n),margin
    mesh=trimesh.creation.annulus(r_min=1,r_max=3,height=5,sections=n)
    radial=mesh.triangles_center.copy();radial[:,2]=0;radial/=np.linalg.norm(radial,axis=1)[:,None]
    keep=np.sum(mesh.face_normals*radial,axis=1)>-.5
    return mesh.submesh([np.flatnonzero(keep)],append=True,repair=False),margin

def assess(mesh,margin,**kwargs):
    return assess_abutment_surface(mesh,margin,[0,0,1],margin_revision_id='confirmed-margin',axis_revision_id='confirmed-axis',
        patient_frame_revision_id='confirmed-frame',target_fdi=46,**kwargs)

@pytest.mark.parametrize('hole',[False,True])
def test_support_contract_and_virtual_channel_cap(hole,tmp_path):
    mesh,margin=fixture_support(hole);before=mesh.export(file_type='stl')
    contract,prepared=assess(mesh,margin,source_sha256=sha256(before).hexdigest())
    if hole:
        assert contract.rejection_code=='SCREW_CHANNEL_CONFIRMATION_REQUIRED'
        loops=[e['loop_id'] for e in contract.diagnostics['loops'] if e['classification']=='SCREW_CHANNEL_CANDIDATE']
        contract,prepared=assess(mesh,margin,source_sha256=sha256(before).hexdigest(),confirmed_loop_ids=loops)
        assert prepared.report['virtual_cap_operations'][0]['operation']=='VIRTUAL_SCREW_CHANNEL_CAP'
    assert contract.readiness=='ABUTMENT_SUPPORT_READY'
    assert contract.ready and contract.eligible
    assert prepared.report['boundary_loop_count']==1
    assert mesh.export(file_type='stl')==before
    source=tmp_path/'source.stl';source.write_bytes(before)
    outer=trimesh.creation.cylinder(radius=4.2,height=6,sections=64).apply_translation([0,0,1.5])
    outer_path=tmp_path/'outer.stl';outer.export(outer_path)
    from app.services.pipeline import finalize_crown
    params={'machine_axes':5,'cement_gap_mm':.06,'cement_gap_start_mm':0,'minimum_thickness_mm':.8,'margin_band_mm':.8}
    _,cavity,final,report=finalize_crown(outer_path,source,tmp_path/'final.stl',tmp_path/'cavity.stl',margin,[0,0,1],params,
        prepared_support=prepared,support_contract=contract)
    assert final.is_watertight
    reloaded=trimesh.load(tmp_path/'final.stl',force='mesh')
    assert reloaded.is_watertight
    assert sha256(source.read_bytes()).hexdigest()==contract.source_asset_sha256
    qc=report['w3']['qc']
    assert qc['SupportSurfaceAudit']['geometry']['conditioned_mesh_sha256']==sha256(Path(report['w3']['artifact_paths']['prepared_support']).read_bytes()).hexdigest()
    assert qc['InsertionAudit']['continuous_insertion_certified'] is False
    if hole:
        assert qc['IntaglioFitAudit']['virtual_cap_excluded_sample_count']>0
        assert qc['VirtualCapDiagnostics']['capped_region_area_mm2']>0
        assert qc['IntaglioFitAudit']['actual_gap']['n']<qc['FinalCrownThicknessAudit']['samples']
        hits=cavity.ray.intersects_location([[0,0,8]],[[0,0,-1]],multiple_hits=True)[0]
        assert len(hits)>=2 and hits[:,2].max()>2.5 # Cavity top covers the channel rather than reproducing its bore.

def test_unknown_hole_and_invalid_margin_are_explicitly_blocked():
    mesh,margin=fixture_support(True)
    v,f=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=.5)
    mesh=trimesh.Trimesh(v,f,process=True)
    faces=np.flatnonzero((mesh.triangles_center[:,2]>.5)&(mesh.triangles_center[:,2]<1.8)&(mesh.face_normals[:,0]>.8))
    assert len(faces)
    damaged=mesh.submesh([np.setdiff1d(np.arange(len(mesh.faces)),faces[:1])],append=True,repair=False)
    contract,_=assess(damaged,margin)
    assert not contract.ready
    assert contract.rejection_code=='UNKNOWN_SUPPORT_HOLE'
    contract,_=assess(mesh,margin+[0,0,9])
    assert contract.rejection_code=='MARGIN_NOT_ON_SUPPORT'
    nonmanifold=trimesh.Trimesh(mesh.vertices,np.vstack((mesh.faces,mesh.faces[:1])),process=False)
    with pytest.raises(SupportExtractionError,match='非流形'):extract_abutment_support(nonmanifold,margin,[0,0,1])

def test_custom_abutment_and_unconfirmed_margin_are_not_crown_inputs():
    mesh,margin=fixture_support()
    contract,_=assess(mesh,margin,indication='CUSTOM_IMPLANT_ABUTMENT')
    assert contract.readiness=='UNSUPPORTED_INDICATION'
    assert contract.rejection_code=='CUSTOM_ABUTMENT_NOT_SUPPORTED'
    contract,_=assess_abutment_surface(mesh,margin,[0,0,1],axis_revision_id='axis',patient_frame_revision_id='frame')
    assert contract.rejection_code=='MARGIN_NOT_CONFIRMED'
    from app.services.pipeline import finalize_crown
    with pytest.raises(ValueError,match='CUSTOM_ABUTMENT_NOT_SUPPORTED'):
        finalize_crown(None,None,None,None,margin,[0,0,1],{'indication':'CUSTOM_IMPLANT_ABUTMENT'})

def test_target_is_not_the_largest_component():
    mesh,margin=fixture_support()
    unrelated=trimesh.creation.icosphere(subdivisions=3,radius=10).apply_translation([25,0,10])
    combined=trimesh.util.concatenate((unrelated,mesh))
    contract,prepared=assess(combined,margin)
    assert contract.ready
    assert prepared.mesh.bounds[1,0]<=3.001
    assert len(prepared.report['source_face_ids'])<len(combined.faces)

def test_anatomical_template_placement_is_pinned_to_its_cervical_reference():
    from app.geometry.registration import initialize_from_margin
    from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
    template,ring=fixture_support()
    template.vertices[:,0]+=.4*template.vertices[:,2] # asymmetric anatomy must not shift the cervical footprint.
    target=ring+[10,20,30]
    frame=confirmed_patient_frame(target.mean(0),[1,0,0],[0,1,0],[0,0,1])
    result=initialize_from_margin(template,target,[0,0,1],template_cervical_points=ring,patient_frame=frame)
    section=result.section(plane_origin=target.mean(0),plane_normal=[0,0,1])
    boundary=max(section.discrete,key=len)
    assert np.linalg.norm((boundary.min(0)+boundary.max(0))[:2]/2-target.mean(0)[:2])<1e-6

@pytest.fixture
def api_client(isolated_db,monkeypatch):
    from app.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)

def create_confirmed_design(client,mesh,margin,axis=(0,0,1),case_id=None,fdi=46,parameters=None):
    case_id=case_id or client.post('/api/cases',json={'name':'existing-abutment-contract'}).json()['id']
    response=client.post(f'/api/cases/{case_id}/assets/support',files={'file':('actual_abutment.stl',mesh.export(file_type='stl'),'model/stl')})
    assert response.status_code==200,response.text
    design=client.post('/api/designs',json={'case_id':case_id,'target_fdi':fdi,'support_type':'abutment','indication':'CROWN_ON_EXISTING_ABUTMENT','draft':True}).json()['id']
    base=f'/api/designs/{design}'
    response=client.post(base+'/margin',json={'points':np.asarray(margin).tolist()});assert response.status_code==200,response.text
    response=client.post(base+'/axis',json={'axis':list(axis)});assert response.status_code==200,response.text
    response=client.put(base+'/profile',json={'material':'engineering fixture','bite_confirmed':True,'confirmed':True,'coverage':'ENGINEERING_ONLY','parameters':{'machine_axes':5,'cement_gap_mm':.06,'cement_gap_start_mm':0,**(parameters or {})}})
    assert response.status_code==200,response.text
    z=np.asarray(axis)/np.linalg.norm(axis);x=np.array([1.,0,0]);x-=z*np.dot(x,z);x/=np.linalg.norm(x);y=np.cross(z,x)
    digest=client.get(base+'/status').json()['input_digest']
    response=client.put(base+'/auto-fit/frame',json={'expected_input_digest':digest,'confirmed':True,'mesial':x.tolist(),'buccal':y.tolist(),'units':'mm'})
    assert response.status_code==200,response.text
    return design

def test_screw_confirmation_api_and_stale_confirmation(api_client):
    client=api_client;mesh,margin=fixture_support(True);design=create_confirmed_design(client,mesh,margin)
    base=f'/api/designs/{design}'
    report=client.get(base+'/prep-surface-contract').json()
    assert report['classification']['rejection_code']=='SCREW_CHANNEL_CONFIRMATION_REQUIRED'
    loop_ids=[loop['loop_id'] for loop in report['diagnostics']['loops'] if loop['classification']=='SCREW_CHANNEL_CANDIDATE']
    digest=client.get(base+'/status').json()['input_digest']
    wrong=client.post(base+'/abutment-support/confirm',json={'expected_input_digest':digest,'confirmed_loop_ids':['unknown']})
    assert wrong.status_code==422
    accepted=client.post(base+'/abutment-support/confirm',json={'expected_input_digest':digest,'confirmed_loop_ids':loop_ids})
    assert accepted.status_code==200,accepted.text
    assert client.get(base+'/prep-surface-contract').json()['classification']['readiness']=='ABUTMENT_SUPPORT_READY'
    before=client.get(base).json()['support_conditioning_json']
    # Reconfirming identical geometry creates a new truth revision; the old cap declaration is stale.
    assert client.post(base+'/margin',json={'points':margin.tolist()}).status_code==200
    assert client.get(base).json()['support_conditioning_json']==before
    report=client.get(base+'/prep-surface-contract').json()
    assert report['classification']['readiness']=='REVIEW_REQUIRED'
    assert report['provenance']['support_confirmation_revision_id'] is None

def test_shoulder_candidate_is_not_support_truth(api_client):
    from app.revisions import record_auto_margin_candidate
    client=api_client;mesh,margin=fixture_support();design=create_confirmed_design(client,mesh,margin)
    base=f'/api/designs/{design}'
    candidate=margin+[0,0,1.4]
    record_auto_margin_candidate(design,candidate.tolist(),algorithm_version='shoulder-test-candidate')
    chosen=margin+[0,0,.3]
    assert client.post(base+'/margin',json={'points':chosen.tolist()}).status_code==200
    digest=client.get(base+'/status').json()['input_digest']
    frame=client.put(base+'/auto-fit/frame',json={'expected_input_digest':digest,'confirmed':True,'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'})
    assert frame.status_code==200,frame.text
    contract=client.get(base+'/prep-surface-contract').json()
    assert contract['classification']['readiness']=='ABUTMENT_SUPPORT_READY'
    assert contract['geometry']['margin_revision_id']==client.get(base).json()['confirmed_margin_revision_id']
    crown=next(e for e in contract['diagnostics']['loops'] if e['classification']=='CROWN_MARGIN')
    assert np.max(np.abs(np.asarray(crown['points'])[:,2]-.3))<1e-7
    assert crown['confirmed']

def test_support_bytes_change_invalidates_axis_and_cap_confirmation(api_client):
    client=api_client;mesh,margin=fixture_support(True);design=create_confirmed_design(client,mesh,margin)
    base=f'/api/designs/{design}';contract=client.get(base+'/prep-surface-contract').json()
    ids=[e['loop_id'] for e in contract['diagnostics']['loops'] if e['classification']=='SCREW_CHANNEL_CANDIDATE']
    digest=client.get(base+'/status').json()['input_digest']
    assert client.post(base+'/abutment-support/confirm',json={'expected_input_digest':digest,'confirmed_loop_ids':ids}).status_code==200
    digest=client.get(base+'/status').json()['input_digest']
    case_id=client.get(base).json()['case_id'];source=Path(client.get('/api/cases/'+case_id).json()['assets'][-1]['path'])
    before=source.read_bytes();source.write_bytes(before+b'changed-test-asset')
    assert client.get(base+'/status').json()['input_digest']!=digest
    contract=client.get(base+'/prep-surface-contract').json()
    assert not contract['classification']['readiness']=='ABUTMENT_SUPPORT_READY'
    assert contract['provenance']['support_confirmation_revision_id'] is None

def test_real_abutment_upload_to_final_stl(api_client,tmp_path):
    run_real_abutment(api_client,tmp_path)


def run_real_abutment(api_client,tmp_path,*,manufacturing_border=False):
    from app.geometry.io import load_mesh
    from app.geometry.implant_candidate import implant_candidates
    from app.geometry.draft_margin import draft_margin
    root=Path(r'D:\code\牙冠自动生成\1.0测试数据\基台测试1-46')
    if not (root/'world_result.stl').is_file():pytest.skip('local real abutment test files unavailable')
    original=sha256((root/'world_result.stl').read_bytes()).hexdigest()
    upper=load_mesh(root/'upperjaw_upper.stl');jaw=load_mesh(root/'world_result.stl')
    support,candidate=implant_candidates(jaw,upper)[0];axis=np.asarray(candidate['axis'])
    z=(support.vertices-support.centroid)@axis;seed=support.vertices[z>=np.percentile(z,98)].mean(0)
    margin,_=draft_margin(support,seed,axis,8)
    client=api_client;case=client.post('/api/cases',json={'name':'real-abutment-test1-46'}).json()['id']
    for kind,name in (('upper','upperjaw_upper.stl'),('lower','world_result.stl')):
        with (root/name).open('rb') as source:
            response=client.post(f'/api/cases/{case}/assets/{kind}',files={'file':(name,source,'model/stl')})
        assert response.status_code==200,response.text
    parameters={'margin_band_mm':2.4}
    if manufacturing_border:parameters.update(border_horizontal_mm=.25,border_angled_mm=.6,border_angle_deg=45,border_vertical_mm=.1,enforce_minimum_thickness=True)
    design=create_confirmed_design(client,support,margin,axis,case,parameters=parameters)
    base=f'/api/designs/{design}'
    contract=client.get(base+'/prep-surface-contract').json()
    assert contract['classification']['readiness']=='ABUTMENT_SUPPORT_READY',contract
    status=client.get(base+'/status').json()
    proposal=client.post(base+'/auto-fit/template-preview',json={'library':'炼金王','confirmed':True,'expected_input_digest':status['input_digest']})
    assert proposal.status_code==200,proposal.text
    status=client.get(base+'/status').json()
    response=client.post(base+f"/auto-fit/{proposal.json()['id']}/accept",json={'expected_input_digest':status['input_digest'],'confirmed':True,'confirmed_template_anatomy':True})
    assert response.status_code==200,response.text
    # Explicit traditional CAD review step: broaden this real template around the unusually tall support.
    outer_record=client.get(base).json()
    response=client.post(base+'/edit-outer',json={'operation':'transform','scale':[1.15,1.15,1],
        'base_outer_path':outer_record['outer_path']})
    assert response.status_code==200,response.text
    status=client.get(base+'/status').json()
    final=client.post(base+'/finalize',json={'expected_input_digest':status['input_digest'],**status['confirmed_revisions']})
    assert final.status_code==200,final.text
    result=final.json();record=client.get(base).json();mesh=load_mesh(Path(record['final_path']))
    assert mesh.is_watertight
    assert result['provenance']['indication']=='CROWN_ON_EXISTING_ABUTMENT'
    paths=result['qc']['artifact_paths']
    expected=result['provenance']['prepared_support']['geometry']['conditioned_mesh_sha256']
    assert sha256(Path(paths['prepared_support']).read_bytes()).hexdigest()==expected
    assert client.get(base+'/final.stl').status_code==200
    assert sha256((root/'world_result.stl').read_bytes()).hexdigest()==original
    # Durable review artifacts only; the actual case database and scans are never changed.
    import shutil
    output=Path(__file__).resolve().parents[3]/'output'/('crown-bottom-review' if manufacturing_border else 'abutment-review')/'real-test1-46'
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(record['final_path'],output/'final.stl')
    shutil.copy2(paths['manufacturing_qc'],output/'manufacturing_qc.json')
    shutil.copy2(paths['prepared_support'],output/'prepared_support.stl')
    (output/'result.json').write_text(json.dumps({'source_case':'基台测试1-46','source_scan_sha256':original,'confirmed_margin':margin,
        'component_role':'EXISTING_ABUTMENT_TEST_DECLARATION_NOT_CLINICAL_IDENTITY_VALIDATION',
        'provenance':result['provenance'],'qc':result['qc']['manufacturing_qc'],'stl_reload_watertight':True,
        'outer_review_adjustment':{'operation':'transform','scale':[1.15,1.15,1],'cervical_band_locked':True},
        'steps':['UPLOAD','MARGIN_CONFIRM','AXIS_CONFIRM','PATIENT_FRAME_CONFIRM','SUPPORT_EXTRACTION','HOLE_CLASSIFICATION','OUTER_TEMPLATE','MANUAL_OUTER_ADJUSTMENT','INTAGLIO','BOOLEAN','SEAM','QC','STL_RELOAD']},ensure_ascii=False,indent=2),encoding='utf-8')
    if manufacturing_border:
        qc=result['qc']['manufacturing_qc']
        assert qc['CrownBorderAudit']['horizontal_mm']==.25
        assert qc['MinimumThicknessEnforcement']['intaglio_preserved']
        assert qc['FinalCrownThicknessAudit']['body_thin_face_sample_area_mm2']<=qc['FinalCrownThicknessAudit']['thin_face_sample_area_mm2']
        assert qc['production_status']=='NOT_PRODUCTION_APPROVED'
    return result
