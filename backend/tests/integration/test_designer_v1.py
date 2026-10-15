"""1.0 edit/save/rebuild regression, isolated from real cases."""
import json
import numpy as np
import trimesh
from fastapi.testclient import TestClient
from app.db import db
from app.api.designer import Edit,edit_mesh


def test_margin_save_requires_ordered_loop_and_preserves_legacy_read(isolated_db):
    from app.main import app
    from app.revisions import record_auto_margin_candidate,record_manual_margin
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'margin-contract'}).json()['id']
    design=client.post('/api/designs',json={'case_id':case,'target_fdi':46,'support_type':'prep','draft':True}).json()['id']
    path=f'/api/designs/{design}'
    theta=np.linspace(0,2*np.pi,12,endpoint=False)
    ring=np.c_[2*np.cos(theta),2*np.sin(theta),np.zeros(12)].tolist()
    small=np.c_[.2*np.cos(theta),.2*np.sin(theta),np.zeros(12)].tolist()
    crossing=[ring[(i*5)%12] for i in range(12)]
    for points in (ring[:3],[[i*.2,0,0] for i in range(12)],small,
                   ring[:6]+[ring[0]]+ring[7:],crossing):
        response=client.post(path+'/margin',json={'points':points})
        assert response.status_code==400,response.text
    accepted=client.post(path+'/margin',json={'points':ring+[ring[0]]})
    assert accepted.status_code==200,accepted.text
    assert len(accepted.json()['points'])==12
    old_triangle=[[0,0,0],[2,0,0],[0,2,0]]
    record_manual_margin(design,old_triangle)
    before=client.get(path+'/revisions').json()['n_revisions']
    assert json.loads(client.get(path).json()['margin_json'])==old_triangle
    assert client.post(path+'/margin',json={'points':old_triangle}).status_code==400
    assert client.get(path+'/revisions').json()['n_revisions']==before
    candidate,applied=record_auto_margin_candidate(design,old_triangle)
    assert not applied
    before=client.get(path+'/revisions').json()['n_revisions']
    assert client.post(path+f'/revisions/{candidate.revision_no}/accept').status_code==400
    assert json.loads(client.get(path).json()['margin_json'])==old_triangle
    assert client.get(path+'/revisions').json()['n_revisions']==before


def test_edit_protects_margin_and_rejects_invalid_transform():
    m=trimesh.creation.icosphere(subdivisions=2,radius=4)
    m.apply_translation([0,0,3])
    margin=[[3,0,0],[0,3,0],[-3,0,0],[0,-3,0]]
    p=Edit(operation='transform',scale=[1.03,1,1],base_outer_path='test')
    edited=edit_mesh(m,margin,[0,0,1],p)
    locked=m.vertices[:,2]<=.4
    assert np.array_equal(edited.vertices[locked],m.vertices[locked])
    assert np.max(np.linalg.norm(edited.vertices-m.vertices,axis=1))>0
    assert edited.is_watertight
    import pytest
    with pytest.raises(ValueError): edit_mesh(m,margin,[0,0,1],p.model_copy(update={'scale':[0,1,1]}))


def test_draft_profile_and_late_support(isolated_db,monkeypatch):
    from app.api import routes
    monkeypatch.setattr(routes,'CASE_DIR',isolated_db/'cases')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'draft-regression'}).json()['id']
    d=client.post('/api/designs',json={'case_id':case,'target_fdi':46,'support_type':'prep','draft':True}).json()['id']
    assert client.post(f'/api/designs/{d}/rough-margin',json={'seeds':[[0,0,0]]*3}).status_code==422
    assert client.post(f'/api/designs/{d}/outer/auto',json={'margin_points':[[0,0,0],[1,0,0],[0,1,0]],'insertion_axis':[0,0,1]}).status_code==409
    assert client.post(f'/api/designs/{d}/suggest-margin').status_code==400
    profile={'material':'test','coverage':'FULL_ARCH','bite_confirmed':True,'confirmed':True,'parameters':{'machine_axes':5}}
    assert client.put(f'/api/designs/{d}/profile',json=profile).status_code==400
    profile['coverage']='ENGINEERING_ONLY'
    assert client.put(f'/api/designs/{d}/profile',json={**profile,'parameters':{'machine_axes':5,'cement_gap_mm':9}}).status_code==422
    assert client.put(f'/api/designs/{d}/profile',json=profile).status_code==200
    mesh=trimesh.creation.cone(radius=3,height=5)
    assert client.post(f'/api/cases/{case}/assets/support',files={'file':('prep.stl',mesh.export(file_type='stl'),'model/stl')}).status_code==200
    angles=np.linspace(0,2*np.pi,12,endpoint=False)
    seeds=np.c_[1.8*np.cos(angles),1.8*np.sin(angles),np.full(12,2)].tolist()
    rough=client.post(f'/api/designs/{d}/rough-margin',json={'seeds':seeds,'center_hint':[0,0,2],'guide_axis':[0,0,1]})
    assert rough.status_code==200,rough.text
    assert len(rough.json()['points'])==64
    front=np.c_[1.8*np.cos(angles),np.full(12,-2),2+1.8*np.sin(angles)].tolist()
    assert client.post(f'/api/designs/{d}/rough-margin',json={'seeds':front,'center_hint':[0,0,2],'guide_axis':[0,0,1]}).status_code==400
    assert client.post(f'/api/designs/{d}/rough-margin',json={'seeds':seeds,'center_hint':[10,0,2],'guide_axis':[0,0,1]}).status_code==400
    crossing=[seeds[(i*5)%12] for i in range(12)]
    assert client.post(f'/api/designs/{d}/rough-margin',json={'seeds':crossing,'center_hint':[0,0,2],'guide_axis':[0,0,1]}).status_code==400
    assert client.get(f'/api/designs/{d}/status').json()['release_status']=='NOT_PRODUCTION_APPROVED'


def test_unconfirmed_auto_margin_can_relocate_prep(isolated_db,monkeypatch):
    from app.api import routes
    monkeypatch.setattr(routes,'CASE_DIR',isolated_db/'cases')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'relocate-prep'}).json()['id']
    teeth=[]
    for x in (-7,7):
        tooth=trimesh.creation.cylinder(radius=2,height=7,sections=64)
        tooth.apply_translation([x,0,0]);teeth.append(tooth)
    jaw=trimesh.util.concatenate(teeth)
    assert client.post(f'/api/cases/{case}/assets/lower',files={'file':('jaw.stl',jaw.export(file_type='stl'),'model/stl')}).status_code==200
    design=client.post('/api/designs',json={'case_id':case,'target_fdi':46,'support_type':'prep','draft':True}).json()['id']
    for x in (-7,7):
        result=client.post(f'/api/designs/{design}/draft-preparation',json={'center':[x,0,3.5],'axis':[0,0,1],'radius_mm':6})
        assert result.status_code==200,result.text
        assert abs(np.mean(np.asarray(result.json()['points'])[:,0])-x)<.1
    current=client.get(f'/api/designs/{design}').json()
    assert abs(np.mean(np.asarray(json.loads(current['margin_json']))[:,0])-7)<.1
    assert client.post(f'/api/designs/{design}/margin',json={'points':result.json()['points']}).status_code==200
    blocked=client.post(f'/api/designs/{design}/draft-preparation',json={'center':[-7,0,3.5],'axis':[0,0,1],'radius_mm':6})
    assert blocked.status_code==409


def test_save_restore_and_boolean_rebuild(isolated_db,monkeypatch):
    from app.api import routes,designer
    monkeypatch.setattr(routes,'CASE_DIR',isolated_db/'cases')
    monkeypatch.setattr(designer,'CASE_DIR',isolated_db/'cases')
    monkeypatch.setattr(routes,'TEMPLATE_DIR',isolated_db/'templates')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'isolated-v1-regression'}).json()['id']
    support=trimesh.creation.cone(radius=3,height=5,sections=48)
    outer=trimesh.creation.cylinder(radius=4.5,height=7,sections=64);outer.apply_translation([0,0,3])
    uploaded=client.post(f'/api/cases/{case}/assets/support',files={'file':('support.stl',support.export(file_type='stl'),'model/stl')})
    assert uploaded.status_code==200,uploaded.text
    outer.export(isolated_db/'templates'/'FDI_36.stl')
    d=client.post('/api/designs',json={'case_id':case,'target_fdi':36,'support_type':'prep'}).json()['id']
    margin=np.c_[3*np.cos(np.linspace(0,2*np.pi,24,endpoint=False)),3*np.sin(np.linspace(0,2*np.pi,24,endpoint=False)),np.zeros(24)].tolist()
    assert client.post(f'/api/designs/{d}/margin',json={'points':margin}).status_code==200
    assert client.post(f'/api/designs/{d}/axis',json={'axis':[0,0,1]}).status_code==200
    # 显式走 legacy 地标对应路径（研究/回归用）：它保留已知几何，且不依赖坐标帧。
    # A1 起这条路径必须由调用方显式开启，live 生成走"确认帧 + 模板放置"。
    pts=[[0,0,0],[1,0,0],[0,1,0]]
    r=client.post(f'/api/designs/{d}/outer',json={'patient_points':pts,'template_points':pts,'legacy_landmarks':True})
    assert r.status_code==200,r.text
    assert r.json()['legacy_flags']==['LEGACY_LANDMARK_KABSCH_RBF']
    row=client.get(f'/api/designs/{d}').json();original=row['outer_path']
    rev=client.get(f'/api/designs/{d}/revisions').json()['revisions'][-1]['revision_no']
    r=client.post(f'/api/designs/{d}/edit-outer',json={'operation':'transform','scale':[1.02,1,1],'base_outer_path':original})
    assert r.status_code==200,r.text
    assert client.post(f'/api/designs/{d}/edit-outer',json={'operation':'transform','base_outer_path':original}).status_code==409
    r=client.post(f'/api/designs/{d}/restore-outer',json={'revision_no':rev})
    assert r.status_code==200,r.text
    assert client.get(f'/api/designs/{d}').json()['outer_path']==original
    parameters={'margin_points':margin,'insertion_axis':[0,0,1],'cement_gap_mm':.06,'margin_gap_mm':0,'margin_band_mm':.8,
        'minimum_thickness_mm':.6,'voxel_pitch_mm':.1,'margin_extension_mm':0,'enforce_minimum_thickness':False,'machine_axes':5,'cement_gap_start_mm':.5}
    profile={'material':'Engineering fixture only','coverage':'ENGINEERING_ONLY','bite_confirmed':True,'confirmed':True,
        'parameters':{k:v for k,v in parameters.items() if k not in ('margin_points','insertion_axis')}}
    assert client.put(f'/api/designs/{d}/profile',json={**profile,'bite_confirmed':False}).status_code==400
    assert client.put(f'/api/designs/{d}/profile',json=profile).status_code==200
    assert client.post(f'/api/designs/{d}/finalize',json={**parameters,'cement_gap_mm':.07}).status_code==422
    status=client.get(f'/api/designs/{d}/status').json()
    request={'expected_input_digest':status['input_digest'],**status['confirmed_revisions']}
    assert client.post(f'/api/designs/{d}/finalize',json={**request,'axis_revision_id':'unconfirmed'}).status_code==409
    r=client.post(f'/api/designs/{d}/finalize',json=request)
    assert r.status_code==200,r.text
    assert r.json()['qc']['final_mesh']['watertight']
    assert r.json()['qc']['artifact_sha256']['final_path']
    assert r.json()['qc']['consumed_revisions']==status['confirmed_revisions']
    from app.geometry.margin_seam import MarginReconstructionRequired
    def reject_nonplanar(*args,**kwargs):
        raise MarginReconstructionRequired({'status':'MARGIN_RECONSTRUCTION_REQUIRED','plane_deviation_max_mm':1.2})
    with monkeypatch.context() as scope:
        scope.setattr(routes,'finalize_crown',reject_nonplanar)
        status=client.get(f'/api/designs/{d}/status').json()
        blocked=client.post(f'/api/designs/{d}/finalize',json={'expected_input_digest':status['input_digest'],**status['confirmed_revisions']})
        assert blocked.status_code==422
        assert blocked.json()['detail']['geometry']['status']=='MARGIN_RECONSTRUCTION_REQUIRED'
    assert client.get(f'/api/designs/{d}/final.stl').status_code==200
    assert client.get(f'/api/designs/{d}/package.zip').status_code==200
    from pathlib import Path
    mesh_path=Path(original);before=mesh_path.read_bytes()
    mesh_path.write_bytes(before+b'changed')
    assert client.get(f'/api/designs/{d}/final.stl').status_code==409
    assert client.get(f'/api/designs/{d}/package.zip').status_code==409
    mesh_path.write_bytes(before)
    assert client.get(f'/api/designs/{d}/final.stl').status_code==200
    final_path=Path(client.get(f'/api/designs/{d}').json()['final_path'])
    final_bytes=final_path.read_bytes()
    final_path.write_bytes(final_bytes+b'tampered')
    assert client.get(f'/api/designs/{d}/final.stl').status_code==409
    final_path.write_bytes(final_bytes)
    from app.services.design_dependencies import dependency_digest,publish_artifacts
    import pytest
    from fastapi import HTTPException
    baseline=client.get(f'/api/designs/{d}').json()
    expected=dependency_digest(baseline)
    with db() as connection:
        connection.execute('UPDATE designs SET neighbors_json=? WHERE id=?',(json.dumps({'mesial':{'status':'ABSENT'}}),d))
    with pytest.raises(HTTPException) as conflict:
        publish_artifacts(baseline,expected,{'status':'stale_result'})
    assert conflict.value.status_code==409
    assert client.get(f'/api/designs/{d}').json()['status']!='stale_result'
    with db() as connection:
        connection.execute('UPDATE designs SET neighbors_json=? WHERE id=?',(baseline['neighbors_json'],d))
    row=client.get(f'/api/designs/{d}').json()
    r=client.post(f'/api/designs/{d}/edit-outer',json={'operation':'add','centers':[[4.5,0,4]],'base_outer_path':row['outer_path']})
    assert r.status_code==200,r.text
    assert client.get(f'/api/designs/{d}/final.stl').status_code==404
    assert client.get(f'/api/designs/{d}').json()['qc_json'] is None
    changed_margin=np.asarray(margin)+[0,0,.2]
    assert client.post(f'/api/designs/{d}/margin',json={'points':changed_margin.tolist()}).status_code==200
    assert client.get(f'/api/designs/{d}/outer.stl').status_code==409
    assert client.post(f'/api/designs/{d}/restore-outer',json={'revision_no':rev}).status_code==409
    row=client.get(f'/api/designs/{d}').json()
    body_bytes=Path(row['outer_path']).read_bytes()
    status=client.get(f'/api/designs/{d}/status').json()
    request={'expected_input_digest':status['input_digest'],**status['confirmed_revisions']}
    assert client.post(f'/api/designs/{d}/finalize',json=request).status_code==409
    rebuilt=client.post(f'/api/designs/{d}/rebuild-cervical',json=request)
    assert rebuilt.status_code==200,rebuilt.text
    assert Path(row['outer_path']).read_bytes()==body_bytes
    assert rebuilt.json()['qc']['manufacturing_qc']['NonplanarMarginBuild']['exported_stl_seam']['margin_to_seam_max_mm']<1e-5
    detailed=client.get(f'/api/designs/{d}/manufacturing-qc')
    assert detailed.status_code==200,detailed.text
    assert detailed.json()['qc']['ContactQC']['status']=='NOT_ASSESSED'
    audit_path=Path(rebuilt.json()['qc']['artifact_paths']['manufacturing_qc'])
    audit_bytes=audit_path.read_bytes();audit_path.write_bytes(audit_bytes+b'changed')
    assert client.get(f'/api/designs/{d}/manufacturing-qc').status_code==409
    assert client.get(f'/api/designs/{d}/final.stl').status_code==409
    audit_path.write_bytes(audit_bytes)
    assert client.post(f'/api/designs/{d}/axis',json={'axis':[.01,0,1]}).status_code==200
    status=client.get(f'/api/designs/{d}/status').json()
    assert client.post(f'/api/designs/{d}/rebuild-cervical',json={'expected_input_digest':status['input_digest'],**status['confirmed_revisions']}).status_code==409
