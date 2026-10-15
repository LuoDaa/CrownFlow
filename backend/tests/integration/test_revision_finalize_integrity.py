"""Negative workflow checks use synthetic geometry and an isolated database."""
import json
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import trimesh
from fastapi.testclient import TestClient

from app.db import db
from app.revisions import DesignRevision, RevisionKind, append_revision, record_auto_margin_candidate


@pytest.fixture
def ready_design(isolated_db,monkeypatch):
    from app.api import routes
    from app.services import design_dependencies
    monkeypatch.setattr(routes,'CASE_DIR',isolated_db/'cases')
    monkeypatch.setattr(routes,'TEMPLATE_DIR',isolated_db/'templates')
    monkeypatch.setattr(design_dependencies,'TEMPLATE_DIR',isolated_db/'templates')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'revision-contract'}).json()['id']
    mesh=trimesh.creation.cone(radius=3,height=5)
    assert client.post(f'/api/cases/{case}/assets/support',files={'file':('prep.stl',mesh.export(file_type='stl'),'model/stl')}).status_code==200
    identifier=client.post('/api/designs',json={'case_id':case,'target_fdi':46,'support_type':'prep','draft':True}).json()['id']
    path=f'/api/designs/{identifier}'
    margin=[[3*math.cos(2*math.pi*i/16),3*math.sin(2*math.pi*i/16),0] for i in range(16)]
    assert client.post(path+'/margin',json={'points':margin}).status_code==200
    assert client.post(path+'/axis',json={'axis':[0,0,1]}).status_code==200
    assert client.put(path+'/profile',json={'material':'synthetic fixture','coverage':'ENGINEERING_ONLY',
        'bite_confirmed':True,'confirmed':True,'parameters':{'machine_axes':5,'enforce_minimum_thickness':False}}).status_code==200
    template=isolated_db/'templates'/'FDI_46.stl'
    mesh.export(template)
    # A1：live 初稿必须建立在已确认坐标帧上（先确认方向与单位，再做模板放置）
    status=client.get(path+'/status').json()
    assert client.put(path+'/auto-fit/frame',json={'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm',
        'confirmed':True,'expected_input_digest':status['input_digest']}).status_code==200
    assert client.post(path+'/outer/auto',json={'margin_points':margin,'insertion_axis':[0,0,1]}).status_code==200
    return client,path,identifier,margin


def final_request(client,path):
    status=client.get(path+'/status').json()
    return {'expected_input_digest':status['input_digest'],**status['confirmed_revisions']}


@pytest.mark.parametrize('kind',[kind for kind in RevisionKind if kind is not RevisionKind.MARGIN_CANDIDATE])
def test_wrong_revision_accept_has_no_side_effect(ready_design,kind):
    client,path,identifier,margin=ready_design
    revision=append_revision(DesignRevision(design_id=identifier,revision_no=0,kind=kind,
        payload={'points':margin,'axis':[0,0,1]}))
    before=client.get(path).json()
    assert client.post(path+f'/revisions/{revision.revision_no}/accept').status_code==409
    assert client.get(path).json()==before


def test_axis_candidate_preserves_effective_design(ready_design,monkeypatch):
    client,path,identifier,margin=ready_design
    from app.api import routes
    monkeypatch.setattr(routes,'auto_axis',lambda *args:SimpleNamespace(as_dict=lambda:{'axis':[.1,0,.99],'status':'REVIEW_REQUIRED'}))
    before=client.get(path).json()
    assert client.post(path+'/suggest-axis',json={}).status_code==200
    assert client.get(path).json()==before
    assert client.get(path+'/status').json()['outer_current']


def test_unconfirmed_and_foreign_inputs_are_blocked(ready_design):
    client,path,identifier,margin=ready_design
    request=final_request(client,path)
    for key in ('margin_revision_id','axis_revision_id','profile_revision_id'):
        assert client.post(path+'/finalize',json={**request,key:'other-design-revision'}).status_code==409
    assert client.post(path+'/finalize',json={**request,'margin_points':margin}).status_code==422
    with db() as connection:
        connection.execute('UPDATE designs SET current_profile_revision_id=NULL WHERE id=?',(identifier,))
    assert client.post(path+'/finalize',json=request).status_code==409


def test_margin_candidate_requires_explicit_acceptance(ready_design):
    client,path,identifier,margin=ready_design
    candidate,applied=record_auto_margin_candidate(identifier,margin)
    assert not applied
    assert client.get(path).json()['confirmed_margin_revision_id']!=candidate.id
    assert client.post(path+f'/revisions/{candidate.revision_no}/accept').status_code==200
    assert client.get(path).json()['confirmed_margin_revision_id']==candidate.id
    assert not client.get(path+'/status').json()['outer_current']


def test_inflight_finalize_cannot_overwrite_new_margin(ready_design,monkeypatch):
    client,path,identifier,margin=ready_design
    from app.api import routes
    entered,release=Event(),Event()
    mesh=trimesh.creation.icosphere(subdivisions=1)
    def paused_geometry(outer,support,final,cutter,*args,**kwargs):   # _finalize 还传 prepared_support/support_contract
        entered.set()
        assert release.wait(10)
        mesh.export(final);mesh.export(cutter)
        return mesh,mesh,mesh,{}
    monkeypatch.setattr(routes,'finalize_crown',paused_geometry)
    monkeypatch.setattr(routes,'qc_from_paths',lambda *args,**kwargs:{'synthetic_test':True})
    request=final_request(client,path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task=pool.submit(client.post,path+'/finalize',json=request)
        assert entered.wait(10)
        changed=[[x,y,z+.3] for x,y,z in margin]
        assert client.post(path+'/margin',json={'points':changed}).status_code==200
        release.set()
        assert task.result(timeout=10).status_code==409
    current=client.get(path).json()
    assert json.loads(current['margin_json'])==changed
    assert current['final_path'] is None and current['qc_json'] is None
    assert not any(r['kind']=='final' for r in client.get(path+'/revisions').json()['revisions'])


def test_version_detects_change_then_restore(ready_design):
    client,path,identifier,_=ready_design
    from app.services.design_dependencies import dependency_digest,publish_artifacts
    from fastapi import HTTPException
    baseline=client.get(path).json()
    expected=dependency_digest(baseline)
    import subprocess,sys
    from app import config
    script='''import sqlite3,sys
with sqlite3.connect(sys.argv[1]) as connection:
    previous=connection.execute("SELECT neighbors_json FROM designs WHERE id=?",(sys.argv[2],)).fetchone()[0]
    connection.execute("UPDATE designs SET neighbors_json='{}' WHERE id=?",(sys.argv[2],))
    connection.execute("UPDATE designs SET neighbors_json=? WHERE id=?",(previous,sys.argv[2]))
'''
    subprocess.run([sys.executable,'-c',script,str(config.DB_PATH),identifier],check=True,capture_output=True,text=True)
    assert client.get(path).json()['design_revision']==baseline['design_revision']+2
    with pytest.raises(HTTPException) as error:
        publish_artifacts(baseline,expected,{'status':'stale'})
    assert error.value.status_code==409
