import json
import trimesh
from fastapi.testclient import TestClient
from app.db import db

def test_manual_inputs_are_versioned_without_geometry_pass(isolated_db,monkeypatch):
    from app.api import routes,design_inputs
    monkeypatch.setattr(routes,'CASE_DIR',isolated_db/'cases')
    monkeypatch.setattr(design_inputs,'CASE_DIR',isolated_db/'cases')
    from app.main import app
    client=TestClient(app)
    case=client.post('/api/cases',json={'name':'manual-input-regression'}).json()['id']
    source={'prescription':{'target_fdi':46,'material_name':'Order label'},'configuration_status':'UNCONFIRMED'}
    with db() as c: c.execute('UPDATE cases SET order_json=? WHERE id=?',(json.dumps(source),case))
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=3)
    for kind in ('upper','lower'):
        assert client.post(f'/api/cases/{case}/assets/{kind}',files={'file':(kind+'.stl',mesh.export(file_type='stl'),'model/stl')}).status_code==200
    d=client.post('/api/designs',json={'case_id':case,'target_fdi':46,'support_type':'prep','draft':True}).json()
    d=client.get(f"/api/designs/{d['id']}").json()
    assert json.loads(d['order_source_json'])==source
    path=f"/api/designs/{d['id']}"
    assert client.put(path+'/bite',json={'viewed':False,'pose_confirmed':True}).status_code==400
    response=client.put(path+'/bite',json={'viewed':True,'pose_confirmed':True})
    assert response.status_code==200,response.text
    assert json.loads(response.json()['bite_json'])['geometry_status']=='NOT_RUN'
    assert client.put(path+'/neighbors/mesial',json={'status':'SELECTED','source':'upper','center':[0,0,3]}).status_code==400
    chosen=client.put(path+'/neighbors/mesial',json={'status':'SELECTED','source':'lower','center':[0,0,3],'radius_mm':4})
    assert chosen.status_code==200,chosen.text
    region=json.loads(chosen.json()['neighbors_json'])['mesial']
    assert region['source']['asset_id'] and len(region['region_sha256'])==64
    assert client.get(path+'/neighbors/mesial/file').status_code==200
    with db() as c: c.execute("UPDATE designs SET final_path='old',cutter_path='old',qc_json='{}' WHERE id=?",(d['id'],))
    absent=client.put(path+'/neighbors/mesial',json={'status':'ABSENT'}).json()
    assert absent['final_path'] is None and absent['qc_json'] is None
    assert client.get(path+'/neighbors/mesial/file').status_code==404
    withdrawn=client.put(path+'/neighbors/mesial',json={'status':'UNSET'}).json()
    assert json.loads(withdrawn['neighbors_json'])['mesial']['status']=='UNSET'
    assert json.loads(client.get(path).json()['bite_json'])['pose_confirmed'] is True
    import pytest
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as conflict:
        design_inputs.save_input(d['id'],'neighbors_json',{},None)
    assert conflict.value.status_code==409
