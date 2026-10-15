"""Real endpoint regression against the user's visually confirmed green curves; isolated DB."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

DATA=Path(r'D:\code\牙冠自动生成\1.0测试数据')
REFERENCE=Path(r'D:\code\牙冠自动生成\调研结果\基台肩线_2026-10-14\rim_results.json')


def _curve_distance(points,curve):
    segments=np.roll(curve,-1,axis=0)-curve
    t=np.clip(np.einsum('nki,ki->nk',points[:,None]-curve,segments)/np.maximum(np.sum(segments*segments,axis=1),1e-16),0,1)
    return np.linalg.norm(points[:,None]-(curve[None]+t[:,:,None]*segments),axis=2).min(axis=1)


@pytest.mark.parametrize('number,fdi',[(1,46),(2,36),(3,46)])
def test_real_abutment_draft_and_redetect_preserve_confirmed_shoulder(isolated_db,number,fdi):
    if not DATA.exists() or not REFERENCE.exists():pytest.skip('Local reviewed corpus unavailable')
    from app.main import app
    from app.geometry.abutment_margin import ALGORITHM_VERSION
    client=TestClient(app)
    folder=DATA/f'基台测试{number}-牙位{fdi}'
    lower=next(folder.glob('下颌*.stl'));upper=folder/'上颌.stl'
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (lower,upper)}
    case=client.post('/api/cases',json={'name':f'shoulder-{number}'}).json()['id']
    for kind,path in [('lower',lower),('upper',upper)]:
        uploaded=client.post(f'/api/cases/{case}/assets/{kind}',files={'file':(path.name,path.read_bytes(),'model/stl')})
        assert uploaded.status_code==200,uploaded.text
    design=client.post('/api/designs',json={'case_id':case,'target_fdi':fdi,'support_type':'abutment','draft':True}).json()['id']
    located=client.post(f'/api/designs/{design}/draft-preparation',json={})
    assert located.status_code==200 and located.json()['points']==[]
    response=client.post(f'/api/designs/{design}/draft-preparation',json={'restorative_abutment_confirmed':True})
    assert response.status_code==200,response.text
    body=response.json();reference=np.asarray(json.loads(REFERENCE.read_text(encoding='utf8'))['cases'][number-1]['points_world'])
    assert body['method']==ALGORITHM_VERSION
    assert np.allclose(np.asarray(body['points']),reference,atol=1e-8,rtol=0)
    assert body['margin_quality']['status']=='REVIEW_REQUIRED'
    assert client.post(f'/api/designs/{design}/axis',json={'axis':body['axis']}).status_code==200
    detected=client.post(f'/api/designs/{design}/margin/detect',json={})
    assert detected.status_code==200,detected.text
    session=detected.json();assert session['algorithm_version']==ALGORITHM_VERSION
    points=np.asarray(session['active_points'])
    distances=np.r_[_curve_distance(points,reference),_curve_distance(reference,points)]
    assert distances.max()<.03
    confirmed=client.post(f'/api/designs/{design}/margin/confirm',json={'proposal_id':session['active_proposal_id']})
    assert confirmed.status_code==200,confirmed.text
    assert confirmed.json()['boundary_semantics']=='ABUTMENT_CROWN_FINISH_LINE'
    assert before=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (lower,upper)}
