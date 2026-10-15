from .test_auto_fit import ready, prepare, confirm


def test_step3_frame_confirmation_does_not_require_step4_profile(ready):
    from app.db import db
    client,path,design_id,_=ready
    with db() as conn:
        conn.execute('UPDATE designs SET current_profile_revision_id=NULL,profile_json=NULL WHERE id=?',(design_id,))
    response=confirm(client,path,'frame',{'mesial':[1,0,0],'buccal':[0,1,0],'units':'mm'})
    assert response.status_code==200,response.text
    assert client.get(path+'/status').json()['frame_confirmed']


def test_status_identifies_missing_margin_separately_from_axis_and_profile(ready):
    from app.db import db
    client,path,design_id,_=ready
    with db() as conn:
        conn.execute('UPDATE designs SET confirmed_margin_revision_id=NULL,current_profile_revision_id=NULL WHERE id=?',(design_id,))
    status=client.get(path+'/status').json()
    assert {blocker['code'] for blocker in status['blockers']}=={'MARGIN_UNCONFIRMED','PROFILE_UNCONFIRMED'}
    assert status['confirmed_revisions']['axis_revision_id']


def test_generation_requires_explicit_template_downgrade_and_acceptance(ready):
    client,path,_,_=ready
    before=client.get(path).json()
    state=client.get(path+'/workflow/eligibility').json()
    assert state['template_allowed'] and state['automatic_mode'] is None
    body={'kind':'AUTO','confirmed':True,'expected_input_digest':state['input_digest']}
    assert client.post(path+'/workflow/generate',json=body).status_code==409
    body['kind']='TEMPLATE_ONLY'
    response=client.post(path+'/workflow/generate',json=body)
    assert response.status_code==200,response.text
    proposal=response.json();assert proposal['payload']['candidate_kind']=='TEMPLATE_ONLY_DRAFT'
    assert {s['stage'] for s in proposal['payload']['blocked_stages']}=={'PROXIMAL','OCCLUSAL'}
    assert client.get(path).json()['outer_path']==before['outer_path']
    accept={'confirmed':True,'expected_input_digest':state['input_digest']}
    endpoint=path+f"/auto-fit/{proposal['id']}/accept"
    assert client.post(endpoint,json=accept).status_code==409
    accept['confirmed_template_anatomy']=True
    assert client.post(endpoint,json=accept).status_code==200
    assert client.get(path).json()['outer_path']==proposal['artifacts']['outer']
    assert client.post(endpoint,json=accept).status_code==409


def test_eligibility_distinguishes_absent_neighbor_and_unverified_bite(ready):
    client,path,_,neighbors=ready
    prepare(client,path,neighbors)
    state=client.get(path+'/workflow/eligibility').json()
    assert state['automatic_mode']=='PROXIMAL_ONLY'
    assert state['neighbors']=={'mesial':'CONFIRMED','distal':'ABSENT'}
    assert not state['bite_confirmed']
    confirm(client,path,'neighbors/distal',{'status':'UNKNOWN','source':'lower','instance_name':'待确认'})
    assert client.get(path+'/workflow/eligibility').json()['automatic_mode'] is None


def test_candidate_reject_preserves_current_outer(ready):
    client,path,_,_=ready
    original=client.get(path).json()['outer_path']
    state=client.get(path+'/workflow/eligibility').json()
    response=client.post(path+'/workflow/generate',json={'kind':'TEMPLATE_ONLY','confirmed':True,'expected_input_digest':state['input_digest']})
    assert response.status_code==200,response.text
    endpoint=path+f"/auto-fit/{response.json()['id']}"
    assert client.post(endpoint+'/reject').status_code==200
    assert client.post(endpoint+'/accept',json={'confirmed':True,'confirmed_template_anatomy':True,'expected_input_digest':state['input_digest']}).status_code==409
    assert client.get(path).json()['outer_path']==original


def test_automatic_generation_returns_reviewable_candidate_without_overwriting(ready):
    client,path,_,neighbors=ready
    prepare(client,path,neighbors)
    before=client.get(path).json()['outer_path']
    state=client.get(path+'/workflow/eligibility').json()
    response=client.post(path+'/workflow/generate',json={'kind':'AUTO','confirmed':True,'expected_input_digest':state['input_digest']})
    assert response.status_code==200,response.text
    candidate=response.json()
    assert candidate['payload']['candidate_kind']=='AUTO_FIT_CANDIDATE'
    assert candidate['payload']['mode']=='PROXIMAL_ONLY'
    assert candidate['artifacts']['baseline']
    assert client.get(path).json()['outer_path']==before
    assert candidate['payload']['manufacturing_status']=='NOT_PRODUCTION_APPROVED'
    endpoint=path+f"/auto-fit/{candidate['id']}/analysis"
    analysis=client.get(endpoint+'?kind=neighbors')
    assert analysis.status_code==200,analysis.text
    assert len(analysis.json()['distance_mm'])==len(analysis.json()['vertices'])
    assert client.get(endpoint+'?kind=antagonist').status_code==409
    confirm(client,path,'neighbors/distal',{'status':'UNKNOWN','source':'lower','instance_name':'修改'})
    assert client.get(endpoint+'?kind=neighbors').status_code==409


def test_abutment_without_confirmed_bite_reports_generation_blocker(ready):
    from app.db import db
    client,path,_,_=ready
    with db() as conn:
        conn.execute("UPDATE designs SET support_type='abutment' WHERE id=?",(path.split('/')[-1],))
    state=client.get(path+'/workflow/eligibility').json()
    assert state['automatic_mode'] is None
    assert not state['template_allowed']
    assert any(b.get('code')=='OCCLUSION_REGISTRATION_UNVERIFIED' for b in state['blockers'])
    response=client.post(path+'/workflow/generate',json={'kind':'TEMPLATE_ONLY','confirmed':True,'expected_input_digest':state['input_digest']})
    assert response.status_code==409,response.text


def test_abutment_with_confirmed_bite_can_attempt_occlusal_auto_fit(ready):
    from app.db import db
    import numpy as np
    client,path,_,_=ready
    with db() as conn:
        conn.execute("UPDATE designs SET support_type='abutment' WHERE id=?",(path.split('/')[-1],))
    response=confirm(client,path,'bite',{'transform':np.eye(4).tolist(),
        'source_method':'REGISTERED_EXPORT','units':'mm'})
    assert response.status_code==200,response.text
    state=client.get(path+'/workflow/eligibility').json()
    assert state['template_allowed']
    assert state['automatic_mode']=='OCCLUSAL_ONLY'
    response=client.post(path+'/workflow/generate',json={'kind':'AUTO','confirmed':True,
        'expected_input_digest':state['input_digest']})
    assert response.status_code==200,response.text
    assert response.json()['payload']['candidate_kind']=='AUTO_FIT_CANDIDATE'
    assert response.json()['payload']['status']!='PASS_ENGINEERING'
    assert any(stage['stage']=='OCCLUSAL' and stage['reason']=='NO_VALID_SURFACE_SAMPLES'
               for stage in response.json()['payload']['stages'])
