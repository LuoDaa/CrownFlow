import numpy as np
import pytest
import trimesh
import json
from pathlib import Path
from .test_auto_fit import ready,prepare
from app.api.designer import Edit
from app.geometry.sculpt import sculpt_edit,cervical_weights
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame


def crown():
    mesh=trimesh.creation.icosphere(subdivisions=2,radius=3);mesh.apply_translation([0,0,2])
    t=np.arange(24)*2*np.pi/24;ring=np.c_[2*np.cos(t),2*np.sin(t),np.zeros(24)]
    frame=confirmed_patient_frame([0,0,0],[1,0,0],[0,1,0],[0,0,1])
    return mesh,ring,frame


@pytest.mark.parametrize('op',['add','remove','smooth','flatten'])
def test_free_patch_preserves_cervical_and_far_field(op):
    mesh,ring,frame=crown();request=Edit(operation=op,centers=[[3,0,2]],radius_mm=1.5,strength_mm=.03,base_outer_path='x')
    result,report=sculpt_edit(mesh,ring,[0,0,1],request,frame)
    locked=cervical_weights(mesh,ring,[0,0,1])==0
    assert np.array_equal(result.vertices[locked],mesh.vertices[locked])
    far=np.linalg.norm(mesh.vertices-[3,0,2],axis=1)>=1.5
    assert np.array_equal(result.vertices[far],mesh.vertices[far])
    assert report['max_displacement_mm']>0 and result.is_watertight


def test_md_transform_equivariant_and_seam_fixed():
    mesh,ring,frame=crown();request=Edit(operation='transform',scale=[1.05,1,1],base_outer_path='x')
    original,report=sculpt_edit(mesh,ring,[0,0,1],request,frame)
    rotation=trimesh.transformations.rotation_matrix(.9,[1,2,3])
    spun=mesh.copy();spun.apply_transform(rotation);margin=trimesh.transform_points(ring,rotation)
    r=rotation[:3,:3];rotated_frame=confirmed_patient_frame([0,0,0],r@frame.x,r@frame.y,r@frame.z)
    result,_=sculpt_edit(spun,margin,r@frame.z,request,rotated_frame)
    np.testing.assert_allclose(result.vertices,trimesh.transform_points(original.vertices,rotation),atol=1e-8)
    assert np.array_equal(original.vertices[mesh.vertices[:,2]<=.4],mesh.vertices[mesh.vertices[:,2]<=.4])


def test_paint_pull_fixed_and_elastic():
    mesh,ring,frame=crown();v=mesh.vertices
    moving=(v[:,0]>2.5)&(v[:,2]>.5);elastic=(v[:,0]>1.5)&~moving&(v[:,2]>.5)
    masks=np.full(len(v),'FIXED',dtype='<U7');masks[moving]='MOVING';masks[elastic]='ELASTIC'
    req=Edit(operation='region_transform',translation=[.05,0,0],region_masks=masks.tolist(),vertex_weights=moving.astype(float).tolist(),base_outer_path='x')
    result,_=sculpt_edit(mesh,ring,frame.z,req,frame)
    assert np.array_equal(v[masks=='FIXED'],result.vertices[masks=='FIXED'])
    assert np.max(np.linalg.norm(v[elastic]-result.vertices[elastic],axis=1))>0
    assert np.max(np.linalg.norm(v[elastic]-result.vertices[elastic],axis=1))<.051


def test_failed_gate_does_not_mutate_source():
    mesh,ring,frame=crown();before=mesh.vertices.copy()
    req=Edit(operation='remove',centers=[[3,0,2]],radius_mm=1.5,strength_mm=.3,base_outer_path='x').model_copy(update={'strength_mm':6})
    with pytest.raises(ValueError):sculpt_edit(mesh,ring,frame.z,req,frame)
    assert np.array_equal(mesh.vertices,before)


def test_semantics_commit_preview_and_restore(ready):
    from app.services.design_dependencies import file_digest
    client,path,design,_=ready
    session=client.get(path+'/sculpt-session');assert session.status_code==200,session.text
    session=session.json();vertices=np.asarray(session['vertices']);weights=((vertices[:,0]>1)&(vertices[:,2]>.4)).astype(float).tolist()
    before=client.get(path).json()['outer_path'];revisions=client.get(path+'/revisions').json()['n_revisions']
    body={'expected_input_digest':session['input_digest'],'source_session_sha256':session['source_session_sha256']}
    preview=client.post(path+'/sculpt-contact-preview',json={**body,'vertices':vertices.tolist()})
    assert preview.status_code==200,preview.text
    assert client.get(path+'/revisions').json()['n_revisions']==revisions
    annotation=client.put(path+'/sculpt-anatomy',json={**body,'region':'mesial','weights':weights,'mapping_confirmed':True})
    assert annotation.status_code==200,annotation.text
    first_revision=annotation.json()['revision_no'];session=client.get(path+'/sculpt-session').json()
    metadata=session['state']['template'];assert metadata['template_sha256'] and metadata['regions']['mesial']['source']=='MANUAL_CONFIRMED'
    assert len(session['state']['canonical_face_ids'])==len(session['vertices'])
    before_geometry=client.get(path).json()['outer_path']
    edit=client.post(path+'/edit-outer',json={'base_outer_path':before_geometry,'operation':'region_transform','translation':[.01,0,0],
        'semantic_region':'mesial','source_session_sha256':session['source_session_sha256'],'expected_input_digest':session['input_digest']})
    assert edit.status_code==200,edit.text
    saved=client.get(path).json();revision=client.get(path+'/revisions').json()['revisions'][-1]
    assert revision['payload']['before_sha256']==file_digest(before_geometry)
    assert revision['payload']['after_sha256']==file_digest(saved['outer_path'])
    assert client.post(path+'/restore-outer',json={'revision_no':first_revision}).status_code==200
    restored=client.get(path+'/sculpt-session').json()
    assert restored['state']['template']==session['state']['template']
    np.testing.assert_allclose(restored['vertices'],session['vertices'])
    assert client.post(path+'/sculpt-contact-preview',json={**body,'vertices':vertices.tolist()}).status_code==409


def test_preview_and_commit_contact_share_signed_truth(ready):
    from app.services.sculpt_session import read_session,measure_contacts
    from app.api.routes import design_row
    client,path,design,neighbors=ready;prepare(client,path,neighbors)
    session=client.get(path+'/sculpt-session').json();n=len(session['vertices'])
    # Preview has a bounded sampling budget; compare exactly the same geometry and sample IDs.
    ids=list(range(min(512,n)));payload={'expected_input_digest':session['input_digest'],'source_session_sha256':session['source_session_sha256'],
        'vertices':session['vertices'],'sample_ids':ids}
    result=client.post(path+'/sculpt-contact-preview',json=payload)
    assert result.status_code==200,result.text
    d=design_row(design);mesh,frame,ref,_=read_session(d);formal=measure_contacts(d,mesh,frame,ref,sample_ids=ids)
    np.testing.assert_allclose(result.json()['map']['signed_mm'],formal['map']['signed_mm'])
    assert result.json()['map']['valid']==formal['map']['valid']
    assert result.json()['reports']['mesial']['implementation']==formal['reports']['mesial']['implementation']


@pytest.mark.parametrize('side',['mesial','distal'])
def test_anatomic_drag_keeps_opposite_side(side):
    mesh,ring,frame=crown();sign=1 if side=='mesial' else -1
    weights=((mesh.vertices[:,0]*sign>2)&(mesh.vertices[:,2]>.4)).astype(float)
    from app.services.sculpt_commit import region_masks
    masks=region_masks(mesh,weights,cervical_weights(mesh,ring,frame.z))
    result,_=sculpt_edit(mesh,ring,frame.z,Edit(operation='region_transform',translation=[.02*sign,0,0],
        vertex_weights=weights.tolist(),region_masks=masks,base_outer_path='x'),frame)
    opposite=mesh.vertices[:,0]*sign< -1
    assert np.array_equal(result.vertices[opposite],mesh.vertices[opposite])


@pytest.mark.parametrize('side',['mesial','distal'])
def test_adapt_uses_existing_solver_without_increasing_penetration(side,monkeypatch):
    from .test_auto_fit import geometry
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface,fit_domains
    from app.services import sculpt_adapt
    mesh,margin,frame,neighbor=geometry();sign=1 if side=='mesial' else -1
    if sign<0:neighbor.vertices[:,0]*=-1;neighbor.invert()
    surface=FitSurface(neighbor)
    monkeypatch.setattr(sculpt_adapt,'confirmed_neighbor_surfaces',lambda *a,**k:({side:(surface,{})},{}))
    domain=fit_domains(mesh,frame,margin,{side:surface},'PROXIMAL')[0][side]
    state={'regions':{side:domain.astype(float).tolist()}}
    request=Edit(operation='adapt_'+side,target_distance_mm=.05,base_outer_path='x')
    result,report=sculpt_adapt.adapt_sculpt({'margin_json':json.dumps(margin.tolist()),'parameters_json':'{}'},mesh,frame,{},state,request)
    assert report['solver']=='auto_fit_adapter.fit_stage'
    old=surface.query(mesh.vertices)[2];new=surface.query(result.vertices)[2]
    assert np.max(np.maximum(-new,0))<=np.max(np.maximum(-old,0))+1e-6


def test_thickness_preview_is_separate_and_formal_gate_rejects_new_thin_wall(tmp_path):
    from app.services.sculpt_thickness import preview_thickness,formal_thickness_gate,basis
    from app.services.design_dependencies import file_digest
    inner=trimesh.creation.icosphere(subdivisions=1,radius=2).apply_translation([0,0,3]);inner.invert()
    path=tmp_path/'inner.stl';inner.export(path)
    before=trimesh.creation.icosphere(subdivisions=1,radius=3).apply_translation([0,0,3])
    theta=np.arange(24)*2*np.pi/24;margin=np.c_[2*np.cos(theta),2*np.sin(theta),np.zeros(24)]
    design={'margin_json':json.dumps(margin.tolist()),'axis_json':'[0,0,1]','parameters_json':json.dumps({'minimum_thickness_mm':.5,'margin_band_mm':.8,'border_horizontal_mm':.25}),
        'confirmed_margin_revision_id':'m','current_axis_revision_id':'a','current_profile_revision_id':'p'}
    state={'thickness_reference':{'path':str(path),'sha256':file_digest(path),'basis':basis(design)}}
    preview=preview_thickness(design,before,state,np.arange(len(before.vertices)))
    assert preview['status']=='PREVIEW_ONLY' and 'measurement' not in preview
    after=before.copy();after.vertices=(after.vertices-[0,0,3])*.7+[0,0,3]
    with pytest.raises(ValueError,match='厚度门'):formal_thickness_gate(design,before,after,state)
    after=before.copy();after.vertices=(after.vertices-[0,0,3])*1.01+[0,0,3]
    formal=formal_thickness_gate(design,before,after,state)
    assert formal['status']=='FORMAL_REFERENCE_AUDIT' and not formal['production_approved']
    assert formal['measurement']['method']=='all_vertices_and_face_centroids_exact_triangle_and_normal_ray_first_hit'
