"""Bounded outward repair of the anatomical body; intaglio and finish line stay fixed."""
import numpy as np
from .manufacturing_qc import _closest, _stats, _normal_ray_distances
from .crown_outer.deform_gate import deformation_gate


def enforce_body_thickness(outside, inside, minimum_mm, *, protected_surfaces=(), max_iterations=12):
    if not np.isfinite(minimum_mm) or minimum_mm<=0:raise ValueError('最小厚度必须为正数')
    if len(outside.faces)+len(inside.faces)>100000:raise ValueError('补厚同步工程预算为 100000 个面，请先简化')
    result=outside.copy();baseline=np.asarray(result.vertices).copy()
    # shortcut: discrete vertices/centroids are engineering coverage, not a global minimum certificate.
    samples=np.vstack((inside.vertices,inside.triangles_center))
    directions=-np.vstack((inside.vertex_normals,inside.face_normals))
    locks=np.zeros(len(baseline),bool)
    for surface,target in protected_surfaces:
        _,_,signed,_,valid=surface.query(baseline)
        if not valid.all():raise ValueError('保护面符号查询无效，不能执行补厚')
        locks|=signed<=float(target)+.2
    history=[]
    def measure(mesh):
        near,dist,faces=_closest(mesh,samples)
        ray=_normal_ray_distances(mesh,samples,directions)
        ray=np.where(np.isfinite(ray),ray,np.inf)
        return near,np.minimum(dist,ray),faces
    _,before,_=measure(result)
    reason=None;last_rejected=None
    for iteration in range(max_iterations):
        near,values,face_ids=measure(result);thin=values<minimum_mm-1e-5
        if not thin.any():break
        delta=np.zeros_like(result.vertices);counts=np.zeros(len(delta))
        toward=near[thin]-samples[thin];length=np.linalg.norm(toward,axis=1)
        if np.any(length<1e-8):reason='ZERO_CLEARANCE_OR_CROSSING';break
        pushes=toward/length[:,None]*np.minimum(minimum_mm-values[thin]+.002,.15)[:,None]
        for corner in range(3):
            ids=np.asarray(result.faces)[face_ids[thin],corner]
            np.add.at(delta,ids,pushes);np.add.at(counts,ids,1)
        delta/=np.maximum(counts[:,None],1);delta[locks]=0
        accepted=False
        for scale in (1.,.5,.25):
            proposal=result.copy();proposal.vertices=np.asarray(result.vertices)+scale*delta
            moved=np.linalg.norm(scale*delta,axis=1)>1e-9
            if not moved.any():reason='PROTECTED_ZONE_THICKNESS_CONFLICT';break
            focused=np.flatnonzero(np.any(moved[np.asarray(result.faces)],axis=1))
            gate=deformation_gate(result.vertices,proposal.vertices,result.faces,scale_mm=.1,
                locked_ids=np.flatnonzero(locks),focus_faces=focused)
            flips=np.einsum('ij,ij->i',result.face_normals,proposal.face_normals)<=0
            conflicts=False
            for surface,target in protected_surfaces:
                _,_,old,_,valid0=surface.query(result.vertices);_,_,new,_,valid1=surface.query(proposal.vertices)
                if not valid0.all() or not valid1.all() or np.any(new<np.minimum(old,float(target))-1e-6):conflicts=True;break
                # Moving faces may cross a patch even when vertex signs appear safe.
                from .crown_outer.auto_fit_adapter import surface_crossings
                crossing=surface_crossings(proposal,surface,moved)
                if crossing.get('status')!='MEASURED_ENGINEERING' or crossing.get('crossing_pairs',0):conflicts=True;break
            _,after,_=measure(proposal)
            improved=np.sum(np.maximum(minimum_mm-after,0))<np.sum(np.maximum(minimum_mm-values,0))-1e-8
            if gate.ok and not flips.any() and not conflicts and improved:
                result=proposal;history.append({'iteration':iteration+1,'scale':scale,'geometry_gate':gate.as_dict()});accepted=True;break
            last_rejected={'iteration':iteration+1,'scale':scale,'geometry_gate':gate.as_dict(),
                'flipped_face_count':int(flips.sum()),'reference_conflict':conflicts,'thickness_improved':bool(improved)}
        if not accepted:
            reason=reason or 'GEOMETRY_OR_CONTACT_CONFLICT';break
    _,after,_=measure(result)
    remaining=int((after<minimum_mm-1e-5).sum())
    return result,{'algorithm':'constrained_outward_body_thickness/1','status':'PASS_SAMPLED_ENGINEERING' if not remaining else 'REVIEW_REQUIRED',
        'reason':reason or ('ITERATION_BUDGET' if remaining else None),'configured_minimum_mm':float(minimum_mm),
        'before':_stats(before),'after':_stats(after),'remaining_thin_samples':remaining,
        'protected_vertex_count':int(locks.sum()),'steps':history,
        'last_rejected_trial':last_rejected,
        'max_displacement_mm':float(np.linalg.norm(result.vertices-baseline,axis=1).max()),
        'intaglio_preserved':True,'scope':'BODY_ABOVE_CERVICAL_TRANSITION',
        'continuous_minimum_certified':False,'contact_coverage':'CONFIRMED_REFERENCES_ONLY' if protected_surfaces else 'NOT_ASSESSED'}
