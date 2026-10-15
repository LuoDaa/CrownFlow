"""Crown sculpt operations; local-band solver and geometry gates remain shared."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .nonplanar_crown import MarginCurtain
from .crown_outer.laplacian import solve_local_band
from .crown_outer.deform_gate import deformation_gate
from .crown_outer.patient_frame import validate_patient_frame
from .crown_shape_quality import deformation_shape_report, shape_gate


class SculptEditRejected(ValueError):
    """Structured, rolled-back Step-5 edit rejection."""

    def __init__(self, report):
        self.report = report
        super().__init__(report.get('message') or report.get('reason') or '编辑未通过几何检查，已回滚')


def _failure_region(frame, vertices, moved):
    ids=np.flatnonzero(np.asarray(moved)>1e-9)
    if not len(ids): return 'UNKNOWN'
    center=np.asarray(vertices)[ids].mean(axis=0)
    rel=(center-np.asarray(frame.origin,float))@frame.matrix()
    ax=np.abs(rel)
    if ax[2]>=max(ax[0],ax[1]): return 'OCCLUSAL_OR_AXIAL'
    if ax[0]>=ax[1]: return 'MESIAL' if rel[0]>=0 else 'DISTAL'
    return 'BUCCAL' if rel[1]>=0 else 'LINGUAL'


def _rejection_report(operation, frame, v0, v1, faces, focus, gate=None, reason='GEOMETRY_GATE', message=None):
    moved=np.linalg.norm(np.asarray(v1)-np.asarray(v0),axis=1)
    focus=np.asarray(focus,int) if focus is not None else np.zeros(0,int)
    severe=[];pairs=[];metrics={}
    if gate is not None:
        metrics=gate.as_dict().get('metrics',{})
        severe=list((metrics.get('flipped') or {}).get('severe_faces') or [])
        if gate.pairs is not None:
            pairs=np.asarray(gate.pairs,int).reshape(-1,2).tolist()[:80]
    failure_faces=list(dict.fromkeys([int(x) for x in severe]+[int(x) for p in pairs for x in p]))[:120]
    if not failure_faces and len(focus): failure_faces=[int(x) for x in focus[:120]]
    self_ev=(metrics.get('self_intersections_focus') or {})
    return {
        'status':'REJECTED_AND_ROLLED_BACK',
        'reason':reason,
        'message':message or '本次修改未保存，已恢复到提交前牙冠',
        'operation':operation,
        'affected_vertex_count':int(np.count_nonzero(moved>1e-9)),
        'max_displacement_mm':float(moved.max()) if len(moved) else 0.0,
        'p95_displacement_mm':float(np.quantile(moved,.95)) if len(moved) else 0.0,
        'locked_displacement_max_mm':float((metrics.get('locked_displacement_max_mm') or 0.0)),
        'severe_flip_count':int((metrics.get('flipped') or {}).get('n_flipped_severe') or 0),
        'new_self_intersection_count':int(self_ev.get('new') or 0),
        'baseline_self_intersection_count':int(self_ev.get('before') or 0),
        'after_self_intersection_count':int(self_ev.get('after') or 0),
        'degenerate_face_count':int(np.count_nonzero(_face_areas(np.asarray(v1),np.asarray(faces))<1e-10)),
        'failure_face_ids':failure_faces,
        'failure_region':_failure_region(frame,v0,moved),
        'suggested_action':'缩小笔刷半径或强度，分小笔修改；若小笔仍失败，请检查基线网格/厚度参考',
        'gate':gate.as_dict() if gate is not None else None,
    }


def _face_areas(v, faces):
    tri=v[faces]
    return .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)


def cervical_weights(mesh,margin,axis):
    height=MarginCurtain(margin,axis).axial_above(mesh.vertices)
    weight=np.clip((height-.4)/1.2,0,1)
    return weight*weight*(3-2*weight)


def sculpt_edit(mesh,margin,axis,request,frame):
    validate_patient_frame(frame)
    v=np.asarray(mesh.vertices).copy();faces=np.asarray(mesh.faces)
    lock=cervical_weights(mesh,margin,axis);basis=frame.matrix();origin=frame.origin
    angle=np.deg2rad(request.frame_angle_deg)
    basis=basis@Rotation.from_euler('z',angle).as_matrix()
    t,r,s=[np.asarray(value,float) for value in (request.translation,request.rotation,request.scale)]
    if any(a.shape!=(3,) or not np.isfinite(a).all() for a in (t,r,s)):
        raise ValueError('变换必须包含三个有限数值')
    if np.any(s<.5) or np.any(s>1.5) or np.any(np.abs(t)>5) or np.any(np.abs(r)>45):
        raise ValueError('单次缩放 0.5–1.5、位移 ≤5 mm、旋转 ≤45°')
    op=request.operation;weights=getattr(request,'vertex_weights',None)
    if op=='region_transform' and getattr(request,'semantic_region','') in ('cusp_1','cusp_2','cusp_3','cusp_4','central_fissure','mesial_marginal_ridge','distal_marginal_ridge'):
        if np.linalg.norm(t[:2])>1 or abs(t[2])>1:
            raise ValueError('解剖区域单步横向和轴向移动分别 ≤1 mm')
    if weights is not None:
        weights=np.asarray(weights,float)
        if weights.shape!=(len(v),) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
            raise ValueError('区域权重必须匹配当前网格并在 0–1 之间')
    if op in ('transform','region_transform'):
        moving=(lock>.999) if weights is None else (weights>.5)&(lock>0)
        pivot=v[moving].mean(0) if op=='region_transform' and moving.any() else origin
        local=(v-pivot)@basis
        transformed=(local*s)@Rotation.from_euler('xyz',r,degrees=True).as_matrix().T
        desired=transformed@basis.T+pivot+t@basis.T
        if op=='transform':result=v+(desired-v)*lock[:,None]
        else:
            masks=np.asarray(getattr(request,'region_masks',None))
            if masks.shape!=(len(v),) or not np.isin(masks,['MOVING','ELASTIC','FIXED']).all():
                raise ValueError('区域变形需要完整 Moving / Elastic / Fixed 掩码')
            moving=(masks=='MOVING')&(lock>0);elastic=(masks=='ELASTIC')&(lock>0)
            if not moving.any():raise ValueError('没有可移动的区域')
            targets=v.copy();targets[moving]+=((desired-v)*lock[:,None])[moving]
            free=np.flatnonzero(elastic);fixed=np.flatnonzero(~elastic)
            result=solve_local_band(v,faces,free,fixed,targets)[0]
    elif op in ('add','remove','smooth','flatten','drag'):
        centers=np.asarray(request.centers,float)
        if centers.ndim!=2 or centers.shape[1]!=3 or not len(centers) or not np.isfinite(centers).all():
            raise ValueError('笔刷需要有效三维轨迹')
        distance=cKDTree(centers).query(v)[0]
        brush=np.clip(1-distance/request.radius_mm,0,1)**float(getattr(request,'falloff',2))
        if weights is not None:brush*=weights
        brush*=lock;ids=np.flatnonzero(brush>0)
        if not len(ids):
            if getattr(request,'source_session_sha256',None):raise ValueError('笔刷没有命中可编辑顶点；颈缘固定，稀疏网格需增大笔刷')
            return mesh.copy(),{'affected_area_mm2':0,'max_displacement_mm':0,'p95_displacement_mm':0,'reason':'LEGACY_NOOP_NO_VERTICES_IN_PATCH'}
        if op=='smooth':
            # Harmonic local patch, retaining the fixed boundary and confirmed cervical band.
            targets=v.copy();edges=np.asarray(mesh.edges_unique);sums=np.zeros_like(v);counts=np.zeros(len(v))
            for a,b in ((edges[:,0],edges[:,1]),(edges[:,1],edges[:,0])):
                np.add.at(sums,a,v[b]);np.add.at(counts,a,1)
            center_ids=ids[brush[ids]>=np.max(brush)*.8]
            targets[center_ids]+=(sums[center_ids]/np.maximum(counts[center_ids,None],1)-v[center_ids])*np.minimum(request.strength_mm*5,1)*brush[center_ids,None]
            free=np.setdiff1d(ids,center_ids)
            result=solve_local_band(v,faces,free,np.setdiff1d(np.arange(len(v)),free),targets)[0]
        elif op=='flatten':
            patch=v[ids];center=np.average(patch,axis=0,weights=brush[ids])
            if len(ids)<3:raise ValueError('压平至少需要三个表面顶点')
            _,_,vectors=np.linalg.svd((patch-center)*np.sqrt(brush[ids,None]),full_matrices=False)
            normal=vectors[-1];distance=(v-center)@normal
            result=v-normal[None,:]*np.clip(distance,-request.strength_mm,request.strength_mm)[:,None]*brush[:,None]
        elif op=='drag':
            if np.linalg.norm(t)>1:raise ValueError('局部移动单步 ≤1 mm')
            result=v+brush[:,None]*(t@basis.T)
        else:result=v+np.asarray(mesh.vertex_normals)*request.strength_mm*(1 if op=='add' else -1)*brush[:,None]
    else:raise ValueError('未知编辑工具')
    result[lock==0]=v[lock==0]
    moved=np.linalg.norm(result-v,axis=1);candidate=mesh.copy();candidate.vertices=result
    focus=np.flatnonzero(np.any(moved[faces]>1e-9,axis=1))
    if len(focus):
        gate=deformation_gate(v,result,faces,scale_mm=.02,locked_ids=np.flatnonzero(lock==0),focus_faces=focus)
        strict_flips=np.flatnonzero(np.einsum('ij,ij->i',mesh.face_normals,candidate.face_normals)<=0)
        degenerate=np.flatnonzero(candidate.area_faces<1e-10)
        shape=deformation_shape_report(v,result,faces,focus_faces=focus)
        shape_check=shape_gate(shape,max_edge_stretch=4.0,p95_edge_stretch=2.0,
                               min_quality_floor=.015,relative_quality_floor=.30,
                               min_area_ratio=.10,max_area_ratio=6.0)
        if not gate.ok or len(strict_flips) or len(degenerate) or not shape_check['ok']:
            report=_rejection_report(op,frame,v,result,faces,focus,gate,reason='GEOMETRY_GATE' if shape_check['ok'] else 'SHAPE_QUALITY_GATE')
            report['failure_face_ids']=list(dict.fromkeys(report['failure_face_ids']+strict_flips[:80].astype(int).tolist()+degenerate[:80].astype(int).tolist()))[:120]
            report['degenerate_face_count']=int(len(degenerate))
            report['shape_quality']=shape;report['shape_gate']=shape_check
            if not shape_check['ok']:
                report['message']='本次修改会造成局部拉伸/薄片形变，未保存并已回滚'
            raise SculptEditRejected(report)
    else:
        shape={'focus_face_count':0};shape_check={'ok':True,'reasons':[]}
    if mesh.is_watertight and not candidate.is_watertight:
        raise SculptEditRejected(_rejection_report(op,frame,v,result,faces,focus,None,
            reason='WATERTIGHTNESS_LOST',message='本次修改会破坏闭合拓扑，未保存并已回滚'))
    return candidate,{'affected_area_mm2':float(mesh.area_faces[focus].sum()),
        'max_displacement_mm':float(moved.max()),'p95_displacement_mm':float(np.quantile(moved,.95)),
        'affected_vertex_count':int((moved>1e-9).sum()),'cervical_locked':True,'frame_revision_id':frame.revision_id,
        'shape_quality':shape,'shape_gate':shape_check}
