"""Immutable workbench adapter for Contact v2 / Occlusion v1 local-band steps.

The research entry points require an open cervical ring; workbench bodies are
closed. Reuse their exact query, local solve and gate without inventing a ring.
"""
from dataclasses import dataclass
import numpy as np
from .laplacian import solve_local_band
from .deform_gate import deformation_gate
from ..nonplanar_crown import MarginCurtain


@dataclass
class FitSurface:
    """接触面查询：符号**一律**取自 `measurement_core.SignSource`（O1 起与 Occlusion 同源）。

    闭合可定向实体 → 实体内外判定；开放裁片 → 必须有调用方声明的外向法向（否则拒绝）；
    两者都不满足时 `SignSource` 会给 ``UNSIGNED_ONLY``，此处直接拒绝构造
    （接触判定不接受"没有符号"的参考面）。
    """

    mesh: object
    outward: object = None

    def __post_init__(self):
        from ..measurement_core import SIGN_UNSIGNED_ONLY, SignSource
        self.source=SignSource.for_mesh(self.mesh,self.outward)
        if self.source.method==SIGN_UNSIGNED_ONLY:
            raise ValueError('开放表面需要人工确认外向法向')
        self.patch=self.source.patch
        self.solid=self.source.method=='SOLID_CONTAINMENT'
        self.outward=self.source.outward
        self.sign_method=self.source.method

    def query(self,points):
        return self.source.query(points)


def measure(vertices,mask,surface,target,*,kind=None,side=None,crown=None):
    """B 阶段：接触报告的**唯一**实现移交给 `services.contact_measurement`。

    本函数只保留求解器需要的返回形状 ``(report, query)``；报告字段由 service 产出，
    保证放置/AutoFit/候选审核/最终 QC 四处说的是同一件事（有符号 clearance）。
    ⚠️ 延迟导入是刻意的：service 会在需要时反向构造 FitSurface/接触域，模块级互导会成环。
    """
    from ...services.contact_measurement import measure_surface
    return measure_surface(surface,vertices,mask,target,kind=kind or 'CONTACT',side=side,crown=crown)


def surface_crossings(mesh,surface,mask):
    """All overlapping AABBs in the audited domain; coplanar cases stay unresolved."""
    from .thickness import _tri_tri_intersect
    face_ids=np.flatnonzero(np.any(np.asarray(mask)[mesh.faces],axis=1))
    pairs=[];tree=surface.mesh.triangles_tree
    for face in face_ids:
        tri=mesh.triangles[face]
        pairs.extend((int(face),int(other)) for other in tree.intersection(np.r_[tri.min(0)-1e-8,tri.max(0)+1e-8]))
        if len(pairs)>200000:
            return {'status':'NOT_ASSESSED','reason':'PAIR_BUDGET_EXCEEDED','candidate_pairs':len(pairs)}
    if not pairs: return {'status':'MEASURED_ENGINEERING','crossing_pairs':0,'near_parallel_candidates':0,'candidate_pairs':0,'intersecting_crown_area_mm2':0}
    pairs=np.asarray(pairs,int)
    hit,parallel,length=_tri_tri_intersect(mesh.triangles[pairs[:,0]],surface.mesh.triangles[pairs[:,1]],return_segment=True)
    faces=np.unique(pairs[hit,0])
    return {'status':'REVIEW_REQUIRED' if parallel else 'MEASURED_ENGINEERING',
        'crossing_pairs':int(hit.sum()),'near_parallel_candidates':int(parallel),'candidate_pairs':len(pairs),
        'intersecting_crown_area_mm2':float(mesh.area_faces[faces].sum()),
        'intersection_segment_total_mm':float(length[hit].sum()),
        'method':'overlapping_triangle_AABB_and_existing_triangle_intersection',
        'coverage_note':'Crown domain only; coplanar overlaps require review; area counts whole hit faces, not overlap area.'}


def fit_domains(mesh,frame,margin,surfaces,stage,protected=None):
    reference=np.asarray(mesh.vertices)
    height=MarginCurtain(margin,frame.z).axial_above(reference)
    neck=height<=.4
    normals=np.asarray(mesh.vertex_normals)
    rel=(reference-frame.origin)@frame.matrix()
    occ=(rel[:,2]>=np.quantile(rel[~neck,2],.75))&(normals@frame.z>.35) if (~neck).any() else np.zeros(len(reference),bool)
    half=max(np.ptp((np.asarray(margin)-frame.origin)@frame.x)/2,1e-6)
    masks={side:(~neck)&(~occ)&((rel[:,0] if side=='mesial' else -rel[:,0])>.6*half)&((normals@frame.x)*(1 if side=='mesial' else -1)>.1) for side in surfaces} if stage=='PROXIMAL' else {'occlusal':occ&~neck}
    locked=neck | (occ if stage=='PROXIMAL' else np.zeros(len(reference),bool))
    if protected is not None: locked |= protected
    masks={side:mask&~locked for side,mask in masks.items()}
    if stage=='PROXIMAL':
        for side,surface in surfaces.items():
            ids=np.flatnonzero(masks[side])
            if not len(ids): continue
            _,_,signed,_,valid=surface.query(reference[ids])
            if valid.any(): masks[side][ids] &= signed<=signed[valid].min()+.6
    return masks,locked


def _broaden_proximal_targets(reference, desired, masks, seed_groups, locked, adj, frame):
    """把最近点产生的“针尖 seed”扩成有基底的近远中接触丘。

    最近点目标只决定**中心位移量**；周围两圈同侧语义顶点沿确认的 M/D 轴按
    0.65 / 0.35 衰减跟随。这样接触来自一块连续的 crown wall，而不是单顶点长触角。
    颈缘、咬合区、protected 顶点仍由 ``locked`` 保持不动。
    """
    support=set()
    per_side={}
    for side,core in seed_groups.items():
        core=set(core)
        if not core or side not in ('mesial','distal'):
            continue
        allowed=set(map(int,np.flatnonzero(np.asarray(masks[side],bool)&~np.asarray(locked,bool))))
        if not allowed:
            continue
        direction=np.asarray(frame.x,float)*(1.0 if side=='mesial' else -1.0)
        direction/=max(float(np.linalg.norm(direction)),1e-12)
        core_ids=np.asarray(sorted(core),int)
        scalar=(desired[core_ids]-reference[core_ids])@direction
        finite=scalar[np.isfinite(scalar)]
        if not len(finite):
            continue
        drive=float(np.median(finite))
        seen=set(core);front=set(core);added=[]
        for decay in (.65,.35):
            ring={v for u in front for v in adj[u]}-seen
            ring&=allowed
            if not ring:
                front=set();continue
            ids=np.asarray(sorted(ring),int)
            # 只沿 M/D 轴铺开，不把最近点的切向/法向噪声复制给邻点。
            desired[ids]=reference[ids]+(decay*drive)*direction[None,:]
            support.update(ring);added.extend(ids.tolist())
            seen|=ring;front=ring
        per_side[side]={'core_seed_count':len(core),'support_seed_count':len(set(added)),
                        'drive_mm':drive,'support_rings':2}
    return support,per_side


def fit_stage(mesh,frame,margin,surfaces,targets,stage,*,protected=None,domain_mesh=None,domain_masks=None,cut_only=False,transition_rings=None):
    reference=np.asarray(mesh.vertices).copy();faces=np.asarray(mesh.faces).copy()
    masks,locked=fit_domains(mesh if domain_mesh is None else domain_mesh,frame,margin,surfaces,stage,protected)
    if domain_masks is not None:
        for side,mask in domain_masks.items():
            mask=np.asarray(mask,bool)
            if mask.shape!=(len(reference),):raise ValueError('冻结适配区域与冠形顶点不匹配')
            masks[side]=mask&~locked
    initial={side:measure(reference,masks[side],surface,targets[side],kind=stage,side=side) for side,surface in surfaces.items()}
    before={side:value[0] for side,value in initial.items()}
    desired=reference.copy();seeds=set();seed_groups={side:set() for side in surfaces};unreachable=False
    for side,surface in surfaces.items():
        _,query=measure(reference,masks[side],surface,targets[side],kind=stage,side=side)
        if query is None: continue
        ids,cp,fids,signed,distance,valid=query
        needed=(signed< -1e-6) if cut_only else (np.abs(signed-targets[side])>1e-6) if stage=='PROXIMAL' else (signed<targets[side]-1e-6)
        unreachable |= bool(np.any(valid & needed & (distance>2.5)))
        active=valid & (distance<=2.5) & needed
        normals_q=surface.patch.normals[fids]*surface.patch.normal_sign
        raw_target=cp[active]+normals_q[active]*targets[side]
        if stage=='PROXIMAL' and side in ('mesial','distal'):
            # Fine contact is allowed to correct only a small residual. Keep the
            # displacement predominantly along the confirmed mesial-distal axis
            # so a nearest-point correspondence cannot pull a narrow spike toward
            # one arbitrary point on the neighboring surface.
            direction=np.asarray(frame.x,float)*(1.0 if side=='mesial' else -1.0)
            direction/=max(float(np.linalg.norm(direction)),1e-12)
            delta=raw_target-reference[ids[active]]
            lateral=(delta@direction)[:,None]*direction[None,:]
            raw_target=reference[ids[active]]+lateral+.15*(delta-lateral)
        active_ids=ids[active]
        desired[active_ids]=raw_target
        seeds.update(map(int,active_ids))
        seed_groups[side].update(map(int,active_ids))
    base={'stage':stage,'before':before,'after':before,'trials':[],
        'locked_vertex_count':int(locked.sum()),'transition_rings':int(transition_rings if transition_rings is not None else (7 if stage=='PROXIMAL' else 3)),'status':'REVIEW_REQUIRED','reason':None}
    if not any(r['valid_count'] for r in before.values()):
        base.update(status='NOT_ASSESSED',reason='NO_VALID_SURFACE_SAMPLES');return mesh.copy(),base,masks
    if not seeds:
        base.update(status='CONSTRAINT_CONFLICT' if unreachable else 'PASS_ENGINEERING',reason='TARGET_OUTSIDE_LOCAL_REACH' if unreachable else 'NO_CORRECTION_NEEDED');return mesh.copy(),base,masks
    adj=[set() for _ in reference]
    for a,b in np.asarray(mesh.edges_unique): adj[a].add(int(b));adj[b].add(int(a))

    # Crown 1.0：近远中接触必须来自连续 wall patch，不允许单点/少数点“长触角”碰邻牙。
    if stage=='PROXIMAL':
        support,patch_report=_broaden_proximal_targets(reference,desired,masks,seed_groups,locked,adj,frame)
        seeds|=support
        base['proximal_patch']=patch_report
        base['proximal_core_seed_count']=int(sum(len(v) for v in seed_groups.values()))
        base['proximal_support_seed_count']=int(len(support))

    seen=seeds|set(map(int,np.flatnonzero(locked)));front=seeds.copy();free=set()
    rings=int(transition_rings if transition_rings is not None else (7 if stage=='PROXIMAL' else 3))
    rings=max(1,min(rings,12))
    for _ in range(rings):
        front={v for u in front for v in adj[u]}-seen;seen|=front;free|=front
    free_ids=np.asarray(sorted(free),int);fixed=np.setdiff1d(np.arange(len(reference)),free_ids)
    for alpha in (1,.5,.25,.125,.0625):
        xt=reference.copy();ids=np.asarray(sorted(seeds));xt[ids]+=alpha*(desired[ids]-reference[ids])
        v=solve_local_band(reference,faces,free_ids,fixed,xt)[0] if len(free_ids) else xt
        moved=np.linalg.norm(v-reference,axis=1)
        gate=deformation_gate(reference,v,faces,scale_mm=.1,locked_ids=np.flatnonzero(locked),
            focus_faces=np.flatnonzero(np.any(moved[faces]>1e-9,axis=1)))
        candidate=mesh.copy();candidate.vertices=v
        measured={side:measure(v,masks[side],surface,targets[side],kind=stage,side=side) for side,surface in surfaces.items()}
        after={side:value[0] for side,value in measured.items()}
        key='target_error' if stage=='PROXIMAL' else 'deficit'
        same_valid=all(measured[s][1] is not None and measured[s][1][5][initial[s][1][5]].all() for s in before if initial[s][1] is not None)
        improved=same_valid and any(after[s].get(key) and after[s][key]['p95_mm']<before[s][key]['p95_mm']-1e-9 for s in before if before[s].get(key))
        nonworse=all(after[s].get(key) and after[s][key]['max_mm']<=before[s][key]['max_mm']+1e-9 for s in before if before[s].get(key))
        strict_flips=int((np.einsum('ij,ij->i',mesh.face_normals,candidate.face_normals)<=0).sum())
        safe=gate.ok and strict_flips==0 and not (candidate.area_faces<1e-10).any() and moved.max()<=2.5
        base['trials'].append({'alpha':alpha,'gate':gate.as_dict(),'all_flips':strict_flips,
            'maximum_displacement_mm':float(moved.max()),'improved':bool(improved),'after':after,'accepted':bool(safe and improved and nonworse)})
        if safe and improved and nonworse:
            base.update(after=after,status='PASS_ENGINEERING',accepted_alpha=alpha,
                locked_displacement_mm=float(moved[locked].max()) if locked.any() else 0)
            return candidate,base,masks
    base.update(status='CONSTRAINT_CONFLICT',reason='CONTACT_CONSTRAINT_CONFLICT' if stage=='PROXIMAL' else 'OCCLUSAL_CONSTRAINT_CONFLICT')
    return mesh.copy(),base,masks
