"""Local cervical reconstruction; anatomical body remains the Boolean input body."""
from dataclasses import dataclass
import numpy as np
import trimesh
from trimesh.proximity import closest_point

from .boolean import crown_minus_cutter
from .crown_outer.collar_body_outer import _clip_by_scalar, _boundary_loops
from .crown_outer.patient_frame import build_patient_frame
from .margin_seam import stitch_cervical_surfaces, margin_plane_diagnostics

#: clean cervical ring 的弧长等距采样点数（工程默认）
CERVICAL_RING_SAMPLES = 128


class MarginBuildError(ValueError):
    pass


class MarginCurtain:
    """An axis-single-valued curve, never its fitted plane."""
    def __init__(self,points,axis):
        self.points=np.asarray(points,float)
        margin_plane_diagnostics(self.points)
        if np.array_equal(self.points[0],self.points[-1]):
            self.points=self.points[:-1]
        self.frame=build_patient_frame(self.points,np.asarray(axis,float))
        self.local=(self.points-self.frame.origin)@self.frame.matrix()
        a=self.local[:,:2];b=np.roll(a,-1,axis=0)
        self.delta=b-a;self.length2=np.sum(self.delta**2,axis=1)
        if np.any(self.length2<1e-16):
            raise MarginBuildError('Margin 投影包含重复点或轴向边，当前重建不支持该曲线')
        # A projected crossing is ambiguous along the insertion axis even if Z differs.
        cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
        for i in range(len(a)):
            ids=np.arange(i+2,len(a))
            if i==0: ids=ids[ids!=len(a)-1]
            if not len(ids): continue
            p,q=a[ids],b[ids]
            o1=cross(b[i]-a[i],p-a[i]);o2=cross(b[i]-a[i],q-a[i])
            o3=cross(q-p,a[i]-p);o4=cross(q-p,b[i]-p)
            overlaps=np.all(np.minimum(a[i],b[i])<=np.maximum(p,q)+1e-10,axis=1)&np.all(np.minimum(p,q)<=np.maximum(a[i],b[i])+1e-10,axis=1)
            if np.any(overlaps&(o1*o2<=1e-20)&(o3*o4<=1e-20)):
                raise MarginBuildError('Margin 自交或投影自交，不允许自动补线')
        import mapbox_earcut
        indices=mapbox_earcut.triangulate_float64(np.ascontiguousarray(a),np.array([len(a)],dtype=np.uint32))
        self.cap=self.local[indices.reshape(-1,3)]

    def height(self,points):
        local=(np.asarray(points)-self.frame.origin)@self.frame.matrix()
        heights=[]
        for chunk in np.array_split(local,max(1,len(local)//512+1)):
            rel=chunk[:,None,:2]-self.local[None,:,:2]
            t=np.clip(np.einsum('nki,ki->nk',rel,self.delta)/self.length2,0,1)
            d2=np.sum((rel-t[:,:,None]*self.delta)**2,axis=2)
            idx=np.argmin(d2,axis=1);rows=np.arange(len(chunk))
            z=self.local[idx,2]+t[rows,idx]*(np.roll(self.local[:,2],-1)[idx]-self.local[idx,2])
            tri=self.cap
            e1=tri[:,1,:2]-tri[:,0,:2];e2=tri[:,2,:2]-tri[:,0,:2]
            cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
            denominator=cross(e1,e2)
            if np.any(np.abs(denominator)<1e-16):
                raise MarginBuildError('Margin 投影三角化退化')
            offset=chunk[:,None,:2]-tri[None,:,0,:2]
            u=cross(offset,e2)/denominator;v=cross(e1,offset)/denominator
            inside=(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)
            enclosed=inside.any(axis=1);face=inside.argmax(axis=1)
            weights=np.c_[1-u[rows,face]-v[rows,face],u[rows,face],v[rows,face]]
            z[enclosed]=np.sum(weights[enclosed]*tri[face[enclosed],:,2],axis=1)
            heights.extend(z)
        return np.asarray(heights)

    def axial_above(self,points):
        return (np.asarray(points)-self.frame.origin)@self.frame.z-self.height(points)


def audit_margin_on_scan(support,curve,*,expected_margin_boundary=False):
    """Reject samples escaping onto an open scan edge; report adherence, not fit PASS."""
    edges=np.asarray(support.edges_unique_length)
    spacing=float(np.median(edges[edges>1e-8]))/2
    samples=[]
    for a,b in zip(curve,np.roll(curve,-1,axis=0)):
        count=max(2,int(np.ceil(np.linalg.norm(b-a)/max(spacing,1e-6)))+1)
        if count>10000: raise MarginBuildError('Margin 段跨越范围过大，请重新绘制')
        samples.extend(a+(b-a)*np.linspace(0,1,count)[:,None])
    samples=np.asarray(samples)
    query=[closest_point(support,samples[i:i+512]) for i in range(0,len(samples),512)]
    distances=np.concatenate([r[1] for r in query]);face_ids=np.concatenate([r[2] for r in query])
    faces=np.asarray(support.faces)
    boundary=_boundary_loops(faces)
    boundary_vertices={i for loop in boundary for i in loop}
    if expected_margin_boundary:
        # A derived, explicitly classified crown boundary is not a missing scan edge.
        from .margin_seam import _polyline_distances
        ids=np.asarray(sorted(boundary_vertices),int)
        near=_polyline_distances(support.vertices[ids],curve)<=.05
        boundary_vertices=set(ids[~near].tolist())
        if distances.max()>.05:raise MarginBuildError('确认终止线与派生支撑偏差超过 0.05 mm')
    boundary_faces=np.any(np.isin(faces,list(boundary_vertices)),axis=1)
    if np.any((distances>1e-5)&boundary_faces[face_ids]):
        raise MarginBuildError('Margin 跨扫描开放边界或缺口，不允许自动连接')
    return {'status':'REVIEW_REQUIRED','sample_count':len(samples),'scan_distance_max_mm':float(distances.max()),
        'scan_distance_p95_mm':float(np.quantile(distances,.95)),'acceptance_limit_mm':None,
        'method':'polyline_samples_to_exact_scan_triangles','scan_adherence_approved':False}


@dataclass
class NonplanarCrownBuild:
    final:trimesh.Trimesh
    outer_surface:trimesh.Trimesh
    intaglio_surface:trimesh.Trimesh
    report:dict


def fitting_support_above_margin(support,margin,axis,*,prepared=False):
    curtain=MarginCurtain(margin,axis)
    scan=audit_margin_on_scan(support,curtain.points,expected_margin_boundary=prepared)
    if prepared:surface=support.copy()
    else:
        vertices,faces,_=_clip_by_scalar(support.vertices,support.faces,-curtain.axial_above(support.vertices))
        if len(faces): faces=faces[curtain.axial_above(vertices)[faces].max(axis=1)>1e-9]
        surface=trimesh.Trimesh(vertices,faces,process=False)
    surface.remove_unreferenced_vertices()
    if len(surface.split(only_watertight=False))!=1 or len(_boundary_loops(surface.faces))!=1:
        raise MarginBuildError('Margin 以上基牙不是单一完整就位面；存在缺口或多余组件，请人工复核来源')
    return surface,{'source_faces':len(support.faces),'fitting_faces':len(surface.faces),
        'method':'3d_margin_curtain_above_domain_no_model_base_projection','scan_adherence':scan}


def build_nonplanar_crown(outer,cutter,support,margin,axis,*,transition_mm,prepared_support=False,parameters=None,protected_surfaces=(),quality_mode='DESIGN'):
    curtain=MarginCurtain(margin,axis)
    if not np.isfinite(transition_mm) or transition_mm<=0:
        raise MarginBuildError('三维颈部过渡高度 margin_band_mm 必须为正值')
    scan=audit_margin_on_scan(support,curtain.points,expected_margin_boundary=prepared_support)
    # 生成出口拓扑校验（只读）：定位 duplicate vertex 首次出现的阶段
    from .topology_validation import record_topology
    record_topology('01_input_outer',outer)
    record_topology('02_input_cutter',cutter)
    boolean=crown_minus_cutter(outer,cutter)
    record_topology('03_boolean_crown_minus_cutter',boolean)
    # **boolean 输出后的安全愈合**（新增阶段）：boolean 是 duplicate vertex 的首个来源
    # （1,466 对 <0.01mm、最小边 1.2e-05mm）。这里做逐条 tiny edge 的 link-condition 折叠
    # + 删退化面 + 重建邻接 + 重定向法向，每一步校验 boundary/nonmanifold/components/watertight，
    # 任一变化立即回退（不用简单 merge vertices）。
    from .boolean_healing import boolean_healing_stage
    try:
        boolean,boolean_healing=boolean_healing_stage(boolean)
    except Exception as exc:      # 安全网：愈合阶段有 bug 时绝不拖垮 finalize，如实记录并退回原始 boolean
        boolean_healing={'status':'ERROR_REVERTED','error':str(exc)[:200],
                         'note':'boolean healing 抛错 ⇒ 使用未愈合的 boolean（不劣化）'}
    values=-curtain.axial_above(boolean.vertices)
    vertices,faces,_=_clip_by_scalar(boolean.vertices,boolean.faces,values,level=-float(transition_mm))
    body=trimesh.Trimesh(vertices,faces,process=False)
    body.remove_unreferenced_vertices()
    record_topology('04_after_transition_clip',body)
    parts=body.split(only_watertight=False)
    if len(parts)!=2 or any(len(_boundary_loops(p.faces))!=1 for p in parts):
        raise MarginBuildError('布尔主体不能分离为两个单颈环面；请检查薄壁、破损或过渡范围')
    scores=[]
    for p in parts:
        sample=np.asarray(p.triangles_center)[np.linspace(0,len(p.faces)-1,min(128,len(p.faces)),dtype=int)]
        scores.append(float(closest_point(outer,sample)[1].mean()-closest_point(cutter,sample)[1].mean()))
    if min(scores)>=0 or max(scores)<=0:
        raise MarginBuildError('内外面来源无法可靠区分')
    outside,inside=parts[int(np.argmin(scores))],parts[int(np.argmax(scores))]
    record_topology('05_split_outside',outside)
    record_topology('06_split_inside',inside)
    parameters=parameters or {};enforcement=None;border=None
    if parameters.get('enforce_minimum_thickness'):
        if not parameters.get('border_horizontal_mm'):
            raise MarginBuildError('自动补厚需要明确启用制造边缘截面；刀口模式不能作为制造厚度约束')
        from .minimum_thickness import enforce_body_thickness
        outside,enforcement=enforce_body_thickness(outside,inside,float(parameters['minimum_thickness_mm']),protected_surfaces=protected_surfaces)
    body_outer_count=len(outside.vertices)
    preservation=np.concatenate([closest_point(outer,np.asarray(outside.vertices)[i:i+512])[1] for i in range(0,len(outside.vertices),512)])
    if parameters.get('border_horizontal_mm'):
        from .crown_border import build_crown_border
        outside,border=build_crown_border(outside,curtain.points,axis,parameters,float(transition_mm))
    # Finalize 前的颈缘处理阶段：**当前只诊断、不改几何**。
    # 实测（两轮）：(a) 直接清理输入 mesh（bulk 顶点合并）会破坏 watertight；
    # (b) 用"弧长等距重采样 + collar 带"替换颈环更糟 —— 原颈环本身比目标采样**更密**
    #     （329 点 / 中位间距 ~0.067mm），降到 128 点等于要求 collar 做抽稀：
    #     self_intersections 5 → 113，并新增 98 个退化面（A_margin 94 / OUTER+OUTER 7）。
    # 因此两者都不启用，几何保持不动；只输出诊断，等确定"不劣化"的方案。
    from .cervical_cleanup import mesh_diagnostics
    cervical_ring_report={'status':'DIAGNOSTIC_ONLY_GEOMETRY_UNCHANGED',
        'reason':'bulk weld breaks watertight; coarse ring+collar regresses (5 -> 113 self-intersections, 98 degenerate faces)',
        'outer_before':mesh_diagnostics(outside,curtain.points),
        'inner_before':mesh_diagnostics(inside,curtain.points),
        'scope':'CROWN_CERVICAL_REGION_ONLY','margin_curve_untouched':True}
    final,seam=stitch_cervical_surfaces(outside,inside,curtain.points,quality_mode=quality_mode)
    # Recover the two open surfaces using the same shared Margin vertex indices.
    n_outer=len(outside.vertices);n_inner=len(inside.vertices);seam_start=n_outer+n_inner
    f=np.asarray(final.faces)
    outer_faces=f[~np.any((f>=n_outer)&(f<seam_start),axis=1)]
    inner_faces=f[~np.any(f<n_outer,axis=1)]
    surfaces=[]
    for fs in (outer_faces,inner_faces):
        mesh=trimesh.Trimesh(np.asarray(final.vertices).copy(),fs,process=False)
        mesh.remove_unreferenced_vertices();surfaces.append(mesh)
    report={'status':'ENGINEERING_ONLY','method':'directional_boolean_body_and_3d_cervical_dp_stitch',
        'margin':margin_plane_diagnostics(curtain.points),'scan_adherence':scan,'seam':seam,
        'transition_mm':float(transition_mm),'body_vertices_moved':int(np.sum(preservation>1e-6)),'body_outer_vertex_count':body_outer_count,'source_body_face_count':len(body.faces),
        'body_to_original_outer_p95_mm':float(np.quantile(preservation,.95)),
        'body_to_original_outer_max_mm':float(preservation.max()),
        'geometry_above_cervical_transition_preserved':not bool(enforcement and enforcement['max_displacement_mm']>1e-9),
        'margin_profile':'PARAMETRIC_BORDER_ENGINEERING' if border else 'SHARED_KNIFE_EDGE_NOT_MANUFACTURING_APPROVED',
        'crown_border':border,'minimum_thickness_enforcement':enforcement,
        'cervical_cleanup':cervical_ring_report,'boolean_healing':boolean_healing,'quality_mode':quality_mode,'design_mode_exception':seam.get('design_mode_exception'),'stitch_warnings':([] if not seam.get('design_mode_exception') else ['Cervical stitch minor self intersection'])}
    return NonplanarCrownBuild(final,surfaces[0],surfaces[1],report)
