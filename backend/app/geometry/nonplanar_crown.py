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
        cross=lambda x,y:x[...,0]*y[...,1]-x[...,0]*0+y[...,0]*0+x[...,0]*y[...,1]-x[...,1]*y[...,0]
        # overwrite the compact expression above with the intended 2D cross
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


def _single_open_surface_above_margin(source, curtain, transition_mm, role, *, invert=False):
    """Extract one open cervical surface directly from a trusted source mesh."""
    values=-curtain.axial_above(source.vertices)
    vertices,faces,_=_clip_by_scalar(source.vertices,source.faces,values,level=-float(transition_mm))
    clipped=trimesh.Trimesh(vertices,faces,process=False)
    clipped.remove_unreferenced_vertices()
    parts=list(clipped.split(only_watertight=False))
    eligible=[p for p in parts if len(p.faces)>=3 and len(_boundary_loops(p.faces))==1]
    if not eligible:
        raise MarginBuildError(f'{role} 在 Margin 过渡面以上不能提取单一开放颈环面')
    chosen=max(eligible,key=lambda p: float(p.area))
    discarded_area=float(sum(float(p.area) for p in parts if p is not chosen))
    chosen.remove_unreferenced_vertices()
    if invert:
        chosen.invert()
    return chosen,{
        'role':role,
        'source_faces':int(len(source.faces)),
        'clipped_faces':int(len(clipped.faces)),
        'component_count':int(len(parts)),
        'eligible_single_loop_components':int(len(eligible)),
        'selected_faces':int(len(chosen.faces)),
        'selected_area_mm2':float(chosen.area),
        'discarded_component_area_mm2':discarded_area,
        'boundary_loop_count':int(len(_boundary_loops(chosen.faces))),
        'inverted_for_material_orientation':bool(invert),
    }


def _separate_cervical_source_surfaces(outside, inside, curtain, transition_mm,
                                       *, min_clearance_mm=0.18, band_mm=2.0,
                                       iterations=4):
    """Move only the *outer* cervical source skin outward when it crosses intaglio.

    DIRECT_SOURCE_CLIP keeps the intaglio authoritative because it comes from the
    prepared-abutment blockout + cement-gap field.  Real IOS meshes can leave the
    source outer skin only ~0.01 mm from that intaglio in the cervical offset band;
    the two skins then intersect hundreds of times before the stitch strip is even
    built.  This stage establishes a small engineering separation by moving the
    external cervical skin radially outward.  It never changes the confirmed
    Margin curve or the intaglio fit surface and fades out coronally.
    """
    result=outside.copy()
    total_moved=set();max_move=0.0;passes=[]
    matrix=np.asarray(curtain.frame.matrix(),float)
    origin=np.asarray(curtain.frame.origin,float)
    center=np.mean(np.asarray(curtain.local,float)[:,:2],axis=0)
    eps=0.02
    for iteration in range(max(1,int(iterations))):
        vertices=np.asarray(result.vertices,float).copy()
        height=curtain.axial_above(vertices)
        mask=(height>=float(transition_mm)-0.05)&(height<=float(transition_mm)+float(band_mm))
        ids=np.flatnonzero(mask)
        if not len(ids):
            break
        cp,dist,_=closest_point(inside,vertices[ids])
        dist=np.asarray(dist,float)
        bad=dist<float(min_clearance_mm)
        if not np.any(bad):
            passes.append({'iteration':iteration+1,'candidate_vertices':int(len(ids)),'moved_vertices':0,
                           'min_clearance_mm':float(dist.min()) if len(dist) else None})
            break
        chosen=ids[bad]
        local=(vertices[chosen]-origin)@matrix
        radial_local=np.c_[local[:,:2]-center,np.zeros(len(chosen))]
        rnorm=np.linalg.norm(radial_local,axis=1)
        fallback=rnorm<1e-9
        if np.any(fallback):
            sep=vertices[chosen[fallback]]-cp[bad][fallback]
            sep-=np.outer(sep@curtain.frame.z,curtain.frame.z)
            radial_local[fallback]=sep@matrix
            radial_local[fallback,2]=0.0
            rnorm=np.linalg.norm(radial_local,axis=1)
        rnorm=np.maximum(rnorm,1e-9)
        direction=(radial_local/rnorm[:,None])@matrix.T
        taper=1.0-np.clip((height[chosen]-float(transition_mm))/max(float(band_mm),1e-6),0.0,1.0)
        needed=np.maximum(float(min_clearance_mm)-dist[bad]+eps,0.0)
        displacement=needed*np.maximum(taper,0.20)
        vertices[chosen]+=direction*displacement[:,None]
        result.vertices=vertices
        total_moved.update(map(int,chosen.tolist()))
        max_move=max(max_move,float(displacement.max(initial=0.0)))
        passes.append({'iteration':iteration+1,'candidate_vertices':int(len(ids)),
                       'moved_vertices':int(len(chosen)),
                       'min_before_mm':float(dist[bad].min()),
                       'max_displacement_mm':float(displacement.max(initial=0.0))})
    final_vertices=np.asarray(result.vertices,float)
    final_h=curtain.axial_above(final_vertices)
    final_ids=np.flatnonzero((final_h>=float(transition_mm)-0.05)&
                             (final_h<=float(transition_mm)+float(band_mm)))
    if len(final_ids):
        _cp,final_dist,_=closest_point(inside,final_vertices[final_ids])
        final_dist=np.asarray(final_dist,float)
        min_after=float(final_dist.min())
        p05_after=float(np.quantile(final_dist,.05))
    else:
        min_after=None;p05_after=None
    return result,{
        'status':'APPLIED' if total_moved else 'NO_CORRECTION_NEEDED',
        'method':'RADIAL_OUTER_ONLY_CERVICAL_SOURCE_SEPARATION',
        'min_clearance_target_mm':float(min_clearance_mm),
        'band_mm':float(band_mm),
        'moved_vertices':int(len(total_moved)),
        'max_displacement_mm':float(max_move),
        'min_after_mm':min_after,
        'p05_after_mm':p05_after,
        'passes':passes,
        'intaglio_unchanged':True,
        'margin_curve_unchanged':True,
    }


def build_nonplanar_crown(outer,cutter,support,margin,axis,*,transition_mm,prepared_support=False,parameters=None,protected_surfaces=(),quality_mode='DESIGN'):
    curtain=MarginCurtain(margin,axis)
    if not np.isfinite(transition_mm) or transition_mm<=0:
        raise MarginBuildError('三维颈部过渡高度 margin_band_mm 必须为正值')
    scan=audit_margin_on_scan(support,curtain.points,expected_margin_boundary=prepared_support)
    from .topology_validation import record_topology
    record_topology('01_input_outer',outer)
    record_topology('02_input_cutter',cutter)
    boolean=crown_minus_cutter(outer,cutter)
    record_topology('03_boolean_crown_minus_cutter',boolean)
    from .boolean_healing import boolean_healing_stage
    try:
        boolean,boolean_healing=boolean_healing_stage(boolean)
    except Exception as exc:
        boolean_healing={'status':'ERROR_REVERTED','error':str(exc)[:200],
                         'note':'boolean healing 抛错 ⇒ 使用未愈合的 boolean（不劣化）'}
    values=-curtain.axial_above(boolean.vertices)
    vertices,faces,_=_clip_by_scalar(boolean.vertices,boolean.faces,values,level=-float(transition_mm))
    body=trimesh.Trimesh(vertices,faces,process=False)
    body.remove_unreferenced_vertices()
    record_topology('04_after_transition_clip',body)
    parts=list(body.split(only_watertight=False))

    surface_extraction={
        'method':'BOOLEAN_SPLIT',
        'boolean_component_count':int(len(parts)),
        'boolean_boundary_loops':[int(len(_boundary_loops(p.faces))) for p in parts],
        'fallback_used':False,
    }
    if len(parts)==2 and all(len(_boundary_loops(p.faces))==1 for p in parts):
        scores=[]
        for p in parts:
            sample=np.asarray(p.triangles_center)[np.linspace(0,len(p.faces)-1,min(128,len(p.faces)),dtype=int)]
            scores.append(float(closest_point(outer,sample)[1].mean()-closest_point(cutter,sample)[1].mean()))
        if min(scores)<0<max(scores):
            outside,inside=parts[int(np.argmin(scores))],parts[int(np.argmax(scores))]
            surface_extraction['boolean_source_scores']=scores
        else:
            surface_extraction['fallback_reason']='BOOLEAN_SOURCE_CLASSIFICATION_AMBIGUOUS'
            outside,outer_extract=_single_open_surface_above_margin(outer,curtain,transition_mm,'OUTER',invert=False)
            inside,inner_extract=_single_open_surface_above_margin(cutter,curtain,transition_mm,'INTAGLIO',invert=True)
            surface_extraction.update(method='DIRECT_SOURCE_CLIP',fallback_used=True,outer=outer_extract,intaglio=inner_extract)
    else:
        surface_extraction['fallback_reason']='BOOLEAN_NOT_TWO_SINGLE_LOOP_SURFACES'
        outside,outer_extract=_single_open_surface_above_margin(outer,curtain,transition_mm,'OUTER',invert=False)
        inside,inner_extract=_single_open_surface_above_margin(cutter,curtain,transition_mm,'INTAGLIO',invert=True)
        surface_extraction.update(method='DIRECT_SOURCE_CLIP',fallback_used=True,outer=outer_extract,intaglio=inner_extract)

    if surface_extraction.get('fallback_used'):
        outside,separation=_separate_cervical_source_surfaces(
            outside,inside,curtain,float(transition_mm),
            min_clearance_mm=0.18,band_mm=max(2.0,float(transition_mm)*2.0),iterations=4,
        )
        surface_extraction['cervical_separation']=separation

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
    from .cervical_cleanup import mesh_diagnostics
    cervical_ring_report={'status':'DIAGNOSTIC_ONLY_GEOMETRY_UNCHANGED',
        'reason':'bulk weld breaks watertight; coarse ring+collar regresses (5 -> 113 self-intersections, 98 degenerate faces)',
        'outer_before':mesh_diagnostics(outside,curtain.points),
        'inner_before':mesh_diagnostics(inside,curtain.points),
        'scope':'CROWN_CERVICAL_REGION_ONLY','margin_curve_untouched':True}
    final,seam=stitch_cervical_surfaces(outside,inside,curtain.points,quality_mode=quality_mode)
    n_outer=len(outside.vertices);n_inner=len(inside.vertices);seam_start=n_outer+n_inner
    f=np.asarray(final.faces)
    outer_faces=f[~np.any((f>=n_outer)&(f<seam_start),axis=1)]
    inner_faces=f[~np.any(f<n_outer,axis=1)]
    surfaces=[]
    for fs in (outer_faces,inner_faces):
        mesh=trimesh.Trimesh(np.asarray(final.vertices).copy(),fs,process=False)
        mesh.remove_unreferenced_vertices();surfaces.append(mesh)
    report={'status':'ENGINEERING_ONLY','method':'directional_source_surfaces_and_3d_cervical_dp_stitch',
        'margin':margin_plane_diagnostics(curtain.points),'scan_adherence':scan,'seam':seam,
        'transition_mm':float(transition_mm),'body_vertices_moved':int(np.sum(preservation>1e-6)),'body_outer_vertex_count':body_outer_count,'source_body_face_count':len(body.faces),
        'body_to_original_outer_p95_mm':float(np.quantile(preservation,.95)),
        'body_to_original_outer_max_mm':float(preservation.max()),
        'geometry_above_cervical_transition_preserved':not bool(enforcement and enforcement['max_displacement_mm']>1e-9),
        'margin_profile':'PARAMETRIC_BORDER_ENGINEERING' if border else 'SHARED_KNIFE_EDGE_NOT_MANUFACTURING_APPROVED',
        'crown_border':border,'minimum_thickness_enforcement':enforcement,
        'cervical_cleanup':cervical_ring_report,'boolean_healing':boolean_healing,
        'surface_extraction':surface_extraction,'quality_mode':quality_mode,
        'design_mode_exception':seam.get('design_mode_exception'),
        'stitch_warnings':([] if not seam.get('design_mode_exception') else ['Cervical stitch minor self intersection'])}
    return NonplanarCrownBuild(final,surfaces[0],surfaces[1],report)
