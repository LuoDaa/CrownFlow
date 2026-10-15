"""Derived fitting support, with explicitly confirmed virtual screw-channel caps."""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
import trimesh
from trimesh.proximity import closest_point
from .nonplanar_crown import MarginCurtain, MarginBuildError, audit_margin_on_scan
from .crown_outer.collar_body_outer import _clip_by_scalar, _boundary_loops
from .margin_seam import _polyline_distances

VERSION='abutment_support/margin_split_seed_flood/1.2'

class SupportExtractionError(MarginBuildError):
    def __init__(self,code,message,diagnostics=None):
        self.code=code
        self.diagnostics=diagnostics or {}
        super().__init__(message)

def mesh_digest(mesh):
    return sha256(mesh.export(file_type='stl')).hexdigest()

def _manifold(mesh):
    _,counts=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    if np.any(counts>2) or not mesh.is_winding_consistent:
        raise SupportExtractionError('NONMANIFOLD_SUPPORT','支撑存在非流形边或不一致面朝向')
    boundary=mesh.edges_unique[np.bincount(mesh.edges_unique_inverse)==1]
    if len(boundary) and np.any(np.bincount(boundary.ravel())[np.unique(boundary)]!=2):
        raise SupportExtractionError('NONMANIFOLD_SUPPORT','开放边界不是互不分叉的闭环')

def _inside(points,polygon):
    p=np.asarray(points);a=np.asarray(polygon);b=np.roll(a,-1,axis=0)
    cross=(a[None,:,1]>p[:,None,1])!=(b[None,:,1]>p[:,None,1])
    denominator=b[:,1]-a[:,1]
    intercept=a[None,:,0]+(p[:,None,1]-a[None,:,1])*(b-a)[None,:,0]/np.where(abs(denominator)>1e-12,denominator,1e-12)[None,:]
    return np.sum(cross&(p[:,None,0]<intercept),axis=1)%2==1

def _embed_boundary(mesh,source_ids,ring,margin):
    vertices=mesh.vertices.tolist();faces=mesh.faces.tolist();sources=source_ids.tolist();anchors=[];errors=[]
    for point in margin:
        point=np.asarray(point,dtype=float)
        if point.shape!=(3,) or not np.isfinite(point).all():
            raise SupportExtractionError('MARGIN_NOT_ON_SUPPORT','确认的终止线包含非有限坐标，无法嵌入支撑边界')
        face_array=np.asarray(faces,dtype=int)
        all_edges=np.sort(face_array[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1)
        unique_edges,counts=np.unique(all_edges,axis=0,return_counts=True)
        edges=unique_edges[counts==1]
        allowed=set(ring)
        edges=np.asarray([e for e in edges if e[0] in allowed and e[1] in allowed],dtype=int)
        if edges.size==0:
            raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','终止线对应边界为空，无法嵌入确认 Margin')
        edges=edges.reshape(-1,2)
        v=np.asarray(vertices,dtype=float)
        if not np.isfinite(v).all():
            raise SupportExtractionError('NONFINITE_SUPPORT_GEOMETRY','支撑网格含 NaN/Inf，禁止继续生成')
        a=v[edges[:,0]]
        delta=v[edges[:,1]]-a
        delta2=np.einsum('ij,ij->i',delta,delta)
        valid=np.isfinite(a).all(axis=1)&np.isfinite(delta).all(axis=1)&np.isfinite(delta2)&(delta2>1e-18)
        if not np.any(valid):
            raise SupportExtractionError('DEGENERATE_MARGIN_BOUNDARY','终止线边界全部为退化边，无法安全投影')
        edges=edges[valid];a=a[valid];delta=delta[valid];delta2=delta2[valid]
        numerator=np.einsum('ij,ij->i',point-a,delta)
        t=np.clip(numerator/delta2,0,1)
        projected=a+t[:,None]*delta
        distance=np.linalg.norm(projected-point,axis=1)
        finite=np.isfinite(projected).all(axis=1)&np.isfinite(distance)
        if not np.any(finite):
            raise SupportExtractionError('NONFINITE_MARGIN_PROJECTION','终止线投影产生非有限值，已中止以避免污染支撑网格')
        edges=edges[finite];projected=projected[finite];distance=distance[finite]
        j=int(np.argmin(distance));u,w=map(int,edges[j]);q=projected[j]
        if not np.isfinite(q).all() or not np.isfinite(distance[j]):
            raise SupportExtractionError('NONFINITE_MARGIN_PROJECTION','终止线投影结果无效，已中止')
        errors.append(float(distance[j]))
        if distance[j]>.05:
            raise SupportExtractionError('MARGIN_NOT_ON_SUPPORT','确认的终止线与切割边界偏差超过 0.05 mm，请重新核对，禁止暗改确认线')
        if np.linalg.norm(q-v[u])<1e-7:anchors.append(u);continue
        if np.linalg.norm(q-v[w])<1e-7:anchors.append(w);continue
        new=len(vertices);vertices.append(q.tolist());ring.append(new);anchors.append(new)
        split=False
        for i,face in enumerate(faces):
            if u not in face or w not in face:continue
            for k in range(3):
                x,y,z=face[k],face[(k+1)%3],face[(k+2)%3]
                if {x,y}=={u,w}:
                    faces[i]=[x,new,z];faces.append([new,y,z]);sources.append(sources[i]);split=True;break
            if split:break
        if not split:
            raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','终止线投影命中边无法在当前支撑拓扑中拆分')
    result=trimesh.Trimesh(vertices,faces,process=False)
    if not np.isfinite(np.asarray(result.vertices,dtype=float)).all():
        raise SupportExtractionError('NONFINITE_SUPPORT_GEOMETRY','Margin 嵌入后支撑网格含 NaN/Inf，已拒绝输出')
    return result,np.asarray(sources),{'embedded_vertex_ids':anchors,'projection_distance_max_mm':max(errors) if errors else 0.0,'confirmed_margin_moved':False}

@dataclass
class PreparedAbutmentSupport:
    mesh:trimesh.Trimesh
    actual_surface:trimesh.Trimesh
    report:dict

def margin_project_added_vertices(vertices,curve,*,n_original,cap_mm=1.2):
    from trimesh.proximity import closest_point
    import trimesh as _tm
    vertices=np.asarray(vertices,float);points=np.asarray(curve,float)
    if len(vertices)<=n_original:return vertices,0
    ids=np.arange(n_original,len(vertices))
    _,distance,_=closest_point(_tm.Trimesh(vertices=np.asarray(curve,float),
        faces=np.column_stack([np.arange(len(points)-1),np.arange(1,len(points)),
                               np.arange(1,len(points))]),process=False),vertices[ids])
    moved=0;out=vertices.copy()
    near=distance<=cap_mm
    if near.any():
        a=points[:-1];b=points[1:];delta=b-a
        for k in ids[near]:
            p=out[k]
            t_=np.clip(np.einsum('ij,ij->i',p-a,delta)/np.maximum(np.einsum('ij,ij->i',delta,delta),1e-18),0,1)
            proj=a+t_[:,None]*delta
            out[k]=proj[int(np.argmin(np.linalg.norm(proj-p,axis=1)))]
            moved+=1
    return out,moved


def margin_ribbon_field(support,curve):
    from scipy.spatial import cKDTree
    from trimesh.proximity import closest_point
    vertices=np.asarray(support.vertices,float);points=np.asarray(curve,float)
    curve_distance,curve_face_ids=closest_point(support,points)[:2]
    curve_normals=np.asarray(support.face_normals,float)[np.asarray(curve_face_ids,int)]
    distance,nearest_id=cKDTree(points).query(vertices)
    offset=np.einsum('ij,ij->i',vertices-points[nearest_id],curve_normals[nearest_id])
    return offset,distance


def _rank_margin_components(cut,faces,labels,vertices,curve,seed=None):
    """Rank distinct post-cut components whose boundary reproduces the Margin."""
    candidates=[]
    seed_point=None if seed is None else np.asarray(seed,float)
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        component_loops=[]
        for loop in _boundary_loops(faces[ids]):
            error=_polyline_distances(vertices[loop],curve)
            p95=float(np.quantile(error,.95));mx=float(np.max(error))
            if p95<.05 and mx<.1:
                component_loops.append((p95,mx,np.asarray(loop,int)))
        if not component_loops:
            continue
        component_loops.sort(key=lambda item:(item[0],item[1],-len(item[2])))
        p95,mx,loop=component_loops[0]
        area=float(np.sum(cut.area_faces[ids]))
        seed_distance=None
        if seed_point is not None:
            component=trimesh.Trimesh(vertices,faces[ids],process=False)
            try:
                _,d,_=closest_point(component,seed_point.reshape(1,3));seed_distance=float(d[0])
            except Exception:
                seed_distance=None
        candidates.append({'label':int(label),'loop':loop,'area_mm2':area,'face_count':int(len(ids)),
            'margin_p95_mm':p95,'margin_max_mm':mx,'seed_distance_mm':seed_distance})
    return candidates


def _select_margin_component(candidates,seed=None):
    public=[{k:v for k,v in c.items() if k!='loop'} for c in candidates]
    diagnostics={'method':'MARGIN_MATCH_COMPONENT_RANK_V2','candidate_count':len(candidates),'candidates':public}
    if not candidates:
        return None,diagnostics
    if len(candidates)==1:
        diagnostics.update(selection='UNIQUE_MARGIN_MATCH',confidence='high',selected_label=candidates[0]['label'])
        return candidates[0],diagnostics

    ranked=list(candidates)
    if seed is not None and any(c['seed_distance_mm'] is not None for c in ranked):
        ranked.sort(key=lambda c:(float('inf') if c['seed_distance_mm'] is None else c['seed_distance_mm'],-c['area_mm2'],c['margin_p95_mm']))
        first,second=ranked[0],ranked[1]
        d0=first['seed_distance_mm'];d1=second['seed_distance_mm']
        if d0 is not None and (d1 is None or d0+0.15<d1):
            diagnostics.update(selection='SEED_DOMINANT',confidence='high',selected_label=first['label'])
            return first,diagnostics

    ranked.sort(key=lambda c:(-c['area_mm2'],c['margin_p95_mm'],c['margin_max_mm'],-c['face_count']))
    first,second=ranked[0],ranked[1]
    area_ratio=first['area_mm2']/max(second['area_mm2'],1e-12)
    face_ratio=first['face_count']/max(second['face_count'],1)
    error_better=(first['margin_p95_mm']+0.005<second['margin_p95_mm'] and first['margin_p95_mm']<=0.70*max(second['margin_p95_mm'],1e-9))
    dominant=area_ratio>=1.20 or face_ratio>=1.25 or error_better
    diagnostics.update(area_ratio_to_second=float(area_ratio),face_ratio_to_second=float(face_ratio))
    if dominant:
        diagnostics.update(selection='DOMINANT_MARGIN_CONNECTED_REGION',confidence='engineering_high',selected_label=first['label'])
        return first,diagnostics
    diagnostics.update(selection='AMBIGUOUS',confidence='low',selected_label=None)
    return None,diagnostics


def extract_abutment_support(support,margin,axis,*,seed=None,confirmed_loop_ids=()):
    """Split the existing abutment at the authoritative Margin and select support."""
    original_hash=mesh_digest(support);_manifold(support)
    curtain=MarginCurtain(margin,axis);curve=curtain.points
    _,dist,margin_faces=closest_point(support,curve)
    if dist.max()>.05:raise SupportExtractionError('MARGIN_NOT_ON_SUPPORT','终止线未贴合实体基台，最大偏差超过 0.05 mm')
    audit_margin_on_scan(support,curve)
    source_labels=trimesh.graph.connected_component_labels(support.face_adjacency,node_count=len(support.faces))
    identities=np.unique(source_labels[margin_faces])
    if len(identities)!=1:raise SupportExtractionError('MULTIPLE_SUPPORT_COMPONENTS','终止线涉及多个支撑组件，需明确目标')
    chosen=int(identities[0]);source_faces=np.flatnonzero(source_labels==chosen)
    if seed is not None:
        point=np.asarray(seed,float)
        if point.shape!=(3,) or not np.isfinite(point).all():raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','支撑种子必须是有限三维点')
        _,distance,ids=closest_point(support,point.reshape(1,3))
        if distance[0]>.5 or source_labels[ids[0]]!=chosen:raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','支撑种子与确认终止线不属于同一表面')

    vertices,faces,added,_mapping=_clip_by_scalar(support.vertices,support.faces[source_faces],-curtain.axial_above(support.vertices),return_face_sources=True)
    kept=curtain.axial_above(vertices)[faces].max(axis=1)>1e-9
    faces=faces[kept];mapping=source_faces[_mapping[kept]]
    if not len(faces):raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','终止线以上没有可用支撑面')
    vertices,projected=margin_project_added_vertices(vertices,curve,n_original=len(support.vertices))
    if not np.isfinite(np.asarray(vertices,dtype=float)).all():
        raise SupportExtractionError('NONFINITE_SUPPORT_GEOMETRY','Margin 投影后支撑网格含 NaN/Inf，禁止继续')
    cut=trimesh.Trimesh(vertices,faces,process=False);_manifold(cut)
    labels=trimesh.graph.connected_component_labels(cut.face_adjacency,node_count=len(faces))

    candidates=_rank_margin_components(cut,faces,labels,vertices,curve,seed=seed)
    selected,selection=_select_margin_component(candidates,seed=seed)
    if selected is None:
        raise SupportExtractionError('SUPPORT_EXTRACTION_AMBIGUOUS','无法唯一识别与确认终止线对应的支撑区域',selection)
    label=int(selected['label'])

    ids=np.flatnonzero(labels==label)
    mesh=trimesh.Trimesh(vertices,faces[ids],process=False);mesh.remove_unreferenced_vertices();mapping=mapping[ids]
    loops=_boundary_loops(mesh.faces)
    crown=min(loops,key=lambda loop:np.max(_polyline_distances(mesh.vertices[loop],curve)))
    mesh,mapping,embedding=_embed_boundary(mesh,mapping,crown.copy(),curve)
    if not np.isfinite(np.asarray(mesh.vertices,dtype=float)).all():
        raise SupportExtractionError('NONFINITE_SUPPORT_GEOMETRY','支撑提取产物含 NaN/Inf，禁止标记 READY')
    loops=_boundary_loops(mesh.faces)
    crown=min(loops,key=lambda loop:np.max(_polyline_distances(mesh.vertices[loop],curve)))
    ring_local=curtain.local[:,:2];next_local=np.roll(ring_local,-1,axis=0)
    area=abs(np.sum(ring_local[:,0]*next_local[:,1]-ring_local[:,1]*next_local[:,0]))/2
    radius=np.sqrt(area/np.pi);classified=[];operations=[];cap_faces=[]
    for loop in loops:
        points=mesh.vertices[loop];local=(points-curtain.frame.origin)@curtain.frame.matrix()
        loop_id=sha256(np.round(np.sort(points,axis=0),6).tobytes()).hexdigest()[:20]
        if loop==crown:kind='CROWN_MARGIN'
        else:
            centered=local[:,:2]-local[:,:2].mean(0);_,_,basis=np.linalg.svd(points-points.mean(0),full_matrices=False)
            candidate=(_inside(local[:,:2],ring_local).all() and np.median(local[:,2])>np.max(curtain.local[:,2])+.3
                and np.linalg.norm(local[:,:2].mean(0))<.55*radius and np.max(np.linalg.norm(centered,axis=1))<.65*radius
                and abs(basis[-1]@curtain.frame.z)>.7)
            kind='SCREW_CHANNEL_CANDIDATE' if candidate else 'UNKNOWN_HOLE'
        entry={'loop_id':loop_id,'classification':kind,'points':points.tolist(),'confirmed':loop_id in confirmed_loop_ids}
        classified.append(entry)
        if kind!='SCREW_CHANNEL_CANDIDATE' or not entry['confirmed']:continue
        import mapbox_earcut
        triangles=np.asarray(loop)[mapbox_earcut.triangulate_float64(np.ascontiguousarray(local[:,:2]),np.array([len(loop)],np.uint32)).reshape(-1,3)]
        u,w=map(int,loop[:2]);existing=next(face for face in mesh.faces if u in face and w in face)
        source_direction=any(existing[k]==u and existing[(k+1)%3]==w for k in range(3))
        cap_direction=any(face[k]==u and face[(k+1)%3]==w for face in triangles for k in range(3))
        if source_direction==cap_direction:triangles=triangles[:,::-1]
        before=mesh_digest(mesh);start=len(mesh.faces)
        mesh=trimesh.Trimesh(mesh.vertices,np.vstack((mesh.faces,triangles)),process=False);_manifold(mesh)
        cap_faces.extend(range(start,len(mesh.faces)));touch=np.any(np.isin(mesh.faces[:start],loop),axis=1)
        operations.append({'operation':'VIRTUAL_SCREW_CHANNEL_CAP','loop_id':loop_id,
            'source_face_ids':sorted(set(mapping[touch[:len(mapping)]].tolist())),
            'before_sha256':before,'after_sha256':mesh_digest(mesh),'cap_face_ids':list(range(start,len(mesh.faces))),
            'area_mm2':float(mesh.area_faces[start:].sum())})
    cut_points=mesh.vertices[crown] if len(crown) else np.zeros((0,3))
    cut_distance=_polyline_distances(cut_points,curve) if len(cut_points) else np.zeros(0)
    margin_cut={'method':'margin_ribbon_along_support_normals/1','margin_curve_points':int(len(curve)),
        'cut_boundary_points':int(len(cut_points)),
        'point_to_margin_distance_mm':({'mean_mm':float(np.mean(cut_distance)),
            'p95_mm':float(np.quantile(cut_distance,.95)),'max_mm':float(np.max(cut_distance)),
            'gate_mm':0.05,'passes':bool(np.quantile(cut_distance,.95)<0.05)} if len(cut_distance) else None)}
    pending=[entry for entry in classified if entry['classification']=='SCREW_CHANNEL_CANDIDATE' and not entry['confirmed']]
    unknown=[entry for entry in classified if entry['classification']=='UNKNOWN_HOLE']
    report={'algorithm_version':VERSION,'loops':classified,'source_face_ids':sorted(set(mapping.tolist())),
        'output_face_source_ids':mapping.tolist()+[-1]*len(cap_faces),'source_to_output_mapping':{str(i):np.flatnonzero(mapping==i).tolist() for i in np.unique(mapping)},
        'margin_embedding':embedding,'support_region_selection':selection,
        'screw_channel_present':any(e['classification']=='SCREW_CHANNEL_CANDIDATE' for e in classified),
        'virtual_cap_operations':operations,'virtual_cap_face_ids':cap_faces,'capped_region_area_mm2':sum(o['area_mm2'] for o in operations),
        'unresolved_support_holes':len(pending)+len(unknown),'source_geometry_sha256':original_hash,'conditioned_support_sha256':mesh_digest(mesh),
        'readiness':'REVIEW_REQUIRED' if pending or unknown else 'ABUTMENT_SUPPORT_READY',
        'rejection_code':'UNKNOWN_SUPPORT_HOLE' if unknown else 'SCREW_CHANNEL_CONFIRMATION_REQUIRED' if pending else 'READY',
        'source_modified':False,'component_count':1,'boundary_loop_count':len(_boundary_loops(mesh.faces)),
        'scope':'AXIS_SINGLE_VALUED_CROWN_FINISH_LINE','margin_cut':{**margin_cut,'projected_boundary_vertices':int(projected)}}
    actual=trimesh.Trimesh(mesh.vertices,mesh.faces[:len(mapping)],process=False);actual.remove_unreferenced_vertices()
    if not np.isfinite(np.asarray(actual.vertices,dtype=float)).all():
        raise SupportExtractionError('NONFINITE_SUPPORT_GEOMETRY','实际基台表面含 NaN/Inf，禁止输出')
    if mesh_digest(support)!=original_hash:raise RuntimeError('source support changed during extraction')
    return PreparedAbutmentSupport(mesh,actual,report)
