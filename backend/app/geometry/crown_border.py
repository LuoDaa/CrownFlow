"""Explicit crown-bottom sections around the confirmed curve; not material approval."""
import numpy as np
import trimesh
from .nonplanar_crown import MarginCurtain, MarginBuildError
from .crown_outer.collar_body_outer import _boundary_loops, _tris_from_path


def _section_strip(vertices, first, second, frame):
    """Preserve circumferential correspondence instead of minimizing 3D diagonal length."""
    rings=[];arcs=[]
    for ids in (first,second):
        ids=np.asarray(ids,int);local=(vertices[ids]-frame.origin)@frame.matrix()
        area=np.sum(local[:,0]*np.roll(local[:,1],-1)-local[:,1]*np.roll(local[:,0],-1))
        if area<0:ids=ids[::-1];local=local[::-1]
        start=np.argmin(np.mod(np.arctan2(local[:,1],local[:,0]),2*np.pi))
        ids=np.roll(ids,-start);v=vertices[ids]
        length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(np.vstack((v,v[:1])),axis=0),axis=1))]
        rings.append(ids);arcs.append(length/length[-1])
    a,b=rings;u,w=arcs;i=j=0;path=[]
    while i<len(a) or j<len(b):
        if j==len(b) or (i<len(a) and u[i+1]<=w[j+1]):path.append('a');i+=1
        else:path.append('b');j+=1
    return _tris_from_path(a,b,path)


def build_crown_border(outside, margin, axis, parameters, transition_mm):
    curtain=MarginCurtain(margin,axis);curve=curtain.points
    horizontal=float(parameters.get('border_horizontal_mm',0))
    angled=float(parameters.get('border_angled_mm',0))
    angle=np.deg2rad(float(parameters.get('border_angle_deg',45)))
    vertical=float(parameters.get('border_vertical_mm',0))
    if not np.isfinite([horizontal,angled,angle,vertical]).all() or horizontal<=0 or angled<0 or vertical<0 or not 0<angle<np.pi/2:
        raise MarginBuildError('冠边缘截面参数无效')
    height=angled*np.cos(angle)+vertical
    if height>=transition_mm-1e-5:
        raise MarginBuildError('边缘截面高度必须小于已确认颈部过渡高度')
    xy=curtain.local[:,:2];edge=np.roll(xy,-1,axis=0)-xy
    edge/=np.linalg.norm(edge,axis=1)[:,None]
    signed=np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1))
    normal=np.c_[edge[:,1],-edge[:,0]]*np.sign(signed)
    bisector=normal+np.roll(normal,1,axis=0)
    denominator=np.einsum('ij,ij->i',bisector,normal)
    if np.any(denominator<.1):raise MarginBuildError('终止线存在尖角，边缘偏置不可靠，请修正曲线')
    offset=(bisector/denominator[:,None])@curtain.frame.matrix()[:,:2].T
    rings=[curve+horizontal*offset]
    if angled: rings.append(curve+(horizontal+angled*np.sin(angle))*offset+angled*np.cos(angle)*curtain.frame.z)
    if vertical:rings.append(rings[-1]+vertical*curtain.frame.z)
    for ring in rings:MarginCurtain(ring,axis) # Reject offset self-crossings rather than fixing the truth curve.
    loops=_boundary_loops(outside.faces)
    if len(loops)!=1:raise MarginBuildError('外冠必须有且仅有一个颈缘环')
    vertices=np.vstack((outside.vertices,*rings));faces=[np.asarray(outside.faces)]
    ids=np.asarray(loops[0]);start=len(outside.vertices)
    # Section rings have prescribed correspondence; arbitrary DP fans can fold a noncircular neck.
    for i in range(len(rings)-1,-1,-1):
        target=np.arange(start+i*len(curve),start+(i+1)*len(curve))
        faces.append(_section_strip(vertices,ids,target,curtain.frame));ids=target
    result=trimesh.Trimesh(vertices,np.vstack(faces),process=False)
    trimesh.repair.fix_normals(result)
    return result,{'algorithm':'crown_border_sections/1','status':'ENGINEERING_ONLY',
        'horizontal_mm':horizontal,'angled_mm':angled,'angle_deg':float(np.rad2deg(angle)),
        'vertical_mm':vertical,'below_margin_mm':0,'profile_height_mm':height,
        'margin_position_preserved':True,'manufacturer_rule_validated':False,
        'body_thickness_domain_start_mm':float(transition_mm),
        'horizontal_land_min_mm':float(np.min(np.linalg.norm(rings[0]-curve,axis=1)))}
