"""Reviewed library proposals, not automatic global template approval."""
import numpy as np
import trimesh
import mapbox_earcut
from ..geometry.crown_outer.template import load_crown_template
from ..geometry.crown_outer.collar_body_outer import _boundary_loops
from ..geometry.nonplanar_crown import MarginCurtain
from ..geometry.registration import initialize_from_margin


def canonical_library_crown(library,fdi):
    template=load_crown_template(library,fdi)
    retained=template.crown_vertex_mask.copy();retained[template.margin_loop_indices]=True
    faces=np.all(retained[template.faces],axis=1)
    open_crown=trimesh.Trimesh(template.vertices,template.faces,process=False).submesh([np.flatnonzero(faces)],append=True,repair=False)
    loops=_boundary_loops(open_crown.faces)
    if len(loops)!=1 or len(open_crown.split(only_watertight=False))!=1:
        raise ValueError('牙库冠部存在额外缺口或组件，不允许自动修补')
    ring=np.asarray(loops[0],int)
    MarginCurtain(open_crown.vertices[ring],[0,0,1])
    triangles=mapbox_earcut.triangulate_float64(np.asarray(open_crown.vertices[ring,:2],np.float64),np.array([len(ring)],np.uint32)).reshape(-1,3)
    # Intentional cervical template cap; this does not fill a patient scan defect.
    open_crown.faces=np.vstack((open_crown.faces,ring[triangles]));trimesh.repair.fix_normals(open_crown)
    if not open_crown.is_watertight or open_crown.volume<=0: raise ValueError('冠模板封底未形成有效实体')
    return open_crown,template


def placed_library_crown(library,fdi,margin,frame):
    open_crown,template=canonical_library_crown(library,fdi)
    # Initialize in the confirmed local frame, then put the result into world space.
    # A0：显式传**规范局部帧**（+Z 就位、X=近中），不再走 "world-X 推断" 的 legacy 分支。
    from ..geometry.crown_outer.patient_frame import confirmed_patient_frame
    local_frame=confirmed_patient_frame(np.zeros(3),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0))
    local_margin=(np.asarray(margin)-frame.origin)@frame.matrix()
    placed=initialize_from_margin(open_crown,local_margin,[0,0,1],template_cervical_points=template.margin_loop,
                                  patient_frame=local_frame)
    placed.vertices=np.asarray(placed.vertices)@frame.matrix().T+frame.origin
    return placed,{'library':library.name,'fdi':fdi,'margin_source':template.margin_source,
        'template_summary':template.summary(),'label_semantics_status':'GEOMETRIC_CANDIDATE_NOT_CLINICALLY_VERIFIED',
        'normalization':'confirmed_mesial_and_axis_frame','original_anatomy_review_required':True}
