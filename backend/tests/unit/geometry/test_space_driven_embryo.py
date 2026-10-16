import numpy as np
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import confirmed_patient_frame
from app.geometry.crown_outer.space_contact import build_proximal_domains
from app.geometry.crown_outer.space_morph import (
    audit_inner_envelope_clearance,
    space_driven_embryo_prefit,
)


def _frame():
    return confirmed_patient_frame([0,0,0],[1,0,0],[0,1,0],[0,0,1])


def _margin(rx=2.5, ry=2.0, z=0.0, n=96):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.c_[rx*np.cos(t),ry*np.sin(t),np.full(n,z)]


def _crown(scale=1.0):
    profile=np.array([[0,0],[1.95,0],[2.05,.35],[2.15,1.2],[2.3,2.8],[2.2,4.3],[1.65,5.8],[0,6]])
    mesh=trimesh.creation.revolve(profile,sections=96)
    mesh.apply_scale([scale,scale,scale])
    return mesh


def _neighbor(x, width_y=7.0):
    m=trimesh.creation.box(extents=[1.0,width_y,8.0]);m.apply_translation([x,0,3.0])
    return FitSurface(m)


def test_global_prefit_uses_patient_envelope_not_template_metric_size():
    frame=_frame();margin=_margin();surfaces={'a':_neighbor(-5.0),'b':_neighbor(5.0)}
    small,rs=space_driven_embryo_prefit(_crown(.70),frame,margin,proximal_surfaces=surfaces,proximal_target_mm=-.02)
    large,rl=space_driven_embryo_prefit(_crown(1.40),frame,margin,proximal_surfaces=surfaces,proximal_target_mm=-.02)

    assert rs['algorithm']=='PATIENT_6_FACE_ENVELOPE_PLUS_LOCAL_REFINEMENT_V5'
    assert rs['template_size_authoritative'] is False
    assert rs['patient_envelope']['global_dimensions_source']=='PATIENT_GEOMETRY_ONLY'
    sw=rs['patient_envelope']['md_width_mm'];lw=rl['patient_envelope']['md_width_mm']
    assert abs(sw-lw)<1e-9
    # Different embryos should end in essentially the same patient-owned body width.
    sb=np.asarray(small.vertices)[np.asarray(small.vertices)[:,2]>2]
    lb=np.asarray(large.vertices)[np.asarray(large.vertices)[:,2]>2]
    assert abs(np.ptp(sb[:,0])-np.ptp(lb[:,0]))<.35


def test_clinical_neighbor_label_does_not_control_physical_side_selection():
    mesh=_crown();frame=_frame();margin=_margin();surface=_neighbor(4.8)
    result,report=space_driven_embryo_prefit(mesh,frame,margin,
        proximal_surfaces={'distal':surface},proximal_target_mm=0.0)
    sides=report['patient_envelope']['sources']['proximal_sides']
    assert len(sides)==1 and sides[0]['physical_sign']>0
    masks,_locked,quality=build_proximal_domains(result,frame,margin,{'distal':surface})
    assert quality['distal']['direction_world'][0]>.70
    assert int(masks['distal'].sum())>=40


def test_support_fallback_expands_material_space_from_margin_root():
    mesh=_crown(.8);frame=_frame();margin=_margin()
    support=trimesh.creation.cylinder(radius=2.20,height=4.0,sections=96);support.apply_translation([0,0,2])
    result,report=space_driven_embryo_prefit(mesh,frame,margin,support_mesh=support,
        minimum_thickness_mm=.80,cement_gap_mm=.05,proximal_surfaces={})
    shell=report['inner_shell']
    assert shell['status'] in ('APPLIED','NO_CORRECTION_NEEDED')
    assert shell['source']=='PREP_SUPPORT_PLUS_CEMENT_GAP_FALLBACK'
    assert abs(shell['required_outer_to_intaglio_mm']-.85)<1e-9
    assert report['patient_envelope']['status']=='APPLIED'
    assert np.isfinite(np.asarray(result.vertices)).all()


def test_actual_intaglio_preview_does_not_double_count_cement_gap():
    mesh=_crown();frame=_frame();margin=_margin()
    intaglio=trimesh.creation.cylinder(radius=2.25,height=4.2,sections=96);intaglio.apply_translation([0,0,2.1])
    result,report=space_driven_embryo_prefit(mesh,frame,margin,inner_envelope_mesh=intaglio,
        minimum_thickness_mm=.80,cement_gap_mm=.05,proximal_surfaces={})
    shell=report['inner_shell']
    assert shell['source']=='ACTUAL_BLOCKOUT_GAP_INTAGLIO_PREVIEW'
    assert shell['actual_intaglio_preview_used'] is True
    assert abs(shell['required_outer_to_intaglio_mm']-.80)<1e-9
    audit=audit_inner_envelope_clearance(result,intaglio,frame,margin,.80)
    assert audit['sample_count']>50
    assert np.isfinite(audit['min_clearance_mm'])


def test_tall_actual_intaglio_sets_vertical_floor_without_template_authority():
    mesh=_crown();frame=_frame();margin=_margin()
    intaglio=trimesh.creation.cylinder(radius=1.8,height=5.7,sections=96);intaglio.apply_translation([0,0,2.85])
    before=float(np.quantile(np.asarray(mesh.vertices)[:,2],.98))
    result,report=space_driven_embryo_prefit(mesh,frame,margin,inner_envelope_mesh=intaglio,
        minimum_thickness_mm=.80,proximal_surfaces={})
    floor=report['height_floor']
    assert floor['applied'] is True
    assert floor['required_top_mm']>before
    assert np.quantile(np.asarray(result.vertices)[:,2],.98)>before


def test_antagonist_height_pass_preserves_xy_and_closes_gap():
    mesh=_crown();frame=_frame();margin=_margin()
    antagonist=trimesh.creation.box(extents=[12,12,1]);antagonist.apply_translation([0,0,8.5])
    xy0=np.asarray(mesh.vertices)[:,:2].copy()
    result,report=space_driven_embryo_prefit(mesh,frame,margin,
        antagonist_surface=FitSurface(antagonist),occlusal_target_mm=.05)
    assert report['patient_envelope']['reason']=='HEIGHT_ONLY_PASS_PRESERVES_XY'
    assert np.allclose(np.asarray(result.vertices)[:,:2],xy0,atol=1e-10)
    assert report['height']['applied'] is True
    accepted=[p for p in report['height']['passes'] if p.get('accepted')]
    assert accepted
    assert accepted[-1]['after']['gap_quantile_mm']<accepted[0]['before']['gap_quantile_mm']


def test_proximal_domain_cannot_collapse_to_a_handful_of_vertices():
    mesh=_crown();frame=_frame();margin=_margin();surface=_neighbor(3.7)
    masks,_locked,quality=build_proximal_domains(mesh,frame,margin,{'mesial':surface})
    assert quality['mesial']['status']=='READY'
    assert int(masks['mesial'].sum())>=40
    assert quality['mesial']['valid_vertex_count']>=24
