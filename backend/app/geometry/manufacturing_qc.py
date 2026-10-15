"""Measured engineering QC. No inferred material or clinical acceptance limits."""
import numpy as np
import trimesh
from trimesh.proximity import closest_point
from .margin_seam import _polyline_distances


def _closest(mesh,points):
    chunks=[closest_point(mesh,points[i:i+64]) for i in range(0,len(points),64)]
    return np.concatenate([c[0] for c in chunks]),np.concatenate([c[1] for c in chunks]),np.concatenate([c[2] for c in chunks])


def _stats(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return {'n':len(values),'min_mm':float(values.min()) if len(values) else None,
        'p05_mm':float(np.quantile(values,.05)) if len(values) else None,
        'median_mm':float(np.median(values)) if len(values) else None,
        'p95_mm':float(np.quantile(values,.95)) if len(values) else None,
        'max_mm':float(values.max()) if len(values) else None}


def _normal_ray_distances(mesh,points,directions):
    distances=np.full(len(points),np.nan)
    # Bound broad-phase ray candidates for long cervical triangles.
    for start in range(0,len(points),128):
        p=points[start:start+128];d=directions[start:start+128]
        hits,ids,_=mesh.ray.intersects_location(p+1e-7*d,d,multiple_hits=False)
        if len(ids):distances[start+ids]=np.linalg.norm(hits-p[ids],axis=1)
    return distances


def _intersection_volume(a,b):
    if not a.is_watertight or not b.is_watertight:
        return {'status':'NOT_ASSESSED','reason':'requires_closed_oriented_input'}
    common=trimesh.boolean.intersection([a,b],engine='manifold')
    volume=0.0 if common is None or not len(common.faces) else float(abs(common.volume))
    return {'status':'MEASURED','intersection_volume_mm3':volume,'acceptance_limit_mm3':None,
        'method':'manifold3d_solid_intersection'}


def measure_wall_thickness(outer,inner,margin,axis,parameters):
    # shortcut: vertices and centroids cannot certify the global minimum; add adaptive certification before manufacturing approval.
    vertices=np.asarray(inner.vertices);centers=np.asarray(inner.triangles_center)
    points=np.vstack((vertices,centers));n=len(vertices)
    # The final shell's inner normals point into the cavity; reverse into material.
    directions=-np.vstack((inner.vertex_normals,inner.face_normals))
    _,distance,_=_closest(outer,points)
    ray=_normal_ray_distances(outer,points,directions)
    d_margin=_polyline_distances(points,margin)
    valid=d_margin>1e-6
    conservative=np.minimum(distance,np.where(np.isfinite(ray),ray,np.inf))
    threshold=float(parameters['minimum_thickness_mm'])
    thin=valid&(conservative<threshold)
    area=np.asarray(inner.area_faces)
    from .nonplanar_crown import MarginCurtain
    height=MarginCurtain(margin,axis).axial_above(points)
    border_enabled=float(parameters.get('border_horizontal_mm',0))>0
    body=height>=float(parameters['margin_band_mm'])-1e-6 if border_enabled else valid
    thickness={'status':'REVIEW_REQUIRED','material_limit_validated':False,'configured_minimum_mm':threshold,
        'method':'all_vertices_and_face_centroids_exact_triangle_and_normal_ray_first_hit',
        'closest':_stats(distance[valid]),'normal_ray':_stats(ray[valid]),'conservative':_stats(conservative[valid]),
        'samples':len(points),'unresolved_normal_ray_count':int((valid&~np.isfinite(ray)).sum()),
        'seam_samples_excluded':int((~valid).sum()),'thin_face_sample_area_mm2':float(area[thin[n:]].sum()),
        'body_thin_face_sample_area_mm2':float(area[(thin&body)[n:]].sum()),
        'body_conservative':_stats(conservative[valid&body]),
        'cervical_conservative':_stats(conservative[valid&~body]),
        'cervical_acceptance':'NOT_ASSESSED_REQUIRES_BORDER_AND_MATERIAL_RULE' if border_enabled else 'KNIFE_EDGE_ENGINEERING_ONLY',
        'thin_vertex_ids':np.flatnonzero(thin[:n]).tolist(),
        'coverage_note':'Samples cover vertices and centroids, not an analytical global minimum proof; knife-edge samples reported separately.'}
    heatmap={'vertices':vertices.tolist(),'faces':np.asarray(inner.faces).tolist(),
        'distance_mm':conservative[:n].tolist(),'normal_ray_mm':[float(v) if np.isfinite(v) else None for v in ray[:n]],
        'thin_vertex_ids':thickness['thin_vertex_ids'],'min_mm':float(conservative[:n][valid[:n]].min()),
        'p05_mm':float(np.quantile(conservative[:n][valid[:n]],.05)),
        'method':'inner_surface_sampled_wall_thickness','note':'真实内面到外面厚度采样；薄区按已确认工程数值标记，材料限值未获验证；颈缘刀口单列。'}
    return thickness,heatmap


def audit_manufacturing(final,outer,inner,support,blockout,margin,axis,parameters,topology,*,virtual_support=None,virtual_cap_face_ids=()):
    thickness,heatmap=measure_wall_thickness(outer,inner,margin,axis,parameters)
    points=np.vstack((inner.vertices,inner.triangles_center))
    directions=-np.vstack((inner.vertex_normals,inner.face_normals));area=np.asarray(inner.area_faces)
    from .nonplanar_crown import MarginCurtain
    height=MarginCurtain(margin,axis).axial_above(points)
    prep_closest,gap,prep_ids=_closest(support,points)
    actual=np.ones(len(points),bool)
    if virtual_support is not None and len(virtual_cap_face_ids):
        _,_,truth_faces=_closest(virtual_support,points)
        actual=~np.isin(truth_faces,virtual_cap_face_ids)
    band=float(parameters.get('cement_gap_start_mm') or 0)
    top=directions@np.asarray(axis)/np.linalg.norm(axis)>.5
    cervical_region_height=max(band,float(parameters['margin_band_mm']))
    cervical=(height<=cervical_region_height)&~top
    axial=~cervical&~top
    fit={'status':'REVIEW_REQUIRED','gap_acceptance_tolerance_mm':None,'configured_cement_gap_mm':parameters['cement_gap_mm'],
        'configured_cement_gap_top_mm':parameters.get('cement_gap_top_mm'),'occlusal_geometric_samples':_stats(gap[top&actual]),
        'configured_gap_start_mm':parameters.get('cement_gap_start_mm'),'method':'exact_triangle_unsigned_clearance',
        'actual_gap':_stats(gap[actual]),'virtual_cap_excluded_sample_count':int((~actual).sum()),
        'support_truth':'PREPARED_SUPPORT_REAL_FACES_ONLY',
        'cervical':_stats(gap[cervical&actual]),'body':_stats(gap[~cervical&actual]),
        'regions':{'cervical_margin_band':_stats(gap[cervical&actual]),
            'axial_wall':_stats(gap[axial&actual]),'occlusal_top':_stats(gap[top&actual])},
        'region_method':'confirmed_axis_height_above_margin_and_intaglio_normal; disjoint sampled regions',
        'cervical_region_height_mm':cervical_region_height,
        'signed_local_normal_reference':_stats(np.einsum('ij,ij->i',points-prep_closest,support.face_normals[prep_ids])[actual]),
        'local_normal_sign_status':'REFERENCE_ONLY_NOT_GLOBAL_INSIDE_OUTSIDE',
        'prep_material_intersection':_intersection_volume(final,support)}
    z=np.asarray(axis,float);z/=np.linalg.norm(z)
    draft=(-np.asarray(inner.face_normals))@z
    insertion={'status':'REVIEW_REQUIRED','negative_draft_area_mm2':float(area[draft < -1e-7].sum()),
        'conservative_shadow_collision':_intersection_volume(final,blockout.blockout_mesh),
        'shadow_resolution_mm':blockout.diagnostics['resolution_mm'],
        'method':'actual_intaglio_draft_and_directional_prep_shadow_material_intersection',
        'continuous_insertion_certified':False,'acceptance_limit':None}
    topology_status='MEASURED'
    if topology.get('self_intersections_truncated'): topology_status='NOT_ASSESSED'
    qc={'status':'ENGINEERING_ONLY','production_status':'NOT_PRODUCTION_APPROVED',
        'NonplanarMarginBuild':{'status':'MEASURED'},'IntaglioFitAudit':fit,'FinalCrownThicknessAudit':thickness,
        'InsertionAudit':insertion,'TopologyQC':{'status':topology_status,'metrics':topology,
            'self_intersection_scope':'candidate_screen_not_exhaustive_certification'},
        'ContactQC':{'status':'NOT_ASSESSED','reason':'confirmed_instance_and_signed_scope_contract_not_connected'},
        'OcclusionQC':{'status':'NOT_ASSESSED','reason':'trusted_bite_transform_and_signed_audit_not_connected'},
        'MaterialProcessQC':{'status':'NOT_ASSESSED','reason':'material_label_is_not_validated_process_configuration'}}
    return qc,heatmap
