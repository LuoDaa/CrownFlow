"""Read-only source/crop provenance; never remove islands or bridge scan gaps."""
import numpy as np
import trimesh
from .nonplanar_crown import MarginCurtain
from .crown_outer.collar_body_outer import _clip_by_scalar,_boundary_loops
from .manufacturing_qc import _closest


def diagnose(support,margin,axis):
    vertices=np.asarray(support.vertices);faces=np.asarray(support.faces)
    labels=trimesh.graph.connected_component_labels(support.face_adjacency,node_count=len(faces))
    loops=_boundary_loops(faces)
    cp,distance,face_ids=_closest(support,np.asarray(margin))
    report={'original_face_component_ids':labels.tolist(),
        'original_components_faces':np.bincount(labels).tolist(),
        'original_boundary_loops_world':[vertices[loop].tolist() for loop in loops],
        'margin_closest_source_face_ids':face_ids.tolist(),
        'margin_closest_component_ids':labels[face_ids].tolist(),
        'margin_scan_distance_mm':distance.tolist(),
        'margin_scan_distance_max_mm':float(distance.max()),'codes':[]}
    if len(report['original_components_faces'])!=1: report['codes'].append('PREP_MULTICOMPONENT')
    try:
        curtain=MarginCurtain(margin,axis)
        v,f,_,source_ids=_clip_by_scalar(vertices,faces,-curtain.axial_above(vertices),return_face_sources=True)
        keep=curtain.axial_above(v)[f].max(axis=1)>1e-9 if len(f) else np.zeros(0,bool)
        f=f[keep];source_ids=source_ids[keep]
        cropped=trimesh.Trimesh(v,f,process=False);cropped.remove_unreferenced_vertices()
        components=trimesh.graph.connected_component_labels(cropped.face_adjacency,node_count=len(f)) if len(f) else np.zeros(0,int)
        loops=_boundary_loops(cropped.faces)
        report.update(cropped_face_source_ids=source_ids.tolist(),cropped_face_original_component_ids=labels[source_ids].tolist(),
            cropped_face_component_ids=components.tolist(),cropped_components_faces=np.bincount(components).tolist(),
            cropped_boundary_loops_world=[cropped.vertices[loop].tolist() for loop in loops])
        if len(loops)!=1: report['codes'].append('MULTIPLE_BORDER_LOOPS')
        if len(report['cropped_components_faces'])!=1: report['codes'].append('PREP_MARGIN_TOPOLOGY_UNSUPPORTED')
        report['cropped_island_source_face_ids']=source_ids[components!=np.argmax(np.bincount(components))].tolist() if len(components) else []
        return report,cropped
    except ValueError as exc:
        report.update(reason=str(exc));report['codes'].append('PREP_MARGIN_TOPOLOGY_UNSUPPORTED');return report,None
