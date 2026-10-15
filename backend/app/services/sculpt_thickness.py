"""Preview clearance and formal wall audit stay separate from final crown QC."""
import json
from pathlib import Path
from hashlib import sha256
import numpy as np
from .design_dependencies import file_digest,require_current_final
from ..geometry.io import load_mesh
from ..geometry.manufacturing_qc import _closest,measure_wall_thickness


def basis(design):
    return sha256(json.dumps({key:design.get(key) for key in ('confirmed_margin_revision_id','current_axis_revision_id','current_profile_revision_id')},sort_keys=True).encode()).hexdigest()


def attach_reference(design,state):
    if design.get('final_path'):
        from fastapi import HTTPException
        try:require_current_final(design)
        except HTTPException as exc:
            if exc.status_code!=409:raise
            state={**state};state.pop('thickness_reference',None)
            return state
        qc=json.loads(design['qc_json']);path=qc.get('artifact_paths',{}).get('intaglio_surface')
        if path:state={**state,'thickness_reference':{'path':path,'sha256':file_digest(path),'basis':basis(design)}}
    return state


def inner_reference(design,state):
    reference=state.get('thickness_reference')
    if not reference or reference['basis']!=basis(design):return None
    path=Path(reference['path'])
    if not path.is_file() or file_digest(path)!=reference['sha256']:raise ValueError('厚度参考内面已变化，请重建内冠')
    return load_mesh(path)


def preview_thickness(design,mesh,state,sample_ids):
    inner=inner_reference(design,state)
    if inner is None:return {'status':'NOT_AVAILABLE','reason':'REBUILD_REQUIRED'}
    values=_closest(inner,mesh.vertices[sample_ids])[1]
    return {'status':'PREVIEW_ONLY','sample_ids':sample_ids.tolist(),'distance_mm':values.tolist(),
        'method':'UNSIGNED_OUTER_TO_LAST_CONFIRMED_INTAGLIO','note':'异步采样距离参考，不是壁厚或制造放行依据'}


def formal_thickness_gate(design,before,after,state):
    inner=inner_reference(design,state)
    if inner is None:return {'status':'NOT_AVAILABLE','reason':'REBUILD_REQUIRED','final_qc_required':True}
    parameters=json.loads(design['parameters_json']);margin=json.loads(design['margin_json']);axis=json.loads(design['axis_json'])
    old,old_heatmap=measure_wall_thickness(before,inner,margin,axis,parameters);new,heatmap=measure_wall_thickness(after,inner,margin,axis,parameters)
    # The confirmed intaglio stays unchanged; reject new body thin samples, never excuse an existing thin region.
    if new['body_thin_face_sample_area_mm2']>old['body_thin_face_sample_area_mm2']+1e-6:
        raise ValueError('正式厚度门检测到新增薄区，已回滚；请减小减材或先修正冠形')
    from ..geometry.nonplanar_crown import MarginCurtain
    height=MarginCurtain(margin,axis).axial_above(inner.vertices)
    body=height>=parameters['margin_band_mm'] if parameters.get('border_horizontal_mm') else height>.4
    old_values=np.asarray(old_heatmap['distance_mm']);new_values=np.asarray(heatmap['distance_mm'])
    if np.any(new_values[body]<np.minimum(old_values[body],parameters['minimum_thickness_mm'])-1e-5):
        raise ValueError('正式厚度门检测到薄区恶化，已回滚')
    return {'status':'FORMAL_REFERENCE_AUDIT','measurement':new,'reference_sha256':state['thickness_reference']['sha256'],
        'full_final_qc_required':True,'production_approved':False,'heatmap':heatmap}
