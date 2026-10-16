"""Reviewed library proposals, not automatic global template approval."""
from pathlib import Path

import numpy as np
import trimesh
import mapbox_earcut
from ..geometry.crown_outer.template import load_crown_template
from ..geometry.crown_outer.collar_body_outer import _boundary_loops
from ..geometry.nonplanar_crown import MarginCurtain
from ..geometry.registration import initialize_from_margin

# ── 库件缓存 ─────────────────────────────────────────────────────────────────
# 这两个函数是**确定性**的（同库件字节 + 同牙位 + 同 Margin/帧 ⇒ 同结果），但每次都要解析
# .eoff、封底、三角化、放置：实测 canonical 1.2s + placed 2.1s。Step-5 每笔微调都会重建语义
# 映射，而**这一步只跟库件与 Margin/帧有关、跟当前外形无关** ⇒ 不缓存就是每笔卡 4 秒。
# 这里按"库件字节指纹 + 牙位（+ Margin/帧）"缓存**数组**，每次返回全新 Trimesh，
# 调用方随便改都不会污染缓存。
_CACHE_LIMIT = 8
_CANONICAL_CACHE: dict = {}
_PLACED_CACHE: dict = {}


def _cache_put(cache, key, value):
    cache[key] = value
    while len(cache) > _CACHE_LIMIT:
        cache.pop(next(iter(cache)))


def _arrays(mesh):
    return np.array(mesh.vertices, float), np.array(mesh.faces, np.int64)


def _mesh_from(arrays):
    return trimesh.Trimesh(vertices=np.array(arrays[0], copy=True),
                           faces=np.array(arrays[1], copy=True), process=False)


def _library_cache_key(library, fdi):
    path = Path(library.template_path(fdi))
    stat = path.stat()
    return (library.name, int(fdi), path.name, stat.st_mtime_ns, stat.st_size)


def canonical_library_crown(library,fdi):
    key=_library_cache_key(library,fdi)
    hit=_CANONICAL_CACHE.get(key)
    if hit is not None:
        return _mesh_from(hit[0]),hit[1]
    open_crown,template=_canonical_library_crown_uncached(library,fdi)
    _cache_put(_CANONICAL_CACHE,key,(_arrays(open_crown),template))
    return open_crown,template


def _canonical_library_crown_uncached(library,fdi):
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
    margin_array=np.asarray(margin,float)
    key=(_library_cache_key(library,fdi),margin_array.tobytes(),
         np.asarray(frame.origin,float).tobytes(),np.asarray(frame.matrix(),float).tobytes())
    hit=_PLACED_CACHE.get(key)
    if hit is not None:
        return _mesh_from(hit[0]),dict(hit[1])
    placed,metadata=_placed_library_crown_uncached(library,fdi,margin_array,frame)
    _cache_put(_PLACED_CACHE,key,(_arrays(placed),dict(metadata)))
    return placed,metadata


def _placed_library_crown_uncached(library,fdi,margin,frame):
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
