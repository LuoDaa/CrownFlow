"""Reviewed library proposals, not automatic global template approval."""
from pathlib import Path

import numpy as np
import trimesh
import mapbox_earcut
from ..geometry.crown_outer.template import load_crown_template
from ..geometry.crown_outer.collar_body_outer import _boundary_loops
from ..geometry.crown_outer.generator import generate_crown_outer_v0
from ..geometry.nonplanar_crown import MarginCurtain

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


def _cap_margin_boundary(open_crown):
    """Step-5 临时封底：只封**已经 hard-fit 到患者 Margin**的唯一边界。

    最终制造几何仍由 finalize_crown 的 blockout/gap/intaglio 重建；这里的 cap 只为了
    浏览器工作台需要 watertight body，绝不当作真实内冠。
    """
    crown = open_crown.copy()
    crown.remove_unreferenced_vertices()
    loops = _boundary_loops(np.asarray(crown.faces, np.int64))
    if len(loops) != 1 or len(crown.split(only_watertight=False)) != 1:
        raise ValueError('Margin hard-fit 后冠体出现额外缺口或组件，拒绝静默封补')
    ring = np.asarray(loops[0], int)
    MarginCurtain(np.asarray(crown.vertices, float)[ring], [0, 0, 1])
    triangles = mapbox_earcut.triangulate_float64(
        np.asarray(crown.vertices, float)[ring, :2],
        np.array([len(ring)], np.uint32)).reshape(-1, 3)
    crown.faces = np.vstack((np.asarray(crown.faces, np.int64), ring[triangles]))
    trimesh.repair.fix_normals(crown)
    if not crown.is_watertight or crown.volume <= 0:
        raise ValueError('患者 Margin 临时封底未形成有效实体')
    return crown


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
    # canonical_library_crown 同时验证库件只有一个合法颈缘边界，并从缓存取 CrownTemplate。
    _canonical,template=canonical_library_crown(library,fdi)

    # 先进入确认的患者局部帧：+Z=就位方向、+X=近中、+Y=颊侧。
    # 不再让 Margin PCA 猜 M/D/B/L；尤其 26 这类近似椭圆颈缘容易出现 90°/180° 二义。
    from ..geometry.crown_outer.patient_frame import confirmed_patient_frame
    local_frame=confirmed_patient_frame(np.zeros(3),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),
                                        revision_id=getattr(frame,'revision_id',None))
    local_margin=(np.asarray(margin)-frame.origin)@frame.matrix()

    # Crown 1.0：从"只对齐颈缘中心/尺度"升级为真正的 3D Margin hard constraint。
    # generate_crown_outer_v0 内部做闭环 forward/reverse+shift correspondence，并用 Laplacian
    # 将模板颈环顶点逐点钉到患者 Margin；咬合区仅弱锚，因而保留牙库 cusp/groove 基础形态。
    result=generate_crown_outer_v0(template,local_margin,[0,0,1],patient_frame=local_frame)
    if result.mesh is None:
        raise ValueError('Margin hard-fit 未生成冠体')
    placed=_cap_margin_boundary(result.mesh)
    placed.vertices=np.asarray(placed.vertices)@frame.matrix().T+frame.origin

    solver=(result.mesh_qc or {}).get('solver') or {}
    return placed,{'library':library.name,'fdi':fdi,'margin_source':template.margin_source,
        'template_summary':template.summary(),'label_semantics_status':'GEOMETRIC_CANDIDATE_NOT_CLINICALLY_VERIFIED',
        'normalization':'confirmed_mesial_and_axis_frame','original_anatomy_review_required':True,
        'margin_fit_mode':'HARD_LAPLACIAN_BOUNDARY',
        'margin_error_median_mm':result.margin_error_median_mm,
        'margin_error_p95_mm':result.margin_error_p95_mm,
        'margin_error_max_mm':result.margin_error_max_mm,
        'margin_constraint_residual_max_mm':solver.get('margin_constraint_residual_max_mm'),
        'crown_outer_status':result.status}
