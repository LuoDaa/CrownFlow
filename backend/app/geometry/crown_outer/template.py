"""牙冠模板标准化（Crown Template Normalization）。

从专业牙库取一个 FDI 的完整解剖冠模板，把它变成**可以往患者牙位上摆的规范对象**：

1. 用 ``library.local_points`` 把顶点表达在**该 FDI 的牙位局部系**里
   （Z = AxisOcclusal 咬合向，X = AxisMesial 近中向，Y = Z×X）。
   左右侧的差异由牙库 XML 的 ``AxisMesial`` 承载，所以这一步出来的点
   **已经是这一侧的形状**，不是"左右通用"的镜像源。
2. 牙颈环从 ``.eoff`` 的 ``VERTEX_FEATURE_LABELS`` 取（``find_margin_loop``）。
   **取不到就报错**，不回退到"最细处"之类的几何猜测 —— 那正是
   ``ml/template_studio.py`` 对另一套无标注牙库只能做的事，而本格式有现成标注。
3. 原点移到**牙颈环质心**，于是模板局部系里：Z 朝咬合面、原点落在颈缘附近、
   牙颈环是原点周围的近平曲线 —— 这就是摆位与形变的参照。
4. 按牙颈环把网格切成**冠部 / 根部**：把环上顶点当缝去掉后取连通分量，
   含最高点的分量是冠部、含最低点的是根部。**这是拓扑切分不是平面切分**，
   所以不假设颈线是平的（真实颈缘轴向起伏 1~3.5mm）。

形变只需要冠部；但环上的顶点必须留在两侧的边界表达里（``margin_loop_indices``），
切掉根会丢掉"环与根侧的连接关系"，摆位时用不到但审计时需要。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ...importers.tooth_library import (
    ToothLibrary,
    bit_members,
    boundary_loop,
    find_margin_loop,
)

#: 牙位局部系的轴下标（Z = AxisOcclusal = 咬合向）。
AXIS_Z = 2

#: 牙颈边界的两种来源。同一接口，下游不必关心模板来自哪类网格：
#: * ``OPEN_BOUNDARY`` —— 未封底的冠形网格（超神2 的 teeth），底边界环即牙颈线；
#: * ``LABEL_LOOP``    —— 封底的带根牙形（炼金王 / 洋溢），牙颈环来自
#:   ``VERTEX_FEATURE_LABELS`` 的一维闭环，冠/根靠环做拓扑切分。
MARGIN_SOURCE_OPEN_BOUNDARY = "OPEN_BOUNDARY"
MARGIN_SOURCE_LABEL_LOOP = "LABEL_LOOP"


class TemplateError(ValueError):
    pass


@dataclass
class CrownTemplate:
    """一个已标准化的牙冠模板（坐标在模板局部系，原点 = 牙颈环质心）。"""

    fdi: int
    library: str
    vertices: np.ndarray             # (V, 3) float64
    faces: np.ndarray                # (F, 3) int64
    margin_loop_indices: np.ndarray  # (L,) int64
    crown_vertex_mask: np.ndarray    # (V,) bool，不含环上顶点
    root_vertex_mask: np.ndarray     # (V,) bool，不含环上顶点
    margin_loop_bit: tuple[int, ...]
    margin_perimeter_mm: float
    margin_loop_centroid_raw: np.ndarray   # 移动原点前的质心（牙位局部系）
    occlusal_direction: np.ndarray         # (3,) 单位向量，模板局部系
    margin_source: str                     # OPEN_BOUNDARY 或 LABEL_LOOP
    split_info: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 派生量
    @property
    def margin_loop(self) -> np.ndarray:
        return self.vertices[self.margin_loop_indices]

    @property
    def margin_centroid(self) -> np.ndarray:
        return self.margin_loop.mean(axis=0)

    @property
    def margin_extent_mm(self) -> float:
        """颈缘在自身平面内的最大两点距离 —— 患者颈缘尺寸的对照量。"""
        p = self.margin_loop - self.margin_centroid
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        return float(d.max())

    @property
    def crown_height_mm(self) -> float:
        """颈缘平面到冠部最高点的距离（沿 +Z）。"""
        z = self.vertices[self.crown_vertex_mask, AXIS_Z]
        return float(z.max()) if z.size else 0.0

    @property
    def crown_span_mm(self) -> tuple[float, float]:
        """冠部在 X/Y 方向的跨度。"""
        p = self.vertices[self.crown_vertex_mask]
        if not p.size:
            return (0.0, 0.0)
        return (float(p[:, 0].max() - p[:, 0].min()),
                float(p[:, 1].max() - p[:, 1].min()))

    def summary(self) -> dict:
        return {
            "fdi": self.fdi, "library": self.library,
            "n_vertices": int(self.vertices.shape[0]),
            "n_faces": int(self.faces.shape[0]),
            "n_margin_loop": int(len(self.margin_loop_indices)),
            "margin_perimeter_mm": round(self.margin_perimeter_mm, 3),
            "margin_extent_mm": round(self.margin_extent_mm, 3),
            "margin_loop_bit": [hex(b) for b in self.margin_loop_bit],
            "margin_source": self.margin_source,
            "n_crown_vertices": int(self.crown_vertex_mask.sum()),
            "n_root_vertices": int(self.root_vertex_mask.sum()),
            "crown_height_mm": round(self.crown_height_mm, 3),
            "crown_span_xy_mm": [round(v, 3) for v in self.crown_span_mm],
            "occlusal_direction": [round(float(v), 4)
                                   for v in self.occlusal_direction],
            **self.split_info,
        }


# ---------------------------------------------------------------- 拓扑切分

def component_labels(n_vertices: int, faces: np.ndarray,
                     exclude: np.ndarray) -> np.ndarray:
    """删掉 ``exclude`` 顶点后对剩余顶点连通分量标号；被删的记 -1。

    邻接由面建立，BFS。模板规模（万级顶点）用纯 Python 足够快。
    """
    adj: list[list[int]] = [[] for _ in range(n_vertices)]
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            adj[u].append(v)
            adj[v].append(u)

    label = np.full(n_vertices, -1, dtype=np.int64)
    comp = 0
    for s in range(n_vertices):
        if exclude[s] or label[s] != -1:
            continue
        label[s] = comp
        stack = [s]
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if not exclude[w] and label[w] == -1:
                    label[w] = comp
                    stack.append(w)
        comp += 1
    return label


def split_crown_and_root(vertices: np.ndarray, faces: np.ndarray,
                         loop_indices: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, dict]:
    """按牙颈环把网格拓扑切成冠部/根部。

    把环上顶点当"缝"去掉，剩余顶点按面连通性分量：
    **含最高点（+Z）的分量是冠部，含最低点的是根部**。
    除这两者之外的碎片分量如实计入 ``info["n_other_components"]``
    并归到根部（保守：它们不属于冠部，不该参与冠部形变），
    同时把数量报出来 —— 静默吞掉碎片会让后面的形变结果无从解释。

    返回 ``(crown_mask, root_mask, info)``；两个 mask 都**不含**环上顶点。
    """
    n = vertices.shape[0]
    seam = np.zeros(n, dtype=bool)
    seam[loop_indices] = True
    label = component_labels(n, faces, seam)

    rest = np.flatnonzero(~seam)
    if rest.size == 0:
        raise TemplateError("删掉牙颈环后没有剩余顶点 —— 环覆盖了整个网格？")
    z = vertices[:, AXIS_Z]
    top = label[rest[np.argmax(z[rest])]]
    bot = label[rest[np.argmin(z[rest])]]
    if top == bot:
        raise TemplateError(
            "牙颈环没有把网格切成两部分（冠与根仍是同一连通分量）—— "
            "该环不是拓扑腰，不能作为形变边界")

    crown = np.zeros(n, dtype=bool)
    root = np.zeros(n, dtype=bool)
    crown[rest[label[rest] == top]] = True
    others = rest[(label[rest] != top) & (label[rest] != bot)]
    root[rest[label[rest] == bot]] = True
    root[others] = True

    info = {
        "n_components_after_seam": int(label.max() + 1),
        "n_other_components": int(len(set(label[others].tolist()))),
        "n_other_vertices": int(others.size),
    }
    return crown, root, info


# ---------------------------------------------------------------- 装配

def _orientation_check(verts: np.ndarray, *, has_root: bool) -> dict:
    """核对 +Z 是咬合向，失败抛 TemplateError。判据随网格类型不同：

    * **带根牙形**（``has_root=True``）：半径收敛 —— 牙根沿轴收敛成尖
      （极值端很窄），咬合面是一张宽台（极值端半径仍有 3~5mm）。
      不能用"颈缘离哪端近"判断：实测 FDI46 的颈缘大致在轴向中部
      （根比冠长），那种判据会误报。
    * **冠形网格**（``has_root=False``）：没有根，磨牙冠最宽处本来就在
      颈部（r_lo > r_hi 是正常形态，不是方向错误）。开放边界的存在已经
      定义了"底"，这里只需再核对边界确实在轴向下半部。
    """
    z = verts[:, AXIS_Z]
    z_min, z_max = float(z.min()), float(z.max())
    if has_root:
        cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
        r = np.hypot(verts[:, 0] - cx, verts[:, 1] - cy)
        lo_band, hi_band = z <= np.percentile(z, 5), z >= np.percentile(z, 95)
        r_lo, r_hi = float(r[lo_band].max()), float(r[hi_band].max())
        if r_lo > r_hi:
            raise TemplateError(
                "牙位局部系的 +Z 指向根尖而不是咬合面（最低 5%% 半径 %.2f > "
                "最高 5%% 半径 %.2f）—— 与 Z=AxisOcclusal 的约定不符"
                % (r_lo, r_hi))
        return {"oz_radius_lo5": round(r_lo, 3), "oz_radius_hi5": round(r_hi, 3)}
    return {"z_range": [round(z_min, 3), round(z_max, 3)]}


def load_crown_template(lib: ToothLibrary, fdi: int) -> CrownTemplate:
    """取 FDI 模板并标准化，自动识别牙颈边界来源。

    优先开放底边界（冠形网格，最直接），否则退到标签环（带根牙形）。
    取不到任何一种、+Z 不是咬合向、环不是拓扑腰 —— 都**直接报错**。
    理由：每一种都会让后面"把环对到患者 Margin"这步拿到错位的边界条件，
    而错位不会自己暴露（冠会歪掉但流程照跑完）。
    """
    _entry, mesh, local_pts = lib.local_points(fdi, labels=True)
    verts = np.asarray(local_pts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # ---- 牙颈边界来源：**显式标签环优先**，开放底边界只作回退。
    # 理由：标签是设计者逐顶点标注的意图；而"网格上有洞"可能只是小的
    # 网格破洞（实测炼金王 FDI46 有一个 82 点、周长 9.7mm 的洞，
    # 在轴向下半部 —— 按位置检查根本拦不住）。反过来，冠形网格（超神2）
    # 的标签不成环，会自然落到回退分支。
    got = find_margin_loop(mesh)
    if got is not None:
        cand, bits = got
        loop_idx = np.asarray(_union_members(mesh, bits), dtype=np.int64)
        margin_source = MARGIN_SOURCE_LABEL_LOOP
        orient = _orientation_check(verts, has_root=True)
        crown, root, info = split_crown_and_root(verts, faces, loop_idx)
        margin_bit = tuple(sorted(bits))
        cand_perimeter = cand.perimeter_mm
    else:
        loop_bl = boundary_loop(mesh)
        if loop_bl is None:
            raise TemplateError(
                "%s 的 FDI%d 既没有开放底边界，也取不到牙颈线标签闭环"
                " —— 无法确定形变边界" % (lib.name, fdi))
        loop_idx = np.asarray(loop_bl, dtype=np.int64)
        # 边界必须在轴向下半部，且周长达到环形截面的下界（π×最小跨度）——
        # 否则它更可能是破洞而不是牙颈边界
        z_min, z_max = float(verts[:, AXIS_Z].min()), float(verts[:, AXIS_Z].max())
        z_loop = float(verts[loop_idx, AXIS_Z].mean())
        if z_loop > (z_min + z_max) / 2:
            raise TemplateError(
                "开放边界不在网格底部（边界均值 %.2f，范围 [%.2f,%.2f]）"
                " —— 不能当作牙颈边界" % (z_loop, z_min, z_max))
        xy = verts[loop_idx][:, :2]
        min_span = float((xy.max(axis=0) - xy.min(axis=0)).min())
        perim0 = float(sum(
            np.linalg.norm(verts[loop_idx[i]] - verts[loop_idx[(i + 1) % len(loop_idx)]])
            for i in range(len(loop_idx))))
        if perim0 < math.pi * min_span:
            raise TemplateError(
                "开放边界周长 %.2fmm 小于环形截面下界 %.2fmm（π×最小跨度 %.2f）"
                " —— 更像网格破洞而不是牙颈边界"
                % (perim0, math.pi * min_span, min_span))
        margin_source = MARGIN_SOURCE_OPEN_BOUNDARY
        orient = _orientation_check(verts, has_root=False)
        crown = np.ones(len(verts), dtype=bool)
        crown[loop_idx] = False
        root = np.zeros(len(verts), dtype=bool)
        info = {"n_components_after_seam": 1, "n_other_components": 0,
                "n_other_vertices": 0}
        margin_bit = ()
        cand_perimeter = None

    # ---- 环排序：对应阶段需要**沿环有序**的顶点序，下标序不行。
    # 标签环是"顶点集合"（下标序≠空间序），必须沿子集邻接走一圈；
    # 开放边界本来就是有序的。
    if margin_source == MARGIN_SOURCE_LABEL_LOOP:
        adj = mesh.vertex_adjacency()
        loop_idx = np.asarray(_order_ring(adj, loop_idx), dtype=np.int64)

    origin = verts[loop_idx].mean(axis=0)
    shifted = verts - origin
    d = shifted[crown].mean(axis=0) if crown.any() else np.array([0.0, 0.0, 1.0])
    d[AXIS_Z] = abs(d[AXIS_Z])
    nrm = np.linalg.norm(d)
    d = d / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])

    # 周长：标签环自带；开放边界自己沿环算
    if cand_perimeter is None:
        cand_perimeter = float(sum(
            np.linalg.norm(shifted[loop_idx[i]] - shifted[loop_idx[(i + 1)
                                                                 % len(loop_idx)]])
            for i in range(len(loop_idx))))
    z_span = float(verts[:, AXIS_Z].max() - verts[:, AXIS_Z].min()) or 1.0

    return CrownTemplate(
        fdi=fdi, library=lib.name, vertices=shifted, faces=faces,
        margin_loop_indices=loop_idx, crown_vertex_mask=crown,
        root_vertex_mask=root, margin_loop_bit=margin_bit,
        margin_perimeter_mm=round(cand_perimeter, 3),
        margin_loop_centroid_raw=origin, occlusal_direction=d,
        margin_source=margin_source,
        split_info={**info, **orient, "margin_loop_n": int(len(loop_idx)),
                    "margin_axial_fraction": round(
                        float((verts[loop_idx, AXIS_Z].mean()
                               - verts[:, AXIS_Z].min()) / z_span), 3)})


def _order_ring(adj: dict[int, set[int]], subset: np.ndarray) -> list[int]:
    """把环顶点子集排成沿环的序（子集内每顶点度数必须恰好 2）。"""
    s = set(int(x) for x in subset)
    start = min(s)
    order = [start]
    prev, cur = None, start
    while True:
        nxts = [w for w in adj[cur] if w in s and w != prev]
        if not nxts:
            raise TemplateError("牙颈环排序中断（子集不是单一闭环）")
        prev, cur = cur, nxts[0]
        if cur == start:
            break
        order.append(cur)
    if len(order) != len(s):
        raise TemplateError("牙颈环排序未覆盖全部顶点（存在多个环？）")
    return order


def _union_members(mesh, bits) -> list[int]:
    out: set[int] = set()
    for b in bits:
        out |= set(bit_members(mesh, b))
    return sorted(out)
