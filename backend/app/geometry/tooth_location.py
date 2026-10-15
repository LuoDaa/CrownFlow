"""目标牙定位与 Prep/Support 分割。

**解决什么问题**：上游（exocad / 3Shape）只告诉你目标牙位是 FDI46，
不告诉你 46 在扫描网格里的 XYZ。这个模块把"牙位编号"变成"网格上的区域"，
再把该区域切成一个能交给 Axis / Margin 的独立 support 网格。

## 双模式，以及自动模式当前的真实状态

    locate_target_tooth(jaw, 46, user_seed=点)   操作员点一下目标牙  -> 已实现并验收
    locate_target_tooth(jaw, 46)                 自动                -> 见下

**自动模式这一版不做**，返回 ``REVIEW_REQUIRED`` 并给出**实测的理由**，
而不是猜一个看起来像那么回事的坐标。原因不是偷懒，是实测不支持：

  1. 没有训练过的牙实例检测器时，自动定位的唯一纯几何路径是
     "把参考牙列刚性配准到扫描，再读目标牙的位置"。
  2. 实测 FDI36 病例的 ``Unsectioned_LowerJaw``（其实是绕 FDI36 的
     **sectioned cast**，只含约 5-6 颗牙，不是整颌）上，多起点 ICP
     前 5 名残差只差 10%（1.034 / 1.136 / 1.146 / 1.188 / 1.193 mm），
     目标函数**不够陡**。取第一名等于把随机结果说成"找到了"。
  3. 参考牙列 16 颗 vs 局部扫描 5-6 颗，本身就不该配准。

所以自动模式要求"整颌牙弓 + 判别度足够的配准"两个条件同时成立，
当前两者都无法在有据可依的前提下满足 —— 这正是需求里
"自动置信度不足时让用户只点一下 46"所描述的情形。

## 已知未解决

**support 补片不是"一条边界的圆盘"。** 实测 FDI36 整颌上，切口边界是
8 条环（理想 1 条），因此状态为 ``REVIEW_REQUIRED``。已做的缓解是
``fill_mask_pinholes``（把被包围的针孔顶点补上，把边界环从 10 降到 8、
把连通分量从 2 降到 1）；根因是闭合流形上按测地半径切割时，区域会绕过
窄颈形成额外的边界环。这一点**没有被解决**，只是被如实报出来。

## 核心机制：凹陷代价测地线

从种子出发做测地线扩张，但**经过凹脊（邻接沟、龈沟）的行程代价被放大**，
于是区域自然停在凹谷处，而不是像 ``roi.spherical_roi`` 那样按直线距离取面
（半径一大就切进邻牙，半径一小就丢掉龈缘）。

    edge_weight = |edge| × (1 + VALLEY_GAIN × deviation)
    另有硬屏障：valley_strength > VALLEY_BLOCK 的边直接断开

## 如实说明能做到什么

**能**：给出目标牙的确定位置；给出单连通、含目标牙、边界尽量落在凹谷上的
support 网格；对"切进邻牙的程度"给出**可测的数字**，不达标就 ``REVIEW_REQUIRED``。

**不能**：保证近远中邻接面的精确分离。两牙接触处是**平滑鞍部**而非尖锐凹脊，
纯局部几何无法可靠切开 —— 这是牙齿分割的公开难题（CrownSegger 一类方法需要
训练模型）。本模块**报告**越界程度，而不是假装解决了它。

## 坐标约定

**绝不依赖世界 XYZ。** 凹/凸判据只用法向与邻面形心的相对位置；
位置只取网格顶点。因此整套结果**刚性变换等变**，有测试用随机刚体变换验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .insertion import _json_safe

# ---------------------------------------------------------------- 解剖学常数

#: 各牙位**近远中冠宽**（mm）。牙体解剖学标准值（Wheeler 一类教科书），
#: 不是从患者数据拟合的、也不是随手取的默认值。它**只用来定 ROI 的量级**，
#: 不作为任何几何真值 —— 真正的位置永远来自网格本身。
CROWN_WIDTH_MM: dict[int, float] = {
    11: 8.5, 12: 6.5, 13: 7.6, 14: 7.1, 15: 6.6, 16: 10.4, 17: 10.0, 18: 8.5,
    21: 8.5, 22: 6.5, 23: 7.6, 24: 7.1, 25: 6.6, 26: 10.4, 27: 10.0, 28: 8.5,
    31: 5.3, 32: 5.7, 33: 6.8, 34: 7.1, 35: 7.1, 36: 11.0, 37: 10.5, 38: 10.0,
    41: 5.3, 42: 5.7, 43: 6.8, 44: 7.1, 45: 7.1, 46: 11.0, 47: 10.5, 48: 10.0,
}

#: 牙位 → 牙型。
TOOTH_TYPE: dict[int, str] = {
    fdi: ("incisor" if fdi % 10 <= 2 else "canine" if fdi % 10 == 3
          else "premolar" if fdi % 10 <= 5 else "molar")
    for fdi in CROWN_WIDTH_MM
}

#: 上颌牙位（FDI 首位 1/2）。
UPPER_FDI = frozenset(f for f in CROWN_WIDTH_MM if f // 10 in (1, 2))

#: 二面角"算平"的阈值（弧度）。低于它的偏离不放大代价。
FLAT_ANGLE_RAD = 0.35
#: valley_strength 超过此值的边直接断开（硬屏障）。
#: 0.45 ≈ 二面角偏离平面 81° —— 一个 90° 的直角台阶交界正好落在这里
#: （strength 0.5）。原先取 0.60（≈108°）太尖锐，连"圆柱预备体与牙龈
#: 的直角交界"都拦不住，区域会顺壁上爬再漫进牙龈；实测在合成台阶上
#: 会把区域撑到 5-8 倍冠面积。
VALLEY_BLOCK = 0.42
#: 凹陷代价放大倍数（软惩罚）。够大才能在区域扩张时"绕开"而不是"穿过"凹谷。
VALLEY_GAIN = 14.0
#: 定位区域半径 = 冠宽 × 该系数。
#: **必须足够大到能走到解剖边界**：从牙冠中心到颈缘的测地行程 = 半径 + 冠高，
#: 一颗磨牙约 5.5 + 8 = 13.5 mm。取 0.72 时半径只有 7.9 mm，
#: 区域**是被半径截断的**而不是被凹脊拦住的，切口自然落在平滑面上
#: （实测合成夹具：切口前沿凹度 0.000，而凹脊就在 1.3 mm 之外）。
REGION_SCALE = 1.10
#: support 在冠部之外的额外龈方余量（mm）。
SUPPORT_MARGIN_MM = 1.5
#: 区域最少顶点数，低于此判失败（种子落在噪声/边界上）。
MIN_REGION_VERTICES = 150
#: 边界平均凹度下限。低于它说明边界走在凸面上 —— 大概率切进了牙冠外形。
MIN_BOUNDARY_VALLEY_STRENGTH = 0.22
#: 目标区域被 support 完整保留的比例下限。
MIN_TARGET_RETENTION = 0.98
#: support 面积 / 期望冠面积 的上限。超过说明明显带进了非目标表面。
MAX_SUPPORT_AREA_RATIO = 3.0


class LocalizationStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


class SupportStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


# ---------------------------------------------------------------- 结果类型

@dataclass
class ToothLocalizationResult:
    """目标牙在网格上的位置与区域。"""

    target_fdi: int
    jaw: str | None
    method: str                                  # SEED / AUTO_UNAVAILABLE
    position: list[float]
    region_vertex_ids: list[int]
    region_area_mm2: float
    region_extent_mm: float
    expected_crown_width_mm: float
    #: 区域扩张半径。``extract_support`` 以它为基准再加龈方余量 ——
    #: 不能拿"冠宽/2 + 余量"重算，那样基准与区域对不上（实测踩过）。
    region_radius_mm: float = 0.0
    seed_vertex_id: int | None = None
    status: LocalizationStatus = LocalizationStatus.FAILED
    confidence: float = 0.0
    uncertainty_mm: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def region_points(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """区域顶点坐标（需要原网格）。"""
        return np.asarray(mesh.vertices, dtype=float)[
            np.asarray(self.region_vertex_ids, dtype=int)]

    def as_dict(self) -> dict:
        return _json_safe({
            "target_fdi": self.target_fdi,
            "jaw": self.jaw,
            "method": self.method,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "position": [round(float(x), 4) for x in self.position],
            "n_region_vertices": len(self.region_vertex_ids),
            "region_area_mm2": round(float(self.region_area_mm2), 4),
            "region_extent_mm": round(float(self.region_extent_mm), 4),
            "region_radius_mm": round(float(self.region_radius_mm), 4),
            "expected_crown_width_mm": self.expected_crown_width_mm,
            "seed_vertex_id": self.seed_vertex_id,
            "uncertainty_mm": (None if self.uncertainty_mm is None
                               else round(float(self.uncertainty_mm), 4)),
            "diagnostics": dict(self.diagnostics),
        })


@dataclass
class SupportSegmentationResult:
    """切出来的 support 网格 + 验收数字。"""

    target_fdi: int
    mesh: trimesh.Trimesh | None
    status: SupportStatus = SupportStatus.FAILED
    confidence: float = 0.0
    cut_radius_mm: float = 0.0
    checks: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        from .io import mesh_info

        info = None
        if self.mesh is not None:
            try:
                info = mesh_info(self.mesh)
            except Exception as exc:      # pragma: no cover - 防御
                info = {"error": f"{type(exc).__name__}: {exc}"}
        return _json_safe({
            "target_fdi": self.target_fdi,
            "status": self.status.value,
            "confidence": round(float(self.confidence), 4),
            "cut_radius_mm": round(float(self.cut_radius_mm), 4),
            "mesh": info,
            "checks": dict(self.checks),
            "diagnostics": dict(self.diagnostics),
        })


# ---------------------------------------------------------------- 凹脊场

def face_pair_convex(mesh: trimesh.Trimesh) -> np.ndarray:
    """每个邻接面对是否**凸**（纯几何判据，与刚体变换等变）。

    取面 1 的法向 ``n1`` 与"面 2 形心 − 面 1 形心" ``d``：
    ``dot(n1, d) < 0`` 为凸（邻面落在切平面下方），``> 0`` 为凹（邻面翻到切平面外侧）。
    """
    pairs = np.asarray(mesh.face_adjacency)
    c1 = np.asarray(mesh.triangles_center)[pairs[:, 0]]
    c2 = np.asarray(mesh.triangles_center)[pairs[:, 1]]
    n1 = np.asarray(mesh.face_normals)[pairs[:, 0]]
    return np.einsum("ij,ij->i", n1, c2 - c1) < 0.0


def vertex_valley_strength(mesh: trimesh.Trimesh) -> np.ndarray:
    """每顶点的**凹谷强度** ∈ [0, 1]。

    取该顶点所有**凹**邻接边里，二面角偏离最大的那条，
    ``strength = clip(angle / pi, 0, 1)``。凸边一律记 0 —— 凸脊是牙冠外形
    （牙尖、边缘嵴），不是分割边界。

    注意 ``mesh.face_adjacency`` 装的是**面**索引对，共享边的顶点在
    ``mesh.face_adjacency_edges`` —— 把前者当顶点索引用会直接越界。
    """
    convex = face_pair_convex(mesh)
    ang = np.asarray(mesh.face_adjacency_angles, dtype=float)
    concave_ang = np.where(convex, 0.0, np.clip(ang, 0.0, np.pi))
    shared_edges = np.asarray(mesh.face_adjacency_edges)

    strength = np.zeros(len(mesh.vertices), dtype=float)
    for col in (0, 1):
        np.maximum.at(strength, shared_edges[:, col], concave_ang)
    return np.clip(strength / np.pi, 0.0, 1.0)


def _edge_valley_strength(mesh: trimesh.Trimesh,
                          v_strength: np.ndarray) -> np.ndarray:
    e = np.asarray(mesh.edges_unique)
    return np.maximum(v_strength[e[:, 0]], v_strength[e[:, 1]])


# ---------------------------------------------------------------- 测地线

def geodesic_distances(mesh: trimesh.Trimesh, sources, *,
                       valley_strength: np.ndarray | None = None,
                       valley_gain: float = VALLEY_GAIN,
                       valley_block: float = VALLEY_BLOCK,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """带凹陷代价的测地距离（从多源点出发）。

    ``edge_weight = |edge| × (1 + gain × deviation)``，其中
    ``deviation = max(valley_strength − FLAT_ANGLE_RAD/pi, 0)``；
    凹谷强度超过 ``valley_block`` 的边权重置 ``inf``（硬屏障）。

    返回 ``(距离, 被屏障断开的边掩码)``。距离单位是**代价 mm**：
    平坦处等于几何长度，穿凹谷时被放大，因此"绕开凹谷"比"穿过"更短。
    """
    e = np.asarray(mesh.edges_unique)
    length = np.asarray(mesh.edges_unique_length, dtype=float)
    n = len(mesh.vertices)

    factor = np.ones(len(e), dtype=float)
    blocked = np.zeros(len(e), dtype=bool)
    if valley_strength is not None:
        es = _edge_valley_strength(mesh, valley_strength)
        deviation = np.clip(es - FLAT_ANGLE_RAD / np.pi, 0.0, None)
        factor = 1.0 + valley_gain * deviation
        blocked = es > valley_block

    w = np.where(blocked, np.inf, length * factor)
    A = coo_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]]))),
        shape=(n, n)).tocsr()

    idx = np.atleast_1d(np.asarray(sources, dtype=int))
    if not len(idx):
        raise ValueError("测地线至少需要一个源点")
    d = dijkstra(A, indices=idx, min_only=True)
    return np.asarray(d, dtype=float), blocked


def fill_mask_pinholes(mesh: trimesh.Trimesh, mask: np.ndarray,
                       iterations: int = 6) -> np.ndarray:
    """把顶点掩码里的**针孔**填掉。

    在闭合流形上按测地半径切一块，理想的补片只有**一条**边界环。
    实测（FDI36 整颌）切出来的边界环是
    ``[399, 98, 38, 25, 20, 13, 12, 10, 8, 8, 3]`` ——
    一条真的加十条针孔，而针孔会让 ``n_boundary_loops > 1`` 报出"切成了多个补片"，
    掩盖真正的问题。

    做法很保守：只**新增**"除至多一个邻居外全在掩码内"的顶点（即被包围的空洞），
    不删除任何顶点，所以不会把区域切碎、也不会越过硬屏障。
    """
    e = np.asarray(mesh.edges_unique)
    deg = np.bincount(np.concatenate([e[:, 0], e[:, 1]]),
                      minlength=len(mask)).astype(float)
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        inside_count = np.zeros(len(out), dtype=float)
        np.add.at(inside_count, e[:, 0], out[e[:, 1]].astype(float))
        np.add.at(inside_count, e[:, 1], out[e[:, 0]].astype(float))
        add = (~out) & (deg > 0) & (inside_count >= deg - 1.0)
        if not add.any():
            break
        out |= add
    return out


def _submesh_from_vertices(mesh: trimesh.Trimesh,
                           vertex_mask: np.ndarray) -> trimesh.Trimesh | None:
    """顶点集诱导的子网格（保留三顶点全在集合内的面）。"""
    face_mask = vertex_mask[np.asarray(mesh.faces)].all(axis=1)
    if not face_mask.any():
        return None
    sub = mesh.submesh([np.nonzero(face_mask)[0]], append=True, repair=False)
    sub.remove_unreferenced_vertices()
    return sub


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """网格的边界边（只被**一个**面使用的边）。

    自己算而不是用 ``mesh.outline()``：``outline()`` 在边界较碎时返回的是
    一堆互不相连的短链，``len(entities)`` 当"边界环数"用会报出 3/9/11 这种
    假数字（实测踩过）。这里直接数唯一无向边的使用次数，稳。
    """
    faces = np.asarray(mesh.faces)
    de = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(de, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    new = np.ones(len(ks), dtype=bool)
    new[1:] = (ks[1:] != ks[:-1]).any(axis=1)
    starts = np.nonzero(new)[0]
    counts = np.diff(np.append(starts, len(ks)))
    return ks[starts][counts == 1]


def boundary_loop_count(mesh: trimesh.Trimesh) -> int:
    """边界环个数：把边界边按连通性分组建图，数连通分量。"""
    be = boundary_edges(mesh)
    if not len(be):
        return 0
    n = len(mesh.vertices)
    w = np.ones(len(be))
    A = coo_matrix((np.concatenate([w, w]),
                    (np.concatenate([be[:, 0], be[:, 1]]),
                     np.concatenate([be[:, 1], be[:, 0]]))), shape=(n, n)).tocsr()
    from scipy.sparse.csgraph import connected_components
    _, labels = connected_components(A, directed=False)
    return int(len(set(labels[np.unique(be)])))


def _boundary_vertices(mesh: trimesh.Trimesh, vertex_mask: np.ndarray) -> np.ndarray:
    """边界顶点：在集合内、且存在集合外的邻接顶点。"""
    e = np.asarray(mesh.edges_unique)
    crossing = vertex_mask[e[:, 0]] ^ vertex_mask[e[:, 1]]
    out = np.zeros(len(mesh.vertices), dtype=bool)
    if crossing.any():
        ce = e[crossing]
        out[ce[:, 0]] = True
        out[ce[:, 1]] = True
    return np.nonzero(out)[0]


def cut_front_strength(mesh: trimesh.Trimesh, inside_ids,
                       valley_strength: np.ndarray) -> tuple[float, int]:
    """**切口前沿**的平均凹度 —— 这才是"边界是否依附解剖凹谷"的正确信号。

    为什么不能只看区域自己的边界顶点：硬屏障把凹脊上的**边**断开了，
    于是区域边界落在屏障**内侧**的平滑面上，边界顶点凹度恒为 0。
    实测（合成 90 度台阶夹具）：区域边界凹度 0.000，而它明明就停在凹脊上。

    正确做法是看"区域外一圈紧邻顶点"：那一圈是凹脊，就说明切口沿着解剖凹谷走 ——
    无论屏障是把凹脊本身排除在外，还是停在它旁边。

    返回 ``(平均凹度, 前沿顶点数)``。
    """
    e = np.asarray(mesh.edges_unique)
    inside = np.zeros(len(mesh.vertices), dtype=bool)
    inside[np.asarray(inside_ids, dtype=int)] = True
    crossing = inside[e[:, 0]] ^ inside[e[:, 1]]
    if not crossing.any():
        return 0.0, 0
    ce = e[crossing]
    front = np.unique(np.where(inside[ce[:, 0]], ce[:, 1], ce[:, 0]))
    if not front.size:
        return 0.0, 0
    # 再往外带一圈：屏障可能把凹脊本身排除在区域外，那"前沿"就已经是凹脊；
    # 也可能停在凹脊内侧，那前沿是平滑面、凹脊在下一圈。两圈并起来取统计。
    inside2 = inside.copy()
    inside2[front] = True
    crossing2 = inside2[e[:, 0]] ^ inside2[e[:, 1]]
    extra = np.array([], dtype=int)
    if crossing2.any():
        ce2 = e[crossing2]
        extra = np.unique(np.where(inside2[ce2[:, 0]], ce2[:, 1], ce2[:, 0]))
    band = np.union1d(front, extra)
    return float(valley_strength[band].mean()), int(front.size)


# ---------------------------------------------------------------- 尺寸

def expected_surface_area_mm2(target_fdi: int) -> float:
    """该牙位**表面面积**的量级参考 = 半径 冠宽/2 的球冠 (2πr²)。

    不要用咬合面足迹 (πr²) 当参考 —— 那漏掉了牙冠侧面，而 support 恰恰
    包含侧面。实测：FDI36 预备体（r 5.5、高 4 的台 + 壁）真实表面 233 mm²，
    用足迹算是 76 mm²，比值 3.06 会被误判成"带进了邻牙"。
    """
    r = expected_crown_width_mm(target_fdi) / 2.0
    return float(2.0 * np.pi * r * r)


def expected_crown_width_mm(target_fdi: int) -> float:
    if target_fdi not in CROWN_WIDTH_MM:
        raise ValueError(
            f"FDI {target_fdi} 不在已知牙位表里，无法给出冠宽量级。"
            f"已知牙位：{sorted(CROWN_WIDTH_MM)}")
    return CROWN_WIDTH_MM[target_fdi]


def jaw_of_fdi(target_fdi: int) -> str:
    return "upper" if target_fdi in UPPER_FDI else "lower"


# ---------------------------------------------------------------- 定位

def locate_target_tooth(jaw_mesh: trimesh.Trimesh, target_fdi: int, *,
                        user_seed=None,
                        region_scale: float = REGION_SCALE) -> ToothLocalizationResult:
    """把牙位编号定位成网格上的区域。

    ``user_seed``：操作员在 3D 上点的一点。**只是种子**，不要求点准 ——
    会被吸附到最近顶点，然后由凹陷感知的区域扩张自动完成 ROI。

    未给 ``user_seed`` 时走自动模式；**当前版本自动模式一律返回
    ``REVIEW_REQUIRED`` 并说明实测原因**（见模块文档），不猜坐标。
    """
    if not isinstance(jaw_mesh, trimesh.Trimesh):
        raise ValueError("jaw_mesh 必须是 trimesh.Trimesh")
    if len(jaw_mesh.faces) == 0:
        raise ValueError("jaw_mesh 没有三角面")
    width = expected_crown_width_mm(target_fdi)     # 牙位不认识就早失败

    if user_seed is None:
        return ToothLocalizationResult(
            target_fdi=target_fdi, jaw=jaw_of_fdi(target_fdi),
            method="AUTO_UNAVAILABLE", position=[0.0, 0.0, 0.0],
            region_vertex_ids=[], region_area_mm2=0.0, region_extent_mm=0.0,
            expected_crown_width_mm=width,
            status=LocalizationStatus.REVIEW_REQUIRED, confidence=0.0,
            diagnostics={
                "reasons": [
                    "自动牙位定位未实现：无训练检测器时唯一纯几何路径是"
                    "『参考牙列刚性配准 → 读目标牙位置』，而实测（FDI36 的 "
                    "Unsectioned_LowerJaw 实为绕该牙位的 sectioned cast）多起点 ICP "
                    "前 5 名残差只差 10%（1.034/1.136/1.146/1.188/1.193 mm），"
                    "判别度不足，取第一名等于赌。",
                ],
                "required_action": "传入 user_seed（在 3D 上点一下目标牙）走种子模式。",
            })
    return _locate_from_seed(jaw_mesh, target_fdi, user_seed,
                             region_scale=region_scale)


def resolve_seed_sources(mesh: trimesh.Trimesh, seed, valley_strength: np.ndarray,
                         *, block: float = VALLEY_BLOCK,
                         max_sources: int = 400) -> np.ndarray:
    """把操作员点的那一下解析成一组**合适的源顶点**。

    直接把种子吸附到"最近的那个顶点"是不够的：**操作员完全可能正好点在凹脊上**
    （实测 FDI36 病例里，点在基台/牙龈交界处时最近顶点凹谷强度 0.90，
    它的每条边都被硬屏障断开，区域只剩它自己一个顶点）。

    所以这里取"种子附近、且**不在硬屏障凹脊上**"的一小片顶点当源：
    半径从 0.5 mm 起逐级放大，直到找到这样的顶点。这样种子只是"大致位置"，
    后续扩张仍然由凹陷感知的测地线决定。
    """
    verts = np.asarray(mesh.vertices, dtype=float)
    seed = np.asarray(seed, dtype=float).reshape(-1)
    d = np.linalg.norm(verts - seed, axis=1)

    d_min = float(d.min())
    # 搜索半径**以最近顶点距离为基准**逐级放大 —— 否则种子落在体内时
    # （例如程序拿质心当种子）会一路放大，源点自己就铺开 2.7 mm，区域凭空胖一圈。
    for extra in (0.25, 0.6, 1.2, 2.0):
        cand = np.nonzero((d <= d_min + extra) & (valley_strength < block))[0]
        if len(cand):
            if len(cand) > max_sources:
                cand = cand[np.argsort(d[cand])[:max_sources]]
            return cand

    # 附近全是凹脊：退回到最不凹的那些顶点（至少不把自己孤立）
    order = np.argsort(d)
    for k in (50, 200, 1000):
        pool = order[:min(k, len(order))]
        sub = pool[valley_strength[pool] < block]
        if len(sub):
            return sub
    return np.array([int(np.argmin(d))], dtype=int)


def _locate_from_seed(jaw_mesh: trimesh.Trimesh, target_fdi: int, user_seed,
                      *, region_scale: float) -> ToothLocalizationResult:
    verts = np.asarray(jaw_mesh.vertices, dtype=float)
    seed = np.asarray(user_seed, dtype=float).reshape(-1)
    if seed.shape[0] != 3:
        raise ValueError("user_seed 必须是三个坐标值")
    nearest_id = int(np.argmin(np.linalg.norm(verts - seed, axis=1)))

    width = expected_crown_width_mm(target_fdi)
    radius = region_scale * width

    v_strength = vertex_valley_strength(jaw_mesh)
    source_ids = resolve_seed_sources(jaw_mesh, seed, v_strength)
    dist, blocked = geodesic_distances(jaw_mesh, source_ids,
                                       valley_strength=v_strength)
    finite = np.isfinite(dist)
    mask = finite & (dist <= radius)
    n_mask = int(mask.sum())

    diagnostics = {
        "seed_input": [round(float(x), 4) for x in seed],
        "nearest_vertex_id": nearest_id,
        "seed_snap_distance_mm": round(float(np.linalg.norm(verts[nearest_id] - seed)), 4),
        "n_source_vertices": int(len(source_ids)),
        "valley_strength_at_nearest_vertex": round(float(v_strength[nearest_id]), 4),
        "region_radius_mm": round(radius, 4),
        "n_blocked_edges": int(blocked.sum()),
        "geodesic_reach_cost_mm": round(float(dist[finite].max()) if finite.any() else 0.0, 4),
        "region_scale": region_scale,
    }

    if n_mask < MIN_REGION_VERTICES:
        diagnostics["reasons"] = [
            f"种子周围区域只有 {n_mask} 个顶点（下限 {MIN_REGION_VERTICES}）。"
            "可能是种子落在噪声/网格边界上，或凹陷屏障把它切碎了。"]
        return ToothLocalizationResult(
            target_fdi=target_fdi, jaw=jaw_of_fdi(target_fdi), method="SEED",
            position=[float(x) for x in verts[nearest_id]], region_vertex_ids=[],
            region_area_mm2=0.0, region_extent_mm=0.0,
            expected_crown_width_mm=width, region_radius_mm=radius,
            seed_vertex_id=nearest_id,
            status=LocalizationStatus.FAILED, confidence=0.0,
            diagnostics=diagnostics)

    pts = verts[mask]
    area = float(np.asarray(jaw_mesh.area_faces)[
        mask[np.asarray(jaw_mesh.faces)].all(axis=1)].sum())
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

    bnd = _boundary_vertices(jaw_mesh, mask)
    valley_frac, n_front = cut_front_strength(jaw_mesh, np.nonzero(mask)[0], v_strength)
    area_ratio = area / expected_surface_area_mm2(target_fdi)

    confidence, reasons, status = _assess_region(
        valley_frac, area_ratio, width, diagnostics)
    diagnostics.update({
        "cut_front_valley_strength_mean": round(valley_frac, 4),
        "n_cut_front_vertices": n_front,
        "n_boundary_vertices": int(len(bnd)),
        "area_ratio_vs_crown": round(float(area_ratio), 4),
        "reasons": reasons,
    })
    return ToothLocalizationResult(
        target_fdi=target_fdi, jaw=jaw_of_fdi(target_fdi), method="SEED",
        position=[float(x) for x in pts.mean(axis=0)],
        region_vertex_ids=np.nonzero(mask)[0].tolist(),
        region_area_mm2=area, region_extent_mm=extent,
        expected_crown_width_mm=width, region_radius_mm=radius,
        seed_vertex_id=nearest_id,
        status=status, confidence=confidence,
        uncertainty_mm=extent / 2.0, diagnostics=diagnostics)


def _assess_region(valley_frac: float, area_ratio: float, width: float,
                   diagnostics: dict):
    """区域质量 -> (confidence, reasons, status)。**不达标就明说。**"""
    reasons: list[str] = []
    if area_ratio < 0.40:
        reasons.append(f"区域面积只有期望冠面积的 {area_ratio:.2f} 倍，明显偏小："
                       "种子可能落在牙冠边缘，或凹陷屏障把区域切碎了。")
        size_term = 0.0
    elif area_ratio > 4.0:
        reasons.append(f"区域面积达期望冠面积的 {area_ratio:.2f} 倍，明显偏大："
                       "很可能已越过凹谷扩到邻牙或牙龈。")
        size_term = 0.0
    else:
        size_term = float(np.clip(1.0 - abs(np.log(area_ratio)) / 0.9, 0.0, 1.0))

    if valley_frac < MIN_BOUNDARY_VALLEY_STRENGTH:
        reasons.append(
            f"切口前沿平均凹度只有 {valley_frac:.3f}（下限 "
            f"{MIN_BOUNDARY_VALLEY_STRENGTH}）：切口大部分走在凸面/平面上，"
            "说明它切进了牙冠外形，而不是停在邻接沟/龈沟里 —— "
            "该处很可能**没有可依附的解剖凹谷**。")

    valley_term = float(np.clip(valley_frac / (2.0 * MIN_BOUNDARY_VALLEY_STRENGTH),
                                0.0, 1.0))
    confidence = float(np.clip(0.5 * valley_term + 0.5 * size_term, 0.0, 1.0))
    status = LocalizationStatus.REVIEW_REQUIRED if reasons else LocalizationStatus.OK
    diagnostics["expected_crown_width_mm"] = width
    return confidence, reasons, status


# ---------------------------------------------------------------- Support 分割

def extract_support(jaw_mesh: trimesh.Trimesh, target_region, *,
                    target_fdi: int,
                    margin_mm: float = SUPPORT_MARGIN_MM,
                    ) -> SupportSegmentationResult:
    """把目标牙区域切成独立的 support 网格。

    ``target_region`` 可以是
      * :class:`ToothLocalizationResult`（用它的区域顶点）
      * ``(N, 3)`` 区域点坐标
      * ``(3,)`` 单点（当作种子）

    切割半径 = ``冠宽/2 + margin_mm``（冠部 + 一点龈方余量）。
    返回结果里带**可复核的验收数字**；任一项不达标就 ``REVIEW_REQUIRED``。
    """
    if not isinstance(jaw_mesh, trimesh.Trimesh) or len(jaw_mesh.faces) == 0:
        raise ValueError("jaw_mesh 必须是非空 trimesh.Trimesh")
    width = expected_crown_width_mm(target_fdi)

    verts = np.asarray(jaw_mesh.vertices, dtype=float)
    source_ids, target_points = _resolve_target_region(target_region, verts)

    # 切割半径的含义取决于"源是什么"，两者混用会把 support 撑大好几倍：
    #
    #   * 源是**已定位的区域**（几千个顶点）：测地线是从整个区域出发的最小值，
    #     所以半径 = 余量本身就够了 —— 结果等于"区域 ∪ 区域外 margin 一圈"。
    #     若再按"冠宽/2 + 余量"给半径，等于把 7000 多个源各扩 5.5mm，
    #     实测把 112 mm2 的目标扩成 662 mm2（8.7 倍冠面积）。
    #   * 源只是**一个种子点**：它还没铺开成区域，半径必须自己铺到冠部，
    #     所以半径 = 冠宽/2 + 余量。
    from_region = isinstance(target_region, ToothLocalizationResult) and bool(source_ids.size)
    base_radius = 0.0 if from_region else width / 2.0
    cut_radius = base_radius + float(margin_mm)
    if len(source_ids) == 0:
        return SupportSegmentationResult(
            target_fdi=target_fdi, mesh=None, status=SupportStatus.FAILED,
            confidence=0.0, cut_radius_mm=cut_radius,
            checks={"target_region_present": False},
            diagnostics={"reasons": ["target_region 为空或无法解析成顶点。"]})

    v_strength = vertex_valley_strength(jaw_mesh)
    dist, blocked = geodesic_distances(jaw_mesh, source_ids,
                                       valley_strength=v_strength)
    finite = np.isfinite(dist)
    mask = fill_mask_pinholes(jaw_mesh, finite & (dist <= cut_radius))

    sub = _submesh_from_vertices(jaw_mesh, mask)
    if sub is None or len(sub.faces) < 20:
        return SupportSegmentationResult(
            target_fdi=target_fdi, mesh=None, status=SupportStatus.FAILED,
            confidence=0.0, cut_radius_mm=cut_radius,
            checks={"has_faces": False},
            diagnostics={"reasons": ["切割半径内没有足够的三角面。"],
                         "n_blocked_edges": int(blocked.sum())})

    # 单连通：只保留最大连通块
    sub.remove_unreferenced_vertices()
    n_components_before = len(trimesh.graph.connected_components(
        sub.face_adjacency, min_len=1)) if len(sub.faces) else 0
    if n_components_before > 1:
        parts = sub.split(only_watertight=False)
        sub = max(parts, key=lambda p: float(p.area))
        sub.remove_unreferenced_vertices()

    # ---- 验收数字
    from scipy.spatial import cKDTree

    sub_verts = np.asarray(sub.vertices, dtype=float)
    # 把原网格的顶点凹度和位置搬到子网格上（顶点来自原网格，最近邻距离应为 0）
    _, nearest = cKDTree(verts).query(sub_verts)

    # 边界：用 outline() 取有序边界环。注意不能用"集合内/外的穿越边"来找 ——
    # 子网格自己的 edge 表里不存在外部顶点，穿越判据恒为假。
    try:
        n_loops: int | None = boundary_loop_count(sub)
    except Exception:      # pragma: no cover - 退化网格
        n_loops = None
    be = boundary_edges(sub)
    bnd_points = (np.asarray(sub.vertices, dtype=float)[np.unique(be)]
                  if len(be) else np.empty((0, 3)))

    # 边界凹度取**切口前沿**（区域外一圈紧邻顶点），不是区域自己的边界顶点 ——
    # 硬屏障把凹脊的边断开后，边界顶点会落在屏障内侧的平滑面上，凹度恒为 0。
    boundary_valley, n_front = cut_front_strength(jaw_mesh, nearest, v_strength)
    if len(bnd_points):
        _, bnd_nearest = cKDTree(verts).query(bnd_points)
        bnd_strength = v_strength[bnd_nearest]
        boundary_vertex_valley = float(bnd_strength.mean())
        convex_cut_fraction = float((bnd_strength < FLAT_ANGLE_RAD / np.pi).mean())
    else:
        bnd_strength = np.empty(0)
        boundary_vertex_valley = 0.0
        convex_cut_fraction = 1.0

    # 目标区域保留率：目标区域的顶点有多少**仍然在 support 顶点集里**
    sub_tree = cKDTree(sub_verts)
    d_target, _ = sub_tree.query(verts[source_ids])
    target_retention = float((d_target <= 1e-6).mean())

    support_extent = float(np.linalg.norm(sub_verts.max(axis=0)
                                          - sub_verts.min(axis=0)))
    area_ratio = float(sub.area / expected_surface_area_mm2(target_fdi))

    checks = {
        "single_connected_component": bool(n_components_before == 1),
        "n_components_before_pruning": int(n_components_before),
        "target_retention": round(target_retention, 4),
        "target_retention_ok": bool(target_retention >= MIN_TARGET_RETENTION),
        "boundary_valley_strength_mean": round(boundary_valley, 4),
        "boundary_valley_ok": bool(boundary_valley >= MIN_BOUNDARY_VALLEY_STRENGTH),
        "n_cut_front_vertices": int(n_front),
        "boundary_vertex_valley_mean": round(boundary_vertex_valley, 4),
        #: 切割边界是否**依附于解剖凹谷**（邻接沟/龈沟）。False 表示这是一条
        #: **度量切割**（按半径切），而不是解剖边界。这两者的可信度差很多，
        #: 所以单独作为一个字段报出来，不藏在 reasons 里。
        "boundary_is_anatomical": bool(
            boundary_valley >= MIN_BOUNDARY_VALLEY_STRENGTH),
        "convex_cut_fraction": round(convex_cut_fraction, 4),
        "area_ratio_ok": bool(0.5 <= area_ratio <= MAX_SUPPORT_AREA_RATIO),
        "n_boundary_points": int(len(bnd_points)),
        "n_boundary_loops": n_loops,
        "watertight": bool(sub.is_watertight),
        "n_vertices": int(len(sub.vertices)),
        "n_faces": int(len(sub.faces)),
        "area_mm2": round(float(sub.area), 4),
        "area_ratio_vs_crown": round(area_ratio, 4),
        "extent_mm": round(support_extent, 4),
        "cut_radius_mm": round(cut_radius, 4),
    }

    reasons: list[str] = []
    if not checks["target_retention_ok"]:
        reasons.append(
            f"目标区域只被保留了 {target_retention:.3f}（下限 {MIN_TARGET_RETENTION}）："
            "切割半径太小或凹陷屏障把目标牙本身切断了。")
    if not checks["boundary_valley_ok"]:
        reasons.append(
            f"support 切口前沿平均凹度只有 {boundary_valley:.3f}（下限 "
            f"{MIN_BOUNDARY_VALLEY_STRENGTH}）：边界走在凸面上，"
            f"其中 {convex_cut_fraction * 100:.1f}% 的边界顶点几乎不凹 —— "
            "这部分**切进了牙冠外形**，很可能带上了邻牙。"
            "纯局部几何无法可靠切开平滑的近远中鞍部；请人工检查或缩小 margin_mm。")
    if n_loops is not None and n_loops > 1:
        reasons.append(f"support 有 {n_loops} 条边界环（期望 1），说明切出了多个补片。")
    if area_ratio < 0.5:
        reasons.append(f"support 面积只有期望冠面积的 {area_ratio:.2f} 倍，偏小："
                       "目标牙可能没被完整切下来。")
    if area_ratio > MAX_SUPPORT_AREA_RATIO:
        reasons.append(
            f"support 面积达期望冠面积的 {area_ratio:.2f} 倍（上限 "
            f"{MAX_SUPPORT_AREA_RATIO}）：切下来的表面明显多于一颗牙，"
            "**很可能带上了邻牙或牙龈**。"
            + ("注意该处没有可依附的解剖凹谷（边界凹度 "
               f"{boundary_valley:.3f}），边界是按半径切的**度量切割**，"
               "无法靠局部几何自动避开邻牙 —— 请人工检查，或调小 margin_mm。"
               if not checks["boundary_is_anatomical"] else ""))

    confidence = float(np.clip(
        0.45 * min(target_retention, 1.0)
        + 0.35 * min(boundary_valley / (2.0 * MIN_BOUNDARY_VALLEY_STRENGTH), 1.0)
        + 0.20 * (1.0 if checks["single_connected_component"] else 0.0), 0.0, 1.0))
    status = SupportStatus.REVIEW_REQUIRED if reasons else SupportStatus.OK
    if target_retention < 0.5:
        status = SupportStatus.FAILED

    return SupportSegmentationResult(
        target_fdi=target_fdi, mesh=sub, status=status, confidence=confidence,
        cut_radius_mm=cut_radius, checks=checks,
        diagnostics={
            "margin_mm": float(margin_mm),
            "base_radius_mm": round(float(base_radius), 4),
            "sources_are_region": bool(from_region),
            "n_source_vertices": int(len(source_ids)),
            "n_blocked_edges": int(blocked.sum()),
            "target_position": [round(float(x), 4) for x in target_points.mean(axis=0)],
            "reasons": reasons,
            "note": ("边界凹度是**报告**出来的越界程度，不是保证。"
                     "近远中接触处的平滑鞍部纯局部几何切不开。"),
        })


def _resolve_target_region(target_region, verts: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """把各种形态的 target_region 统一成 (源顶点 id 数组, 区域点数组)。"""
    if isinstance(target_region, ToothLocalizationResult):
        ids = np.asarray(target_region.region_vertex_ids, dtype=int)
        if len(ids):
            return ids, verts[ids]
        return (np.array([], dtype=int),
                np.asarray(target_region.position, dtype=float).reshape(1, 3))
    if isinstance(target_region, (int, np.integer)):
        vid = int(target_region)
        if not (0 <= vid < len(verts)):
            raise ValueError(f"顶点 id {vid} 越界")
        return np.array([vid]), verts[[vid]]

    pts = np.asarray(target_region, dtype=float)
    if pts.ndim == 1:
        if pts.shape[0] != 3:
            raise ValueError("单点 target_region 必须是三个坐标值")
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("target_region 必须是 (N,3) 点集、(3,) 单点、顶点 id "
                         "或 ToothLocalizationResult")
    if len(pts) == 0:
        return np.array([], dtype=int), pts
    from scipy.spatial import cKDTree
    ids = np.unique(cKDTree(verts).query(pts)[1])
    return ids, pts
