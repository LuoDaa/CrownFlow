"""Contact Engine v1 —— 邻接接触优化（提交 I）。

只解决一件事：**把牙冠的近中/远中邻接区调整到合理接触，同时不破坏
Margin、咬合面和整体解剖。**

流程（用户规格）：

    CrownOuter H.1 输出（冠部外表面 + 颈环）
        + 近中邻牙 patch + 远中邻牙 patch
        ↓
    接触区顶点沿"到邻牙最近表面点"方向移动到 target_gap
    （穿透则反向退让）
        ↓
    Laplacian 行替换传播（颈环行恒等=LOCK，咬合区弱锚，颊舌平滑）

变形范围锁死：

    MESIAL/DISTAL_CONTACT_REGION   可调整
    Margin / Cervical              LOCK（硬约束）
    Occlusal                       强限制（弱锚）
    Buccal / Lingual               平滑传播

## DESIGN_TARGET vs REALIZED_GEOMETRY（永久区分，用户规格）

实测证据：5 例 XML ``DistanceToNeighbor = 0.0``，而技师 Final Crown 实际
mesial 0.079 / distal 0.036 mm。所以：

* ``ContactParameters.design_target_mm`` 只记录 DESIGN_TARGET（XML），
  **不作为求解目标**；
* 求解目标是 ``target_gap_mm``（PROVISIONAL，取 GT 反求区间中值）；
* 验收比较的是与 REALIZED_GEOMETRY（GT）的距离/面积/位置。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.spatial import cKDTree

from .laplacian import DeformInputs, solve_laplacian
from .models import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REVIEW_REQUIRED,
    CrownOuterConfig,
)
from .patient_frame import PatientFrame

#: 求解目标间隙。GT 反求（18 例中唯一可用病例）：mesial 0.079 / distal
#: 0.036 mm —— 取中值。**PROVISIONAL**：更多 REALIZED 数据回来后修订。
DEFAULT_TARGET_GAP_MM = 0.05


class ContactError(ValueError):
    pass


@dataclass
class ContactParameters:
    """Contact v1 的全部可调参数。真实回归前 PROVISIONAL。"""

    #: 求解目标间隙（REALIZED 口径，见模块 docstring）。
    target_gap_mm: float = DEFAULT_TARGET_GAP_MM  # PROVISIONAL

    #: DESIGN_TARGET（XML DistanceToNeighbor）。**只记录，不驱动求解**。
    design_target_mm: float | None = None

    #: 接触区横向带宽：sign·u ≥ 该比例 × margin 半跨（0.6 ≈ 邻接面）。
    contact_band_fraction: float = 0.6  # PROVISIONAL

    #: 颈缘锁死带宽：距颈环 < 该值的顶点不动（mm）。
    #: 颈环本身另由硬约束 LOCK，这里只需挡住紧贴颈环的一圈。
    #: 实测 1.5mm 会把"颈缘上方 1.5mm 内的邻接面"排除掉，在那里留下
    #: 1.2~1.5mm 的残留干涉（56/62 个残留顶点都落在这一带）。
    cervical_lock_mm: float = 0.3  # PROVISIONAL

    #: 外层迭代次数。**默认 1（单次求解）**。
    #: 为什么默认 1：朴素外层迭代（每轮重新测量-重解、并累积约束）在真实
    #: 病例上**发散** —— 实测 3 轮后法向翻转 32%（3779 面）、lingual 位移
    #: +4.97mm、cervical +3.75mm。原因是每轮的膨胀锚点钉在**已移动**的
    #: 位置上，误差逐步累积。要真正迭代必须加阻尼（或改为对邻牙建符号
    #: 距离场后一次性投影）。在解决之前默认单次，残留穿透如实报
    #: REVIEW_REQUIRED，不假装收敛。**PROVISIONAL**。
    n_iterations: int = 1

    #: 约束集膨胀圈数。硬钉住一片顶点时，紧邻它的环会被 Laplacian 平滑
    #: 拖向邻牙，形成"接触区外一圈"的残留穿透 —— 膨胀一圈把他们一起
    #: 约束住即可（实测残留穿透从 1.16mm 降到目标量级）。
    dilate_steps: int = 1  # PROVISIONAL

    #: 咬合锚分位（与 CrownOuterConfig 同义，弱锚保持解剖）。
    occlusal_anchor_percentile: float = 75.0  # PROVISIONAL
    #: **Contact v2：局部带状求解**（默认开）。自由区 = 修改区 + 过渡环；
    #: 颈环与咬合区是固定边界 —— 位移由构造精确为 0。v1 的全局路径
    #: （弱锚 α=300）保留做对照：注释里写"硬锁"但 solve_laplacian 实际把
    #: 咬合放进弱锚块，实测真实例咬合被拖 1.08mm。
    local_band: bool = True
    #: 修改区外扩的过渡环数（顶点 BFS）。
    transition_rings: int = 3
    #: **咬合"表面"的法向判据**：|n̂·就位轴| ≥ 此值的 z-p75 区顶点才算咬合面
    #: （锁定）。磨牙的邻接壁在牙尖高度也有高 z，但法向是径向 —— 只按 z 分位
    #: 会把邻接接触区误锁死（实测真实例残留穿透 2.0mm 全在那里）。
    occ_table_normal_dot: float = 0.6

    #: 单顶点最大位移；需要更多 → DESIGN_INFEASIBLE（邻牙太远，不能把
    #: 整颗冠拉过去）。GT 反求显示技师在近中面削掉了 ~1.5mm 的凸度，
    #: 所以 2.5 留有余量。
    max_displacement_mm: float = 2.5  # PROVISIONAL

    #: "内部"判定的欧氏距离上限：signed<0 且 euclid ≤ inside_reach 才算
    #: 真穿透（带符号距离对切向偏远的顶点会高估穿透深度）。
    inside_reach_mm: float = 2.0  # PROVISIONAL

    #: 间隙已达标就不动的容差。
    gap_tolerance_mm: float = 0.03  # PROVISIONAL

    #: 捕捉带宽：只把 gap ≤ (min_gap + capture_band) 的顶点投影到目标间隙
    #: —— 接触区是**最接近处附近的一小块**，侧翼（wrap-around）不投影，
    #: 否则冠面会被压平到邻牙平面上。留给 Laplacian 平滑过渡。
    capture_band_mm: float = 0.6  # PROVISIONAL

    #: 接触判定距离（面积/质心统计口径）。
    contact_mm: float = 0.15  # PROVISIONAL

    #: 解除穿透后退让的目标间隙与判定。
    penetration_tolerance_mm: float = 0.05  # PROVISIONAL

    #: **形变增益**（硬化 line search 用）：顶点只走 alpha×(目标−当前位置)。
    #: 1.0 = 直接到目标（旧行为）；硬化器依次试 1, 1/2, 1/4 …
    alpha: float = 1.0
    #: 几何硬门尺度（mm）：翻面/自交按 (scale/2)² 面积分级 micro。
    gate_scale_mm: float = 0.10  # 构造尺度，非临床阈值

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in sorted(self.__dataclass_fields__)
                } | {"_provisional": True}


@dataclass
class SideContactReport:
    """单侧接触报告（mesial / distal 共用）。"""

    side: str
    status: str = "NOT_APPLICABLE"
    min_distance_mm: float | None = None
    penetration_mm: float = 0.0          # 正值 = 穿透深度
    contact_area_mm2: float = 0.0
    contact_centroid = None              # (3,) world
    vertices_modified: int = 0
    max_required_displacement_mm: float = 0.0
    n_region_vertices: int = 0

    def as_dict(self) -> dict:
        return {
            "side": self.side, "status": self.status,
            "min_distance_mm": (round(self.min_distance_mm, 4)
                                if self.min_distance_mm is not None else None),
            "penetration_mm": round(self.penetration_mm, 4),
            "contact_area_mm2": round(self.contact_area_mm2, 3),
            "contact_centroid": (None if self.contact_centroid is None
                                 else [round(float(v), 3)
                                       for v in self.contact_centroid]),
            "vertices_modified": self.vertices_modified,
            "max_required_displacement_mm": round(
                self.max_required_displacement_mm, 4),
            "n_region_vertices": self.n_region_vertices,
        }


@dataclass
class ContactResult:
    mesh=None                            # trimesh 或 (vertices, faces)
    status: str = STATUS_FAILED
    notes: list = field(default_factory=list)
    mesial: SideContactReport = field(default_factory=lambda: SideContactReport("mesial"))
    distal: SideContactReport = field(default_factory=lambda: SideContactReport("distal"))
    margin_max_displacement_mm: float | None = None
    occlusal_p95_displacement_mm: float | None = None
    n_flipped_faces: int | None = None
    patch_normal_flipped: bool = False
    parameters: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"status": self.status, "notes": self.notes,
                "mesial": self.mesial.as_dict(),
                "distal": self.distal.as_dict(),
                "margin_max_displacement_mm": self.margin_max_displacement_mm,
                "occlusal_p95_displacement_mm": self.occlusal_p95_displacement_mm,
                "n_flipped_faces": self.n_flipped_faces,
                "patch_normal_flipped": self.patch_normal_flipped}


# ---------------------------------------------------------------- 邻牙 patch

def closest_point_on_triangle(p, a, b, c):
    """点到三角形的最近点（Ericson《Real-Time Collision Detection》§5.1.5）。"""
    ab, ac = b - a, c - a
    ap = p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        t = d1 / (d1 - d3)
        return a + t * ab
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        t = d2 / (d2 - d6)
        return a + t * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + t * (c - b)
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return a + ab * v + ac * w


class NeighborPatch:
    """邻牙局部表面：AABB 候选三角形上的精确最近点。"""

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        if len(self.faces) == 0:
            raise ContactError("邻牙 patch 没有三角形")
        tri = self.vertices[self.faces]
        self.centroids = tri.mean(axis=1)
        # 法向（单位）；退化面剔除
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        keep = ln > 1e-12
        self.faces = self.faces[keep]
        self.normals = n[keep] / ln[keep, None]
        self.centroids = self.centroids[keep]
        self.normal_sign = 1.0              # 自动定向后可能 −1

        # ---- 开放 patch 的边界（I.1）：最近点落在边界附近时法向不可靠，
        # 必须能被识别出来（未识别时实测出现 −4mm 的伪 signed distance）。
        from collections import Counter
        cnt: Counter = Counter()
        for a, b, c3 in self.faces:
            for u, v in ((a, b), (b, c3), (c3, a)):
                cnt[(min(int(u), int(v)), max(int(u), int(v)))] += 1
        bedges = [e for e, n in cnt.items() if n == 1]
        self.n_boundary_edges = len(bedges)
        samples = []
        for u, v in bedges:
            pu, pv = self.vertices[u], self.vertices[v]
            seg = float(np.linalg.norm(pv - pu))
            n_s = max(2, int(seg / 0.2) + 1)
            for t in np.linspace(0.0, 1.0, n_s):
                samples.append(pu + t * (pv - pu))
        self._boundary_tree = cKDTree(np.asarray(samples)) \
            if samples else None

    def closest_with_validity(self, points, *, boundary_safety_mm: float = 0.5):
        """``closest`` + 边界可靠性（I.1）。

        返回 ``(cp, fids, signed, dist, valid)``；``valid=False`` 表示最近点
        落在开放边界的安全带内 —— 那里的法向不能当接触约束用，
        语义记 ``AMBIGUOUS_PATCH_EDGE``。
        """
        cp, fids, signed, dist = self.closest(points)
        if self._boundary_tree is None:
            valid = np.ones(len(cp), dtype=bool)
        else:
            d_b = self._boundary_tree.query(cp, k=1)[0]
            valid = d_b >= boundary_safety_mm
        return cp, fids, signed, dist, valid

    # ---- 全项目唯一 clearance 口径（用户 2026-10-09 决策 ③）-----------------
    #: 决策路径（H.2.x / K0 / Contact solver / Contact audit / Contact Gate）
    #: **只允许**调用下面两个方法。raw `closest`/`closest_with_validity` 保留给
    #: 实现内部与取证用 —— 实测 raw dot 在三角形 crease / 顶点最近点处会跳变：
    #: 深穿 1.2mm 读成 0.03mm，8.5µm 位移后读到 −1.13mm。0.2mm 验收阈值不变，
    #: 变的是测量 primitive（测量 bugfix，不是 Gate 放宽）。
    CLEARANCE_BOUNDARY_SAFETY_MM = 0.6
    CLEARANCE_OBLIQUITY_MIN = 0.90

    def clearance_detail(self, points, *,
                         boundary_safety_mm: float | None = None,
                         obliquity_min: float | None = None):
        """→ ``(signed, valid, cp, fids, parts)``；``parts`` 带两类歧义掩码。

        * ``signed = sign(dot(p−c, n̂)·normal_sign) · euclid``
          （幅值单规则用 euclid：连续、单调；dot 会随获胜面切换跳变）
        * ``valid = 边界有效 & 法向可靠``
          （``AMBIGUOUS_PATCH_EDGE`` / ``AMBIGUOUS_CREASE``）
        * 正 = 邻牙外侧（安全），负 = 穿进邻牙
        """
        bs = (self.CLEARANCE_BOUNDARY_SAFETY_MM if boundary_safety_mm is None
              else float(boundary_safety_mm))
        ob = (self.CLEARANCE_OBLIQUITY_MIN if obliquity_min is None
              else float(obliquity_min))
        cp, fids, signed, dist, valid_b = self.closest_with_validity(
            points, boundary_safety_mm=bs)
        signed = np.asarray(signed, np.float64)
        dist = np.asarray(dist, np.float64)
        obl = np.abs(signed) / np.maximum(dist, 1e-12)
        valid_c = obl >= ob
        valid_b = np.asarray(valid_b, bool)
        # ⚠️ `closest` 返回的 signed **已经**含 normal_sign（见其末行
        # `out_s[i] = dot(p−q, n̂) * self.normal_sign`）——这里绝不能再乘一次：
        # 双重翻转在合成夹具（normal_sign=−1）上把 3.8mm 外侧读成 3.8mm 穿入，
        # 真实数据两侧都是 +1 所以一直没暴露。幅值单规则用 euclid、符号沿用。
        sc = np.sign(signed) * dist
        parts = {"valid_boundary": valid_b, "valid_crease": valid_c,
                 "obliquity": obl}
        return sc, (valid_b & valid_c), cp, fids, parts

    def clearance(self, points, **kw):
        """→ ``(signed, valid, cp, fids)``（见 :meth:`clearance_detail`）。"""
        sc, valid, cp, fids, _parts = self.clearance_detail(points, **kw)
        return sc, valid, cp, fids

    def closest(self, points):
        """返回精确最近三角面点、面 ID、局部法向符号、欧氏距离。"""
        points = np.asarray(points, dtype=np.float64)
        if not len(points): return points.copy(),np.zeros(0,int),np.zeros(0),np.zeros(0)
        import trimesh
        if not hasattr(self,'_query_mesh'):
            self._query_mesh=trimesh.Trimesh(self.vertices,self.faces,process=False)
        cp=np.empty_like(points);distance=np.empty(len(points));ids=np.empty(len(points),int)
        for start in range(0,len(points),512):
            batch=points[start:start+512]
            candidates=[np.asarray(sorted(faces),int) for faces in trimesh.proximity.nearby_faces(self._query_mesh,batch)]
            lengths=np.asarray([len(faces) for faces in candidates]);flat=np.concatenate(candidates)
            query=np.repeat(batch,lengths,axis=0)
            closest_all=trimesh.triangles.closest_point(self._query_mesh.triangles[flat],query)
            squared_all=np.sum((closest_all-query)**2,axis=1);begin=0
            for offset,faces in enumerate(candidates):
                end=begin+len(faces);closest=closest_all[begin:end];squared=squared_all[begin:end]
                # Shared-edge ties must use the same face after rigid transforms.
                best=np.flatnonzero(squared<=squared.min()+1e-12)[0]
                index=start+offset;cp[index]=closest[best];ids[index]=faces[best];distance[index]=np.sqrt(squared[best])
                begin=end
        signed=np.einsum('ij,ij->i',points-cp,self.normals[ids])*self.normal_sign
        return cp,ids,signed,distance

    def orient_toward(self, points):
        """**已废弃**：多数投票对开放 patch 不成立。

        邻牙 patch 没有严格 inside/outside：若冠整体陷入邻牙表面之后，
        多数票必然把法向翻到"看起来都在正侧"，于是穿透检测失效
        （实测：均匀穿透的合成例报 penetration=0）。保留仅为兼容，
        新代码请用 ``orient_against_crown``。"""
        _p, _f, s, _d = self.closest(points)
        flipped = False
        if float(np.median(s)) < 0:
            self.normal_sign = -1.0
            flipped = True
            _p, _f, s, _d = self.closest(points)
        return flipped

    def orient_against_crown(self, points, crown_normals):
        """用**冠的自身外法向**定向 patch 法向。返回是否翻转。

        判据有明确的物理语义：接触区里邻牙外法向与冠外法向应当**相对**
        （面对面），即 ``dot(crown_n, patch_n) < 0``。这不需要 patch 是
        封闭体，也不受"整体穿透"影响 —— 冠的法向是自己网格的固有属性。
        """
        _p, fids, _s, _d = self.closest(points)
        pn = self.normals[fids]
        d = float(np.mean(np.einsum("ij,ij->i", crown_normals, pn)))
        flipped = d > 0.0
        if flipped:
            self.normal_sign = -1.0
        return flipped

    def normals_at(self, points):
        """查询点处（最近面）的法向 × normal_sign。"""
        _p, fids, _s, _d = self.closest(points)
        return self.normals[fids] * self.normal_sign


# ---------------------------------------------------------------- 区域选择

def _boundary_loop(vertices, faces):
    """开放面片的边界环（有序）。复用 eoff 的拓扑判据。"""
    from collections import Counter, defaultdict
    cnt = Counter()
    adj = defaultdict(list)
    for a, b, c3 in faces:
        for u, v in ((a, b), (b, c3), (c3, a)):
            e = (min(u, v), max(u, v))
            cnt[e] += 1
    boundary = [e for e, n in cnt.items() if n == 1]
    if not boundary:
        return None
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)
    if any(len(v) != 2 for v in adj.values()):
        return None
    start = min(adj)
    loop = [start]
    prev, cur = None, start
    while True:
        nxt = [w for w in adj[cur] if w != prev]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    return loop if len(loop) == len(adj) else None


def _region_mask(vertices, frame, loop_indices, params, side):
    """接触区顶点 mask：横向带宽内、颈缘锁外、**咬合区分位以下**。

    轴向 cut 是规格要求（Occlusal 强限制、不属于接触区）；没有它，
    咬合面边缘的顶点离邻牙平面很远，会虚假地撑爆 max_displacement。"""
    origin, x, y, z = frame.origin, frame.x, frame.y, frame.z
    rel = vertices - origin
    u = rel @ x
    sign = 1.0 if side == "mesial" else -1.0
    loop = vertices[loop_indices]
    loop_u = (loop - origin) @ x
    u_edge = float(np.max(sign * loop_u))
    u_min = params.contact_band_fraction * u_edge
    # 颈缘锁：到颈环顶点的距离
    from scipy.spatial import cKDTree
    d_loop = cKDTree(loop).query(vertices, k=1)[0]
    crown = np.ones(len(vertices), dtype=bool)
    crown[loop_indices] = False
    z_thr = np.percentile(vertices[crown, 2], params.occlusal_anchor_percentile)
    return ((sign * u >= u_min) & (d_loop >= params.cervical_lock_mm)
            & (vertices[:, 2] < z_thr))


# ---------------------------------------------------------------- 主入口

#: Contact 求解专用的咬合弱锚权重 —— 比 CrownOuter 默认（0.3）强一个量级：
#: 接触区的位移（可达 ~1mm）沿 Laplacian 传播会明显拖动咬合面，
#: 实测 0.3 时咬合 p95 位移 0.56mm，3.0 时应显著更小。**PROVISIONAL**。
#: 咬合锚强度。用**正确的**邻牙 patch 后约束数从 ~700 增到 ~8400 个顶点、
#: 单点修正达 1.3mm，10 压不住（实测咬合仍被拖 0.99mm，违反规格
#: "Occlusal ≈ 0"）。提到 300 使其接近硬约束 —— 规格把咬合定义为"强限制"，
#: 与"接触区可调整"本就该差一个量级以上。
CONTACT_OCCLUSAL_ANCHOR = 300.0


def _side_need(vertices, patch, mask, params, adj, exclude=None, wide=None):
    """单侧的约束集与目标位置。

    返回 ``(signed, dist, need_idx, target_xyz, max_disp, penetration_mm)``。
    ``need_idx`` 已按 ``dilate_steps`` 沿网格邻接膨胀：硬钉住一片顶点时，
    紧邻它的环会被 Laplacian 平滑拖向邻牙，形成"接触区外一圈"的残留穿透。
    """
    empty = np.array([], dtype=np.int64)
    # 度量域 = wide（该侧半边，排除颈环）：**穿透消除不限带宽** ——
    # 哪里陷入邻牙就要在哪里修，带宽只用于"把接触区拉向邻牙"那一类。
    # 用带宽当度量域时，紧贴带宽边界的穿透点会被切在外面，留下 1mm 级
    # 残留干涉（实测 dilate 1~5 圈都消不掉，因为根本不是传播问题）。
    wide = mask if wide is None else wide
    wids = np.flatnonzero(wide)
    if len(wids) == 0:
        return None, None, empty, np.zeros((0, 3)), 0.0, 0.0
    # **统一口径**（用户 2026-10-09 决策 ③）：幅值 = euclid（连续）、
    # 有效性 = 边界 + crease；`dist ≡ |signed|` 由此定义。
    signed, valid, cp, fids, _parts = patch.clearance_detail(vertices[wids])
    dist = np.abs(np.asarray(signed, np.float64))
    valid = np.asarray(valid, bool)
    in_band = mask[wids]
    # 穿透统计用"真穿透"口径：signed<0、欧氏距离不过大、且点本身有效
    # （|signed| = euclid，所以上界同时限制深度不被切向远处顶点夸大）。
    inside = (signed < 0) & (dist <= params.inside_reach_mm) & valid
    pen = float(max(0.0, -np.min(signed[inside])
                    - params.penetration_tolerance_mm)) if inside.any() else 0.0

    # 捕捉 = 并集（两个语义上都要修）：
    #   1) 穿透集 signed < target_gap —— 任何深度的陷入都要推出来，与切向
    #      距离无关（否则会在邻牙体内留下 1mm 级残留干涉）；
    #   2) 邻近集 dist <= dist_min + band —— 邻牙还在几 mm 外时，把"离它
    #      最近的那一带"拉过去（规格场景 1：邻牙太远 -> 向邻牙移动）。
    nhat = patch.normals[fids] * patch.normal_sign
    # 穿透集用与统计一致的"真穿透"口径：signed < target_gap、欧氏距离
    # 不过大、且点有效。无效点 = 开放 patch 切口边/crease 的法向伪影
    # （实测误报 4.09mm 不可行）。
    penetrate = (signed < params.target_gap_mm) & (
        dist <= params.inside_reach_mm) & valid
    dmin = float(np.min(dist[valid])) if valid.any() else 0.0
    near = in_band & valid & (dist <= dmin + params.capture_band_mm)
    need_local = (penetrate | near) & (
        np.abs(params.target_gap_mm - signed) > params.gap_tolerance_mm)
    # 修正量 = |target_gap - signed|（沿邻牙面法向的**最小修正**）——
    # 单点位移始终有界，物理上就是技师在邻接面磨掉/补上的量。
    # 真正需要修的顶点（决定可行性）：max_disp 只看它们 —— 位移上限
    # 是"把干涉修掉需要磨多少"，不该被下面的数值装置放大。
    move_ids = wids[need_local]
    if exclude is not None and len(move_ids):
        move_ids = move_ids[~exclude[move_ids]]
    inv = {int(v): i for i, v in enumerate(wids)}
    if len(move_ids) == 0:
        return signed, dist, empty, np.zeros((0, 3)), 0.0, pen
    loc = np.asarray([inv[int(v)] for v in move_ids], dtype=np.int64)
    sign_mv = signed[loc]
    target_mv = vertices[move_ids] + nhat[loc] * (
        params.target_gap_mm - sign_mv)[:, None]
    max_disp = float(np.max(np.abs(params.target_gap_mm - sign_mv)))

    # 膨胀环：把接触区**外一圈钉在原位**，挡住 Laplacian 平滑把它拖进邻牙。
    # 注意语义是"锚定"而不是"投影到邻牙"：把这一圈也投到邻牙上会让一个
    # 5mm 外的顶点被要求移动 4.95mm（实测 distal 因此误报 4.55mm 不可行）。
    anchor_ids = np.array([], dtype=np.int64)
    if params.dilate_steps > 0:
        front = set(int(v) for v in move_ids)
        seen = set(front)
        for _ in range(params.dilate_steps):
            nxt = set()
            for v in front:
                nxt |= adj.get(v, set())
            front = nxt - seen
            seen |= nxt
        anchor_ids = np.asarray(sorted(front), dtype=np.int64)
        if exclude is not None and len(anchor_ids):
            anchor_ids = anchor_ids[~exclude[anchor_ids]]

    need = np.concatenate([move_ids, anchor_ids])
    target = np.vstack([target_mv, vertices[anchor_ids]]) if len(anchor_ids)         else target_mv
    return signed, dist, need, target, max_disp, pen


def optimize_contacts(crown_mesh, patient_frame: PatientFrame,
                      mesial_neighbor: NeighborPatch,
                      distal_neighbor: NeighborPatch,
                      contact_parameters: ContactParameters | None = None,
                      *,
                      laplacian_config: CrownOuterConfig | None = None,
                      ) -> ContactResult:
    """Contact v1 主入口（外层迭代）。

    **为什么要迭代**：把顶点沿"当前最近面的法向"移到目标间隙是一阶近似；
    邻牙是曲面，移动后最近面会变，单次解会留下 1mm 级的残留穿透
    （实测 mesial 1.16 / distal 0.98）。外层迭代重新测量-重解，逐次收敛。
    """
    params = contact_parameters or ContactParameters()
    res = None
    mesh = crown_mesh
    for it in range(max(1, params.n_iterations)):
        res = _solve_once(mesh, patient_frame, mesial_neighbor,
                          distal_neighbor, params, laplacian_config)
        res.parameters = params.as_dict() | {"iterations_used": it + 1}
        pen = max(res.mesial.penetration_mm, res.distal.penetration_mm)
        # 收敛判据是**穿透是否消除**，不是 status：REVIEW_REQUIRED 恰恰是
        # "还有穿透、该继续迭代"的状态。只有 INFEASIBLE/FAILED 才停手。
        if (pen <= params.penetration_tolerance_mm
                or res.status in ("DESIGN_INFEASIBLE", STATUS_FAILED)
                or res.mesh is None or res.mesh is crown_mesh):
            break
        mesh = res.mesh
    return res


def harden_contacts(crown_mesh, patient_frame: PatientFrame,
                    mesial_neighbor: NeighborPatch,
                    distal_neighbor: NeighborPatch,
                    contact_parameters: ContactParameters | None = None,
                    *,
                    alphas=(1.0, 0.5, 0.25, 0.125, 0.0625)) -> dict:
    """Contact **硬化**（用户 2026-10-08 规格）：α line search + 几何硬门。

    接受一个步，必须同时满足

        editable penetration p95 ↓  AND  max 不恶化
        AND 新增自交 = 0  AND 非流形不新增  AND 翻面 = 0
        AND margin 位移 = 0  AND 咬合锁定区位移 = 0

    所有 α 都不行 → ``CONTACT_CONSTRAINT_CONFLICT``（**拒绝以折叠换穿透清零**）。
    返回 ``{"result": ContactResult, "alpha": …, "trials": […], "conflict": bool}``。
    """
    from .deform_gate import deformation_gate

    base_params = contact_parameters or ContactParameters()
    # baseline：α=0 走一遍同样的求解链（无位移，只拿测量）
    p0 = replace(base_params, alpha=0.0)
    base = _solve_once(crown_mesh, patient_frame, mesial_neighbor,
                       distal_neighbor, p0)
    v_base = np.asarray(base.mesh.vertices, dtype=np.float64)
    faces = np.asarray(base.mesh.faces, dtype=np.int64)
    # 锁定区 = 颈环 ∪ 咬合面（与 _solve_once 的 local_band 分支同一口径）
    loop_b = _boundary_loop(v_base, faces)
    locked_ids = set()
    if loop_b is not None:
        locked_ids |= {int(i) for i in np.asarray(loop_b)}
    from .contact_refine import occ_table_mask as _occ
    occ_t = _occ(v_base, faces, patient_frame, base_params)
    locked_ids |= {int(i) for i in np.asarray(occ_t)}
    pen0 = max(base.mesial.penetration_mm, base.distal.penetration_mm)
    trials = []
    # ⚠️ α=0 也是**一次求解**（Laplacian 会有微量漂移，实测咬合面 0.156mm）。
    # 所以"不做形变"时必须返回**原网格**，而不是 α=0 的解 —— 否则等于
    # 偷偷动了不该动的面。
    base.mesh = crown_mesh
    if pen0 <= base_params.penetration_tolerance_mm:
        return {"result": base, "alpha": 0.0, "trials": trials,
                "conflict": False,
                "note": "baseline 已在穿透容差内，无需形变"}
    for alpha in alphas:
        p = replace(base_params, alpha=float(alpha))
        try:
            res = _solve_once(crown_mesh, patient_frame, mesial_neighbor,
                              distal_neighbor, p)
        except Exception as exc:                     # noqa: BLE001
            trials.append({"alpha": alpha, "error": str(exc)[:120]})
            continue
        v1 = np.asarray(res.mesh.vertices, dtype=np.float64)
        if len(v1) != len(v_base):
            trials.append({"alpha": float(alpha),
                           "error": "topology_changed(%d→%d)"
                           % (len(v_base), len(v1))})
            continue
        moved = np.flatnonzero(np.abs(v1 - v_base).max(axis=1) > 1e-12)
        focus = np.flatnonzero(np.isin(faces, moved).any(axis=1))
        gate = deformation_gate(v_base, v1, faces,
                                scale_mm=base_params.gate_scale_mm,
                                focus_faces=focus,
                                locked_ids=locked_ids)
        pen1 = max(res.mesial.penetration_mm, res.distal.penetration_mm)
        improved = pen1 < pen0 - 1e-9
        rec = {"alpha": float(alpha), "penetration_before": round(pen0, 4),
               "penetration_after": round(pen1, 4),
               "improved": bool(improved), "gate": gate.as_dict(),
               "accepted": False}
        if improved and gate.ok:
            rec["accepted"] = True
            trials.append(rec)
            return {"result": res, "alpha": float(alpha), "trials": trials,
                    "conflict": False}
        trials.append(rec)
    return {"result": base, "alpha": 0.0, "trials": trials, "conflict": True,
            "note": ("所有 α 均无法在保持几何安全的前提下改善 → "
                     "CONTACT_CONSTRAINT_CONFLICT")}


def _solve_once(crown_mesh, patient_frame: PatientFrame,
                mesial_neighbor: NeighborPatch,
                distal_neighbor: NeighborPatch,
                params: ContactParameters,
                laplacian_config: CrownOuterConfig | None = None,
                ) -> ContactResult:
    """单次测量-求解。"""
    res = ContactResult(parameters=params.as_dict())
    try:
        vertices = np.asarray(crown_mesh.vertices, dtype=np.float64)
        faces = np.asarray(crown_mesh.faces, dtype=np.int64)
        loop = _boundary_loop(vertices, faces)
        if loop is None:
            raise ContactError("冠网格没有可识别的颈环边界（应开放底边）")
        loop_idx = np.asarray(loop, dtype=np.int64)

        adj: dict[int, set] = {}
        for a, b, c3 in faces:
            for uu, vv in ((a, b), (b, c3), (c3, a)):
                adj.setdefault(int(uu), set()).add(int(vv))
                adj.setdefault(int(vv), set()).add(int(uu))

        patches = {"mesial": mesial_neighbor, "distal": distal_neighbor}
        # 防御：接触区绝不能含颈环顶点（颈环是硬 LOCK，两者会冲突）
        loop_set = np.zeros(len(vertices), dtype=bool)
        loop_set[loop_idx] = True
        masks, wides = {}, {}
        u_lat = (vertices - patient_frame.origin) @ patient_frame.x
        for s in ("mesial", "distal"):
            masks[s] = _region_mask(
                vertices, patient_frame, loop_idx, params, s) & ~loop_set
            sign = 1.0 if s == "mesial" else -1.0
            wides[s] = (sign * u_lat >= -0.5) & ~loop_set

        for patch in patches.values():
            patch.orient_toward(vertices[~loop_set])

        desired = np.full_like(vertices, np.nan)
        modified = np.zeros(len(vertices), dtype=bool)
        reports = {}
        infeasible = []
        for side, patch in patches.items():
            rep = SideContactReport(side=side)
            rep.n_region_vertices = int(masks[side].sum())
            signed, dist, need, target, max_disp, pen = _side_need(
                vertices, patch, masks[side], params, adj, loop_set,
                wide=wides[side])
            if signed is None:
                rep.status = "NO_REGION"
                reports[side] = rep
                continue
            rep.min_distance_mm = float(np.min(dist))
            rep.penetration_mm = pen
            rep.vertices_modified = int(len(need))
            rep.max_required_displacement_mm = max_disp
            if len(need) and max_disp > params.max_displacement_mm:
                rep.status = "INFEASIBLE"
                infeasible.append(side)
            else:
                rep.status = "OK"
                desired[need] = target
                modified[need] = True
            reports[side] = rep
        res.mesial, res.distal = reports["mesial"], reports["distal"]

        if infeasible:
            res.status = "DESIGN_INFEASIBLE"
            res.notes.append(
                "需要位移超过 max_displacement_mm=%.2f（%s）—— 邻牙太远，"
                "不能把整颗冠拉过去" % (params.max_displacement_mm,
                                        "/".join(infeasible)))
            res.mesh = crown_mesh
            return res

        mod_idx = np.flatnonzero(modified)
        if len(mod_idx) == 0:
            res.status = STATUS_OK
            res.notes.append("两侧间隙已在容差内，未修改几何")
            res.mesh = crown_mesh
            _finalize_metrics(res, vertices, faces, patches, params,
                              patient_frame)
            return res

        crown = ~loop_set
        z_thr = np.percentile(vertices[crown, 2],
                              params.occlusal_anchor_percentile)
        occ = np.flatnonzero(crown & (vertices[:, 2] >= z_thr))
        # 咬合面 = 高 z 且法向朝轴（径向法向的高 z 顶点是邻接壁，可修）
        from .contact_refine import occ_table_mask
        occ_table = np.flatnonzero(occ_table_mask(vertices, faces,
                                                  patient_frame, params))
        if params.local_band:
            # **Contact v2**：修改区顶点硬设到目标位，只解修改区外侧的过渡环；
            # 颈环与咬合区是固定边界（位移由构造**精确为 0**）。
            # v1 的教训：注释写"都硬锁"，但 solve_laplacian 把咬合放进弱锚块
            # （α=300 对硬行 W=1e4 无效），真实例咬合被拖 1.08mm。
            from .laplacian import solve_local_band
            loop_set_b = np.zeros(len(vertices), dtype=bool)
            loop_set_b[loop_idx] = True
            locked = loop_set_b.copy()
            locked[occ_table] = True
            locked_idx = set(int(i) for i in np.flatnonzero(locked))
            seeds = set(int(i) for i in mod_idx) - locked_idx
            seen = set(seeds) | locked_idx
            free, cur = set(), set(seeds)
            for _ in range(params.transition_rings):
                nxt = set()
                for v_ in cur:
                    for w_ in adj.get(v_, ()):
                        if w_ not in seen:
                            nxt.add(w_)
                free |= nxt
                seen |= nxt
                cur = nxt
                if not cur:
                    break
            if free:
                free_ids = np.asarray(sorted(free), dtype=np.int64)
                fixed_ids = np.setdiff1d(np.arange(len(vertices)), free_ids)
                xt = vertices.copy()
                # **锁定区（颈环 ∪ 咬合面）的目标一律忽略**：落在锁定区的
                # 修改顶点本迭代不动（fail-safe）。否则咬合面顶点被硬推到
                # 目标位，"表面零位移"就是空话（实测真实例 1.12mm 教训）。
                movable = ~locked[mod_idx]
                ids_m = mod_idx[movable]
                xt[ids_m] = vertices[ids_m] + params.alpha * (
                    desired[ids_m] - vertices[ids_m])
                v2, _info = solve_local_band(vertices, faces, free_ids,
                                             fixed_ids, xt)
            else:
                v2 = vertices.copy()
                v2[mod_idx] = vertices[mod_idx] + params.alpha * (
                    desired[mod_idx] - vertices[mod_idx])
        else:
            v2, _info = solve_laplacian(DeformInputs(
                vertices_placed=vertices, faces=faces,
                margin_vertex_indices=mod_idx, margin_targets=desired[mod_idx],
                root_vertex_indices=loop_idx,          # 颈环 LOCK（恒等锚定）
                occlusal_vertex_indices=occ,
                config=laplacian_config or CrownOuterConfig(
                    weight_occlusal_anchor=CONTACT_OCCLUSAL_ANCHOR)))

        res.margin_max_displacement_mm = float(np.max(np.linalg.norm(
            v2[loop_idx] - vertices[loop_idx], axis=1)))
        # 咬合**表面**位移（必须为 0）；与"接触修正波及的高 z 邻接壁"分开报，
        # 否则修接触的动作会被误读成咬合拖动（v1 的 1.08mm 教训）。
        dt = (np.linalg.norm(v2[occ_table] - vertices[occ_table], axis=1)
              if len(occ_table) else np.array([0.0]))
        res.occlusal_p95_displacement_mm = float(np.percentile(dt, 95))
        modified_set = set(int(i) for i in mod_idx)
        occ_wall = np.asarray([i for i in occ if i not in modified_set],
                              dtype=np.int64) if len(occ) else np.array([],
                                                                       dtype=np.int64)
        dw = (np.linalg.norm(v2[occ_wall] - vertices[occ_wall], axis=1)
              if len(occ_wall) else np.array([0.0]))
        res.notes.append("occlusal_table_p95=%.4f occ_wall_drag_p95=%.4f"
                         % (float(np.percentile(dt, 95)),
                            float(np.percentile(dw, 95))))
        n0 = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
        n1 = np.cross(v2[faces[:, 1]] - v2[faces[:, 0]],
                      v2[faces[:, 2]] - v2[faces[:, 0]])
        den = np.linalg.norm(n0, axis=1) * np.linalg.norm(n1, axis=1)
        okd = den > 1e-18
        res.n_flipped_faces = int(np.sum(
            np.einsum("ij,ij->i", n0[okd], n1[okd]) / den[okd] < 0))

        import trimesh
        res.mesh = trimesh.Trimesh(vertices=v2, faces=faces, process=False)
        _finalize_metrics(res, v2, faces, patches, params, patient_frame)

        notes = []
        if res.margin_max_displacement_mm > 0.01:
            notes.append("margin 位移 %.4f 超过 0.01"
                         % res.margin_max_displacement_mm)
        if res.occlusal_p95_displacement_mm > 0.3:
            notes.append("咬合区位移 p95 %.3f 偏大"
                         % res.occlusal_p95_displacement_mm)
        for rep in (res.mesial, res.distal):
            if rep.penetration_mm > params.penetration_tolerance_mm:
                notes.append("%s 仍有穿透 %.3f" % (rep.side,
                                                   rep.penetration_mm))
        res.notes = notes
        res.status = STATUS_REVIEW_REQUIRED if notes else STATUS_OK
        return res
    except ContactError:
        raise
    except Exception as exc:                     # noqa: BLE001
        res.status = STATUS_FAILED
        res.notes = res.notes + [f"{type(exc).__name__}: {exc}"]
        raise


def _finalize_metrics(res, vertices, faces, patches, params, frame):
    """变形后逐侧指标：min_distance / penetration / contact area / centroid。

    每侧只度量**该侧半边**的顶点：对侧顶点到本侧 patch 的带符号距离
    必然是大负值，会伪造"穿透"（实测 −9.5mm 假报警）。"""
    tri_area = 0.5 * np.linalg.norm(
        np.cross(  # 面积加权：网格总面积/顶点数 = 每顶点当量面积
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=1).sum()
    apv = float(tri_area) / max(len(vertices), 1)
    for side, patch in patches.items():
        rep = getattr(res, side)
        sign = 1.0 if side == "mesial" else -1.0
        u_lat = (vertices - frame.origin) @ frame.x
        sel = sign * u_lat >= -0.5
        # **统一口径**（决策 ③）：幅值 euclid + 边界/crease 有效性
        signed, valid, cp, fids, _parts = patch.clearance_detail(vertices[sel])
        dist = np.abs(np.asarray(signed, np.float64))
        valid = np.asarray(valid, bool)
        rep.min_distance_mm = float(np.min(dist))
        inside = (signed < 0) & (dist <= params.inside_reach_mm) & valid
        rep.penetration_mm = float(max(
            0.0, -np.min(signed[inside]) - params.penetration_tolerance_mm)
            ) if inside.any() else 0.0
        # 接触带 = 间隙落在 [target−contact_mm, target+contact_mm] 的顶点
        # （目标是留有 target_gap 的邻接，不是 0 距离贴合）
        gap = np.abs(signed)
        near = np.abs(gap - params.target_gap_mm) <= params.contact_mm
        rep.contact_area_mm2 = float(near.sum()) * apv
        if near.any():
            rep.contact_centroid = vertices[sel][near].mean(axis=0)
        else:
            rep.contact_centroid = None
