"""Thickness v1 —— 壁厚**审计 + 冲突归因**（只测不修）。

用户 2026-10-02 规格：第一提交**不做修复**，只回答三件事：

    1. 这颗冠到底哪里薄、薄多少（分区账，不是一个全冠 min）；
    2. 薄的地方**能不能修**（自由度 vs Contact/Occlusion 硬约束）；
    3. Occlusion v1 削掉 1.5mm 咬合干涉后，有没有把咬合厚度削穿安全线。

## 为什么不是"薄了就往外推"

此刻已经有三个互相竞争的硬约束：Intaglio/Fit 不能动、Contact 不能重新撞邻牙、
Occlusion 不能重新撞对颌。所以先归因（A/B/C/D 四类），再谈修：

| 类 | 语义 | v1 处置 |
|---|---|---|
| ``REPAIRABLE_THIN`` | 薄区在自由外表面，往外补**不碰**任何约束 | 留给 Repair 提交 |
| ``CONTACT_THICKNESS_CONFLICT`` | 往外补会撞邻牙 | REVIEW |
| ``OCCLUSION_THICKNESS_CONFLICT`` | 往外补会撞对颌 | REVIEW（交给联合优化） |
| ``CERVICAL_THIN`` | 颈缘带内 | 按近颈缘规则单独判定 |

归因判据是**几何 clearance** 而不是区域标签：把外表面点沿其外法向推 ``deficit``
之后，检验它与邻牙/对颌 patch 的带符号距离是否仍 ≥ 目标间隙 + 余量。

## 两个厚度测量

* ``t_closest`` —— 最近表面距离（**与法向无关**，不受翻面影响）；
* ``t_ray``     —— 沿内表面外法向的**射线穿透厚度**（真正的"壁"）。

两者差 > ``max(ambiguous_abs, ambiguous_rel·t_closest)`` → ``AMBIGUOUS_THICKNESS``
（凹坑/牙尖侧壁上"最近点"并不代表真正材料厚度）。薄判定用 ``t_ray``（有效时），
射线无解时回退 ``t_closest`` 并记 ``FALLBACK_CLOSEST``。

## 输入 QC —— Occlusion v1 留了 2.08% 翻面，必须先量出来

``outer`` 的绕向一致性 / 非流形边 / 边界边 / 自交 /（给参考网格时）翻面率。
**审计不"顺便"掩盖翻面问题**：这些数原样进报告；``t_closest`` 不依赖法向。

## 阈值 provenance

``min_thickness_mm`` 等**必须由调用方给出**（来自病例设计参数 / 材料 / 加工规则）。
引擎内不写任何临床默认值。真实例 cf_b1b73bae221211b1 的 ``.modelInfo``：
``MinThickness=0.6`` / ``MinThicknessNearPreparationMargin=0.2`` /
``MinThicknessFullDistanceToPreparationMargin=0.8`` / ``CementGapThickness=0.08``。

## 内表面从哪来

CrownOuter 链路目前只出外表面（开放壳，颈环 = 边界）。内表面 =
``intaglio_from_prep(prep, gap)``：代型表面沿其外法向偏移 cement gap。
代型获取见 ``tools/stage_prep_case.py`` + ``tools/align_prep_to_design.py``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .contact import NeighborPatch
from .gt_compare import anatomical_regions
from .patient_frame import PatientFrame

STATUS_PASS = "PASS"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"

THIN_REPAIRABLE = "REPAIRABLE_THIN"
THIN_CONTACT = "CONTACT_THICKNESS_CONFLICT"
THIN_OCCLUSION = "OCCLUSION_THICKNESS_CONFLICT"
THIN_CERVICAL = "CERVICAL_THIN"

#: 区域表行（词表与 gt_compare.anatomical_regions 一致）
REGION_ROWS = ("occlusal", "mesial", "distal", "buccal", "lingual",
               "cervical", "lateral_unknown")

#: 射线状态
RAY_HIT = 0
RAY_NO_HIT = 1
RAY_STARTS_OUTSIDE = 2


class ThicknessError(Exception):
    pass


@dataclass
class ThicknessParameters:
    """全部阈值来自调用方；引擎不写临床默认。"""

    #: 设计最小壁厚（mm）。**必填**。
    min_thickness_mm: float
    #: 颈缘附近放宽（设计参数）。None = 不启用（只按 min_thickness 判）。
    near_margin_thickness_mm: float | None = None
    #: 放宽生效距离（mm）：d ≥ 此值用 min_thickness。
    near_margin_distance_mm: float | None = None
    #: ``linear``（默认读法）| ``step``。两种读法都会在报告里给出对照。
    near_margin_mode: str = "linear"
    #: 内表面 = 代型 + 此 gap（provenance，不参与几何）。
    cement_gap_mm: float = 0.0
    #: 区域划分
    cervical_band_mm: float = 1.0
    occlusal_fraction: float = 0.65
    #: 到外表面开放边界的距离小于此值的最近点不可靠（颈缘刀口带）
    boundary_safety_mm: float = 0.6
    #: 射线步进 / 上限 / 二分
    ray_step_mm: float = 0.1
    ray_max_mm: float = 4.0
    ray_bisect_iters: int = 12
    #: 双测法一致性
    ambiguous_abs_mm: float = 0.15
    ambiguous_rel: float = 0.30
    #: 内/外判定：绕数阈值（开放壳的边界带绕数 → 0.5）
    outside_winding_threshold: float = 0.5
    winding_band: float = 0.15
    #: 颈缘刀口带（d_margin 小于此值不参与薄判定，单独计数）
    knife_edge_mm: float = 0.10
    #: 冲突判定：推进 deficit 后仍需保留的余量（mm）
    clearance_margin_mm: float = 0.02
    #: 最近点候选数（质心/顶点采样 kNN）
    k_candidates: int = 24
    #: 查询网格细分的最大边长（mm）；0 = 不细分（会显著低估精度）
    max_query_edge_mm: float = 0.35
    #: 自交 QC：每个三角形取 k 个质心近邻当候选对
    si_neighbor_k: int = 8
    si_max_pairs: int = 2_000_000
    #: 输入 QC 门
    max_flipped_ratio: float | None = None
    max_nonmanifold_edges: int = 0
    max_self_intersections: int | None = 0


@dataclass
class ThicknessResult:
    status: str = STATUS_FAILED
    input_qc: dict = field(default_factory=dict)
    #: 分区账：{n, min, p05, median, p95, thin_area_mm2, worst_deficit_mm, …}
    regions: dict = field(default_factory=dict)
    #: 薄区归因合计（面积 mm²）
    thin_area: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    #: 逐顶点原始量（heatmap 用；不进 JSON）
    thickness_map: dict = field(default_factory=dict)
    #: B1：本域测量契约标签（域 / 符号来源(平滑法向) / 有效性 / 冲突含义 / 负责相交审计）
    domain_report: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def summary_lines(self) -> str:
        s = self.summary
        out = ["状态: %s" % self.status]
        for k in ("n_samples", "n_evaluated", "min_thickness_mm",
                  "p05_thickness_mm", "median_thickness_mm",
                  "thin_vertex_count", "thin_area_mm2", "max_deficit_mm",
                  "p95_deficit_mm"):
            if k in s:
                out.append("  %-22s %s" % (k, s[k]))
        return "\n".join(out)


# --------------------------------------------------------------------------
# point-triangle 最近点（向量化；退化三角形回退到边界）
# --------------------------------------------------------------------------

def _seg_nearest(p, a, b):
    """点到线段最近点（向量化，零长线段回退到端点 a）。"""
    ab = b - a
    l2 = np.einsum("...i,...i->...", ab, ab)
    denom = np.where(l2 > 1e-18, l2, 1.0)
    t = np.einsum("...i,...i->...", p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    out = a + t[..., None] * ab
    return np.where((l2 <= 1e-18)[..., None], a, out)


def _point_tri_closest(p, a, b, c):
    """点到三角形最近点（向量化，(...,3) 广播）—— **Ericson §5.1.5** 逐区选择。

    区域判定的**先后顺序即优先级**（``np.select`` 取第一个为真的分支），这正是
    Ericson 原算法的语义。⚠️ 曾用"重心坐标三式全非负"的等价写法：
    边函数取错（``cross(ac, ap)`` 而非 ``cross(bc, bp)``），7% 的三角形会返回
    三角形**外**的投影点 → 距离偏小（实测最大偏 24mm）。所以本函数与
    ``contact.closest_point_on_triangle``（同一算法）及 trimesh 逐点对过账。
    """
    p, a, b, c = np.broadcast_arrays(p, a, b, c)
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("...i,...i->...", ab, ap)
    d2 = np.einsum("...i,...i->...", ac, ap)
    bp = p - b
    d3 = np.einsum("...i,...i->...", ab, bp)
    d4 = np.einsum("...i,...i->...", ac, bp)
    cp_ = p - c
    d5 = np.einsum("...i,...i->...", ab, cp_)
    d6 = np.einsum("...i,...i->...", ac, cp_)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) > 1e-300, denom, 1.0)
    v = vb / safe
    w = vc / safe
    p_ab = a + (d1 / np.where(np.abs(d1 - d3) > 1e-300, d1 - d3, 1.0)) \
        [..., None] * ab
    p_ac = a + (d2 / np.where(np.abs(d2 - d6) > 1e-300, d2 - d6, 1.0)) \
        [..., None] * ac
    den_bc = (d4 - d3) + (d5 - d6)
    p_bc = b + (((d4 - d3) / np.where(np.abs(den_bc) > 1e-300, den_bc, 1.0))
                [..., None]) * (c - b)
    p_in = a + ab * v[..., None] + ac * w[..., None]
    p_in = np.where(np.isfinite(p_in).all(axis=-1, keepdims=True), p_in, a)
    conds = [
        (d1 <= 0) & (d2 <= 0),                                    # 顶点 A
        (d3 >= 0) & (d4 <= d3),                                   # 顶点 B
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),                        # 边 AB
        (d6 >= 0) & (d5 <= d6),                                   # 顶点 C
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),                        # 边 AC
        (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),          # 边 BC
    ]
    choices = [np.broadcast_to(a, p.shape), np.broadcast_to(b, p.shape),
               p_ab, np.broadcast_to(c, p.shape), p_ac, p_bc, p_in]
    sel = np.select(conds, np.arange(6), default=6)
    cands = np.stack(choices, axis=-2)                       # (...,7,3)
    res = np.take_along_axis(cands, sel[..., None, None],
                             axis=-2)[..., 0, :]
    # 完全退化（denom≈0）且所有区域判据都没命中时，落到三条边（永远正确）
    deg = (np.abs(denom) <= 1e-300) & (sel == 6)
    if deg.any():
        e1 = _seg_nearest(p, a, b)
        e2 = _seg_nearest(p, b, c)
        e3 = _seg_nearest(p, c, a)
        cand = np.stack([e1, e2, e3], axis=-2)
        dd = np.linalg.norm(cand - p[..., None, :], axis=-1)
        best = np.take_along_axis(
            cand, np.argmin(dd, axis=-1)[..., None, None], axis=-2)[..., 0, :]
        res = np.where(deg[..., None], best, res)
    return res


class SurfaceQuery:
    """三角网格点-最近点查询（**全向量化**），语义对齐 ``NeighborPatch.closest``。

    候选面 = 质心 KD-tree 的 k 近邻（细密网格上足够；等价性由单测对
    ``NeighborPatch`` 逐点验收）。

    ``signed`` 用**平滑法向**（最近面三顶点的面积加权法向均值）而不是单面法向：
    Occlusion v1 留了 2.08% 翻面，单面法向会让翻面处的符号整体翻错（实测 8301
    点里 244 点属此类）。单面法向与平滑法向**反向**的点额外返回 ``flipped``
    ——翻面问题如实计数，不掩盖。
    """

    def __init__(self, vertices, faces, *, boundary_safety_mm: float = 0.5,
                 k_candidates: int = 24, max_query_edge_mm: float = 0.35):
        v = np.asarray(vertices, dtype=np.float64)
        f = np.asarray(faces, dtype=np.int64)
        n_faces_in = len(f)
        if max_query_edge_mm and max_query_edge_mm > 0:
            v, f = refine_max_edge(v, f, max_query_edge_mm)
        self.n_faces_input = n_faces_in
        self.n_faces_query = len(f)
        self.vertices = v
        #: 原始（未细分）三角形 —— 精确射线求交用（面更少，且就是真实曲面）
        self.tri_original = np.asarray(
            np.asarray(vertices, dtype=np.float64)[np.asarray(faces,
                                                              dtype=np.int64)],
            dtype=np.float64)
        tri = self.vertices[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        keep = ln > 1e-14
        self.faces = f[keep]
        self.tri = tri[keep]
        self.normals = n[keep] / ln[keep, None]
        self.vnormals = vertex_normals(self.vertices, self.faces)
        self.centroids = self.tri.mean(axis=1)
        self.k = int(k_candidates)
        # 采样点 = 面质心 + 顶点。**只用质心是不够的**：本网格面尺寸不均，
        # 真最近面的质心可以排到第 127 近（实测），kNN 必然漏。顶点采样的
        # 命中再展开成关联面 —— 与 ``NeighborPatch`` 同一套做法，并配合
        # ``refine_max_edge`` 把面尺寸差异压到采样间距以下。
        samples = np.vstack([self.centroids, self.vertices])
        self._sample_face = np.concatenate([
            np.arange(len(self.centroids), dtype=np.int64),
            np.full(len(self.vertices), -1, dtype=np.int64)])
        self._sample_vertex = np.concatenate([
            np.full(len(self.centroids), -1, dtype=np.int64),
            np.arange(len(self.vertices), dtype=np.int64)])
        self._tree = cKDTree(samples)
        self._centroid_tree = cKDTree(self.centroids)
        vid2face: dict[int, list[int]] = {}
        for fid, (a, b, c) in enumerate(self.faces):
            for v in (int(a), int(b), int(c)):
                vid2face.setdefault(v, []).append(fid)
        self.max_inc = max((len(v) for v in vid2face.values()), default=1)
        inc = np.full((len(self.vertices), self.max_inc), -1, dtype=np.int64)
        for v, fs in vid2face.items():
            inc[v, :len(fs)] = fs
        self._vid2face_inc = inc
        self.normal_sign = 1.0
        self.boundary_safety_mm = float(boundary_safety_mm)
        self.boundary_edges = _boundary_edges(self.faces)
        self.n_boundary_edges = len(self.boundary_edges)
        pts = []
        for u, v in self.boundary_edges:
            pu, pv = self.vertices[u], self.vertices[v]
            ns = max(2, int(float(np.linalg.norm(pv - pu)) / 0.2) + 1)
            for t in np.linspace(0.0, 1.0, ns):
                pts.append(pu + t * (pv - pu))
        self._boundary_tree = cKDTree(np.asarray(pts)) if pts else None

    def _narrow(self, points, idx):
        """给定候选面 (N,K)，返回 (最近点, 面 id, 距离)。idx<0 视为无效。"""
        idx = np.where(idx < 0, 0, idx)
        tri = self.tri[idx]
        cp = _point_tri_closest(points[:, None, :], tri[..., 0, :],
                                tri[..., 1, :], tri[..., 2, :])
        dd = np.linalg.norm(cp - points[:, None, :], axis=-1)
        j = np.argmin(dd, axis=1)
        rows = np.arange(len(points))
        return cp[rows, j], idx[rows, j], dd[rows, j]

    def update_vertices(self, vertices):
        """**拓扑不变、只挪顶点**时刷新查询结构（逐迭代测距必需）。

        重新建 SurfaceQuery 每次都要跑 Python 级的边界遍历；迭代里每个候选位形
        都要测一次距离，建对象会成为瓶颈（实测真实例每步 3~5s）。这里只重算
        向量化部分 + 边界采样（边界边通常 ~150 条）。
        """
        v = np.asarray(vertices, dtype=np.float64)
        if len(v) != len(self.vertices):
            raise ThicknessError("update_vertices 顶点数必须不变（拓扑不改）")
        self.vertices = v
        self.tri = v[self.faces]
        n = np.cross(self.tri[:, 1] - self.tri[:, 0],
                     self.tri[:, 2] - self.tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        self.normals = n / np.maximum(ln, 1e-14)[:, None]
        self.vnormals = vertex_normals(v, self.faces)
        self.centroids = self.tri.mean(axis=1)
        self._tree = cKDTree(np.vstack([self.centroids, v]))
        pts = []
        for u, w in self.boundary_edges:
            pu, pv = v[u], v[w]
            ns = max(2, int(float(np.linalg.norm(pv - pu)) / 0.2) + 1)
            for t in np.linspace(0.0, 1.0, ns):
                pts.append(pu + t * (pv - pu))
        self._boundary_tree = cKDTree(np.asarray(pts)) if pts else None
        return self

    def closest_normal(self, fids):
        """最近面处的**平滑单位法向**（面积加权顶点法向在三点上的均值）。"""
        fids = np.asarray(fids, dtype=np.int64)
        vn = self.vnormals[self.faces[fids]].sum(axis=1)
        ln = np.linalg.norm(vn, axis=1)
        return vn / np.maximum(ln, 1e-18)[:, None] * self.normal_sign

    def query(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        k = min(self.k, len(self._sample_face))
        _d, sidx = self._tree.query(points, k=k)
        sidx = np.atleast_2d(sidx)
        n_pts = len(points)
        # 质心命中 → 该面；顶点命中 → 其关联面（稠密展开，−1 位无害）
        fids = self._sample_face[sidx]                       # (N,k)
        vids = self._sample_vertex[sidx]
        inc = np.where(vids[:, :, None] >= 0,
                       self._vid2face_inc[np.maximum(vids, 0)], -1)
        cand = np.concatenate([fids, inc.reshape(n_pts, -1)], axis=1)
        best_p, best_f, best_d = self._narrow(points, cand)
        # 兜底：若某行的 k 近邻**全是**"无关联面"的顶点样本（查询点在网格
        # 内部/很远时会发生，如拿子面片当查询网格），候选会全是 −1 填充。
        # 这时退回纯质心 kNN（质心永远对应有效面）。
        dead = (cand < 0).all(axis=1)
        if dead.any():
            _d2, cidx = self._centroid_tree.query(points[dead], k=min(8, len(self.centroids)))
            p2, f2, d2 = self._narrow(points[dead], np.atleast_2d(cidx))
            rows = np.flatnonzero(dead)
            best_p[rows], best_f[rows], best_d[rows] = p2, f2, d2
        vn = self.vnormals[self.faces[best_f]].sum(axis=1)
        ln = np.linalg.norm(vn, axis=1)
        vn = vn / np.maximum(ln, 1e-18)[:, None]
        signed = np.einsum("ij,ij->i", points - best_p, vn) \
            * self.normal_sign
        flipped = np.einsum("ij,ij->i", self.normals[best_f], vn) < 0.0
        if self._boundary_tree is None:
            valid = np.ones(n_pts, dtype=bool)
        else:
            valid = self._boundary_tree.query(best_p, k=1)[0] >= \
                self.boundary_safety_mm
        return best_p, best_f, signed, best_d, valid, flipped


def _boundary_edges(faces):
    from collections import Counter
    cnt: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            cnt[(min(u, v), max(u, v))] += 1
    return np.asarray([e for e, m in cnt.items() if m == 1], dtype=np.int64)


def winding_number(points, tri, *, chunk: int = 400) -> np.ndarray:
    """广义绕数（Van Oosterom & Strackee）：≈1 在闭合壳内，≈0 在外。

    **为什么不用"最近面法向"判内外**：Occlusion v1 留下 628 个翻面 + 376 处自交，
    折叠区里最近面的法向会把"其实在里面"的点判成外面（实测 1128 vs 746：有 450
    个点被法向判错，它们的绕数 0.65~0.90 明明在里面）。绕数只用**有向立体角**
    求和，对翻面/自交/开放边界都稳健；本实现与"封底 + 射线奇偶"独立法一致
    （746 vs 749）。
    """
    points = np.asarray(points, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.float64)
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        p = points[i:i + chunk]
        a = tri[None, :, 0, :] - p[:, None, :]
        b = tri[None, :, 1, :] - p[:, None, :]
        c = tri[None, :, 2, :] - p[:, None, :]
        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)
        num = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc
               + np.einsum("ijk,ijk->ij", a, b) * lc
               + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", a, c) * lb)
        out[i:i + chunk] = 2.0 * np.arctan2(num, den).sum(axis=1) / (4.0 * np.pi)
    return out


def ray_first_hit(origins, directions, tri, *, max_t: float = 4.0,
                  chunk: int = 64):
    """沿方向求与三角网格的**首个交点**（Möller–Trumbore，向量化分块）。

    返回 ``(t, face_id)``；未命中 t=inf、face_id=-1。**与面法向无关** —— 翻面
    不会让射线提前"穿出"（早期用带符号距离场步进，在翻面处会当场终止：实测
    咬合区一半采样点的 t_ray 被这种伪终止污染）。
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    tri = np.asarray(tri, dtype=np.float64)
    if tri.ndim == 2:                       # 单个三角形 (3,3) → (1,3,3)
        tri = tri[None, :, :]
    origins = np.atleast_2d(origins)
    directions = np.atleast_2d(directions)
    n = len(origins)
    t_out = np.full(n, np.inf)
    f_out = np.full(n, -1, dtype=np.int64)
    e1 = tri[:, 1, :] - tri[:, 0, :]
    e2 = tri[:, 2, :] - tri[:, 0, :]
    for i in range(0, n, chunk):
        o = origins[i:i + chunk]
        d = directions[i:i + chunk]
        h = np.cross(d[:, None, :], e2[None, :, :])
        a = np.einsum("ijk,ijk->ij", e1[None, :, :], h)
        ok = np.abs(a) > 1e-12
        f = 1.0 / np.where(ok, a, 1.0)
        s = o[:, None, :] - tri[None, :, 0, :]
        u = f * np.einsum("ijk,ijk->ij", s, h)
        q = np.cross(s, e1[None, :, :])
        v = f * np.einsum("ijk,ijk->ij", d[:, None, :], q)
        t = f * np.einsum("ijk,ijk->ij", e2[None, :, :], q)
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) \
            & (t > 1e-6) & (t <= max_t)
        tt = np.where(hit, t, np.inf)
        j = np.argmin(tt, axis=1)
        best = tt[np.arange(len(o)), j]
        t_out[i:i + chunk] = best
        f_out[i:i + chunk] = np.where(np.isfinite(best), j, -1)
    return t_out, f_out


def max_edge_lengths(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    return np.stack([
        np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1),
        np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1),
        np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)]).max(0)


def refine_max_edge(vertices, faces, max_edge_mm: float, *,
                    max_rounds: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """把过长边做 **red-green 保形细分**（几何不变，只在边上插中点）。

    为什么必须做：查询法（采样点 KD-tree + 候选面投影）的精度依赖"任意面内点到
    最近采样点"的距离上界。本网格面尺寸极不均（实测 occluded：最长边 p50=0.25mm、
    max=3.5mm，23% 的面 >0.4mm），大面会让真最近面的采样点排到 kNN 之外 ——
    对 NeighborPatch 与本实现都会给出 0.28mm vs 真值 0.13mm 的错误距离。

    ⚠️ 必须**保形**：只按"面内三点"做 1→4 会在与未分裂邻居共享的边上留下
    T-junction，那些边会被当成开放边界（实测把 93% 的采样点判成"边界带不可信"）。
    这里逐边标记 → 每面按"哪几条边被标记"分裂（1 边→2 面，2 边→3 面，3 边→4 面），
    共享边两侧用同一个中点 id，保形。
    """
    vlist = [tuple(p) for p in np.asarray(vertices, dtype=np.float64)]
    f = np.asarray(faces, dtype=np.int64).copy()
    cache: dict[tuple[int, int], int] = {}

    def mid(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        got = cache.get(key)
        if got is None:
            got = len(vlist)
            va = np.asarray(vlist[key[0]], dtype=np.float64)
            vb = np.asarray(vlist[key[1]], dtype=np.float64)
            vlist.append(tuple(0.5 * (va + vb)))
            cache[key] = got
        return got

    def total_area(vv, ff):
        t = vv[ff]
        return 0.5 * np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum()

    def normal_sum(vv, ff):
        """面积加权法向和 —— 平面细分下**逐位守恒**。"""
        t = vv[ff]
        return np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]).sum(axis=0)

    a0 = total_area(np.asarray(vlist), f)
    ns0 = normal_sum(np.asarray(vlist), f)
    for _ in range(max_rounds):
        v = np.asarray(vlist, dtype=np.float64)
        if len(f) == 0:
            break
        ab = np.linalg.norm(v[f[:, 1]] - v[f[:, 0]], axis=1)
        bc = np.linalg.norm(v[f[:, 2]] - v[f[:, 1]], axis=1)
        ca = np.linalg.norm(v[f[:, 0]] - v[f[:, 2]], axis=1)
        marked = np.stack([ab > max_edge_mm, bc > max_edge_mm,
                           ca > max_edge_mm], axis=1)
        if not marked.any():
            break
        out = list(f[~marked.any(axis=1)])
        for fi in np.flatnonzero(marked.any(axis=1)):
            a, b, c = int(f[fi, 0]), int(f[fi, 1]), int(f[fi, 2])
            m0, m1, m2 = marked[fi]
            if m0 and m1 and m2:
                mab, mbc, mca = mid(a, b), mid(b, c), mid(c, a)
                out.extend([(a, mab, mca), (mab, b, mbc), (mca, mbc, c),
                            (mab, mbc, mca)])
            elif m0 and m1:                       # 切 b 角 + 四边形 (mab,mbc,c,a)
                mab, mbc = mid(a, b), mid(b, c)
                out.extend([(a, mab, mbc), (mab, b, mbc), (a, mbc, c)])
            elif m0 and m2:                       # 切 a 角 + 四边形 (mab,b,c,mca)
                mab, mca = mid(a, b), mid(c, a)
                out.extend([(a, mab, mca), (mab, b, c), (mab, c, mca)])
            elif m1 and m2:                       # 切 c 角 + 四边形 (a,b,mbc,mca)
                mbc, mca = mid(b, c), mid(c, a)
                out.extend([(a, b, mbc), (a, mbc, mca), (mbc, c, mca)])
            elif m0:
                mab = mid(a, b)
                out.extend([(a, mab, c), (mab, b, c)])
            elif m1:
                mbc = mid(b, c)
                out.extend([(b, mbc, a), (mbc, c, a)])
            else:
                mca = mid(c, a)
                out.extend([(c, mca, b), (mca, a, b)])
        f = np.asarray(out, dtype=np.int64)
        a1 = total_area(np.asarray(vlist), f)
        if abs(a1 - a0) > 1e-9 * max(a0, 1.0):
            raise ThicknessError(
                "细分破坏了面积守恒（%.6f → %.6f）—— 分裂组合写错了"
                % (a0, a1))
        ns1 = normal_sum(np.asarray(vlist), f)
        # **必须同时守绕向**：分裂组合里只要有一个子三角形绕反，面积照样守恒
        # （实测 (0,1,1) 那组分错 → 37.6% 的子面翻向、顶点法向被抵消成切向），
        # 而**面积**这一条看不出来。平面细分下面积加权法向和逐位守恒。
        if not np.allclose(ns1, ns0, rtol=0, atol=1e-9 * max(np.abs(ns0).max(),
                                                            1.0)):
            raise ThicknessError(
                "细分破坏了绕向（Σ面法向 %s → %s）—— 分裂组合把子三角形绕反了"
                % (np.round(ns0, 4).tolist(), np.round(ns1, 4).tolist()))
    return np.asarray(vlist, dtype=np.float64), f


def vertex_normals(vertices, faces):
    """面积加权顶点法向。"""
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], fn)
    ln = np.linalg.norm(out, axis=1)
    ln[ln < 1e-18] = 1.0
    return out / ln[:, None]


# --------------------------------------------------------------------------
# 内表面：代型 + cement gap
# --------------------------------------------------------------------------

def intaglio_from_prep(prep_vertices, prep_faces, cement_gap_mm: float,
                       *, axis=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """内表面 = 代型表面沿其外法向偏移 cement gap。

    法向定向检查：轴向壁上 ``n·radial`` 的中位数应为正（背离牙体）。整体反向时
    翻号并在返回的 info 里如实记录。
    """
    v = np.asarray(prep_vertices, dtype=np.float64)
    f = np.asarray(prep_faces, dtype=np.int64)
    n = vertex_normals(v, f)
    info = {"normal_flipped": False, "axial_wall_median_dot": None}
    if axis is not None:
        z = np.asarray(axis, dtype=np.float64)
        z = z / np.linalg.norm(z)
        rel = v - v.mean(axis=0)
        radial = rel - np.outer(rel @ z, z)
        rn = np.linalg.norm(radial, axis=1)
        wall = (rn > 1e-6) & (np.abs(n @ z) < 0.5)
        if int(wall.sum()) > 10:
            dot = np.einsum("ij,ij->i", n[wall], radial[wall] / rn[wall, None])
            med = float(np.median(dot))
            info["axial_wall_median_dot"] = med
            if med < 0:
                n = -n
                info["normal_flipped"] = True
    return v + float(cement_gap_mm) * n, f, info


# --------------------------------------------------------------------------
# 输入 QC
# --------------------------------------------------------------------------

def input_qc(vertices, faces, *, reference_vertices=None,
             si_neighbor_k: int = 8, si_max_pairs: int = 2_000_000) -> dict:
    """绕向一致性 / 非流形边 / 边界边 / 自交 / 相对参考的翻面率。"""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    cnt, ori = _edge_table(f)
    n_boundary = sum(1 for m in cnt.values() if m == 1)
    n_nonmanifold = sum(1 for m in cnt.values() if m > 2)
    inconsistent = 0
    for e, m in cnt.items():
        if m == 2 and (ori.get((e, True), 0) != 1 or ori.get((e, False), 0) != 1):
            inconsistent += 1
    tri = v[f]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(fn, axis=1)
    out = {
        "n_vertices": int(len(v)), "n_faces": int(len(f)),
        "boundary_edges": int(n_boundary),
        "nonmanifold_edges": int(n_nonmanifold),
        "winding_inconsistent_edges": int(inconsistent),
        "degenerate_faces": int(np.sum(area2 <= 1e-14)),
    }
    if reference_vertices is None:
        out["flipped_ratio"] = None
        out["flipped_note"] = "未提供参考网格（Occlusion 前），翻面率未测"
    else:
        rv = np.asarray(reference_vertices, dtype=np.float64)
        if len(rv) != len(v):
            out["flipped_ratio"] = None
            out["flipped_note"] = "参考网格顶点数不同，翻面率不可比"
        else:
            rt = rv[f]
            rn = np.cross(rt[:, 1] - rt[:, 0], rt[:, 2] - rt[:, 0])
            den = np.linalg.norm(rn, axis=1) * area2
            ok = den > 1e-18
            cos = np.ones(len(f))
            cos[ok] = np.einsum("ij,ij->i", rn[ok], fn[ok]) / den[ok]
            n_flip = int(np.sum(cos < 0))
            out["flipped_faces_vs_reference"] = n_flip
            out["flipped_ratio"] = n_flip / max(1, len(f))
            # **面积加权**才是"折叠"：一个大面翻向 = 把表面折过去；数值级微面
            # 翻向只是退化（面积→0）。计数口径会把后者当缺陷 —— J.1 的大变形
            # 线搜索在 α=0.0625 时就有 1 个微面翻向，计数门因此拒掉每一步。
            fa = 0.5 * area2
            tot = float(fa.sum())
            out["flipped_area_mm2"] = float(fa[cos < 0].sum())
            out["flipped_area_ratio"] = (float(fa[cos < 0].sum()) / tot
                                         if tot > 0 else 0.0)
    out.update(self_intersections(v, f, neighbor_k=si_neighbor_k,
                                  max_pairs=si_max_pairs))
    return out


def _edge_table(faces):
    from collections import Counter
    cnt: Counter = Counter()
    ori: Counter = Counter()
    for a, b, c in faces:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (min(u, v), max(u, v))
            cnt[key] += 1
            ori[(key, u < v)] += 1
    return cnt, ori


def self_intersections(vertices, faces, *, neighbor_k: int = 8,
                       max_pairs: int = 2_000_000,
                       severe_min_area_mm2: float | None = None,
                       severe_max_normal_dot: float | None = None,
                       focus_faces=None, return_pairs: bool = False) -> dict:
    """自交检测：质心 kNN 候选对（非相邻）+ 精确三角形相交。

    候选对生成是**近邻近似**（自交必然发生在空间近邻之间），测试本身精确；
    候选数超上限时如实报告截断，不假装测全。
    """
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces,
                                                            dtype=np.int64)]
    n = len(tri)
    # 顶点未焊接的 triangle soup：近邻法会给出无意义的成对结果 → 跳过并说明
    from collections import Counter
    ec: Counter = Counter()
    for t in tri.reshape(-1, 3):
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        for u, v in ((a, b), (b, c), (c, a)):
            ec[(min(u, v), max(u, v))] += 1
    if ec and sum(1 for m in ec.values() if m == 1) > 0.9 * len(ec):
        return {"self_intersections": None,
                "self_intersections_truncated": True,
                "self_intersections_note": "网格顶点未焊接，自交检测跳过"}
    centroids = tri.mean(axis=1)
    k = min(neighbor_k + 1, n)
    if focus_faces is not None:
        # **聚焦模式**（形变步的几何门用）：只找"至少牵涉一个关注面"的候选对。
        # 依据：形变只动一小片顶点，**新增**自交必然牵涉被动过的面；同一片区域
        # 在形变前若有（baseline），两次都跑一遍再相减就能分开"新"与"旧"。
        # 代价从 O(全网格) 降到 O(关注面 × k)，这是让"每步都过硬门"可行的关键。
        fsel = np.flatnonzero(np.asarray(focus_faces, dtype=bool))
        if len(fsel) == 0:
            return {"self_intersections": 0,
                    "self_intersections_truncated": False, "pair_tests": 0,
                    "self_intersections_focused": True,
                    "self_intersections_note": "关注面为空"}
        _d, idx = cKDTree(centroids).query(centroids[fsel], k=k)
        idx = np.atleast_2d(idx)
        i = np.repeat(fsel, idx.shape[1])
        j = idx.ravel()
        m = i != j
        i, j = i[m], j[m]
        i, j = np.minimum(i, j), np.maximum(i, j)
        pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    else:
        _d, idx = cKDTree(centroids).query(centroids, k=k)
        idx = np.atleast_2d(idx)
        i = np.repeat(np.arange(n), idx.shape[1])
        j = idx.ravel()
        m = i < j                      # 去重 + 去自对
        i, j = i[m], j[m]
        pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    if len(pairs) > max_pairs:
        return {"self_intersections": None,
                "self_intersections_truncated": True,
                "self_intersections_note":
                    "候选对 %d 超上限 %d" % (len(pairs), max_pairs),
                "pair_tests": int(len(pairs))}
    keep = ~_share_any_vertex(np.asarray(faces)[pairs[:, 0]],
                              np.asarray(faces)[pairs[:, 1]])
    pairs = pairs[keep]
    hit, n_parallel, seglen = _tri_tri_intersect(tri[pairs[:, 0]],
                                                 tri[pairs[:, 1]],
                                                 return_segment=True)
    # **证据**：单靠计数会把"两片曲面几乎相切地贴在一起"也算成自交。这里给出
    # 命中对的面积与法向夹角，便于区分"真交叉"与"数值级相切"（外挂核如
    # manifold3d 的输出在数值上可能相切相叠；我们不该凭一个计数就否掉它）。
    hp = pairs[hit]
    diag = []
    if len(hp):
        t1 = tri[hp[:, 0]]
        t2 = tri[hp[:, 1]]
        n1 = np.cross(t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 0])
        n2 = np.cross(t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0])
        a1 = 0.5 * np.linalg.norm(n1, axis=1)
        a2 = 0.5 * np.linalg.norm(n2, axis=1)
        u1 = n1 / np.maximum(np.linalg.norm(n1, axis=1), 1e-18)[:, None]
        u2 = n2 / np.maximum(np.linalg.norm(n2, axis=1), 1e-18)[:, None]
        dot = np.abs(np.einsum("ij,ij->i", u1, u2))
        gap = np.linalg.norm(t1.mean(axis=1) - t2.mean(axis=1), axis=1)
        for k in range(min(5, len(hp))):
            diag.append({"faces": [int(hp[k, 0]), int(hp[k, 1])],
                         "area_mm2": [round(float(a1[k]), 6),
                                      round(float(a2[k]), 6)],
                         "abs_normal_dot": round(float(dot[k]), 4),
                         "centroid_gap_mm": round(float(gap[k]), 4)})
    out = {"self_intersections": int(hit.sum()),
           "self_intersections_truncated": False,
           "pair_tests": int(len(pairs)),
           "self_intersections_near_parallel_pairs": int(n_parallel),
           "self_intersections_min_area_mm2": float(
               min([d["area_mm2"][0] for d in diag] +
                   [d["area_mm2"][1] for d in diag])) if diag else None,
           "self_intersections_max_normal_dot": float(
               max(d["abs_normal_dot"] for d in diag)) if diag else None,
           "self_intersections_diag": diag,
           "self_intersections_note":
               "候选对=质心 kNN 近似；相交判定=平面交线段区间重叠（精确）；"
               "近平面平行的候选对未判定（另行计数）。hp 为命中对证据"}
    if len(hp) and severe_min_area_mm2 is not None:
        # **逐对**分级（不再用"最小面积+最大法向角"这种全有全无的判据）：
        # 一对命中两个面都够大、且法向不近平行 → 真交叉；否则是数值级相切。
        small = np.minimum(a1, a2) < severe_min_area_mm2
        par = dot > (severe_max_normal_dot if severe_max_normal_dot
                     else 0.999)
        sev = ~(small | par)
        out["self_intersections_severe"] = int(sev.sum())
        out["self_intersections_micro_defect"] = int((small | par).sum())
        paired = np.minimum(a1, a2)
        out["self_intersections_pair_area_mm2"] = {
            "p50": float(np.median(paired)),
            "p95": float(np.quantile(paired, .95)),
            "max": float(paired.max())}
        out["self_intersections_pair_normal_dot_p50"] = float(np.median(dot))
        # 逐对证据（修缝用）：哪两个面、多大、多平行、交线段多长
        sg = seglen[hit] if len(seglen) else np.zeros(len(hp))
        out["self_intersections_severe_pairs"] = [
            {"face_a": int(hp[k, 0]), "face_b": int(hp[k, 1]),
             "area_a_mm2": round(float(a1[k]), 6),
             "area_b_mm2": round(float(a2[k]), 6),
             "area_min_mm2": round(float(paired[k]), 6),
             "abs_normal_dot": round(float(dot[k]), 4),
             "centroid_gap_mm": round(float(gap[k]), 4),
             "intersection_segment_mm": round(float(sg[k]), 4)}
            for k in np.flatnonzero(sev)[:20]]
        out["self_intersections_severe_face_ids"] = [
            [int(hp[k, 0]), int(hp[k, 1])] for k in np.flatnonzero(sev)[:200]]
        out["self_intersections_hit_face_ids"] = [
            [int(hp[k, 0]), int(hp[k, 1])] for k in range(min(len(hp), 500))]
    if focus_faces is not None:
        out["self_intersections_focused"] = True
        if return_pairs:
            out["self_intersections_pairs"] = hp          # (m,2) int64 —— **全部命中**
    if return_pairs and severe_min_area_mm2 is not None and len(hp):
        # **severe 子集**（面积与法向角都过阈值；相切微面对不算）—— 供形变硬门用。
        # 只用 self_intersections_pairs（全命中）会把 dot=1.0 的相切对也算成违规
        # （实测：α=1 报 90 个"severe"，其实全是 dot=1.0、面积 0.0004~0.0055mm² 的相切对）。
        keep_sev = (~small) & (~par)
        out["self_intersections_severe_pairs_list"] = hp[keep_sev]
        # severe 子集**自己的证据**（面积/法向角/间距/交线段长）——
        # 原 diag 只装前 5 个"原始命中"，位置常被相切对占满。
        idx_sev = np.flatnonzero(keep_sev)
        out["self_intersections_severe_diag"] = [
            {"faces": [int(hp[k, 0]), int(hp[k, 1])],
             "area_mm2": [round(float(a1[k]), 6), round(float(a2[k]), 6)],
             "abs_normal_dot": round(float(dot[k]), 4),
             "centroid_gap_mm": round(float(gap[k]), 4),
             "intersection_segment_mm": round(float(sg[k]), 4)}
            for k in idx_sev[:10]]
    return out


def _share_any_vertex(t1, t2):
    out = np.zeros(len(t1), dtype=bool)
    for a in range(3):
        for b in range(3):
            out |= t1[:, a] == t2[:, b]
    return out


def _cross_interval(tri, dvals, q0, d_hat):
    """三角形与另一平面的交线段在交线上的投影区间 [lo, hi]（无交 → inf/-inf）。

    **必须用"与对方平面的交线段"而不是"整块三角形的投影"**：整块投影会给出
    更大的区间，把邻近但不真正相交的非相邻面对判成相交（实测在干净 H.1 上
    假报 1887 对）。
    """
    p = len(tri)
    lo = np.full(p, np.inf)
    hi = np.full(p, -np.inf)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        di, dj = dvals[:, i], dvals[:, j]
        cross = (di * dj <= 0.0) & (np.abs(di - dj) > 1e-300)
        t = np.where(cross, di / np.where(cross, di - dj, 1.0), 0.0)
        pt = tri[:, i] + t[:, None] * (tri[:, j] - tri[:, i])
        proj = np.einsum("ij,ij->i", pt - q0, d_hat)
        lo = np.where(cross, np.minimum(lo, proj), lo)
        hi = np.where(cross, np.maximum(hi, proj), hi)
    return lo, hi


def _tri_tri_intersect(t1, t2, *, return_segment: bool = False):
    """Möller 三角形相交（向量化）→ ``(命中掩码, 近平行对数[, 交线段长度])``。

    判定：两三角形与对方平面的交线段在交线上**区间重叠**。三顶点同侧分离则先跳过。
    ``return_segment=True`` 时第三个返回值是重叠段长度（mm）——修缝时判断
    "真交叉 vs 数值级相切"要用到它（数值级相切的重叠段极短）。
    """
    p = len(t1)
    hit = np.zeros(p, dtype=bool)
    seg = np.zeros(p, dtype=np.float64)
    n1 = np.cross(t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 0])
    n2 = np.cross(t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0])
    ln1 = np.linalg.norm(n1, axis=1)
    ln2 = np.linalg.norm(n2, axis=1)
    d = np.cross(n1, n2)
    dd = np.linalg.norm(d, axis=1)
    n_parallel = int(np.sum(dd <= 1e-12 * ln1 * ln2))
    d1 = np.einsum("ijk,ik->ij", t1 - t2[:, 0][:, None, :], n2)
    d2 = np.einsum("ijk,ik->ij", t2 - t1[:, 0][:, None, :], n1)
    sep = ((d1 > 0).all(1) | (d1 < 0).all(1)
           | (d2 > 0).all(1) | (d2 < 0).all(1))
    cand = ~sep & (dd > 1e-12 * ln1 * ln2)
    if not cand.any():
        return (hit, n_parallel, seg) if return_segment else (hit, n_parallel)
    ids = np.flatnonzero(cand)
    a = t1[ids, 0]
    a2 = t2[ids, 0]
    n1c, n2c, dc = n1[ids], n2[ids], d[ids]
    c1 = np.einsum("ij,ij->i", n1c, a)
    c2 = np.einsum("ij,ij->i", n2c, a2)
    d2n = np.maximum(dd[ids] ** 2, 1e-30)
    # 交线上一点 x = (c1·(n2×d) + c2·(d×n1)) / |d|²
    q0 = (c1[:, None] * np.cross(n2c, dc)
          + c2[:, None] * np.cross(dc, n1c)) / d2n[:, None]
    d_hat = dc / dd[ids, None]
    lo1, hi1 = _cross_interval(t1[ids], d1[ids], q0, d_hat)
    lo2, hi2 = _cross_interval(t2[ids], d2[ids], q0, d_hat)
    hit[ids] = (lo1 <= hi2) & (lo2 <= hi1)
    seg[ids] = np.maximum(np.minimum(hi1, hi2) - np.maximum(lo1, lo2), 0.0)
    return (hit, n_parallel, seg) if return_segment else (hit, n_parallel)


# --------------------------------------------------------------------------
# 阈值规则
# --------------------------------------------------------------------------

def geometry_safety_gate(vertices, faces, *, reference_vertices=None,
                         max_flip_ratio: float | None = 0.0,
                         max_flipped_area_ratio: float | None = None,
                         max_self_intersections: int | None = 0,
                         max_nonmanifold_edges: int = 0,
                         expect_open_boundary: bool | None = None,
                         si_neighbor_k: int = 8,
                         si_max_pairs: int = 2_000_000) -> dict:
    """**各阶段通用几何安全守门员**（与 Thickness Audit 同一套实现）。

    用户 2026-10-02 规格：CrownOuter / Contact / Occlusion / Thickness Repair /
    Final Stitch 每一步都必须能立刻回答"这一步改善了什么、又破坏了什么"。
    所以翻面 / 自交 / 非流形 / 绕向 / 边界 这些判据只能有**一份实现**，
    阈值由调用方给（各阶段松紧不同），判定结果与指标一起返回。

    ``reference_vertices``：同连接关系的"上一步"网格（算翻面率必需）。
    ``expect_open_boundary``：True = 必须恰好一条开放边界（颈环）。
    """
    qc = input_qc(np.asarray(vertices, dtype=np.float64),
                  np.asarray(faces, dtype=np.int64),
                  reference_vertices=reference_vertices,
                  si_neighbor_k=si_neighbor_k, si_max_pairs=si_max_pairs)
    violations = []
    if qc["nonmanifold_edges"] > max_nonmanifold_edges:
        violations.append("nonmanifold_edges=%d" % qc["nonmanifold_edges"])
    if qc["winding_inconsistent_edges"] > 0:
        violations.append("winding_inconsistent=%d"
                          % qc["winding_inconsistent_edges"])
    if max_self_intersections is not None:
        if qc.get("self_intersections") is None:
            violations.append("self_intersections=UNMEASURED(%s)"
                              % qc.get("self_intersections_note", ""))
        elif qc["self_intersections"] > max_self_intersections:
            violations.append("self_intersections=%d"
                              % qc["self_intersections"])
    if max_flip_ratio is not None:
        if qc.get("flipped_ratio") is None:
            violations.append("flipped_ratio=UNMEASURED(%s)"
                              % qc.get("flipped_note", ""))
        elif qc["flipped_ratio"] > max_flip_ratio:
            violations.append("flipped_ratio=%.4f" % qc["flipped_ratio"])
    if max_flipped_area_ratio is not None:
        if qc.get("flipped_area_ratio") is None:
            violations.append("flipped_area_ratio=UNMEASURED")
        elif qc["flipped_area_ratio"] > max_flipped_area_ratio:
            violations.append("flipped_area_ratio=%.6f"
                              % qc["flipped_area_ratio"])
    if expect_open_boundary is True and qc["boundary_edges"] == 0:
        violations.append("boundary_edges=0（颈环缺失）")
    return {"ok": not violations, "violations": violations, "metrics": qc}


def thresholds_of(d_margin, p: ThicknessParameters):
    """→ (T_i, near_mask)。近颈缘规则：linear 为默认读法，step 作对照。"""
    if p.near_margin_thickness_mm is None or \
            p.near_margin_distance_mm is None:
        return np.full(len(d_margin), float(p.min_thickness_mm)), \
            np.zeros(len(d_margin), dtype=bool)
    t_near = float(p.near_margin_thickness_mm)
    t_full = float(p.min_thickness_mm)
    dist = max(float(p.near_margin_distance_mm), 1e-9)
    frac = np.clip(d_margin / dist, 0.0, 1.0)
    if p.near_margin_mode == "step":
        thr = np.where(d_margin >= dist, t_full, t_near)
    elif p.near_margin_mode == "linear":
        thr = t_near + (t_full - t_near) * frac
    else:
        raise ThicknessError("未知 near_margin_mode=%r" % p.near_margin_mode)
    return thr, d_margin < dist


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------

def audit_thickness(outer_mesh, intaglio_mesh, patient_frame: PatientFrame, *,
                    axis_mesial=None, axis_buccal=None,
                    contact_patches: dict | None = None,
                    antagonist_patch: NeighborPatch | None = None,
                    contact_target_mm: float = 0.0,
                    antagonist_target_mm: float = 0.0,
                    reference_mesh=None,
                    params: ThicknessParameters | None = None
                    ) -> ThicknessResult:
    """壁厚审计 + 冲突归因（**只测不修**）。

    ``outer_mesh`` / ``intaglio_mesh``：只需有 ``.vertices`` / ``.faces``；
    ``contact_patches``：{side: NeighborPatch}（已定向）；``antagonist_patch``：
    同类型 patch。任何缺失的约束会被如实记进 ``notes``，不假装验过。
    """
    from ..measurement_core import DOMAIN_THICKNESS, THICKNESS_CONTRACT, domain_report
    if params is None:
        raise ThicknessError("ThicknessParameters 必须显式给出（阈值不猜）")
    if not np.isfinite(params.min_thickness_mm) or params.min_thickness_mm <= 0:
        raise ThicknessError("min_thickness_mm 必须是正有限值")
    p = params
    res = ThicknessResult()
    ov = np.asarray(outer_mesh.vertices, dtype=np.float64)
    of = np.asarray(outer_mesh.faces, dtype=np.int64)
    iv = np.asarray(intaglio_mesh.vertices, dtype=np.float64)
    ifaces = np.asarray(intaglio_mesh.faces, dtype=np.int64)
    # 逐顶点面积（1/3 面面积），全程用它做"面积账"
    iv_area = np.zeros(len(iv))
    ia = _tri_areas(iv, ifaces)
    np.add.at(iv_area, ifaces.ravel(), np.repeat(ia / 3.0, 3))

    res.input_qc = input_qc(
        ov, of,
        reference_vertices=(None if reference_mesh is None
                            else np.asarray(reference_mesh.vertices)),
        si_neighbor_k=p.si_neighbor_k, si_max_pairs=p.si_max_pairs)

    sq = SurfaceQuery(ov, of, boundary_safety_mm=p.boundary_safety_mm,
                      k_candidates=p.k_candidates,
                      max_query_edge_mm=p.max_query_edge_mm)
    res.input_qc["query_faces"] = sq.n_faces_query
    # 颈缘曲线取**原始网格**的开放边界（query 网格是细分过的，索引不同）
    bedges = _boundary_edges(of)
    if len(bedges) == 0:
        raise ThicknessError("外表面没有开放边界 —— 颈缘缺失，无法定位 margin")
    loop_idx = np.asarray(sorted({int(v) for e in bedges for v in e}),
                          dtype=np.int64)
    d_margin = cKDTree(ov[loop_idx]).query(iv, k=1)[0]

    z = np.asarray(patient_frame.z, dtype=np.float64)
    z = z / np.linalg.norm(z)
    origin = np.asarray(patient_frame.origin, dtype=np.float64)
    rel = iv - origin
    axial = rel @ z
    ln = vertex_normals(iv, ifaces)
    radial = rel - np.outer(axial, z)
    rn = np.linalg.norm(radial, axis=1)
    wall = (rn > 1e-6) & (np.abs(ln @ z) < 0.5)
    if int(wall.sum()) > 10 and float(np.median(np.einsum(
            "ij,ij->i", ln[wall], radial[wall] / rn[wall, None]))) < 0:
        ln = -ln
        res.notes.append("内表面法向整体反向 —— 按径向判据翻号（记录在案）")

    cp, fids, signed, t_closest, valid, face_flipped = sq.query(iv)
    # 内/外判定用**绕数**（对翻面/自交稳健）；法向符号只留作翻面交互诊断
    w_out = winding_number(iv, sq.tri_original)
    outside = w_out < p.outside_winding_threshold
    w_band = np.abs(w_out - p.outside_winding_threshold) < p.winding_band
    res.notes.append(
        "内/外：绕数判在冠外 %d 点（%.3fmm²，深度中位 %.3fmm）；法向判在冠外 "
        "%d 点，差 %d 点落在翻面/折叠区（Occlusion 留下 %d 个最近面翻面）—— "
        "法向判据在这类区域不可信，两个数都留在报告里"
        % (int(outside.sum()), float(iv_area[outside].sum()),
           float(np.median(t_closest[outside])) if outside.any() else 0.0,
           int((signed >= 0).sum()),
           int(((signed >= 0) & ~outside).sum()),
           int(face_flipped.sum())))
    res.input_qc["outside_by_winding"] = int(outside.sum())
    res.input_qc["outside_by_face_normal"] = int((signed >= 0).sum())
    res.input_qc["face_flipped_at_closest"] = int(face_flipped.sum())
    res.input_qc["winding_ambiguous_points"] = int(w_band.sum())

    t_ray, ray_face = ray_first_hit(iv, ln, sq.tri_original,
                                    max_t=p.ray_max_mm)
    ray_status = np.where(np.isfinite(t_ray), RAY_HIT, RAY_NO_HIT)
    ray_status[outside] = RAY_STARTS_OUTSIDE
    t_ray = np.where(np.isfinite(t_ray), t_ray, p.ray_max_mm)
    amb = (np.abs(t_ray - t_closest)
           > np.maximum(p.ambiguous_abs_mm,
                        p.ambiguous_rel * np.maximum(t_closest, 1e-6))) \
        & (ray_status == RAY_HIT)
    primary = np.where(ray_status == RAY_HIT, t_ray, t_closest)
    flags = np.zeros(len(iv), dtype=np.int64)
    flags[ray_status == RAY_NO_HIT] = 1              # FALLBACK_CLOSEST
    flags[ray_status == RAY_STARTS_OUTSIDE] = 2      # OUTSIDE_OUTER
    flags[amb] = 3                                   # AMBIGUOUS_THICKNESS
    flags[~valid] = 4                                # 边界安全带（颈缘刀口）
    flags[w_band] = 5                                # 绕数在边界带（内外不可分）

    # ---- 区域：颈缘带（d_margin 与 anatomical_regions 的并集）优先
    regions = anatomical_regions(iv, origin, z, axis_mesial, axis_buccal,
                                 occlusal_fraction=p.occlusal_fraction,
                                 cervical_band_mm=p.cervical_band_mm)
    reg_of = np.full(len(iv), "lateral_unknown", dtype=object)
    for name in ("occlusal", "mesial", "distal", "buccal", "lingual"):
        if name in regions:
            reg_of[regions[name]] = name
    cerv = (d_margin < p.cervical_band_mm) | regions.get(
        "cervical", np.zeros(len(iv), dtype=bool))
    reg_of[cerv] = "cervical"

    # ---- 阈值与薄判定（颈缘刀口带与"穿出外表面"的点不参与判定）
    thr, near_band = thresholds_of(d_margin, p)
    thr_step = thr
    if p.near_margin_thickness_mm is not None and \
            p.near_margin_distance_mm is not None:
        thr_step = np.where(d_margin >= float(p.near_margin_distance_mm),
                            float(p.min_thickness_mm),
                            float(p.near_margin_thickness_mm))
    evaluated = (flags != 4) & (d_margin >= p.knife_edge_mm) & ~outside
    deficit = np.where(evaluated, np.maximum(thr - primary, 0.0), 0.0)
    thin = evaluated & (deficit > 0)

    # ---- 冲突归因：把外表面点沿外法向推 deficit 后，与各约束的剩余间隙
    push_dir = sq.normals[fids] * sq.normal_sign
    hit_pt = iv + np.maximum(t_ray, 0.0)[:, None] * ln
    outer_pt = np.where((ray_status == RAY_HIT)[:, None], hit_pt, cp)
    cand = outer_pt + deficit[:, None] * push_dir
    clearance_contact = np.full(len(iv), np.inf)
    clearance_occl = np.full(len(iv), np.inf)
    blockers = np.zeros(len(iv), dtype=np.int64)
    if contact_patches:
        for _name, patch in contact_patches.items():
            _c, _f, s_p, _d, v_p = patch.closest_with_validity(
                cand, boundary_safety_mm=p.boundary_safety_mm)
            left = np.where(v_p, s_p - float(contact_target_mm), np.inf)
            clearance_contact = np.minimum(clearance_contact, left)
        thin_c = thin & (clearance_contact < p.clearance_margin_mm)
        blockers[thin_c] |= 1
    else:
        res.notes.append("未提供 contact patch —— CONTACT_THICKNESS_CONFLICT "
                         "未验（薄区一律按可修计入是**不安全**的，见 note）")
    if antagonist_patch is not None:
        _c, _f, s_a, _d, v_a = antagonist_patch.closest_with_validity(
            cand, boundary_safety_mm=p.boundary_safety_mm)
        clearance_occl = np.where(v_a, s_a - float(antagonist_target_mm),
                                  np.inf)
        thin_o = thin & (clearance_occl < p.clearance_margin_mm)
        blockers[thin_o] |= 2
    else:
        res.notes.append("未提供 antagonist patch —— OCCLUSION_THICKNESS_"
                         "CONFLICT 未验")
    clearance = np.minimum(clearance_contact, clearance_occl)

    kind = np.full(len(iv), "", dtype=object)
    kind[thin & cerv] = THIN_CERVICAL
    rest = thin & (kind == "")
    kind[rest & (blockers == 0)] = THIN_REPAIRABLE
    b_c = rest & (blockers == 1)
    b_o = rest & (blockers == 2)
    b_b = rest & (blockers == 3)
    kind[b_c] = THIN_CONTACT
    kind[b_o] = THIN_OCCLUSION
    # 两侧都挡：记更紧的那一侧
    kind[b_b] = np.where(clearance_occl[b_b] <= clearance_contact[b_b],
                         THIN_OCCLUSION, THIN_CONTACT)

    # ---- 面积：用内表面自己的面片（iv_area 已在入口算好）
    cov = {
        "n_points": int(outside.sum()),
        "area_mm2": float(iv_area[outside].sum()),
        "depth_mm_p50": float(np.median(t_closest[outside]))
        if outside.any() else None,
        "depth_mm_p95": float(np.quantile(t_closest[outside], .95))
        if outside.any() else None,
        "winding_p50": float(np.median(w_out[outside]))
        if outside.any() else None,
        "outside_by_face_normal": int((signed >= 0).sum()),
        "region_area_mm2": {},
    }

    # ---- 分区账
    res.regions = {}
    for name in REGION_ROWS:
        sel = (reg_of == name) & evaluated
        out_sel = (reg_of == name) & outside
        cov["region_area_mm2"][name] = float(iv_area[out_sel].sum())
        if not sel.any() and not out_sel.any():
            res.regions[name] = {"n": 0}
            continue
        tt = primary[sel] if sel.any() else np.array([0.0])
        res.regions[name] = {
            "n": int(sel.sum()),
            "min": float(tt.min()),
            "p05": float(np.quantile(tt, .05)),
            "median": float(np.median(tt)),
            "p95": float(np.quantile(tt, .95)),
            "thin_area_mm2": float(iv_area[thin & sel].sum()),
            "worst_deficit_mm": float(deficit[sel].max()) if sel.any() else 0.0,
            "n_outside_outer": int(out_sel.sum()),
            "outside_area_mm2": float(iv_area[out_sel].sum()),
            "n_ambiguous": int(((flags == 3) & sel).sum()),
        }

    # ---- 汇总
    thin_area = {k: float(iv_area[kind == k].sum()) for k in
                 (THIN_REPAIRABLE, THIN_CONTACT, THIN_OCCLUSION,
                  THIN_CERVICAL)}
    res.thin_area = thin_area
    ev = evaluated
    res.summary = {
        "n_samples": int(len(iv)),
        "n_evaluated": int(ev.sum()),
        "n_knife_edge_skipped": int((d_margin < p.knife_edge_mm).sum()),
        "n_boundary_invalid": int((flags == 4).sum()),
        "n_outside_outer": int(outside.sum()),
        "n_ambiguous_thickness": int((flags == 3).sum()),
        "n_ray_no_hit": int((flags == 1).sum()),
        "min_thickness_mm": float(primary[ev].min()) if ev.any() else None,
        "p05_thickness_mm": float(np.quantile(primary[ev], .05))
        if ev.any() else None,
        "median_thickness_mm": float(np.median(primary[ev]))
        if ev.any() else None,
        "thin_vertex_count": int(thin.sum()),
        "thin_area_mm2": float(iv_area[thin].sum()),
        "max_deficit_mm": float(deficit.max()),
        "p95_deficit_mm": float(np.quantile(deficit[deficit > 0], .95))
        if (deficit > 0).any() else 0.0,
        "thin_area_by_kind_mm2": dict(thin_area),
        "negative_coverage": cov,
        "region_thin_area_mm2": {name: res.regions[name].get(
            "thin_area_mm2", 0.0) for name in REGION_ROWS},
        "rule": {
            "min_thickness_mm": p.min_thickness_mm,
            "near_margin_thickness_mm": p.near_margin_thickness_mm,
            "near_margin_distance_mm": p.near_margin_distance_mm,
            "near_margin_mode": p.near_margin_mode,
            "cement_gap_mm": p.cement_gap_mm,
            "n_near_margin_band": int(near_band.sum()),
            "knife_edge_mm": p.knife_edge_mm,
        },
        "clearance_need_mm": float(p.clearance_margin_mm),
        "contact_target_mm": float(contact_target_mm),
        "antagonist_target_mm": float(antagonist_target_mm),
        "n_blocked_by_contact": int((blockers & 1).sum()),
        "n_blocked_by_antagonist": int((blockers & 2).sum()),
    }
    res.summary["thin_area_mm2_step_rule"] = float(
        iv_area[ev & (thr_step - primary > 0)].sum())
    # 保守账：min(t_ray, t_closest)。结构上 t_ray ≥ t_closest 必然成立（射线命中点
    # 本身就在表面上），所以保守账 = "两种读法谁薄算谁"，不给乐观解释留口子。
    conservative = np.minimum(np.where(ray_status == RAY_HIT, t_ray, np.inf),
                              t_closest)
    thin_cons = ev & (thr - conservative > 0)
    res.summary["thin_area_mm2_conservative"] = float(iv_area[thin_cons].sum())
    res.summary["region_thin_area_conservative_mm2"] = {
        name: float(iv_area[thin_cons & (reg_of == name)].sum())
        for name in REGION_ROWS}
    res.thickness_map["t_conservative_mm"] = conservative
    res.thickness_map = {
        "vertices": iv, "t_closest_mm": t_closest, "t_ray_mm": t_ray,
        "t_primary_mm": primary, "deficit_mm": deficit,
        "threshold_mm": thr, "d_margin_mm": d_margin, "region": reg_of,
        "kind": kind, "flags": flags, "clearance_mm": clearance,
        "blockers": blockers, "vertex_area_mm2": iv_area,
        "outer_closest_point": cp, "outer_face_id": fids,
    }

    # ---- 状态
    fail = []
    qc = res.input_qc
    if qc.get("nonmanifold_edges", 0) > p.max_nonmanifold_edges:
        fail.append("nonmanifold_edges=%d" % qc["nonmanifold_edges"])
    if qc.get("self_intersections") is not None and \
            p.max_self_intersections is not None and \
            qc["self_intersections"] > p.max_self_intersections:
        fail.append("self_intersections=%d" % qc["self_intersections"])
    if qc.get("flipped_ratio") is not None and \
            p.max_flipped_ratio is not None and \
            qc["flipped_ratio"] > p.max_flipped_ratio:
        fail.append("flipped_ratio=%.4f" % qc["flipped_ratio"])
    if int(outside.sum()) > 0:
        res.notes.append(
            "**负壁厚（覆盖率失败）**：%d 个采样点（%.3fmm²）的内表面落到冠外，"
            "深度中位 %.3fmm —— 外表面切进了预备体；这不是薄，是外表面没盖住"
            "预备体（H.1 模板形变只保证颈环对齐，不保证 outer ≥ prep）"
            % (cov["n_points"], cov["area_mm2"], cov["depth_mm_p50"] or 0.0))
    if fail:
        res.status = STATUS_FAILED
        res.notes.append("输入 QC 未过：" + ", ".join(fail))
    elif int(thin.sum()) == 0 and int(outside.sum()) == 0:
        res.status = STATUS_PASS
    else:
        res.status = STATUS_REVIEW_REQUIRED
        total = sum(thin_area.values())
        res.notes.append(
            "薄区 %.4fmm²（可修 %.4f / 接触冲突 %.4f / 咬合冲突 %.4f / "
            "颈缘 %.4f）+ 负壁厚 %.4fmm²"
            % (total, thin_area[THIN_REPAIRABLE], thin_area[THIN_CONTACT],
               thin_area[THIN_OCCLUSION], thin_area[THIN_CERVICAL],
               cov["area_mm2"]))
    # B1：本域测量契约显式标注（符号用**平滑法向**；有效性仅边界安全；相交审计属本域）。
    res.domain_report = domain_report(
        DOMAIN_THICKNESS, thin_area=dict(thin_area),
        thin_area_mm2=float(sum(thin_area.values())),
        n_evaluated=int(res.summary.get("n_evaluated", 0)), status=res.status)
    res.notes.append("本域口径：%s；有效性=%s；符号来源=%s（与 Contact 的实体内外判定不同）"
                     % (THICKNESS_CONTRACT.penetration_meaning, THICKNESS_CONTRACT.validity,
                        THICKNESS_CONTRACT.sign_method))
    return res


def _tri_areas(v, f):
    tri = v[f]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                         tri[:, 2] - tri[:, 0]), axis=1)

