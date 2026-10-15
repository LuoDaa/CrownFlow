"""真值冠颈缘提取（acceptance truth extractor）。

**这不是患者 Prep 的自动 Margin 算法。** 两者要分开：

- 本模块：从**已完工的最终冠**里提取 intaglio 与 exterior 的真实交界缝 ——
  用于给验收套件提供可靠真值。输入是已知正确的成品冠。
- 患者 Margin 算法：从 Prep/Abutment 扫描推断 finish line，需要面对噪声、
  牙龈遮挡、扫描裁切边界。那是另一件事。

---

## 为什么不能沿用旧方法

旧 `_metrics.cervical_margin()` 取"真值冠最低 0.4 mm 带的顶点"。实测该假设不成立：

| 取带宽度 | 方位角覆盖 | 是否闭环 |
|---|---|---|
| 0.2 mm | 15/24 | ❌ 残弧 |
| 0.4 mm | 16/24 | ❌ 残弧 |
| 0.8 mm | 19/24 | ❌ 残弧 |
| 1.5 mm | 24/24 | ✅ 但已跨 1.5 mm 高度 |

而该冠的颈缘相对最佳拟合平面残差 **mean 1.07 / max 3.83 mm** ——
**平面假设不成立**。所以"最低 Z 带""法向正负""拟合平面切一圈"三种做法全部出局：
颈缘是空间三维闭环，不能假设近平面，也不能靠世界 Z。

同样不能用"选最大环"：FDI36 的冠实测 **χ = 0（genus 1）**，有两条缝
（颈缘 + 螺丝通道口），而通道口环在某些度量下更大。

---

## 实际采用的判据：朝向场符号，无阈值

对每个冠面片，取到 support 的**精确最近点**，算

```
cos = n · (nearest_support_point − face_centroid) / |...|
```

语义：面片是否"朝向"support。
- 外表面：材料在内侧，法向背离 support → `cos < 0`
- 内表面：材料在外侧，法向指向 support → `cos > 0`

**标签直接取 `sign(cos)` 即可，不需要阈值，也不需要图割或标签传播。**
这不是偷懒，是有实测依据的选择。

FDI36 实测：

```
近 Support (d ≤ 0.16):  10275 面，其中 cos > 0.1 的 99.9%   → 内表面
远 Support (d ≥ 0.5):   29264 面，其中 cos > 0.1 的  0.0%   → 外表面
距离歧义带 (0.16<d<0.5):   887 面（2.2%）
```

朝向场的符号在两端的纯度接近 100%，零交叉是**内蕴**的（不需要选分离点）；
而距离场本身是双峰的（内表面 0.047–0.151 mm，然后一个明显空隙，外表面 0.8–3.15 mm），
这说明几何上内外表面确实清晰可分。

**为什么不用距离阈值硬切**：exocad 数据实测 `CementGapThickness` 跨 0→0.08，
还有 Start / Top / Additional spacing 变体。任何固定距离阈值都会在某些病例上切错。
朝向场没有这个问题。`fitting_params` 若提供，只用于**诊断记录**与
"距离场是否仍呈双峰"的一致性检查，不参与标签判定。

`sign(cos)` 的实测结果（FDI36）：

```
内表面 10482   外表面 29944
界面边 314     界面顶点 314     度数分布 {2: 314}     ← 全部 degree 2，干净流形缝
检出环 2（正好等于 genus 1 应有的两条缝）
```

---

## 环选择：不能取最大，也不能取第一个

FDI36 实测三条候选（`sign(cos)` 给出 2 条真缝 + 1 个 4 点伪影）：

| 环 | 点数 | 周长 mm | 轴向相对位置 | 半径 mean/max | 到 Support 中位 mm |
|---|---|---|---|---|---|
| 0 | 208 | 10.70 | 0.71（咬合侧） | 1.29 / 2.01 | 0.071 |
| 1 | **102** | **21.76** | **0.10（颈侧）** | **3.36 / 3.83** | **0.237** |
| 2 | 4 | 0.46 | 0.76 | 3.23 / 3.32 | 0.901 |

环 1 是颈缘（周长最大、半径最大、位于颈侧、环绕整个基台）；
环 0 是螺丝通道口。

**注意一个反直觉的实测**：通道口到 support 的中位距离 **0.071** 比颈缘的 **0.237**
更近 —— 因为它就在基台顶部正上方。所以"离 support 近"这个直觉判据是**反向的**，
不能用作主判据。

同理，"沿 −axis 是否被冠自身遮挡"也不判别：通道口与内腔同轴，
向下的射线穿孔后直接出颈缘开口，实测两个环的被挡比例都是 0.000。

**真正判别的是**：环的投影是否包围**该环自身轴向高度附近**的 support 截面。

| 环 | 支撑点被包围比例 |
|---|---|
| 0（通道口） | 0.095 |
| 1（颈缘） | **1.000** |
| 2（伪影） | 0.000 |

10 倍分离度。这是主判据，其余判据（周长、半径、自交、闭合、简单性、
方位角覆盖）作为 gate 与辅助分项。
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh

from .insertion import _orthonormal_basis

#: 判定某个面的朝向为"明确朝向 support / 背离 support"的余弦阈值。
#: 只用于**统计诊断**（锐利度），不用于标签判定 —— 标签是 sign(cos)。
FACING_AMBIGUOUS_BAND = 0.1

#: 若 |cos| < 该值的面片占比超过此比例，说明朝向场不够锐利，标 REVIEW_REQUIRED。
MAX_AMBIGUOUS_FRACTION = 0.20

#: 候选环的最小周长（mm）。用于滤掉拓扑噪声产生的小环（实测伪影周长 0.46 mm）。
MIN_LOOP_PERIMETER_MM = 2.0

#: 自交判定容差（mm）。
SELF_INTERSECTION_TOL_MM = 1e-6

#: 判定"包围"的阈值。
ENCLOSURE_MIN = 0.90

#: 方位角分箱数。颈缘是闭合环，应覆盖接近 360°。
AZIMUTH_BINS = 24
MIN_AZIMUTH_COVERAGE = 0.95


class MarginStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


@dataclass
class CandidateLoop:
    """一条候选缝的完整度量。"""

    index: int
    vertex_ids: list[int]
    perimeter_mm: float
    is_closed: bool
    is_simple: bool
    closure_gap_mm: float
    n_self_intersections: int
    spacing_cv: float
    azimuth_coverage: float
    enclosure_fraction: float
    support_distance_median_mm: float
    support_distance_max_mm: float
    radius_mean_mm: float
    axial_relative_position: float
    score: float = 0.0
    rejected_reason: str | None = None

    @property
    def passes_gates(self) -> bool:
        return (self.is_closed and self.is_simple and self.n_self_intersections == 0
                and self.perimeter_mm >= MIN_LOOP_PERIMETER_MM
                and self.rejected_reason is None)

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "n_points": len(self.vertex_ids),
            "perimeter_mm": round(self.perimeter_mm, 4),
            "is_closed": self.is_closed,
            "is_simple": self.is_simple,
            "closure_gap_mm": round(self.closure_gap_mm, 6),
            "n_self_intersections": self.n_self_intersections,
            "spacing_cv": round(self.spacing_cv, 6),
            "azimuth_coverage": round(self.azimuth_coverage, 4),
            "enclosure_fraction": round(self.enclosure_fraction, 4),
            "support_distance_median_mm": round(self.support_distance_median_mm, 4),
            "support_distance_max_mm": round(self.support_distance_max_mm, 4),
            "radius_mean_mm": round(self.radius_mean_mm, 4),
            "axial_relative_position": round(self.axial_relative_position, 4),
            "score": round(self.score, 6),
            "passes_gates": self.passes_gates,
            "rejected_reason": self.rejected_reason,
        }


@dataclass
class MarginExtractionResult:
    """颈缘真值提取结果。

    **不包含 ``margin_plane_height`` 之类的核心语义。** 最佳拟合平面只在
    ``diagnostics`` 里作为诊断量出现 —— 实测真实颈缘相对最优平面的残差可达毫米级，
    强行平面化会直接损失真实几何。
    """

    points: list[list[float]]
    edge_ids: list[list[int]]
    is_closed: bool
    is_simple: bool
    perimeter_mm: float
    spacing_cv: float
    azimuth_coverage: float
    support_distance_stats: dict
    confidence: float
    status: MarginStatus
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "points": [[round(float(c), 6) for c in p] for p in self.points],
            "edge_ids": [[int(a), int(b)] for a, b in self.edge_ids],
            "is_closed": self.is_closed,
            "is_simple": self.is_simple,
            "perimeter_mm": round(float(self.perimeter_mm), 4),
            "spacing_cv": round(float(self.spacing_cv), 6),
            "azimuth_coverage": round(float(self.azimuth_coverage), 4),
            "support_distance_stats": self.support_distance_stats,
            "confidence": round(float(self.confidence), 4),
            "status": self.status.value,
            "diagnostics": self.diagnostics,
        }


# --------------------------------------------------------------------- 基础量

def facing_field(
    crown: trimesh.Trimesh, support: trimesh.Trimesh
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(distance, cosine)``：

    - ``distance``：面片质心到 support **表面**的精确最近距离（``trimesh.proximity.closest_point``）
    - ``cosine``：面片法向与"指向最近 support 点"方向的余弦

    余弦 > 0 表示面片朝向 support（内表面），< 0 表示背离（外表面）。
    """
    centroids = np.asarray(crown.triangles_center, dtype=float)
    normals = np.asarray(crown.face_normals, dtype=float)
    nearest, distance, _ = trimesh.proximity.closest_point(support, centroids)

    vec = nearest - centroids
    mag = np.linalg.norm(vec, axis=1)
    safe = mag > 1e-9
    cosine = np.zeros(len(centroids), dtype=float)
    cosine[safe] = np.einsum("ij,ij->i", normals[safe], vec[safe]) / mag[safe]
    return distance, cosine


def _shared_edge_map(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            out[(min(u, v), max(u, v))].append(int(fi))
    return out


def _interface_edges(edge_map, labels: np.ndarray) -> list[tuple[int, int]]:
    """两侧标签不同的共享边。"""
    return [(u, v) for (u, v), fl in edge_map.items()
            if len(fl) == 2 and labels[fl[0]] != labels[fl[1]]]


def _trace_loops(
    interface_edges: list[tuple[int, int]], vertices: np.ndarray
) -> tuple[list[list[int]], Counter]:
    """把界面边追踪成闭环。

    与 ``margin.boundary_loops`` 的关键差别：**遇到度数不为 2 的顶点时按"最直延续"
    选择下一跳，而不是取 ``choices[0]``**。后者会在 T 型连接/非流形处静默跳环，
    实测在真实扫描上产出跨 14 mm 的退化环。

    度数直方图随结果返回，便于发现界面缝不是干净流形的情况。
    """
    graph: dict[int, list[int]] = defaultdict(list)
    for u, v in interface_edges:
        graph[u].append(v)
        graph[v].append(u)
    degrees = Counter(len(v) for v in graph.values())

    loops: list[list[int]] = []
    used_edges: set[tuple[int, int]] = set()

    def next_hop(prev: int | None, cur: int) -> int | None:
        candidates = []
        for nxt in graph[cur]:
            if prev is not None and nxt == prev:
                continue
            if tuple(sorted((cur, nxt))) in used_edges:
                continue
            candidates.append(nxt)
        if not candidates:
            return None
        if len(candidates) == 1 or prev is None:
            return candidates[0]
        # 度数 > 2：选"最直延续"，避免跳环
        d_prev = vertices[cur] - vertices[prev]
        d_prev /= max(float(np.linalg.norm(d_prev)), 1e-12)
        def straightness(n):
            d = vertices[n] - vertices[cur]
            d /= max(float(np.linalg.norm(d)), 1e-12)
            return float(np.dot(d_prev, d))
        return max(candidates, key=straightness)

    for start in list(graph):
        if all(tuple(sorted((start, n))) in used_edges for n in graph[start]):
            continue
        path = [start]
        prev, cur = None, start
        while True:
            nxt = next_hop(prev, cur)
            if nxt is None:
                break
            used_edges.add(tuple(sorted((cur, nxt))))
            if nxt == start:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            if len(path) > 100_000:
                break
        if len(path) >= 8:
            loops.append(path)

    return loops, degrees


# ------------------------------------------------------------------- 度量

def _self_intersection_count(points: np.ndarray) -> int:
    """非相邻线段相交计数。

    **传入的是开环点列**（不要 append 首个点来闭合）。曾因 append 闭合点而多出一个
    重复端点：数组变成 ``[v0..v_{m-1}, v0]``，于是段 0 与闭合段在几何上共享 v0，
    但模运算的邻接判断认不出来，把正确的颈缘环误判为"自交 1 处"并整条拒绝。
    改用模索引后邻接关系是精确的。
    """
    n = len(points)
    if n < 4:
        return 0
    seg_a = points
    seg_b = np.roll(points, -1, axis=0)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (i + 1) % n == j or (j + 1) % n == i:
                continue          # 共享端点，相邻
            d = _segment_distance(seg_a[i], seg_b[i], seg_a[j], seg_b[j])
            if d < SELF_INTERSECTION_TOL_MM:
                count += 1
    return count


def _segment_distance(p1, p2, q1, q2) -> float:
    """两线段间最短距离（标准实现）。"""
    u = p2 - p1
    v = q2 - q1
    w = p1 - q1
    a, b, c = float(u @ u), float(u @ v), float(v @ v)
    d, e = float(u @ w), float(v @ w)
    denom = a * c - b * b
    eps = 1e-12
    if denom < eps:
        s = 0.0
        t = (e / c) if c > eps else 0.0
    else:
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
    s = min(1.0, max(0.0, s))
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(w + s * u - t * v))


def _azimuth_coverage(points: np.ndarray, axis: np.ndarray, bins: int = AZIMUTH_BINS) -> float:
    """把环投影到 ⊥axis 的平面，统计方位角分箱覆盖比例。

    作用：防止"只取到最低点附近半圈"的残弧再次混进验收
    （旧方法实测只覆盖 16/24）。不要求每箱点数相同，只要求每箱非空。
    """
    e1, e2 = _orthonormal_basis(axis)
    rel = points - points.mean(axis=0)
    xy = np.c_[rel @ e1, rel @ e2]
    ang = np.arctan2(xy[:, 1], xy[:, 0])
    idx = ((ang + math.pi) / (2 * math.pi) * bins).astype(int) % bins
    return float(len(set(idx.tolist())) / bins)


def _point_in_polygon_2d(polygon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """射线交叉法（向量化）。polygon 为闭合顺序点列。"""
    n = len(polygon)
    x = polygon[:, 0]
    y = polygon[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    px = points[:, 0]
    py = points[:, 1]
    j = n - 1
    for i in range(n):
        cond = ((y[i] > py) != (y[j] > py))
        with np.errstate(divide="ignore", invalid="ignore"):
            xin = (x[j] - x[i]) * (py - y[i]) / np.where(y[j] - y[i] == 0, 1e-30, y[j] - y[i]) + x[i]
        inside ^= cond & (px < xin)
        j = i
    return inside


def _enclosure_fraction(
    loop_points: np.ndarray, support_vertices: np.ndarray, axis: np.ndarray,
    *, band_mm: float = 0.35,
) -> float:
    """环的投影是否包围**该环自身轴向高度附近**的 support 截面。

    这是主判据。实测区分度（FDI36）：

        螺丝通道口 0.095   |   颈缘 1.000   |   拓扑伪影 0.000

    **不要改成"是否包含 support 中心"** —— 通道口与基台同轴，
    包含中心这一点对它同样成立，不判别。
    """
    e1, e2 = _orthonormal_basis(axis)
    loop_s = loop_points @ axis
    lo = float(loop_s.min()) - band_mm
    hi = float(loop_s.max()) + band_mm
    s = support_vertices @ axis
    band = support_vertices[(s >= lo) & (s <= hi)]
    if len(band) < 10:
        band = support_vertices

    sample = band[:: max(1, len(band) // 400)]
    poly = np.c_[loop_points @ e1, loop_points @ e2]
    q = np.c_[sample @ e1, sample @ e2]
    if len(poly) < 3:
        return 0.0
    return float(_point_in_polygon_2d(poly, q).mean())


def _score_loop(
    loop_points: np.ndarray,
    support_vertices: np.ndarray,
    axis: np.ndarray,
    crown_axis_range: tuple[float, float],
    support_distance_median: float,
) -> tuple[float, dict]:
    """候选环评分。主项是包围比例，其余为辅助。"""
    enclosure = _enclosure_fraction(loop_points, support_vertices, axis)
    perimeter = float(np.linalg.norm(np.diff(loop_points, axis=0), axis=1).sum())

    s = loop_points @ axis
    lo, hi = crown_axis_range
    axial_rel = float((s.mean() - lo) / max(hi - lo, 1e-9))

    # 轴向位置：颈缘应在冠的颈侧。用 (1 - axial_rel) 作为辅助分，
    # 但它不是主判据（主判据是包围），避免退化成"最低 Z 带"。
    axial_term = 1.0 - axial_rel

    score = (2.0 * enclosure
             + 0.3 * min(perimeter / 20.0, 2.0)
             + 0.3 * axial_term)

    detail = {
        "enclosure_fraction": enclosure,
        "perimeter_mm": perimeter,
        "axial_relative_position": axial_rel,
        "axial_term": axial_term,
        "support_distance_median_mm": support_distance_median,
    }
    return float(score), detail


# ------------------------------------------------------------------- 主入口

def extract_gt_crown_margin(
    final_crown_mesh: trimesh.Trimesh,
    support_mesh: trimesh.Trimesh,
    insertion_axis,
    *,
    fitting_params: dict | None = None,
) -> MarginExtractionResult:
    """从已完工的最终冠提取 intaglio ↔ exterior 的真实颈缘缝。

    ``insertion_axis`` 只用于：局部坐标系构造、方位角分箱、轴向相对位置。
    **所有判定都基于几何，不依赖世界坐标轴**（否则破坏刚性等变性）。

    ``fitting_params``（如 ``CementGapThickness``）只写入诊断，不参与标签判定 ——
    exocad 实测该值跨 0→0.08，任何固定阈值都会在某些病例上切错。
    """
    if len(final_crown_mesh.faces) == 0 or len(support_mesh.faces) == 0:
        raise ValueError("冠或 support 为空网格")
    axis = np.asarray(insertion_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) < 1e-9:
        raise ValueError("insertion_axis 无效")
    axis = axis / float(np.linalg.norm(axis))

    crown = final_crown_mesh
    distance, cosine = facing_field(crown, support_mesh)
    labels = (cosine > 0).astype(np.int8)

    ambiguous_fraction = float((np.abs(cosine) < FACING_AMBIGUOUS_BAND).mean())

    # 拓扑：χ = V - E + F，genus = (2 - χ)/2。用于解释"为什么会有多条缝"。
    chi = len(crown.vertices) - len(crown.edges_unique) + len(crown.faces)
    genus = int(round((2 - chi) / 2)) if crown.is_watertight else None

    edge_map = _shared_edge_map(np.asarray(crown.faces))
    interface = _interface_edges(edge_map, labels)
    vertices = np.asarray(crown.vertices, dtype=float)
    raw_loops, degrees = _trace_loops(interface, vertices)

    crown_s = vertices @ axis
    crown_axis_range = (float(crown_s.min()), float(crown_s.max()))
    support_vertices = np.asarray(support_mesh.vertices, dtype=float)

    candidates: list[CandidateLoop] = []
    for i, loop in enumerate(raw_loops):
        pts = vertices[loop]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        perimeter = float(seg.sum())
        closure_gap = float(np.linalg.norm(pts[-1] - pts[0]))
        is_closed = closure_gap <= max(0.05 * perimeter, 0.05)
        # 简单性：除闭合点外无重复顶点
        is_simple = len(set(loop)) == len(loop)
        n_si = _self_intersection_count(pts)
        spacing_cv = float(np.std(seg) / np.mean(seg)) if len(seg) and np.mean(seg) > 0 else float("inf")
        azimuth = _azimuth_coverage(pts, axis)
        _, d_loop, _ = trimesh.proximity.closest_point(support_mesh, pts)

        score, detail = _score_loop(pts, support_vertices, axis, crown_axis_range,
                                    float(np.median(d_loop)))

        rejected = None
        if not is_closed:
            rejected = f"非闭环（首尾间距 {closure_gap:.4f} mm）"
        elif not is_simple:
            rejected = "含重复顶点，不是简单环"
        elif n_si:
            rejected = f"自交 {n_si} 处"
        elif perimeter < MIN_LOOP_PERIMETER_MM:
            rejected = f"周长 {perimeter:.3f} mm 低于下限 {MIN_LOOP_PERIMETER_MM}（拓扑噪声）"
        elif detail["enclosure_fraction"] < ENCLOSURE_MIN:
            rejected = (f"未包围 support 截面（{detail['enclosure_fraction']:.3f} "
                        f"< {ENCLOSURE_MIN}）—— 不是颈缘")

        candidates.append(CandidateLoop(
            index=i, vertex_ids=loop, perimeter_mm=perimeter, is_closed=is_closed,
            is_simple=is_simple, closure_gap_mm=closure_gap, n_self_intersections=n_si,
            spacing_cv=spacing_cv, azimuth_coverage=azimuth,
            enclosure_fraction=detail["enclosure_fraction"],
            support_distance_median_mm=float(np.median(d_loop)),
            support_distance_max_mm=float(d_loop.max()),
            radius_mean_mm=float(np.linalg.norm(pts - vertices.mean(axis=0), axis=1).mean()),
            axial_relative_position=detail["axial_relative_position"],
            score=score, rejected_reason=rejected,
        ))

    diagnostics: dict = {
        "n_faces": int(len(crown.faces)),
        "n_intaglio_faces": int((labels == 1).sum()),
        "n_outer_faces": int((labels == 0).sum()),
        "facing_ambiguous_fraction": round(ambiguous_fraction, 6),
        "interface_edges": len(interface),
        "interface_vertex_degrees": dict(sorted(degrees.items())),
        "euler_characteristic": int(chi),
        "genus": genus,
        "n_candidate_loops": len(candidates),
        "candidates": [c.as_dict() for c in candidates],
        "axis": [float(v) for v in axis],
        "fitting_params": dict(fitting_params or {}),
        "fitting_params_note": (
            "仅记录，不参与标签判定。exocad 实测 CementGapThickness 跨 0–0.08，"
            "任何固定距离阈值都会在某些病例上切错，因此标签取 sign(cos)。"
        ),
        "reasons": [],
    }

    usable = [c for c in candidates if c.passes_gates]
    if not usable:
        diagnostics["reasons"].append(
            "没有任何候选环通过 gate（闭合 / 简单 / 无自交 / 周长 / 包围 support）")
        return MarginExtractionResult(
            points=[], edge_ids=[], is_closed=False, is_simple=False, perimeter_mm=0.0,
            spacing_cv=float("nan"), azimuth_coverage=0.0,
            support_distance_stats={}, confidence=0.0,
            status=MarginStatus.FAILED, diagnostics=diagnostics,
        )

    best = max(usable, key=lambda c: c.score)
    pts = vertices[best.vertex_ids]
    # 边列表：环上相邻顶点对（含闭合边）
    closed_ids = best.vertex_ids + [best.vertex_ids[0]]
    edge_ids = [[closed_ids[k], closed_ids[k + 1]] for k in range(len(closed_ids) - 1)]

    # 诊断：最佳拟合平面残差。**只作诊断** —— 实测真实颈缘残差可达毫米级，
    # 强行平面化会直接损失真实几何。
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center)
    normal = vh[-1]
    plane_residual = np.abs((pts - center) @ normal)

    reasons: list[str] = []
    status = MarginStatus.OK
    if ambiguous_fraction > MAX_AMBIGUOUS_FRACTION:
        status = MarginStatus.REVIEW_REQUIRED
        reasons.append(f"朝向场不够锐利（|cos|<{FACING_AMBIGUOUS_BAND} 占 {ambiguous_fraction:.1%}）")
    if any(d != 2 for d in degrees):
        # 界面缝不是干净流形：追踪可能跳环（这正是旧 boundary_loops 在真实扫描上
        # 产出跨 14 mm 退化环的原因）。结果仍返回，但必须标记待复核。
        status = MarginStatus.REVIEW_REQUIRED
        reasons.append(f"界面缝含非二度顶点：{dict(sorted(degrees.items()))}")
    if best.azimuth_coverage < MIN_AZIMUTH_COVERAGE:
        status = MarginStatus.REVIEW_REQUIRED
        reasons.append(f"方位角覆盖不足（{best.azimuth_coverage:.2f} < {MIN_AZIMUTH_COVERAGE}）")
    if len(usable) > 1:
        reasons.append(f"存在 {len(usable)} 条通过的候选环，已按包围比例与周长选分"
                       f"（选定 #{best.index}, 其余 {[c.index for c in usable if c is not best]}）")
    diagnostics["reasons"] = reasons
    diagnostics["selected_candidate_index"] = best.index

    confidence = float(np.clip(
        0.45 * best.enclosure_fraction
        + 0.25 * best.azimuth_coverage
        + 0.30 * (1.0 - min(ambiguous_fraction / MAX_AMBIGUOUS_FRACTION, 1.0)),
        0.0, 1.0))

    return MarginExtractionResult(
        points=pts.tolist(),
        edge_ids=edge_ids,
        is_closed=best.is_closed,
        is_simple=best.is_simple,
        perimeter_mm=best.perimeter_mm,
        spacing_cv=best.spacing_cv,
        azimuth_coverage=best.azimuth_coverage,
        support_distance_stats={
            "median_mm": round(best.support_distance_median_mm, 6),
            "max_mm": round(best.support_distance_max_mm, 6),
        },
        confidence=confidence,
        status=status,
        diagnostics={
            **diagnostics,
            "best_fit_plane": {
                "_note": "仅诊断。颈缘是三维曲线，不要压平到该平面。",
                "origin": [float(v) for v in center],
                "normal": [float(v) for v in normal],
                "residual_mean_mm": round(float(plane_residual.mean()), 6),
                "residual_max_mm": round(float(plane_residual.max()), 6),
            },
        },
    )
