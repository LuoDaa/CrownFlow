"""提交 2.1 —— 弓段 / 整颌资产审计（Arch Asset Audit）。

**这一版只做资产分类与诊断**：不碰 Target Locator，不碰任何牙冠生成算法，
也不改 GT Set 规则（那要等 2.2 拿到真实分布之后）。

## 为什么需要它

提交 2 用 ``bbox.max_extent >= ARCH_MIN_MM(40)`` 判"整颌尺度网格"，于是 672 个
候选里混进了多少真整颌、多少 sectioned modelbase、多少只是 3–4 单位长桥被尺寸
阈值误判 —— **未知**。而 ``LocalizationGTSet`` 有没有训练集，完全取决于这里面
到底有多少真·未分割整颌。所以先把"承重假设"验死，再谈后续路线。

一座长桥也能达到 40–50 mm，所以尺度单指标不够。这里给每个 STL 提取一组
**不依赖世界坐标**的几何描述（主方向展布、面积、连通分量…），并额外做中心邻域
占据度诊断；分类结论必须带 ``classification_basis``（为什么被归到这一类），
免得日后没人说得清某条记录是怎么进 GT Set 的。

## 阈值：这一版刻意保守，且集中在一处

真正的阈值应当由 2.2 的全量分布决定，本版只提供**临时**阈值（见 ``THRESHOLDS``），
并把全部特征与分位数写进 ``arch_asset_summary.json``，让下一版用数据替换数字。
所有几何分类都带 ``classification_confidence``：离阈值越近越低。

## 元数据优先

CI 的 ``ScanFiles/ScanFile/PartType`` 是**真值**（实测取值 ``UnsegmentedJaw`` /
``Antagonist`` / ``BiteRegister`` / ``SituScan`` / ``MergedBridgeSLM``）。
所以角色判定先查元数据（``METADATA_*``），查不到才退回几何启发式
（``GEOMETRY_*``）—— 和修 bridge 时同一条原则：结构化真值优先，几何/文件名只作 fallback。

**注意**：真值里 ``UnsegmentedJaw`` / ``Antagonist`` 指向的是 ``.DCM`` 扫描，
而 DCM 需要 ``hpsdecode+numpy+trimesh``，数据服务器上**没有**（见 ``assets.py`` 开头）。
因此这里只把元数据当**标签**用（按文件名匹配），不对 DCM 做几何；STL 才是几何对象。

## 用法

    python -m app.groundtruth.arch_audit --root /data1/welo/dentistry/exo \\
        --out /data1/welo/gt_index

纯 stdlib（有测试断言 ``app/groundtruth`` 不 import numpy/trimesh）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .assets import (
    ARCH_MIN_MM,
    VertexGrid,
    CENTER_SURFACE_NEAR_MM,
    CENTER_SURFACE_VOID_MM,
    SINGLE_TOOTH_MAX_MM,
    assert_phi_free,
    content_hash,
    probe_stl,
    safe_case_id,
    safe_text_token,
)
from .indexer import CaseFiles, discover_cases

# ------------------------------------------------------------------ 阈值（临时）

#: **临时阈值**：数值来源写在这里，2.2 用全量分布替换。
#:
#: * ``SINGLE_TOOTH_MAX_MM`` / ``ARCH_MIN_MM``：沿用提交 1/2 在真实样本上量到的范围
#:   （单牙 7~17 mm、弓段 50~65 mm、整颌 61~87 mm）。
#: * ``ARCH_SECOND_PLANE_MIN_RATIO``：第二主方向展布 / 第一主方向展布。
#:   完整牙弓是"马蹄形"，两个平面方向都有明显展布；长桥则细长（比值小）。
#:   0.35 是**观测前的猜测**，只为把分布切开看，不作为临床规则。
#: * ``ARCH_THIRD_PLANE_MAX_RATIO``：第三主方向（厚度方向）应明显小于前两者。
#: * ``CONFIDENT_MARGIN``：离阈值 15% 以上算高置信，落在带内算低置信。
THRESHOLDS: dict[str, float] = {
    "single_tooth_max_mm": SINGLE_TOOTH_MAX_MM,
    "arch_min_mm": ARCH_MIN_MM,
    "arch_second_plane_min_ratio": 0.35,
    "arch_third_plane_max_ratio": 0.45,
    "confident_margin": 0.15,
}

#: 资产语义分类（比 sectioned/unsectioned/unknown 三态细，直接对应"能不能用"）。
ASSET_CLASSES: tuple[str, ...] = (
    "FULL_ARCH_WITH_TARGET",      # 完整牙弓 + 目标牙位置确有牙面 → 定位正样本
    "FULL_ARCH_TARGET_MISSING",   # 完整牙弓 + 目标位置是空洞（代型被抽走）
    "PARTIAL_ARCH",               # 弓段（一个方向为主）
    "MULTI_UNIT_RESTORATION",     # 多单位桥 / 连冠
    "SINGLE_TOOTH",               # 单牙
    "ANTAGONIST",                 # 元数据明确标为对颌
    "UNKNOWN",
)

#: 分类依据。``METADATA_*`` = 结构化真值，``GEOMETRY_*`` = 几何启发式。
BASES: tuple[str, ...] = (
    "METADATA_UNSEGMENTED_JAW",
    "METADATA_ANTAGONIST",
    "GEOMETRY_FULL_ARCH",
    "GEOMETRY_PARTIAL_ARCH",
    "GEOMETRY_MULTI_UNIT",
    "GEOMETRY_SINGLE_TOOTH",
    "UNKNOWN",
)

#: 元数据里出现的 PartType（实测）。
#: 对颌弓。**是完整弓，但目标牙在对侧颌上** —— 绝不能当 LocalizationGT 正样本。
#: 提交 2.1 里它曾被判成 FULL_ARCH_WITH_TARGET，虚增了 41 行 / 18 例。
FULL_ARCH_ANTAGONIST = "FULL_ARCH_ANTAGONIST"
PARTTYPE_UNSEGMENTED = "unsegmentedjaw"
PARTTYPE_ANTAGONIST = "antagonist"

#: 牙位侧别提示（用于把"上颌/下颌"这种信息留成标签，不参与判分）。
_JAW_HINTS = (("upper", "upper"), ("ober", "upper"), ("maxill", "upper"),
              ("lower", "lower"), ("unter", "lower"), ("mandib", "lower"))


# ------------------------------------------------------------------ 几何特征

@dataclass
class MeshFeatures:
    """一个 STL 的几何描述（不依赖世界坐标/朝向）。"""

    faces: int = 0
    vertices: int = 0
    bytes: int = 0
    extents: tuple[float, float, float] = (0.0, 0.0, 0.0)      # 已排序，升序
    principal: tuple[float, float, float] = (0.0, 0.0, 0.0)    # 已排序，升序（主方向展布）
    surface_area: float = 0.0
    bbox_volume: float = 0.0
    components: int | None = None          # 连通分量数（None = 点太多，没算）
    largest_component_ratio: float | None = None
    sampled_points: int = 0
    topology_skipped: bool = False
    ok: bool = True
    error: str | None = None

    @property
    def max_extent(self) -> float:
        return self.extents[-1]

    @property
    def second_plane_ratio(self) -> float:
        """第二主方向展布 / 第一主方向展布 —— 区分"弓形"与"细长条"的核心量。

        ``principal`` 按**升序**存，最大的是 ``[-1]``。
        """
        _small, middle, large = self.principal
        return (middle / large) if large > 0 else 0.0

    @property
    def third_plane_ratio(self) -> float:
        small, _middle, large = self.principal
        return (small / large) if large > 0 else 0.0


def _jacobi_eigen(m: list[list[float]], *, sweeps: int = 24) -> tuple[list[float], list[list[float]]]:
    """3×3 对称矩阵的特征值/特征向量（Jacobi 旋转）。纯 Python，无依赖。

    返回 ``(按降序排列的特征值, 对应的单位特征向量)``。
    """
    a = [row[:] for row in m]
    v = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(sweeps):
        off = abs(a[0][1]) + abs(a[0][2]) + abs(a[1][2])
        if off < 1e-12:
            break
        for p, q in ((0, 1), (0, 2), (1, 2)):
            if abs(a[p][q]) < 1e-15:
                continue
            theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
            t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            for k in range(3):
                akp, akq = a[k][p], a[k][q]
                a[k][p] = c * akp - s * akq
                a[k][q] = s * akp + c * akq
            for k in range(3):
                apk, aqk = a[p][k], a[q][k]
                a[p][k] = c * apk - s * aqk
                a[q][k] = s * apk + c * aqk
            for k in range(3):
                vkp, vkq = v[k][p], v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq
    pairs = sorted(((a[i][i], (v[0][i], v[1][i], v[2][i])) for i in range(3)),
                   key=lambda pair: -pair[0])
    return [p[0] for p in pairs], [list(p[1]) for p in pairs]


def _principal_extents(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """点云在三个主方向上的展布（max-min），升序返回。

    展布比方差更贴近"这个方向有多长"，也和包围盒直接可比。
    """
    if len(points) < 3:
        return (0.0, 0.0, 0.0)
    n = float(len(points))
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    cov = [[0.0] * 3 for _ in range(3)]
    for x, y, z in points:
        dx, dy, dz = x - cx, y - cy, z - cz
        cov[0][0] += dx * dx; cov[0][1] += dx * dy; cov[0][2] += dx * dz
        cov[1][1] += dy * dy; cov[1][2] += dy * dz; cov[2][2] += dz * dz
    cov[1][0], cov[2][0], cov[2][1] = cov[0][1], cov[0][2], cov[1][2]
    _values, axes = _jacobi_eigen(cov)
    spans = []
    for axis in axes:
        lo = math.inf
        hi = -math.inf
        for x, y, z in points:
            d = x * axis[0] + y * axis[1] + z * axis[2]
            if d < lo:
                lo = d
            if d > hi:
                hi = d
        spans.append(hi - lo)
    spans.sort()
    return (spans[0], spans[1], spans[2])


def _components(keys: list[tuple], parent: dict, rank: dict) -> tuple[int, float | None]:
    """并查集：返回 ``(连通分量数, 最大分量顶点占比)``。

    ``keys`` 是每个面的三个顶点键（已展平）。
    """
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for i in range(0, len(keys), 3):
        a, b, c = keys[i], keys[i + 1], keys[i + 2]
        union(a, b)
        union(b, c)
    counts: dict = {}
    for k in parent:
        r = find(k)
        counts[r] = counts.get(r, 0) + 1
    if not counts:
        return 0, None
    return len(counts), max(counts.values()) / len(parent)



def analyze_asset(path: str, *, max_points: int = 120_000, grid_cell: float = 1.0,
                   topology_budget: int = 400_000,
                   need_grid: bool = True) -> tuple[MeshFeatures, "VertexGrid | None"]:
    """**单趟读取**：一次读盘同时得到几何特征与顶点网格。

    实测真实整颌单文件 0.3M~0.76M 面，而"探包围盒 / 量特征 / 建网格"各读一遍就把
    单资产耗时抬到 4.5 秒 —— 全量跑不起。这里把包围盒、面积、抽样主方向展布、
    中心邻域用的顶点网格合并成一趟；连通分量按面数预算决定做不做
    （纯 Python 并查集在 0.3M 面上约 4 秒，超预算就标记跳过，而不是装作算过）。

    返回值：``(MeshFeatures, VertexGrid | None)``。
    """
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return MeshFeatures(ok=False, error=f"stat_failed:{exc.errno}"), None
    if size < 84:
        return MeshFeatures(bytes=size, ok=False, error="too_small"), None
    try:
        with open(path, "rb") as f:
            head = f.read(84)
            if head[:5] == b"solid":
                chunk = head + f.read(2048 - 84)
                printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in chunk)
                if printable / max(len(chunk), 1) > 0.95:
                    return MeshFeatures(bytes=size, ok=False, error="ascii_stl"), None
            n = struct.unpack("<I", head[80:84])[0]
            if 84 + n * 50 != size:
                n = max(0, (size - 84) // 50)
            buf = f.read(n * 50)
    except OSError as exc:
        return MeshFeatures(ok=False, error=f"read_failed:{exc.errno}"), None

    stride = max(1, (n * 3) // max_points) if n * 3 > max_points else 1
    points: list[tuple[float, float, float]] = []
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    area = 0.0
    keys: list[tuple] = []
    parent: dict = {}
    rank: dict = {}
    want_topology = n <= topology_budget
    for i in range(n):
        try:
            v = struct.unpack_from("<12f", buf, i * 50)
        except struct.error:
            break
        a = (v[3], v[4], v[5])
        b = (v[6], v[7], v[8])
        c = (v[9], v[10], v[11])
        for p in (a, b, c):
            for k in range(3):
                x = p[k]
                if x < lo[k]:
                    lo[k] = x
                if x > hi[k]:
                    hi[k] = x
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        area += 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)      # |u×v|/2
        if i % stride == 0:
            points.append(a); points.append(b); points.append(c)
        if want_topology:
            for p in (a, b, c):
                key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
                if key not in parent:
                    parent[key] = key
                    rank[key] = 0
                keys.append(key)
    if not (math.isfinite(lo[0]) and math.isfinite(hi[0])):
        return MeshFeatures(bytes=size, ok=False, error="no_vertices"), None

    feats = MeshFeatures(faces=n, bytes=size,
                         extents=tuple(sorted(hi[k] - lo[k] for k in range(3))))
    feats.bbox_volume = math.prod(feats.extents)
    feats.surface_area = area
    feats.sampled_points = len(points)
    feats.principal = _principal_extents(points)
    feats.vertices = len(parent)
    if want_topology and parent:
        feats.components, feats.largest_component_ratio = _components(keys, parent, rank)
    else:
        feats.topology_skipped = True

    grid = None
    if need_grid and points:
        base = (min(p[0] for p in points), min(p[1] for p in points),
                min(p[2] for p in points))
        buckets: dict[tuple[int, int, int], list] = {}
        for p in points:
            key = (int(math.floor((p[0] - base[0]) / grid_cell)),
                   int(math.floor((p[1] - base[1]) / grid_cell)),
                   int(math.floor((p[2] - base[2]) / grid_cell)))
            buckets.setdefault(key, []).append(p)
        grid = VertexGrid(cell=grid_cell, lo=base, buckets=buckets, n_points=len(points))
    return feats, grid


def mesh_features(path: str, *, max_points: int = 40_000,
                  topology_budget: int = 400_000) -> MeshFeatures:
    """只要特征、不要网格的便捷入口（测试与分布观察用）。"""
    feats, _grid = analyze_asset(path, max_points=max_points,
                                 topology_budget=topology_budget, need_grid=False)
    return feats


@dataclass
class CenterFeatures:
    """目标牙 Center 周围的表面占据情况（诊断用，不是最终分类）。"""

    dist_min: float | None = None
    dist_p10: float | None = None
    dist_median: float | None = None
    count_r2: int = 0
    count_r4: int = 0
    count_r6: int = 0
    coverage: float | None = None
    saturated: bool = False
    radius_mm: float = 12.0

    @property
    def occupancy_class(self) -> str:
        """三态粗判：预备体还在 → 中心附近有密集表面；代型被抽走 → 中心空。

        单看最近距离不稳（中心可能挨着洞边），所以同时看"半径内有几个顶点"。
        """
        if self.dist_min is None:
            return "unknown"
        if self.saturated:
            return "void"                      # 搜索半径内一个顶点都没有 → 中心周围是空的
        if self.dist_min <= CENTER_SURFACE_NEAR_MM and self.count_r4 > 0:
            return "surface_present"           # 中心附近就有面，而且密
        if self.dist_min >= CENTER_SURFACE_VOID_MM:
            return "void"                      # 最近的表面也远在"空洞"阈值之外
        return "ambiguous"                     # 3.5~6 mm 之间：宁可说不知道


def center_features(grid, center, *, radius_mm: float = 12.0,
                    search_radius_mm: float = CENTER_SURFACE_VOID_MM * 2.5,
                    coverage_radius_mm: float = 6.0) -> CenterFeatures:
    """以 Center 为中心统计邻近表面点分布。

    单看"最近距离"不够：预备体还在与"代型被抽走但中心挨着洞边"都可能很近。
    所以同时给出距离分位数与若干半径内的顶点计数 —— 这两种分布是可分的。
    """
    feats = CenterFeatures(radius_mm=radius_mm)
    if grid is None or center is None:
        return feats
    dist_min, found = grid.nearest_distance(tuple(center), search_radius_mm=search_radius_mm)
    feats.saturated = not found
    feats.dist_min = dist_min
    feats.coverage = grid.coverage(tuple(center), coverage_radius_mm)

    ds: list[float] = []
    for bucket in grid.buckets.values():
        for p in bucket:
            d = math.dist(center, p)
            if d <= radius_mm:
                ds.append(d)
                if d <= 2.0:
                    feats.count_r2 += 1
                if d <= 4.0:
                    feats.count_r4 += 1
                if d <= 6.0:
                    feats.count_r6 += 1
    if ds:
        ds.sort()
        feats.dist_median = statistics.median(ds)
        feats.dist_p10 = ds[min(len(ds) - 1, int(round(0.10 * (len(ds) - 1))))]
    return feats


# ------------------------------------------------------------------ 元数据

@dataclass
class CaseMetadata:
    """来自 constructionInfo 的结构化真值（只当标签，不做几何）。

    ``scan_labels`` 里的 ``ref`` 是**脱敏后的**文件引用 token（原文件名可能含患者
    姓名，绝不写进索引）；``side`` 是从原名里抽出的上/下颌提示，ASCII 字符串。
    """

    role_by_file: dict[str, str] = field(default_factory=dict)   # 归一化文件名 -> PartType
    scan_labels: list[dict] = field(default_factory=list)        # [{'role','ref','side'}]
    n_targets: int = 0                                           # 有 Center 的牙位数

    def labels(self, part_type: str) -> list[dict]:
        low = part_type.strip().lower()
        return [x for x in self.scan_labels if x.get("role", "").lower() == low]

    @property
    def has_unsegmented_jaw_label(self) -> bool:
        return bool(self.labels(PARTTYPE_UNSEGMENTED))

    @property
    def has_antagonist_label(self) -> bool:
        return bool(self.labels(PARTTYPE_ANTAGONIST))


def _norm_name(name: str | None) -> str:
    """归一化文件名用于匹配：小写、去掉扩展名、把空格压成一个。"""
    if not name:
        return ""
    base = os.path.basename(name.strip()).lower()
    stem, _dot, _ext = base.rpartition(".")
    return (stem or base).replace("  ", " ")


def _jaw_side(name: str | None) -> str | None:
    low = (name or "").lower()
    for token, side in _JAW_HINTS:
        if token in low:
            return side
    return None


def case_metadata(case) -> CaseMetadata:
    """从已解析的 CI 病例里取出角色真值。"""
    meta = CaseMetadata()
    for sf in getattr(case, "scan_files", ()) or ():
        name = getattr(sf, "file_name", None)
        role = (getattr(sf, "part_type", None) or "").strip()
        key = _norm_name(name)
        if key and role:
            meta.role_by_file[key] = role
        if role:
            meta.scan_labels.append({
                "role": role,
                "ref": safe_text_token(name, prefix="scan") or "",
                "side": _jaw_side(name) or "",
            })
    meta.n_targets = sum(1 for t in (getattr(case, "teeth", ()) or ())
                         if getattr(t, "center", None) is not None)
    return meta



def role_for_asset(meta: CaseMetadata, stl_path: str) -> str | None:
    """这个 STL 在 CI 元数据里是什么角色（``None`` = 元数据里没有它）。

    先按完整文件名匹配，再退到"同 stem 换扩展名"——实测整颌/对颌的
    ``ScanFile`` 指向 ``.DCM``，而同一病例里常有同 stem 的转换版 STL。
    """
    base = _norm_name(stl_path)
    if not base:
        return None
    if base in meta.role_by_file:
        return meta.role_by_file[base]
    for key, role in meta.role_by_file.items():
        if key and (key.startswith(base) or base.startswith(key)):
            return role
    return None


# ------------------------------------------------------------------ 分类

@dataclass
class Classification:
    asset_class: str
    basis: str
    confidence: str          # high / medium / low
    why: str                 # 中文解释：给人看（JSON / 日志），不写进 CSV
    evidence: str = ""       # ASCII 证据串：只含 token 与数字，写进 CSV（PHI 闸要求 ASCII）


def _fdi_jaw(fdi) -> str | None:
    """FDI 首位定颌：1/2 上颌，3/4 下颌。"""
    try:
        q = int(fdi) // 10
    except (TypeError, ValueError):
        return None
    if q in (1, 2):
        return "upper"
    if q in (3, 4):
        return "lower"
    return None


def classify_asset(feats: MeshFeatures, *, role: str | None,
                   center: CenterFeatures | None, threshold: dict | None = None,
                   asset_jaw: str | None = None,
                   target_fdi=None) -> Classification:
    """资产语义分类。元数据优先，几何兜底，结论必须带依据与置信度。

    ``asset_jaw`` / ``target_fdi`` 用于**严格准入**：``UnsegmentedJaw`` 按定义
    是"目标牙所在的那一颌"。若该资产的颌别与目标 FDI 的颌别不一致，
    说明角色与资产配错了（或它其实是对颌），此时**降级为
    ``FULL_ARCH_ANTAGONIST``**，不给 target-present 结论。

    这是把"6 个候选"当关键训练集冻结之前必须有的闸门：
    数量已经小到个位数，任何一个假正样本都会显著污染它。
    """
    t = dict(THRESHOLDS)
    t.update(threshold or {})
    if not feats.ok:
        return Classification("UNKNOWN", "UNKNOWN", "low",
                              feats.error or "几何读取失败", "probe_failed")

    role_token = safe_text_token(role, prefix="pt") or ""
    role_low = (role or "").strip().lower()
    if role_low == PARTTYPE_ANTAGONIST:
        # 对颌弓是完整弓，但**目标牙在对侧颌**。判成 target-present 是语义错误：
        # 直接虚增 LocalizationGT 正样本（提交 2.1 实测虚增了 41/47 行）。
        return Classification(FULL_ARCH_ANTAGONIST, "METADATA_ANTAGONIST", "high",
                              "CI ScanFiles.PartType=Antagonist：对颌弓，"
                              "目标牙在对侧颌，不是 target-present",
                              f"part_type={role_token}")
    ext = feats.max_extent
    center_token = center.occupancy_class if center is not None else "no_center"

    if role_low == PARTTYPE_UNSEGMENTED:
        why = "CI ScanFiles.PartType=UnsegmentedJaw"
        ev = (f"part_type={role_token};center={center_token};"
              f"second_ratio={feats.second_plane_ratio:.2f}")
        # 严格准入 1：颌别必须一致。不一致 -> 这是对颌，不是 target-present。
        want_jaw = _fdi_jaw(target_fdi)
        if want_jaw is not None and asset_jaw is not None and asset_jaw != want_jaw:
            return Classification(
                FULL_ARCH_ANTAGONIST, "METADATA_UNSEGMENTED_JAW", "high",
                why + f"：但资产颌别 {asset_jaw} 与目标 FDI {target_fdi} 的 "
                      f"{want_jaw} 不一致，判为对颌弓",
                ev + f";jaw_mismatch={asset_jaw}!={want_jaw}")
        # 严格准入 2：必须真的查到 Center 邻域，且邻域里确实有表面。
        # 提交 2.1 曾在"没有 Center"时按几何展布猜 target-present —— 那是猜测，
        # 现在一律 UNKNOWN。个位数级的集合上不允许猜。
        if center is None:
            return Classification("UNKNOWN", "METADATA_UNSEGMENTED_JAW", "low",
                                  why + "：该病例没有 Center，无法验证目标牙在不在",
                                  ev + ";no_center")
        if center.occupancy_class == "surface_present":
            return Classification("FULL_ARCH_WITH_TARGET",
                                  "METADATA_UNSEGMENTED_JAW", "high",
                                  why + "：目标 Center 邻域有表面",
                                  ev + f";occupancy={center.occupancy_class}")
        if center.occupancy_class == "void":
            return Classification("FULL_ARCH_TARGET_MISSING",
                                  "METADATA_UNSEGMENTED_JAW", "high",
                                  why + "：目标 Center 邻域是空的",
                                  ev + f";occupancy={center.occupancy_class}")
        return Classification("UNKNOWN", "METADATA_UNSEGMENTED_JAW", "low",
                              why + f"：Center 邻域判定为 {center.occupancy_class}",
                              ev + f";occupancy={center.occupancy_class}")

    if ext <= 0:
        return Classification("UNKNOWN", "UNKNOWN", "low", "包围盒为空", "empty_bbox")
    margin = t["confident_margin"]
    arch_scale = ext >= t["arch_min_mm"]
    near_single = ext <= t["single_tooth_max_mm"]
    second = feats.second_plane_ratio
    third = feats.third_plane_ratio
    arch_like = second >= t["arch_second_plane_min_ratio"] and \
        third <= t["arch_third_plane_max_ratio"]
    evidence = (f"max_extent={ext:.1f};second_ratio={second:.2f};third_ratio={third:.2f};"
                f"center={center_token};arch_scale={int(arch_scale)};arch_like={int(arch_like)}")

    if near_single:
        conf = "high" if ext <= t["single_tooth_max_mm"] * (1 - margin) else "low"
        return Classification("SINGLE_TOOTH", "GEOMETRY_SINGLE_TOOTH", conf,
                              f"最长边 {ext:.1f} mm ≤ 单牙上限 {t['single_tooth_max_mm']:.0f}",
                              evidence)

    if arch_scale and arch_like:
        cls, why = "UNKNOWN", "尺寸达弓形尺度且两个平面方向都有展布"
        if center is None:
            why += "；但缺 Center（无法判目标在不在）"
        elif center.occupancy_class == "surface_present":
            cls = "FULL_ARCH_WITH_TARGET"; why += "；中心附近有密集表面"
        elif center.occupancy_class == "void":
            cls = "FULL_ARCH_TARGET_MISSING"; why += "；中心是空洞"
        else:
            why += "；中心邻域不明确"
        conf = "high" if second >= t["arch_second_plane_min_ratio"] * (1 + margin) else "low"
        return Classification(cls, "GEOMETRY_FULL_ARCH", conf, why, evidence)

    if arch_scale or not arch_like:
        why = (f"最长边 {ext:.1f} mm，第二主方向/第一主方向 {second:.2f}"
               f"（< {t['arch_second_plane_min_ratio']}）→ 不是马蹄形")
        conf = "high" if second <= t["arch_second_plane_min_ratio"] * (1 - margin) else "low"
        return Classification("PARTIAL_ARCH" if arch_scale else "MULTI_UNIT_RESTORATION",
                              "GEOMETRY_PARTIAL_ARCH" if arch_scale else "GEOMETRY_MULTI_UNIT",
                              conf, why, evidence)

    return Classification("UNKNOWN", "UNKNOWN", "low",
                          f"最长边 {ext:.1f} mm，形态不在已知区间", evidence)


# ------------------------------------------------------------------ 审计主流程

#: 审计表列顺序（也是 CSV 表头）。只追加，不改名 —— 与 ``models.COLUMNS`` 同一约定。
AUDIT_COLUMNS: tuple[str, ...] = (
    "case_uuid",
    "source_case_id",
    "asset_id",
    "asset_kind",
    "metadata_role",
    "candidate_class",
    "classification_basis",
    "classification_confidence",
    "classification_evidence",
    "triangle_count",
    "size_bytes",
    "bbox_extent_1",
    "bbox_extent_2",
    "bbox_extent_3",
    "principal_extent_1",
    "principal_extent_2",
    "principal_extent_3",
    "second_plane_ratio",
    "third_plane_ratio",
    "surface_area",
    "bbox_volume",
    "components",
    "largest_component_ratio",
    "topology_skipped",
    "target_fdi",
    "has_center",
    "n_case_targets",
    "target_surface_distance",
    "near_dist_min",
    "near_dist_p10",
    "near_dist_median",
    "near_count_r2",
    "near_count_r4",
    "near_count_r6",
    "near_coverage",
    "near_occupancy",
    "near_radius_mm",
    "case_has_metadata_unsegmented_jaw",
    "case_unsegmented_jaw_side",
    "case_has_metadata_antagonist",
    "antagonist_asset_id",
    "antagonist_basis",
    "antagonist_metadata_ref",
    "antagonist_metadata_side",
)

#: 对颌判定依据。
ANTAGONIST_BASES = ("METADATA_ANTAGONIST", "METADATA_ANTAGONIST_REFERENCE",
                    "GEOMETRY_FARTHEST_ARCH", "NONE")


def _round(value, digits: int = 3):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return round(value, digits)
    return value


def _asset_id(path: str, do_hash: bool) -> str:
    if do_hash:
        h = content_hash(path)
        if h:
            return h
    return "sha:" + hashlib.sha256(
        os.path.basename(path).encode("utf-8")).hexdigest()[:16]


def _pick_target(grid, probes_center, targets):
    """选出这个资产"最可能承载"的目标牙位：到资产表面距离最小的那个 Center。

    返回 ``(fdi, center, distance, found)``。
    """
    best = None
    for fdi, center in targets:
        if grid is not None:
            d, found = grid.nearest_distance(tuple(center),
                                            search_radius_mm=CENTER_SURFACE_VOID_MM * 2.5)
        else:
            d, found = math.dist(center, probes_center), True
        if best is None or d < best[2]:
            best = (fdi, center, d, found)
    return best


def resolve_antagonist(meta: CaseMetadata, stl_paths, *, probes, arch_candidates,
                       target_center, do_hash: bool) -> tuple[str | None, str, str, str]:
    """对颌：**元数据优先**，没有元数据才用几何。

    返回 ``(asset_id|None, basis, meta_ref, meta_side)``。
    元数据里对颌是 ``.DCM``（服务器读不了），所以常常只能给出**引用**而不能给出
    几何资产 —— 那也比"拿最远的弓候选顶替"可靠。
    """
    for path in stl_paths:
        role = role_for_asset(meta, path)
        if role and role.strip().lower() == PARTTYPE_ANTAGONIST:
            return _asset_id(path, do_hash), "METADATA_ANTAGONIST", "", _jaw_side(path) or ""
    labels = meta.labels(PARTTYPE_ANTAGONIST)
    if labels:
        return None, "METADATA_ANTAGONIST_REFERENCE", labels[0].get("ref", ""), \
            labels[0].get("side", "")
    if len(arch_candidates) >= 2 and target_center is not None:
        far = max(arch_candidates, key=lambda p: math.dist(probes[p].center, target_center))
        return _asset_id(far, do_hash), "GEOMETRY_FARTHEST_ARCH", "", ""
    return None, "NONE", "", ""


def audit_cases(cases: dict[str, CaseFiles], *, max_assets_per_case: int = 6,
                all_assets: bool = False, do_hash: bool = True, topology: bool = True,
                threshold: dict | None = None, progress=None) -> list[dict]:
    """对每个病例产出若干行（一行一个候选 STL 资产）。"""
    rows: list[dict] = []
    for n, (cid, files) in enumerate(sorted(cases.items()), 1):
        if not files.has_construction_info:
            continue
        if progress and n % 25 == 0:
            progress(f"  ...已审计 {n}/{len(cases)} 个病例，当前资产行数 {len(rows)}")
        try:
            rows.extend(_audit_case(cid, files, max_assets_per_case=max_assets_per_case,
                                    all_assets=all_assets, do_hash=do_hash,
                                    topology=topology, threshold=threshold))
        except Exception as exc:                      # 单个病例失败不拖垮全量
            rows.append({
                "case_uuid": _case_uuid(cid), "source_case_id": safe_case_id(cid),
                "asset_id": "", "asset_kind": "STL",
                "candidate_class": "UNKNOWN", "classification_basis": "UNKNOWN",
                "classification_confidence": "low",
                "classification_evidence": f"case_failed={type(exc).__name__}",
                "case_has_metadata_unsegmented_jaw": 0,
                "case_has_metadata_antagonist": 0,
                "antagonist_basis": "NONE",
            })
    return rows


def _case_uuid(cid: str) -> str:
    return "cf_" + hashlib.sha256(cid.encode("utf-8")).hexdigest()[:16]


def _audit_case(cid: str, files: CaseFiles, *, max_assets_per_case: int, all_assets: bool,
                do_hash: bool, topology: bool, threshold: dict | None) -> list[dict]:
    from ..importers.exocad.construction_info import parse_construction_case

    case = parse_construction_case(files.construction_infos[0])
    meta = case_metadata(case)
    case_uuid = _case_uuid(cid)
    safe_id = safe_case_id(cid)
    targets = [(t.tooth_fdi, tuple(t.center)) for t in (case.teeth or ())
               if t.tooth_fdi is not None and t.center is not None and len(t.center) == 3]

    probes = {}
    for p in files.stls:
        try:
            pr = probe_stl(p)
        except Exception:
            continue
        if pr.ok and pr.faces > 0:
            probes[p] = pr
    if not probes:
        return []

    # 候选 = 弓段尺度以上；--all-assets 时连单牙也一起量（供分布观察）
    archs = sorted((p for p, pr in probes.items()
                    if pr.max_extent_mm >= ARCH_MIN_MM), key=lambda p: -probes[p].bytes)
    singles = sorted((p for p, pr in probes.items()
                      if pr.max_extent_mm < ARCH_MIN_MM), key=lambda p: -probes[p].bytes)
    candidates = archs[:max_assets_per_case] if not all_assets \
        else (archs + singles)[:max_assets_per_case]

    unseg_side = (meta.labels(PARTTYPE_UNSEGMENTED) or [{}])[0].get("side", "")
    rows: list[dict] = []
    for path in candidates:
        feats, grid = analyze_asset(
            path, max_points=120_000,
            topology_budget=400_000 if topology else -1,
            need_grid=bool(targets))
        picked = _pick_target(grid, probes[path].center, targets) if targets else None
        target_fdi, target_center, target_d, _found = picked if picked else ("", None, None, False)
        center = center_features(grid, target_center) if (grid is not None and target_center) else None
        role = role_for_asset(meta, path)
        cls = classify_asset(feats, role=role, center=center,
                             threshold=threshold,
                             asset_jaw=_jaw_side(path),
                             target_fdi=target_fdi)
        arch_asset, ant_basis, ant_ref, ant_side = resolve_antagonist(
            meta, list(probes), probes=probes, arch_candidates=archs,
            target_center=target_center, do_hash=do_hash)

        ext = list(feats.extents) + [0.0] * 3
        prin = list(feats.principal) + [0.0] * 3
        rows.append({
            "case_uuid": case_uuid,
            "source_case_id": safe_id,
            "asset_id": _asset_id(path, do_hash),
            "asset_kind": "STL",
            "metadata_role": safe_text_token(role, prefix="pt") or "",
            "candidate_class": cls.asset_class,
            "classification_basis": cls.basis,
            "classification_confidence": cls.confidence,
            "classification_evidence": cls.evidence,
            "triangle_count": feats.faces,
            "size_bytes": feats.bytes,
            "bbox_extent_1": _round(ext[0], 2),
            "bbox_extent_2": _round(ext[1], 2),
            "bbox_extent_3": _round(ext[2], 2),
            "principal_extent_1": _round(prin[0], 2),
            "principal_extent_2": _round(prin[1], 2),
            "principal_extent_3": _round(prin[2], 2),
            "second_plane_ratio": _round(feats.second_plane_ratio),
            "third_plane_ratio": _round(feats.third_plane_ratio),
            "surface_area": _round(feats.surface_area, 1),
            "bbox_volume": _round(feats.bbox_volume, 1),
            "components": _round(feats.components),
            "largest_component_ratio": _round(feats.largest_component_ratio),
            "topology_skipped": int(feats.topology_skipped),
            "target_fdi": target_fdi if target_fdi != "" else "",
            "has_center": int(target_center is not None),
            "n_case_targets": meta.n_targets,
            "target_surface_distance": _round(target_d, 3),
            "near_dist_min": _round(center.dist_min, 3) if center else "",
            "near_dist_p10": _round(center.dist_p10, 3) if center else "",
            "near_dist_median": _round(center.dist_median, 3) if center else "",
            "near_count_r2": center.count_r2 if center else "",
            "near_count_r4": center.count_r4 if center else "",
            "near_count_r6": center.count_r6 if center else "",
            "near_coverage": _round(center.coverage, 3) if center else "",
            "near_occupancy": center.occupancy_class if center else "",
            "near_radius_mm": center.radius_mm if center else "",
            "case_has_metadata_unsegmented_jaw": int(meta.has_unsegmented_jaw_label),
            "case_unsegmented_jaw_side": unseg_side,
            "case_has_metadata_antagonist": int(meta.has_antagonist_label),
            "antagonist_asset_id": arch_asset or "",
            "antagonist_basis": ant_basis,
            "antagonist_metadata_ref": ant_ref,
            "antagonist_metadata_side": ant_side,
        })
    return rows


# ------------------------------------------------------------------ 输出

def _quantiles(values: list[float]) -> dict:
    clean = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if not clean:
        return {}
    def pick(q: float) -> float:
        return round(clean[min(len(clean) - 1, int(round(q * (len(clean) - 1))))], 3)
    return {"n": len(clean), "p05": pick(0.05), "p25": pick(0.25), "p50": pick(0.50),
            "p75": pick(0.75), "p95": pick(0.95), "max": round(clean[-1], 3)}


def build_summary(rows: list[dict]) -> dict:
    """给 2.2 用：让数据决定阈值。"""
    def counts(key: str) -> dict:
        out: dict = {}
        for r in rows:
            v = r.get(key, "")
            out[str(v)] = out.get(str(v), 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    per_class: dict = {}
    for r in rows:
        cls = r.get("candidate_class", "UNKNOWN")
        per_class.setdefault(cls, []).append(r)

    def class_stats(items: list[dict]) -> dict:
        return {
            "n": len(items),
            "max_extent": _quantiles([r.get("bbox_extent_3") for r in items]),
            "second_plane_ratio": _quantiles([r.get("second_plane_ratio") for r in items]),
            "third_plane_ratio": _quantiles([r.get("third_plane_ratio") for r in items]),
            "surface_area": _quantiles([r.get("surface_area") for r in items]),
            "largest_component_ratio": _quantiles(
                [r.get("largest_component_ratio") for r in items]),
        }

    # 阈值扫描：换一个第二主方向比阈值，各类会有多少个 —— 供 2.2 挑阈值用
    sweep = []
    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        arch_like = sum(1 for r in rows
                        if isinstance(r.get("second_plane_ratio"), (int, float))
                        and isinstance(r.get("bbox_extent_3"), (int, float))
                        and r["bbox_extent_3"] >= THRESHOLDS["arch_min_mm"]
                        and r["second_plane_ratio"] >= threshold)
        sweep.append({"second_plane_min_ratio": threshold,
                      "arch_scale_and_arch_like": arch_like,
                      "arch_scale_total": sum(
                          1 for r in rows
                          if isinstance(r.get("bbox_extent_3"), (int, float))
                          and r["bbox_extent_3"] >= THRESHOLDS["arch_min_mm"])})

    return {
        "n_assets": len(rows),
        "n_cases": len({r.get("case_uuid") for r in rows}),
        "class_counts": counts("candidate_class"),
        "basis_counts": counts("classification_basis"),
        "confidence_counts": counts("classification_confidence"),
        "occupancy_counts": counts("near_occupancy"),
        "metadata_role_counts": counts("metadata_role"),
        "asset_kind_counts": counts("asset_kind"),
        "antagonist_basis_counts": counts("antagonist_basis"),
        "cases_with_metadata_unsegmented_jaw": len({
            r.get("case_uuid") for r in rows
            if r.get("case_has_metadata_unsegmented_jaw") == 1}),
        "cases_with_metadata_antagonist": len({
            r.get("case_uuid") for r in rows
            if r.get("case_has_metadata_antagonist") == 1}),
        "class_case_counts": {
            name: len({r.get("case_uuid") for r in items})
            for name, items in per_class.items()},
        "n_cases_with_any_audited_asset": len({r.get("case_uuid") for r in rows}),
        "topology_skipped": sum(1 for r in rows if r.get("topology_skipped") == 1),
        "per_class": {name: class_stats(items) for name, items in per_class.items()},
        "all_assets_feature_quantiles": class_stats(rows),
        "arch_min_mm": THRESHOLDS["arch_min_mm"],
        "threshold_sweep": sweep,
        "provisional_thresholds": THRESHOLDS,
        "note": ("provisional thresholds only - freeze them from threshold_sweep + "
                 "per_class quantiles in submission 2.2; do not treat as clinical rules"),
    }


def write_outputs(rows: list[dict], out_dir: str | Path) -> dict:
    """写 ``arch_asset_audit.csv`` + ``arch_asset_summary.json``（写前过 PHI 闸）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    flat = [{k: r.get(k, "") for k in AUDIT_COLUMNS} for r in rows]
    assert_phi_free(flat)
    with (out / "arch_asset_audit.csv").open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(AUDIT_COLUMNS))
        w.writeheader()
        w.writerows(flat)
    summary = build_summary(flat)
    (out / "arch_asset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2), encoding="ascii")
    return summary


# ------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WelooAI Arch Asset Audit（提交 2.1）")
    ap.add_argument("--root", action="append", required=True,
                    help="病例根目录，可重复（如 exo/EXO 与 exo/all_data）")
    ap.add_argument("--out", required=True, help="输出目录（不要写在只读数据目录里）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个病例（调试）")
    ap.add_argument("--max-assets-per-case", type=int, default=6,
                    help="每个病例最多审计几个资产（按体积从大到小）")
    ap.add_argument("--all-assets", action="store_true",
                    help="连单牙尺度的 STL 也一起量（观察分布用，更慢）")
    ap.add_argument("--no-hash", action="store_true", help="跳过内容哈希（更快，但不可追溯）")
    ap.add_argument("--no-topology", action="store_true",
                    help="跳过连通分量（大整颌上快很多，components 会是空）")
    a = ap.parse_args(argv)

    print(f"扫描 {len(a.root)} 个根目录 ...", flush=True)
    cases = discover_cases(a.root)
    with_ci = {k: v for k, v in cases.items() if v.has_construction_info}
    print(f"  病例 id 总数 {len(cases)}，其中有 constructionInfo 的 {len(with_ci)}", flush=True)
    if a.limit:
        with_ci = dict(sorted(with_ci.items())[:a.limit])
        print(f"  --limit 生效，只处理 {len(with_ci)} 个", flush=True)

    rows = audit_cases(with_ci, max_assets_per_case=a.max_assets_per_case,
                       all_assets=a.all_assets, do_hash=not a.no_hash,
                       topology=not a.no_topology,
                       progress=lambda m: print(m, flush=True))
    summary = write_outputs(rows, a.out)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("per_class")},
                     ensure_ascii=True, indent=2))
    print(f"\n写出到 {a.out}/arch_asset_audit.csv")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())

