"""Intaglio 链路：方向 blockout → 变量 gap → cutter。

拆成两段（对应两次提交）：

    A. support → directional blockout geometry   （:func:`build_blockout`）
    B. blockout → variable gap → cutter          （:func:`apply_gap_field` / :func:`build_intaglio_cutter`）

这样若最终 clearance 改善，还能分辨是哪个环节贡献的。

---

## 为什么旧实现是错的

旧 ``build_blockout_cutter`` 把 support 体素化后，**对每个 XY 列从底部填充到该列最高
占用 voxel**。它填的是**整个上部 support 的投影**，不是局部截面。实测在 FDI36 基台上：

| 高度 | 基台截面 | cutter 截面 | 倍数 |
|---|---|---|---|
| z=5.0 | 7.10 mm² | 34.98 mm² | **5×** |
| z=6.0 | 17.15 mm² | 35.20 mm² | **2×** |
| z=7.0 | 29.99 mm² | 33.49 mm² | 1.1× |
| z=8.0 | 29.69 mm² | 31.35 mm² | 1.1× |

即边缘附近被撑大 2–5 倍，颈部缩窄的 emergence profile 被抹掉。实测内腔间隙
（边缘带内中位）**0.523 mm**，而目标量级是 cement gap（0.02–0.06 mm）。

---

## A. 方向阴影体（当前实现）

把 insertion axis 设为局部 +Z，在 axis frame 内构造**方向阴影体**：

```
Support  →  投影到 XY  →  从 +Z 向 -Z 求 first hit  →  h(x,y)

阴影体 B = { (x,y,z) | (x,y) ∈ covered,  z_bottom ≤ z ≤ h(x,y) }
```

几何语义：每个 XY 上只保留从戴入方向**最先碰到**的表面；隐藏在它后面的凹陷
自动被阴影体填掉。所以 B 沿就位方向是单值、无隐藏倒凹的 ——
这正是"可无碰撞插入"的定义。

**这不是 3D voxelization。** 只在 2D 上做规则采样得到高度场，最终输出是
三角化的水密网格，再由 ``manifold3d`` 做精确 Boolean。2D 采样 ≠ 3D 体素化。

掠射边界用保守栅格化处理：每个 cell 取 2×2 子采样，cell 高度取子采样命中的
**最大值**（保守 = 包络更高），并把覆盖掩码**向外膨胀一个 cell** 后以邻居高度补齐，
避免侧壁正好落在 grid cell 之间而被漏掉。

---

## B. 变量 gap field（当前实现）

gap 不是 ``blockout.offset(一个数)``，而是空间变化的场：

    g(p) = f(d_margin, region, parameters)

| 区域 | gap |
|---|---|
| ``CementGapStart`` 带内（紧邻 Margin） | **0** —— 官方明确此处不施加 cement gap |
| 正常配合面 | ``CementGapThickness`` |
| 顶部（咬合面） | ``CementGapTopThickness``，**若病例未显式写出则保持 MISSING，不猜** |

另外 ``CrownAdditionalSpacingX/Y/Z`` 与 ``CrownAdditionalSpacingZOnTop`` 是独立的
XYZ 方向附加间隙，**不合并成一个法向偏移**。

### provenance 纪律

``CementGapTopThickness`` 与 ``CementGapEnd`` 在本次上传的 exocad
*Material Configuration Guide* 中**未找到对应的 technical name**。
所以代码支持它们，但 provenance 标为 ``constructionInfo_observed``，
**不能**标成 ``exocad_official_dictionary``。当前病例未显式写出时保持
``MISSING_INTERNAL_DEFAULT_UNKNOWN``，不偷偷取默认值。

---

## Policy：先建语义，再实现几何

``MillingMachineAxes`` 与 ``AdditionalBlockOutAngleAfterCementGap`` /
``UntouchableZoneNearPreparationMargin`` 是**功能依赖**（实测相关度 100%，
25 条记录无例外）。所以不要写 ``if value == -1: magic()``，而是先建
:class:`BlockoutPolicy`：

| 机轴数 | ``additional_blockout_after_gap`` | ``untouchable_zone_mm`` |
|---|---|---|
| 3 | 0.0（启用，无附加角度） | 0.0 |
| 5 | ``None``（**禁用**） | 0.1 |

``machine_axes = 4`` 等未知取值 → 保持 ``UNRESOLVED``，不套用近似策略。

> 注：这两个值（0 与 ``None``）在本项目现有几何里的实际效果都是"不加额外角度"。
> 之所以仍分开建模，是因为 ``-1`` 的语义是**禁用**而非"角度为 0" ——
> 将来出现非零角度时行为会不同，不能在代码里把两者混为一谈。
> 目前没有任何证据说明非零附加角度的几何含义，故不实现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import binary_dilation

from .insertion import _orthonormal_basis

#: 高度场的目标采样密度（沿最大 XY 跨度的 cell 数）。
TARGET_CELLS_ACROSS = 256
#: 分辨率夹取范围（mm）。
MIN_RESOLUTION_MM = 0.04
MAX_RESOLUTION_MM = 0.12
#: 保守栅格化的子采样数（每轴）。
SUBSAMPLES_PER_AXIS = 2
#: 阴影体底部相对 support 底部的下沉量（mm），用于封闭成实体。
BOTTOM_EXTENSION_MM = 1.0

#: 倒凹判据的拔模角容差（度）。与就位方向求解器保持一致。
DRAFT_TOLERANCE_DEG = 1.0
#: ``CementGapTopThickness`` 等参数的 provenance 标签。
#: 实测这两个参数名未出现在本次上传的 exocad 官方指南里，因此**不能**标为官方字典。
PROVENANCE_CONSTRUCTION_INFO = "constructionInfo_observed"
PROVENANCE_OFFICIAL = "exocad_official_dictionary"


class BlockoutStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"
    #: 参数未解析（例如机轴数未知），无法确定策略。
    UNRESOLVED = "UNRESOLVED"


# ------------------------------------------------------------------ policy

@dataclass(frozen=True)
class BlockoutPolicy:
    """由已解析参数导出的 blockout 语义策略。"""

    machine_axes: int
    #: ``AdditionalBlockOutAngleAfterCementGap``。``None`` = 该项**禁用**（exocad 的 -1）。
    #: 注意与"角度为 0"是两回事，不要合并。
    additional_blockout_after_gap: float | None
    #: ``UntouchableZoneNearPreparationMargin``：Margin 周围即使上方有 undercut
    #: 也永远不做 blockout 的区域宽度（mm）。
    untouchable_zone_mm: float
    provenance: dict = field(default_factory=dict)

    @property
    def is_three_axis(self) -> bool:
        return self.machine_axes == 3

    def as_dict(self) -> dict:
        return {
            "machine_axes": self.machine_axes,
            "additional_blockout_after_gap": self.additional_blockout_after_gap,
            "untouchable_zone_mm": self.untouchable_zone_mm,
            "provenance": dict(self.provenance),
        }


#: 机轴数 → 策略的语义表。
#:
#: 实测（25 条同时写出这三个参数的记录，相关度 100%）：
#:     3 轴 → AdditionalBlockOutAngleAfterCementGap =  0,  UntouchableZone = 0
#:     5 轴 → AdditionalBlockOutAngleAfterCementGap = -1,  UntouchableZone = 0.1
#: 与 exocad 官方对 3 轴 / 5 轴材料的区分一致（0 → 3 axis，-1 → 5 axis）。
_POLICY_TABLE: dict[int, tuple[float | None, float]] = {
    3: (0.0, 0.0),
    5: (None, 0.1),
}


def blockout_policy_from_resolved(resolved) -> tuple[BlockoutPolicy | None, dict]:
    """从 ``ResolvedParameterSet`` 构造策略。

    返回 ``(policy, diagnostics)``。机轴数未解析或其取值不在语义表内时，
    ``policy`` 为 ``None`` 且 diagnostics 里带 ``status = UNRESOLVED`` ——
    **不套用近似策略**。
    """
    from ..parameters.models import ParameterState

    diag: dict = {"status": BlockoutStatus.OK.value}

    axes_param = resolved["MillingMachineAxes"] if "MillingMachineAxes" in resolved else None
    if axes_param is None or axes_param.value is None:
        diag.update({
            "status": BlockoutStatus.UNRESOLVED.value,
            "reason": ("MillingMachineAxes 未解析，无法确定 blockout 策略。"
                       "派生参数（AdditionalBlockOutAngleAfterCementGap / "
                       "UntouchableZoneNearPreparationMargin）也随之保持未解析。"),
        })
        return None, diag

    axes = int(round(float(axes_param.value)))
    if axes not in _POLICY_TABLE:
        diag.update({
            "status": BlockoutStatus.UNRESOLVED.value,
            "reason": (f"MillingMachineAxes = {axes} 不在已知语义表 "
                       f"{sorted(_POLICY_TABLE)} 内；不套用近似策略。"),
            "machine_axes": axes,
        })
        return None, diag

    angle, zone = _POLICY_TABLE[axes]

    # 已解析的显式值优先于语义表（保持"病例显式值优先"的一贯规则）
    if "AdditionalBlockOutAngleAfterCementGap" in resolved:
        p = resolved["AdditionalBlockOutAngleAfterCementGap"]
        if p.value is not None:
            angle = None if float(p.value) == -1.0 else float(p.value)
    if "UntouchableZoneNearPreparationMargin" in resolved:
        p = resolved["UntouchableZoneNearPreparationMargin"]
        if p.value is not None:
            zone = float(p.value)

    provenance = {
        "machine_axes": {"source": "resolved", "value": axes},
        "additional_blockout_after_gap": {
            "source": "semantic_table_from_machine_axes",
            "note": "exocad 官方：0 → 3 轴材料，-1 → 5 轴材料。此处 -1 建模为『禁用』。",
        },
        "untouchable_zone_mm": {
            "source": "constructionInfo_observed",
            "official_definition": ("exocad 官方定义 UntouchableZoneNearPreparationMargin 为"
                                    "『Preparation Margin 周围即使上方存在 undercut 也永远"
                                    "不做 blockout 的区域』。"),
        },
    }
    return BlockoutPolicy(machine_axes=axes,
                          additional_blockout_after_gap=angle,
                          untouchable_zone_mm=zone,
                          provenance=provenance), diag


# ------------------------------------------------------------------ result

@dataclass
class BlockoutResult:
    blockout_mesh: trimesh.Trimesh
    undercut_faces: np.ndarray
    undercut_area_mm2: float
    undercut_area_ratio: float
    max_blockout_depth_mm: float
    margin_protected_area_mm2: float
    #: blockout 网格自身的拔模负角面积比。应接近 0 —— 这是"倒凹已被封闭"的直接证据。
    blockout_undercut_area_ratio: float
    status: BlockoutStatus
    diagnostics: dict = field(default_factory=dict)
    #: 构造 blockout 用的高度场与网格参数。B 段（gap field）需要它来做**外扩**，
    #: 而不是把整个实体平移 —— 纯平移不改变体积，对布尔没有作用（实测体积
    #: 13 位小数完全相同，是 gap 实现的一个真 bug）。
    grid: dict = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict:
        return {
            "undercut_area_mm2": round(float(self.undercut_area_mm2), 4),
            "undercut_area_ratio": round(float(self.undercut_area_ratio), 6),
            "max_blockout_depth_mm": round(float(self.max_blockout_depth_mm), 4),
            "margin_protected_area_mm2": round(float(self.margin_protected_area_mm2), 4),
            "blockout_undercut_area_ratio": round(float(self.blockout_undercut_area_ratio), 6),
            "n_undercut_faces": int(len(self.undercut_faces)),
            "watertight": bool(self.blockout_mesh.is_watertight),
            "n_components": int(len(self.blockout_mesh.split(only_watertight=False))),
            "volume_mm3": round(float(abs(self.blockout_mesh.volume)), 4)
            if self.blockout_mesh.is_watertight else None,
            "status": self.status.value,
            "diagnostics": self.diagnostics,
        }


# ------------------------------------------------------------------ helpers

def _surface_geodesic_from_margin(
    support: trimesh.Trimesh, margin_points: np.ndarray, *, sources_mm: float = 1e-3
) -> np.ndarray:
    """support 每个顶点到 Margin 的**测地**距离（网格图最短路径）。

    用测地距离而不是 ``z - margin_plane_z``：以后病例的 Margin 未必像 FDI36 这么平
    （真实 finish line 可以沿龈下轮廓走），用轴向差会把保护区域切错。

    做法：把 Margin 点吸附到最近的 support 顶点作为源点，在顶点邻接图上跑
    Dijkstra（边权为边长）。这是网格图上的真实测地距离。
    """
    verts = np.asarray(support.vertices, dtype=float)
    edges = np.asarray(support.edges_unique, dtype=np.int64)
    lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)

    n = len(verts)
    graph = sparse.coo_matrix(
        (np.concatenate([lengths, lengths]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(n, n),
    ).tocsr()

    # 源点：离每个 Margin 点最近的 support 顶点
    from scipy.spatial import cKDTree
    mp = np.asarray(margin_points, dtype=float)
    src = sorted(set(cKDTree(verts).query(mp, k=1, workers=-1)[1].tolist()))
    dist = dijkstra(graph, indices=src, directed=False)
    # 多源时取最小值
    return np.min(np.atleast_2d(dist), axis=0)


def _resolution_mm(extent_xy: np.ndarray) -> float:
    span = float(np.max(extent_xy))
    if span <= 0:
        return MAX_RESOLUTION_MM
    return float(np.clip(span / TARGET_CELLS_ACROSS, MIN_RESOLUTION_MM, MAX_RESOLUTION_MM))


def _fix_diagonal_connections(mask: np.ndarray) -> np.ndarray:
    """消除"仅对角相连"的 2×2 配置，否则共享角点处会有 4 个侧壁面交汇 → 非流形。

    若一个 2×2 块里恰好只有对角两个 cell 覆盖，补上其中一个正交 cell 使其变成 L 形。

    **必须迭代到不动点。** 实测单遍扫描会漏：补一个 cell 会让相邻的 2×2 块从
    "对角"变成另一种形态，而那一块可能已经被扫过了，于是残留对角 → 网格非水密
    （实测残留 5 条非流形边，全部是竖直角边被 4 面共享）。
    """
    m = mask.copy()
    changed = True
    guard = 0
    while changed and guard < 1000:
        changed = False
        guard += 1
        for i in range(m.shape[0] - 1):
            for j in range(m.shape[1] - 1):
                blk = m[i:i + 2, j:j + 2]
                if int(blk.sum()) != 2:
                    continue
                if blk[0, 0] and blk[1, 1] and not (blk[0, 1] or blk[1, 0]):
                    m[i, j + 1] = True
                    changed = True
                elif blk[0, 1] and blk[1, 0] and not (blk[0, 0] or blk[1, 1]):
                    m[i, j] = True
                    changed = True
    return m


def _count_diagonal_connections(mask: np.ndarray) -> int:
    """统计残留的"仅对角相连"2×2 配置数（应为 0，否则网格会非流形）。"""
    n = 0
    for i in range(mask.shape[0] - 1):
        for j in range(mask.shape[1] - 1):
            blk = mask[i:i + 2, j:j + 2]
            if int(blk.sum()) != 2:
                continue
            if (blk[0, 0] and blk[1, 1] and not (blk[0, 1] or blk[1, 0])) or                (blk[0, 1] and blk[1, 0] and not (blk[0, 0] or blk[1, 1])):
                n += 1
    return n


def _height_field_local(
    support_local: trimesh.Trimesh, resolution: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """在局部坐标系（Z = insertion axis）里构造方向高度场。

    返回 ``(height, covered, protected, origin_x, origin_y)``，
    ``height`` 为 ``(H, W)``，未覆盖处为 ``-inf``。
    """
    verts = np.asarray(support_local.vertices, dtype=float)
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)

    # 保守栅格化：向外多留一个 cell，配合膨胀
    nx = int(math.ceil((hi[0] - lo[0]) / resolution)) + 3
    ny = int(math.ceil((hi[1] - lo[1]) / resolution)) + 3
    origin_x = lo[0] - 1.5 * resolution
    origin_y = lo[1] - 1.5 * resolution

    xs = origin_x + (np.arange(nx) + 0.5) * resolution
    ys = origin_y + (np.arange(ny) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys, indexing="xy")

    # 子采样（保守栅格化）
    sub = (np.arange(SUBSAMPLES_PER_AXIS) + 0.5) / SUBSAMPLES_PER_AXIS - 0.5
    height = np.full((ny, nx), -np.inf, dtype=float)

    for dy in sub:
        for dx in sub:
            ox = gx.ravel() + dx * resolution
            oy = gy.ravel() + dy * resolution
            origins = np.c_[ox, oy, np.full(ox.size, hi[2] + BOTTOM_EXTENSION_MM)]
            dirs = np.tile(np.array([0.0, 0.0, -1.0]), (ox.size, 1))
            loc, index_ray, _ = support_local.ray.intersects_location(
                origins, dirs, multiple_hits=False)
            z = np.full(ox.size, -np.inf)
            z[index_ray] = loc[:, 2]
            height = np.maximum(height, z.reshape(ny, nx))

    covered = np.isfinite(height)
    # 边界半像素等效的膨胀：把覆盖掩码向外扩一个 cell，缺高度处用邻居最大值补齐
    dilated = binary_dilation(covered, structure=np.ones((3, 3), dtype=bool), iterations=1)
    newly = dilated & ~covered
    if newly.any():
        filled = np.where(covered, height, -np.inf)
        for _ in range(2):     # 两轮传播足以覆盖一圈
            nb = np.full_like(filled, -np.inf)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = np.maximum(nb, np.roll(np.roll(filled, dy, 0), dx, 1))
            filled = np.where(newly & (filled == -np.inf) & np.isfinite(nb), nb, filled)
        height = np.where(dilated, filled, -np.inf)
        covered = dilated

    # 去掉膨胀出去但始终没拿到高度的 cell
    covered = np.isfinite(height)
    height = np.where(covered, height, -np.inf)
    return height, covered, origin_x, origin_y


def _heightfield_to_mesh(
    height: np.ndarray, covered: np.ndarray, origin_x: float, origin_y: float,
    resolution: float, z_bottom: float,
) -> trimesh.Trimesh:
    """把高度场三角化成**水密**网格。

    结构：上表面（高度场）+ 下表面（平面）+ 侧壁（边界处竖直四边形）。
    顶点落在 grid corner 上，相邻 quad 共享顶点 → 水密。
    """
    mask = _fix_diagonal_connections(covered)
    H, W = mask.shape
    cnx, cny = W + 1, H + 1

    def cx(j): return origin_x + j * resolution
    def cy(i): return origin_y + i * resolution

    # corner 高度 = 相邻已覆盖 cell 的均值
    corner_h = np.full((cny, cnx), np.nan)
    corner_cnt = np.zeros((cny, cnx))
    h_safe = np.where(np.isfinite(height), height, 0.0)
    for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
        sl = mask
        corner_h[di:di + H, dj:dj + W] = np.where(
            np.isnan(corner_h[di:di + H, dj:dj + W]), 0.0, corner_h[di:di + H, dj:dj + W]
        ) + np.where(sl, h_safe, 0.0)
        corner_cnt[di:di + H, dj:dj + W] += sl.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        corner_h = np.where(corner_cnt > 0, corner_h / np.maximum(corner_cnt, 1.0), np.nan)

    verts: list[list[float]] = []
    top_idx = np.full((cny, cnx), -1, dtype=np.int64)
    bot_idx = np.full((cny, cnx), -1, dtype=np.int64)

    def ensure(i: int, j: int) -> None:
        if top_idx[i, j] >= 0:
            return
        top_idx[i, j] = len(verts)
        verts.append([cx(j), cy(i), float(corner_h[i, j])])
        bot_idx[i, j] = len(verts)
        verts.append([cx(j), cy(i), float(z_bottom)])

    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            for di, dj in ((0, 0), (0, 1), (1, 0), (1, 1)):
                ensure(i + di, j + dj)

    faces: list[list[int]] = []
    # 上/下表面
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            a, b, c, d = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
            t = [top_idx[x] for x in (a, b, c, d)]
            bo = [bot_idx[x] for x in (a, b, c, d)]
            faces += [[t[0], t[1], t[2]], [t[0], t[2], t[3]]]
            faces += [[bo[0], bo[2], bo[1]], [bo[0], bo[3], bo[2]]]

    # 侧壁：cell 的四条边中，越界或邻居未覆盖的那条
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                continue
            edges = (
                ((i, j), (i, j + 1), (i - 1, j)),      # 下
                ((i + 1, j), (i + 1, j + 1), (i + 1, j)),  # 上
                ((i, j), (i + 1, j), (i, j - 1)),      # 左
                ((i, j + 1), (i + 1, j + 1), (i, j + 1)),  # 右
            )
            for c0, c1, (ni, nj) in edges:
                outside = not (0 <= ni < H and 0 <= nj < W and mask[ni, nj])
                if not outside:
                    continue
                t0, t1 = top_idx[c0], top_idx[c1]
                b0, b1 = bot_idx[c0], bot_idx[c1]
                faces += [[t0, b0, b1], [t0, b1, t1]]

    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=float),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass
    return mesh


# ------------------------------------------------------------------ A. blockout

def build_blockout(
    support_mesh: trimesh.Trimesh,
    margin_curve,
    insertion_axis,
    policy: BlockoutPolicy,
    *,
    resolution_mm: float | None = None,
) -> BlockoutResult:
    """构造方向阴影体（A 段）。

    所有几何在由 ``insertion_axis`` 建立的局部坐标系内完成；因此结果对
    刚体变换等变（``T(support) + R(axis)`` 与 ``T(blockout)`` 等价）。
    """
    if len(support_mesh.faces) == 0:
        raise ValueError("support 为空网格")
    axis = np.asarray(insertion_axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or float(np.linalg.norm(axis)) < 1e-9:
        raise ValueError("insertion_axis 无效")
    axis = axis / float(np.linalg.norm(axis))

    margin = np.asarray(margin_curve, dtype=float) if margin_curve is not None \
        else np.zeros((0, 3))

    # ---- 局部坐标系：完全由 axis 建立（Z = axis），不依赖世界轴
    e1, e2 = _orthonormal_basis(axis)
    frame = np.eye(4)
    frame[:3, 0] = e1
    frame[:3, 1] = e2
    frame[:3, 2] = axis
    support_local = support_mesh.copy()
    support_local.apply_transform(np.linalg.inv(frame))

    verts_local = np.asarray(support_local.vertices, dtype=float)
    resolution = float(resolution_mm) if resolution_mm else _resolution_mm(
        np.array([np.ptp(verts_local[:, 0]), np.ptp(verts_local[:, 1])]))

    height, covered, origin_x, origin_y = _height_field_local(support_local, resolution)

    # ---- Margin 保护带：测地距离 < untouchable_zone 的列不做 blockout
    protected_mask = np.zeros_like(covered)
    protected_area = 0.0
    if policy.untouchable_zone_mm > 0 and len(margin) >= 3:
        geo = _surface_geodesic_from_margin(support_mesh, margin)
        faces = np.asarray(support_mesh.faces)
        face_geo = geo[faces].min(axis=1)
        face_centers_local = np.asarray(support_local.triangles_center)
        protected_faces = face_geo < policy.untouchable_zone_mm
        protected_area = float(np.asarray(support_mesh.area_faces)[protected_faces].sum())

        jj = np.floor((face_centers_local[protected_faces, 0] - origin_x) / resolution).astype(int)
        ii = np.floor((face_centers_local[protected_faces, 1] - origin_y) / resolution).astype(int)
        ok = (ii >= 0) & (ii < covered.shape[0]) & (jj >= 0) & (jj < covered.shape[1])
        protected_mask[ii[ok], jj[ok]] = True

    effective_covered = covered & ~protected_mask

    z_bottom = float(verts_local[:, 2].min()) - BOTTOM_EXTENSION_MM
    blockout_local = _heightfield_to_mesh(height, effective_covered, origin_x, origin_y,
                                         resolution, z_bottom)
    blockout = blockout_local.copy()
    blockout.apply_transform(frame)

    # ---- undercut 度量：面心落在阴影内（z < h(x,y)）的面即被 blockout 掉的部分
    centers_local = np.asarray(support_local.triangles_center, dtype=float)
    jj = np.floor((centers_local[:, 0] - origin_x) / resolution).astype(int)
    ii = np.floor((centers_local[:, 1] - origin_y) / resolution).astype(int)
    inside = (ii >= 0) & (ii < height.shape[0]) & (jj >= 0) & (jj < height.shape[1])
    h_at = np.full(len(centers_local), -np.inf)
    h_at[inside] = height[ii[inside], jj[inside]]
    depth = h_at - centers_local[:, 2]
    faces_arr = np.asarray(support_mesh.faces)
    face_in_protection = np.zeros(len(faces_arr), dtype=bool)
    if protected_mask.any():
        pil = protected_mask[ii.clip(0, height.shape[0] - 1), jj.clip(0, height.shape[1] - 1)]
        face_in_protection = pil & inside

    # 倒凹判据用**拔模角**，不用"低于该 XY 最高点"。
    #
    # 后者对与轴平行的侧壁会把整面侧壁算成倒凹（实测比例 0.91–0.94，几乎全中），
    # 与之前就位方向求解器上踩过的掠射陷阱同源：平行面既非倒凹也非可脱模，
    # 它只是"与轴平行"，而 d(z)/dz=0 处高度场无法区分。
    # 拔模角 n·a < -sin(tol) 是无歧义的判据。
    normals = np.asarray(support_mesh.face_normals, dtype=float)
    draft = normals @ axis
    draft_negative = draft < -math.sin(math.radians(DRAFT_TOLERANCE_DEG))

    areas = np.asarray(support_mesh.area_faces, dtype=float)

    # 就位面限定在 **Margin 平面之上** —— 与就位方向求解器一致。
    # 不加这条会把 Margin 以下的颈部算进来（颈部相对任何轴都是倒凹），
    # 使倒凹比虚高（实测 0.3109 vs 求解器的 0.1170，差异全部来自作用域）。
    above_margin = np.ones(len(areas), dtype=bool)
    if len(margin) >= 3:
        from .insertion import _robust_plane
        mc, mn, _ = _robust_plane(margin)
        if float(np.dot(mn, axis)) < 0:
            mn = -mn
        above_margin = (np.asarray(support_mesh.triangles_center) - mc) @ mn > 0.05

    fitting = inside & ~face_in_protection & above_margin
    undercut = fitting & draft_negative
    fill_depth = np.where(undercut, np.maximum(depth, 0.0), 0.0)

    diagnostics = {
        "resolution_mm": round(resolution, 6),
        "grid_shape": [int(height.shape[0]), int(height.shape[1])],
        "n_covered_cells": int(covered.sum()),
        "n_cells_after_margin_protection": int(effective_covered.sum()),
        "n_cells_protected": int((covered & protected_mask).sum()),
        "policy": policy.as_dict(),
        "z_bottom_local": round(z_bottom, 6),
        "axis": [float(v) for v in axis],
        "local_frame": {"e1": [float(v) for v in e1], "e2": [float(v) for v in e2]},
        "bottom_extension_mm": BOTTOM_EXTENSION_MM,
        "diagonal_connections_before_fix": int(_count_diagonal_connections(effective_covered)),
        "diagonal_connections_after_fix": int(
            _count_diagonal_connections(_fix_diagonal_connections(effective_covered))),
        "conservative_rasterization": {
            "subsamples_per_axis": SUBSAMPLES_PER_AXIS,
            "mask_dilation_cells": 1,
            "note": "掠射边界用子采样取最大高度 + 掩码外扩一个 cell，避免侧壁落在 grid cell 之间。",
        },
    }

    # blockout 自身的倒凹比只统计**包络面**（顶面 + 侧壁）。
    # 必须排除底面盖片：它的法向朝下（-axis），会被算成拔模负角，贡献约 6% 的假倒凹。
    # 底面从不受牙冠接触，与"能否就位"无关。
    bl_normals = np.asarray(blockout.face_normals, dtype=float)
    bl_areas = np.asarray(blockout.area_faces, dtype=float)
    bl_centers = np.asarray(blockout.triangles_center, dtype=float)
    envelope = bl_centers @ axis > (z_bottom + 2.0 * resolution)
    bl_draft = bl_normals @ axis
    bl_negative = envelope & (bl_draft < -math.sin(math.radians(DRAFT_TOLERANCE_DEG)))
    blockout_undercut_ratio = float(bl_areas[bl_negative].sum() / max(bl_areas[envelope].sum(), 1e-9))

    status = BlockoutStatus.OK
    if not blockout.is_watertight:
        status = BlockoutStatus.REVIEW_REQUIRED
        diagnostics["reasons"] = ["blockout 网格不水密"]
    elif len(blockout.split(only_watertight=False)) != 1:
        status = BlockoutStatus.REVIEW_REQUIRED
        diagnostics["reasons"] = ["blockout 网格非单连通"]

    return BlockoutResult(
        blockout_mesh=blockout,
        undercut_faces=np.flatnonzero(undercut),
        undercut_area_mm2=float(areas[undercut].sum()),
        undercut_area_ratio=float(areas[undercut].sum() / max(areas[fitting].sum(), 1e-9)),
        blockout_undercut_area_ratio=blockout_undercut_ratio,
        max_blockout_depth_mm=float(fill_depth.max()) if undercut.any() else 0.0,
        margin_protected_area_mm2=protected_area,
        grid={
            "height": height, "covered": effective_covered,
            "origin_x": origin_x, "origin_y": origin_y,
            "resolution": resolution, "z_bottom": z_bottom, "frame": frame,
            "axis": axis,
        },
        status=status,
        diagnostics=diagnostics,
    )


# ------------------------------------------------------------------ B. gap field

@dataclass(frozen=True)
class GapFieldParams:
    """变量 gap 的参数。**每个都用四态解析结果构造，缺失即缺失，不猜 0。**

    provenance 纪律：``CementGapTopThickness`` / ``CementGapEnd`` 未出现在本次上传的
    exocad 官方指南中，标签必须是 :data:`PROVENANCE_CONSTRUCTION_INFO`。
    """

    cement_gap_mm: float | None
    cement_gap_start_mm: float | None
    cement_gap_top_mm: float | None
    additional_spacing_xyz_mm: tuple[float, float, float] | None
    additional_spacing_z_on_top_mm: float | None
    provenance: dict = field(default_factory=dict)

    @property
    def has_top_gap(self) -> bool:
        return self.cement_gap_top_mm is not None

    def as_dict(self) -> dict:
        return {
            "cement_gap_mm": self.cement_gap_mm,
            "cement_gap_start_mm": self.cement_gap_start_mm,
            "cement_gap_top_mm": self.cement_gap_top_mm,
            "additional_spacing_xyz_mm": list(self.additional_spacing_xyz_mm)
            if self.additional_spacing_xyz_mm else None,
            "additional_spacing_z_on_top_mm": self.additional_spacing_z_on_top_mm,
            "provenance": dict(self.provenance),
        }


def gap_field_params_from_resolved(resolved) -> GapFieldParams:
    """从 ``ResolvedParameterSet`` 构造 gap 参数，保留 MISSING 语义。"""
    def val(name):
        if name not in resolved:
            return None
        p = resolved[name]
        return p.value if p.is_resolved else None

    def prov(name, official: bool):
        if name not in resolved:
            return {"present": False}
        p = resolved[name]
        return {
            "present": p.value is not None,
            "source": (PROVENANCE_OFFICIAL if official else PROVENANCE_CONSTRUCTION_INFO),
            "resolver_source": p.source.value,
            "state": p.state.value,
        }

    xyz = (val("CrownAdditionalSpacingX"), val("CrownAdditionalSpacingY"),
           val("CrownAdditionalSpacingZ"))
    xyz = xyz if all(v is not None for v in xyz) else None

    return GapFieldParams(
        cement_gap_mm=val("CementGapThickness"),
        cement_gap_start_mm=val("CementGapStart"),
        cement_gap_top_mm=val("CementGapTopThickness"),
        additional_spacing_xyz_mm=xyz,
        additional_spacing_z_on_top_mm=val("CrownAdditionalSpacingZOnTop"),
        provenance={
            "CementGapThickness": prov("CementGapThickness", official=True),
            "CementGapStart": prov("CementGapStart", official=True),
            "CementGapTopThickness": prov("CementGapTopThickness", official=False),
            "CrownAdditionalSpacingX/Y/Z": prov("CrownAdditionalSpacingX", official=False),
            "CrownAdditionalSpacingZOnTop": prov("CrownAdditionalSpacingZOnTop", official=False),
            "_note": ("CementGapTopThickness 与 CementGapEnd 在本次上传的 exocad "
                      "Material Configuration Guide 中未找到 technical name，"
                      "provenance 标为 constructionInfo_observed，不得标为官方字典。"
                      "病例未显式写出时保持 MISSING，不猜默认值。"),
        },
    )


def apply_gap_field(
    blockout: BlockoutResult | trimesh.Trimesh,
    support_mesh: trimesh.Trimesh,
    margin_curve,
    insertion_axis,
    params: GapFieldParams,
) -> trimesh.Trimesh:
    """把变量 gap 场施加到 blockout 上，得到 intaglio cutter（B 段）。

    ``g(p) = f(d_margin, region, parameters)``：

    - ``d_margin`` 小于 ``CementGapStart`` 的带内（紧邻 Margin）gap = **0**
    - 正常配合面 gap = ``CementGapThickness``
    - 顶部（咬合面附近）gap = ``CementGapTopThickness``（未显式给出则不加）

    **实现方式是在高度场域内做外扩，不是平移整个实体。**
    实体 ``{(x,y,z): z_bottom ≤ z ≤ h(x,y), (x,y)∈M}`` 的**外扩** =
    顶面抬升 ``h → h + g`` **且** 覆盖掩码外扩 ``M → M ⊕ disk(g)``。
    只做前者会漏掉侧壁的法向分量；把全部顶点一起平移则体积完全不变
    （实测 13 位小数相同），对后续布尔毫无作用 —— 那是实现初版的一个真 bug，
    由 ``test_gap_increases_cutter_volume_monotonically`` 抓到。

    不使用"对整块 mesh 做一次统一法向 offset"：``CrownAdditionalSpacingX/Y/Z``
    与顶端间距是独立的自由度，合并成一个偏移量会丢掉它们。
    """
    if isinstance(blockout, trimesh.Trimesh):
        raise TypeError(
            "apply_gap_field 需要 build_blockout 返回的 BlockoutResult（携带高度场网格）；"
            "只传 mesh 无法做掩码外扩，会导致 gap 退化为平移。")

    grid = blockout.grid
    if not grid:
        raise ValueError("BlockoutResult 缺少 grid，无法施加 gap field")

    base_gap = params.cement_gap_mm
    if base_gap is None:
        raise ValueError(
            "CementGapThickness 未解析。gap 是必填项，不能用 0 或猜测值代替 —— "
            "0 粘接间隙意味着牙冠与预备体直接贴合、无法就位。")

    axis = np.asarray(insertion_axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    frame = grid["frame"]
    height = np.asarray(grid["height"], dtype=float)
    covered = np.asarray(grid["covered"], dtype=bool)
    resolution = float(grid["resolution"])
    origin_x, origin_y, z_bottom = grid["origin_x"], grid["origin_y"], grid["z_bottom"]

    support_local = support_mesh.copy()
    support_local.apply_transform(np.linalg.inv(frame))
    support_z_top = float(np.asarray(support_local.vertices)[:, 2].max())

    margin = np.asarray(margin_curve, dtype=float) if margin_curve is not None         else np.zeros((0, 3))
    margin_local = (np.linalg.inv(frame) @ np.c_[margin, np.ones(len(margin))].T).T[:, :3]         if len(margin) else np.zeros((0, 3))

    # ---- 逐 cell 的 gap
    H, W = height.shape
    xs = origin_x + (np.arange(W) + 0.5) * resolution
    ys = origin_y + (np.arange(H) + 0.5) * resolution
    gx, gy = np.meshgrid(xs, ys, indexing="xy")

    gap_cell = np.full((H, W), float(base_gap))

    if params.cement_gap_start_mm and params.cement_gap_start_mm > 0 and len(margin_local):
        # 未覆盖 cell 的高度是 -inf，直接送入 KD-tree 会报 must be finite。
        # 用有限占位值替换；这些 cell 本来就不参与（gap_cell 只在 covered 处被使用）。
        h_finite = np.where(np.isfinite(height), height, 0.0)
        d_margin = _nearest_axis_distance(
            margin, np.c_[gx.ravel(), gy.ravel(), h_finite.ravel()], axis, frame)
        gap_cell = np.where(d_margin.reshape(H, W) < float(params.cement_gap_start_mm),
                            0.0, gap_cell)

    if params.has_top_gap:
        is_top = (support_z_top - height) < 0.25
        gap_cell = np.where(is_top & np.isfinite(height), float(params.cement_gap_top_mm),
                            gap_cell)
    if params.additional_spacing_xyz_mm is not None:
        gap_cell = gap_cell + float(params.additional_spacing_xyz_mm[2])
    if params.additional_spacing_z_on_top_mm is not None:
        is_top = (support_z_top - height) < 0.25
        gap_cell = np.where(is_top & np.isfinite(height),
                            gap_cell + float(params.additional_spacing_z_on_top_mm), gap_cell)

    # ---- 外扩：顶面抬升 + 掩码按 gap 外扩
    height_offset = np.where(covered, height + gap_cell, height)
    g_max = float(np.nanmax(gap_cell[covered])) if covered.any() else 0.0
    if g_max > 0:
        radius_cells = int(math.ceil(g_max / resolution))
        yy, xx = np.mgrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
        disk = (xx ** 2 + yy ** 2) <= (radius_cells ** 2)
        dilated = binary_dilation(covered, structure=disk, iterations=1)
        newly = dilated & ~covered
        if newly.any():
            # 新覆盖的 cell 用邻域内最高的抬升后高度补齐，保持包络单调
            filled = np.where(covered, height_offset, -np.inf)
            for _ in range(radius_cells + 1):
                nb = np.full_like(filled, -np.inf)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = np.maximum(nb, np.roll(np.roll(filled, dy, 0), dx, 1))
                filled = np.where(newly & (filled == -np.inf) & np.isfinite(nb), nb, filled)
            height_offset = np.where(dilated, filled, -np.inf)
            covered_out = dilated
        else:
            covered_out = covered
    else:
        covered_out = covered

    covered_out = np.isfinite(height_offset) & covered_out
    local = _heightfield_to_mesh(height_offset, covered_out, origin_x, origin_y,
                                 resolution, z_bottom)
    out = local.copy()
    out.apply_transform(frame)
    return out


def _nearest_axis_distance(margin_points, query_points, axis, frame) -> np.ndarray:
    """查询点到 Margin 的近似测地距离。

    用 support 表面上的顶点测地距离更严格（见 ``_surface_geodesic_from_margin``），
    但 cutter 的顶点不在 support 上。此处在**局部轴坐标**里用
    "轴向差 + 同轴半径差"的混合度量近似表面距离 —— 够用于 ``CementGapStart``
    这种 0.1–1 mm 量级的带内判定，且不引入世界轴依赖。
    """
    mp = (np.linalg.inv(frame) @ np.c_[margin_points, np.ones(len(margin_points))].T).T[:, :3]
    qp = (np.linalg.inv(frame) @ np.c_[query_points, np.ones(len(query_points))].T).T[:, :3]
    from scipy.spatial import cKDTree
    # 在 (径向, 轴向) 平面上比较，避免轴向差被忽略
    def feat(p):
        r = np.linalg.norm(p[:, :2], axis=1)
        return np.c_[r, p[:, 2]]
    d, _ = cKDTree(feat(mp)).query(feat(qp), k=1, workers=-1)
    return d


# ------------------------------------------------------------------ 兼容入口

def build_blockout_cutter(
    support_mesh: trimesh.Trimesh,
    margin_points,
    insertion_axis,
    cement_gap_mm: float | None = None,
    margin_gap_mm: float | None = None,
    margin_band_mm: float = 0.8,
    pitch_mm: float = 0.10,
    entry_extension_mm: float = 2.0,
    *,
    policy: BlockoutPolicy | None = None,
    gap_params: GapFieldParams | None = None,
) -> trimesh.Trimesh:
    """旧签名的组合入口：A（方向 blockout）+ B（变量 gap）→ cutter。

    保留它是为了让调用方在迁移期不至于中断。新代码请分别调用
    :func:`build_blockout` 与 :func:`apply_gap_field`，以便分辨改善来自哪一段。

    ``pitch_mm`` 与 ``entry_extension_mm`` 在新实现里不再使用（新实现是
    2D 高度场 + 精确布尔，不是 3D 体素化）。保留参数仅为兼容，并在
    diagnostics 里标注忽略。
    """
    p = policy or BlockoutPolicy(machine_axes=5,
                                 additional_blockout_after_gap=None,
                                 untouchable_zone_mm=0.0,
                                 provenance={"source": "legacy_default"})
    gp = gap_params or GapFieldParams(
        cement_gap_mm=cement_gap_mm,
        cement_gap_start_mm=None,
        cement_gap_top_mm=None,
        additional_spacing_xyz_mm=None,
        additional_spacing_z_on_top_mm=None,
        provenance={"source": "legacy_signature"},
    )
    result = build_blockout(support_mesh, margin_points, insertion_axis, p)
    return apply_gap_field(result, support_mesh, margin_points, insertion_axis, gp)
