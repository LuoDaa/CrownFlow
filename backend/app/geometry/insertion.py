"""就位方向求解器。

与旧 ``suggest_insertion_axis`` 的区别：旧实现直接对整块 support 顶点做 PCA 取主轴，
底部、颈缘和扫描裁切形状都会把主轴拉歪（实测在真实颈缘上稳定偏 60.9–64.2°）。

本实现按以下结构求解：

1. **初始轴**由两个独立来源合成
   - A = Margin 的稳健拟合平面法向（按"support 主体位于法向正侧"定向）
   - B = 侧壁法向拟合轴（面积加权的法向协方差最小特征值方向；对圆柱侧壁，
     所有法向都垂直于轴，因此这是该轴的最小二乘估计）
   两者夹角过大时不硬平均，返回 ``REVIEW_REQUIRED``。
2. **锥内搜索**：在初始轴 ±``max_deviation_deg`` 内粗—细两层采样。
3. **逐候选评分**：面积加权倒凹比 + 归一化倒凹深度 + (1 − Margin 可见率) + 偏角罚。
4. **状态判定**：``OK`` / ``REVIEW_REQUIRED`` / ``FAILED``，不允许"挑最不坏的一个
   然后声称 OK"。

关于倒凹判据的选择（有实测依据，勿改回射线法）：

- 用**拔模角** ``n·a < -sin(tol)`` 判定倒凹，而不是"从点出发沿 a 发射线是否命中"。
  原因是后者对**与就位轴平行的侧壁**数值退化：掠射几乎必然误命中，而侧壁恰是
  最需要精确判定的区域。实测在 FDI36 基台上，射线法给出 56.8% 不可达，而拔模法
  给出 45.6%，两者的差异主要来自掠射噪声。
- 对拔模角合格（``n·a >= 0``）但仍被上方几何遮挡的面，另用射线法补一个"阴影"
  深度统计；此时射线与表面不平行，数值是可靠的。

关于目标函数的局限（重要，见 ``diagnostics["objective_flatness"]``）：

倒凹面积、射线不可达、blockout 代价这三类目标，在**紧凑且轴对称的基台**上区分度很低
（实测 FDI36 基台上三者跨轴变化都在 ±8% 以内，且最小值不落在该病例的临床正确方向上）。
因此本模块除了返回最优轴，还会显式计算**候选集上的目标平坦度**并把
``objective_is_flat`` 置位 —— 用来告诉调用方"这个病例的轴其实没被目标函数定住，
必须人工确认"，而不是让一个平坦目标上的 argmin 冒充一个可信结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh

#: 默认最大偏角。来源：当前 Lab Profile 的实测证据
#: （``CrownBottomInsertionAngleBetweenTeethLimit`` 在 25/25 条 exocad 记录上为 20）。
#: **这是该实验室的病例级证据，不是 exocad 全局规律** —— 上传的 exocad
#: Material Configuration Guide 并未给出该技术字段的定义。换实验室需重新取值。
DEFAULT_MAX_DEVIATION_DEG = 20.0

#: 拔模角容差：|n·a| 小于此值视为"侧壁"，不计入倒凹。
DRAFT_TOLERANCE_DEG = 1.0

#: 判定目标"平坦"的阈值：候选集上最优与最差评分之差小于此值时认为轴未被定住。
FLATNESS_THRESHOLD = 0.02

#: 面片必须高出 Margin 平面多少毫米才算入就位面。
#: 取正值以排除位于平面上的底面/裁切盖片（否则会贡献面积占比 16% 量级的假倒凹）。
ABOVE_MARGIN_TOLERANCE_MM = 0.05

#: 状态判定阈值。
MAX_ACCEPTABLE_UNDERCUT_RATIO = 0.15
MIN_ACCEPTABLE_MARGIN_VISIBILITY = 0.90
MAX_INITIAL_DISAGREEMENT_DEG = 35.0


class AxisStatus(str, Enum):
    OK = "OK"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"


def _json_safe(value):
    """把非有限浮点转成 ``None``。

    必要性：降级路径（无 Margin）用 NaN 表达"未知"，而 NaN 不是合法 JSON，
    FastAPI 序列化时会直接抛 ``ValueError: Out of range float values``。
    语义上也更准确 —— "未知"应当是 null，不是 NaN。
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


@dataclass
class InsertionAxisResult:
    """就位方向求解结果。"""

    axis: list[float]
    initial_axis: list[float]
    score: float

    undercut_area_mm2: float
    undercut_area_ratio: float
    max_undercut_depth_mm: float
    mean_undercut_depth_mm: float

    margin_visibility_ratio: float

    deviation_deg: float

    confidence: float

    status: AxisStatus
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return _json_safe({
            "axis": [round(float(v), 6) for v in self.axis],
            "initial_axis": [round(float(v), 6) for v in self.initial_axis],
            "score": round(float(self.score), 6),
            "undercut_area_mm2": round(float(self.undercut_area_mm2), 4),
            "undercut_area_ratio": round(float(self.undercut_area_ratio), 6),
            "max_undercut_depth_mm": round(float(self.max_undercut_depth_mm), 4),
            "mean_undercut_depth_mm": round(float(self.mean_undercut_depth_mm), 4),
            "margin_visibility_ratio": round(float(self.margin_visibility_ratio), 6),
            "deviation_deg": round(float(self.deviation_deg), 4),
            "confidence": round(float(self.confidence), 4),
            "status": self.status.value,
            "diagnostics": self.diagnostics,
        })


# --------------------------------------------------------------------- helpers

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("零向量无法归一化")
    return v / n


def angle_deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(_unit(a), _unit(b)))), -1.0, 1.0))))


def _orthonormal_basis(axis: np.ndarray, *, e1_hint: np.ndarray | None = None):
    """构造与 axis 正交的一组基。

    ``e1_hint`` 用来让基随几何一同旋转（刚性等变性）：传入 Margin 的平面内主轴时，
    锥采样的方位角起点会跟着模型转，从而保证候选集合整体等变。
    """
    axis = _unit(axis)
    if e1_hint is not None:
        e1 = np.asarray(e1_hint, dtype=float)
        e1 = e1 - axis * float(np.dot(e1, axis))
        if float(np.linalg.norm(e1)) > 1e-6:
            e1 = _unit(e1)
            return e1, _unit(np.cross(axis, e1))
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(axis, helper))
    return e1, _unit(np.cross(axis, e1))


# --------------------------------------------------------------- margin handling

@dataclass
class MarginValidation:
    """Margin 校验结果。

    ``status`` 的语义：``None`` = 无问题；``REVIEW_REQUIRED`` = 可继续但需人工确认；
    ``FAILED`` = 无法继续（点数不足、坐标非有限、点重合等结构性缺陷）。
    """

    ok: bool
    status: AxisStatus | None = None
    reasons: list[str] = field(default_factory=list)
    n_points: int = 0
    closure_ratio: float | None = None
    planarity_rms_mm: float | None = None
    spacing_cv: float | None = None
    plane_origin: list[float] | None = None
    plane_normal: list[float] | None = None
    in_plane_e1: list[float] | None = None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status.value if self.status else None,
            "reasons": list(self.reasons),
            "n_points": self.n_points,
            "closure_ratio": self.closure_ratio,
            "planarity_rms_mm": self.planarity_rms_mm,
            "spacing_cv": self.spacing_cv,
        }


def _robust_plane(points: np.ndarray, *, iterations: int = 3, keep: float = 0.8):
    """迭代剔除离群点的平面拟合（比纯 SVD 对退化 Margin 稳健得多）。"""
    pts = np.asarray(points, dtype=float)
    mask = np.ones(len(pts), dtype=bool)
    center = pts.mean(axis=0)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(iterations):
        sub = pts[mask]
        center = sub.mean(axis=0)
        _, _, vh = np.linalg.svd(sub - center)
        normal = _unit(vh[-1])
        resid = np.abs((pts - center) @ normal)
        thr = np.quantile(resid[mask], keep)
        new_mask = resid <= max(thr, 1e-9)
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            mask = new_mask
            break
        mask = new_mask
    sub = pts[mask]
    center = sub.mean(axis=0)
    _, _, vh = np.linalg.svd(sub - center)
    normal = _unit(vh[-1])
    return center, normal, mask


def validate_margin(margin_points, support_mesh: trimesh.Trimesh) -> MarginValidation:
    """校验 Margin，并对退化输入给出明确状态（不允许静默输出）。

    检查项：点数、闭环性（首尾间距 vs 相邻点中位间距）、平面性（RMS 残差）、
    采样均匀性（相邻间距变异系数）。
    """
    pts = np.asarray(margin_points, dtype=float)
    v = MarginValidation(ok=False, status=AxisStatus.FAILED, n_points=len(pts))
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        v.reasons.append(f"Margin 点数不足（{len(pts)} < 3）")
        return v
    if not np.isfinite(pts).all():
        v.reasons.append("Margin 含非有限坐标")
        return v

    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    median_d = float(np.median(d)) if len(d) else 0.0
    if median_d <= 1e-9:
        v.reasons.append("Margin 相邻点几乎重合，无法定义曲线")
        return v
    v.spacing_cv = float(np.std(d) / median_d)
    v.closure_ratio = float(np.linalg.norm(pts[-1] - pts[0]) / median_d)

    center, normal, inlier_mask = _robust_plane(pts)
    resid = np.abs((pts - center) @ normal)
    v.planarity_rms_mm = float(np.sqrt(np.mean(resid ** 2)))
    v.plane_origin = [float(x) for x in center]
    v.plane_normal = [float(x) for x in normal]

    # 平面内主轴：供锥采样基使用，使候选集随几何一同旋转
    sub = pts[inlier_mask] if inlier_mask.sum() >= 3 else pts
    rel = sub - center
    rel = rel - np.outer(rel @ normal, normal)
    if len(rel) >= 3:
        _, _, vh2 = np.linalg.svd(rel)
        v.in_plane_e1 = [float(x) for x in _unit(vh2[0])]

    # 方向：support 主体应位于法向正侧
    centroid = np.asarray(support_mesh.vertices).mean(axis=0)
    if float(np.dot(centroid - center, normal)) < 0:
        normal = -normal
        v.plane_normal = [float(x) for x in normal]

    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0

    if v.closure_ratio is not None and v.closure_ratio > 3.0:
        v.reasons.append(
            f"Margin 非闭环（首尾间距为相邻点间距的 {v.closure_ratio:.1f} 倍）")
    if v.planarity_rms_mm > 0.25 * span:
        v.reasons.append(
            f"Margin 严重非平面（RMS 残差 {v.planarity_rms_mm:.2f} mm，跨度 {span:.2f} mm）")
    if v.spacing_cv is not None and v.spacing_cv > 1.5:
        v.reasons.append(f"Margin 采样严重不均（间距变异系数 {v.spacing_cv:.2f}）")

    v.ok = not v.reasons
    if v.reasons:
        # 非闭环/非平面/采样不均属于"需要人工确认"，仍可求解；
        # 结构性缺陷（点数不足、坐标非法、点重合）在上面已直接返回 FAILED。
        v.status = AxisStatus.REVIEW_REQUIRED
    else:
        v.status = None
    return v


# ------------------------------------------------------------------ wall axis

def _wall_axis_from_normals(normals: np.ndarray, areas: np.ndarray):
    """面积加权法向协方差的最小特征值方向 + 拟合质量。

    对圆柱侧壁，所有法向都垂直于轴，故 ``Σ w (n·a)²`` 在真实轴处取最小。
    拟合质量 = 最小特征值 / 次小特征值，越小说明法向越集中在一个平面内（越像圆柱）。
    """
    w = areas / areas.sum()
    m = np.einsum("i,ij,ik->jk", w, normals, normals)
    vals, vecs = np.linalg.eigh(m)
    axis = _unit(vecs[:, 0])
    ratio = float(vals[0] / vals[1]) if vals[1] > 1e-12 else float("inf")
    return axis, ratio, [float(x) for x in vals]


def estimate_wall_axis(
    support: trimesh.Trimesh,
    reference_axis,
    *,
    iterations: int = 3,
    wall_cone_deg: float = 60.0,
):
    """侧壁法向拟合轴（迭代选面）。

    以 ``reference_axis`` 为临时轴，选 ``|n·a| < cos(wall_cone_deg 的余角)`` 的面为侧壁，
    再重拟合，迭代数次。返回 ``(axis, quality, n_faces, converged)``。

    **方向约定**：结果定向为与 ``reference_axis`` 同向。

    不要改成"取 +Z 那侧"之类的世界坐标约定 —— 那样的符号规则不是旋转等变的。
    实测：一旦轴被翻转，拔模角 ``n·a`` 全部反号，倒凹比会在刚体变换下跳变
    （面积比从 0.844 变成 0.156，两者互补），而轴夹角本身仍显示 0°，很难发现。
    刚性等变验收测试正是为拦住这类缺陷而设。
    """
    normals = np.asarray(support.face_normals, dtype=float)
    areas = np.asarray(support.area_faces, dtype=float)
    reference_axis = _unit(reference_axis)
    axis = reference_axis
    limit = math.cos(math.radians(90.0 - wall_cone_deg))

    quality = float("inf")
    n_faces = 0
    for _ in range(iterations):
        sel = np.abs(normals @ axis) < limit
        if sel.sum() < 20:
            return axis, float("inf"), int(sel.sum()), False
        n_faces = int(sel.sum())
        axis, quality, _ = _wall_axis_from_normals(normals[sel], areas[sel])
        if float(np.dot(axis, reference_axis)) < 0:
            axis = -axis
    return axis, quality, n_faces, True


# ------------------------------------------------------------------ metrics

#: 掠射过滤阈值：命中面的法向与射线方向的夹角须小于此值的余角，才算"真正的遮挡物"。
#:
#: 必要性（实测教训）：当 Margin 位于**垂直侧壁的底部**时（纯圆柱预备体），
#: 从 Margin 点沿轴发射的射线与侧壁平行 —— 掠射几乎必然误命中某个相邻三角形，
#: 使 margin_visibility 在正确轴处被算成 0，其罚项（权重 0.8）压过倒凹项，
#: 最终把求解器推向一个 17° 倾斜、倒凹 41.6% 的错误轴（正确轴倒凹为 0）。
#: 过滤掉掠射命中后该缺陷消失。
BLOCKER_FACING_DOT = -0.2


def _first_blocker(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    direction: np.ndarray,
    *,
    min_distance_mm: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """沿 ``direction`` 找每条射线的"真正遮挡物"。

    返回 ``(blocked_mask, distance)``。只有满足以下两条的命中才算遮挡：

    1. 命中距离 > ``min_distance_mm``（排除起点附近的自身接触）
    2. 命中面的法向与射线方向显著相对（``n·dir < BLOCKER_FACING_DOT``），
       即确实是迎面挡住射线的面，而不是与射线平行的掠射面

    条件 2 是必需的：没有它，垂直侧壁会被自己的掠射命中误判为遮挡。
    """
    origins = np.asarray(origins, dtype=float)
    d = np.asarray(direction, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    dirs = np.broadcast_to(d, origins.shape)

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        origins, dirs, multiple_hits=False)

    blocked = np.zeros(len(origins), dtype=bool)
    distance = np.zeros(len(origins), dtype=float)
    if len(index_ray) == 0:
        return blocked, distance

    normals = np.asarray(mesh.face_normals, dtype=float)[index_tri]
    facing = normals @ d
    dist = np.linalg.norm(locations - origins[index_ray], axis=1)
    genuine = (dist > min_distance_mm) & (facing < BLOCKER_FACING_DOT)

    blocked[np.unique(index_ray[genuine])] = True
    for k in np.flatnonzero(genuine):
        distance[index_ray[k]] = dist[k]
    return blocked, distance


def _margin_visibility(support: trimesh.Trimesh, margin_pts: np.ndarray,
                       axis: np.ndarray, *, start_offset_mm: float = 1e-3) -> float:
    """Margin 点沿 +axis 的可见比例（已过滤掠射误命中）。

    对**正确的就位方向**，Margin 位于预备体最宽环或缩窄起点，沿轴通常可见。
    若 Margin 落在外扩（flare）之上，可见率会天然偏低 —— 这不是求解错误，
    而是该 Margin 位置与轴不自洽，调用方应把它当作"需人工确认"的信号。
    """
    a = _unit(axis)
    p = np.asarray(margin_pts, dtype=float) + a * start_offset_mm
    blocked, _ = _first_blocker(support, p, a)
    return float(1.0 - blocked.mean()) if len(p) else 0.0


def _undercut_metrics(
    support: trimesh.Trimesh,
    axis: np.ndarray,
    margin_pts: np.ndarray,
    *,
    draft_tol_deg: float = DRAFT_TOLERANCE_DEG,
    shadow_samples: int = 4000,
    seed: int = 17,
):
    """面积加权倒凹指标。

    - ``undercut_area``：拔模负角面的面积（逐面判定，天然面积加权）。
    - 阴影深度：对拔模合格面按面积比例采样后沿轴发射线，取命中距离。
      （拔模合格面的法向与轴不平行，射线数值可靠。）
    """
    a = _unit(axis)
    normals = np.asarray(support.face_normals, dtype=float)
    areas = np.asarray(support.area_faces, dtype=float)
    centers = np.asarray(support.triangles_center, dtype=float)
    total_area = float(areas.sum())

    draft = normals @ a
    tol = math.sin(math.radians(draft_tol_deg))

    # 只统计就位面相关区域：**严格在 Margin 平面之上**的面才有临床意义。
    #
    # 容差必须 > 0 而不是"平面附近的都算进去"：实测在一颗合成无倒凹圆台上，
    # 若把 Margin 平面附近也算入，底面圆盘（法向朝下）会被计入倒凹，贡献
    # 16.17% 的假倒凹 —— 恰好等于底面积占比 π r²/(侧面积+底面+顶面)。
    # 真实病例里 Margin 平面以下同样只是扫描裁切边界或牙龈，不参与就位。
    if len(margin_pts) >= 3:
        try:
            center, normal, _ = _robust_plane(np.asarray(margin_pts, dtype=float))
            if np.dot(normal, a) < 0:
                normal = -normal
            above = (centers - center) @ normal > ABOVE_MARGIN_TOLERANCE_MM
        except Exception:
            above = np.ones(len(areas), dtype=bool)
    else:
        above = np.ones(len(areas), dtype=bool)

    undercut_face = above & (draft < -tol)
    undercut_area = float(areas[undercut_face].sum())
    usable_area = float(areas[above].sum()) or total_area

    # 阴影：拔模合格但仍被真正的遮挡物挡住（同样过滤掠射命中）
    draft_ok = above & (draft >= -tol)
    idx = np.flatnonzero(draft_ok)
    max_depth = 0.0
    mean_depth = 0.0
    if len(idx) and shadow_samples > 0:
        rng = np.random.default_rng(seed)
        probs = areas[idx] / areas[idx].sum()
        pick = rng.choice(idx, size=min(shadow_samples, len(idx)), replace=True, p=probs)
        tri = np.asarray(support.triangles, dtype=float)[pick]
        r1 = np.sqrt(rng.random(len(pick)))
        r2 = rng.random(len(pick))
        pts = ((1 - r1)[:, None] * tri[:, 0]
               + (r1 * (1 - r2))[:, None] * tri[:, 1]
               + (r1 * r2)[:, None] * tri[:, 2])
        blocked, depths = _first_blocker(support, pts, a)
        if blocked.any():
            max_depth = float(depths[blocked].max())
            mean_depth = float(depths[blocked].mean())
            # 阴影面积按"命中样本的面积占比"折算
            shadow_area = float(areas[pick][blocked].sum() / max(len(pick), 1) * len(pick))
            undercut_area += shadow_area

    return {
        "undercut_area_mm2": undercut_area,
        "undercut_area_ratio": undercut_area / usable_area if usable_area > 0 else 0.0,
        "max_undercut_depth_mm": max_depth,
        "mean_undercut_depth_mm": mean_depth,
        "draft_negative_area_mm2": float(areas[undercut_face].sum()),
        "total_area_mm2": total_area,
        "usable_area_mm2": usable_area,
    }


# ------------------------------------------------------------------- search

def _cone_candidates(initial_axis, e1, e2, *, max_deg, step_deg, azimuths):
    out = [(_unit(initial_axis), 0.0)]
    for theta in np.arange(step_deg, max_deg + 1e-9, step_deg):
        t = math.radians(float(theta))
        for phi in np.linspace(0.0, 2.0 * math.pi, azimuths, endpoint=False):
            d = (math.cos(t) * _unit(initial_axis)
                 + math.sin(t) * (math.cos(phi) * e1 + math.sin(phi) * e2))
            out.append((_unit(d), float(theta)))
    return out


def default_weights() -> dict:
    return {"undercut_area": 1.0, "undercut_depth": 0.35,
            "margin_visibility": 0.8, "deviation": 0.25}


def score_axis(
    support: trimesh.Trimesh,
    margin_curve,
    axis,
    *,
    initial_axis,
    max_deviation_deg: float = DEFAULT_MAX_DEVIATION_DEG,
    weights: dict | None = None,
) -> dict:
    """给任意一条候选轴打分 —— 与 :func:`solve_insertion_axis` 内部使用的评分完全一致。

    公开出来的用途：验收测试可以断言"求解结果确实优于任何替代方向"（真正的最优性
    性质），而不是断言某个单一指标最小 —— 综合评分是面积/深度/可见率/偏角的折中，
    单看其中一项并不反映求解器实际优化的目标。
    """
    w = default_weights()
    if weights:
        w.update(weights)

    margin = np.asarray(margin_curve, dtype=float) if margin_curve is not None \
        else np.zeros((0, 3))
    a = _unit(axis)

    metrics = _undercut_metrics(support, a, margin)
    visibility = _margin_visibility(support, margin, a)
    deviation = angle_deg(a, initial_axis)
    depth_scale = float(np.linalg.norm(support.extents)) or 1.0
    depth_norm = min(metrics["mean_undercut_depth_mm"] / (0.05 * depth_scale + 1e-9), 1.0)

    score = (w["undercut_area"] * metrics["undercut_area_ratio"]
             + w["undercut_depth"] * depth_norm
             + w["margin_visibility"] * (1.0 - visibility)
             + w["deviation"] * (deviation / max(max_deviation_deg, 1e-6)))

    return {
        "score": float(score),
        "undercut_area_ratio": float(metrics["undercut_area_ratio"]),
        "undercut_area_mm2": float(metrics["undercut_area_mm2"]),
        "max_undercut_depth_mm": float(metrics["max_undercut_depth_mm"]),
        "mean_undercut_depth_mm": float(metrics["mean_undercut_depth_mm"]),
        "margin_visibility_ratio": float(visibility),
        "deviation_deg": float(deviation),
        "weights": dict(w),
    }


def solve_insertion_axis(
    support_mesh: trimesh.Trimesh,
    margin_curve,
    *,
    neighbors=None,
    max_deviation_deg: float = DEFAULT_MAX_DEVIATION_DEG,
    coarse_step_deg: float = 5.0,
    coarse_azimuths: int = 12,
    refine: bool = True,
    weights: dict | None = None,
) -> InsertionAxisResult:
    """求解就位方向。

    ``neighbors`` 目前仅作为诊断记录；在缺少可靠邻牙分割时不参与评分，
    也绝不会因为没有它而中止求解（``neighbors=None`` 可完整运行）。
    """
    w = default_weights()
    if weights:
        w.update(weights)

    support = support_mesh
    margin = np.asarray(margin_curve, dtype=float) if margin_curve is not None else np.zeros((0, 3))

    validation = validate_margin(margin, support)
    if validation.status is AxisStatus.FAILED:
        return InsertionAxisResult(
            axis=[0.0, 0.0, 1.0], initial_axis=[0.0, 0.0, 1.0], score=float("nan"),
            undercut_area_mm2=0.0, undercut_area_ratio=float("nan"),
            max_undercut_depth_mm=0.0, mean_undercut_depth_mm=0.0,
            margin_visibility_ratio=0.0, deviation_deg=0.0, confidence=0.0,
            status=AxisStatus.FAILED,
            diagnostics={"margin_validation": validation.as_dict(),
                         "reason": "Margin 不可用，无法求解就位方向"},
        )

    # ---- 1. 初始轴：A（Margin 平面法向）+ B（侧壁法向拟合轴）
    axis_a = _unit(validation.plane_normal)
    axis_b, wall_quality, wall_faces, wall_ok = estimate_wall_axis(support, axis_a)
    if float(np.dot(axis_b, axis_a)) < 0:
        axis_b = -axis_b

    disagreement = angle_deg(axis_a, axis_b)
    # 用拟合质量做权重：B 的 eigenvalue ratio 越小越可信
    wb = 1.0 / (1.0 + max(wall_quality, 0.0) * 4.0) if wall_ok else 0.0
    wa = 1.0 - 0.5 * min(disagreement / max(MAX_INITIAL_DISAGREEMENT_DEG, 1e-6), 1.0)
    initial_axis = _unit(axis_a * max(wa, 1e-6) + axis_b * wb)

    centroid = np.asarray(support.vertices).mean(axis=0)
    if float(np.dot(centroid - np.asarray(validation.plane_origin), initial_axis)) < 0:
        initial_axis = -initial_axis

    e1_hint = np.asarray(validation.in_plane_e1, dtype=float) if validation.in_plane_e1 else None
    e1, e2 = _orthonormal_basis(initial_axis, e1_hint=e1_hint)

    # ---- 2. 锥内搜索
    candidates = _cone_candidates(initial_axis, e1, e2,
                                  max_deg=max_deviation_deg, step_deg=coarse_step_deg,
                                  azimuths=coarse_azimuths)

    def evaluate(axis):
        r = score_axis(support, margin, axis, initial_axis=initial_axis,
                       max_deviation_deg=max_deviation_deg, weights=w)
        return (r, np.asarray(axis, dtype=float))

    scored = [evaluate(a) for a, _ in candidates]
    best = min(scored, key=lambda kv: kv[0]["score"])
    best_axis = best[1]

    # ---- 3. 细化：围绕当前最优再做一轮小锥
    if refine:
        e1b, e2b = _orthonormal_basis(best_axis, e1_hint=e1_hint)
        fine = _cone_candidates(best_axis, e1b, e2b,
                                max_deg=max(coarse_step_deg, 2.0), step_deg=1.0,
                                azimuths=max(coarse_azimuths, 8))
        scored += [evaluate(a) for a, _ in fine]
        best = min(scored, key=lambda kv: kv[0]["score"])
        best_axis = best[1]

    metrics = best[0]

    # 方向收敛：统一为 cervical -> occlusal（从 Margin 平面指向 support 主体）。
    # 这一步必须是几何约定而非世界坐标约定，否则破坏旋转等变性。
    centroid = np.asarray(support.vertices).mean(axis=0)
    if float(np.dot(centroid - np.asarray(validation.plane_origin), best_axis)) < 0:
        best_axis = -best_axis
        metrics = score_axis(support, margin, best_axis, initial_axis=initial_axis,
                             max_deviation_deg=max_deviation_deg, weights=w)

    score = metrics["score"]
    vis = metrics["margin_visibility_ratio"]
    dev = metrics["deviation_deg"]
    scores = np.array([s[0]["score"] for s in scored], dtype=float)
    spread = float(scores.max() - scores.min())
    objective_is_flat = spread < FLATNESS_THRESHOLD

    # ---- 4. 状态判定
    reasons: list[str] = list(validation.reasons)
    status = AxisStatus.OK
    if validation.status is AxisStatus.REVIEW_REQUIRED:
        status = AxisStatus.REVIEW_REQUIRED
        reasons.append("Margin 校验未通过（见 margin_validation）")
    if disagreement > MAX_INITIAL_DISAGREEMENT_DEG:
        status = AxisStatus.REVIEW_REQUIRED
        reasons.append(
            f"两个初始轴来源分歧过大（{disagreement:.1f}° > {MAX_INITIAL_DISAGREEMENT_DEG}°）："
            "Margin 平面法向与侧壁拟合轴不自洽")
    if metrics["undercut_area_ratio"] > MAX_ACCEPTABLE_UNDERCUT_RATIO:
        status = AxisStatus.REVIEW_REQUIRED
        reasons.append(
            f"最优轴仍有较大倒凹（面积比 {metrics['undercut_area_ratio']:.3f} "
            f"> {MAX_ACCEPTABLE_UNDERCUT_RATIO}），需要 blockout 或人工调整")
    if vis < MIN_ACCEPTABLE_MARGIN_VISIBILITY:
        status = AxisStatus.REVIEW_REQUIRED
        reasons.append(f"Margin 沿最优轴可见率偏低（{vis:.3f} < {MIN_ACCEPTABLE_MARGIN_VISIBILITY}）")
    if objective_is_flat:
        status = AxisStatus.REVIEW_REQUIRED
        reasons.append(
            f"目标函数在候选集上近似平坦（极差 {spread:.4f} < {FLATNESS_THRESHOLD}）："
            "该病例的就位方向未被目标函数定住，argmin 不具备可信度，必须人工确认")

    # 置信度：拟合质量 + 目标区分度共同决定
    quality_term = 1.0 / (1.0 + max(wall_quality, 0.0) * 4.0) if wall_ok else 0.0
    separation = min(spread / (4.0 * FLATNESS_THRESHOLD), 1.0)
    confidence = float(np.clip(0.5 * quality_term + 0.5 * separation, 0.0, 1.0))
    if objective_is_flat:
        confidence = min(confidence, 0.25)

    return InsertionAxisResult(
        axis=[float(x) for x in best_axis],
        initial_axis=[float(x) for x in initial_axis],
        score=float(score),
        undercut_area_mm2=float(metrics["undercut_area_mm2"]),
        undercut_area_ratio=float(metrics["undercut_area_ratio"]),
        max_undercut_depth_mm=float(metrics["max_undercut_depth_mm"]),
        mean_undercut_depth_mm=float(metrics["mean_undercut_depth_mm"]),
        margin_visibility_ratio=float(vis),
        deviation_deg=float(dev),
        confidence=confidence,
        status=status,
        diagnostics={
            "margin_validation": validation.as_dict(),
            "initial_axis_sources": {
                "margin_plane_normal": [float(x) for x in axis_a],
                "wall_fit_axis": [float(x) for x in axis_b],
                "wall_fit_quality": float(wall_quality),
                "wall_faces": int(wall_faces),
                "wall_fit_converged": bool(wall_ok),
                "disagreement_deg": float(disagreement),
            },
            "search": {
                "n_candidates": len(scored),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "score_spread": spread,
                "flatness_threshold": FLATNESS_THRESHOLD,
            },
            "objective_is_flat": bool(objective_is_flat),
            "best_axis_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in metrics.items()},
            "weights": w,
            "neighbors_provided": neighbors is not None,
            "neighbors_note": (
                "邻牙未参与评分：缺少可靠邻牙分割时，邻牙碰撞项会把求解器拖向不可靠的结论。"
                "本字段仅作记录。" if neighbors is None else
                "已提供邻牙，但当前版本仍未将其纳入评分，仅记录。"),
            "max_deviation_deg_source": (
                "Lab Profile 实测证据（CrownBottomInsertionAngleBetweenTeethLimit 25/25 = 20）。"
                "该字段在 exocad 公开的 Material Configuration Guide 中未定义，"
                "属实验室级证据而非全局规律。"),
            "reasons": reasons,
        },
    )


def suggest_insertion_axis(support_mesh, margin_points=None):
    """兼容旧调用点的薄包装：返回单纯的轴向量。

    新代码请直接使用 :func:`solve_insertion_axis` 以获得状态、置信度与诊断。
    """
    if margin_points is None or len(margin_points) < 3:
        # 无 Margin 时退化为侧壁拟合轴（仍不使用顶点 PCA）。
        # 方向用几何约定（质心相对包围盒中心）确定，避免依赖世界 Z。
        #
        # 必须先判断零向量再归一化：对称网格（如居中的 cylinder）的质心与包围盒
        # 中心重合，直接 _unit() 会抛异常 —— 这是实测踩到过的崩溃点。
        raw = (np.asarray(support_mesh.vertices).mean(axis=0)
               - np.asarray(support_mesh.bounds).mean(axis=0))
        if not np.isfinite(raw).all() or float(np.linalg.norm(raw)) < 1e-9:
            raw = np.array([0.0, 0.0, 1.0])
        axis, _, _, _ = estimate_wall_axis(support_mesh, _unit(raw))
        return axis
    return np.asarray(solve_insertion_axis(support_mesh, margin_points).axis, dtype=float)
