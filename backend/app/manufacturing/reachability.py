"""刀具可达性。

**本模块当前状态：只交付已验证正确的构件，完整的可加工性判定尚未实现。**

原因（实测于真实 exocad 最终冠 ``1629-王煜 1.stl``）：

1. 牙冠是**壳**（含内表面），外表面与内表面的可加工路径完全不同：
   外表面从外部下刀，内表面只能从颈缘开口进入。用一个固定方向测全部表面
   会把内表面全部判为不可达 —— 那是错的。
2. 固定轴的可见率在真实冠上只有约 54%（外表面沿 +Z 遮挡 45.9%），
   **但这不代表不可加工**：真实 3 轴加工会把牙冠按优化角度装夹，
   正确的问题是"是否存在一个装夹方向使全部外表面可达"。
3. 因此"不可加工"的判定需要一个装夹方向优化 + 内外表面分离的加工路径模型，
   而不是逐点切线平面测试。在完成之前，任何 ``machinable=False`` 都可能是假闸门，
   会把合格设计挡掉。

已经验证可用、可以安全依赖的构件（均有实测记录）：

- :class:`ManufacturingProfile` 与 3 轴 / 5 轴分离
- :func:`candidate_directions` —— 3 轴 1 个方向、5 轴 19 个方向
- :func:`split_outer_intaglio` —— 外/内表面分离（真实冠 73.9% / 26.1%）
- :func:`visibility_fraction` —— 单向可见性测量（sound primitive）

**未实现**：装夹方向优化、内表面经颈缘开口的可达性、刀体-特征干涉（
"局部曲率半径 > 刀具半径"只是必要条件，且不能单独作为通过依据）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh

from .models import ManufacturingProfile


class ReachabilityStatus(str, Enum):
    """可达性判定的置信状态。"""

    #: 全部表面在给定方向集合下可达。
    REACHABLE = "REACHABLE"
    #: 存在不可达区域 —— **仅在完整求解器实现后才可信**。
    UNREACHABLE = "UNREACHABLE"
    #: 尚未实现生产级判定。**不得作为拒绝设计的依据。**
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


def split_outer_intaglio(crown: trimesh.Trimesh) -> np.ndarray:
    """按面法向相对质心的朝向分离外表面（True）与内表面（False）。

    对壳状牙冠有效：外表面片法向背离质心（点积 > 0），内表面片法向指向内腔
    （点积 < 0）。实测于 ``1629-王煜 1.stl``：外表面 29858 片 / 内表面 10568 片。
    """
    centers = np.asarray(crown.triangles_center)
    normals = np.asarray(crown.face_normals)
    centroid = np.asarray(crown.vertices).mean(axis=0)
    return np.einsum("ij,ij->i", normals, centers - centroid) > 0


def visibility_fraction(
    crown: trimesh.Trimesh,
    direction: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
    n_samples: int = 2000,
    seed: int = 17,
) -> float:
    """测量给定方向上表面的可见比例（不可见 = 射线在离开前再次命中本体）。

    这是 sound primitive：语义明确，不做"可加工"的推断。
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)

    if face_mask is None:
        face_mask = np.ones(len(crown.faces), dtype=bool)

    idx = np.flatnonzero(face_mask)
    if len(idx) == 0:
        return 0.0

    rng = np.random.default_rng(seed)
    areas = np.asarray(crown.area_faces, dtype=float)[idx]
    probs = areas / areas.sum()
    pick = rng.choice(idx, size=n_samples, replace=True, p=probs)
    tri = np.asarray(crown.triangles, dtype=float)[pick]
    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    pts = ((1 - r1)[:, None] * tri[:, 0]
           + (r1 * (1 - r2))[:, None] * tri[:, 1]
           + (r1 * r2)[:, None] * tri[:, 2])

    dirs = np.broadcast_to(direction, pts.shape)
    hits = crown.ray.intersects_first(pts + dirs * 1e-4, dirs)
    return float((hits < 0).mean())


def candidate_directions(
    insertion_axis: np.ndarray,
    profile: ManufacturingProfile,
    *,
    half_angle_deg: float = 20.0,
    n_directions: int = 21,
) -> np.ndarray:
    """构造候选刀具方向。

    - **3 轴**：只有固定加工轴一个方向。
    - **5 轴**：在加工轴周围 ``half_angle_deg`` 锥内采样（实测
      ``CrownBottomInsertionAngleBetweenTeethLimit = 20``，故默认 20°）。

    实测输出：3 轴 → 1 个方向；5 轴 → 19 个方向。
    """
    axis = np.asarray(insertion_axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    if profile.is_three_axis:
        return axis[None, :]

    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= max(float(np.linalg.norm(e1)), 1e-12)
    e2 = np.cross(axis, e1)

    dirs = [axis]
    half = np.radians(half_angle_deg)
    per_ring = max(4, (n_directions - 1) // 3)
    for theta in np.linspace(half / 3.0, half, 3):
        for phi in np.linspace(0.0, 2.0 * np.pi, per_ring, endpoint=False):
            d = (np.cos(theta) * axis
                 + np.sin(theta) * (np.cos(phi) * e1 + np.sin(phi) * e2))
            dirs.append(d / np.linalg.norm(d))
    return np.array(dirs)


@dataclass
class ReachabilityResult:
    """可达性测量结果。

    ``machinable`` 为 ``None`` 表示**未知**，不代表不可加工。调用方不得把
    ``None`` 当作 False 来做拒绝决策。
    """

    status: ReachabilityStatus
    machinable: bool | None
    #: 该判定是否可作为拒绝设计的依据。当前恒为 False。
    is_authoritative: bool
    machine_axes: int
    tool_radius_mm: float
    candidate_direction_count: int
    #: 诊断数据，供人工核对与后续求解器使用。
    diagnostics: dict = field(default_factory=dict)
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "status": self.status.value,
            "machinable": self.machinable,
            "is_authoritative": self.is_authoritative,
            "machine_axes": self.machine_axes,
            "tool_radius_mm": self.tool_radius_mm,
            "candidate_direction_count": self.candidate_direction_count,
            "diagnostics": self.diagnostics,
            "note": self.note,
        }


def evaluate_reachability(
    crown: trimesh.Trimesh,
    profile: ManufacturingProfile,
    insertion_axis: np.ndarray,
    *,
    n_samples: int = 800,
) -> ReachabilityResult:
    """测量可达性诊断数据，但**不给出生产级可加工结论**。

    返回 ``status = NOT_IMPLEMENTED`` 与 ``machinable = None``，并附带：
    - 外表面 / 内表面的采样数
    - 每个候选方向下外表面的可见比例（以及最佳值）

    这些数字可用于研发比较（例如不同刀具直径、不同装夹角度的相对优劣），
    但不能用于判定"这个设计做不出来"。
    """
    if len(crown.faces) == 0:
        raise ValueError("crown 为空网格")

    outer_mask = split_outer_intaglio(crown)
    n_outer = int(outer_mask.sum())
    n_intaglio = int((~outer_mask).sum())
    directions = candidate_directions(insertion_axis, profile)

    per_direction = []
    for d in directions:
        frac_outer = visibility_fraction(crown, d, face_mask=outer_mask, n_samples=n_samples)
        per_direction.append(float(frac_outer))

    best = max(per_direction) if per_direction else 0.0

    return ReachabilityResult(
        status=ReachabilityStatus.NOT_IMPLEMENTED,
        machinable=None,
        is_authoritative=False,
        machine_axes=int(profile.machine_axes),
        tool_radius_mm=profile.tool_radius_mm,
        candidate_direction_count=len(directions),
        diagnostics={
            "outer_faces": n_outer,
            "intaglio_faces": n_intaglio,
            "outer_visibility_per_direction": [round(v, 4) for v in per_direction],
            "best_outer_visibility": round(best, 4),
            "sampled_per_direction": n_samples,
        },
        note=(
            "可达性判定尚未实现生产版本。缺失部分：(1) 装夹方向优化——3 轴加工会把"
            "牙冠按优化角度装夹，固定轴可见率不能代表可加工性；(2) 内表面经颈缘开口的"
            "可达性——内表面无法从外部下刀，必须单独建模；(3) 刀体-特征干涉——"
            "『局部曲率半径 > 刀具半径』只是必要条件。在这些完成前，"
            "machinable 保持 None，任何基于它的拒绝决策都是假闸门。"
        ),
    )
