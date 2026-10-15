"""O1：Occlusion 符号来源正确性 —— **验收矩阵**（冻结的 6 行 + 2 条边界）。

这一批只修"符号来源是否可信"，不动 occlusion 的目标间隙、Gate、统计口径。
矩阵把"每一行的 sign_method / signed 预期 / 能否下穿透结论 / deficit"钉死：

| 场景 | sign_method | signed | 穿透结论 | deficit |
|---|---|---|---|---|
| watertight 不相交 | SOLID_CONTAINMENT | > 0 | 0 | max(target−signed,0) |
| watertight 轻微穿透 | SOLID_CONTAINMENT | < 0 | 可下结论 | target + penetration |
| 冠顶点在对颌实体内 | SOLID_CONTAINMENT | < 0 | 可下结论 | 正值 |
| open patch + 已复核外向法向 | REVIEWED_NORMAL_LOCAL | 正负可用 | **局部可信** | 正常计算 |
| open patch 无已复核法向 | UNSIGNED_ONLY | **不提供** | **必须 NOT_ASSESSED** | 只报 clearance shortfall |
| 边界：d < ε（接触点） | 任一种 | 不许随机翻号 | 视为歧义，不下结论 | 按歧义计 |
| 职责：signed_min ≥ 0 | — | — | **不能推出** crossings == 0 | — |

命名与 Contact 共享：`SOLID_CONTAINMENT` / `REVIEWED_NORMAL_LOCAL` 是**同一个 primitive**
产出的同一种符号（`measurement_core`），域名 interpretation 不同：
Contact `penetration = max(−signed, 0)`；Occlusion `deficit = max(target_gap − signed, 0)`。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry import measurement_core as MC
from app.geometry.crown_outer.contact import NeighborPatch
from app.geometry.crown_outer.occlusion import (OcclusionParams, solve_occlusion_collision)
from app.geometry.crown_outer.patient_frame import PatientFrame

FRAME = PatientFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def safe_sign(value: float) -> int:
    return 0 if not np.isfinite(value) or abs(value) < 1e-12 else int(np.sign(value))


def crown(*, top_z: float = 5.0) -> trimesh.Trimesh:
    """带颈环的锥形冠（底部开放 = 颈环）。"""
    theta = np.arange(48) * 2 * np.pi / 48
    ring = np.c_[2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(48)]
    apex = np.array([[0.0, 0.0, top_z]])
    vertices = np.vstack((ring, apex))
    faces = [[i, (i + 1) % 48, 48] for i in range(48)]
    return trimesh.Trimesh(vertices, np.asarray(faces, np.int64), process=False)


def plate_patch(*, z: float, half: float = 6.0) -> NeighborPatch:
    """开放对颌裁片（网格板，法向朝 −Z = 朝冠）。"""
    grid = 12
    verts = [(x, y, z) for x in np.linspace(-half, half, grid) for y in np.linspace(-half, half, grid)]
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a, b, c, d = i * grid + j, i * grid + j + 1, (i + 1) * grid + j, (i + 1) * grid + j + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return NeighborPatch(np.asarray(verts, float), np.asarray(faces, np.int64))


def solid_antagonist(*, bottom_z: float) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(10.0, 10.0, 1.0)).apply_translation([0, 0, bottom_z + 0.5])


def _run(antagonist, *, target_gap=0.05, **kwargs):
    return solve_occlusion_collision(crown(), FRAME, antagonist,
                                     params=OcclusionParams(target_gap_mm=target_gap, inside_reach_mm=2.0),
                                     **kwargs)


# ---------------------------------------------------------------- 1. 不相交

def test_row1_solid_no_intersection_reports_gap_and_zero_penetration():
    res = _run(solid_antagonist(bottom_z=5.6))          # 冠顶 5.0，对颌底 5.6 ⇒ 0.6 间隙
    first = res.history[0]
    assert res.sign_method == MC.SIGN_SOLID_CONTAINMENT
    assert first["signed_available"] is True
    assert first["signed_min_mm"] > 0, first
    assert first["penetration_mm"] is None or first["penetration_mm"]["max_mm"] == 0.0
    # deficit = max(target − signed, 0)：间隙大于目标 ⇒ 亏欠 0
    assert first["clearance_shortfall_mm"]["max_mm"] == 0.0
    assert first["count"] == 0
    assert res.modified_vertex_count == 0
    assert res.status == "OK"


# ---------------------------------------------------------------- 2. 轻微穿透

def test_row2_solid_slight_penetration_is_conclusive():
    res = _run(solid_antagonist(bottom_z=4.9))          # 冠顶 5.0 穿进对颌 0.1
    first = res.history[0]
    assert res.sign_method == MC.SIGN_SOLID_CONTAINMENT
    assert first["signed_min_mm"] < 0
    assert first["penetration_mm"]["max_mm"] == pytest.approx(0.1, abs=0.02)
    # deficit = target + penetration
    assert first["clearance_shortfall_mm"]["max_mm"] == pytest.approx(0.15, abs=0.03)


# ---------------------------------------------------------------- 3. 冠顶点在实体内

def test_row3_engulfed_crown_vertex_is_penetration_not_no_interference():
    """O1 的核心修复：质心投票曾把"冠被对颌包住"判成无干涉。

    几何：对颌实体 z∈[4.2, 5.2]，冠顶 5.0 在其内部 ⇒ 到最近实体面（顶面 5.2）
    的距离 0.2 就是穿透深度。**关键不是这个数字，而是它不再被报成"无干涉"。**
    """
    res = _run(solid_antagonist(bottom_z=4.2))
    first = res.history[0]
    assert res.sign_method == MC.SIGN_SOLID_CONTAINMENT
    assert first["signed_min_mm"] < 0, "实体内 ⇒ 必须判为穿透"
    assert first["penetration_mm"]["max_mm"] == pytest.approx(0.2, abs=0.05)
    assert first["count"] > 0, "不得报成无干涉（旧质心投票的失效模式）"
    assert first["clearance_shortfall_mm"]["max_mm"] == pytest.approx(0.25, abs=0.05)  # 0.2 + 目标 0.05
    assert res.modified_vertex_count > 0, "确凿穿透必须触发修正"


# ---------------------------------------------------------------- 4. 已复核的开放裁片

def test_row4_reviewed_open_patch_keeps_local_signed():
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface

    surface = FitSurface(plate_mesh(z=4.6), [0.0, 0.0, -1.0])     # 人工声明外向法向
    res = _run(surface)
    first = res.history[0]
    assert res.sign_method == MC.SIGN_REVIEWED_NORMAL_LOCAL
    assert first["signed_available"] is True and first["signed_min_mm"] < 0
    assert first["penetration_mm"]["max_mm"] == pytest.approx(0.4, abs=0.05)


def plate_mesh(*, z: float, half: float = 6.0) -> trimesh.Trimesh:
    grid = 12
    verts = [(x, y, z) for x in np.linspace(-half, half, grid) for y in np.linspace(-half, half, grid)]
    faces = []
    for i in range(grid - 1):
        for j in range(grid - 1):
            a, b, c, d = i * grid + j, i * grid + j + 1, (i + 1) * grid + j, (i + 1) * grid + j + 1
            faces.append([a, b, c])
            faces.append([b, d, c])
    return trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces, np.int64), process=False)


# ---------------------------------------------------------------- 5. 未复核的开放裁片

def test_row5_unreviewed_open_patch_never_concludes_penetration():
    """**O1 第 4 条验收标准**：没有已复核法向的开放裁片 ⇒ 只报 clearance shortfall。

    两个子场景都钉住同一件事：**"咬穿"没有结论**（`penetration_mm is None`、NOT_ASSESSED、不动网格）。
    shortfall 用的是 unsigned 幅值：``max(target_gap − unsigned, 0)``。
    """
    # (a) 幅值 0.4 > 目标 0.05：几何上看着像穿透，但符号不可信 ⇒ shortfall 也是 0（不能反推方向）
    far = _run(plate_patch(z=4.6))
    first = far.history[0]
    assert far.sign_method == MC.SIGN_UNSIGNED_ONLY
    assert first["signed_available"] is False
    assert first["signed_min_mm"] is None, "不提供 signed 值（不是 0，也不是猜出来的）"
    assert first["penetration_mm"] is None, "**不得**把间隙不足说成穿透"
    assert first["clearance_shortfall_mm"]["max_mm"] == 0.0, "0.4 > 目标 0.05 ⇒ 幅值上不缺间隙"
    assert far.modified_vertex_count == 0, "符号不可信时不得动几何"
    assert far.status == "NOT_ASSESSED"
    assert any("SIGN" in note or "符号" in note for note in far.notes), far.notes

    # (b) 幅值 0.02 < 目标 0.05：shortfall 报 0.03，但**仍然**没有穿透结论
    near = _run(plate_patch(z=5.02))
    near_first = near.history[0]
    assert near.sign_method == MC.SIGN_UNSIGNED_ONLY
    assert near_first["clearance_shortfall_mm"]["max_mm"] == pytest.approx(0.03, abs=0.01)
    assert near_first["penetration_mm"] is None
    assert near.status == "NOT_ASSESSED" and near.modified_vertex_count == 0


# ---------------------------------------------------------------- 边界：ε 附近

def test_boundary_epsilon_makes_contact_points_ambiguous_instead_of_flipping_sign():
    """d < ε 的接触点必须判为**歧义**，不许出现 ±1e-12 级别的随机翻号。"""
    surface = MC.SignSource.for_mesh(solid_antagonist(bottom_z=5.0))    # 冠顶正好贴面
    assert surface.method == MC.SIGN_SOLID_CONTAINMENT
    points = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0 + 1e-9], [0.0, 0.0, 5.2]])
    _cp, _fids, signed, distance, valid = surface.query(points)
    assert not bool(valid[0]), "ε 以内视为歧义（不判内外）"
    assert bool(valid[2]) is True
    assert distance[0] <= surface.epsilon_mm + 1e-12, "歧义点确实在 ε 带里"
    assert safe_sign(signed[0]) in (-1, 0, 1), "不许出现 ±1e-12 级别的随机符号被当成结论"


# ---------------------------------------------------------------- 职责分离

def test_signed_distance_and_surface_crossing_are_separate_questions():
    """signed_min ≥ 0 **不能**推出没有表面相交（低分辨率网格上的经典反例）。"""
    from app.geometry.crown_outer.auto_fit_adapter import surface_crossings

    from app.geometry.crown_outer.auto_fit_adapter import FitSurface

    slab = trimesh.creation.box(extents=(4.0, 4.0, 0.2)).apply_translation([0, 0, 0.5])
    blade = trimesh.Trimesh(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                            np.array([[0, 1, 2]], np.int64), process=False)
    source = MC.SignSource.for_mesh(slab)
    _cp, _fids, signed, _distance, valid = source.query(np.asarray(blade.vertices))
    assert source.method == MC.SIGN_SOLID_CONTAINMENT
    assert (signed > 0).all() and bool(valid.all()), "顶点全在实体外 ⇒ signed 全正"
    report = surface_crossings(blade, FitSurface(slab), np.ones(len(blade.vertices), bool))
    assert report["crossing_pairs"] >= 1, "表面确实相交 —— 与 signed 是两个问题"
