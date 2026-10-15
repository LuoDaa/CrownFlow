"""Occlusion 域测量的**特征测试**（B1 第一步：先把旧行为钉住，再谈统一口径）。

``occlusion.py`` 的"符号"和 Contact **不是同一件事**，B1 的结论是**保留它自己的领域语义**，
只共用 validity / closest primitive，并把语义显式标注出来。本文件把六类场景的现状钉住：

1. 闭合对颌实体（watertight antagonist）——符号从哪来（一次性质心定向，**不是**实体内外判定）；
2. 开放对颌裁片（open patch）——validity 只有边界安全，没有折痕掩码；
3. 无效 / 歧义顶点——是否进入统计（``ambiguous`` 计数 + hit 集合排除）；
4. 没有外向法向 / 没有可识别颈环——抛什么；
5. 三角形相交（crossing）——**本模块不负责**，由服务层 ``surface_crossings`` 审（职责边界）；
6. 零 / 近零间隙——零间隙不动、穿透按"相对目标间隙的亏欠"计、锁定区绝不动。

**这些断言的价值**在于：任何一次"顺手改成 Contact 口径"都会让本文件红掉，
从而强制先说明"occlusion 的 penetration 与 contact 的 penetration 不是同一个量"。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.contact import NeighborPatch
from app.geometry.crown_outer.occlusion import (OcclusionError, OcclusionParams,
                                                solve_occlusion_collision)
from app.geometry.crown_outer.patient_frame import PatientFrame

FRAME = PatientFrame(np.zeros(3), np.array([0.0, 0.0, 1.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))


def crown(*, top_z: float = 5.0, open_bottom: bool = True) -> trimesh.Trimesh:
    """一颗带颈环的冠：闭合时无边界环，``open_bottom=True`` 时底部是颈环。"""
    theta = np.arange(48) * 2 * np.pi / 48
    ring = np.c_[2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(48)]
    top = np.array([[0.0, 0.0, top_z]])
    vertices = np.vstack((ring, top))
    apex = len(ring)
    faces = [[i, (i + 1) % len(ring), apex] for i in range(len(ring))]
    if not open_bottom:
        center = np.array([[0.0, 0.0, 0.0]])
        bottom = len(vertices)
        vertices = np.vstack((vertices, center))
        faces += [[(i + 1) % len(ring), i, bottom] for i in range(len(ring))]
    return trimesh.Trimesh(vertices, np.asarray(faces, np.int64), process=False)


def antagonist_plate(*, z: float, half: float = 4.0) -> trimesh.Trimesh:
    """开放对颌裁片（一块平板，有开放边界）。"""
    return trimesh.Trimesh(
        np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]]),
        np.array([[0, 1, 2], [0, 2, 3]], np.int64), process=False)


def antagonist_solid(*, z: float, half: float = 4.0) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(2 * half, 2 * half, 1.0))\
        .apply_translation([0, 0, z + 0.5])


def patch_of(mesh) -> NeighborPatch:
    return NeighborPatch(mesh.vertices, mesh.faces)


# ---------------------------------------------------------------- 1. 符号来源

def test_unreviewed_open_patch_is_unsigned_only():
    """**O1 之后的事实**：开放裁片没有调用方声明的外向法向 ⇒ ``UNSIGNED_ONLY``。

    O1 之前这里会用一次性质心投票猜一个符号、照样给出穿透结论。
    现在：不下穿透结论、状态 NOT_ASSESSED、不动几何（见 sign matrix 第 5 行）。
    """
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                   patch_of(antagonist_plate(z=4.6)),
                                   params=OcclusionParams(target_gap_mm=0.05))
    assert res.sign_method == "UNSIGNED_ONLY"
    assert res.signed_available is False
    assert res.history[0]["penetration_mm"] is None
    assert res.status == "NOT_ASSESSED"
    assert res.modified_vertex_count == 0


def test_sign_source_is_solid_containment_for_closed_antagonist():
    """闭合可定向对颌 ⇒ 符号来自**实体内外判定**，与 Contact 同一个 primitive。

    （O1 之前这里是一次性质心投票 `ONE_SHOT_CENTROID_VOTE`；该做法已退役。）
    """
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                   patch_of(antagonist_solid(z=4.6)),
                                   params=OcclusionParams(target_gap_mm=0.05))
    assert res.sign_method == "SOLID_CONTAINMENT"
    assert res.signed_available is True
    assert res.history[0]["count"] > 0, "冠顶在对颌实体内 ⇒ 必须识别为穿透"
    assert res.history[0]["penetration_mm"]["max_mm"] > 0
    assert res.history[0]["max"] == pytest.approx(0.5, abs=0.05), res.history[0]


def test_reviewed_open_patch_uses_declared_normal():
    """开放裁片 + **调用方声明**的外向法向 ⇒ ``REVIEWED_NORMAL_LOCAL``（局部可信）。

    声明方式是显式的（``antagonist_outward=`` 或直接传 ``FitSurface``），
    不再是"猜一个"，报告里也写得出是哪一种。
    """
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface

    mesh = crown(top_z=5.0)
    plate = antagonist_plate(z=4.6)
    res = solve_occlusion_collision(mesh, FRAME, patch_of(plate),
                                   params=OcclusionParams(target_gap_mm=0.05),
                                   antagonist_outward=[0.0, 0.0, -1.0])
    assert res.sign_method == "REVIEWED_NORMAL_LOCAL"
    assert res.history[0]["penetration_mm"]["max_mm"] == pytest.approx(0.4, abs=0.05)
    assert res.history[0]["max"] == pytest.approx(0.45, abs=0.02)

    surface = FitSurface(plate, [0.0, 0.0, -1.0])
    res2 = solve_occlusion_collision(mesh, FRAME, surface,
                                     params=OcclusionParams(target_gap_mm=0.05))
    assert res2.sign_method == "REVIEWED_NORMAL_LOCAL"
    assert res2.history[0]["max"] == pytest.approx(res.history[0]["max"], abs=1e-9)


# ---------------------------------------------------------------- 2/3. validity

def test_validity_is_boundary_safety_only_no_crease_mask():
    """开放裁片的 validity 只剔除**边界带**；本模块没有 ``clearance_detail`` 的折痕掩码。"""
    p = OcclusionParams(target_gap_mm=0.05, boundary_safety_mm=0.6)
    # 一块很窄的裁片：冠顶投影落在裁片外/边界带上 ⇒ 大量点判为 ambiguous
    narrow = patch_of(antagonist_plate(z=4.6, half=0.3))
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME, narrow, params=p,
                                    antagonist_outward=[0.0, 0.0, -1.0])
    assert res.history[0]["ambiguous"] > 0, "窄裁片应产生无效采样"
    assert res.history[0]["count"] == 0, "无效采样不得进入干涉统计"


def test_invalid_vertices_are_excluded_from_stats_but_counted():
    """无效顶点不参与统计，但必须在报告里可数（``ambiguous``）。"""
    p = OcclusionParams(target_gap_mm=0.05, boundary_safety_mm=0.6)
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                   patch_of(antagonist_plate(z=4.6, half=4.0)), params=p,
                                   antagonist_outward=[0.0, 0.0, -1.0])
    entry = res.history[0]
    assert entry["ambiguous"] == 0, "大裁片 + 冠顶落在中部 ⇒ 无无效采样"
    assert set(entry) >= {"count", "p50", "p95", "max", "sum", "ambiguous"}


# ---------------------------------------------------------------- 4. 前置条件

def test_missing_cervical_ring_raises_with_reason():
    """没有可识别颈环的网格必须抛 OcclusionError（不能默默当成没有锁定区）。"""
    closed = crown(top_z=5.0, open_bottom=False)
    outcome = None
    try:
        solve_occlusion_collision(closed, FRAME, patch_of(antagonist_plate(z=4.6)))
    except OcclusionError as exc:
        outcome = str(exc)
    assert outcome is not None and "颈环" in outcome, outcome


# ---------------------------------------------------------------- 5. 职责边界

def test_signed_distance_does_not_imply_absence_of_crossing():
    """**第 6 类场景（O1 补）**：signed 全为正 ⇏ 没有表面相交。

    低分辨率网格上可以出现"三角形 A 穿过三角形 B，但**所有采样顶点都在实体外**"：
    顶点距离全为正，而表面确实相交。所以

        signed_min >= 0        ⇏        surface_crossings == 0

    签名距离回答"查询点的方向距离"，`surface_crossings` 回答"表面是否相交" ——
    两个问题，两个模块（本模块不负责相交审计，见 `measurement_core` 契约表）。
    这条断言在 O1 前后都必须成立：换符号 primitive 不许把两个概念混起来。
    """
    from app.geometry.crown_outer.auto_fit_adapter import FitSurface, surface_crossings

    slab = trimesh.creation.box(extents=(4.0, 4.0, 0.2)).apply_translation([0, 0, 0.5])
    # 三个顶点都在板外（z=0 / z=1，板体 z∈[0.4,0.6]），但面**穿过**板
    blade = trimesh.Trimesh(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                            np.array([[0, 1, 2]], np.int64), process=False)
    surface = FitSurface(slab)
    _cp, _fids, signed, _dist, valid = surface.query(np.asarray(blade.vertices))
    assert (signed > 0).all() and bool(valid.all()),         "三个顶点都在对颌实体之外 ⇒ signed 全正"
    crossings = surface_crossings(blade, surface, np.ones(len(blade.vertices), bool))
    assert crossings["crossing_pairs"] >= 1,         "顶点 s>0 但表面确实相交 —— 这正是不能被 signed 距离替代的那件事"


def test_crossing_audit_is_not_this_module_job():
    """三角形相交（crossing）不在本模块职责内 —— 由服务层 `surface_crossings` 审。

    这里钉住的是**职责边界**：即使冠与对颌明显相交，本模块也只报穿透统计，
    不会宣称做过相交审计（避免"看起来审过了"）。
    """
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                   patch_of(antagonist_plate(z=4.6)),
                                   params=OcclusionParams(target_gap_mm=0.05))
    assert not hasattr(res, "surface_crossings")
    assert all("crossing" not in key for key in res.history[0])


# ---------------------------------------------------------------- 6. 零/近零间隙

def test_zero_clearance_moves_nothing_and_target_clearance_converges():
    """**恰好零间隙不算干涉**（``hit`` 判据是 ``signed < 0`` 严格小于）。

    这是本模块的一个真实边界：冠顶正好贴在对颌面上时，亏欠虽为 ``target_gap``，
    但因为不存在真穿透，hit 集合为空 ⇒ 不动、报 0。把它钉住，
    避免以后有人把判据改成 ``<= 0`` 而悄悄改变"贴面即干涉"的行为。
    """
    p = OcclusionParams(target_gap_mm=0.05, inside_reach_mm=2.0, pen_tolerance_mm=0.05)
    res = solve_occlusion_collision(crown(top_z=4.6), FRAME,
                                   patch_of(antagonist_plate(z=4.6)), params=p,
                                   antagonist_outward=[0.0, 0.0, -1.0])
    assert res.history[0]["count"] == 0, "零间隙不得进入 hit 集合"
    assert res.history[0]["max"] == 0.0
    assert res.modified_vertex_count == 0, "≤ 容差/无真穿透时不得移动"
    assert res.status == "OK"

    # 只要有**真穿透**（哪怕 1 µm），hit 立刻非空，且亏欠 = 真穿透 + 目标间隙
    tiny = solve_occlusion_collision(crown(top_z=4.601), FRAME,
                                    patch_of(antagonist_plate(z=4.6)),
                                    params=OcclusionParams(target_gap_mm=0.05, inside_reach_mm=2.0),
                                    antagonist_outward=[0.0, 0.0, -1.0])
    assert tiny.history[0]["count"] > 0
    assert tiny.history[0]["max"] == pytest.approx(0.051, abs=0.005)

    # 真的穿透 0.4 mm：应移动，且颈环/锁定区位移严格为 0
    res2 = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                    patch_of(antagonist_plate(z=4.6)), params=p,
                                    antagonist_outward=[0.0, 0.0, -1.0])
    assert res2.modified_vertex_count > 0
    assert res2.margin_max_displacement_mm == 0.0
    assert res2.proximal_core_max_displacement_mm == 0.0
    assert res2.n_flipped_faces == 0
    assert res2.history[-1]["p95"] < res2.history[0]["p95"], "line search 必须单调改善"


def test_penetration_is_target_relative_and_hit_set_is_true_penetration():
    """两个量的口径必须分开看（这正是"不能和 Contact 混"的地方）：

    - ``hit`` 集合 = **真穿透**（``signed < 0``）且在对颌 2 mm 内；
    - 报告出的 ``pen`` = ``target_gap − signed``（**相对目标间隙的亏欠**），真穿透时为
      ``真穿透 + target_gap``。
    """
    p = OcclusionParams(target_gap_mm=0.10, inside_reach_mm=2.0)
    res = solve_occlusion_collision(crown(top_z=5.0), FRAME,
                                   patch_of(antagonist_plate(z=4.8)), params=p,
                                   antagonist_outward=[0.0, 0.0, -1.0])
    # 真穿透 0.2 mm ⇒ 亏欠 ≈ 0.30，而不是 0.20
    assert res.history[0]["max"] == pytest.approx(0.30, abs=0.02), res.history[0]
