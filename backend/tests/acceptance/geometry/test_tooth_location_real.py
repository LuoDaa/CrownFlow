"""目标牙定位 / Support 分割的真实病例验收 —— **fixture 不进 Git**。

病例：FDI36 基准病例的 ``01_主要病例模型/Unsectioned_LowerJaw.stl``。
真值来自同一坐标系里的基台模型 ``1629-王煜 0.stl`` ——
实测基台顶点到整颌表面的中位距离是 **0.000 mm**，说明两者同一坐标系，
而且**整颌网格里确实包含了那颗被扫描的基台**。所以这里可以断言
"定位到的位置就是基台所在的位置"，不是自说自话。

## 关于"交界有没有凹脊"（一条被推翻的中间结论，留在这里备查）

中途我曾断言"基台/牙龈交界没有凹脊"，理由是实测 max 只有 0.054。
**那是错的**：那个数字来自只有 **11** 个顶点的样本（用 ``d < 0.3 mm`` 匹配
基台顶点到整颌顶点，龈方端只剩 11 个）。把匹配容差放宽到 0.5 mm 之后：

    龈方端 30%：n=46    mean 0.547  p90 0.862  max 0.873
    龈方端 45%：n=1176  mean 0.226  p90 0.868  max 0.971

交界上的凹脊很强。所以这个病例**可以**依附解剖边界，
``boundary_is_anatomical`` 实测为 True。教训是：拿"最大值"下结论之前
先看样本量 —— 11 个顶点不足以支撑一个否定结论。

## 当前真实状态

    locate  : OK，置信度 0.689，区域半径 12.1 mm，切口前沿凹度 0.401
    support : REVIEW_REQUIRED，置信度 0.930
              单连通 ✓  目标保留 0.9998 ✓  面积比 2.07 ✓  边界依附凹脊 ✓
              唯一未过项：**8 条边界环**（理想是 1 条）

也就是说：位置找对了、目标完整、边界也在解剖凹谷上，
但切出来的补片还不是"一条边界的圆盘"。这是本模块**已知未解决**的部分，
本文件把它如实钉住，不允许被当成通过。
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from app.geometry.io import load_mesh
from app.geometry.tooth_location import (
    LocalizationStatus,
    SupportStatus,
    extract_support,
    locate_target_tooth,
    vertex_valley_strength,
)

TARGET_FDI = 36


@pytest.fixture(scope="module")
def fdi36_geometry(fdi36_case_dir):
    main = fdi36_case_dir / "01_主要病例模型"
    jaw = load_mesh(main / "Unsectioned_LowerJaw.stl")
    abutment = load_mesh(main / "1629-王煜 0.stl")
    return {"jaw": jaw, "abutment": abutment,
            "abutment_centroid": np.asarray(abutment.vertices, float).mean(axis=0)}


@pytest.fixture(scope="module")
def filed_jaw(fdi36_geometry):
    """模拟一次真实点击：点在最靠近基台中心的整颌表面上。"""
    jaw = fdi36_geometry["jaw"]
    V = np.asarray(jaw.vertices, float)
    click = V[int(np.argmin(np.linalg.norm(
        V - fdi36_geometry["abutment_centroid"], axis=1)))]
    return jaw, V, click


# --------------------------------------------------------------- 数据事实

def test_jaw_contains_the_scanned_abutment_in_the_same_frame(fdi36_geometry):
    """前置事实：基台与整颌同一坐标系，且基台表面就在整颌上（中位距离 0）。"""
    jaw, abut = fdi36_geometry["jaw"], fdi36_geometry["abutment"]
    d, _ = cKDTree(np.asarray(jaw.vertices, float)).query(
        np.asarray(abut.vertices, float), workers=-1)
    # 实测中位距离 2e-6 mm（同坐标系、且基台表面就是整颌表面的一部分）
    assert float(np.median(d)) < 1e-3, f"基台与整颌不同坐标系：中位距离 {np.median(d)}"
    assert float(np.percentile(d, 90)) < 2.0


def test_abutment_gingival_junction_has_strong_concave_creases(filed_jaw,
                                                               fdi36_geometry):
    """交界上**有**强凹脊 —— 本病例最关键的一条数据事实。

    匹配容差必须给足：用 0.3 mm 只能匹配到 11 个龈方端顶点，
    恰好都落在平滑处，于是得出"没有凹脊"的**错误**结论（已推翻）。
    0.5 mm 下 sample 有 46 个，凹脊强度 p90 0.862 / max 0.873。
    """
    jaw, V, _ = filed_jaw
    abut = fdi36_geometry["abutment"]
    vs = vertex_valley_strength(jaw)
    av = np.asarray(abut.vertices, float)

    z = av[:, 2]
    gingival = (z - z.min()) / (z.max() - z.min()) < 0.30
    d, nearest = cKDTree(V).query(av)

    junction_tight = np.unique(nearest[(d < 0.3) & gingival])
    junction = np.unique(nearest[(d < 0.5) & gingival])

    # 记录"窄匹配会给出误导性样本"这件事本身
    assert len(junction) > 3 * len(junction_tight), (
        f"容差放宽后样本没变大：{len(junction_tight)} -> {len(junction)}；"
        "本文件关于样本量的结论需要重新核对")
    assert len(junction) >= 20, f"交界顶点太少，不足以支撑结论：{len(junction)}"

    measured_p90 = float(np.percentile(vs[junction], 90))
    assert measured_p90 > 0.5, (
        f"交界凹脊没达到记录的强度：p90={measured_p90:.3f}（记录 0.862）")
    assert float(vs[junction].max()) > 0.6


# --------------------------------------------------------------- 定位

def test_locate_finds_the_prepared_tooth(filed_jaw, fdi36_geometry):
    """点击预备体表面 -> 位置必须落在基台自身范围内。"""
    jaw, V, click = filed_jaw
    ac = fdi36_geometry["abutment_centroid"]
    ab = np.asarray(fdi36_geometry["abutment"].vertices, float)

    res = locate_target_tooth(jaw, TARGET_FDI, user_seed=click)
    pos = np.asarray(res.position, float)

    assert res.method == "SEED"
    assert res.jaw == "lower"
    # 位置与基台中心的距离，应当远小于基台自身尺度（半对角线约 5 mm）
    half_diag = float(np.linalg.norm(ab.max(0) - ab.min(0)) / 2.0)
    err = float(np.linalg.norm(pos - ac))
    assert err < half_diag, f"位置偏离基台中心 {err:.2f} mm（半对角线 {half_diag:.2f}）"
    # 定位区域必须命中基台表面
    region = V[np.asarray(res.region_vertex_ids)]
    covered = float((cKDTree(region).query(ab)[0] <= 0.5).mean())
    assert covered > 0.20, f"定位区域几乎没盖住基台：{covered:.3f}"
    assert len(res.region_vertex_ids) > 1000
    assert res.confidence > 0.3


def test_locate_from_a_click_off_the_surface_still_works(filed_jaw, fdi36_geometry):
    """操作员点到基台内部（离表面 2.3 mm）也要能定位 —— 种子只是大致位置。"""
    jaw, V, _ = filed_jaw
    ac = fdi36_geometry["abutment_centroid"]
    res = locate_target_tooth(jaw, TARGET_FDI, user_seed=ac)
    assert len(res.region_vertex_ids) > 1000, res.diagnostics
    assert res.status is not LocalizationStatus.FAILED


def test_auto_mode_declines_on_a_sectioned_cast(filed_jaw):
    """该扫描是绕 FDI36 的 sectioned cast，不是整颌 —— 自动模式必须拒绝。"""
    jaw, _, _ = filed_jaw
    res = locate_target_tooth(jaw, TARGET_FDI)
    assert res.method == "AUTO_UNAVAILABLE"
    assert res.status is LocalizationStatus.REVIEW_REQUIRED
    assert res.confidence == 0.0
    reasons = res.diagnostics["reasons"]
    assert reasons and "判别度不足" in reasons[0]
    assert "user_seed" in res.diagnostics["required_action"]


# --------------------------------------------------------------- support

def test_support_good_on_everything_except_boundary_loops(filed_jaw):
    """真实病例上 support 的**全部实测状态**，包括唯一未过的那一项。

    已达标：单连通、目标完整保留、面积量级合理、边界依附解剖凹谷。
    未达标：边界环 8 条（理想 1 条）—— 这是本模块已知未解决的部分。
    """
    jaw, _, click = filed_jaw
    res = locate_target_tooth(jaw, TARGET_FDI, user_seed=click)
    sup = extract_support(jaw, res, target_fdi=TARGET_FDI)

    assert sup.mesh is not None
    # --- 已达标
    assert sup.checks["single_connected_component"] is True
    assert sup.checks["target_retention"] > 0.98
    assert sup.checks["boundary_is_anatomical"] is True
    assert sup.checks["boundary_valley_strength_mean"] > 0.25
    assert sup.checks["area_ratio_vs_crown"] < 4.0

    # --- 未达标（已知）：补片不是"一条边界的圆盘"
    assert sup.checks["n_boundary_loops"] > 1
    assert sup.status is SupportStatus.REVIEW_REQUIRED
    reasons = sup.diagnostics["reasons"]
    assert any("边界环" in r for r in reasons), reasons
    # 但"目标没切坏"这件事不能因此被含糊掉
    assert not any("目标区域只被保留" in r for r in reasons)


def test_jaw_mesh_is_a_closed_manifold(fdi36_geometry):
    """整颌是**闭合流形**（每条边恰好 2 个面、无边界边）。

    这解释了为什么 support 会碎：我们是在一个闭合曲面上按测地半径切一块下来，
    切口两侧的面片一旦在某些窄区断开，就会得到多条边界环。
    """
    F = np.asarray(fdi36_geometry["jaw"].faces)
    counts: dict[tuple[int, int], int] = {}
    for a, b in ((0, 1), (1, 2), (2, 0)):
        for key in map(tuple, np.sort(F[:, [a, b]], axis=1)):
            counts[key] = counts.get(key, 0) + 1
    values = np.array(list(counts.values()))
    assert int((values == 1).sum()) == 0, "整颌不该有边界边"
    assert int((values > 2).sum()) == 0, "整颌不该有非流形边"


def test_support_cut_radius_is_reported_and_matches_margin(filed_jaw):
    """半径与"源是区域"配套：cut_radius = margin，而不是冠宽/2 + margin。"""
    jaw, _, click = filed_jaw
    res = locate_target_tooth(jaw, TARGET_FDI, user_seed=click)
    sup = extract_support(jaw, res, target_fdi=TARGET_FDI)
    assert sup.diagnostics["sources_are_region"] is True
    assert sup.cut_radius_mm == pytest.approx(sup.diagnostics["margin_mm"])
    # 定位区域本身已经铺开到冠宽量级（否则说明半径又被截断了）
    assert res.region_radius_mm > 8.0


def test_support_mesh_is_usable_downstream(filed_jaw):
    """切出来的网格要能真的交给 Axis / Margin：非空、连通、量级合理。"""
    jaw, _, click = filed_jaw
    res = locate_target_tooth(jaw, TARGET_FDI, user_seed=click)
    sup = extract_support(jaw, res, target_fdi=TARGET_FDI)

    assert isinstance(sup.mesh, trimesh.Trimesh)
    assert len(sup.mesh.faces) > 1000
    assert sup.mesh.area > 20.0
    assert sup.checks["n_vertices"] > 500
    # 网格本身可用，但见上一条：它还不是单连通补片
    assert sup.checks["n_boundary_loops"] is not None
