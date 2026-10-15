"""真值颈缘提取器的验收 —— 7 条。

这一组**先于**任何 clearance 指标。理由：如果验收用的 Margin 真值本身不是可靠闭环，
后面所有按高度分层的断言都被错误分区污染，届时数值改善也无法归因
（是 blockout 真改好了，还是 Margin 分带方式在误导）。

对应旧方法的失败模式（会被这 7 条全部拦住）：

    旧：最低 0.4 mm 带 → 只覆盖 16/24 方位角，是**残弧不是闭环**；spacing_cv 1.89
    新：intaglio ↔ exterior 真实交界缝

① 恰好选中一条环
② closed == True
③ simple == True
④ self_intersection == 0
⑤ 方位角覆盖 ≥ 24/24
⑥ 投影包围 support 截面（"包含 support 中心"不够 —— 螺丝通道口也满足它）
⑦ 刚性等变
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from app.geometry.gt_margin import (
    AZIMUTH_BINS,
    ENCLOSURE_MIN,
    MIN_LOOP_PERIMETER_MM,
    MarginStatus,
    extract_gt_crown_margin,
)
from tests.acceptance._synthetic import random_rigid
from tests.acceptance.conftest import require_fdi36

# 真值可构造时的紧容差
TIGHT_TOL_DEG = 0.5


@pytest.fixture(scope="module")
def extraction():
    from app.geometry.io import load_mesh

    case = require_fdi36()
    crown = load_mesh(case / "01_主要病例模型" / "1629-王煜 1.stl")
    support = load_mesh(case / "01_主要病例模型" / "1629-王煜 0.stl")
    axis = np.array([0.0, 0.0, 1.0])
    return crown, support, axis, extract_gt_crown_margin(crown, support, axis)


# ------------------------------------------------------------ ① 恰好一条环

def test_exactly_one_loop_selected(extraction):
    """恰好一条候选通过 gate，且状态为 OK。

    FDI36 的冠实测 χ = 0（genus 1）→ 有**两条**缝：颈缘 + 螺丝通道口。
    所以"恰好一条"不是靠拓扑天然成立，而是靠语义判别选出来的。
    这一步是这组断言里最容易做错的地方 —— 旧实现用"最大环"，会把通道口选中。
    """
    _, _, _, result = extraction
    assert result.status is MarginStatus.OK, result.diagnostics.get("reasons")
    passing = [c for c in result.diagnostics["candidates"] if c["passes_gates"]]
    assert len(passing) == 1, (
        f"通过 gate 的候选环有 {len(passing)} 条，期望 1 条："
        f"{[(c['index'], c['enclosure_fraction']) for c in passing]}"
    )


def test_genus_is_reported_not_assumed(extraction):
    """拓扑必须被报告而不是假定。genus=1 解释了"为什么会有多条缝"。"""
    _, _, _, result = extraction
    diag = result.diagnostics
    assert diag["euler_characteristic"] == 0
    assert diag["genus"] == 1
    assert diag["n_candidate_loops"] >= 2, "genus 1 应至少有两条缝"
    # 全部界面顶点二度 → 界面缝是干净流形
    assert diag["interface_vertex_degrees"] == {2: diag["interface_edges"]}


# ------------------------------------------------------- ②③④ 闭环 / 简单 / 无自交

def test_margin_is_closed(extraction):
    """必须闭环。旧方法产出的是残弧（16/24 方位角），首尾不相接。"""
    _, _, _, result = extraction
    assert result.is_closed is True
    points = np.asarray(result.points)
    gap = float(np.linalg.norm(points[-1] - points[0]))
    perimeter = result.perimeter_mm
    assert gap <= max(0.05 * perimeter, 0.05), f"首尾间距 {gap:.4f} mm 相对周长 {perimeter:.2f} mm 过大"


def test_margin_is_simple(extraction):
    """不得含重复顶点（自接触/夹点）。"""
    _, _, _, result = extraction
    assert result.is_simple is True
    edge_ids = result.edge_ids
    assert len(edge_ids) == len(result.points), "边数应等于点数（环上每条边一个）"
    # 每条边都是环上相邻两点，且首尾相接
    assert edge_ids[0][0] == edge_ids[-1][1], "首尾顶点必须相接（闭环）"
    for a, b in edge_ids:
        assert a != b, "不应出现自环边"


def test_margin_has_no_self_intersection(extraction):
    """非相邻线段不得相交。

    注意这里**不需要**用容差掩盖问题：曾因 append 闭合点造成重复端点，
    使正确的颈缘环被误报"自交 1 处"并整条拒绝 —— 那是实现缺陷，不是几何问题。
    """
    _, _, _, result = extraction
    for candidate in result.diagnostics["candidates"]:
        if candidate["passes_gates"]:
            assert candidate["n_self_intersections"] == 0


# --------------------------------------------------- ⑤⑤ 方位角覆盖（旧方法的核心失败）

def test_azimuth_coverage_is_full(extraction):
    """方位角必须覆盖接近 360° —— 直接针对"只取到最低点附近半圈"这种残弧。

    旧方法实测 0.667（16/24）；新方法 1.000。
    """
    _, _, _, result = extraction
    assert result.azimuth_coverage >= 0.999, (
        f"方位角覆盖 {result.azimuth_coverage:.3f}（{AZIMUTH_BINS} 箱）"
        f" —— 低于全覆说明提取到的是残弧"
    )


def test_rejected_candidates_are_reported_not_silently_dropped(extraction):
    """被拒的候选必须带明确理由留存，不能静默丢弃。

    这样即使将来某条真实缝被误拒，也能从 diagnostics 里看出来。
    """
    _, _, _, result = extraction
    rejected = [c for c in result.diagnostics["candidates"] if not c["passes_gates"]]
    assert rejected, "本病例应至少有一条被拒候选（螺丝通道口）"
    for c in rejected:
        assert c["rejected_reason"], f"候选 #{c['index']} 被拒但没有理由"
    # 通道口被拒的理由应是"未包围 support 截面"
    assert any("包围" in c["rejected_reason"] for c in rejected)


# ------------------------------------------------------- ⑥ 包围 support 截面

def test_margin_encloses_support_cross_section(extraction):
    """环的投影必须包围该环自身高度附近的 support 截面。

    **这是主判别量。** 不要退化成"是否包含 support 中心"：
    螺丝通道口与基台同轴，包含中心对它同样成立，实测包围比例只有 0.095。

    实测区分度：通道口 0.095 ｜ 颈缘 1.000 ｜ 拓扑伪影 0.000
    """
    _, _, _, result = extraction
    best = [c for c in result.diagnostics["candidates"] if c["passes_gates"]][0]
    assert best["enclosure_fraction"] >= ENCLOSURE_MIN

    # 被拒的通道口应显著低于阈值
    others = [c for c in result.diagnostics["candidates"] if not c["passes_gates"]]
    if others:
        assert max(c["enclosure_fraction"] for c in others) < ENCLOSURE_MIN


def test_enclosure_discriminates_not_just_threshold(extraction):
    """包围比例在颈缘与被拒候选之间必须有数量级差异 —— 防止阈值碰巧通过。"""
    _, _, _, result = extraction
    best = [c for c in result.diagnostics["candidates"] if c["passes_gates"]][0]
    others = [c["enclosure_fraction"] for c in result.diagnostics["candidates"]
              if not c["passes_gates"]]
    if others:
        assert best["enclosure_fraction"] >= 5 * max(others), (
            f"区分度不足：选中 {best['enclosure_fraction']:.3f} vs 其余 {max(others):.3f}"
        )


# ------------------------------------------------------------ ⑦ 刚性等变

def test_rigid_equivariance(extraction):
    """对冠、support 与轴施加同一刚体变换，提取的点集必须随之变换。

    这是唯一能拦住"依赖世界坐标轴"这类缺陷的测试。本项目在就位方向求解器上
    已经因此抓到一个真 bug（``if axis[2] < 0`` 世界 Z 符号约定）。
    """
    crown, support, axis, base = extraction
    base_points = np.asarray(base.points)

    rng = np.random.default_rng(7)
    for k in range(3):
        T = random_rigid(rng)
        crown_t = crown.copy()
        crown_t.apply_transform(T)
        support_t = support.copy()
        support_t.apply_transform(T)
        axis_t = T[:3, :3] @ axis

        got = extract_gt_crown_margin(crown_t, support_t, axis_t)
        assert got.status is MarginStatus.OK, f"变换 {k}: {got.diagnostics.get('reasons')}"

        expected = (T[:3, :3] @ base_points.T).T + T[:3, 3]
        got_points = np.asarray(got.points)
        # 点集的起点/方向可能不同，用双向 Hausdorff 比较
        d1, _ = cKDTree(expected).query(got_points, k=1, workers=-1)
        d2, _ = cKDTree(got_points).query(expected, k=1, workers=-1)
        hausdorff = max(float(d1.max()), float(d2.max()))
        assert hausdorff < 1e-6, (
            f"变换 {k}: 点集 Hausdorff 距离 {hausdorff:.3e} mm —— 提取器不是刚性等变的"
        )


def test_rigid_invariance_of_scalar_metrics(extraction):
    """所有标量指标必须在刚体变换下严格不变。"""
    crown, support, axis, base = extraction
    rng = np.random.default_rng(11)
    for k in range(2):
        T = random_rigid(rng)
        crown_t = crown.copy()
        crown_t.apply_transform(T)
        support_t = support.copy()
        support_t.apply_transform(T)
        got = extract_gt_crown_margin(crown_t, support_t, T[:3, :3] @ axis)

        assert abs(got.perimeter_mm - base.perimeter_mm) < 1e-6
        assert abs(got.azimuth_coverage - base.azimuth_coverage) < 1e-9
        assert abs(got.spacing_cv - base.spacing_cv) < 1e-6
        assert abs(got.confidence - base.confidence) < 1e-6


# ------------------------------------------------------- 输出语义（防退化）

def test_plane_is_diagnostic_only(extraction):
    """最佳拟合平面只能出现在 diagnostics 里，不得成为核心语义。

    实测本病例的缝接近平面（残差 max 0.043 mm），但那是**该病例的性质**，
    不是保证 —— 沿龈下轮廓走的 finish line 并不平面。若把 ``margin_plane_height``
    做成输出字段，下游就会去用它，损失真实几何。
    """
    _, _, _, result = extraction
    payload = result.as_dict()
    for forbidden in ("margin_plane_height", "plane_normal", "plane_origin"):
        assert forbidden not in payload, f"输出里不应有核心语义字段 {forbidden}"
    assert "best_fit_plane" in payload["diagnostics"]
    assert "_note" in payload["diagnostics"]["best_fit_plane"]


def test_self_intersection_helper_contract():
    """helper 的契约是**接收开环点列**。这条测试把踩过的坑固定下来。

    曾为了闭合而 append 首个点，使数组末尾多出一个重复端点：段 0 与闭合段在几何上
    共享端点，但模运算的邻接判断认不出"末尾那个点就是首点"，于是把**正确的颈缘环**
    误报为"自交 1 处"并整条拒绝。

    所以契约是"传开环"，不是"传什么都行"—— 下面第三条断言把误用的后果也钉住，
    防止将来有人给 helper 加容差把问题"绕过"而不是修正调用方。
    """
    from app.geometry.gt_margin import _self_intersection_count

    square = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert _self_intersection_count(square) == 0          # 正确用法：开环

    bowtie = np.array([[0.0, 0, 0], [1, 1, 0], [1, 0, 0], [0, 1, 0]])
    assert _self_intersection_count(bowtie) >= 1          # 真自交能被检出

    # 误用（传闭合点列）会被检出而非静默放过 —— 正是当初暴露问题的方式
    assert _self_intersection_count(np.vstack([square, square[:1]])) == 1
