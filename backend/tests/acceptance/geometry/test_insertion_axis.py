"""就位方向求解器验收 —— 六类测试。

这一组是求解器**正确性的真正证明**：合成几何的真值是构造出来的，因此可以严格断言。
真实 FDI36 病例做不到这一点（其临床正确方向无法从数据独立确定，见
``test_fdi36_golden.py`` 里的说明），所以精度断言放在这里，真实病例上只断言
"不劣于平凡解 + 与几何自洽 + 该报不确定时报不确定"。

A. 刚性等变        —— axis(T·M) == R·axis(M)，且所有指标严格不变
B. 完美锥台        —— 无倒凹时轴误差 0
C. 倾斜预备体      —— 输出 = 倾斜后的 +Z
D. 人工倒凹        —— 优化轴的倒凹指标必须优于错误轴
E. 退化 Margin     —— 必须明确 REVIEW_REQUIRED / FAILED，不得静默输出
F. 网格分辨率鲁棒性 —— 同一几何不同面数，轴不得漂移
"""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.insertion import (
    ABOVE_MARGIN_TOLERANCE_MM,
    AxisStatus,
    _undercut_metrics,
    angle_deg,
    score_axis,
    solve_insertion_axis,
    validate_margin,
)
from tests.acceptance._synthetic import (
    apply_rigid,
    frustum,
    random_rigid,
    transform_points,
    true_axis_for_tilt,
)

#: 合成几何上允许的轴误差。真值可构造，所以阈值可以很紧。
TIGHT_TOL_DEG = 0.5


# ------------------------------------------------------------------ A. 刚性等变

def test_rigid_equivariance_of_axis():
    """对 support 与 Margin 施加同一刚体变换，解出的轴必须随之旋转。

    这是数学正确性测试，也是唯一能拦住"依赖世界坐标符号约定"这类缺陷的测试。
    实测它抓到过一个真 bug：``estimate_wall_axis`` 曾用 ``if axis[2] < 0`` 定向，
    轴被翻转后拔模角全部反号，倒凹比在刚体变换下从 0.844 跳到 0.156
    —— 而那时轴夹角仍显示 0.000°，肉眼看指标完全不察觉。
    """
    rng = np.random.default_rng(0)
    mesh, base = frustum(tilt_deg=11.0, bulge=1.1)
    ref = solve_insertion_axis(mesh, base)

    for k in range(4):
        T = random_rigid(rng)
        got = solve_insertion_axis(apply_rigid(mesh, T), transform_points(base, T))
        expected = T[:3, :3] @ np.asarray(ref.axis)

        assert np.dot(got.axis, expected) > 0, (
            f"变换 {k}: 轴方向翻转（angle_deg 用绝对值会掩盖这点，故显式检查同向）"
        )
        assert angle_deg(got.axis, expected) < TIGHT_TOL_DEG, (
            f"变换 {k}: 轴偏差 {angle_deg(got.axis, expected):.4f}°"
        )


def test_rigid_invariance_of_metrics():
    """所有标量指标必须在刚体变换下严格不变。"""
    rng = np.random.default_rng(1)
    mesh, base = frustum(tilt_deg=11.0, bulge=1.1)
    ref = solve_insertion_axis(mesh, base)

    for k in range(3):
        T = random_rigid(rng)
        got = solve_insertion_axis(apply_rigid(mesh, T), transform_points(base, T))

        assert abs(got.undercut_area_ratio - ref.undercut_area_ratio) < 1e-9, (
            f"变换 {k}: 倒凹面积比变化 {abs(got.undercut_area_ratio - ref.undercut_area_ratio):.3e}"
        )
        assert abs(got.margin_visibility_ratio - ref.margin_visibility_ratio) < 1e-9
        assert abs(got.max_undercut_depth_mm - ref.max_undercut_depth_mm) < 1e-6
        assert abs(got.score - ref.score) < 1e-9


# ------------------------------------------------------------------ B. 完美锥台

@pytest.mark.parametrize("taper", [(3.0, 2.2), (3.0, 3.0), (2.5, 1.0)])
def test_undercut_free_frustum_has_no_undercut(taper):
    """纯收缩圆台（无倾斜）没有倒凹，解出的轴必须是 +Z 且倒凹为 0。"""
    r_bottom, r_top = taper
    mesh, base = frustum(r_bottom, r_top, 8.0)

    result = solve_insertion_axis(mesh, base)

    assert angle_deg(result.axis, [0.0, 0.0, 1.0]) < TIGHT_TOL_DEG
    assert result.undercut_area_ratio < 1e-6, (
        f"无倒凹几何却报倒凹比 {result.undercut_area_ratio:.6f}"
    )
    assert result.margin_visibility_ratio > 0.999
    assert result.status is AxisStatus.OK, result.diagnostics["reasons"]


def test_no_false_undercut_from_bottom_cap():
    """底面/裁切盖片不得被算成倒凹。

    底面位于 Margin 平面上、法向朝下，若把 Margin 平面附近也算入就位面，
    它会贡献恰好等于底面积占比的假倒凹（实测 16.17%）。真实病例里 Margin
    平面以下同样只是扫描裁切边界或牙龈。
    """
    mesh, base = frustum(3.0, 2.2, 8.0)
    metrics = _undercut_metrics(mesh, np.array([0.0, 0.0, 1.0]), base)

    assert metrics["undercut_area_ratio"] < 1e-6
    # 确认底面确实被排除在"可用面积"之外
    assert metrics["usable_area_mm2"] < metrics["total_area_mm2"]
    assert ABOVE_MARGIN_TOLERANCE_MM > 0


# ------------------------------------------------------------------ C. 倾斜预备体

@pytest.mark.parametrize("tilt", [13.0, 25.0, -18.0, 32.0])
def test_tilted_prep_recovers_tilt(tilt):
    """整体倾斜后，解出的轴必须等于倾斜后的 +Z。"""
    mesh, base = frustum(tilt_deg=tilt)
    result = solve_insertion_axis(mesh, base)
    truth = true_axis_for_tilt(tilt)

    assert angle_deg(result.axis, truth) < TIGHT_TOL_DEG, (
        f"倾斜 {tilt}°：解出 {np.round(result.axis, 4)}，真值 {np.round(truth, 4)}"
    )
    assert result.status is AxisStatus.OK, result.diagnostics["reasons"]


def test_tilt_is_reported_in_deviation():
    """倾斜越大，与初始轴的偏角越大 —— 但不得超出 max_deviation_deg。"""
    for tilt in (13.0, 25.0):
        mesh, base = frustum(tilt_deg=tilt)
        result = solve_insertion_axis(mesh, base, max_deviation_deg=40.0)
        assert result.deviation_deg <= 40.0 + TIGHT_TOL_DEG


# ------------------------------------------------------------------ D. 人工倒凹

def test_solver_returns_score_optimal_axis():
    """有外凸环时，求解结果必须优于任何替代方向。

    断言的是**综合评分**最优性，而不是单一指标最小。这不是放宽标准：
    评分是倒凹面积 / 倒凹深度 / Margin 可见率 / 偏角的折中，单看面积会与
    求解器实际优化的目标不一致（实测 15° 倾斜的倒凹面积更小，但最大埋深
    3.75 mm 远大于求解结果的埋深，因此综合评分更差）。
    """
    mesh, base = frustum(bulge=1.25)
    result = solve_insertion_axis(mesh, base)
    initial = np.asarray(result.initial_axis)

    def score(axis):
        return score_axis(mesh, base, axis, initial_axis=initial)["score"]

    best = score(np.asarray(result.axis))
    alternatives = {"+Z": [0.0, 0.0, 1.0]}
    for deg in (15.0, 30.0, 45.0):
        alternatives[f"tilt{int(deg)}"] = [np.sin(np.radians(deg)), 0.0, np.cos(np.radians(deg))]
    for label, axis in alternatives.items():
        assert best <= score(axis) + 1e-9, (
            f"求解结果评分 {best:.6f} 不优于 {label} 的 {score(axis):.6f}"
        )


def test_bulge_is_flagged_not_hidden():
    """外凸环无法靠换轴消除，必须报 REVIEW_REQUIRED，不能声称 OK。"""
    mesh, base = frustum(bulge=1.25)
    result = solve_insertion_axis(mesh, base)

    assert result.status is AxisStatus.REVIEW_REQUIRED
    assert result.undercut_area_ratio > 0.1
    assert any("倒凹" in r for r in result.diagnostics["reasons"]), \
        result.diagnostics["reasons"]


# ------------------------------------------------------------------ E. 退化 Margin

def test_margin_too_few_points_fails():
    mesh, base = frustum()
    result = solve_insertion_axis(mesh, base[:2])

    assert result.status is AxisStatus.FAILED
    assert result.diagnostics["margin_validation"]["ok"] is False
    assert "点数不足" in " ".join(result.diagnostics["margin_validation"]["reasons"])


def test_margin_not_closed_is_review_required():
    """非闭环的 Margin（残弧）必须被识别 —— 这正是本项目真实病例上踩过的坑。"""
    mesh, base = frustum()
    arc = base[10:70]          # 只取一段弧，丢掉两端
    result = solve_insertion_axis(mesh, arc)

    assert result.status is AxisStatus.REVIEW_REQUIRED
    assert result.diagnostics["margin_validation"]["ok"] is False
    assert "非闭环" in " ".join(result.diagnostics["margin_validation"]["reasons"])


def test_margin_severely_non_planar_is_review_required():
    mesh, base = frustum()
    bumpy = np.vstack([base, base + [0.0, 0.0, 3.0]])
    result = solve_insertion_axis(mesh, bumpy)

    assert result.status is AxisStatus.REVIEW_REQUIRED
    assert result.diagnostics["margin_validation"]["ok"] is False


def test_margin_non_finite_fails():
    mesh, base = frustum()
    bad = base.copy()
    bad[0, 0] = np.nan
    result = solve_insertion_axis(mesh, bad)

    assert result.status is AxisStatus.FAILED


@pytest.mark.parametrize("band_scale,expect", [(0.02, True), (0.5, False)])
def test_validate_margin_closure_metric(band_scale, expect):
    """闭环性判据本身要可靠：完整环通过，小区段不通过。"""
    mesh, base = frustum(sections=100)
    n = int(len(base) * band_scale)
    points = base if expect else base[:max(3, n)]
    validation = validate_margin(points, mesh)
    assert validation.ok is expect


# ------------------------------------------------------- F. 网格分辨率鲁棒性

@pytest.mark.parametrize("tilt,bulge", [(0.0, 1.0), (13.0, 1.1)])
def test_mesh_resolution_robustness(tilt, bulge):
    """同一几何不同面数，解出的轴不得漂移。

    否则换一台扫描仪、换一次网格简化，结果就会重新坏掉。
    """
    axes = []
    for sections in (40, 120, 360):
        mesh, base = frustum(sections=sections, tilt_deg=tilt, bulge=bulge)
        result = solve_insertion_axis(mesh, base)
        axes.append(np.asarray(result.axis))

    assert angle_deg(axes[0], axes[-1]) < TIGHT_TOL_DEG, (
        f"sections 40 与 360 的轴相差 {angle_deg(axes[0], axes[-1]):.4f}°"
    )
    for a in axes[1:-1]:
        assert angle_deg(a, axes[-1]) < TIGHT_TOL_DEG


# --------------------------------------------------------------- 其它不变量

def test_solver_never_silently_returns_worst_of_bad_set():
    """状态机必须能表达"没有好方向"，不能永远 OK。"""
    mesh, base = frustum(bulge=1.6)          # 极端倒凹
    result = solve_insertion_axis(mesh, base)

    assert result.status is not AxisStatus.OK
    assert result.diagnostics["reasons"], "REVIEW_REQUIRED 必须带理由"


def test_result_serializes_with_all_declared_fields():
    """``as_dict`` 必须覆盖接口约定的全部字段，防止字段被悄悄删掉。"""
    mesh, base = frustum()
    payload = solve_insertion_axis(mesh, base).as_dict()

    for key in ("axis", "initial_axis", "score", "undercut_area_mm2", "undercut_area_ratio",
                "max_undercut_depth_mm", "mean_undercut_depth_mm", "margin_visibility_ratio",
                "deviation_deg", "confidence", "status", "diagnostics"):
        assert key in payload, f"结果缺少字段 {key}"

    assert payload["diagnostics"]["max_deviation_deg_source"], \
        "20° 的出处必须随结果一起返回（它是 Lab Profile 证据，不是全局规律）"


def test_neighbors_none_fully_supported():
    """没有邻牙分割时必须能完整求解（neighbors=None）。"""
    mesh, base = frustum(tilt_deg=13.0)
    result = solve_insertion_axis(mesh, base, neighbors=None)

    assert result.status is AxisStatus.OK
    assert result.diagnostics["neighbors_provided"] is False
    assert "未参与评分" in result.diagnostics["neighbors_note"]
