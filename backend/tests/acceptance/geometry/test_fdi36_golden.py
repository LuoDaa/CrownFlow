"""FDI36 基准病例的几何验收。

断言的对象是 ``baseline.json`` 里的 ``targets``，不是 ``measurements``。
``measurements`` 只用于在 diff 里看到每次几何改动的 before/after。

**完成标准不是"STL 成功生成"，而是"与真实病例真值的定量误差下降"。**

已知缺陷的 ``xfail(strict=True)`` 标记由 conftest 里的
``pytest_collection_modifyitems`` 统一施加，理由取自 ``baseline.json`` 的
``known_defects``（见 ``baselines/regenerate.py``）。修好某项后 XPASS 会让 strict
模式报错，从而强制移除标记。目标状态：0 xfail。
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.acceptance import _metrics as M


# ======================================================= 通过项（有断言价值）

def test_boolean_backend_is_exact(baseline):
    """布尔后端必须是精确引擎。体素降级会把体积放大 10%、引入 0.089 mm 中位误差。"""
    assert baseline["measurements"]["boolean_backend"] == \
        baseline["targets"]["boolean_backend"]["target"]


def test_plane_clip_is_watertight_on_planar_margin():
    """平面 Margin 上的平面裁切必须水密且零开放边界。

    这是本项目的原始 P0：``cap=False`` 静默退化会留下开放边界，最终得到真值 13%
    体积的碎冠。该断言只在 **Margin 数值平面**时有意义（也是平面裁切唯一合法的场合）；
    本病例的真值 Margin 非平面，走这条路会被守卫拒绝 —— 见下一条测试。
    """
    from tests.acceptance.synthetic_prep import prep_case

    case = prep_case(amplitude_mm=0.0)
    detail = M.measure_plane_clip(case["gt_crown"], case["margin"], case["axis"])

    assert detail["margin_plane"]["numerically_planar"] is True
    assert detail["watertight"] is True, (
        f"裁切后非水密，剩余开放边界 {detail['boundary_edges']} 条"
    )
    assert detail["boundary_edges"] == 0


def test_nonplanar_gt_margin_is_refused_not_clipped(fdi36_meshes, baseline):
    """非平面真值 Margin 禁止平面裁切：必须 fail loud，并把真实偏离量记录在案。

    Q0（2026-10-12）之前这里断言"平面裁切后的网格水密"，但那条链路对非平面 Margin
    是**被守卫禁止**的（非平面 3D Margin 用平面裁切会静默产出错误几何：共享颈缘缝、
    方向 blockout 内腔、壁厚审计三件事都做不了）。所以本用例改为钉住守卫本身：
    拒绝是正确行为，偏离量是唯一允许出现的数字。
    """
    from app.geometry.margin_seam import MarginReconstructionRequired

    margin = _seam(fdi36_meshes)
    status = M.margin_plane_status(margin)
    assert status["numerically_planar"] is False, "本病例真值 Margin 不是平面"
    assert status["plane_deviation_max_mm"] == pytest.approx(0.0433, abs=5e-4)

    with pytest.raises(MarginReconstructionRequired) as refusal:
        M.measure_plane_clip(fdi36_meshes["gt_crown"], margin, M.OCCLUSAL_AXIS)
    diagnostics = refusal.value.diagnostics
    assert diagnostics["status"] == "MARGIN_RECONSTRUCTION_REQUIRED"
    assert diagnostics["plane_deviation_max_mm"] == pytest.approx(0.0433, abs=5e-4)
    assert set(diagnostics["missing_adapters"]) == {
        "reviewed_outer_cervical_open_surface",
        "direction_blockout_gap_intaglio_with_prescribed_margin_boundary",
        "shared_seam_fit_thickness_audit"}
    assert baseline["targets"]["margin_trim_watertight"]["target"] is True


def test_boolean_roundtrip_is_exact(fdi36_meshes, baseline):
    """未被裁减的牙冠过一遍布尔，误差必须接近 0。

    实测：manifold3d 下为 0.0；体素降级在 pitch 0.10 下达到 0.132 mm p95 并放大体积 10%。
    """
    target = baseline["targets"]["boolean_roundtrip_deviation_p95_mm"]["target_max"]
    rt = M.measure_boolean_roundtrip(fdi36_meshes["gt_crown"], pitch=0.10)

    assert rt["deviation_p95_mm"] <= target, rt
    assert abs(rt["volume_ratio"] - 1.0) <= 0.01, f"体积比 {rt['volume_ratio']}"
    assert rt["faces"] <= rt["input_faces"] * 1.2, (
        f"面数从 {rt['input_faces']} 膨胀到 {rt['faces']}"
    )


def test_live_chain_final_mesh_is_usable_and_balanced(baseline):
    """CAM 可用性底线（**sanctioned 非平面链路**）：水密、单连通、共享颈缘缝、质量守恒。

    为什么不再用 ``_isolated(...)``：那条链路（平面裁切 + ``enforce_clearance`` + 体素布尔）
    对非平面 Margin 已被守卫禁止，且 ``enforce_clearance`` 本身就是 P0 缺陷
    （用"顶点到 cutter"距离当壁厚，外推 37% 顶点最多 1.63 mm）。
    现在测的是与 API ``finalize_crown`` 同族的链路，输入见 ``synthetic_prep.prep_case``。
    """
    r = _live_chain()
    targets = baseline["targets"]
    assert r["margin_plane"]["numerically_planar"] is False, "本用例必须走非平面路径"
    assert r["watertight"] is True
    assert r["components"] == targets["live_chain_components"]["target"], (
        f"碎冠回归：组件数 {r['components']}"
    )
    assert r["boundary_edges"] == targets["live_chain_boundary_edges"]["target"]
    assert r["nonmanifold_edges"] == 0
    assert r["seam_shared_edges_incidence_two"] is True
    assert r["seam_margin_to_seam_max_mm"] <= targets["live_chain_seam_max_mm"]["target_max"]
    assert r["seam_to_margin_max_mm"] <= targets["live_chain_seam_max_mm"]["target_max"]
    assert r["mass_balance_relative_error"] <= \
        targets["live_chain_mass_balance_relative_error"]["target_max"], (
        f"final ≠ outer − 内腔（相对误差 {r['mass_balance_relative_error']:.4f}）："
        "要么崩坏，要么内腔没被真正扣减"
    )


def test_live_chain_refuses_abutment_support_with_reason(fdi36_meshes):
    """真实基台病例：Margin 以上不是单一完整就位面 ⇒ 必须拒绝，不许硬做。

    这是 Q0 查明的**输入契约**：live 链路要的是"分段好的单一预备体面"。
    FDI36 是种植基台（Margin 以上有螺丝通道开口），FDI46 病例包的 ``prep.stl``
    开放边界跨过 Margin、``die.stl`` 与 FDI36 同因 —— 三个真实输入都被正确拒绝。
    记录在此，避免以后有人把它当 bug"修好"，也标明真实病例验收需要一个满足契约的输入。
    """
    from app.geometry.nonplanar_crown import MarginBuildError, fitting_support_above_margin

    with pytest.raises(MarginBuildError) as refusal:
        fitting_support_above_margin(fdi36_meshes["support"], _seam(fdi36_meshes), M.OCCLUSAL_AXIS)
    assert "单一完整就位面" in str(refusal.value)


# ========================================= 就位方向（判据已修正，见下）

def test_insertion_axis_agrees_with_fitted_wall_axis(fdi36_meshes, baseline):
    """求解的就位方向必须与该基台侧壁法向独立拟合出的轴一致。

    **为什么不用"+Z 误差 < 10°"作为断言**（本项目修正掉的一个错误判据）：

    - 基台侧壁法向独立拟合出的轴本身就偏 +Z **12.9°**（以 +Z 与以 Margin 平面法向
      为参考各拟合一次，两次一致到 0.01°，拟合质量 eigenvalue ratio ≈ 0.105）。
      种植基台相对颌骨咬合平面倾斜是常见情况，所以 +Z 不是可从数据独立确认的真值。
    - 倒凹面积、射线不可达、blockout 代价这三类目标在该基台上跨轴变化都在 ±8% 以内，
      最小值并不落在 +Z。把 +Z 当目标会激励"恰好把方向旋到 +Z"，而不是求出更
      合理的就位方向。

    精度断言放在合成几何上（``test_insertion_axis.py``，真值可构造，当前误差 0）。
    与 +Z 的夹角仍记录在 baseline 的 ``error_vs_plus_z_deg_DIAGNOSTIC_ONLY``，仅作追踪。
    """
    diag = _axis_diagnostics(fdi36_meshes)
    target = baseline["targets"]["insertion_axis_vs_fitted_wall_axis_deg"]["target_max"]

    assert diag["axis_vs_fitted_wall_axis_deg"] <= target, (
        f"求解轴与基台拟合壁轴相差 {diag['axis_vs_fitted_wall_axis_deg']}°（上限 {target}°）"
    )
    assert diag["fitted_wall_axis_quality"] < 0.3, (
        f"壁轴拟合质量过差（eigenvalue ratio {diag['fitted_wall_axis_quality']}），"
        "该参考不可用，本断言失去意义"
    )


def test_insertion_axis_improves_undercut_over_trivial_axis(fdi36_meshes, baseline):
    """求解轴必须显著优于平凡解 +Z，否则求解器没有实际收益。"""
    diag = _axis_diagnostics(fdi36_meshes)
    rel = diag["undercut_vs_plus_z"]["undercut_area_ratio_rel_change"]
    target = baseline["targets"]["insertion_axis_undercut_rel_improvement"]["target_max"]

    assert rel <= target, (
        f"倒凹面积比仅改善 {rel:+.3f}（要求 ≤ {target}）。"
        f"+Z {diag['undercut_vs_plus_z']['undercut_area_ratio_plus_z']:.4f} → "
        f"求解 {diag['undercut_vs_plus_z']['undercut_area_ratio_solver']:.4f}"
    )


def test_insertion_axis_margin_stays_visible(fdi36_meshes, baseline):
    """Margin 沿最优轴必须基本可见。"""
    diag = _axis_diagnostics(fdi36_meshes)
    minimum = baseline["targets"]["insertion_axis_margin_visibility"]["target_min"]
    assert diag["margin_visibility_ratio"] >= minimum, (
        f"Margin 可见率 {diag['margin_visibility_ratio']} < {minimum}"
    )


def test_insertion_axis_status_matches_margin_quality(fdi36_meshes, baseline):
    """状态应为 OK —— 这是用**正确**颈缘真值重算后的结论。

    **本条判据被修正过，前因后果记录在 baseline 的 ``retracted_expectations``。**

    上一轮曾断言该病例必须报 REVIEW_REQUIRED，理由是"该基台有轴对称外扩的
    emergence profile，相对任何轴都是倒凹，OK 是错答案"。该结论已推翻：
    它建立在错误的 Margin 真值上 —— 旧的"最低 0.4 mm 带"给出一个几乎水平的面
    （偏离 +Z **0.80°**），而真实颈缘缝的平面偏离 +Z **11.67°**，与基台自身拟合轴
    （**12.9°**）一致（边缘平面本就垂直于基台轴线）。用错平面切分"就位面"会把
    倒凹统计的作用域放错，虚高倒凹比（0.214 → 修正后 0.117）。

    "求解器必须能表达没有好方向"这一 fail-loud 性质仍被覆盖 —— 由合成几何的
    ``test_solver_never_silently_returns_worst_of_bad_set`` 与
    ``test_bulge_is_flagged_not_hidden`` 断言，不依赖本病例。
    """
    diag = _axis_diagnostics(fdi36_meshes)
    expected = baseline["targets"]["insertion_axis_status"]["target"]

    assert diag["status"] == expected, (
        f"倒凹比 {diag['undercut_area_ratio']:.3f}、可见率 {diag['margin_visibility_ratio']:.3f} "
        f"下状态为 {diag['status']}，期望 {expected}；理由：{diag['reasons']}"
    )
    assert not diag["reasons"], f"OK 状态下不应有 REVIEW 理由：{diag['reasons']}"
    assert not diag["objective_is_flat"], (
        "本病例目标函数不应平坦；若变平坦说明判据退化，需重新审视"
    )


# =============================================== 已知缺陷项（xfail 待消除）

def test_margin_band_clearance_within_range(fdi36_meshes, baseline):
    """边缘带内腔间隙中位数，目标 0.015–0.050 mm（exocad 真实 margin_gap 为 0.025）。

    边缘带是精度最关键的区域。
    """
    band = _margin_band(fdi36_meshes)
    lo, hi = baseline["targets"]["margin_band_clearance_median_mm"]["target_range"]
    med = band["margin_band_median_mm"]

    assert med is not None and lo <= med <= hi, (
        f"边缘带间隙中位数 {med} mm 超出 [{lo}, {hi}]。"
        "偏差来自投影柱 blockout 把内腔基面撑大，而非 cement gap 参数本身。"
    )


def test_supra_margin_clearance_within_limit(fdi36_meshes, baseline):
    """配合面（边缘带以上）内腔间隙 p95，目标 < 0.10 mm。"""
    band = _margin_band(fdi36_meshes)
    target = baseline["targets"]["supra_margin_clearance_p95_mm"]["target_max"]
    assert band["supra_margin_p95_mm"] <= target, band


def test_live_chain_preserves_outer_anatomy(baseline):
    """喂真值外冠给 sanctioned 链路，链路对外冠形态的改写必须 p95 < 0.05 mm。

    这是"外冠形态不许被内腔环节改写"的守门断言。**这条断言 2026-10-12 之前是
    ``xfail``**（已知缺陷 ``isolated_chain_outer_deviation_p95_mm``），因为当时测的是
    退役链路的 ``enforce_clearance``：它用"顶点到 cutter"的距离当壁厚，误判并外推 37%
    顶点最多 1.63 mm（实测 p95 0.865 mm）。live 链路**不做自动补厚**（薄区由
    ``audit_manufacturing`` 报告并阻断放行），所以这条现在实测 1.1e-7 mm 并通过 ——
    缺陷随退役链路一并撤销，而不是被放宽阈值。
    """
    r = _live_chain()
    target = baseline["targets"]["isolated_chain_outer_deviation_p95_mm"]["target_max"]

    assert r["outer_deviation_p95_mm"] <= target, (
        f"CAD 链路把完美外冠改动了 {r['outer_deviation_p95_mm']} mm (p95)、"
        f"{r['outer_deviation_max_mm']} mm (max)。误差全部来自链路自身，与模板质量无关。"
    )


# ============================================================ 其余量化守门

def test_live_chain_mass_balance(baseline):
    """``final ≈ outer − 内腔``（相对误差 ≤ 目标值）。

    取代旧的 ``isolated_chain_volume_ratio ∈ [0.90, 1.10]``：那个区间只对"外冠本身
    就是薄壳冠"的输入成立，换输入就失效；质量守恒与几何形状无关，且能同时抓到
    "崩坏"（历史 P0 是 0.13）与"内腔没被真正扣减"两种失败。
    """
    r = _live_chain()
    target = baseline["targets"]["live_chain_mass_balance_relative_error"]["target_max"]
    assert r["mass_balance_relative_error"] <= target, (
        f"final={r['volume_final_mm3']:.3f} mm³ 与 outer−内腔="
        f"{r['volume_outer_mm3'] - r['volume_cavity_mm3']:.3f} mm³ 不符"
        f"（相对误差 {r['mass_balance_relative_error']:.4f} > {target}）"
    )


def test_measurements_are_current(fdi36_meshes, baseline):
    """baseline.measurements 必须与当前代码实测值一致。

    用途：几何指标改善时 diff 里必然出现 measurements 变化，从而强制重新审视
    targets 与 known_defects，避免"改了代码但 baseline 没更新"。
    """
    recorded = baseline["measurements"]["insertion_axis"]
    diag = _axis_diagnostics(fdi36_meshes)

    assert abs(diag["axis_vs_fitted_wall_axis_deg"]
               - recorded["axis_vs_fitted_wall_axis_deg"]) <= 1.0, (
        f"轴与壁轴偏差实测 {diag['axis_vs_fitted_wall_axis_deg']:.2f}° 与 baseline "
        f"{recorded['axis_vs_fitted_wall_axis_deg']:.2f}° 不符。若为有意改进，请运行 "
        "`python -m tests.acceptance.baselines.regenerate --write` 刷新 baseline "
        "并同步更新 targets / known_defects。"
    )
    assert abs(diag["undercut_area_ratio"] - recorded["undercut_area_ratio"]) <= 0.02, (
        f"倒凹比实测 {diag['undercut_area_ratio']:.4f} 与 baseline "
        f"{recorded['undercut_area_ratio']:.4f} 不符"
    )


def test_known_defects_are_documented(baseline):
    """每个已知缺陷都必须在 baseline 里有对应的文字理由，防止 xfail 沦为静默忽略。"""
    for key in ("margin_band_clearance_median_mm", "supra_margin_clearance_p95_mm"):
        assert key in baseline["known_defects"], f"{key} 缺少已知缺陷说明"
        assert len(baseline["known_defects"][key]) > 20, f"{key} 的说明过于简略"


# ===================================================================== helpers

def _live_chain() -> dict:
    """sanctioned 非平面链路的不变量（会话内缓存，避免每个用例重跑一遍）。"""
    if "live" not in _LIVE_CHAIN_CACHE:
        from tests.acceptance.synthetic_prep import prep_case

        case = prep_case()
        _LIVE_CHAIN_CACHE["live"] = M.measure_live_chain(
            case["gt_crown"], case["support"], case["margin"], case["axis"], case["params"])
    return _LIVE_CHAIN_CACHE["live"]


_SEAM_CACHE: dict = {}
_AXIS_CACHE: dict = {}
_LIVE_CHAIN_CACHE: dict = {}


def _seam(meshes) -> np.ndarray:
    """真值颈缘（会话内缓存）。

    **度量定义已变**：旧方法取最低 0.4 mm 带，只覆盖 16/24 方位角（残弧）。
    因此旧 baseline 数值已移入 ``legacy_measurement``，
    新旧数值不可直接比较 —— 变化里混了 Margin 真值修正，不是 blockout 改善。
    """
    key = "fdi36"
    if key not in _SEAM_CACHE:
        points, diag = M.gt_cervical_seam(meshes["gt_crown"], meshes["support"])
        _SEAM_CACHE[key] = (points, diag)
    return _SEAM_CACHE[key][0]


def _axis_diagnostics(meshes) -> dict:
    """就位方向诊断（会话内缓存，避免重复求解）。"""
    global _AXIS_CACHE
    key = "fdi36"
    if key not in _AXIS_CACHE:
        _AXIS_CACHE[key] = M.measure_insertion_axis(meshes["support"], _seam(meshes))
    return _AXIS_CACHE[key]


def _margin_band(meshes) -> dict:
    from app.geometry.intaglio import build_blockout_cutter

    margin = _seam(meshes)
    cutter = build_blockout_cutter(meshes["support"], margin, M.OCCLUSAL_AXIS, cement_gap_mm=0.06)
    return M.measure_margin_band_clearance(
        meshes["support"], cutter, z_margin=float(margin[:, 2].mean()))
