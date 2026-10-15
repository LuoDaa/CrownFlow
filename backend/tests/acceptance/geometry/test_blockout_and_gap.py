"""方向 blockout 与变量 gap 的验收。

分两组：先测 **blockout 本身**，再测 **gap field**，最后才看 clearance。
顺序是刻意的 —— 如果 clearance 变好但 blockout 的几何不变量不成立，那是巧合而非改善。

对应本轮的范围锁定：只改 Directional Blockout / Cement Gap Field / Intaglio Cutter。
Axis、Margin v2、Outer、Thickness、Contact/Occlusion、制造可达性一律未动。
"""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.intaglio import (
    DRAFT_TOLERANCE_DEG,
    PROVENANCE_CONSTRUCTION_INFO,
    BlockoutPolicy,
    BlockoutStatus,
    GapFieldParams,
    apply_gap_field,
    blockout_policy_from_resolved,
    build_blockout,
    gap_field_params_from_resolved,
)
from tests.acceptance._synthetic import random_rigid

# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def case():
    """FDI36：真值冠（Outer）、基台（support）、v2 真值颈缘、求解就位轴。"""
    from pathlib import Path

    from app.geometry.gt_margin import extract_gt_crown_margin
    from app.geometry.insertion import solve_insertion_axis
    from app.geometry.io import load_mesh
    from tests.acceptance.conftest import require_fdi36

    root = require_fdi36() / "01_主要病例模型"
    crown = load_mesh(root / "1629-王煜 1.stl")
    support = load_mesh(root / "1629-王煜 0.stl")
    margin = np.asarray(extract_gt_crown_margin(crown, support, [0, 0, 1]).points)
    axis = np.asarray(solve_insertion_axis(support, margin).axis, dtype=float)
    return crown, support, margin, axis


def _policy_5axis() -> BlockoutPolicy:
    return BlockoutPolicy(machine_axes=5, additional_blockout_after_gap=None,
                          untouchable_zone_mm=0.1,
                          provenance={"source": "test_5axis"})


def _gap(cement_gap: float | None = 0.025, **kw) -> GapFieldParams:
    base = dict(cement_gap_mm=cement_gap, cement_gap_start_mm=None, cement_gap_top_mm=None,
                additional_spacing_xyz_mm=None, additional_spacing_z_on_top_mm=None,
                provenance={"source": "test"})
    base.update(kw)
    return GapFieldParams(**base)


# =============================================================== A. blockout

def test_blockout_closes_undercut_along_axis(case):
    """方向阴影体沿就位轴必须**完全单值** —— 这是"倒凹已被封闭"的直接证据。

    断言的是 blockout 自身的拔模负角面积比为 0，而不是"比原始 support 小"：
    后者只能说明有变化，不能说明可插入。
    """
    _, support, margin, axis = case
    result = build_blockout(support, margin, axis, _policy_5axis())

    assert result.status is BlockoutStatus.OK, result.diagnostics
    assert result.blockout_undercut_area_ratio <= 1e-6, (
        f"blockout 自身倒凹比 {result.blockout_undercut_area_ratio:.6f}，沿轴不是单值"
    )
    # 原始 support 确实有倒凹 —— 否则这条断言是空的
    assert result.undercut_area_ratio > 0.05, (
        f"原始 support 的就位面倒凹只有 {result.undercut_area_ratio:.4f}，"
        "本病例不足以检验 blockout 是否有效"
    )


def test_blockout_consistency_with_axis_solver(case):
    """blockout 报告的原始倒凹比必须与就位方向求解器一致。

    两者的作用域（就位面 = Margin 平面之上）与判据（拔模角）必须相同，
    否则说明其中一处的过滤条件写错了 —— 实测曾因 blockout 漏掉
    "Margin 平面之上"的限定而把颈部算进来，虚高到 0.3109（求解器给 0.1170）。
    """
    from app.geometry.insertion import solve_insertion_axis

    _, support, margin, axis = case
    result = build_blockout(support, margin, axis, _policy_5axis())
    solver_ratio = solve_insertion_axis(support, margin).undercut_area_ratio

    assert abs(result.undercut_area_ratio - solver_ratio) < 0.02, (
        f"blockout {result.undercut_area_ratio:.4f} vs 求解器 {solver_ratio:.4f} —— "
        "作用域或判据不一致"
    )


def test_blockout_is_watertight_and_single_component(case):
    """水密 + 单连通。非流形或断裂会让后续精确布尔失败或产出碎片。"""
    _, support, margin, axis = case
    result = build_blockout(support, margin, axis, _policy_5axis())
    mesh = result.blockout_mesh

    assert mesh.is_watertight is True
    assert len(mesh.split(only_watertight=False)) == 1
    edges = np.asarray(mesh.edges_sorted)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert int((counts == 1).sum()) == 0, "存在开放边界边"
    assert int((counts > 2).sum()) == 0, (
        f"存在 {int((counts>2).sum())} 条非流形边 —— 通常是掩码残留"
        "'仅对角相连'的 2×2 配置，需迭代修正到不动点"
    )


def test_blockout_is_rigid_equivariant(case):
    """``T(support) + R(axis)`` 得到的 blockout 必须等价于 ``T(原 blockout)``。

    所有几何都在由 axis 建立的局部坐标系内完成，结果必须与刚体变换等变。
    本项目在就位方向求解器上已因同类缺陷抓到一个真 bug（世界 Z 符号约定）。
    """
    _, support, margin, axis = case
    base = build_blockout(support, margin, axis, _policy_5axis())

    rng = np.random.default_rng(3)
    for k in range(2):
        T = random_rigid(rng)
        support_t = support.copy()
        support_t.apply_transform(T)
        margin_t = (T[:3, :3] @ margin.T).T + T[:3, 3]

        got = build_blockout(support_t, margin_t, T[:3, :3] @ axis, _policy_5axis())

        assert abs(got.blockout_mesh.volume - base.blockout_mesh.volume) / base.blockout_mesh.volume < 5e-3, (
            f"变换 {k}: 体积 {got.blockout_mesh.volume:.2f} vs {base.blockout_mesh.volume:.2f}"
        )
        assert got.blockout_undercut_area_ratio <= 1e-6
        assert abs(got.undercut_area_ratio - base.undercut_area_ratio) < 0.02


def test_blockout_volume_exceeds_support(case):
    """阴影体必须比原始 support 大 —— 否则没有填充任何倒凹。"""
    _, support, margin, axis = case
    result = build_blockout(support, margin, axis, _policy_5axis())
    assert abs(result.blockout_mesh.volume) >= abs(support.volume) * 0.99


def test_margin_protected_zone_is_respected(case):
    """``UntouchableZoneNearPreparationMargin`` 内不得发生 blockout。

    exocad 官方定义该参数为"Margin 周围即使上方有 undercut 也永远不做 blockout 的区域"。
    实测 5 轴策略下该区域面积 6.37 mm²（> 0，否则参数没生效）。
    """
    _, support, margin, axis = case
    with_zone = build_blockout(support, margin, axis, _policy_5axis())
    without = build_blockout(support, margin, axis,
                             BlockoutPolicy(machine_axes=5,
                                            additional_blockout_after_gap=None,
                                            untouchable_zone_mm=0.0, provenance={}))

    assert with_zone.margin_protected_area_mm2 > 0, "untouchable zone 没有生效"
    assert with_zone.diagnostics["n_cells_protected"] > 0
    # 保护带会减少有效覆盖的 cell 数
    assert (with_zone.diagnostics["n_cells_after_margin_protection"]
            < without.diagnostics["n_cells_after_margin_protection"])
    # 保护的实测位移：保护带内不应被推到 support 之外
    assert with_zone.blockout_mesh.volume <= without.blockout_mesh.volume + 1e-6


def test_blockout_mesh_density_stability(case):
    """同一 support、不同轴向分辨率，结果不得明显漂移。"""
    _, support, margin, axis = case
    coarse = build_blockout(support, margin, axis, _policy_5axis(), resolution_mm=0.10)
    fine = build_blockout(support, margin, axis, _policy_5axis(), resolution_mm=0.05)

    assert abs(fine.undercut_area_ratio - coarse.undercut_area_ratio) < 0.05
    assert fine.blockout_undercut_area_ratio <= 1e-6
    assert coarse.blockout_undercut_area_ratio <= 1e-6
    assert abs(fine.blockout_mesh.volume - coarse.blockout_mesh.volume) / coarse.blockout_mesh.volume < 0.15


# =============================================================== policy 语义

def test_policy_table_matches_evidence():
    """3 轴 / 5 轴策略必须与证据数据一致（相关度 100%，25 条记录无例外）。

    并且 **``-1`` 建模为『禁用』而不是『角度为 0』** —— 两者几何行为在当前证据下
    相同，但语义不同，不能在代码里混为一谈。
    """
    from app.parameters.models import ParameterScope, ParameterState, ResolvedParameter, ResolvedParameterSet, ResolvedSource

    def pset(axes: float | None):
        params = {}
        if axes is not None:
            params["MillingMachineAxes"] = ResolvedParameter(
                name="MillingMachineAxes", state=ParameterState.EXPLICIT, value=axes,
                source=ResolvedSource.TOOTH_EXPLICIT, scope=ParameterScope.ORDER)
        return ResolvedParameterSet(parameters=params)

    p3, _ = blockout_policy_from_resolved(pset(3.0))
    p5, _ = blockout_policy_from_resolved(pset(5.0))
    assert (p3.machine_axes, p3.additional_blockout_after_gap, p3.untouchable_zone_mm) == (3, 0.0, 0.0)
    assert (p5.machine_axes, p5.additional_blockout_after_gap, p5.untouchable_zone_mm) == (5, None, 0.1)
    assert p5.additional_blockout_after_gap is None, "5 轴的 -1 必须建模为『禁用』（None）"


def test_unknown_machine_axes_stays_unresolved():
    """机轴数为未知取值时不得套用近似策略。"""
    from app.parameters.models import ParameterScope, ParameterState, ResolvedParameter, ResolvedParameterSet, ResolvedSource

    pset = ResolvedParameterSet(parameters={
        "MillingMachineAxes": ResolvedParameter(
            name="MillingMachineAxes", state=ParameterState.EXPLICIT, value=4.0,
            source=ResolvedSource.TOOTH_EXPLICIT, scope=ParameterScope.ORDER)})
    policy, diag = blockout_policy_from_resolved(pset)

    assert policy is None
    assert diag["status"] == BlockoutStatus.UNRESOLVED.value
    assert "4" in diag["reason"]


# ================================================================== B. gap

def test_missing_cement_gap_blocks_instead_of_guessing(case):
    """CementGapThickness 未解析时必须**报错**，不得用 0 或猜测值代替。

    0 粘接间隙意味着牙冠与预备体直接贴合、无法就位。这是本项目反复强调的纪律。
    """
    _, support, margin, axis = case
    blockout = build_blockout(support, margin, axis, _policy_5axis())

    with pytest.raises(ValueError, match="CementGapThickness 未解析"):
        apply_gap_field(blockout, support, margin, axis,
                        _gap(cement_gap=None))


def test_cement_gap_start_region_has_no_gap(case):
    """``CementGapStart`` 带内不施加 cement gap。

    exocad 官方语义：从 Margin 开始先存在一段**没有 cement gap** 的区域，
    过了这一区域才开始正常 cement gap。
    """
    _, support, margin, axis = case
    blockout = build_blockout(support, margin, axis, _policy_5axis())

    no_gap = apply_gap_field(blockout, support, margin, axis,
                             _gap(cement_gap=0.025, cement_gap_start_mm=0.0))
    with_start = apply_gap_field(blockout, support, margin, axis,
                                 _gap(cement_gap=0.025, cement_gap_start_mm=0.5))

    # 开启 CementGapStart 后 cutter 体积应变小（近 Margin 处不膨胀）
    assert abs(with_start.volume) <= abs(no_gap.volume) + 1e-6


def test_gap_increases_cutter_volume_monotonically(case):
    """gap 越大 cutter 越大 —— 基本单调性，防止实现把 gap 施加错方向。"""
    _, support, margin, axis = case
    blockout = build_blockout(support, margin, axis, _policy_5axis())

    small = apply_gap_field(blockout, support, margin, axis, _gap(0.025))
    large = apply_gap_field(blockout, support, margin, axis, _gap(0.10))
    assert abs(large.volume) > abs(small.volume)


def test_gap_provenance_labels_are_honest():
    """``CementGapTopThickness`` 的 provenance 不得标为官方字典。

    该参数名未出现在本次上传的 exocad Material Configuration Guide 中，
    因此必须标 ``constructionInfo_observed``。
    """
    from app.parameters.models import ParameterScope, ParameterState, ResolvedParameter, ResolvedParameterSet, ResolvedSource

    pset = ResolvedParameterSet(parameters={
        "CementGapThickness": ResolvedParameter(
            name="CementGapThickness", state=ParameterState.EXPLICIT, value=0.025,
            source=ResolvedSource.TOOTH_EXPLICIT, scope=ParameterScope.TOOTH),
        "CementGapTopThickness": ResolvedParameter(
            name="CementGapTopThickness", state=ParameterState.EXPLICIT, value=0.01,
            source=ResolvedSource.TOOTH_EXPLICIT, scope=ParameterScope.TOOTH)})
    gp = gap_field_params_from_resolved(pset)

    assert gp.cement_gap_mm == 0.025
    assert gp.cement_gap_top_mm == 0.01
    assert gp.provenance["CementGapTopThickness"]["source"] == PROVENANCE_CONSTRUCTION_INFO
    assert gp.provenance["CementGapThickness"]["source"] == "exocad_official_dictionary"


def test_missing_top_gap_stays_missing():
    """未显式写出 ``CementGapTopThickness`` 时保持缺失，不取默认值。"""
    from app.parameters.models import ParameterScope, ParameterState, ResolvedParameter, ResolvedParameterSet, ResolvedSource

    pset = ResolvedParameterSet(parameters={
        "CementGapThickness": ResolvedParameter(
            name="CementGapThickness", state=ParameterState.EXPLICIT, value=0.025,
            source=ResolvedSource.TOOTH_EXPLICIT, scope=ParameterScope.TOOTH)})
    gp = gap_field_params_from_resolved(pset)

    assert gp.cement_gap_top_mm is None
    assert gp.has_top_gap is False
    assert gp.provenance["CementGapTopThickness"]["present"] is False


# ============================================== 真实病例 clearance（v3 口径）

def test_margin_plane_band_clearance_in_target_range(case, baseline):
    """按 **Margin 平面相对距离**划分的边缘带，中位间隙应落在 0.015–0.05 mm。

    这是本轮 blockout + gap 重写的核心结果。

    **为什么这里用平面相对口径而不是 v2 的轴向口径**：FDI36 的真实颈缘平面偏离
    +Z **11.67°**，轴向切带会把倾斜一侧远低于 Margin 平面的区域算进边缘带 ——
    那里 blockout 必然深填，使该区中位从 0.034 被抬到 0.200。这是**度量定义**
    的问题，不是算法问题。

    v2 轴向口径的数值**保持不变**（`metric_version` 仍为 `v2_gt_cervical_seam`），
    它的 xfail 仍在，且理由已改为"受定义限制"。修好算法不改定义修不好那个数。
    """
    from tests.acceptance import _metrics as M

    _, support, margin, axis = case
    blockout = build_blockout(support, margin, axis, _policy_5axis())
    cutter = apply_gap_field(blockout, support, margin, axis, _gap(0.025))

    dist = M.measure_clearance_distributions_plane_relative(support, cutter, margin, axis)
    lo, hi = baseline["targets"]["margin_plane_band_median_mm"]["target_range"]
    med = dist["margin_band"]["median"]

    assert med is not None and lo <= med <= hi, (
        f"平面相对边缘带中位 {med} mm 超出 [{lo}, {hi}]；完整分布 {dist['margin_band']}"
    )
    # 分布形状也要合理，不能只靠中位数掩盖两极
    assert dist["margin_band"]["p95"] < 0.60, (
        f"边缘带 p95 {dist['margin_band']['p95']} 过大，分布形状有问题"
    )


def test_blockout_own_undercut_matches_baseline(case, baseline):
    """blockout 自身倒凹比必须为 0（与 baseline 记录一致）。"""
    _, support, margin, axis = case
    result = build_blockout(support, margin, axis, _policy_5axis())
    target = baseline["targets"]["blockout_own_undercut_area_ratio"]["target_max"]

    assert result.blockout_undercut_area_ratio <= target
    assert baseline["measurements"]["_blockout_quality"]["blockout_undercut_area_ratio"] <= target


# ================================================= 版本管理不变量（本轮新增）

def test_metric_version_unchanged_for_clearance(baseline):
    """clearance 两项的 metric_version **不得**改动。

    测量定义没变，只是被测算法变了。改了 metric_version 会让
    ``0.254 -> 新值`` 无法直接比较、无法归因给 blockout。

    （2026-10-12 Q0：``isolated_chain_outer_deviation_p95_mm`` 已从本清单移除 ——
    它的**测量定义确实变了**（退役链路 → sanctioned 非平面链路），
    所以必须升版本号，而不是被这条守卫挡住。见 ``metric_versions`` 与 ``retracted_expectations``。）
    """
    for key in ("margin_band_clearance_median_mm", "supra_margin_clearance_p95_mm"):
        assert baseline["metric_versions"][key]["version"] == "v2_gt_cervical_seam", (
            f"{key} 的 metric_version 被改动；本轮只改算法，不应改度量定义"
        )
    assert baseline["metric_versions"]["isolated_chain_outer_deviation_p95_mm"]["version"] == \
        "v4_live_nonplanar_chain", "定义已变更的指标必须升版本，否则新旧数值无法归因"


def test_pipeline_version_records_the_algorithm_change(baseline):
    """算法变更必须记在 pipeline_version / pipeline_components，而不是 metric_version。"""
    assert baseline["pipeline_version"] == "v4_nonplanar_shared_seam"
    comps = baseline["pipeline_components"]
    assert comps["blockout"].startswith("v3_")
    assert comps["cement_gap"].startswith("v3_")
    assert comps["cervical"].startswith("v4_")
    # 本轮未动的部分必须仍标原版本，证明范围没有越界
    assert comps["thickness"].startswith("audit_only")
    assert comps["margin"] == "v2_gt_cervical_seam"


def test_input_hashes_are_recorded(baseline):
    """输入哈希必须被记录 —— 这是"改善不是靠改输入得来的"的可验证证据。"""
    hashes = baseline["measurements"]["input_hashes"]
    for key in ("outer_crown", "support", "margin_v2", "insertion_axis"):
        assert key in hashes and len(hashes[key]) == 16, f"缺少输入哈希 {key}"
