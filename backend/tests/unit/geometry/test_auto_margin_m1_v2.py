"""M1 v2 环状 DP 的单元测试 —— 用合成格点，不需要网格。

直接构造 ``FaceStat``（fid 即列表下标）与 ``{sector: [(score, fid)...]}``，
因此测的是**路径优化本身**：闭环是否真闭环、硬门是否生效、高分离群点是否被拒绝。
"""
from __future__ import annotations

import math

from app.geometry.auto_margin import graph_search
from app.geometry.auto_margin.models import AutoMarginConfig
from app.geometry.auto_margin.prep_frame import build_frame
from app.geometry.auto_margin.features import FaceStat


def _stat(fid, h, r, theta):
    return FaceStat(fid=fid, centroid=(r*math.cos(theta), h, r*math.sin(theta)),
                    normal=(1.0, 0.0, 0.0), area=1.0, h=h, r=r, theta=theta,
                    nx=0.0, sharp_deg=10.0, radial_slope=0.0)


def _frame(n_sectors):
    return build_frame([(1.0, 0.0, 0.0)], (0.0, 1.0, 0.0))


def _lattice(n=24, decoy_h=3.0, decoy_sectors=(5, 6)):
    """每个扇区 2 个候选：A（h=0，正确环）+ D（h=decoy_h，个别扇区里分数更高）。

    D 只在 decoy_sectors 出现且分数更高 → 独立 argmax 会在那两个扇区跳上去，
    环状 DP 因硬门 |Δh| ≤ max_dh 而必须拒绝它。
    """
    stats = []
    cands = {}
    for s in range(n):
        th = 2 * math.pi * s / n
        fa = len(stats)
        stats.append(_stat(fa, 0.0, 5.0, th))
        items = [(1.0, fa)]
        if s in decoy_sectors:
            fd = len(stats)
            stats.append(_stat(fd, decoy_h, 5.0, th))
            items.append((2.0, fd))          # 分数更高
        items.sort(key=lambda x: -x[0])
        cands[s] = items
    return stats, cands


def test_argmax_picks_the_high_score_outlier():
    """v1 的已知失败模式：独立 argmax 会跳到高分离群点。"""
    stats, cands = _lattice()
    path = graph_search.argmax_path(cands, stats)
    hs = [stats[fid].h for _s, fid in sorted(path.items())]
    assert max(hs) == 3.0, "argmax 本应被高分离群点勾走（这正是 v1 的问题）"


def test_cyclic_dp_rejects_the_outlier_via_hard_gate():
    """v2：硬门禁止 |Δh| > max_dh 的转移 → 整圈留在 h=0。"""
    stats, cands = _lattice(decoy_h=3.0)
    cfg = AutoMarginConfig(method="dp", max_dh_mm=1.2, max_dr_mm=0.6)
    path, _e1, _e2 = graph_search.cyclic_dp_path(cands, stats, _frame(24), cfg)
    assert path, "应求出闭环"
    assert len(path) == 24
    hs = [stats[fid].h for _s, fid in sorted(path.items())]
    assert max(hs) == 0.0, "闭环应全部留在 h=0（离群点被硬门挡住）"


def test_dp_prefers_smooth_ring_even_without_hard_gate():
    """即使把硬门放宽到不生效，平滑项也应当把离群点压掉（λ 起作用的证明）。"""
    stats, cands = _lattice(decoy_h=1.0)
    cfg = AutoMarginConfig(method="dp", max_dh_mm=99.0, max_dr_mm=99.0,
                           lambda_h=10.0, huber_delta_mm=0.2)
    path, _e1, _e2 = graph_search.cyclic_dp_path(cands, stats, _frame(24), cfg)
    hs = [stats[fid].h for _s, fid in sorted(path.items())]
    assert max(hs) == 0.0


def test_dp_path_is_a_true_cycle():
    """闭环性：每个扇区恰好一个点，且末→首的转移也满足硬门。"""
    stats, cands = _lattice()
    cfg = AutoMarginConfig(method="dp", max_dh_mm=1.2, max_dr_mm=0.6)
    path, _e1, _e2 = graph_search.cyclic_dp_path(cands, stats, _frame(24), cfg)
    assert sorted(path) == list(range(24))
    secs = sorted(path)
    for i in range(len(secs)):
        a = stats[path[secs[i]]]
        b = stats[path[secs[(i + 1) % len(secs)]]]
        assert abs(b.h - a.h) <= cfg.max_dh_mm + 1e-9
        assert abs(b.r - a.r) <= cfg.max_dr_mm + 1e-9


def test_dp_reports_energy_gap():
    """新增：返回 (path, E1, E2)，E2 是**起点候选不同**的另一条环的能量。"""
    stats, cands = _lattice(n=24, decoy_h=3.0)
    cfg = AutoMarginConfig(method="dp", max_dh_mm=1.2, max_dr_mm=0.6)
    path, e1, e2 = graph_search.cyclic_dp_path(cands, stats, _frame(24), cfg)
    assert path and e1 is not None
    assert e2 is None or e2 >= e1, "次优能量不应小于最优"


def test_dp_returns_empty_when_gates_make_it_infeasible():
    """候选之间全被硬门隔断时返回 {}（调用方回退并报 REVIEW，不静默）。"""
    stats = []
    cands = {}
    for s in range(24):
        th = 2 * math.pi * s / 24
        f = len(stats)
        # 高度在 0 和 10 之间交替 → 任何相邻转移都超门
        stats.append(_stat(f, 0.0 if s % 2 == 0 else 10.0, 5.0, th))
        cands[s] = [(1.0, f)]
    cfg = AutoMarginConfig(method="dp", max_dh_mm=1.0)
    assert graph_search.cyclic_dp_path(cands, stats, _frame(24), cfg)[0] == {}


def test_dp_handles_varying_candidate_counts():
    """各扇区候选数不同（1~5）也要能跑通。"""
    stats = []
    cands = {}
    for s in range(30):
        th = 2 * math.pi * s / 30
        items = []
        for k in range(1 + (s % 5)):
            f = len(stats)
            stats.append(_stat(f, 0.05 * k, 5.0 + 0.02 * k, th))
            items.append((1.0 - 0.1 * k, f))
        items.sort(key=lambda x: -x[0])
        cands[s] = items
    cfg = AutoMarginConfig(method="dp", max_dh_mm=1.2, max_dr_mm=0.6)
    path, _e1, _e2 = graph_search.cyclic_dp_path(cands, stats, _frame(30), cfg)
    assert len(path) == 30


def test_dp_too_few_sectors_returns_empty():
    stats, cands = _lattice(n=6)
    cfg = AutoMarginConfig(method="dp")
    assert graph_search.cyclic_dp_path(cands, stats, _frame(6), cfg)[0] == {}


# ============================================================ B：fail-loud 语义
def test_no_feasible_cycle_is_review_without_margin(monkeypatch):
    """**生产语义**：无可行闭环时必须 REVIEW_REQUIRED 且 margin=None。

    直接针对引擎分支测（合成件上存在等高环，收小门也可能有可行环，不能靠几何
    构造"必然不可行"）。monkeypatch DP 返回不可行 —— 这正是生产上会遇到的那条路。
    """
    from app.geometry.auto_margin import engine, graph_search
    from app.geometry.auto_margin.models import (STATUS_REVIEW_REQUIRED,
                                                 STATUS_AUTO_ACCEPTED)
    from tests.unit.geometry.test_auto_margin_m1 import die_with_margin
    tris, _y = die_with_margin()
    # 先验正常路径（此时还没打补丁）：AUTO_ACCEPTED + 齐备的 confidence/provenance
    ok = engine.predict_margin(tris, (0.0, 1.0, 0.0), AutoMarginConfig())
    assert ok.status == STATUS_AUTO_ACCEPTED, ok.status
    assert ok.margin is not None and ok.loop_energy is not None
    assert ok.config_id and ok.score_method and ok.transition_method
    assert "energy_gap" in ok.confidence
    assert ok.candidate_rank_stats.get("mean") is not None
    # 再验生产语义：DP 不可行 → REVIEW + 无曲线
    monkeypatch.setattr(graph_search, "cyclic_dp_path",
                        lambda *a, **k: ({}, None, None))
    res = engine.predict_margin(tris, (0.0, 1.0, 0.0),
                                AutoMarginConfig(fallback_mode="REVIEW"))
    assert res.status == STATUS_REVIEW_REQUIRED, res.status
    assert res.reason == "NO_FEASIBLE_CYCLE", res.reason
    assert res.margin is None, "生产语义下不得输出兜底曲线"
    assert res.config_id, "provenance 必须仍然写入"


def test_dev_argmax_fallback_still_available_and_labelled(monkeypatch):
    """研发模式保留回退，但必须**如实标注**（供诊断，不做生产语义）。"""
    from app.geometry.auto_margin import engine, graph_search
    from tests.unit.geometry.test_auto_margin_m1 import die_with_margin
    tris, _y = die_with_margin()
    monkeypatch.setattr(graph_search, "cyclic_dp_path",
                        lambda *a, **k: ({}, None, None))
    res = engine.predict_margin(tris, (0.0, 1.0, 0.0),
                                AutoMarginConfig(fallback_mode="ARGMAX"))
    assert res.margin is not None
    assert "dp_infeasible_fallback_argmax" in res.reason
