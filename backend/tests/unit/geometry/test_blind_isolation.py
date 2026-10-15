"""Blind 隔离与 AUTO_ACCEPT 规则的测试（用户 2026-10-01 §1/§6）。

两条最要紧的：
  1. generation manifest **结构上不可能带 GT**（允许清单 + 点列检测）；
  2. 生成侧 runner **不引用** evaluation manifest、**不读任何 GT 字段**。
"""
from __future__ import annotations

import ast
import os

import pytest

from app.geometry.auto_margin import qc_rules
from app.geometry.auto_margin.blind_guard import assert_generation_manifest_clean
from app.geometry.auto_margin.models import (STATUS_AUTO_ACCEPTED,
                                             STATUS_REVIEW_REQUIRED)

TOOLS = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "tools")


def _src(name):
    p = os.path.join(TOOLS, name)
    if not os.path.isfile(p):
        pytest.skip("缺 %s" % p)
    return open(p, encoding="utf-8").read()


# ============================================================ manifest 隔离
def _gen_entry(**over):
    e = {"case_uuid": "cf_x", "fdi": "46", "source": "modeldie",
         "die_asset_id": "sha:ab", "die_sha256": "sha:ab",
         "axis": [0.0, 0.0, 1.0], "mesh_n_faces": 12,
         "input_schema_version": "blind_gen_v1"}
    e.update(over)
    return e


def test_generation_manifest_accepts_only_deployment_inputs():
    assert_generation_manifest_clean(_gen_entry())


def test_generation_manifest_rejects_gt_derived_fields():
    """**收紧版核心**：不只禁 Margin 点，连 GT 派生统计也一律不许进生成侧。"""
    for bad in ({"margin_gt": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]},
                {"points": [[0, 0, 0], [1, 1, 1], [2, 2, 2]]},
                {"margin_source": "MODEL_INFO"},
                {"proj_p95_mm": 0.0},
                {"input_qc_margin_projection_p95_mm": 0.0},
                {"side_method": "ABOVE_FRAC"},
                {"asset_kind": "DIE_LIKE"},
                {"topology_qc": "pass"},
                {"gt_path": "x"},
                {"provenance": {}}):
        with pytest.raises(AssertionError):
            assert_generation_manifest_clean(_gen_entry(**bad))


def test_split_selection_evidence_keeps_gt_stats_on_eval_side():
    """选择证据（投影统计/形态结论）只能留在 ①/③，不许进 ②。"""
    from app.geometry.auto_margin.blind_guard import split_selection_evidence
    row = {"case_uuid": "cf_x", "fdi": "46", "source": "modeldie",
           "best_asset_id": "sha:ab", "axis": [0.0, 0.0, 1.0],
           "best_n_faces": 12, "best_n_vertices": 10, "best_method": "MODEL_INFO",
           "best_p50_mm": 0.0, "best_p95_mm": 0.0, "best_max_mm": 0.0,
           "decision": "MARGIN_ON_DIE", "n_snapshots": 2, "n_candidates": 3,
           "n_margin": 300}
    g, e = split_selection_evidence(row)
    assert_generation_manifest_clean(g)              # ② 干净
    for k in ("best_p95_mm", "best_method", "decision", "best_n_faces"):
        assert k not in g, k
    assert e["gt_qc"]["proj_p95_mm"] == 0.0          # 统计留在 ③
    assert e["margin_source"] == "MODEL_INFO"
    assert "why_selected" in e["selection_evidence"]


def test_generation_manifest_allowlist_is_enforced():
    with pytest.raises(AssertionError):
        assert_generation_manifest_clean({"case_uuid": "cf_x",
                                          "some_new_field": 1})


def test_generation_runner_has_no_gt_access():
    """AST 级判定：不含评估 manifest 文件名、不读 GT 字段、不调用取 GT 的函数。"""
    src = _src("run_blind_generation.py")
    assert "blind_evaluation_manifest" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice,
                                                          ast.Constant):
            k = str(node.slice.value).lower()
            assert "gt" not in k, "生成侧读取了 GT 字段 %r" % node.slice.value
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "margin_points_of" not in called
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            assert "blind_manifest" not in mod, "生成侧不该依赖 manifest 工具"


# ============================================================ AUTO_ACCEPT 规则
def _q(**over):
    q = {"closed": True, "n_self_intersections": 0, "filled_sector_frac": 1.0,
         "proj_disp_max_mm": 0.05, "max_transition_mm": 0.6}
    q.update(over)
    return q


class _Cfg:
    max_proj_disp_mm = 0.60
    max_transition_mm = 1.50


def test_auto_accept_requires_all_qc():
    st, why = qc_rules.decide_status(feasible=True, quality=_q(), config=_Cfg())
    assert st == STATUS_AUTO_ACCEPTED and why == []


def test_review_on_each_qc_violation():
    for over in ({"closed": False}, {"n_self_intersections": 2},
                 {"filled_sector_frac": 0.9}, {"proj_disp_max_mm": 0.9},
                 {"max_transition_mm": 2.0}):
        st, why = qc_rules.decide_status(feasible=True, quality=_q(**over),
                                         config=_Cfg())
        assert st == STATUS_REVIEW_REQUIRED and why, over


def test_no_feasible_cycle_is_review_not_accept():
    st, why = qc_rules.decide_status(feasible=False, quality={}, config=_Cfg())
    assert st == STATUS_REVIEW_REQUIRED and why == ["NO_FEASIBLE_CYCLE"]


def test_energy_gap_is_recorded_but_never_rejects():
    """§6：energy_gap 在验证其与误差相关之前**只记录、不参与拒绝**。"""
    q = _q(energy_gap=1e-9, ambiguous_second_cycle=True)
    st, why = qc_rules.decide_status(feasible=True, quality=q, config=_Cfg())
    assert st == STATUS_AUTO_ACCEPTED, "置信度信号不得参与拒绝（尚未验证）"
