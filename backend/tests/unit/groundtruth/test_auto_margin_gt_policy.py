"""`AutoMarginGTSet` 准入判据的测试（用户 2026-09-30 拍板 a + 边界）。

核心断言：**本判据独立于 `is_geometric_gt_source()`** —— 接受 MODEL_INFO
作为 AutoMargin 的 Margin 来源，绝不改变几何 GT 体系对 MODEL_INFO 的排除。
"""
from __future__ import annotations

from app.groundtruth.auto_margin_gt import (
    ALLOWED_MARGIN_SOURCES,
    EVIDENCED_SELECTION_METHODS,
    FROZEN_RULE,
    MAX_PROJ_MAX_MM,
    MAX_PROJ_P95_MM,
    REQUIRED_ASSET_KIND,
    REQUIRED_SIDE_METHOD,
    auto_margin_exclusion_reasons,
    auto_margin_gt_set,
    is_auto_margin_gt_source,
)
from app.groundtruth.models import (
    CANONICAL_DESIGN_SOURCE,
    ToothGroundTruth,
    is_geometric_gt_source,
)


def _ok_row(**over):
    r = {
        "case_uuid": "cf_x", "fdi": "46", "qc_status": "OK",
        "asset_kind": REQUIRED_ASSET_KIND, "side_method": REQUIRED_SIDE_METHOD,
        "proj_p95_mm": "0.0", "proj_max_mm": "0.0",
        "selection_method": "MARGIN_PROJECTION_VERIFIED",
    }
    r.update(over)
    return r


# ============================================================ 接受 MODEL_INFO
def test_model_info_margin_is_accepted_for_auto_margin_only():
    """本题拍板的核心：MODEL_INFO 的 Margin 在有几何证据时可以进本集合。"""
    for src in ALLOWED_MARGIN_SOURCES:
        assert is_auto_margin_gt_source(_ok_row(), src), src


def test_unknown_or_other_margin_source_is_excluded():
    for bad in (None, "UNKNOWN"):
        why = auto_margin_exclusion_reasons(_ok_row(), bad)
        assert "excluded_margin_source_unknown" in why or \
            "excluded_margin_source_%s" % bad in why


# ============================================================ 回退永远不算证据
def test_fallback_asset_is_never_evidence():
    why = auto_margin_exclusion_reasons(
        _ok_row(selection_method="NEAREST_TO_CENTER_FALLBACK"), "MODEL_INFO")
    assert "excluded_not_evidenced_NEAREST_TO_CENTER_FALLBACK" in why
    assert not is_auto_margin_gt_source(
        _ok_row(selection_method="NEAREST_TO_CENTER_FALLBACK"), "MODEL_INFO")
    for m in EVIDENCED_SELECTION_METHODS:
        assert is_auto_margin_gt_source(_ok_row(selection_method=m), "MODEL_INFO")


def test_missing_selection_method_is_excluded():
    """构建表没有证据字段时宁可排除，不能默认放行。"""
    why = auto_margin_exclusion_reasons(_ok_row(selection_method=None),
                                        "MODEL_INFO")
    assert "excluded_selection_method_unknown" in why


# ============================================================ 形态 / 侧别 / 投影 / QC
def test_crown_shell_is_excluded():
    why = auto_margin_exclusion_reasons(
        _ok_row(asset_kind="CROWN_SHELL_INTAGLIO"), "MODEL_INFO")
    assert "excluded_asset_kind_CROWN_SHELL_INTAGLIO" in why


def test_wrong_side_method_is_excluded():
    """侧别必须由 above_frac 决定（实测 92/92 与旧跨度判据相反）。"""
    why = auto_margin_exclusion_reasons(_ok_row(side_method="GEOMETRY"),
                                        "MODEL_INFO")
    assert "excluded_side_method_GEOMETRY" in why


def test_projection_thresholds_are_enforced():
    assert is_auto_margin_gt_source(_ok_row(proj_p95_mm="0.1999"), "MODEL_INFO")
    why = auto_margin_exclusion_reasons(_ok_row(proj_p95_mm="0.25"), "MODEL_INFO")
    assert "excluded_proj_p95_over_threshold" in why
    why = auto_margin_exclusion_reasons(_ok_row(proj_max_mm="0.7"), "MODEL_INFO")
    assert "excluded_proj_max_over_threshold" in why
    # 缺数值也要排除（不能当成 0）
    why = auto_margin_exclusion_reasons(_ok_row(proj_p95_mm=""), "MODEL_INFO")
    assert "excluded_proj_p95_over_threshold" in why


def test_non_ok_qc_is_excluded():
    for st in ("REJECT", "REVIEW", None):
        why = auto_margin_exclusion_reasons(_ok_row(qc_status=st), "MODEL_INFO")
        assert any(w.startswith("excluded_qc_") for w in why)


# ============================================================ 集合级 + 对账
def test_auto_margin_gt_set_partitions_rows():
    rows = [_ok_row(), _ok_row(fdi="47"), _ok_row(qc_status="REJECT"),
            _ok_row(asset_kind="UNCLEAR")]
    mm = {"cf_x|46": "MODEL_INFO", "cf_x|47": "CONSTRUCTION_INFO"}
    kept, reasons = auto_margin_gt_set(rows, lambda u, f: mm.get(u + "|" + f))
    assert len(kept) == 2 and len(reasons) == 2, (len(kept), reasons)


# ============================================================ 边界：全局规则不动
def test_geometric_gt_source_untouched():
    """**这是本模块存在的理由**：MODEL_INFO 仍然不是几何 GT 源。"""
    common = dict(case_uuid="cf_x", source_case_id="s",
                  source_order_id="o", fdi=46,
                  reconstruction_type="AnatomicCrown")
    mi = ToothGroundTruth(design_source_kind="MODEL_INFO", **common)
    ci = ToothGroundTruth(design_source_kind=CANONICAL_DESIGN_SOURCE, **common)
    assert is_geometric_gt_source(mi) is False
    assert is_geometric_gt_source(ci) is True
    # 同一条 MODEL_INFO 行在 AutoMargin 口径下（有证据）可以通过 —— 两个
    # 判据服务不同任务，互不影响
    assert is_auto_margin_gt_source(_ok_row(), "MODEL_INFO") is True
    assert is_geometric_gt_source(mi) is False


def test_frozen_rule_is_documented():
    """规则文本必须与常量一致 —— 改常量不同步改文档就失败。"""
    for token in ("MODEL_INFO", "DIE_LIKE", "ABOVE_FRAC",
                  "%.2f" % MAX_PROJ_P95_MM, "%.2f" % MAX_PROJ_MAX_MM,
                  "case-level"):
        assert token in FROZEN_RULE, token
