"""`AutoMarginGTSet` 的准入判据 —— **独立于 `is_geometric_gt_source()`**。

用户 2026-09-30 拍板（选项 a）与边界：

    「`.modelInfo` 不是被"全局洗白"为可信几何 GT；只是对于这 122 个经过真实
     modeldie 几何配对、零投影误差、拓扑与人工检查全部通过的样本，可以作为
     `AutoMarginGTSet v1` 的 Margin GT 来源。」

所以本模块**只服务 Auto Margin 这一个任务**，绝不修改
``app.groundtruth.models.is_geometric_gt_source()``（那是 Axis / Support /
OuterCrown / SingleAnatomicCrown / Localization / EndToEnd 的共同闸门，
两套坐标语义必须继续严格分开）。

与「几何 GT」的区别在于**证据类型**：几何 GT 体系要求 "设计快照是 canonical 的
constructionInfo"；本集合要求 "Margin 与真实代型**几何上确认同帧**"，后者的直接
证明是 `proj_p95 = 0.0000mm`（若 Margin 来自另一个设计阶段，不可能精确落在代型
表面；抖动核验已排除"投影恒返回 0"的假象）。

**未采纳的做法（用户明确否决）**：把 `is_geometric_gt_source()` 从
`CONSTRUCTION_INFO only` 改成 `CONSTRUCTION_INFO or MODEL_INFO` —— 那会污染
上面列出的全部其它集合。同理，也不为了让 `SupportGTSet` 2028→10、
`SingleAnatomicCrownGTSet` 177→3 的数字"好看"而把 MODEL_INFO 放回去：旧
SupportGTSet 的规模本来就被 94.9% 的假代型高估了，那个数字不是要恢复的目标。
"""

from __future__ import annotations

#: 允许的 Margin 来源。**只在本集合内允许 MODEL_INFO**；离开本模块，
#: MODEL_INFO 仍然不是几何 GT 源（见模块 docstring）。
ALLOWED_MARGIN_SOURCES = ("CONSTRUCTION_INFO", "MODEL_INFO")

#: 投影门限（mm）。分布双峰：96.9% 精确为 0，坏例直接 2~10mm。
MAX_PROJ_P95_MM = 0.20
MAX_PROJ_MAX_MM = 0.60

#: 代型形态必须是"真的代型"（不是冠壳/桥体）。见 docs/AUTO_MARGIN_M0_ASSET_AUDIT.md。
REQUIRED_ASSET_KIND = "DIE_LIKE"

#: 侧别必须由 above_frac 决定（实测 92/92 与旧跨度判据相反）。
REQUIRED_SIDE_METHOD = "ABOVE_FRAC"

#: "有证据的代型"的选择方式。回退（NEAREST_TO_CENTER_FALLBACK）永远不算证据。
EVIDENCED_SELECTION_METHODS = ("MARGIN_PROJECTION_VERIFIED", "FILENAME_AND_FDI")

#: 拓扑/QC 必须通过（构建器里非 OK 的行不会进集合）。
REQUIRED_QC_STATUS = "OK"

FROZEN_RULE = (
    "AutoMarginGTSet 准入：margin_source ∈ {CONSTRUCTION_INFO, MODEL_INFO}\n"
    "  AND 真实 modeldie 已确认（selection_method ∈ "
    "{MARGIN_PROJECTION_VERIFIED, FILENAME_AND_FDI}；回退不算）\n"
    "  AND asset_kind == DIE_LIKE\n"
    "  AND proj_p95 <= 0.20mm AND proj_max <= 0.60mm\n"
    "  AND topology QC == OK\n"
    "  AND side_method == ABOVE_FRAC\n"
    "  AND split 必须是 case-level（按 case_uuid 分组）"
)


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def auto_margin_exclusion_reasons(build_row, margin_method=None):
    """逐条给出排除原因（空列表 = 准入）。`build_row` 是构建表的一行。"""
    why = []
    if build_row.get("qc_status") != REQUIRED_QC_STATUS:
        why.append("excluded_qc_%s" % (build_row.get("qc_status") or "missing"))
    if build_row.get("asset_kind") != REQUIRED_ASSET_KIND:
        why.append("excluded_asset_kind_%s"
                   % (build_row.get("asset_kind") or "missing"))
    if build_row.get("side_method") != REQUIRED_SIDE_METHOD:
        why.append("excluded_side_method_%s"
                   % (build_row.get("side_method") or "missing"))
    p95 = _f(build_row, "proj_p95_mm")
    pmax = _f(build_row, "proj_max_mm")
    if p95 is None or p95 > MAX_PROJ_P95_MM:
        why.append("excluded_proj_p95_over_threshold")
    if pmax is None or pmax > MAX_PROJ_MAX_MM:
        why.append("excluded_proj_max_over_threshold")
    if margin_method is None:
        why.append("excluded_margin_source_unknown")
    elif margin_method not in ALLOWED_MARGIN_SOURCES:
        why.append("excluded_margin_source_%s" % margin_method)
    sel = build_row.get("selection_method")
    if sel is None:
        # 构建表没有该列时（早期产物）用代型 id 是否存在代替不了 —— 宁可排除
        why.append("excluded_selection_method_unknown")
    elif sel not in EVIDENCED_SELECTION_METHODS:
        why.append("excluded_not_evidenced_%s" % sel)
    return why


def is_auto_margin_gt_source(build_row, margin_method=None) -> bool:
    """该构建行能否作为 AutoMarginGTSet 的一员（用户冻结规则）。"""
    return not auto_margin_exclusion_reasons(build_row, margin_method)


def auto_margin_gt_set(build_rows, margin_method_of=None):
    """按冻结规则过滤构建表行，并给出排除原因统计。

    ``margin_method_of`` 是 ``(case_uuid, fdi) -> margin_method`` 的取值函数；
    缺省视为未知（→ 排除），因为"Margin 来源"是这条规则的必需证据。
    """
    kept, reasons = [], []
    for r in build_rows:
        mm = (margin_method_of(r["case_uuid"], r["fdi"])
              if margin_method_of else None)
        why = auto_margin_exclusion_reasons(r, mm)
        if why:
            reasons.extend(why)
        else:
            kept.append(r)
    return kept, reasons
