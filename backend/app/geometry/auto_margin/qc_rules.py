"""AUTO_ACCEPT / REVIEW / FAILED 的判定规则 —— **只用 GT-independent QC**。

用户 2026-10-01 §6 规格：第一版**不许用 GT 调置信度阈值**。`energy_gap` 在证明
与误差相关之前**只记录、不参与拒绝**（否则很容易在开发集上把阈值调过头）。

规则（冻结候选版）：

    AUTO_ACCEPTED
      - feasible cycle（DP 求出了闭环）
      - closed = true
      - self_intersection == 0
      - filled_sector_frac == 1.0
      - projection QC 通过（预测点投回表面的位移在阈值内）
      - transition sanity 通过（沿环的最大 3D 步长在阈值内）

    REVIEW_REQUIRED
      - no feasible cycle
      - candidate coverage incomplete
      - 结构性不同的第二解与最优解几何上太接近（歧义）—— **先只记录**
      - transition / perimeter / 几何异常

    FAILED
      - 网格非法 / 缺轴 / 数值失败 / 候选场无法构造

`AUTO_ACCEPTED` 是**合规**声明，不是精度声明；精度由 Accepted-only 指标单独报。
"""

from __future__ import annotations

from .models import (STATUS_AUTO_ACCEPTED, STATUS_FAILED,
                     STATUS_REVIEW_REQUIRED)


def decide_status(*, feasible: bool, quality: dict, config) -> tuple:
    """→ (status, reasons)。``quality`` 是 engine 的 QC 字典。"""
    q = quality or {}
    if not feasible:
        return STATUS_REVIEW_REQUIRED, ["NO_FEASIBLE_CYCLE"]
    reasons = []
    if q.get("closed") is not True:
        reasons.append("loop_not_closed")
    if (q.get("n_self_intersections") or 0) != 0:
        reasons.append("self_intersection=%s" % q.get("n_self_intersections"))
    frac = q.get("filled_sector_frac")
    if frac is not None and frac < 1.0:
        reasons.append("candidate_coverage=%.3f" % frac)
    disp = q.get("proj_disp_max_mm")
    if disp is not None and disp > config.max_proj_disp_mm:
        reasons.append("projection_disp=%.3f" % disp)
    mt = q.get("max_transition_mm")
    if mt is not None and mt > config.max_transition_mm:
        reasons.append("transition=%.3f" % mt)
    if reasons:
        return STATUS_REVIEW_REQUIRED, reasons
    return STATUS_AUTO_ACCEPTED, []


def is_failed_reason(reason: str) -> bool:
    """哪些 reason 属于 FAILED（输入/数值层），而不是 REVIEW（几何层）。"""
    return reason.split("(")[0] in {
        "invalid_mesh", "invalid_axis", "numerical_failure",
        "candidate_field_failed", "candidate_coverage_insufficient",
    }
