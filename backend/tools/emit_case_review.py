#!/usr/bin/env python3
"""病例评审工件（`case_review.json`）—— 用户 2026-10-09 决策 ①。

当锁定链（模板 → H.2.1 → J.2 → K0 → Contact）在 Contact Gate 无法自动通过时，
**不改病例设计参数、不放宽 Gate**，而是产出一份正式的 REVIEW 工件，
供前端明确告知技师"为什么不能自动生成、需要人工做什么"。

字段与理由码（用户给定 schema）：

    status = REVIEW_CONTACT
    reason_code = SPACE_CONSTRAINT_CONFLICT
    evidence:
        available_md_space_mm          # 患者可用近远中空间（分层剖面，非单值）
        template_md_width_range_mm     # 同 FDI 库存模板总宽范围
        residual_mesial_p95_mm
        residual_distal_p95_mm
        geometry_constraints_blocking_repair

FDI46（`cf_b1b73bae221211b1`）是第一个正式 REVIEW 正例：
A 同 FDI 模板重选 = NO_TEMPLATE_SOLUTION；B 参数化宽度重塑在锁定验收下全 α 被拒；
Contact 硬化全 α 被拒（几何门）⇒ SPACE_CONSTRAINT_CONFLICT。

用法::

    python tools/emit_case_review.py --run <chain 输出目录> [--case <病例包>]

输入：`<run>/h2_2b_chain.json`（+ `<run>/contact/contact_report.json`）。
输出：`<run>/case_review.json`（PHI-free：只有 uuid、坐标无关数值、计数）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: 库模板总宽（同 FDI 三基型，H.2.2-A inventory 实测，`73f8cb6`）
TEMPLATE_MD_WIDTH_RANGE_MM = (10.9, 11.4)


def _available_md_space(width_field: dict, template_range: tuple) -> dict:
    """可用近远中空间：**分层剖面**（每层 半宽 − deficit_p95），不是单值。

    `md_half_width_mm` 是模板/冠在该层的 MD 半宽；`deficit_p95_mm` 是该层
    proximal 墙到目标间隙的欠缺 —— 差值即"患者实际允许的半宽"。
    """
    prof = []
    for b in width_field.get("band_table", []):
        half = b.get("md_half_width_mm")
        need = b.get("deficit_p95_mm")
        if half is None or need is None:
            continue
        prof.append({"t": b.get("t"), "half_mm": round(float(half) - float(need), 4)})
    if not prof:
        return {"profile": [], "half_min_mm": None, "total_min_mm": None}
    half_min = min(p["half_mm"] for p in prof)
    return {
        "profile": prof,
        "half_min_mm": round(half_min, 4),
        "total_min_mm": round(2.0 * half_min, 4),
        "template_total_range_mm": [round(template_range[0], 3),
                                    round(template_range[1], 3)],
        "shortfall_per_side_mm": round(
            max(0.0, template_range[0] / 2.0 - half_min), 4),
    }


def build_review(run_dir: str, contact_report_path: str | None = None,
                 patch_qc: dict | None = None) -> dict:
    chain = json.load(open(os.path.join(run_dir, "h2_2b_chain.json"),
                           encoding="utf-8"))
    pq = patch_qc or chain.get("patch_qc") or {}
    contaminated = (pq.get("status") == "PATCH_SCOPE_CONTAMINATED")
    contact = chain.get("contact", {}) or {}
    gate = contact.get("gate", {}) or {}
    crep = {}
    cpath = contact_report_path or os.path.join(run_dir, "contact",
                                                "contact_report.json")
    if os.path.exists(cpath):
        crep = json.load(open(cpath, encoding="utf-8"))

    # ---- 阻止自动修复的几何约束（按本轮实测证据，逐条给出）
    blocking = []
    wf = chain.get("width_field", {}) or {}
    stages = {r.get("tag"): r for r in chain.get("stages", [])}
    h21 = stages.get("B+H.2.1") or stages.get("H.2.1") or {}
    j2s = stages.get("J.2") or stages.get("K0") or {}
    ratio = None
    if h21.get("mesial_p95") and j2s.get("mesial_p95"):
        ratio = j2s["mesial_p95"] / h21["mesial_p95"]
    if gate.get("status") == "CONTACT_CONSTRAINT_CONFLICT":
        blocking.append({
            "code": "CONTACT_ALPHA_LADDER_ALL_REJECTED",
            "detail": ("Contact 硬化（v2→I.1→I.2）所有 α 都被几何门拒绝："
                       "消除残留穿透所需的形变量必然产生翻面/自交；"
                       "硬几何不变量优先于穿透清零（不接受折叠换清零）")})
    blocking.append({
        "code": "MIN_THICKNESS_ENVELOPE_REINFLATION",
        "detail": ("J.2 的 minimum-thickness envelope 不知道邻牙空间，"
                   "会把 H.2.1 已收窄的 proximal 重新撑出去（本轮实测 "
                   "B+H.2.1 mesial %.4f → J.2 mesial %.4f）"
                   % (h21.get("mesial_p95", float("nan")),
                      j2s.get("mesial_p95", float("nan"))))})
    if wf:
        blocking.append({
            "code": "WIDTH_FIELD_SAFETY_FADE",
            "detail": ("参数化宽度重塑必须为咬合锁定区/牙尖形态在 t≥0.70 "
                       "淡出、t≥0.80 归零，而 deficit 质量正落在该区")})
    blocking.append({
        "code": "OCCLUSAL_LOCKED_AND_MARGIN_INVARIANTS",
        "detail": "margin 与咬合锁定区位移必须为 0，进一步压缩可行形变空间"})

    qc_findings = []
    if contaminated:
        qc_findings.extend(pq.get("findings", []))
        qc_findings.append({
            "code": "PATCH_SCOPE_QC",
            "detail": ("证据：%s｜近接大陆上 GT 冠零穿透（间隙 1.0–1.8mm），"
                       "污染集中在颊舌侧非近接区；patch u 跨段覆盖目标牙自身区域"
                       % (pq.get("gt_control") or {}))})
    if gate.get("geometry_clean") is False and gate.get("self_intersections"):
        qc_findings.append({
            "code": "J2_GEOMETRY_QC_FAIL_INHERITED",
            "detail": ("本次链的 J.2 输出残留 %d 处（micro）自交，被 Contact 继承 ——"
                       "frozen J.2 的 seam repair 对这份 anatomy 输入未收敛到 0，"
                       "按冻结标准应判 J2_GEOMETRY_QC_FAIL（见下一轮确定性修复）"
                       % int(gate.get("self_intersections") or 0))})

    review = {
        "status": "REVIEW_CONTACT",
        "reason_code": ("PATCH_SCOPE_CONTAMINATED" if contaminated
                        else "SPACE_CONSTRAINT_CONFLICT"),
        "secondary_reason_code": ("SPACE_CONSTRAINT_CONFLICT_UNVERIFIED"
                                  if contaminated else None),
        "case_uuid": chain.get("case_uuid")
        or os.path.basename(os.path.normpath(run_dir)),
        "message_zh": (
            ("本病例包的邻牙 patch 作用域被污染（整段牙弓 patch 含目标牙自身/颈缘）："
             "临床已交付的设计冠（GT）在同一 patch 上读出的穿透 p95 ≈ %.3fmm，"
             "因此'邻间隙不足'的结论**未经证实**；本次链的全部接触数字不可信。"
             "需要先重新裁剪 patch（只保留邻牙朝向本牙的近接壁）再重测。"
             % (pq.get("gt_control", {}).get("worst_penetration_p95_mm") or 0.0))
            if contaminated else
            ("当前邻间隙不足以在保持既定 Margin、厚度、咬合面和合法几何"
             "的条件下自动生成符合目标邻接的冠，需要人工调整设计。")),
        "evidence": {
            "available_md_space_mm": _available_md_space(
                wf, TEMPLATE_MD_WIDTH_RANGE_MM),
            "template_md_width_range_mm": list(TEMPLATE_MD_WIDTH_RANGE_MM),
            "template_reselection": "NO_TEMPLATE_SOLUTION",
            "width_reshape": "ALL_ALPHA_REJECTED",
            "residual_mesial_p95_mm": j2s.get("mesial_p95"),
            "residual_distal_p95_mm": j2s.get("distal_p95"),
            "contact_residual_editable_p95_mm": crep.get("editable", {}).get("p95"),
            "contact_residual_allow_mm": crep.get("residual_allow_mm"),
            "contact_geometry_clean": gate.get("geometry_clean"),
            "contact_self_intersections": gate.get("self_intersections"),
            "margin_displacement_mm": gate.get("margin_displacement_mm"),
            "occlusal_locked_displacement_mm":
                gate.get("occlusal_locked_displacement_mm"),
            "j2_reinflation_ratio": (round(ratio, 4) if ratio else None),
            "geometry_constraints_blocking_repair": blocking,
        },
        "qc_findings": qc_findings,
        "route_taken": {"A": "NO_TEMPLATE_SOLUTION",
                        "B": "MD_WIDTH_FIELD_ALL_ALPHA_REJECTED",
                        "C": "SPACE_CONSTRAINT_CONFLICT"},
        "next_action": "REVIEW_BY_TECHNICIAN",
    }
    return review


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--contact-report", default=None)
    ap.add_argument("--patch-qc", default=None,
                    help="neighbor_patch_scope_qc.json（默认读链 JSON 里的 patch_qc）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pq = (json.load(open(a.patch_qc, encoding="utf-8"))
          if a.patch_qc and os.path.exists(a.patch_qc) else None)
    review = build_review(a.run, a.contact_report, patch_qc=pq)
    out = a.out or os.path.join(a.run, "case_review.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(review, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"status": review["status"],
                      "reason_code": review["reason_code"],
                      "out": out}, ensure_ascii=False))
    print("available_md_space: half_min=%.3fmm total_min=%.3fmm | "
          "template=%.1f–%.1fmm | 每侧差 %.3fmm"
          % (review["evidence"]["available_md_space_mm"]["half_min_mm"],
             review["evidence"]["available_md_space_mm"]["total_min_mm"],
             TEMPLATE_MD_WIDTH_RANGE_MM[0], TEMPLATE_MD_WIDTH_RANGE_MM[1],
             review["evidence"]["available_md_space_mm"]["shortfall_per_side_mm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
