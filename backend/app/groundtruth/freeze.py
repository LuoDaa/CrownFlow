"""GT Set Freeze —— 把索引与审计 join 起来，冻结一份**版本化的真值清单**。

    python3 -m app.groundtruth.freeze \\
        --index /tmp/gt_index --audit /tmp/gt_audit_full \\
        --out /tmp/gt_frozen --dataset-version exocad_gt_v1

## 为什么要 join

两个流水线各管一半，**判定不能有两处**：

  * ``GroundTruthIndexer`` 管 颈缘 / 就位方向 / 中心 / 导出冠 / 代型 的有效性
  * ``Arch Asset Audit``   管 **弓资产到底是什么**（含不含目标预备体、是不是对颌）

弓资产那一列由审计回填（``arch_asset_class``），索引不另算一套 ——
否则两处阈值漂移，没人说得清某个病例为什么进/不进集合。

## 只写脚本算出来的数字

manifest 里每个数字都是本次运行实际数出来的。**不预留、不估计、不填期望值。**
同时还写出 ``exclusion_reasons``，让"672 例为什么只剩 N 例"一条命令可答。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .human_labels import (
    LOCALIZATION_ROUTE_EVIDENCE,
    LOCALIZATION_ROUTE_STATUS,
    QA2_CONFUSION,
    QA2_STRATA,
    labels_agree_across_rounds,
    qa_round_summary,
)
from .models import (
    COLUMNS,
    HUMAN_PREP_POSITIVE,
    LOCALIZATION_ADMITTED_CLASS,
    LOCALIZATION_REQUIRES_HUMAN_PREP,
    ToothGroundTruth,
    build_all_sets,
    case_group_key,
    human_prep_label,
    localization_exclusion_reasons,
)

#: 弓资产类别之间的"强度"顺序。同一病例里多个候选资产时取最强的一类 ——
#: 只要有一个候选真的是"整颌含目标"，就不该被同病例的其他候选拉低。
_CLASS_STRENGTH: tuple[str, ...] = (
    LOCALIZATION_ADMITTED_CLASS,          # FULL_ARCH_WITH_TARGET
    "FULL_ARCH_ANTAGONIST",
    "FULL_ARCH_TARGET_MISSING",
    "PARTIAL_ARCH",
    "MULTI_UNIT_RESTORATION",
    "SINGLE_TOOTH",
    "UNKNOWN",
)


def _rank(cls: str) -> int:
    try:
        return _CLASS_STRENGTH.index(cls)
    except ValueError:
        return len(_CLASS_STRENGTH)


def _to_row(d: dict) -> ToothGroundTruth:
    """把索引 CSV 的一行还原成 ``ToothGroundTruth``。

    CSV 里全是字符串，所以要按列类型还原。布尔只有 ``True``/``False``/空。
    """
    def b(key):
        v = (d.get(key) or "").strip()
        return v == "True"

    def i(key):
        v = (d.get(key) or "").strip()
        return int(v) if v.lstrip("-").isdigit() else 0

    def f(key):
        v = (d.get(key) or "").strip()
        try:
            return float(v)
        except ValueError:
            return None

    def opt_b(key):
        v = (d.get(key) or "").strip()
        return None if v == "" else (v == "True")

    def opt_i(key):
        v = (d.get(key) or "").strip()
        return int(v) if v.lstrip("-").isdigit() else None

    center = None
    raw = (d.get("center") or "").strip()
    if raw:
        parts = raw.split(";")
        if len(parts) == 3:
            try:
                center = (float(parts[0]), float(parts[1]), float(parts[2]))
            except ValueError:
                center = None

    row = ToothGroundTruth(
        case_uuid=d.get("case_uuid") or "",
        source_case_id=d.get("source_case_id") or "",
        source_order_id=d.get("source_order_id") or None,
        fdi=i("fdi"),
        reconstruction_type=d.get("reconstruction_type") or "Unknown",
        has_construction_info=b("has_construction_info"),
        has_parameters=b("has_parameters"),
        parameter_count=i("parameter_count"),
        has_margin=b("has_margin"),
        margin_valid=b("margin_valid"),
        margin_point_count=i("margin_point_count"),
        margin_perimeter_mm=f("margin_perimeter_mm"),
        margin_max_segment_mm=f("margin_max_segment_mm"),
        has_axis=b("has_axis"),
        axis_valid=b("axis_valid"),
        has_center=b("has_center"),
        center=center,
        has_tooth_frame=b("has_tooth_frame"),
        has_tooth_library=b("has_tooth_library"),
        tooth_library_name=d.get("tooth_library_name") or None,
        tooth_model_matrix_valid=b("tooth_model_matrix_valid"),
        has_modeldie=b("has_modeldie"),
        modeldie_asset_id=d.get("modeldie_asset_id") or None,
        modeldie_same_frame=b("modeldie_same_frame"),
        modeldie_center_distance_mm=f("modeldie_center_distance_mm"),
        nearest_asset_id=d.get("nearest_asset_id") or None,
        nearest_asset_center_distance_mm=f("nearest_asset_center_distance_mm"),
        asset_kind=d.get("asset_kind") or None,
        asset_selection_method=d.get("asset_selection_method") or None,
        has_full_arch_candidate=b("has_full_arch_candidate"),
        full_arch_asset_id=d.get("full_arch_asset_id") or None,
        full_arch_bbox_mm=d.get("full_arch_bbox_mm") or None,
        target_center_surface_distance_mm=f("target_center_surface_distance_mm"),
        target_center_surface_coverage=f("target_center_surface_coverage"),
        target_center_has_surface=opt_b("target_center_has_surface"),
        arch_is_sectioned_or_unsectioned=(
            d.get("arch_is_sectioned_or_unsectioned") or "unknown"),
        arch_asset_class="UNKNOWN",        # 由审计回填
        has_antagonist=b("has_antagonist"),
        antagonist_asset_id=d.get("antagonist_asset_id") or None,
        has_final_crown=b("has_final_crown"),
        final_crown_asset_id=d.get("final_crown_asset_id") or None,
        final_crown_fdi_count=i("final_crown_fdi_count"),
        final_crown_single_unit=b("final_crown_single_unit"),
        final_crown_watertight=opt_b("final_crown_watertight"),
        final_crown_euler=opt_i("final_crown_euler"),
        margin_crown_consistent=b("margin_crown_consistent"),
        margin_crown_p95_mm=f("margin_crown_p95_mm"),
        design_source_kind=d.get("design_source_kind") or "UNKNOWN",
        design_source_asset_id=d.get("design_source_asset_id") or None,
        coordinate_frame_kind=d.get("coordinate_frame_kind") or "UNKNOWN",
        geometry_validation_status=(
            d.get("geometry_validation_status") or "NOT_APPLICABLE"),
        notes=d.get("notes") or "",
    )
    return row


def load_asset_classes(audit_csv: Path,
                       ) -> dict[tuple[str, int], str]:
    """``(case_uuid, target_fdi) -> 最强弓资产类别``。"""
    best: dict[tuple[str, int], str] = {}
    with audit_csv.open(newline="", encoding="ascii") as f:
        for d in csv.DictReader(f):
            fdi = (d.get("target_fdi") or "").strip()
            if not fdi.lstrip("-").isdigit():
                continue
            key = (d.get("case_uuid") or "", int(fdi))
            cls = (d.get("candidate_class") or "UNKNOWN").strip()
            cur = best.get(key)
            if cur is None or _rank(cls) < _rank(cur):
                best[key] = cls
    return best


def freeze(index_dir: Path, audit_dir: Path, out_dir: Path,
           dataset_version: str) -> dict:
    index_csv = index_dir / "tooth_index.csv"
    audit_csv = audit_dir / "arch_asset_audit.csv"
    for p in (index_csv, audit_csv):
        if not p.is_file():
            raise FileNotFoundError(f"缺少输入：{p}")

    classes = load_asset_classes(audit_csv)

    rows: list[ToothGroundTruth] = []
    with index_csv.open(newline="", encoding="ascii") as f:
        for d in csv.DictReader(f):
            row = _to_row(d)
            row.arch_asset_class = classes.get(
                (row.case_uuid, row.fdi), "UNKNOWN")
            rows.append(row)

    sets = build_all_sets(rows)
    reasons = localization_exclusion_reasons(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_version": dataset_version,
        "generated_from": {
            "index_csv": str(index_csv),
            "audit_csv": str(audit_csv),
        },
        "n_findex_rows": len(rows),
        "n_cases": len({r.case_uuid for r in rows}),
        "n_case_groups": len({case_group_key(r) for r in rows}),
        # 集合大小：**本次实际数出来的**
        "set_counts": {k: len(v) for k, v in sets.items()},
        "set_case_counts": {
            k: len({case_group_key(m) for m in v}) for k, v in sets.items()},
        "localization_admitted_class": LOCALIZATION_ADMITTED_CLASS,
        # 定位这条路线的**状态**是冻结的一部分：它证明 LocalizationGTSet
        # 为什么这么小，也防止后人以为它只是"还没跑完"。
        "localization_route_status": LOCALIZATION_ROUTE_STATUS,
        "localization_route_evidence": LOCALIZATION_ROUTE_EVIDENCE,
        "localization_requires_human_prep": LOCALIZATION_REQUIRES_HUMAN_PREP,
        "human_label_summary": qa_round_summary(),
        "human_label_conflicts": labels_agree_across_rounds(),
        "human_prep_positive_rows": sum(
            1 for r in rows if human_prep_label(r) == HUMAN_PREP_POSITIVE),
        "qa2_confusion": QA2_CONFUSION,
        "qa2_strata": QA2_STRATA,
        "localization_exclusion_reasons": reasons,
        "arch_asset_class_counts": _counts(r.arch_asset_class for r in rows),
        "reconstruction_type_counts": _counts(r.reconstruction_type for r in rows),
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="ascii")
    (out_dir / "localization_exclusion_reasons.json").write_text(
        json.dumps(reasons, indent=2, ensure_ascii=True), encoding="ascii")

    with (out_dir / "gt_set_members.csv").open("w", newline="",
                                               encoding="ascii") as f:
        w = csv.writer(f)
        w.writerow(["set_name", "case_uuid", "source_order_id", "fdi",
                    "reconstruction_type", "arch_asset_class",
                    "human_prep_label"])
        for name, members in sets.items():
            for m in members:
                w.writerow([name, m.case_uuid, m.source_order_id, m.fdi,
                            m.reconstruction_type, m.arch_asset_class,
                            human_prep_label(m) or ""])

    # 冻结后的行也写出来，方便后续直接按 arch_asset_class 复查
    with (out_dir / "frozen_tooth_index.csv").open("w", newline="",
                                                   encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow(r.as_row())
    return manifest


def _counts(iterable) -> dict:
    out: dict = {}
    for x in iterable:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WelooAI GT Set Freeze（提交 2.2）")
    ap.add_argument("--index", required=True, help="GroundTruthIndexer 的输出目录")
    ap.add_argument("--audit", required=True, help="Arch Asset Audit 的输出目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-version", default="exocad_gt_v1")
    a = ap.parse_args(argv)

    man = freeze(Path(a.index), Path(a.audit), Path(a.out), a.dataset_version)
    print(json.dumps(man, indent=2, ensure_ascii=True))
    print(f"\n写出到 {a.out}/dataset_manifest.json")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
