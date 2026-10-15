"""从真实 exocad 证据数据生成 Lab Profile。

用法：

    python -m app.parameters.build_lab_profile \
        --raw "D:/code/牙冠自动生成/数据/15个完整病例" \
        --lab-id inhouse_76007

生成物写入 ``app/parameters/lab_profiles/<lab-id>.json``，只含参数名/数值/计数，
不含任何 PHI（写出前强制校验），可提交进 Git。

**分组方式**：按 ``source_order_id``，不按目录。实测同一订单出现在两个目录，
按目录统计会重复计数并造成训练/验证泄露。
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from ..importers.exocad.case_loader import discover_cases, load_case
from ..importers.exocad.models import ExocadToothRecord
from .classify import build_lab_profile
from .models import ParameterClass
from .profiles import save_lab_profile


def collect_records(raw_root: Path) -> tuple[list[ExocadToothRecord], int, Counter]:
    """按订单分组收集牙位记录。

    返回 ``(records, n_orders, skipped)``。同一订单的多个目录只取信息最全的那个作为
    代表，避免同一份设计被重复计数。
    """
    groups = discover_cases(raw_root)
    records: list[ExocadToothRecord] = []
    skipped: Counter = Counter()

    for order_id, dirs in sorted(groups.items()):
        best: list[ExocadToothRecord] = []
        for d in dirs:
            case = load_case(d)
            if len(case.teeth) > len(best):
                best = case.teeth
        if not best:
            skipped["no_construction_info"] += 1
            continue
        if len(dirs) > 1:
            skipped["duplicate_dirs_collapsed"] += 1
        records.extend(best)

    n_orders = len([g for g in groups.values()])
    return records, n_orders, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="从 exocad 病例数据生成 Lab Profile")
    ap.add_argument("--raw", required=True, help="病例根目录")
    ap.add_argument("--lab-id", required=True, help="实验室标识，用于 profile 文件名")
    ap.add_argument("--out", default=None, help="输出路径，默认写入 lab_profiles/<lab-id>.json")
    ap.add_argument("--notes", default="", help="备注，写入 profile（禁止包含 PHI）")
    a = ap.parse_args()

    raw = Path(a.raw)
    records, n_orders, skipped = collect_records(raw)
    if not records:
        raise SystemExit(f"在 {raw} 下没有找到任何 .constructionInfo 牙位记录")

    profile = build_lab_profile(records, lab_id=a.lab_id,
                               n_orders_basis=n_orders, notes=a.notes)
    path = save_lab_profile(profile, a.out)

    print(f"Lab Profile 已写出: {path}")
    print(f"  依据: {n_orders} 个订单组 / {len(records)} 条牙位记录")
    if skipped:
        print(f"  跳过: {dict(skipped)}")
    print()
    for cls in (ParameterClass.LAB_GLOBAL_PRESET, ParameterClass.RESTORATION_PRESET,
                ParameterClass.LAB_PRESET_VARIANT, ParameterClass.CASE_VARIABLE,
                ParameterClass.UNKNOWN):
        items = profile.of_class(cls)
        print(f"  {cls.value:24s} {len(items):3d} 个")
    print()
    print("  重点参数:")
    for name in ("MinThickness", "CementGapThickness", "MillingDiameter",
                 "MillingMachineAxes", "UntouchableZoneNearPreparationMargin",
                 "CrownBottomInsertionAngleBetweenTeethLimit",
                 "CrownAdaptationShapePreservation",
                 "CrownAdaptationMarginForShapePreservation"):
        c = profile.get(name)
        if c is None:
            print(f"    {name:44s} 未出现")
            continue
        print(f"    {name:44s} {c.parameter_class.value:22s} "
              f"n={c.n_records_with_param:2d} {c.value_counts}")


if __name__ == "__main__":
    main()
