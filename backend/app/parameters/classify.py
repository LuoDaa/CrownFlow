"""从真实 exocad 证据数据推导参数分类。

分类规则（必须由数据推导，不得硬编码参数名 → 类别）：

1. 所有已写出记录的取值完全一致                  -> ``LAB_GLOBAL_PRESET``
2. 每个修复体类型内部一致、类型之间不同（≥2 类型）  -> ``RESTORATION_PRESET``
3. 取值种类 ≤ 3 但不满足 2                      -> ``LAB_PRESET_VARIANT``
4. 取值种类 ≥ 4                                -> ``CASE_VARIABLE``
5. 样本 < 3 条                                  -> ``UNKNOWN``

规则 3 与 4 的阈值依据真实数据：``CementGapThickness`` 与 ``MillingDiameter``
各有 5 种取值（真变量），而 ``UntouchableZoneNearPreparationMargin``、
``AdditionalBlockOutAngleAfterCementGap``、``CrownAdditionalSpacing*``、
``MillingMachineAxes`` 只有 2–3 种（工艺菜单）。
"""

from __future__ import annotations

import collections
from collections.abc import Iterable

from ..importers.exocad.models import (
    ExocadToothRecord,
    ParameterScope,
    ParameterState,
)
from .derived import DERIVED_TARGETS, rules_for
from .models import (
    CASE_VARIABLE_MIN_DISTINCT,
    MIN_RECORDS_FOR_PRESET,
    LabProfile,
    ParameterClass,
    ParameterClassification,
)


def classify_parameters(
    records: Iterable[ExocadToothRecord],
) -> dict[str, ParameterClassification]:
    """按参数名汇总全部牙位记录并给出分类。"""
    records = list(records)

    value_counts: dict[str, collections.Counter[float]] = collections.defaultdict(collections.Counter)
    by_type: dict[str, collections.defaultdict[str, collections.Counter[float]]] = \
        collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    scopes: dict[str, ParameterScope] = {}

    for rec in records:
        rt = rec.reconstruction_type.value
        for name, cell in rec.parameters.items():
            # 只统计 EXPLICIT：NaN（无约束）与语义未定的 0 都不构成"这个值被用过"的证据。
            if cell.state is not ParameterState.EXPLICIT or cell.value is None:
                continue
            value_counts[name][cell.value] += 1
            by_type[name][rt][cell.value] += 1
            scopes.setdefault(name, cell.scope)

    out: dict[str, ParameterClassification] = {}
    for name, counter in value_counts.items():
        n = sum(counter.values())
        distinct = len(counter)
        per_type = {rt: dict(c) for rt, c in by_type[name].items()}

        if n < MIN_RECORDS_FOR_PRESET:
            cls, rationale = ParameterClass.UNKNOWN, f"仅 {n} 条记录，证据不足"
        elif distinct == 1:
            cls = ParameterClass.LAB_GLOBAL_PRESET
            rationale = (
                f"{n} 条记录取值全部为 {counter.most_common(1)[0][0]}。"
                "含义是该**实验室**在这批记录上取值一致，可作 Lab Profile 默认值，"
                "不是全局临床真理，不得硬编码进算法核心。"
            )
        elif all(len(c) == 1 for c in per_type.values()) and len(per_type) >= 2:
            cls = ParameterClass.RESTORATION_PRESET
            detail = "; ".join(f"{rt}={next(iter(c))}" for rt, c in sorted(per_type.items()))
            rationale = f"每个修复体类型内部一致、类型之间不同: {detail}"
        elif distinct <= CASE_VARIABLE_MIN_DISTINCT - 1:
            cls = ParameterClass.LAB_PRESET_VARIANT
            rationale = (
                f"{distinct} 种取值（{dict(sorted(counter.items()))}），"
                "且并非按修复体类型分组 → 判定为并存的多套工艺风格，"
                "必须保留为可选项，不得压平成单一默认值。"
            )
            if name in DERIVED_TARGETS:
                # 双峰不等于两套风格：有些双峰其实是另一个参数的函数。
                # 实测 AdditionalBlockOutAngleAfterCementGap 与
                # UntouchableZoneNearPreparationMargin 的双峰完全由
                # MillingMachineAxes（3/5）决定，相关度 100%。
                rule = rules_for(name)[0]
                rationale = (
                    f"{distinct} 种取值（{dict(sorted(counter.items()))}）看似双峰，"
                    f"但已确认是 {rule.depends_on} 的确定性函数，不是并列的风格取值。"
                    f"解析链应派生而非查表。依据：{rule.evidence}"
                )
        else:
            cls = ParameterClass.CASE_VARIABLE
            rationale = (
                f"{distinct} 种取值（{dict(sorted(counter.items()))}），"
                "是真正的逐病例变量，绝不可作为默认值。"
            )

        out[name] = ParameterClassification(
            name=name,
            parameter_class=cls,
            scope=scopes.get(name, ParameterScope.TOOTH),
            n_records_with_param=n,
            value_counts=dict(sorted(counter.items())),
            by_restoration_type=per_type,
            rationale=rationale,
        )

    return out


def build_lab_profile(
    records: Iterable[ExocadToothRecord],
    *,
    lab_id: str,
    n_orders_basis: int,
    notes: str = "",
) -> LabProfile:
    records = list(records)
    return LabProfile(
        lab_id=lab_id,
        n_orders_basis=n_orders_basis,
        n_tooth_records_basis=len(records),
        parameters=classify_parameters(records),
        notes=notes,
    )
