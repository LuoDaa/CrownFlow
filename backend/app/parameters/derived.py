"""由其它参数派生的参数。

实测发现（`数据/15个完整病例`，25 条同时写出这两个参数的记录）：
``AdditionalBlockOutAngleAfterCementGap`` 与 ``UntouchableZoneNearPreparationMargin``
**不是独立的实验室 preset，而是机轴数的确定性函数**：

    MillingMachineAxes = 3  ->  AdditionalBlockOutAngleAfterCementGap =  0   (10/10)
                                UntouchableZoneNearPreparationMargin =  0   (10/10)
    MillingMachineAxes = 5  ->  AdditionalBlockOutAngleAfterCementGap = -1   (15/15)
                                UntouchableZoneNearPreparationMargin = 0.1  (15/15)

相关度 100%，无例外。

**这纠正了本项目之前的一个误判。** 参数分类器把这两个参数标成了
``LAB_PRESET_VARIANT``（"两种工艺风格并存"），因为它们的取值分布是双峰的
（0 与 0.1；-1 与 0）。但它们并不是"同一参数的两个风格值"，而是**另一个参数的
函数** —— 把它们当独立 preset 查表，等于把一个功能依赖压平成经验值。

工程后果：解析链应在 ``MillingMachineAxes`` 确定之后**推导**这两个参数，
而不是各自独立查表；并且当某病例显式写了这两个参数时以显式值为准。

``DERIVED_RULES`` 里每条规则都带证据规模；验收测试会用真实数据核验规则本身，
数据一旦不再支持（出现例外）测试会失败 —— 而不是让一个过时的经验值悄悄留在代码里。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DerivedRule:
    """``target`` 由 ``depends_on`` 通过 ``table`` 确定性派生。"""

    target: str
    depends_on: str
    table: dict[float, float]
    evidence: str
    #: 该规则成立时依赖参数的合法取值集合（用于判断规则是否适用）。
    depends_on_known_values: frozenset[float] = field(default_factory=frozenset)

    def derive(self, dependency_value: float | None) -> float | None:
        if dependency_value is None:
            return None
        return self.table.get(float(dependency_value))


DERIVED_RULES: tuple[DerivedRule, ...] = (
    DerivedRule(
        target="AdditionalBlockOutAngleAfterCementGap",
        depends_on="MillingMachineAxes",
        table={3.0: 0.0, 5.0: -1.0},
        evidence=(
            "实测 25 条记录：3 轴 10/10 取 0，5 轴 15/15 取 -1，相关度 100%。"
            "与 exocad 官方对 3 轴/5 轴材料区分该参数的说法一致。"
        ),
        depends_on_known_values=frozenset({3.0, 5.0}),
    ),
    DerivedRule(
        target="UntouchableZoneNearPreparationMargin",
        depends_on="MillingMachineAxes",
        table={3.0: 0.0, 5.0: 0.1},
        evidence=(
            "实测 25 条记录：3 轴 10/10 取 0，5 轴 15/15 取 0.1，相关度 100%。"
            "原先被分类器标为 LAB_PRESET_VARIANT（双峰），实为机轴数的函数。"
        ),
        depends_on_known_values=frozenset({3.0, 5.0}),
    ),
)

DERIVED_TARGETS: frozenset[str] = frozenset(r.target for r in DERIVED_RULES)


def rules_for(target: str) -> list[DerivedRule]:
    return [r for r in DERIVED_RULES if r.target == target]
