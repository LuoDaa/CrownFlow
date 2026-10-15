"""Parameter Resolver。

固定的解析优先级（不得改动顺序）：

    Tooth Explicit
        -> Case / Order Explicit
            -> Restoration-type Preset
                -> Lab Preset
                    -> System Default

两条硬性规则：

1. **``MISSING_INTERNAL_DEFAULT_UNKNOWN`` 只能"继续找下一层"，绝不能"跌落成 0"。**
   某一层没有这个参数表示"该层未设定"，继续向下是正确的；最终仍找不到时返回
   ``MISSING_INTERNAL_DEFAULT_UNKNOWN`` 且 ``value is None``。
2. **某一层已经写了值（含 ``EXPLICIT_BUT_UNRESOLVED`` / ``NO_CONSTRAINT``）就不继续向下找。**
   exocad 写了 ``CementGapThickness = 0`` 时，用 lab preset 的 0.025 去顶替它，
   等于把"待确认"静默改成"已确认"，这正是要避免的。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from ..importers.exocad.models import (
    ExocadToothRecord,
    ParameterScope,
    ParameterState,
)
from .derived import DERIVED_RULES, DERIVED_TARGETS
from .models import (
    LabProfile,
    ParameterClass,
    ParameterResolutionError,
    ResolvedParameter,
    ResolvedParameterSet,
    ResolvedSource,
)

#: 这些参数即使允许 system default，也不允许由系统默认值补齐 ——
#: 它们的 0 语义未定，必须由人工或真实 profile 提供。
SYSTEM_DEFAULT_FORBIDDEN: frozenset[str] = frozenset({
    "CementGapThickness",
    "CementGapTopThickness",
    "MinThickness",
    "MinThicknessNearPreparationMargin",
})

#: 系统级默认值。**故意为空。**
#:
#: 本项目没有可引用的临床/法规来源来定义系统级默认值，而"猜一个数"比留空更危险：
#: 留空会让解析结果为 MISSING（可见、可断言），猜数会让它看起来像真值。
#: 需要系统默认值时，应由实验室提供并写入 ``lab_profiles/*.json``。
SYSTEM_DEFAULTS: dict[str, float] = {}


class ParameterResolver:
    """按固定优先级把各层参数合成一次设计所需的参数集。"""

    def __init__(
        self,
        profile: LabProfile,
        *,
        system_defaults: Mapping[str, float] | None = None,
        lab_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.system_defaults = dict(system_defaults if system_defaults is not None
                                   else SYSTEM_DEFAULTS)
        self.lab_id = lab_id or profile.lab_id

    # ---------------------------------------------------------------- public

    def resolve(
        self,
        *,
        tooth: ExocadToothRecord | None = None,
        restoration_type: str | None = None,
        material: str | None = None,
        order_id: str | None = None,
        case_overrides: Mapping[str, float] | None = None,
        order_overrides: Mapping[str, float] | None = None,
        names: Iterable[str] | None = None,
        allow_system_defaults: bool = False,
    ) -> ResolvedParameterSet:
        """合成参数集。

        ``allow_system_defaults`` 默认为 ``False``：在这个阶段，缺失参数应当保持为缺失，
        而不是被一个未经验证的系统默认值填上。
        """
        case_overrides = dict(case_overrides or {})
        order_overrides = dict(order_overrides or {})
        restoration_type = restoration_type or (
            tooth.reconstruction_type.value if tooth is not None else None)

        if names is None:
            names = self._known_names(tooth, case_overrides, order_overrides)

        resolved: dict[str, ResolvedParameter] = {}
        for name in sorted(set(names)):
            resolved[name] = self._resolve_one(
                name,
                tooth=tooth,
                restoration_type=restoration_type,
                case_overrides=case_overrides,
                order_overrides=order_overrides,
                allow_system_defaults=allow_system_defaults,
            )

        # 派生遍：把「由其它参数确定性决定」的参数补齐。
        # 必须在逐名解析之后，因为派生依赖的目标可能自己也是从 preset 解析出来的。
        # 只补齐**未解析**的项；病例显式写过的值优先，绝不被派生值覆盖。
        self._apply_derived_rules(resolved)

        return ResolvedParameterSet(
            parameters=resolved,
            context={
                "lab_id": self.lab_id,
                "order_id": order_id,
                "fdi": tooth.tooth_fdi if tooth is not None else None,
                "restoration_type": restoration_type,
                "material": material,
                "profile_basis": {
                    "n_orders": self.profile.n_orders_basis,
                    "n_tooth_records": self.profile.n_tooth_records_basis,
                },
                "allow_system_defaults": allow_system_defaults,
            },
        )

    # --------------------------------------------------------------- private

    def _apply_derived_rules(self, resolved: dict[str, ResolvedParameter]) -> None:
        """按 :data:`DERIVED_RULES` 补齐未解析的派生参数。

        规则未覆盖依赖取值时不强行套用 —— 保持 MISSING 并说明原因，
        而不是用最近似的表项敷衍（那会把"规则不适用"伪装成"规则已应用"）。
        """
        if not (DERIVED_TARGETS & set(resolved)):
            return

        for rule in DERIVED_RULES:
            target = resolved.get(rule.target)
            if target is None:
                continue
            if target.state is not ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN:
                continue

            dependency = resolved.get(rule.depends_on)
            if dependency is None:
                continue
            dependency_value = dependency.value if dependency.is_resolved else None
            derived_value = rule.derive(dependency_value)

            if derived_value is None:
                resolved[rule.target] = replace(
                    target,
                    provenance={**target.provenance,
                                "derived_rule_not_applicable": {
                                    "depends_on": rule.depends_on,
                                    "dependency_value": dependency_value,
                                    "known_values": sorted(rule.depends_on_known_values),
                                }},
                    unresolved_reason=(
                        f"{rule.target} 未写出，且派生规则不适用："
                        f"{rule.depends_on}={dependency_value} 不在规则的已知取值 "
                        f"{sorted(rule.depends_on_known_values)} 内。"
                    ),
                )
                continue

            resolved[rule.target] = ResolvedParameter(
                name=rule.target,
                state=ParameterState.EXPLICIT,
                value=float(derived_value),
                source=ResolvedSource.DERIVED_FROM_PARAMETER,
                scope=target.scope,
                provenance={
                    "derived_from": rule.depends_on,
                    "dependency_value": dependency_value,
                    "rule_evidence": rule.evidence,
                },
            )

    def _known_names(
        self,
        tooth: ExocadToothRecord | None,
        case_overrides: Mapping[str, float],
        order_overrides: Mapping[str, float],
    ) -> set[str]:
        names = set(self.profile.parameters)
        if tooth is not None:
            names |= set(tooth.parameters)
        names |= set(case_overrides) | set(order_overrides)
        return names

    def _resolve_one(
        self,
        name: str,
        *,
        tooth: ExocadToothRecord | None,
        restoration_type: str | None,
        case_overrides: Mapping[str, float],
        order_overrides: Mapping[str, float],
        allow_system_defaults: bool,
    ) -> ResolvedParameter:
        scope = self._scope_of(name)

        # ---- 1. Tooth Explicit ------------------------------------------------
        if tooth is not None:
            cell = tooth.parameter(name)
            if cell.state is not ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN:
                # 这一层写了东西就定下来，不再向下找 —— 即使它是未定/无约束。
                return ResolvedParameter(
                    name=name,
                    state=cell.state,
                    value=cell.value,
                    source=ResolvedSource.TOOTH_EXPLICIT,
                    scope=cell.scope,
                    provenance=dict(cell.provenance),
                    unresolved_reason=cell.unresolved_reason,
                )

        # ---- 2. Case / Order Explicit ----------------------------------------
        for overrides, src in ((order_overrides, ResolvedSource.CASE_ORDER_EXPLICIT),
                               (case_overrides, ResolvedSource.CASE_ORDER_EXPLICIT)):
            if name in overrides:
                return ResolvedParameter(
                    name=name, state=ParameterState.EXPLICIT, value=float(overrides[name]),
                    source=src, scope=scope,
                    provenance={"origin": "case_or_order_override", "order_id": None},
                )

        # ---- 3. Restoration-type Preset --------------------------------------
        cls = self.profile.get(name)
        if cls is not None and cls.parameter_class is ParameterClass.RESTORATION_PRESET:
            value = cls.preset_value(restoration_type)
            if value is not None:
                return ResolvedParameter(
                    name=name, state=ParameterState.EXPLICIT, value=value,
                    source=ResolvedSource.RESTORATION_PRESET, scope=scope,
                    provenance={
                        "lab_id": self.lab_id,
                        "restoration_type": restoration_type,
                        "basis_records": cls.n_records_with_param,
                    },
                )

        # ---- 4. Lab Preset ---------------------------------------------------
        if cls is not None and cls.parameter_class is ParameterClass.LAB_GLOBAL_PRESET:
            value = cls.preset_value()
            if value is not None:
                return ResolvedParameter(
                    name=name, state=ParameterState.EXPLICIT, value=value,
                    source=ResolvedSource.LAB_PRESET, scope=scope,
                    provenance={"lab_id": self.lab_id,
                                "basis_records": cls.n_records_with_param},
                )

        # ---- 5. System Default ----------------------------------------------
        if allow_system_defaults and name in self.system_defaults:
            if name in SYSTEM_DEFAULT_FORBIDDEN:
                return self._unresolved(
                    name, scope,
                    reason=(f"{name} 在 SYSTEM_DEFAULT_FORBIDDEN 中：其 0/近 0 取值语义未定，"
                            "不允许由系统默认值补齐。"),
                )
            return ResolvedParameter(
                name=name, state=ParameterState.EXPLICIT, value=float(self.system_defaults[name]),
                source=ResolvedSource.SYSTEM_DEFAULT, scope=scope,
                provenance={"origin": "system_default", "caveat": "无临床来源，仅为占位"},
            )

        # ---- 都没有 ----------------------------------------------------------
        return self._unresolved(name, scope, reason=None)

    def _unresolved(self, name: str, scope: ParameterScope,
                    reason: str | None) -> ResolvedParameter:
        cls = self.profile.get(name)
        base = (
            "该参数在所有层级都没有可用取值：牙位未写出（exocad 内部默认值未知）、"
            "病例/订单无覆盖，且 Lab Profile 未提供默认值"
        )
        if cls is not None and cls.parameter_class is ParameterClass.CASE_VARIABLE:
            base += f"（该参数被分类为 CASE_VARIABLE，{cls.n_records_with_param} 条记录有 "
            base += f"{len(cls.distinct_values)} 种取值，本来就不应有默认值）"
        return ResolvedParameter(
            name=name,
            state=ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN,
            value=None,
            source=ResolvedSource.UNRESOLVED,
            scope=scope,
            provenance={"lab_id": self.lab_id},
            unresolved_reason=reason or base,
        )

    def _scope_of(self, name: str) -> ParameterScope:
        cls = self.profile.get(name)
        if cls is not None:
            return cls.scope
        return ParameterScope.TOOTH


def resolve_for_tooth(
    profile: LabProfile,
    tooth: ExocadToothRecord,
    **kwargs: Any,
) -> ResolvedParameterSet:
    """便捷入口：直接用一条 ``ExocadToothRecord`` 解析。"""
    resolver = ParameterResolver(profile)
    return resolver.resolve(tooth=tooth, **kwargs)


__all__ = [
    "ParameterResolver",
    "resolve_for_tooth",
    "SYSTEM_DEFAULTS",
    "SYSTEM_DEFAULT_FORBIDDEN",
    "ParameterResolutionError",
]
