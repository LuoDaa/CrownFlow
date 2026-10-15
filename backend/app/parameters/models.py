"""参数分类与 Lab Profile 的数据模型。

分类标签的来源逻辑见 classify.py。这里只定义数据结构与"能否用作默认值"的判定。

关键设计约束：**分类结果是实验室级别的，不是全局真理。**
``LAB_GLOBAL_PRESET`` 只说明"在导入的这批记录里取值一致"，因此它可以成为该 Lab
Profile 的默认值，但**不得硬编码进算法核心** —— 换一个实验室数据可能不同。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator

from ..importers.exocad.models import ParameterScope, ParameterState


class ParameterClass(str, Enum):
    """一个参数在证据数据里表现出的稳定性等级。"""

    #: 所有已写出的记录取值完全一致。可作**该实验室**的默认值。
    LAB_GLOBAL_PRESET = "LAB_GLOBAL_PRESET"

    #: 在每个修复体类型内部取值一致，但类型之间不同（如 MinThickness：冠 0.6 / 减量桥体 0.4）。
    RESTORATION_PRESET = "RESTORATION_PRESET"

    #: 取值种类很少（≤3）但并非按修复体类型分组 —— 表现为两种工艺风格并存
    #: （如 UntouchableZoneNearPreparationMargin：0 与 0.1 分布为 10/15）。
    #: 不得压平成单一值，须保留为可选项。
    LAB_PRESET_VARIANT = "LAB_PRESET_VARIANT"

    #: 取值种类多（≥4），是真正的逐病例变量。**绝不可作为默认值。**
    CASE_VARIABLE = "CASE_VARIABLE"

    #: 证据不足（样本 < 2 条），无法判定。
    UNKNOWN = "UNKNOWN"

    @property
    def may_supply_default(self) -> bool:
        """该类别是否允许为解析链提供一个默认值。"""
        return self in (ParameterClass.LAB_GLOBAL_PRESET, ParameterClass.RESTORATION_PRESET)


#: 判定"取值种类多"的阈值。≥4 种取值意味着它不可能是一个 preset 菜单。
CASE_VARIABLE_MIN_DISTINCT = 4
#: 判定 LAB_GLOBAL_PRESET 所需的最少记录数。
MIN_RECORDS_FOR_PRESET = 3


@dataclass
class ParameterClassification:
    """单个参数在证据数据上的分类结果。"""

    name: str
    parameter_class: ParameterClass
    scope: ParameterScope
    n_records_with_param: int
    #: 取值 → 出现次数（全样本）。
    value_counts: dict[float, int] = field(default_factory=dict)
    #: 修复体类型 → {取值: 次数}。
    by_restoration_type: dict[str, dict[float, int]] = field(default_factory=dict)
    #: 该类别的判定理由，供人工复核。
    rationale: str = ""

    @property
    def distinct_values(self) -> list[float]:
        return sorted(self.value_counts)

    @property
    def modal_value(self) -> float | None:
        if not self.value_counts:
            return None
        return max(self.value_counts.items(), key=lambda kv: kv[1])[0]

    def preset_value(self, restoration_type: str | None = None) -> float | None:
        """该参数能为 ``restoration_type`` 提供的默认值。不适用时返回 ``None``。"""
        if self.parameter_class is ParameterClass.LAB_GLOBAL_PRESET:
            return self.modal_value
        if self.parameter_class is ParameterClass.RESTORATION_PRESET and restoration_type:
            bucket = self.by_restoration_type.get(restoration_type)
            if bucket:
                return max(bucket.items(), key=lambda kv: kv[1])[0]
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.parameter_class.value,
            "scope": self.scope.value,
            "n_records_with_param": self.n_records_with_param,
            "value_counts": {str(k): v for k, v in sorted(self.value_counts.items())},
            "by_restoration_type": {
                rt: {str(k): v for k, v in sorted(vals.items())}
                for rt, vals in sorted(self.by_restoration_type.items())
            },
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterClassification":
        return cls(
            name=data["name"],
            parameter_class=ParameterClass(data["class"]),
            scope=ParameterScope(data["scope"]),
            n_records_with_param=data["n_records_with_param"],
            value_counts={float(k): v for k, v in data["value_counts"].items()},
            by_restoration_type={
                rt: {float(k): v for k, v in vals.items()}
                for rt, vals in data.get("by_restoration_type", {}).items()
            },
            rationale=data.get("rationale", ""),
        )


@dataclass
class LabProfile:
    """一个实验室的参数档案。由真实 exocad 证据数据导出，不含任何 PHI。"""

    lab_id: str
    #: 证据基础：用于统计的独立订单数 / 牙位记录数。用于让使用者知道置信度。
    n_orders_basis: int
    n_tooth_records_basis: int
    parameters: dict[str, ParameterClassification] = field(default_factory=dict)
    #: 导入时的来源说明（不含路径与姓名）。
    notes: str = ""

    def get(self, name: str) -> ParameterClassification | None:
        return self.parameters.get(name)

    def preset_value(self, name: str, restoration_type: str | None = None) -> float | None:
        cls = self.parameters.get(name)
        if cls is None:
            return None
        return cls.preset_value(restoration_type)

    def of_class(self, parameter_class: ParameterClass) -> list[ParameterClassification]:
        return [c for c in self.parameters.values() if c.parameter_class is parameter_class]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lab_id": self.lab_id,
            "n_orders_basis": self.n_orders_basis,
            "n_tooth_records_basis": self.n_tooth_records_basis,
            "notes": self.notes,
            "parameters": {n: c.as_dict() for n, c in sorted(self.parameters.items())},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabProfile":
        return cls(
            lab_id=data["lab_id"],
            n_orders_basis=data.get("n_orders_basis", 0),
            n_tooth_records_basis=data.get("n_tooth_records_basis", 0),
            notes=data.get("notes", ""),
            parameters={n: ParameterClassification.from_dict(d)
                        for n, d in data["parameters"].items()},
        )


class ResolvedSource(str, Enum):
    """解析链上真正提供了该值的层级。"""

    TOOTH_EXPLICIT = "tooth_explicit"
    CASE_ORDER_EXPLICIT = "case_order_explicit"
    RESTORATION_PRESET = "restoration_preset"
    LAB_PRESET = "lab_preset"
    SYSTEM_DEFAULT = "system_default"
    #: 由另一个已解析参数按确定性规则派生（见 parameters/derived.py）。
    #: 与 preset 的区别：preset 是"经验上常用这个值"，派生是"功能依赖"。
    DERIVED_FROM_PARAMETER = "derived_from_parameter"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ResolvedParameter:
    """解析后的单个参数。数值访问必须走 ``.value`` 并同时检查 ``.state``。"""

    name: str
    state: ParameterState
    value: float | None
    source: ResolvedSource
    scope: ParameterScope
    provenance: dict[str, Any] = field(default_factory=dict)
    unresolved_reason: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.state is ParameterState.EXPLICIT and self.value is not None

    @property
    def blocks_auto_design(self) -> bool:
        """该参数是否必须阻止自动设计继续。

        ``EXPLICIT_BUT_UNRESOLVED``（如 CementGapThickness = 0）与未知值
        都必须阻止 —— 前者是语义未定，后者是根本没有值。
        """
        return self.state is not ParameterState.EXPLICIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "value": self.value,
            "source": self.source.value,
            "scope": self.scope.value,
            "provenance": dict(self.provenance),
            "unresolved_reason": self.unresolved_reason,
        }


class ParameterResolutionError(RuntimeError):
    """在自动设计路径上请求了一个未解析或被阻塞的参数。"""


@dataclass
class ResolvedParameterSet:
    """一次 ``resolve()`` 的完整结果。

    支持 ``resolved["MinThickness"].value`` 与 ``resolved.MinThickness.value`` 两种写法，
    与用户在验收套件里的断言写法一致。
    """

    parameters: dict[str, ResolvedParameter] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> ResolvedParameter:
        try:
            return self.parameters[name]
        except KeyError as exc:
            raise KeyError(
                f"参数 {name!r} 未被解析。解析链只覆盖被显式请求或由 profile 提供的参数；"
                "请把它加入请求列表，不要假定它存在。"
            ) from exc

    def __getattr__(self, name: str) -> ResolvedParameter:
        if name.startswith("_") or name in ("parameters", "context"):
            raise AttributeError(name)
        params = object.__getattribute__(self, "parameters")
        if name in params:
            return params[name]
        raise AttributeError(f"参数 {name!r} 未被解析")

    def __contains__(self, name: str) -> bool:
        return name in self.parameters

    def __iter__(self) -> Iterator[str]:
        return iter(self.parameters)

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def blockers(self) -> list[ResolvedParameter]:
        """会阻止自动设计的参数（未知或语义未定）。"""
        return [p for p in self.parameters.values() if p.blocks_auto_design]

    def require(self, name: str) -> float:
        """在自动设计路径上取值。未解析或被阻塞时抛异常，**绝不返回 0**。"""
        p = self[name]
        if p.blocks_auto_design:
            raise ParameterResolutionError(
                f"参数 {name} 处于 {p.state.value}，不能参与自动设计。"
                + (f" 原因: {p.unresolved_reason}" if p.unresolved_reason else "")
            )
        assert p.value is not None  # EXPLICIT 恒有值
        return p.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "context": dict(self.context),
            "parameters": {n: p.as_dict() for n, p in sorted(self.parameters.items())},
        }
