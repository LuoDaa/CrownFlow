"""制造约束：加工档案与刀具可达性。

实测依据（`数据/15个完整病例`，31 条带加工参数的记录）：

- ``MillingMachineAxes`` 取 **3（12 条）或 5（19 条）** —— 必须分开处理。
- ``MillingDiameter`` 在 **0.5–1.2 mm** 之间取 5 种值（2.4 倍跨度），是**订单/病例级**
  变量，``scope = ORDER``。**不属于材料 profile** —— 把它写进 ``materials/zirconia.json``
  是错误的键位。
- ``MillingDiameterFlatToolRatio = 0.8`` 是全局 preset。

因此 ``materials/*.json`` 只承载材料学参数（间隙、壁厚、接触/咬合目标），
刀具与机台走 :class:`ManufacturingProfile`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..importers.exocad.models import ParameterScope


class MachineAxes(int, Enum):
    """加工机轴数。决定可达性测试是单向还是多向搜索。"""

    THREE = 3
    FIVE = 5


@dataclass(frozen=True)
class ManufacturingProfile:
    """一个订单/病例的加工能力。由 Parameter Resolver 解析后构造。"""

    machine_axes: MachineAxes
    #: 刀具直径（mm）。**注意 exocad 写的是 MillingDiameter，语义为直径**，
    #: 构造几何判定时必须换算成半径。
    milling_diameter_mm: float
    #: 平底刀比例。1.0 = 纯平底，<1 表示带圆角。
    flat_tool_ratio: float = 0.8
    machine_profile: str | None = None
    source: str = "resolved"

    @property
    def tool_radius_mm(self) -> float:
        return self.milling_diameter_mm / 2.0

    @property
    def is_three_axis(self) -> bool:
        return self.machine_axes is MachineAxes.THREE

    def as_dict(self) -> dict:
        return {
            "machine_axes": int(self.machine_axes),
            "milling_diameter_mm": self.milling_diameter_mm,
            "tool_radius_mm": self.tool_radius_mm,
            "flat_tool_ratio": self.flat_tool_ratio,
            "machine_profile": self.machine_profile,
            "source": self.source,
            "parameter_scope": ParameterScope.ORDER.value,
        }
