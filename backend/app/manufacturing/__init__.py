"""制造约束：加工档案（3 轴 / 5 轴分离）+ 刀具可达性构件。"""

from .models import MachineAxes, ManufacturingProfile
from .reachability import (
    ReachabilityResult,
    ReachabilityStatus,
    candidate_directions,
    evaluate_reachability,
    split_outer_intaglio,
    visibility_fraction,
)

__all__ = [
    "MachineAxes",
    "ManufacturingProfile",
    "ReachabilityResult",
    "ReachabilityStatus",
    "candidate_directions",
    "evaluate_reachability",
    "split_outer_intaglio",
    "visibility_fraction",
]
