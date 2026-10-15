"""设计修订链的数据模型。

存在理由（实测发现）：

1. ``POST /api/designs/{id}/margin`` 在旧版里前端从不调用（``api.saveMargin`` 是死代码），
   人工绘制的 Margin 只有在走到 finalize 时才落库。
2. ``suggest-margin`` 会直接覆盖 ``designs.margin_json``，即**静默覆盖已有人工结果**。

因此每次 Margin 变更都必须产生一条不可变的修订记录，且自动提案不得覆盖人工结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class RevisionKind(str, Enum):
    """修订类型。``*_CANDIDATE`` 表示尚未被采纳的提案。"""

    MARGIN_AUTO_ACCEPTED = "margin_auto_accepted"
    MARGIN_CANDIDATE = "margin_candidate"
    MARGIN_MANUAL = "margin_manual"
    AXIS_ACCEPTED = "axis_accepted"
    AXIS_CANDIDATE = "axis_candidate"
    OUTER = "outer"
    OUTER_AUTOFIT_CANDIDATE = "outer_autofit_candidate"
    TEMPLATE_SELECTED = "template_selected"
    AUTOFIT_REJECTED = "autofit_rejected"
    GEOMETRY_FRAME = "geometry_frame"
    NEIGHBOR_CONFIRMED = "neighbor_confirmed"
    BITE_CONFIRMED = "bite_confirmed"
    FINAL = "final"
    PROFILE = "profile"
    SUPPORT_CONDITIONING = "support_conditioning"


class MarginSource(str, Enum):
    AUTO = "auto"
    MANUAL = "manual"
    AUTO_THEN_MANUAL = "auto+manual"

    @property
    def is_human_authored(self) -> bool:
        """该来源是否含人工成分 —— 含人工的修订不得被自动提案静默覆盖。"""
        return self in (MarginSource.MANUAL, MarginSource.AUTO_THEN_MANUAL)


@dataclass
class DesignRevision:
    """一条不可变的设计修订记录。"""

    design_id: str
    revision_no: int
    kind: RevisionKind
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: uuid4().hex)
    algorithm_version: str | None = None
    parameter_set: dict[str, Any] | None = None
    margin_source: MarginSource | None = None
    operator: str | None = None
    approved_by: str | None = None
    operator_actions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    created_at: str | None = None
    #: 该修订是否为当前生效版本。
    is_current: bool = False

    @property
    def is_candidate(self) -> bool:
        return self.kind.value.endswith("_candidate")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "design_id": self.design_id,
            "revision_no": self.revision_no,
            "kind": self.kind.value,
            "is_candidate": self.is_candidate,
            "is_current": self.is_current,
            "algorithm_version": self.algorithm_version,
            "parameter_set": self.parameter_set,
            "margin_source": self.margin_source.value if self.margin_source else None,
            "operator": self.operator,
            "approved_by": self.approved_by,
            "operator_actions": list(self.operator_actions),
            "artifacts": dict(self.artifacts),
            "created_at": self.created_at,
            "payload": dict(self.payload),
        }

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "design_id": self.design_id,
            "revision_no": self.revision_no,
            "kind": self.kind.value,
            "algorithm_version": self.algorithm_version,
            "parameter_set_json": (json.dumps(self.parameter_set, ensure_ascii=False)
                                   if self.parameter_set else None),
            "margin_source": self.margin_source.value if self.margin_source else None,
            "operator": self.operator,
            "approved_by": self.approved_by,
            "operator_actions_json": json.dumps(self.operator_actions, ensure_ascii=False),
            "artifacts_json": json.dumps(self.artifacts, ensure_ascii=False),
            "payload_json": json.dumps(self.payload, ensure_ascii=False),
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: Any, *, current_id: str | None = None) -> "DesignRevision":
        def _loads(value: Any, default: Any) -> Any:
            if not value:
                return default
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return default

        raw_source = row["margin_source"]
        return cls(
            id=row["id"],
            design_id=row["design_id"],
            revision_no=int(row["revision_no"]),
            kind=RevisionKind(row["kind"]),
            payload=_loads(row["payload_json"], {}),
            algorithm_version=row["algorithm_version"],
            parameter_set=_loads(row["parameter_set_json"], None),
            margin_source=MarginSource(raw_source) if raw_source else None,
            operator=row["operator"],
            approved_by=row["approved_by"],
            operator_actions=_loads(row["operator_actions_json"], []),
            artifacts=_loads(row["artifacts_json"], {}),
            created_at=row["created_at"],
            is_current=(current_id is not None and row["id"] == current_id),
        )


def margin_payload(points: list[list[float]], *, note: str | None = None,
                   boundary_semantics: str | None = None) -> dict[str, Any]:
    """构造 Margin 修订的 payload。

    ``boundary_semantics`` 记录这条边界**是什么**（天然牙预备体边缘线 vs 已有实体基台上的
    冠终止线）。同一条曲线在不同指征下含义不同，不可从 ``support_type`` 事后猜。
    """
    return {
        "type": "margin",
        "points": points,
        "point_count": len(points),
        "note": note,
        "boundary_semantics": boundary_semantics,
    }
