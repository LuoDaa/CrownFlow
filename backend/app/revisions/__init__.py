"""设计修订链：不可变修订记录 + 人工 Margin 不被静默覆盖。"""

from .models import DesignRevision, MarginSource, RevisionKind, margin_payload
from .service import (
    accept_revision,
    append_revision,
    current_margin,
    get_revision,
    latest_axis_candidate,
    list_revisions,
    record_axis,
    record_auto_margin_candidate,
    record_manual_margin,
    revision_chain,
)

__all__ = [
    "DesignRevision",
    "MarginSource",
    "RevisionKind",
    "margin_payload",
    "append_revision",
    "list_revisions",
    "get_revision",
    "latest_axis_candidate",
    "current_margin",
    "accept_revision",
    "record_manual_margin",
    "record_auto_margin_candidate",
    "record_axis",
    "revision_chain",
]
