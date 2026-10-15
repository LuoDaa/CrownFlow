"""``.dentalProject`` 解析器（病例级标注 + OrderID）。

``<WorkParams>`` 是 Base64 且带签名的 .NET 序列化负载，本模块**不**尝试解析它。
只取明文部分：

- ``<Notes>``                牙位 / 修复体类型 / 材料，人可读
- ``<TrayNo>``               托盘号
- ``<DateTime>``             病例时间
- ``订单: <uuid>``            **病例身份主键来源**

``Notes`` 里的牙位是"修复体牙位"的字符串，与 ``.constructionInfo`` 的 per-tooth 记录
互为交叉验证。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_ORDER_ID_RE = re.compile(r"订单:\s*([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                          r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
_TEETH_RE = re.compile(r"牙齿\s+([\d|]+)\s*:")
_RESTORATION_RE = re.compile(r"修复体:\s*(.*?)\s*材料:", re.S)
_MATERIAL_RE = re.compile(r"材料:\s*(.*?)\s*颜色:", re.S)


@dataclass
class DentalProjectInfo:
    source_file: str
    order_id: str | None = None
    tray_no: str | None = None
    datetime_text: str | None = None
    restoration_note: str | None = None
    material_note: str | None = None
    teeth_from_notes: list[int] = field(default_factory=list)
    #: Notes 里出现的患者姓名等，供脱敏阶段处理。
    phi_fields: dict[str, str] = field(default_factory=dict)


def parse_dental_project(path: Path | str) -> DentalProjectInfo:
    path = Path(path)
    root = ET.parse(path).getroot()

    notes = root.findtext("Notes") or ""
    flat = re.sub(r"\s+", " ", notes).strip()

    teeth: set[int] = set()
    for group in _TEETH_RE.findall(flat):
        teeth.update(int(t) for t in group.split("|") if t.isdigit())

    restoration = _RESTORATION_RE.search(flat)
    material = _MATERIAL_RE.search(flat)

    info = DentalProjectInfo(
        source_file=str(path),
        order_id=(m.group(1) if (m := _ORDER_ID_RE.search(flat)) else None),
        tray_no=root.findtext("TrayNo"),
        datetime_text=root.findtext("DateTime"),
        restoration_note=restoration.group(1) if restoration else None,
        material_note=material.group(1) if material else None,
        teeth_from_notes=sorted(teeth),
    )

    return info
