"""``.3ox`` / ``.3OX`` 解析器（3Shape 订单交换消息，UTF-16 XML）。

这是 PHI 密度最高的来源文件。实测包含：

- ``Patient_LastName``    患者姓氏
- ``PatientId``           3Shape 侧患者 GUID
- ``LabID`` / ``LabName`` / ``ClinicID``
- ``OperatorID`` / ``LabOperator``
- ``OrderID``             **病例身份主键来源**
- ``ScanSource``          扫描仪型号（TRIOS）
- ``ThreeShapeOrderNo``   实验室订单号

本模块只**提取**，不修改源文件。脱敏由 sanitizer.py 在导入阶段输出到新目录完成。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

#: 这些叶子标签的值一律视为 PHI。
PHI_TAGS: tuple[str, ...] = (
    "Patient_LastName",
    "Patient_FirstName",
    "PatientId",
    "LabID",
    "LabName",
    "ClinicID",
    "ClinicName",
    "OperatorID",
    "LabOperator",
    "DoctorName",
)


@dataclass
class ThreeOxInfo:
    source_file: str
    order_id: str | None = None
    three_shape_order_no: str | None = None
    scan_source: str | None = None
    lab_id: str | None = None
    phi_fields: dict[str, str] = field(default_factory=dict)


def parse_three_ox(path: Path | str) -> ThreeOxInfo:
    """解析 UTF-16 编码的 3Shape 订单 XML。

    文件带 UTF-16 BOM（``ff fe``）且 XML 声明为 ``encoding="utf-16"``，
    因此必须让 ET 自己嗅探编码，不要手工 ``read_text(encoding="utf-8")``。
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    phi: dict[str, str] = {}
    for tag in PHI_TAGS:
        value = root.findtext(f".//{tag}")
        if value:
            phi[tag] = value

    return ThreeOxInfo(
        source_file=str(path),
        order_id=root.findtext(".//OrderID"),
        three_shape_order_no=root.findtext(".//ThreeShapeOrderNo"),
        scan_source=root.findtext(".//ScanSource"),
        lab_id=root.findtext(".//LabID"),
        phi_fields=phi,
    )
