"""订单 XML -> CasePrescription + 牙位 + 扫描列表。

结构路径（实测确认）：

    /TDM_Container/TDM_List_Order/TDM_Item_Order              CacheMaterialName
    /TDM_Container/TDM_List_ModelElement/..ModelElement       MaterialID / ProcessStatusID
    /TDM_Container/TDM_List_ToothElement/..ToothElement       ToothNumber / toothElementTypeID
    /TDM_Container/TDM_List_Scan/TDM_Item_Scan                ScanType / ScanID / ScanName

牙位编号是 **Universal（美国）1-32**，不是 FDI。实测 ToothNumber=30 -> FDI 46。
转换走 models.UNIVERSAL_TO_FDI；绝不能把 Universal 数值当 FDI 用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._xml import Node, load_object_graph
from .models import CasePrescription, ToothElement

#: 修复体类型 ID 片段 -> 规范化名称。
#: 只映射**能确证**的片段；其余保留原始 ID 并留空，不硬猜。
_RESTORATION_HINTS: tuple[tuple[str, str], ...] = (
    ("IDCrown", "CROWN"),
    ("IDInlay", "INLAY"),
    ("IDOnlay", "ONLAY"),
    ("IDVeneer", "VENEER"),
    ("IDBridge", "BRIDGE"),
    ("IDAbutment", "ABUTMENT"),
    ("IDPostAndCore", "POST_AND_CORE"),
)

#: 订单 XML 里属 PHI 的字段（与 exocad 侧 sanitizer 的处理口径一致）。
PHI_PROPERTY_NAMES: tuple[str, ...] = (
    "Patient_FirstName", "Patient_LastName", "PatientGuid", "Patient_RefNo",
    "IntOrderID", "ExtOrderID", "ClientContactPerson", "OperatorName",
    "DeliveryAddress1", "DeliveryAddress2", "DeliveryCity", "DeliveryZip",
    "ShipToID", "ClientID", "ERPCustomerNo",
)


def _bool(v: str | None) -> bool | None:
    if v is None or v == "":
        return None
    return v.strip().lower() == "true"


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except ValueError:
        return None


def restoration_type_from_id(tooth_element_type_id: str | None) -> str | None:
    """从 toothElementTypeID 推断修复体类型。无法确证时返回 None。"""
    if not tooth_element_type_id:
        return None
    for needle, name in _RESTORATION_HINTS:
        if needle.lower() in tooth_element_type_id.lower():
            return name
    return None


def parse_order(path: Path | str) -> dict[str, Any]:
    """解析订单 XML。返回 prescription / teeth / scans / meta / phi。"""
    node: Node = load_object_graph(path)

    # **必须用 descendants，不能用 children。**
    # 实测层级是 TDM_Container / TDM_List_ToothElement / TDM_Item_ToothElement ——
    # 条目对象是孙节点。用 children 会一条都找不到（起初就是这样，处方全空）。
    def first(type_name: str) -> Node:
        found = node.descendants(type_name)
        return found[0] if found else Node("", "")

    order = first("TDM_Item_Order")
    model = first("TDM_Item_ModelElement")
    tooth_items = node.descendants("TDM_Item_ToothElement")
    scan_items = node.descendants("TDM_Item_Scan")

    teeth = [
        ToothElement(
            tooth_element_id=t.get("ToothElementID"),
            tooth_number_universal=_int(t.get("ToothNumber")),
            tooth_element_type_id=t.get("toothElementTypeID"),
            cache_tooth_type_class=t.get("CacheToothTypeClass"),
            anatomical=_bool(t.get("Anatomical")),
            post_and_core=_bool(t.get("PostAndCore")),
        )
        for t in tooth_items
    ]

    scans = [
        {
            "scan_id": s.get("ScanID"),
            "scan_type": s.get("ScanType"),
            "scan_name": s.get("ScanName"),
            "file_name": s.get("FileName"),
        }
        for s in scan_items
    ]

    primary = teeth[0] if teeth else None
    rx = CasePrescription(
        target_fdi=primary.target_fdi if primary else None,
        restoration_type=restoration_type_from_id(
            primary.tooth_element_type_id if primary else None),
        material_name=order.get("CacheMaterialName") or model.get("CacheMaterialName"),
        material_id=model.get("MaterialID"),
        tooth_element_type_id=primary.tooth_element_type_id if primary else None,
        process_status=model.get("ProcessStatusID"),
    )

    phi = {
        k: v
        for k, v in {**order.properties, **model.properties}.items()
        if k in PHI_PROPERTY_NAMES and v
    }

    return {
        "prescription": rx,
        "teeth": teeth,
        "scans": scans,
        "order_meta": dict(order.properties),
        "model_meta": dict(model.properties),
        "phi_fields": phi,
    }
