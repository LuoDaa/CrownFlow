"""Materials.xml -> 材料条目 + CAD 参数，并映射到 WelooAI 规范参数。

**这个文件不只是材料清单。** 实测里面有 3Shape 的牙科 CAD 参数对象：

    TDM_Item_CementSpace   ×35    Name / MaterialID / RemoveUndercut / LinerSpace /
                                  CementGap / ExtraCementGap / DistToPrepLine / SmoothDist
    TDM_Item_CrownOverlay  ×14    ContactSpaceOffset / Cut2AntagonistEnabled /
                                  Cut2AntagonistOffset / Cut2NeighborOffset /
                                  MarginLineConnectionOffset*
    TDM_Item_Overlay       ×14    Name="Zircon Crown" / WallThickness /
                                  PrepLineOffset / OffsetAngle / ExtensionOffset

它们**按 MaterialID 索引**，而订单 XML 里的 ``MaterialID`` 指向具体材料 ——
所以订单包确实带着该病例的 CAD 参数，不需要另外要配置。

---

## 映射纪律

用户要求：**只有语义确认过的字段才映射**，不确定的保留 UNRESOLVED。
实测发现三处**不能**按名字硬映射，这里是证据：

| 3Shape 字段 | 值 | 看似对应 | 问题 |
|---|---|---|---|
| `CementSpace.CementGap` | 0.01 | exocad `CementGapThickness`（0.02–0.08） | 3Shape **把间隙拆成两个**：`CementGap` + `ExtraCementGap`（0.06–0.07）。单取一个都不等于 exocad 的单一值，且不知道二者是分区还是叠加 |
| `Overlay.WallThickness` | 0.4 | exocad `MinThickness`（0.6） | 数值差 1.5 倍。一个是**设计目标**、一个是**最小值**，语义不同 |
| `CementSpace.DistToPrepLine` | 0.8–1.0 | exocad `UntouchableZoneNearPreparationMargin`（0–0.1） | **差一个量级**，几乎肯定不是同一个量 |
| `CementSpace.RemoveUndercut` | True/False | exocad `BlockoutMode` / `AdditionalBlockOutAngleAfterCementGap` | 3Shape 是布尔开关，exocad 是模式 + 角度。形态不同，无法直接互换 |

所以：**能确证的标 SEMANTIC_CONFIRMED，语义合理但未获文档确认的标 PROPOSED，
存在冲突的标 UNRESOLVED** —— 不允许"看名字差不多"就硬映射。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ._xml import Node, load_object_graph

#: 参数来源系统的标识（与 exocad 侧分开，不揉成一套"默认参数"）。
SOURCE_SYSTEM = "3shape"


class MappingStatus(str, Enum):
    """字段级映射的确认状态。"""

    #: 语义无歧义，且与规范参数一一对应。
    SEMANTIC_CONFIRMED = "SEMANTIC_CONFIRMED"
    #: 语义合理但未获文档确认（按用户建议先记 PROPOSED，不当作既定事实）。
    PROPOSED = "PROPOSED"
    #: 已知存在冲突或语义不明，**不得参与自动设计**。
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ParameterValue:
    """统一参数模型：规范名 + 值 + 来源系统 + 来源字段 + 状态。

    exocad 与 3Shape 各自产出这种结构，才能做
    "3Shape 怎么配 / exocad 怎么配 / 我们最终怎么配" 的三方对照。
    """

    canonical_name: str | None
    value: float | bool | None
    unit: str | None
    state: MappingStatus
    source_system: str
    source_field: str
    scope: str = "case"
    note: str | None = None
    conflict_with: str | None = None

    def as_dict(self) -> dict:
        return {
            "canonical_name": self.canonical_name,
            "value": self.value,
            "unit": self.unit,
            "state": self.state.value,
            "source_system": self.source_system,
            "source_field": self.source_field,
            "scope": self.scope,
            "note": self.note,
            "conflict_with": self.conflict_with,
        }


#: 3Shape 参数对象字段 -> 规范名 + 确认状态。
#:
#: 只列真正有证据的。``canonical_name`` 为 None = 保持未映射。
_FIELD_MAP: dict[tuple[str, str], tuple[str | None, MappingStatus, str | None]] = {
    # --- 咬合/邻接：布尔开关与偏移量语义清楚 ---
    ("TDM_Item_CrownOverlay", "Cut2AntagonistEnabled"):
        ("occlusion.enabled", MappingStatus.PROPOSED, None),
    ("TDM_Item_CrownOverlay", "Cut2AntagonistOffset"):
        ("occlusion.target_offset", MappingStatus.PROPOSED, None),
    ("TDM_Item_CrownOverlay", "Cut2NeighborOffset"):
        ("contact.target_offset", MappingStatus.PROPOSED, None),
    ("TDM_Item_CrownOverlay", "ContactSpaceOffset"):
        ("contact.space_offset", MappingStatus.PROPOSED, None),

    # --- 间隙：3Shape 拆成两个，不能只取一个当 exocad 的单一值 ---
    ("TDM_Item_CementSpace", "CementGap"):
        ("fitting.cement_gap", MappingStatus.UNRESOLVED,
         "3Shape 把粘接间隙拆成 CementGap + ExtraCementGap；"
         "单取其一不等于 exocad 的单一 CementGapThickness，"
         "且未知二者是分区应用还是叠加。需厂商文档或技师确认。"),
    ("TDM_Item_CementSpace", "ExtraCementGap"):
        ("fitting.extra_cement_gap", MappingStatus.PROPOSED,
         "作为独立项保留；不要与 CementGap 相加后再当作 exocad 的 CementGapThickness。"),
    ("TDM_Item_CementSpace", "LinerSpace"):
        ("fitting.liner_space", MappingStatus.PROPOSED, None),
    ("TDM_Item_CementSpace", "SmoothDist"):
        ("fitting.smooth_distance", MappingStatus.PROPOSED, None),

    # --- 明确冲突：量级差一个数量级 ---
    ("TDM_Item_CementSpace", "DistToPrepLine"):
        (None, MappingStatus.UNRESOLVED,
         "实测 0.8–1.0 mm，而 exocad 的 UntouchableZoneNearPreparationMargin 是 0–0.1 mm，"
         "差一个量级。两者几乎肯定不是同一个量，禁止互相对应。"),

    # --- 厚度：设计目标 vs 最小值，语义不同 ---
    ("TDM_Item_Overlay", "WallThickness"):
        ("thickness.design_target", MappingStatus.UNRESOLVED,
         "实测 0.4 mm，exocad 的 MinThickness 是 0.6 mm。"
         "「设计目标厚度」与「最小允许厚度」不是同一语义，"
         "不能因为都叫厚度就映射到同一个规范名。"),

    # --- 边缘/延伸：语义合理但未确认 ---
    ("TDM_Item_Overlay", "PrepLineOffset"):
        ("fitting.prep_line_offset", MappingStatus.PROPOSED, None),
    ("TDM_Item_Overlay", "ExtensionOffset"):
        ("fitting.extension_offset", MappingStatus.PROPOSED, None),
    ("TDM_Item_Overlay", "OffsetAngle"):
        ("fitting.offset_angle", MappingStatus.PROPOSED, "单位疑为度，未确认。"),

    # --- 倒凹：形态差异导致无法直接互换 ---
    ("TDM_Item_CementSpace", "RemoveUndercut"):
        ("fitting.remove_undercut", MappingStatus.UNRESOLVED,
         "3Shape 是布尔开关；exocad 对应项是 BlockoutMode + 附加角度。"
         "布尔与「模式+角度」不能直接互换。"),
}

#: 明确不参与自动设计、但保留记录的对象类型。
_RECORD_ONLY_TYPES: tuple[str, ...] = (
    "TDM_Item_Attachment", "TDM_Item_Articulator", "TDM_Item_PnCType",
    "TDM_Item_WaxProfileStrip", "TDM_Item_StippledWax",
)


def _to_value(raw: str) -> float | bool | None:
    s = (raw or "").strip()
    if s == "":
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return float(s)
    except ValueError:
        return None


def parse_materials(path: Path | str) -> Node:
    return load_object_graph(path)


def list_material_ids(node: Node) -> dict[str, str]:
    """返回 {MaterialID: 材料名}。"""
    out: dict[str, str] = {}
    for item in node.descendants("TDM_Item_Material"):
        mid = item.get("ItemID") or item.get("MaterialID")
        name = item.get("Name")
        if mid:
            out[mid] = name or ""
    return out


#: 这些子类型词出现时，条目不属于"单冠"，应被排除。
#: 实测 Material1 下有 10 个 CementSpace：Crown / Coping(tangent/chamfer/shoulder/
#: on abutment) / Crown Pontic / Inlay / Veneer / Tabletop。
#: 材质相同但修复体子类型不同就是**不同的参数集**，不能混。
_NON_CROWN_SUBTYPE_TOKENS: tuple[str, ...] = (
    "pontic", "inlay", "onlay", "veneer", "tabletop", "coping", "abutment",
)


def subtype_hint_from_tooth_element_type(tooth_element_type_id: str | None) -> str | None:
    """从 ``toothElementTypeID`` 取修复体子类型提示。

    实测 ``Configuration_11_IDCrownZircon10`` -> ``"Crown"``。
    这是**启发式**：ID 的构词未被文档确认，所以结果只用于收窄候选，
    且收窄行为会记进 provenance，不当作已确认事实。
    """
    if not tooth_element_type_id:
        return None
    low = tooth_element_type_id.lower()
    # 用已知修复体词表匹配，而不是切 ID 词素 ——
    # 实测 ``Configuration_11_IDCrownZircon10`` 切出来是 ``CrownZircon``（不是 ``Crown``），
    # 拿它去过滤会把全部条目都排除掉。
    # 顺序有意：先匹配更具体的词，避免 "crown" 抢到 "crownpontic"。
    for word in ("crownpontic", "pontic", "inlay", "onlay", "veneer",
                 "tabletop", "coping", "crown"):
        if word in low:
            return word
    return None


def collect_cad_parameters(
    node: Node,
    material_id: str | None,
    *,
    subtype_hint: str | None = None,
) -> list[ParameterValue]:
    """收集与 ``material_id``（可选再按 ``subtype_hint`` 收窄）关联的 CAD 参数。

    两级收窄，缺一不可：
    1. **材质**：库里同时定义了几十种材料的参数，拿错材料的参数比拿不到更糟。
    2. **修复体子类型**：同一材质下 CementSpace 按 Crown / Coping / Inlay / Veneer
       分别定义（实测 Material1 下有 10 个），混用等于用错参数集。

    ``subtype_hint`` 为空时只按材质过滤，并把"未收窄"记进 note。
    """
    if not material_id:
        # 拿错材料的参数比拿不到更糟：库里同时定义了几十种材料。
        # 宁可不返回，由验证闸门给出 WARN 让调用方去补 MaterialID。
        return []

    out: list[ParameterValue] = []
    wanted_types = {t for t, _ in _FIELD_MAP}
    hint = (subtype_hint or "").lower()
    excluded_by_subtype = 0
    for type_name in sorted(wanted_types):
        for item in node.descendants(type_name):
            if item.get("MaterialID") != material_id:
                continue
            item_name = (item.get("Name") or "").lower()
            if hint:
                # 子类型收窄：排除明确属于别的修复体类型的条目
                if any(tok in item_name for tok in _NON_CROWN_SUBTYPE_TOKENS):
                    excluded_by_subtype += 1
                    continue
                if hint.lower() not in item_name:
                    excluded_by_subtype += 1
                    continue
            for field_name in item.properties:
                key = (type_name, field_name)
                if key not in _FIELD_MAP:
                    continue
                canonical, status, note = _FIELD_MAP[key]
                raw = item.get(field_name)
                out.append(ParameterValue(
                    canonical_name=canonical,
                    value=_to_value(raw),
                    unit=None if isinstance(_to_value(raw), bool) else "mm",
                    state=status,
                    source_system=SOURCE_SYSTEM,
                    source_field=f"{type_name.split('_', 2)[-1]}.{field_name}",
                    note=(note or "") + (
                        f" [子类型收窄：hint={subtype_hint!r}，排除 {excluded_by_subtype} 条]"
                        if hint else " [未按修复体子类型收窄]"),
                    conflict_with=("exocad.CementGapThickness"
                                   if field_name in ("CementGap", "ExtraCementGap")
                                   else "exocad.MinThickness" if field_name == "WallThickness"
                                   else "exocad.UntouchableZoneNearPreparationMargin"
                                   if field_name == "DistToPrepLine" else None),
                ))
    return out


#: 材料记录可能挂在这几种对象上。实测该包里没有独立的 TDM_Item_Material，
#: 材料名是通过 ToothElementType（如 TDM_Item_ToothElementType_Crown
#: Name='Crown Zircon '）+ CementSpace/Overlay 的 MaterialID 关联出来的。
_MATERIAL_CARRIER_TYPES: tuple[str, ...] = (
    "TDM_Item_ToothElementType_Crown", "TDM_Item_ToothElementType",
    "TDM_Item_Material", "TDM_Item_Material2",
)


def material_record(node: Node, material_id: str | None,
                    *, subtype_hint: str | None = None) -> dict[str, Any] | None:
    """取该材料的记录（名称、ID、绑定关系）。

    找不到时返回 ``None`` —— 由验证闸门给出 WARN，不伪造一条记录。
    """
    if not material_id:
        return None
    hint = (subtype_hint or "").lower()
    for type_name in _MATERIAL_CARRIER_TYPES:
        for item in node.descendants(type_name):
            # 两个字段都要查：实测 TDM_Item_ToothElementType_Crown 的 ItemID 装的是
            # 修复体类型 ID（Configuration_11_IDCrownZircon10），材料 ID 在 MaterialID 里。
            # 用 `ItemID or MaterialID` 会短路到错误的那个。
            idents = {item.get("ItemID"), item.get("MaterialID")}
            if material_id not in idents:
                continue
            name = item.get("Name") or ""
            # 有子类型提示时优先取名字匹配的那条（同一材质下有 Crown/Inlay/… 多条）
            if hint and hint not in name.lower():
                continue
            return {
                "material_id": material_id,
                "name": name.strip() or None,
                "carrier_type": type_name,
                "properties": {k: v for k, v in item.properties.items() if v},
            }
    return None
