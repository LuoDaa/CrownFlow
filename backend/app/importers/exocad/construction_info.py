"""``.constructionInfo`` 解析器。

exocad 的 per-tooth 设计参数记录。结构：

    <ConstuctionInfo>            <- 注意 exocad 自己拼错了（少了 r）
      <Teeth>
        <Tooth>
          <Number>26</Number>
          <ReconstructionType>AnatomicCrown</ReconstructionType>
          <ParametersUsed>
            <Parameter>
              <Value>0.025</Value>
              <MinValue>0</MinValue>
              <MaxValue>0.5</MaxValue>
              <StepSize xsi:nil="true" />
              <Type>CementGapThickness</Type>
            </Parameter>
            ...

**必须使用 XML parser。** ``<Value xsi:nil="true" />`` 这类自闭合标签不匹配
``<Value>(.*?)</Value>``，正则实现会跳过它们，使其后所有 Value/Type 错位配对，
参数值被静默张冠李戴（实测：``CementGapThickness`` 的总记录数被算成 7 而非 25）。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

from .models import (
    ExocadConstructionCase,
    ExocadConstructionFile,
    ExocadScanFile,
    ExocadToothRecord,
    FunctionalRole,
    ParameterCell,
    ParameterScope,
    ParameterState,
    ReconstructionType,
)

#: ``<Value>`` 为这些值时的文本表示，统一视为"无约束"。
_NAN_TOKENS = {"nan", "NaN", "NAN"}

#: 这些参数取 0 时临床上不可信，必须人工定性后才能参与自动设计。
#:
#: 判定标准：0 会产生**不可就位或不可加工**的结果（不安全），而不是一种合理的设计意图。
#: 反例（不在此表内，因为 0 是合理意图）：``DistanceToNeighbor = 0``（有意接触）、
#: ``UntouchableZoneNearPreparationMargin = 0``（不要保护带）、
#: ``CrownAdditionalSpacing* = 0``（在 cement gap 之上不再加额外间隙）。
ZERO_IS_UNRESOLVED: dict[str, str] = {
    "CementGapThickness": (
        "0 粘接间隙意味着牙冠内表面与预备体直接贴合，无法就位，"
        "临床上不可能是设计意图。需确认这是 exocad 的未配置哨兵值还是真实表达。"
    ),
    "CementGapTopThickness": "0 咬合面间隙会导致咬合面直接贴死对颌牙。",
    "MinThickness": "0 最小壁厚意味着该修复体无材料强度约束。",
    "MinThicknessNearPreparationMargin": "0 边缘区壁厚意味着颈缘无材料。",
}


def _classify_value(name: str, raw_text: str | None) -> tuple[ParameterState, float | None, str | None]:
    """把 exocad 的文本值映射到四态之一。返回 (state, value, unresolved_reason)。"""
    if raw_text is None:
        # <Parameter> 存在但里面没有 <Value> 子元素 —— 写了这个参数却没有可用取值。
        return (
            ParameterState.EXPLICIT_BUT_UNRESOLVED,
            None,
            "该 Parameter 未包含 <Value> 子元素。",
        )

    text = raw_text.strip()
    if text == "":
        return (
            ParameterState.EXPLICIT_BUT_UNRESOLVED,
            None,
            "该 Parameter 的 <Value> 为空。",
        )
    if text in _NAN_TOKENS:
        return ParameterState.NO_CONSTRAINT, None, None

    try:
        value = float(text)
    except ValueError:
        return (
            ParameterState.EXPLICIT_BUT_UNRESOLVED,
            None,
            f"<Value> 无法解析为数值: {text!r}",
        )

    # -1 也是 exocad 用于表达"关闭/不适用"的哨兵，但它出现在
    # AdditionalBlockOutAngleAfterCementGap 上时是合理的显式取值，故不在此处拦截。

    if value == 0.0 and name in ZERO_IS_UNRESOLVED:
        return ParameterState.EXPLICIT_BUT_UNRESOLVED, 0.0, ZERO_IS_UNRESOLVED[name]

    return ParameterState.EXPLICIT, value, None


def parse_construction_info(
    path: Path | str,
    *,
    source_order_id: str | None = None,
) -> list[ExocadToothRecord]:
    """解析一个 ``.constructionInfo``，返回其全部牙位记录。

    ``source_order_id`` 由调用方从 ``.3ox`` / ``.dentalProject`` 取得后传入；
    病例身份只认它，不认目录名。

    只关心牙位记录时用这个；需要扫描清单 / 导出清单 / 坐标系变换时用
    :func:`parse_construction_case`。
    """
    return parse_construction_case(path, source_order_id=source_order_id).teeth


# ---------------------------------------------------------------- 设计几何解析

def _num(text: str | None) -> float | None:
    """把一段文本转成 float；``NaN``/空/非数字一律返回 None。

    exocad 真的会写 ``<MinValue>NaN</MinValue>``，所以不能直接 float()。
    """
    if text is None:
        return None
    t = text.strip()
    if not t or t.lower() in _NAN_TOKENS:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if v == v and v not in (float("inf"), float("-inf")) else None


def _vec3(node: ET.Element | None) -> list[float] | None:
    """读一个三分量向量。

    两种实测写法都要吃：
      * ``<Margin><Vec3><x>..</x><y>..</y><z>..</z></Vec3>``（子标签是 x/y/z）
      * ``<Axis><..>v1</..><..>v2</..><..>v3</..></Axis>``（子标签名未知，按文档序）

    任一分量不是有限数就返回 None —— 半个向量比没有更危险。
    """
    if node is None:
        return None
    kids = list(node)
    if len(kids) == 3:
        named = { (k.tag or "").lower(): k.text for k in kids }
        if all(c in named for c in ("x", "y", "z")):
            vals = [_num(named[c]) for c in ("x", "y", "z")]
        else:
            vals = [_num(k.text) for k in kids]
        if all(v is not None for v in vals):
            return [float(v) for v in vals]
        return None
    # 少数情况写成一行 "x y z" / "x,y,z"
    if node.text:
        parts = node.text.replace(",", " ").split()
        if len(parts) == 3:
            vals = [_num(p) for p in parts]
            if all(v is not None for v in vals):
                return [float(v) for v in vals]
    return None


def _mat4x4(node: ET.Element | None) -> list[float] | None:
    """读一个 4x4 矩阵，按 ``_rc`` 的行主序返回 16 个数。

    exocad 用 ``<_00>``…``<_33>`` 这样的子标签；按标签定位，标签不认得就退到文档序。
    任何一个数缺失/非有限就返回 None，不返回"半个矩阵"。

    **存储约定（实测 82404080845，两个矩阵都是）：平移在最后一行，最后一列是
    ``[0,0,0,1]``。** 也就是按行主序展平时 ``_30 _31 _32`` 是平移。
    按"最后一列是平移"的列向量直觉去用会得到转置后的错误变换。

    另外实测 ``ToothModelMatrix`` 的 3x3 部分**带均匀缩放**：
    奇异值 ``[1.1927, 1.1927, 1.1927]``（即 旋转 x 均匀缩放 1.19 x 平移），
    而 ``MatrixToScanDataFiles`` 的 3x3 是纯旋转（奇异值全 1，纯坐标系变换）。
    所以"exocad 的模板摆放是刚体+均匀缩放"，**各向异性适配发生在后面的
    freeform 造型阶段，不在这个矩阵里** —— 不要把两者混为一谈。
    """
    if node is None:
        return None
    kids = list(node)
    if len(kids) != 16:
        return None
    grid: dict[tuple[int, int], float] = {}
    for k in kids:
        tag = (k.tag or "").strip().lstrip("_")
        v = _num(k.text)
        if v is None or len(tag) != 2 or not tag.isdigit():
            grid = {}
            break
        r, c = int(tag[0]), int(tag[1])
        if not (0 <= r < 4 and 0 <= c < 4):
            grid = {}
            break
        grid[(r, c)] = v
    if len(grid) == 16:
        return [grid[(r, c)] for r in range(4) for c in range(4)]
    # 退路：按文档序
    vals = [_num(k.text) for k in kids]
    if all(v is not None for v in vals):
        return [float(v) for v in vals]
    return None


def _int_list(node: ET.Element | None) -> list[int]:
    """读 ``<ToothNumbers><int>17</int>…</ToothNumbers>`` 这类牙位列表。"""
    if node is None:
        return []
    out: list[int] = []
    for k in node.iter():
        if k is node:
            continue
        t = (k.text or "").strip()
        if t.lstrip("-").isdigit():
            out.append(int(t))
    if out:
        return out
    t = (node.text or "").replace(",", " ").split()
    return [int(x) for x in t if x.lstrip("-").isdigit()]


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    t = (node.text or "").strip()
    return t or None


def _bool(node: ET.Element | None) -> bool | None:
    t = _text(node)
    if t is None:
        return None
    low = t.lower()
    if low in ("true", "1"):
        return True
    if low in ("false", "0"):
        return False
    return None


def _margin_points(tooth: ET.Element) -> tuple[list[list[float]], dict]:
    """读颈缘线。返回 ``(点集, 诊断)``；坏点被计数并跳过，不静默丢弃。"""
    node = tooth.find("Margin")
    if node is None:
        return [], {"present": False, "n_total": 0, "n_bad": 0}
    children = list(node)
    pts: list[list[float]] = []
    bad = 0
    for k in children:
        v = _vec3(k)
        if v is None:
            bad += 1
            continue
        pts.append(v)
    return pts, {"present": True, "n_total": len(children), "n_bad": bad}


def parse_construction_case(
    path: Path | str,
    *,
    source_order_id: str | None = None,
) -> ExocadConstructionCase:
    """解析一个 ``.constructionInfo`` 的**全部**内容。

    比 :func:`parse_construction_info` 多出：

      * 每牙位的设计几何（Margin / Axis / 牙架 / Center / 库牙 / 4x4 变换）
      * 病例级 ``ScanFiles``（含 ``PartType``，即预备体颌/对颌的真值）
      * 病例级 ``ConstructionFileList``（导出文件名 → 与 ``*_cad.stl`` 对应）
      * ``MatrixToScanDataFiles`` 与身份字段（PHI 单独收集）

    **单个元素解析失败不会让整份文件失败** —— 记录进 ``diagnostics``，
    其余字段照常返回。real 文件里字段缺失是常态，不是异常。
    """
    path = Path(path)
    root = ET.parse(path).getroot()

    teeth: list[ExocadToothRecord] = []
    diagnostics: dict[str, dict] = {}

    for tooth in root.iter("Tooth"):
        fdi_text = tooth.findtext("Number")
        try:
            fdi = int(fdi_text) if fdi_text else None
        except ValueError:
            fdi = None

        rt = ReconstructionType.parse(tooth.findtext("ReconstructionType"))

        parameters: dict[str, ParameterCell] = {}
        used = tooth.find("ParametersUsed")
        if used is not None:
            for prm in used.findall("Parameter"):
                name = prm.findtext("Type")
                if not name:
                    continue
                raw = prm.find("Value")
                raw_text = raw.text if raw is not None else None
                state, value, reason = _classify_value(name, raw_text)
                cell = ParameterCell(
                    name=name,
                    state=state,
                    value=value,
                    source="constructionInfo",
                    scope=_default_scope(name),
                    raw_text=raw_text,
                    provenance={
                        "order_id": source_order_id,
                        "fdi": fdi,
                        "file": path.name,
                        "min": prm.findtext("MinValue"),
                        "max": prm.findtext("MaxValue"),
                    },
                    unresolved_reason=reason,
                )
                if name in parameters and parameters[name].value != cell.value:
                    # 同一牙位同名参数出现冲突值时以最后一个为准，但保留痕迹。
                    cell = replace(
                        cell,
                        provenance={
                            **cell.provenance,
                            "conflict_with": parameters[name].value,
                        },
                    )
                parameters[name] = cell

        margin, margin_diag = _margin_points(tooth)
        if margin_diag["present"]:
            diagnostics[f"fdi_{fdi}_margin"] = margin_diag

        teeth.append(
            ExocadToothRecord(
                source_file=str(path),
                source_order_id=source_order_id,
                case_dir_name=path.parent.name,
                tooth_fdi=fdi,
                reconstruction_type=rt,
                functional_role=FunctionalRole.from_reconstruction_type(rt),
                parameters=parameters,
                margin_points=margin,
                margin_bottom_type=_text(tooth.find("MarginBottomType")),
                insertion_axis=_vec3(tooth.find("Axis")),
                axis_occlusal_of_tooth_model=_vec3(
                    tooth.find("AxisOcclusalOfToothModel")),
                axis_mesial=_vec3(tooth.find("AxisMesial")),
                axis_buccal=_vec3(tooth.find("AxisBuccal")),
                center=_vec3(tooth.find("Center")),
                tooth_library_name=_text(tooth.find("ToothLibraryName")),
                tooth_library_tooth_number=_text(
                    tooth.find("ToothLibraryToothNumber")),
                tooth_model_matrix=_mat4x4(tooth.find("ToothModelMatrix")),
                tooth_scan_file_name=_text(tooth.find("ToothScanFileName")),
            )
        )

    scans: list[ExocadScanFile] = []
    for sf in root.findall("ScanFiles/ScanFile"):
        scans.append(ExocadScanFile(
            file_name=_text(sf.find("FileName")),
            part_type=_text(sf.find("PartType")),
            tooth_numbers=_int_list(sf.find("ToothNumbers")),
            transformation_matrix=_mat4x4(sf.find("TransformationMatrix")),
            file_hash=_text(sf.find("FileHash")),
            loaded_by=_text(sf.find("LoadedBy")),
            part_name=_text(sf.find("PartName")),
        ))

    constructions: list[ExocadConstructionFile] = []
    for cf in root.findall("ConstructionFileList/ConstructionFile"):
        constructions.append(ExocadConstructionFile(
            filename=_text(cf.find("Filename")),
            material=_text(cf.find("Material")),
            material_name=_text(cf.find("MaterialName")),
            material_postfix=_text(cf.find("MaterialPostfix")),
            material_color=_text(cf.find("MaterialColor")),
            height=_text(cf.find("Height")),
            tooth_numbers=_int_list(cf.find("ToothNumbers")),
            reconstruction_type_hash=_text(cf.find("ReconstructionTypeHashSHA1")),
            z_rotation_matrix=_mat4x4(cf.find("ZRotationMatrix")),
            estimated_direction_for_cam=_vec3(cf.find("EstimatedDirectionForCAM")),
            is_screw_retained=_bool(cf.find("IsScrewRetained")),
            export_information=_text(cf.find("ExportInformation")),
            save_time=_text(cf.find("SaveTime")),
        ))

    return ExocadConstructionCase(
        source_file=str(path),
        case_dir_name=path.parent.name,
        teeth=teeth,
        scan_files=scans,
        construction_files=constructions,
        matrix_to_scan_data_files=_mat4x4(root.find("MatrixToScanDataFiles")),
        patient_id=_text(root.find("PatientId")),
        practice_id=_text(root.find("PracticeId")),
        technician_id=_text(root.find("TechnicianId")),
        project_name=_text(root.find("ProjectName")),
        project_dir=_text(root.find("ProjectDir")),
        design_guid=_text(root.find("DesignGUID")),
        diagnostics=diagnostics,
    )


def _default_scope(name: str) -> ParameterScope:
    """参数的默认作用域。

    实测依据（8 个病例 / 38 条牙位记录）：
    - ``MillingDiameter`` 在 0.5–1.2 之间取值（5 种），是机台/订单相关的加工参数，
      **不属于材料 profile**。
    - ``MillingMachineAxes`` 取 3 或 5，是机台能力，属订单/实验室维度。
    - ``CementGapThickness`` 是真正的逐病例变量。
    """
    if name.startswith("Milling") or name.startswith("ProductionBlank"):
        return ParameterScope.ORDER
    if name.startswith("Model"):
        return ParameterScope.CASE
    if name.startswith("Connector") or name.startswith("Bridge") or name.startswith("MaximumBridge"):
        return ParameterScope.CASE
    if name.startswith("AbutmentEmergenceProfile") or name.startswith("AdaptEmergence"):
        return ParameterScope.RESTORATION
    return ParameterScope.TOOTH
