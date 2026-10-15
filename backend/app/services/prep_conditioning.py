"""Prep Surface Contract / Conditioning **编排**（P1，2026-10-13）。

**这一批刻意不写几何算法。** 目标只有一个：把"哪些真实病例属于 1.0、哪些输入满足后续
几何链的输入契约"变成**机器可判定**的事，并给出可信分母。

消费的现有积木（不复制判据）：

- ``geometry.nonplanar_crown.fitting_support_above_margin`` —— **权威契约谓词**
  （单一完整就位面 + 单一颈环；内部还会跑 ``audit_margin_on_scan`` 拒"Margin 跨开放边界"）；
- ``geometry.prep_margin_diagnostics.diagnose`` —— 只读诊断（组件数 / 边界环 / 只读源码面 id /
  Margin 到扫描的距离），**从不删岛、不补洞**；
- ``geometry.roi.spherical_roi`` / ``implant_candidate.component_at_point`` —— 资产准备阶段已有的选取原语。

纪律：
- ``readiness`` 只由上面那个谓词决定 —— 绝不另写一份"看起来能过"的判据；
- 三档 ``PREP_SURFACE_READY`` / ``REVIEW_REQUIRED`` / ``UNSUPPORTED_INDICATION``，不是 PASS/FAIL；
- 任何"自动封洞 / 缝螺丝孔 / 智能删组件 / 重建 prep"都**不在本模块**；
- 契约里保留操作日志与前后 sha256，来源可追。

1.0 支持范围（冻结）：**Natural-tooth single anatomical crown**。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

#: 1.0 indication 名称（冻结，不要改口径）
INDICATION_NATURAL_SINGLE_CROWN = "NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN"
SCOPE_1_0 = INDICATION_NATURAL_SINGLE_CROWN


class Indication(str, Enum):
    """1.0 支持范围内的牙位类型；非本枚举值 = 不在 1.0 范围。"""

    NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN = INDICATION_NATURAL_SINGLE_CROWN
    ANATOMIC_PONTIC = "ANATOMIC_PONTIC"
    REDUCED_PONTIC = "REDUCED_PONTIC"
    COPING_ONLY = "COPING_ONLY"
    IMPLANT_ABUTMENT = "IMPLANT_ABUTMENT"
    CROWN_ON_EXISTING_ABUTMENT = "CROWN_ON_EXISTING_ABUTMENT"
    CUSTOM_IMPLANT_ABUTMENT = "CUSTOM_IMPLANT_ABUTMENT"
    ANTAGONIST_DESIGN_ARTIFACT = "ANTAGONIST_DESIGN_ARTIFACT"
    HEALTHY_TOOTH = "HEALTHY_TOOTH"
    BRIDGE_UNIT = "BRIDGE_UNIT"
    UNKNOWN_INDICATION = "UNKNOWN_INDICATION"
    #: 应用侧：support_type='prep' 且无种植组件候选
    NATURAL_PREP = "NATURAL_PREP"


#: 只有这个 indication 才算 1.0 的合格单位
ELIGIBLE_INDICATIONS = {
    Indication.CROWN_ON_EXISTING_ABUTMENT.value,
    Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value,
    Indication.NATURAL_PREP.value,
}

#: 明确"模型不该继续往下试"的 indication（产品边界，不是算法失败）
UNSUPPORTED_INDICATIONS = {
    Indication.CUSTOM_IMPLANT_ABUTMENT.value,
    Indication.ANATOMIC_PONTIC.value, Indication.REDUCED_PONTIC.value,
    Indication.COPING_ONLY.value, Indication.IMPLANT_ABUTMENT.value,
    Indication.ANTAGONIST_DESIGN_ARTIFACT.value, Indication.HEALTHY_TOOTH.value,
    Indication.BRIDGE_UNIT.value,
}


class Readiness(str, Enum):
    PREP_SURFACE_READY = "PREP_SURFACE_READY"
    ABUTMENT_SUPPORT_READY = "ABUTMENT_SUPPORT_READY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    UNSUPPORTED_INDICATION = "UNSUPPORTED_INDICATION"
    #: 连支撑资产都没有（不是"试用失败"，是根本没东西可判）
    NO_SUPPORT_ASSET = "NO_SUPPORT_ASSET"


#: Margin 到支撑表面的**同坐标系健全性门**（mm）。
#: 依据：真实病例里 margin 就贴在预备体/代型表面上（实测 1629 病例 0.24 mm 量级）；
#: 30–45 mm 这种距离只可能是**两个不同坐标系**（例如设计系 margin vs 模型/打印系代型），
#: 而不是"边缘线画歪了"。不设这道门时，拓扑谓词会**必然通过**（margin 离网格很远 ⇒
#: "margin 不落在开放边界上"平凡成立），产出"自信地错"的 PREP_SURFACE_READY。
MARGIN_ON_SURFACE_MAX_MM = 1.0

#: 拒绝分类：A 类=indication 外；B 类=资产准备问题（可人工重新裁取/选组件后再试）
REJECTION_CLASS_INDICATION = "UNSUPPORTED_INDICATION"
REJECTION_CLASS_ASSET = "ASSET_PREPARATION_REQUIRED"
REJECTION_CLASS_REVIEW = "REVIEW_REQUIRED"
REJECTION_CLASS_NONE = "READY"


#: ``fitting_support_above_margin`` / ``audit_margin_on_scan`` 的失败消息 → (code, class)。
#: 这些字符串来自那两个函数本体，本模块**只做映射，不做几何判断**。
_MESSAGE_CODES: tuple[tuple[str, str, str], ...] = (
    ("Margin 跨扫描开放边界或缺口", "MARGIN_CROSSES_OPEN_BOUNDARY", REJECTION_CLASS_ASSET),
    ("单一完整就位面", "FITTING_SURFACE_NOT_SINGLE_COMPLETE", REJECTION_CLASS_ASSET),
    ("Margin 投影三角化退化", "MARGIN_CURTAIN_DEGENERATE", REJECTION_CLASS_ASSET),
    ("Margin 段跨越范围过大", "MARGIN_SEGMENT_TOO_LONG", REJECTION_CLASS_ASSET),
    ("三维颈部过渡高度", "TRANSITION_HEIGHT_INVALID", REJECTION_CLASS_REVIEW),
    ("内外面来源无法可靠区分", "INTAGLIO_EXTERIOR_AMBIGUOUS", REJECTION_CLASS_REVIEW),
    ("布尔主体不能分离为两个单颈环面", "BOOLEAN_BODY_NOT_TWO_SINGLE_LOOPS", REJECTION_CLASS_ASSET),
)

#: ``diagnose`` 的只读 codes → 拒绝码（与上面同表不同来源；两处都出现在报告里）
_DIAGNOSE_CODES = {
    "PREP_MULTICOMPONENT": "MULTIPLE_COMPONENTS_RAW_SUPPORT",
    "MULTIPLE_BORDER_LOOPS": "EXTRA_BOUNDARY_LOOPS_ABOVE_MARGIN",
    "PREP_MARGIN_TOPOLOGY_UNSUPPORTED": "MARGIN_TOPOLOGY_UNSUPPORTED",
}


def classify_failure_message(message: str) -> tuple[str, str]:
    """把现成谓词的失败消息映射成 (拒绝码, 分类)。**只做字符串映射。**"""
    text = str(message or "")
    for needle, code, category in _MESSAGE_CODES:
        if needle in text:
            return code, category
    return "PREP_SURFACE_CONTRACT_UNMAPPED", REJECTION_CLASS_REVIEW


# ---------------------------------------------------------------- indication 分诊

def _enum_value(value):
    """枚举取 ``value``（``ExocadToothRecord`` 的字段是 str-Enum 或 None）。"""
    if value is None:
        return None
    return getattr(value, "value", value)


def classify_tooth_metadata(recon: str | None, implant: str | None) -> Indication:
    """从 exocad 逐牙位元数据判定 indication（``recon`` / ``implant`` 字段）。"""
    imp = str(_enum_value(implant) or "").strip()
    if imp and imp.lower() not in ("none", "null", "-"):
        return Indication.IMPLANT_ABUTMENT
    kind = str(_enum_value(recon) or "").strip()
    mapping = {
        "AnatomicCrown": Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN,
        "AnatomicPontic": Indication.ANATOMIC_PONTIC,
        "ReducedPontic": Indication.REDUCED_PONTIC,
        "Coping": Indication.COPING_ONLY,
        "Antagonist": Indication.ANTAGONIST_DESIGN_ARTIFACT,
        "HealthyTooth": Indication.HEALTHY_TOOTH,
    }
    return mapping.get(kind, Indication.UNKNOWN_INDICATION)


def _quadrant_order(fdi: int) -> tuple[int, int]:
    """(象限, 象限内序号)，用于判断两个牙位是否相邻。"""
    quadrant, position = divmod(int(fdi), 10)
    return quadrant, position


def triage_units(teeth: list[dict]) -> list[dict]:
    """逐牙位分诊；并把"与桥体相邻的冠"判为 ``BRIDGE_UNIT``。

    规则（可审计、不猜）：同一象限内，若一个 ``AnatomicCrown`` 单位在 ±1 牙位内有
    ``AnatomicPontic``/``ReducedPontic``，则它属于桥，不是 1.0 的游离单冠。
    """
    units = []
    for tooth in teeth or []:
        if not isinstance(tooth, dict):
            continue
        fdi_raw = tooth.get("fdi")
        indication = classify_tooth_metadata(tooth.get("recon"), tooth.get("implant"))
        notes = []
        try:
            fdi = int(str(fdi_raw))
        except (TypeError, ValueError):
            fdi = None
        if fdi is not None and indication is Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN:
            quadrant = _quadrant_order(fdi)[0]
            neighbours = []
            for other in teeth:
                if not isinstance(other, dict) or other is tooth:
                    continue
                try:
                    other_fdi = int(str(other.get("fdi")))
                except (TypeError, ValueError):
                    continue
                if _quadrant_order(other_fdi)[0] != quadrant or abs(other_fdi - fdi) != 1:
                    continue
                if classify_tooth_metadata(other.get("recon"), other.get("implant")) in (
                        Indication.ANATOMIC_PONTIC, Indication.REDUCED_PONTIC):
                    neighbours.append(other_fdi)
            if neighbours:
                indication = Indication.BRIDGE_UNIT
                notes.append("桥体相邻（FDI %s）⇒ 非游离单冠" % neighbours)
        units.append({"fdi": fdi, "indication": indication.value,
                      "eligible": indication.value in ELIGIBLE_INDICATIONS,
                      "recon": str(_enum_value(tooth.get("recon")) or "") or None,
                      "implant": str(_enum_value(tooth.get("implant")) or "") or None,
                      "margin_point_count": tooth.get("margin_n"),
                      "notes": notes})
    return units


# ---------------------------------------------------------------- 契约对象

@dataclass
class PreparedSurfaceContract:
    """进入 crown-bottom chain 之前的**输入契约**（机器可判定）。"""

    case_id: str | None = None
    target_fdi: int | None = None
    source_asset_sha256: str | None = None
    source_revision_id: str | None = None
    conditioned_mesh_sha256: str | None = None
    source_face_ids: list | None = None
    component_count: int | None = None
    boundary_loop_count: int | None = None
    margin_revision_id: str | None = None
    insertion_axis_revision_id: str | None = None
    diagnostics: dict = field(default_factory=dict)
    operations: list = field(default_factory=list)
    indication: str = Indication.UNKNOWN_INDICATION.value
    readiness: str = Readiness.REVIEW_REQUIRED.value
    rejection_code: str = "NOT_ASSESSED"
    rejection_class: str = REJECTION_CLASS_REVIEW
    rejection_reason: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.readiness in (Readiness.PREP_SURFACE_READY.value,Readiness.ABUTMENT_SUPPORT_READY.value)

    @property
    def eligible(self) -> bool:
        return self.indication in ELIGIBLE_INDICATIONS

    def as_report(self) -> dict:
        return {
            "identity": {"case_id": self.case_id, "target_fdi": self.target_fdi,
                         "source_asset_sha256": self.source_asset_sha256,
                         "source_revision_id": self.source_revision_id},
            "geometry": {"conditioned_mesh_sha256": self.conditioned_mesh_sha256,
                         "source_face_ids": self.source_face_ids,
                         "component_count": self.component_count,
                         "boundary_loop_count": self.boundary_loop_count,
                         "margin_revision_id": self.margin_revision_id,
                         "insertion_axis_revision_id": self.insertion_axis_revision_id},
            "diagnostics": dict(self.diagnostics),
            "operations": list(self.operations),
            "classification": {"indication": self.indication, "readiness": self.readiness,
                               "eligible_for_1_0": self.eligible,
                               "rejection_code": self.rejection_code,
                               "rejection_class": self.rejection_class,
                               "rejection_reason": self.rejection_reason},
            "provenance": dict(self.provenance),
        }


def assess_prep_surface(support, margin, axis, *, case_id=None, target_fdi=None,
                        source_asset_sha256=None, source_revision_id=None,
                        margin_revision_id=None, insertion_axis_revision_id=None,
                        margin_point_count=None,
                        indication: str | None = None,
                        screw_channel_present: bool | None = None,
                        provenance: dict | None = None) -> PreparedSurfaceContract:
    """评估一个预备体资产是否满足 crown-bottom chain 的输入契约。

    **只读**：不修改 ``support``，不做任何修复。``readiness`` 由现成谓词
    ``fitting_support_above_margin`` 决定；诊断数字来自 ``prep_margin_diagnostics.diagnose``。
    """
    from ..geometry.prep_margin_diagnostics import diagnose
    from ..geometry.nonplanar_crown import MarginBuildError, fitting_support_above_margin

    contract = PreparedSurfaceContract(
        case_id=case_id, target_fdi=target_fdi,
        source_asset_sha256=source_asset_sha256, source_revision_id=source_revision_id,
        margin_revision_id=margin_revision_id,
        insertion_axis_revision_id=insertion_axis_revision_id,
        indication=indication or Indication.UNKNOWN_INDICATION.value,
        provenance=dict(provenance or {}),)

    # 1) indication 先判：不在 1.0 范围就不进几何链（这是产品边界，不是算法失败）
    if contract.indication in UNSUPPORTED_INDICATIONS:
        contract.readiness = Readiness.UNSUPPORTED_INDICATION.value
        contract.rejection_code = ("IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL"
                                   if contract.indication == Indication.IMPLANT_ABUTMENT.value
                                   else "UNSUPPORTED_INDICATION_%s" % contract.indication)
        contract.rejection_class = REJECTION_CLASS_INDICATION
        contract.rejection_reason = "1.0 支持范围限定为天然牙单冠（%s）" % SCOPE_1_0
        contract.diagnostics = {"assessed": False, "reason": "indication_out_of_scope",
                                "screw_channel_present": screw_channel_present}
        return contract
    # indication 未判定**不短路**：几何结论照样要给（"是否属于 1.0 范围"与
    # "输入是否满足契约"是两条正交轴，混在一起会掩盖真实拒绝码）。

    # 2) 只读诊断（组件 / 边界环 / Margin 到扫描的距离 / 只读源码面 id）
    margin_array = np.asarray(margin, float)
    axis_array = np.asarray(axis, float)
    report, _cropped = diagnose(support, margin_array, axis_array)
    codes = list(report.get("codes") or [])
    contract.component_count = len(report.get("original_components_faces") or []) or None
    cropped_components = len(report.get("cropped_components_faces") or [])
    contract.boundary_loop_count = len(report.get("cropped_boundary_loops_world") or []) or cropped_components + 1
    contract.diagnostics = {
        "assessed": True,
        "diagnose_codes": codes,
        "margin_scan_distance_max_mm": report.get("margin_scan_distance_max_mm"),
        "cropped_component_count": cropped_components,
        "cropped_island_source_face_ids": report.get("cropped_island_source_face_ids") or [],
        "neighbor_or_base_contamination_suspected": bool(report.get("cropped_island_source_face_ids")),
        "margin_point_count": margin_point_count,
        "screw_channel_present": screw_channel_present,
        "conditioning_operations": "NONE（本批不做任何几何修复）",
    }
    contract.source_face_ids = report.get("cropped_face_source_ids")

    # 2.5) 同坐标系健全性门：margin 必须**贴着**支撑表面。
    # 这一步必须在拓扑谓词之前 —— 否则"margin 在别的坐标系里"会被拓扑判据当成通过。
    margin_gap = report.get("margin_scan_distance_max_mm")
    if margin_gap is not None and float(margin_gap) > MARGIN_ON_SURFACE_MAX_MM:
        contract.readiness = Readiness.REVIEW_REQUIRED.value
        contract.rejection_code = "MARGIN_NOT_ON_SUPPORT_SURFACE"
        contract.rejection_class = REJECTION_CLASS_ASSET
        contract.rejection_reason = (
            "Margin 到支撑表面的最大距离 %.3f mm 超过 %.2f mm —— 两者很可能不在同一坐标系"
            "（设计系 margin vs 模型/打印系网格），拓扑判据在这种情形下会平凡通过，"
            "因此不给出 READY。" % (float(margin_gap), MARGIN_ON_SURFACE_MAX_MM))
        contract.diagnostics["margin_to_surface_max_mm"] = float(margin_gap)
        contract.diagnostics["coordinate_frame_sane"] = False
        return contract
    contract.diagnostics["margin_to_surface_max_mm"] = margin_gap
    contract.diagnostics["coordinate_frame_sane"] = True

    # 3) 权威契约谓词（唯一判据；失败消息映射成拒绝码，不另写判据）
    try:
        fitting_support_above_margin(support, margin_array, axis_array)
    except MarginBuildError as exc:
        # 以**契约谓词自己的判词**为主（它直接说明违反了哪条契约、也是操作员要处理的那件事）；
        # diagnose 的只读 codes 作为佐证代码一并留在报告里，不覆盖主码。
        code, category = classify_failure_message(str(exc))
        if code == "PREP_SURFACE_CONTRACT_UNMAPPED":
            for diagnose_code in codes:
                if diagnose_code in _DIAGNOSE_CODES:
                    code, category = _DIAGNOSE_CODES[diagnose_code], REJECTION_CLASS_ASSET
                    break
        contract.diagnostics["corroborating_codes"] = [_DIAGNOSE_CODES[c] for c in codes
                                                       if c in _DIAGNOSE_CODES]
        contract.readiness = Readiness.REVIEW_REQUIRED.value
        contract.rejection_code = code
        contract.rejection_class = category
        contract.rejection_reason = str(exc)
        return contract
    except ValueError as exc:                       # 参数级错误：如实报 REVIEW
        contract.readiness = Readiness.REVIEW_REQUIRED.value
        contract.rejection_code = "PREP_INPUT_INVALID"
        contract.rejection_class = REJECTION_CLASS_REVIEW
        contract.rejection_reason = str(exc)
        return contract

    contract.readiness = Readiness.PREP_SURFACE_READY.value
    contract.rejection_code = "READY"
    contract.rejection_class = REJECTION_CLASS_NONE
    contract.rejection_reason = ""
    return contract


# ---------------------------------------------------------------- 应用侧接线（只读）

def assess_design_prep(design, *, support_path=None, asset_row=None) -> PreparedSurfaceContract:
    """从 designs 行构造契约（只读；不落库、不改设计）。供 UI 在尝试构建内冠之前自检。"""
    from ..geometry.io import load_mesh
    from .design_dependencies import file_digest
    from .abutment_support import indication_of,prepare_design_support
    if indication_of(design) in ('CROWN_ON_EXISTING_ABUTMENT','CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'):
        reference={**(asset_row or {}),'path':str(support_path)} if support_path is not None else None
        return prepare_design_support(design,require_ready=False,lookup_asset=False,reference_override=reference)[0]

    support_type = (design.get("support_type") or "").lower()
    mesh_info = {}
    if asset_row and asset_row.get("mesh_info_json"):
        try:
            mesh_info = json.loads(asset_row["mesh_info_json"])
        except (TypeError, ValueError):
            mesh_info = {}
    implant_candidate = bool(mesh_info.get("implant_candidate"))
    confirmed_abutment = bool(mesh_info.get("restorative_abutment_confirmed"))
    if support_type == "abutment" or (implant_candidate and not confirmed_abutment):
        indication = Indication.IMPLANT_ABUTMENT.value
    elif support_type == "prep":
        indication = Indication.NATURAL_PREP.value
    else:
        indication = Indication.UNKNOWN_INDICATION.value

    screw = True if implant_candidate else (False if support_type == "prep" else None)
    # 顺序：indication 外 → 输入未确认 → 缺资产 → 才跑几何（范围外的病例不浪费在几何上）
    if indication in UNSUPPORTED_INDICATIONS:
        return PreparedSurfaceContract(
            case_id=design.get("case_id"), target_fdi=design.get("target_fdi"),
            indication=indication, readiness=Readiness.UNSUPPORTED_INDICATION.value,
            rejection_code=("IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL"
                            if indication == Indication.IMPLANT_ABUTMENT.value
                            else "UNSUPPORTED_INDICATION_%s" % indication),
            rejection_class=REJECTION_CLASS_INDICATION,
            rejection_reason="1.0 支持范围限定为天然牙单冠（%s）" % SCOPE_1_0,
            diagnostics={"assessed": False, "reason": "indication_out_of_scope",
                         "screw_channel_present": screw},
            provenance={"source": "design_row"})
    try:
        margin = json.loads(design.get("margin_json") or "null")
        axis = json.loads(design.get("axis_json") or "null")
    except (TypeError, ValueError):
        margin = axis = None
    if not margin or not axis:
        contract = PreparedSurfaceContract(
            case_id=design.get("case_id"), target_fdi=design.get("target_fdi"),
            indication=indication, readiness=Readiness.REVIEW_REQUIRED.value,
            rejection_code="MARGIN_OR_AXIS_UNCONFIRMED", rejection_class=REJECTION_CLASS_REVIEW,
            rejection_reason="请先确认 Margin 与插入轴", provenance={"source": "design_row"})
        return contract
    if support_path is None:
        contract = PreparedSurfaceContract(
            case_id=design.get("case_id"), target_fdi=design.get("target_fdi"),
            indication=indication, readiness=Readiness.NO_SUPPORT_ASSET.value,
            rejection_code="NO_SUPPORT_ASSET", rejection_class=REJECTION_CLASS_ASSET,
            rejection_reason="尚未定位/上传目标基牙支撑", provenance={"source": "design_row"})
        return contract

    path = Path(support_path)
    sha = file_digest(path) if path.is_file() else None
    return assess_prep_surface(
        load_mesh(path), np.asarray(margin, float), np.asarray(axis, float),
        case_id=design.get("case_id"), target_fdi=design.get("target_fdi"),
        source_asset_sha256=sha, source_revision_id=design.get("design_revision"),
        margin_revision_id=design.get("confirmed_margin_revision_id"),
        insertion_axis_revision_id=design.get("current_axis_revision_id"),
        margin_point_count=len(margin), indication=indication,
        screw_channel_present=screw,
        provenance={"source": "design_row", "support_path_name": path.name})


# ---------------------------------------------------------------- 语料报表

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    try:
        from .design_dependencies import file_digest
        return file_digest(path) if path.is_file() else None
    except Exception:                                            # noqa: BLE001
        return None


def triage_cases(cases_root: Path | str) -> dict:
    """对 exocad 病例目录做**逐牙位分诊**（只看元数据，不读几何）。"""
    root = Path(cases_root)
    cases = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        infos = sorted(case_dir.glob("*.constructionInfo")) or sorted(case_dir.glob("*.modelInfo"))
        if not infos:
            others = sorted(p.name for p in case_dir.iterdir() if p.suffix.lower() in (".3ox", ".dentalcad"))
            cases.append({"case": case_dir.name, "units": [], "metadata": None,
                          "notes": ["无 constructionInfo/modelInfo —— 设计元数据只在二进制快照里（%s）"
                                    % ", ".join(others) if others else "目录里没有设计元数据文件"]})
            continue
        try:
            from ..importers.exocad.construction_info import parse_construction_case
            parsed = parse_construction_case(infos[0])
            teeth = []
            for tooth in getattr(parsed, "teeth", []) or []:
                teeth.append({"fdi": getattr(tooth, "tooth_fdi", None),
                              "recon": _enum_value(getattr(tooth, "reconstruction_type", None)),
                              "implant": _enum_value(getattr(tooth, "implant_type", None)),
                              "margin_n": len(getattr(tooth, "margin_points", []) or [])})
            units = triage_units(teeth)
            cases.append({"case": case_dir.name, "metadata": str(infos[0].name), "units": units,
                          "notes": []})
        except Exception as exc:                                 # noqa: BLE001
            cases.append({"case": case_dir.name, "metadata": str(infos[0].name), "units": [],
                          "notes": ["元数据解析失败：%s" % type(exc).__name__]})
    return {"cases": cases}


def assess_bundle_dir(bundles_root: Path | str, *, support_names=("prep.stl", "die.stl")) -> list[dict]:
    """对病例包（case.json + 可选 prep/die）逐个跑契约；没有支撑资产就如实记为 NO_SUPPORT_ASSET。"""
    from ..geometry.io import load_mesh

    results = []
    for bundle in sorted(p for p in Path(bundles_root).iterdir() if p.is_dir()):
        meta_path = bundle / "case.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        margin = meta.get("margin_points")
        axis = meta.get("insertion_axis")
        support = next((bundle / name for name in support_names if (bundle / name).is_file()), None)
        entry = {"bundle": bundle.name, "fdi": meta.get("fdi"),
                 "margin_point_count": len(margin or []),
                 "source_json_sha256": _sha256(meta_path)}
        if support is None or not margin or not axis:
            entry["contract"] = {"classification": {
                "indication": Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value,
                "readiness": Readiness.NO_SUPPORT_ASSET.value,
                "eligible_for_1_0": True, "rejection_code": "NO_SUPPORT_ASSET",
                "rejection_class": REJECTION_CLASS_ASSET,
                "rejection_reason": "病例包内无 prep.stl/die.stl（或 case.json 缺 Margin/轴）"}}
            results.append(entry)
            continue
        try:
            mesh = load_mesh(support)
            contract = assess_prep_surface(
                mesh, np.asarray(margin, float), np.asarray(axis, float),
                case_id=bundle.name, target_fdi=meta.get("fdi"),
                source_asset_sha256=_sha256(support), margin_point_count=len(margin),
                indication=Indication.NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN.value,
                screw_channel_present=None,
                provenance={"source": "crown_outer_bundle", "support_name": support.name})
            entry["contract"] = contract.as_report()
        except Exception as exc:                                 # noqa: BLE001
            entry["contract"] = {"classification": {
                "readiness": Readiness.REVIEW_REQUIRED.value,
                "rejection_code": "ASSET_LOAD_FAILED",
                "rejection_reason": "%s: %s" % (type(exc).__name__, exc)}}
        results.append(entry)
    return results


def build_corpus_report(*, cases_root=None, bundles_root=None) -> dict:
    """生成 ``prep_conditioning_report.json`` 的全部内容（纯统计，无副作用）。"""
    report = {"generated_at": _now(), "scope": SCOPE_1_0,
              "bridge_rule": "与桥体**直接相邻**（同象限 ±1 牙位）的冠判为 BRIDGE_UNIT；"
                             "仅间接相邻者可能仍被计入，故 eligible_units 是**上界**",
              "total_cases": 0, "total_units": 0, "eligible_units": 0,
              "by_indication": {}, "contract_attempts": 0,
              "by_rejection_code": {}, "readiness": {},
              "eligible_without_asset": 0, "assets": {}, "cases": [], "bundles": []}
    if cases_root and Path(cases_root).is_dir():
        triage = triage_cases(cases_root)
        report["cases"] = triage["cases"]
        report["total_cases"] = len(triage["cases"])
        units = [u for case in triage["cases"] for u in case["units"]]
        report["total_units"] = len(units)
        report["eligible_units"] = sum(1 for u in units if u.get("eligible"))
        for unit in units:
            report["by_indication"][unit["indication"]] = \
                report["by_indication"].get(unit["indication"], 0) + 1
        # 元数据侧没有任何支撑网格可判 ⇒ 合格单位全部卡在"没有资产"这一步，
        # 这本身就是要报的结论（流程缺口，不是数据缺口）。
        report["eligible_without_asset"] = report["eligible_units"]
    if bundles_root and Path(bundles_root).is_dir():
        bundles = assess_bundle_dir(bundles_root)
        report["bundles"] = bundles
        for entry in bundles:
            contract = entry.get("contract") or {}
            classification = contract.get("classification") or {}
            if contract.get("geometry"):
                readiness = classification.get("readiness", Readiness.REVIEW_REQUIRED.value)
                report["readiness"][readiness] = report["readiness"].get(readiness, 0) + 1
                code = classification.get("rejection_code", "UNKNOWN")
                report["by_rejection_code"][code] = report["by_rejection_code"].get(code, 0) + 1
                report["contract_attempts"] += 1
            else:
                report["eligible_without_asset"] += 1
        report["assets"]["bundles_with_support_asset"] = sum(
            1 for e in bundles if (e.get("contract") or {}).get("geometry"))
        report["assets"]["bundles_total"] = len(bundles)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Prep Surface Contract 语料报表（只读，无副作用）")
    parser.add_argument("--cases", default=None, help="exocad 完整病例目录（逐牙位分诊）")
    parser.add_argument("--bundles", default=None, help="crown_outer 病例包目录（跑契约）")
    parser.add_argument("--out", default="prep_conditioning_report.json")
    args = parser.parse_args()
    report = build_corpus_report(cases_root=args.cases, bundles_root=args.bundles)
    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("total_cases", "total_units", "eligible_units",
                                      "by_indication", "by_rejection_code", "readiness", "assets")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("written: %s" % out)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
