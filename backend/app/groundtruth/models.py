"""真值索引的数据模型与 GT 集合规则。

**规则写死，数量不写死。** 每个 GT Set 都是对索引行的一个纯函数谓词；
"到底有多少例"必须由索引器算出来，不可以预先填一个数字。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from app.groundtruth.human_labels import (
    HUMAN_PREP_POSITIVE,
    LOCALIZATION_ROUTE_STATUS,
    human_prep_verdict,
)

#: 索引列顺序（也是 CSV 表头）。新增字段只能追加，不能插入或改名 ——
#: 否则已生成的索引与比较脚本会错位。
COLUMNS: tuple[str, ...] = (
    # --- 身份与分组
    "case_uuid",
    "source_case_id",
    "source_order_id",
    "fdi",
    "reconstruction_type",
    # --- constructionInfo
    "has_construction_info",
    "has_parameters",
    "parameter_count",
    # --- 颈缘线
    "has_margin",
    "margin_valid",
    "margin_point_count",
    "margin_perimeter_mm",
    "margin_max_segment_mm",
    # --- 就位方向
    "has_axis",
    "axis_valid",
    # --- 位置 / 牙架 / 库牙
    "has_center",
    "center",
    "has_tooth_frame",
    "has_tooth_library",
    "tooth_library_name",
    "tooth_model_matrix_valid",
    # --- 代型（support）
    "has_modeldie",
    "modeldie_asset_id",
    "modeldie_same_frame",
    "modeldie_center_distance_mm",
    #: 回退候选（bbox 中心离设计中心最近的 STL）—— **不是代型证据**，只作诊断。
    #: 用户决策 2026-09-30 ③：没有文件名/几何证据时绝不能把它命名成 modeldie。
    "nearest_asset_id",
    "nearest_asset_center_distance_mm",
    #: 资产形态（DIE_LIKE / CROWN_SHELL_INTAGLIO / UNCLEAR），来自网格形态审计
    "asset_kind",
    #: modeldie 的选择依据。NEAREST_TO_CENTER_FALLBACK 时 has_modeldie 恒为 False
    "asset_selection_method",
    # --- 整颌候选 + 承重假设
    "has_full_arch_candidate",
    "full_arch_asset_id",
    "full_arch_bbox_mm",
    "target_center_surface_distance_mm",
    "target_center_surface_coverage",
    "target_center_has_surface",
    "arch_is_sectioned_or_unsectioned",
    "arch_asset_class",
    # --- 对颌
    "has_antagonist",
    "antagonist_asset_id",
    # --- 最终冠
    "has_final_crown",
    "final_crown_asset_id",
    "final_crown_fdi_count",
    "final_crown_single_unit",
    "final_crown_watertight",
    "final_crown_euler",
    # --- 颈缘 ↔ 冠一致性
    "margin_crown_consistent",
    "margin_crown_p95_mm",
    # --- 设计快照来源（提交 2.2 的 provenance 审计产物）
    "design_source_kind",
    "design_source_asset_id",
    "coordinate_frame_kind",
    "geometry_validation_status",
    # --- 诊断
    "notes",
)


@dataclass
class ToothGroundTruth:
    """一行 = 一个「病例 × FDI」。

    指针字段一律只存 ``asset_id``（``内容哈希``）与**合成**路径，
    不存原始文件名、不存内网路径 —— 原文件名里实测出现过患者姓名。
    """

    case_uuid: str
    source_case_id: str
    source_order_id: str | None
    fdi: int
    reconstruction_type: str

    has_construction_info: bool = False
    has_parameters: bool = False
    parameter_count: int = 0

    has_margin: bool = False
    margin_valid: bool = False
    margin_point_count: int = 0
    margin_perimeter_mm: float | None = None
    #: 相邻控制点之间的最大间距。断开的环会出现超长段，这是最直接的坏样本信号。
    margin_max_segment_mm: float | None = None

    has_axis: bool = False
    axis_valid: bool = False

    has_center: bool = False
    center: tuple[float, float, float] | None = None
    has_tooth_frame: bool = False
    has_tooth_library: bool = False
    tooth_library_name: str | None = None
    tooth_model_matrix_valid: bool = False

    has_modeldie: bool = False
    modeldie_asset_id: str | None = None
    modeldie_same_frame: bool = False
    modeldie_center_distance_mm: float | None = None
    nearest_asset_id: str | None = None
    nearest_asset_center_distance_mm: float | None = None
    asset_kind: str | None = None
    asset_selection_method: str | None = None

    has_full_arch_candidate: bool = False
    full_arch_asset_id: str | None = None
    full_arch_bbox_mm: str | None = None
    target_center_surface_distance_mm: float | None = None
    target_center_surface_coverage: float | None = None
    target_center_has_surface: bool | None = None
    #: ``unsectioned`` / ``sectioned`` / ``unknown``
    arch_is_sectioned_or_unsectioned: str = "unknown"
    #: 来自 Arch Asset Audit 的资产语义类（FULL_ARCH_WITH_TARGET / FULL_ARCH_ANTAGONIST /
    #: FULL_ARCH_TARGET_MISSING / PARTIAL_ARCH / MULTI_UNIT_RESTORATION /
    #: SINGLE_TOOTH / UNKNOWN）。由 ``freeze.py`` 从审计 CSV 回填 ——
    #: **弓资产的判定只有审计那一处**，索引不另算一套。
    arch_asset_class: str = "UNKNOWN"

    has_antagonist: bool = False
    antagonist_asset_id: str | None = None

    has_final_crown: bool = False
    final_crown_asset_id: str | None = None
    final_crown_fdi_count: int = 0
    final_crown_single_unit: bool = False
    final_crown_watertight: bool | None = None
    final_crown_euler: int | None = None

    margin_crown_consistent: bool = False
    margin_crown_p95_mm: float | None = None

    # --- 设计快照 provenance（design-source audit 的结论，见 docs）----------
    # 一个 Tooth GT 必须来自**一个明确的设计快照**，绝不允许
    # "Margin 取自 A、Axis 取自 B" 的拼接真值。当前实现从结构上保证这一点：
    # 一行的 margin/axis/center/参数全部来自同一个被解析的文件。
    #: CONSTRUCTION_INFO | MODEL_INFO
    design_source_kind: str = "UNKNOWN"
    #: 来源文件的内容哈希（可追溯到资产，不含路径与文件名）
    design_source_asset_id: str | None = None
    #: EXPORT_CANONICAL | MODEL_INFO_NATIVE | MODEL_INFO_CANONICALIZED | UNKNOWN
    coordinate_frame_kind: str = "UNKNOWN"
    #: CANONICAL（ci，政策认定的几何 GT 帧）| VALIDATED / INCONSISTENT /
    #: UNVALIDATABLE_NO_CROWN / NOT_APPLICABLE
    geometry_validation_status: str = "NOT_APPLICABLE"

    notes: str = ""

    # ------------------------------------------------------------ 序列化

    def as_row(self) -> dict:
        """一行扁平记录。向量用 ``;`` 连接（避免 CSV 里的逗号歧义）。"""
        d = asdict(self)
        if self.center is not None:
            d["center"] = ";".join(f"{x:.4f}" for x in self.center)
        if isinstance(self.target_center_surface_distance_mm, float):
            d["target_center_surface_distance_mm"] = round(
                self.target_center_surface_distance_mm, 4)
        if isinstance(self.target_center_surface_coverage, float):
            d["target_center_surface_coverage"] = round(
                self.target_center_surface_coverage, 4)
        if isinstance(self.margin_crown_p95_mm, float):
            d["margin_crown_p95_mm"] = round(self.margin_crown_p95_mm, 4)
        if isinstance(self.margin_perimeter_mm, float):
            d["margin_perimeter_mm"] = round(self.margin_perimeter_mm, 4)
        if isinstance(self.margin_max_segment_mm, float):
            d["margin_max_segment_mm"] = round(self.margin_max_segment_mm, 4)
        if isinstance(self.modeldie_center_distance_mm, float):
            d["modeldie_center_distance_mm"] = round(
                self.modeldie_center_distance_mm, 4)
        if isinstance(self.nearest_asset_center_distance_mm, float):
            d["nearest_asset_center_distance_mm"] = round(
                self.nearest_asset_center_distance_mm, 4)
        return {k: d.get(k) for k in COLUMNS}

    @property
    def is_restorative(self) -> bool:
        return self.reconstruction_type in RESTORATIVE_TYPES

    @property
    def is_single_anatomic_crown(self) -> bool:
        return self.reconstruction_type == "AnatomicCrown"


# ---------------------------------------------------------------- 快照来源政策
#
# design-source audit（全量 3566 对配对拟合）的结论：
#   * modelInfo 的颈缘与 constructionInfo 的颈缘是**同一条折线**
#     （段长差 med 0.078mm），但二者**不是**刚体/相似/镜像关系
#     （rigid 残差 med 1.69mm，0/3566 对 <=0.01mm；|scale-1| med 0.125）；
#   * 拟合出的变换与 XML 里所有矩阵（MatrixToScanDataFiles / ToothModelMatrix /
#     ScanFiles.TransformationMatrix，两个方向）**精确匹配 0/3566**；
#   * modelInfo-only 病例的 418 个 Margin 牙位**全部没有**导出冠可验证；
#   * 参数键集一致仅 366/1573 —— 两者连参数快照都不同。
# 因此 `.modelInfo` 不是 `.constructionInfo` 的坐标重表达，而是另一个
# 设计阶段的快照：**几何 GT 只认 constructionInfo**；
# modelInfo 记录保留为 metadata/research candidate，不进几何 GT。

#: 几何 GT（颈缘/就位方向/中心）唯一认定的设计快照来源。
CANONICAL_DESIGN_SOURCE = "CONSTRUCTION_INFO"

#: 坐标帧语义。FULL_ARCH_WITH_TARGET 只表示"当前可观察 STL 几何中存在与目标牙
#: 一致的表面证据"，**不能**提升成"原始 IOS 未分割扫描" —— ci 里真正的
#: UnsegmentedJaw 引用的 PLY 在语料中全部缺失，现存 STL 是 exocad 后处理模型。
FRAME_EXPORT_CANONICAL = "EXPORT_CANONICAL"
FRAME_MODEL_INFO_NATIVE = "MODEL_INFO_NATIVE"
FRAME_MODEL_INFO_CANONICALIZED = "MODEL_INFO_CANONICALIZED"
FRAME_UNKNOWN = "UNKNOWN"


def is_geometric_gt_source(row: "ToothGroundTruth") -> bool:
    """该行是否可作为**几何**真值（颈缘/就位方向/中心）使用。

    政策（由上面的实测证据冻结）：只认 constructionInfo 快照。
    modelInfo 行要进入几何 GT，必须先经过显式的 canonical transform
    并通过验证 —— 而该变换目前不存在（XML 无出处、拟合不成立）。
    """
    return row.design_source_kind == CANONICAL_DESIGN_SOURCE


#: 需要修复的修复体类型（排除 Antagonist/HealthyTooth/Pontic 这类邻牙记录）。
RESTORATIVE_TYPES: frozenset[str] = frozenset({
    "AnatomicCrown", "Coping", "PrimaryTelescope", "Veneer",
    "AnatomicInlay", "OffsetCoping",
})


# ---------------------------------------------------------------- 有效性判据

def is_margin_valid(row: ToothGroundTruth) -> bool:
    """颈缘线是否可用。

    判据（都是可复核的几何量，不是"看着像"）：
      * 点数 ≥ 20（实测真实颈缘 200~530 点）
      * 周长 ≥ 5 mm（一颗磨牙颈缘周长 ~25 mm，5 mm 已是量级下限）
      * 最大段长 ≤ 3 mm（断开的环会有超长段）
    单位是 mm，全部来自设计坐标系，不做任何缩放假设。
    """
    return (row.has_margin
            and row.margin_point_count >= 20
            and (row.margin_perimeter_mm or 0.0) >= 5.0
            and (row.margin_max_segment_mm or 1e9) <= 3.0)


def add_note(row: ToothGroundTruth, text: str) -> None:
    """往 ``notes`` 里追加一条诊断（分号分隔，保持单列）。"""
    parts = [p for p in row.notes.split(";") if p]
    parts.append(text)
    row.notes = ";".join(parts)


# ---------------------------------------------------------------- GT 集合

def margin_gt_set(rows) -> list[ToothGroundTruth]:
    """可做 Auto Margin 的牙位：修复体 + 有效颈缘线 + **快照来源合规**。"""
    return [r for r in rows if r.is_restorative and is_margin_valid(r)
            and is_geometric_gt_source(r)]


def axis_gt_set(rows) -> list[ToothGroundTruth]:
    """可做 Insertion Axis 评测的牙位：颈缘线 + 合法就位方向。"""
    return [r for r in margin_gt_set(rows) if r.has_axis and r.axis_valid]


#: ``LocalizationGTSet`` 的自动预筛值。
#:
#: 只允许 FULL_ARCH_WITH_TARGET。明确排除 ``FULL_ARCH_ANTAGONIST``、
#: ``FULL_ARCH_TARGET_MISSING``、``PARTIAL_ARCH`` / ``UNKNOWN``。
LOCALIZATION_ADMITTED_CLASS = "FULL_ARCH_WITH_TARGET"

#: **准入的最终判据是人工判定，不是上面那个自动值。**
#:
#: 自动判据（v2 "Center 附近有表面" → v3.2 格搜索 + margin 轮廓）经两轮人工
#: QA 证伪：v3.2 的最高峰约 79% 落在目标牙自身 margin 轮廓之外，附加的
#: 肩台/覆盖条件没有区分度。所以 ``FULL_ARCH_WITH_TARGET`` 降级为**必要不充分**
#: 的预筛，只有人工判为 A（预备体确实在，且峰落在它上面）才准入。
#: 见 ``app/groundtruth/human_labels.py`` 与 ``docs/LOCALIZATION_QA2.md``。
LOCALIZATION_REQUIRES_HUMAN_PREP = True


def human_prep_label(row: "ToothGroundTruth") -> str | None:
    """该行的人工 Prep 判定标签（A/B/C/U），无判定或跨轮次冲突时为 ``None``。"""
    v = human_prep_verdict(row.case_uuid, row.fdi)
    return v.label if v is not None else None


def localization_gt_set(rows) -> list[ToothGroundTruth]:
    """可做 ``Full Jaw → Target Tooth`` 的样本。

    **两个条件都要满足**（顺序即理由）：

    1. 自动预筛 ``arch_asset_class == FULL_ARCH_WITH_TARGET`` —— 只排除
       对颌弓、sectioned base、非整颌这类"结构上根本不可能"的样本；
    2. **人工判定 == A** —— 因为 (1) 的精度实测只有约 0.17~0.2，
       仅凭几何形态无法把天然牙尖/邻牙/底座与预备体分开。

    因此这个集合现在很小，且**只用于回归与人工验证过的参考定位**，
    不作为训练集使用。
    """
    return [r for r in axis_gt_set(rows)
            if r.has_center and r.has_full_arch_candidate
            and r.arch_asset_class == LOCALIZATION_ADMITTED_CLASS
            and (not LOCALIZATION_REQUIRES_HUMAN_PREP
                 or human_prep_label(r) == HUMAN_PREP_POSITIVE)]


def localization_exclusion_reasons(rows) -> dict:
    """``LocalizationGTSet`` 的排除原因计数 —— 回答「672 例为什么只剩 N 例」。

    顺序即判定的先后：先过 ``AxisGTSet`` 的门，再逐条排除。
    """
    out: dict[str, int] = {}
    for r in rows:
        if not (r.is_restorative and is_margin_valid(r)):
            out["excluded_not_margin_gt"] = out.get("excluded_not_margin_gt", 0) + 1
        elif not (r.has_axis and r.axis_valid):
            out["excluded_no_valid_axis"] = out.get("excluded_no_valid_axis", 0) + 1
        elif not r.has_center:
            out["excluded_missing_center"] = out.get("excluded_missing_center", 0) + 1
        elif not is_geometric_gt_source(r):
            # margin_gt_set 的第三道门。**必须单独列出来**：漏掉它会让
            # admitted 比 LocalizationGTSet 实际大小多出 modelInfo 行的数量，
            # 报告与实际集合对不上（回归测试 test_reasons_reconcile_with_set 钉住）。
            out["excluded_not_geometric_gt_source"] = \
                out.get("excluded_not_geometric_gt_source", 0) + 1
        elif not r.has_full_arch_candidate:
            out["excluded_no_arch_candidate"] = out.get("excluded_no_arch_candidate", 0) + 1
        elif r.arch_asset_class == "FULL_ARCH_ANTAGONIST":
            out["excluded_antagonist"] = out.get("excluded_antagonist", 0) + 1
        elif r.arch_asset_class == "FULL_ARCH_TARGET_MISSING":
            out["excluded_target_missing"] = out.get("excluded_target_missing", 0) + 1
        elif r.arch_asset_class == "PARTIAL_ARCH":
            out["excluded_partial_arch"] = out.get("excluded_partial_arch", 0) + 1
        elif r.arch_asset_class == "UNKNOWN":
            out["excluded_unknown"] = out.get("excluded_unknown", 0) + 1
        elif r.arch_asset_class != LOCALIZATION_ADMITTED_CLASS:
            out["excluded_other_class"] = out.get("excluded_other_class", 0) + 1
        else:
            # 自动预筛通过后的去向，全部由人工判定决定。
            label = human_prep_label(r)
            if label is None:
                out["excluded_no_human_verdict"] = \
                    out.get("excluded_no_human_verdict", 0) + 1
            elif label == HUMAN_PREP_POSITIVE:
                out["admitted"] = out.get("admitted", 0) + 1
            else:
                key = "excluded_human_label_" + label
                out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def support_gt_set(rows) -> list[ToothGroundTruth]:
    """可做 Support 分割的牙位：**有证据的**代型且与设计同坐标系。

    注意（2026-09-30 起）：``has_modeldie`` 只对有文件名或几何证据的资产为真；
    indexer 的回退候选写在 ``nearest_asset_id``，**不再**冒充代型。所以本集合
    在 v1.1 里会显著收窄 —— 这是修正，不是退化。
    """
    return [r for r in axis_gt_set(rows)
            if r.has_modeldie and r.modeldie_same_frame]


def outer_crown_gt_set(rows) -> list[ToothGroundTruth]:
    """可做 Crown Outer 真值的牙位：存在导出冠且几何自洽。"""
    return [r for r in axis_gt_set(rows)
            if r.has_final_crown and r.final_crown_watertight
            and r.margin_crown_consistent]


def single_anatomic_crown_gt_set(rows) -> list[ToothGroundTruth]:
    """**最重要的一个集合**：单单位解剖全冠的完整真值。

    规则（写死，数量由索引器算）：

        ReconstructionType == AnatomicCrown
        AND 该冠 STL 只对应 1 个 FDI
        AND Margin 有效 AND Axis 有效
        AND Final Crown 存在且水密
        AND Margin 与 Crown 同一坐标系
        AND Support / modeldie 可用
    """
    return [r for r in rows
            if r.reconstruction_type == "AnatomicCrown"
            and is_geometric_gt_source(r)
            and r.final_crown_fdi_count == 1
            and r.final_crown_single_unit
            and is_margin_valid(r)
            and r.has_axis and r.axis_valid
            and r.has_final_crown
            and r.final_crown_watertight is True
            and r.margin_crown_consistent
            and r.has_modeldie and r.modeldie_same_frame]


def end_to_end_gt_set(rows) -> list[ToothGroundTruth]:
    """端到端（整颌 → 定位 → support → margin → axis → outer）都齐的样本。

    **同样要求人工 Prep == A**：这个集合声称"端到端"就包含定位环节，
    而定位只能靠人工确认。少了这一条，它就会声称一个已经证伪的环节是通的。
    """
    return [r for r in single_anatomic_crown_gt_set(rows)
            if r.has_full_arch_candidate
            and r.arch_asset_class == LOCALIZATION_ADMITTED_CLASS
            and human_prep_label(r) == HUMAN_PREP_POSITIVE
            and r.has_antagonist]


#: 集合名 -> 谓词。CLI 与报告都从这里取，保证只有一处定义。
GT_SETS: dict[str, callable] = {
    "MarginGTSet": margin_gt_set,
    "AxisGTSet": axis_gt_set,
    "LocalizationGTSet": localization_gt_set,
    "SupportGTSet": support_gt_set,
    "OuterCrownGTSet": outer_crown_gt_set,
    "SingleAnatomicCrownGTSet": single_anatomic_crown_gt_set,
    "EndToEndGTSet": end_to_end_gt_set,
}


def build_all_sets(rows) -> dict[str, list[ToothGroundTruth]]:  # noqa: E302
    return {name: fn(rows) for name, fn in GT_SETS.items()}


def case_group_key(row: ToothGroundTruth) -> str:
    """训练/验证拆分必须用的分组键。

    同一患者的多颗牙（同一订单）必须落在同一个 split，否则同一患者
    既进 train 又进 test，指标是假的。优先用 ``source_order_id``，
    没有就退到 ``case_uuid``。
    """
    return row.source_order_id or row.case_uuid
