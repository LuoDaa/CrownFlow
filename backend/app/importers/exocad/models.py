"""exocad 导入器数据模型。

核心是参数的三态表达：exocad 的 ``.constructionInfo`` 只写出"被显式设定或偏离默认"
的参数，所以"参数没写出来"和"参数写了 0"必须是两种不同的东西，否则未知会被静默当成 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ParameterState(str, Enum):
    """单个参数的取值状态。四态，不可合并。"""

    #: exocad 明确写出了这个值，且语义无歧义，可以直接参与计算。
    EXPLICIT = "EXPLICIT"

    #: exocad 写出了值，但该值在临床上不可信，必须先人工定性才能参与自动设计。
    #: 典型：``CementGapThickness = 0``（0 粘接间隙意味着冠与预备体直接贴合，无法就位）。
    EXPLICIT_BUT_UNRESOLVED = "EXPLICIT_BUT_UNRESOLVED"

    #: 该牙位根本没有写出这个参数。含义是"exocad 内部默认值"，该默认值不在文件里，
    #: 因此是**未知**，不是 0，也不是任何可由本系统假定的数值。
    MISSING_INTERNAL_DEFAULT_UNKNOWN = "MISSING_INTERNAL_DEFAULT_UNKNOWN"

    #: exocad 写出了 NaN。表达"此项无约束"，不应被替换成任何数值。
    NO_CONSTRAINT = "NO_CONSTRAINT"


class ParameterScope(str, Enum):
    """参数在 exocad 工程中的作用域，决定它应该落在哪一层配置里。"""

    TOOTH = "TOOTH"          # 逐牙位
    CASE = "CASE"            # 逐病例
    ORDER = "ORDER"          # 逐订单（含加工设备、托盘等）
    RESTORATION = "RESTORATION"  # 按修复体类型
    LAB = "LAB"              # 实验室级
    SYSTEM = "SYSTEM"        # 系统级


class ReconstructionType(str, Enum):
    """exocad 记录的修复体重建类型。实测出现的全部取值。"""

    ANATOMIC_CROWN = "AnatomicCrown"
    COPING = "Coping"
    ANATOMIC_PONTIC = "AnatomicPontic"
    REDUCED_PONTIC = "ReducedPontic"
    ANTAGONIST = "Antagonist"
    HEALTHY_TOOTH = "HealthyTooth"
    UNKNOWN = "Unknown"

    @classmethod
    def parse(cls, raw: str | None) -> "ReconstructionType":
        if not raw:
            return cls.UNKNOWN
        try:
            return cls(raw)
        except ValueError:
            return cls.UNKNOWN


class FunctionalRole(str, Enum):
    """该牙位记录在本病例中扮演的角色。

    由 ReconstructionType 推导。``ANTAGONIST`` 是 exocad 记录"哪颗牙被它当作对颌"，
    可直接作为对颌识别算法的真值标签，与"上下颌咬合空间变换"是**两件不同的事**。
    """

    RESTORATION = "RESTORATION"      # 需要设计的修复体
    ANTAGONIST = "ANTAGONIST"        # 对颌牙（识别真值）
    REFERENCE = "REFERENCE"          # 邻牙 / 健康牙 / 其他参考几何
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_reconstruction_type(cls, rt: ReconstructionType) -> "FunctionalRole":
        if rt in (ReconstructionType.ANATOMIC_CROWN, ReconstructionType.COPING,
                  ReconstructionType.ANATOMIC_PONTIC, ReconstructionType.REDUCED_PONTIC):
            return cls.RESTORATION
        if rt is ReconstructionType.ANTAGONIST:
            return cls.ANTAGONIST
        if rt in (ReconstructionType.HEALTHY_TOOTH,):
            return cls.REFERENCE
        return cls.UNKNOWN


@dataclass(frozen=True)
class ParameterCell:
    """一个参数在一个牙位记录上的原始取值。

    绝不提供 ``float(cell)`` 或 ``cell or 0`` 这类语义化访问；取值必须走
    ``.value`` 并同时检查 ``.state``。这是为了防止"未知变零"。
    """

    name: str
    state: ParameterState
    value: float | None
    source: str
    scope: ParameterScope
    raw_text: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    #: 该值为何被标为 EXPLICIT_BUT_UNRESOLVED，供人工复核时显示。
    unresolved_reason: str | None = None

    @property
    def usable_for_auto_design(self) -> bool:
        return self.state is ParameterState.EXPLICIT

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "value": self.value,
            "source": self.source,
            "scope": self.scope.value,
            "raw_text": self.raw_text,
            "provenance": dict(self.provenance),
            "unresolved_reason": self.unresolved_reason,
        }


@dataclass
class ExocadToothRecord:
    """``.constructionInfo`` 里的一条 ``<Tooth>`` 记录。

    ``parameters`` 只包含**被写出的**参数。exocad 不为每个牙位写全部参数
    （实测每牙位 47–134 个不等），所以"某参数不在 ``parameters`` 里"就是
    ``MISSING_INTERNAL_DEFAULT_UNKNOWN``。要拿到这个四态结果请用 :meth:`parameter`，
    不要直接 ``parameters.get(name)`` 然后当 None 处理。
    """

    source_file: str
    source_order_id: str | None
    case_dir_name: str
    tooth_fdi: int | None
    reconstruction_type: ReconstructionType
    functional_role: FunctionalRole
    parameters: dict[str, ParameterCell] = field(default_factory=dict)

    # --- 设计几何（实测存在于 .constructionInfo，本轮起解析）----------------
    #
    # 这些不是"参数"，而是 exocad 记录下来的**设计结果**：
    # 技师画的那条颈缘线、他确认的就位方向、他把库牙放进患者空间的变换。
    # 它们是 Auto Margin / Insertion Axis / CrownOuter 的**真值来源**。
    #
    # 旧文件可能没有这些元素，所以全部默认空 —— 空表示"文件里没写"，
    # 不表示"值为 0"。判定可用性用 :attr:`has_design_geometry`。

    #: 颈缘线控制点（三维闭合环；实测 200~530 点，段长均匀）。
    margin_points: list[list[float]] = field(default_factory=list)
    margin_bottom_type: str | None = None
    #: 技师确认的就位方向。**与 ConstructionFileList 里的
    #: ``EstimatedDirectionForCAM`` 不是同一语义**（一个是设计就位、一个是加工方向），
    #: 不要混用 —— 实测同一病例里两者可以不同。
    insertion_axis: list[float] | None = None
    #: 完整正交牙架：牙合面方向 / 近中方向 / 颊侧方向。
    axis_occlusal_of_tooth_model: list[float] | None = None
    axis_mesial: list[float] | None = None
    axis_buccal: list[float] | None = None
    #: 该牙在设计坐标系里的中心。目标牙定位（"在整颌里找 46"）的真值。
    center: list[float] | None = None
    #: exocad 选用的是哪个库里的哪颗牙，以及"库牙 → 患者空间"的 4x4 变换。
    #: 这是判断"我们手上的 32 颗模板是否就是 exocad 用的那套"的关键依据，
    #: 但**必须先做几何验证** —— 名字像不能当同源证据。
    tooth_library_name: str | None = None
    tooth_library_tooth_number: str | None = None
    tooth_model_matrix: list[float] | None = None
    tooth_scan_file_name: str | None = None

    @property
    def has_design_geometry(self) -> bool:
        """是否拿到了可用于真值的设计几何（颈缘线 + 就位方向）。"""
        return len(self.margin_points) >= 3 and self.insertion_axis is not None

    @property
    def written_names(self) -> list[str]:
        return sorted(self.parameters)

    @property
    def written_parameter_count(self) -> int:
        return len(self.parameters)

    def parameter(self, name: str) -> ParameterCell:
        """取参数；未写出时返回 ``MISSING_INTERNAL_DEFAULT_UNKNOWN`` 单元。

        绝不返回 ``None``，也绝不返回带数值的单元 —— 未知不能被消费成 0。
        """
        cell = self.parameters.get(name)
        if cell is not None:
            return cell
        return ParameterCell(
            name=name,
            state=ParameterState.MISSING_INTERNAL_DEFAULT_UNKNOWN,
            value=None,
            source="constructionInfo",
            scope=ParameterScope.TOOTH,
            raw_text=None,
            provenance={
                "order_id": self.source_order_id,
                "fdi": self.tooth_fdi,
                "file": Path(self.source_file).name,
            },
            unresolved_reason=(
                "该牙位未写出此参数。含义是 exocad 内部默认值，该默认值不在文件里，"
                "因此未知；不得当作 0 或任何假定值参与自动设计。"
            ),
        )

    @property
    def unresolved_parameters(self) -> list[ParameterCell]:
        return [c for c in self.parameters.values()
                if c.state is ParameterState.EXPLICIT_BUT_UNRESOLVED]


@dataclass
class ExocadCase:
    """一个 exocad 病例的完整导入结果（跨多个来源文件的聚合）。"""

    #: 系统内主键。与任何 PHI 无关。
    case_uuid: str
    #: 外部病例分组键。训练/验证拆分必须 group by 它，否则同一订单会跨集泄露。
    source_order_id: str | None
    case_dir_name: str
    teeth: list[ExocadToothRecord] = field(default_factory=list)
    restoration_note: str | None = None
    material_note: str | None = None
    teeth_from_notes: list[int] = field(default_factory=list)
    scan_source: str | None = None
    three_shape_order_no: str | None = None
    #: 被识别为 PHI、需要脱敏的字段，形如 {"field": 值}。
    phi_fields: dict[str, str] = field(default_factory=dict)
    #: 被识别为 PHI 的路径（相对病例目录）。
    phi_paths: list[str] = field(default_factory=list)
    #: .matrix4 的角色判定结果。
    matrix4_role: str | None = None

    @property
    def restoration_teeth(self) -> list[ExocadToothRecord]:
        return [t for t in self.teeth if t.functional_role is FunctionalRole.RESTORATION]

    @property
    def antagonist_teeth(self) -> list[ExocadToothRecord]:
        """exocad 明确标记为对颌的牙位 —— 对颌识别算法的现成真值。"""
        return [t for t in self.teeth if t.functional_role is FunctionalRole.ANTAGONIST]

    def tooth(self, fdi: int) -> ExocadToothRecord | None:
        for t in self.teeth:
            if t.tooth_fdi == fdi:
                return t
        return None


# ---------------------------------------------------------------- 病例级设计记录

@dataclass
class ExocadScanFile:
    """``<ScanFiles><ScanFile>`` —— 这个病例用了哪些扫描、各自扮演什么角色。

    ``part_type`` 是**对颌/预备体颌识别的现成真值**：实测取值有
    ``UnsegmentedJaw``（未分割整颌，即预备体所在的颌）与 ``Antagonist``。
    """
    file_name: str | None
    part_type: str | None
    tooth_numbers: list[int] = field(default_factory=list)
    transformation_matrix: list[float] | None = None
    file_hash: str | None = None
    loaded_by: str | None = None
    part_name: str | None = None


@dataclass
class ExocadConstructionFile:
    """``<ConstructionFileList><ConstructionFile>`` —— 导出了哪些加工件。

    ``filename`` 是**导出的 CAD STL 的文件名**，这是把
    "constructionInfo 里的某个牙位" 和 "all_data 目录里的某个 ``*_cad.stl``"
    对应起来的唯一可靠依据（实测：文件名里带 FDI，桥件则带一串 FDI）。
    """
    filename: str | None
    material: str | None = None
    material_name: str | None = None
    material_postfix: str | None = None
    material_color: str | None = None
    height: str | None = None
    #: 该导出件覆盖的牙位。桥件会是多个。
    tooth_numbers: list[int] = field(default_factory=list)
    reconstruction_type_hash: str | None = None
    z_rotation_matrix: list[float] | None = None
    #: **加工方向**，不是就位方向。不要拿去和 ``<Axis>`` 比。
    estimated_direction_for_cam: list[float] | None = None
    is_screw_retained: bool | None = None
    export_information: str | None = None
    save_time: str | None = None

    @property
    def looks_like_cad_export(self) -> bool:
        """文件名是否像 exocad 的 CAD 导出（``*_cad.stl``）。

        实测命名：``crown_cad`` / ``coping_cad`` / ``telescope_cad`` /
        ``bridge_slm_cad`` / ``inlay_cad`` / ``pontic_cad``。
        """
        n = (self.filename or "").lower()
        return n.endswith(".stl") and "_cad" in n


@dataclass
class ExocadConstructionCase:
    """一个 ``.constructionInfo`` 的**完整**解析结果。

    比 :func:`parse_construction_info` 的返回值多出病例级的扫描清单、
    导出清单、坐标系变换与身份字段。``teeth`` 与旧接口返回的是同一批对象。
    """

    source_file: str
    case_dir_name: str
    teeth: list[ExocadToothRecord] = field(default_factory=list)
    scan_files: list[ExocadScanFile] = field(default_factory=list)
    construction_files: list[ExocadConstructionFile] = field(default_factory=list)
    #: 设计坐标系 → 扫描坐标系 的 4x4 变换。
    matrix_to_scan_data_files: list[float] | None = None
    #: 下面这些是**身份字段，属于 PHI**，单独收集，不要混进正常字段里传播。
    #: 实测有值：PatientId（门诊内部患者号）、PracticeId（门诊号）、
    #: ProjectDir（会泄露内网服务器路径）。
    patient_id: str | None = None
    practice_id: str | None = None
    technician_id: str | None = None
    project_name: str | None = None
    project_dir: str | None = None
    design_guid: str | None = None
    #: 逐元素的解析诊断。单个元素坏掉不影响整份文件，但必须留痕：
    #: 例如 ``fdi_17_margin -> {"present": True, "n_total": 434, "n_bad": 0}``。
    diagnostics: dict[str, dict] = field(default_factory=dict)

    @property
    def phi_fields(self) -> dict[str, str]:
        """需要脱敏的身份字段。空值不出现在结果里。"""
        raw = {
            "PatientId": self.patient_id,
            "PracticeId": self.practice_id,
            "TechnicianId": self.technician_id,
            "ProjectName": self.project_name,
            "ProjectDir": self.project_dir,
            "DesignGUID": self.design_guid,
        }
        return {k: v for k, v in raw.items() if v}

    @property
    def unsegmented_jaw_scan(self) -> ExocadScanFile | None:
        """预备体所在的颌（exocad 记作 ``UnsegmentedJaw``）。"""
        for s in self.scan_files:
            if (s.part_type or "").strip().lower() == "unsegmentedjaw":
                return s
        return None

    @property
    def antagonist_scan(self) -> ExocadScanFile | None:
        for s in self.scan_files:
            if (s.part_type or "").strip().lower() == "antagonist":
                return s
        return None

    @property
    def restoration_teeth(self) -> list[ExocadToothRecord]:
        return [t for t in self.teeth
                if t.functional_role is FunctionalRole.RESTORATION]

    @property
    def teeth_with_design_geometry(self) -> list[ExocadToothRecord]:
        return [t for t in self.teeth if t.has_design_geometry]

    def tooth(self, fdi: int) -> ExocadToothRecord | None:
        for t in self.teeth:
            if t.tooth_fdi == fdi:
                return t
        return None
