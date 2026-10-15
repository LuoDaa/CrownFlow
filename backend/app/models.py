from pydantic import BaseModel, Field
from typing import Literal

SupportType = Literal["prep", "abutment"]
GeneratorMode = Literal["template", "ai"]

class CaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)

class DesignCreate(BaseModel):
    draft: bool = False
    case_id: str
    target_fdi: int
    support_type: SupportType
    indication: Literal['NATURAL_PREP','NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN','CROWN_ON_EXISTING_ABUTMENT','CUSTOM_IMPLANT_ABUTMENT','SCREW_RETAINED_CROWN'] | None = None
    generator_mode: GeneratorMode = "template"
    cement_gap_mm: float = Field(default=0.060, ge=0.0, le=0.5)
    margin_gap_mm: float = Field(default=0.0, ge=0.0, le=0.3)
    margin_band_mm: float = Field(default=0.8, gt=0.0, le=4.0)
    minimum_thickness_mm: float = Field(default=0.8, ge=0.2, le=5.0)
    voxel_pitch_mm: float = Field(default=0.10, ge=0.04, le=0.4)
    margin_extension_mm: float = Field(default=0.0, ge=-1.0, le=2.0)
    enforce_minimum_thickness: bool = False
    border_horizontal_mm: float = Field(default=0, ge=0, le=1)
    border_angled_mm: float = Field(default=0, ge=0, le=2)
    border_angle_deg: float = Field(default=45, gt=0, lt=90)
    border_vertical_mm: float = Field(default=0, ge=0, le=1)
    proximal_target_mm: float | None = Field(default=None, ge=0, le=.5)
    occlusal_target_mm: float | None = Field(default=None, ge=0, le=.5)
    # 加工/加工策略：缺失时由 blockout_policy_from_resolved 报 UNRESOLVED，
    # 不套用近似策略。
    machine_axes: int | None = None
    # Margin 周围不做 blockout 的保护带宽度（mm）。exocad 官方定义见 intaglio.py 文档。
    untouchable_zone_mm: float | None = Field(default=None, ge=0.0, le=2.0)
    # 紧邻 Margin 的一段不施加 cement gap（exocad CementGapStart 语义）
    cement_gap_start_mm: float | None = Field(default=None, ge=0.0, le=5.0)
    # 咬合面间隙。未显式给出则不加 —— 该参数名未见于本次上传的官方指南，
    # provenance 标为 constructionInfo_observed，不猜默认值。
    cement_gap_top_mm: float | None = Field(default=None, ge=0.0, le=0.5)

    # ── 参数驱动生成（CrownGenerationProfile）新增字段 ─────────────────────────
    # 纪律：**语义参数一律 None = MISSING**，由 profile 显式登记"缺哪个"，
    # 绝不在解析处 `or 0`（那会让"没填"和"填了 0"不可区分）。
    proximal_contact_area_mode: Literal['SMALL', 'MEDIUM', 'LARGE'] | None = None
    proximal_contact_area_target_mm2: float | None = Field(default=None, gt=0.0, le=60.0)
    proximal_mesial_enabled: bool | None = None
    proximal_distal_enabled: bool | None = None
    occlusal_enabled: bool | None = None
    occlusal_require_confirmed_bite_pose: bool | None = None
    #: 倒凹策略：REMOVE 用已确认就位轴自动 blockout；KEEP 不假装安全（warning + 禁止生产放行）
    undercut_policy: Literal['REMOVE', 'KEEP'] | None = None
    blockout_angle_deg: float | None = Field(default=None, ge=0.0, le=45.0)
    #: 颈缘过渡（工程级参数化，不是材料学）
    border_smooth_transition: bool | None = None
    border_blend_radius_mm: float | None = Field(default=None, ge=0.0, le=2.0)
    border_blend_width_mm: float | None = Field(default=None, ge=0.0, le=3.0)
    #: 车针/铣刀补偿（作用于内腔，不作用于外冠）
    tool_radius_mm: float | None = Field(default=None, ge=0.05, le=3.0)
    tool_compensation_enabled: bool | None = None
    #: 初始缩放（放置阶段；None = 用既有默认，不静默变 1.0）
    outer_uniform_scale: float | None = Field(default=None, gt=0.2, le=3.0)
    outer_md_scale: float | None = Field(default=None, gt=0.2, le=3.0)
    outer_bl_scale: float | None = Field(default=None, gt=0.2, le=3.0)
    outer_height_scale: float | None = Field(default=None, gt=0.2, le=3.0)

class RoiRequest(BaseModel):
    source: Literal["upper", "lower"]
    center: list[float]
    radius_mm: float = Field(gt=1.0, lt=30.0)

class MarginRequest(BaseModel):
    points: list[list[float]]


class AutoMarginM1Request(BaseModel):
    """M1 自动 Margin 提案（工作台）。``axis`` 省略时用侧壁拟合临时轴并如实标注。"""
    model_config = {"extra": "forbid"}
    axis: list[float] | None = None
    mode: Literal["SUPRA", "SUB"] = "SUPRA"
    radius_mm: float = Field(default=1.0, ge=0.1, le=5.0)
    apply_snap: bool = True


class MarginSnapRequest(BaseModel):
    """把当前边缘线按模式吸到凸棱/凹沟（exocad Supragingival / Subgingival）。"""
    model_config = {"extra": "forbid"}
    points: list[list[float]]
    axis: list[float] | None = None
    mode: Literal["SUPRA", "SUB"] = "SUPRA"
    radius_mm: float = Field(default=1.0, ge=0.1, le=5.0)


class SectionPreviewRequest(BaseModel):
    """悬停剖面气泡：给定表面点（与可选法向），返回一张剖面折线（3D + 2D）。"""
    model_config = {"extra": "forbid"}
    point: list[float]
    normal: list[float] | None = None
    span_mm: float = Field(default=8.0, ge=2.0, le=30.0)


class AxisRequest(BaseModel):
    margin_points: list[list[float]] | None = None


class AxisSetRequest(BaseModel):
    """人工确认/修改就位方向。必须进入 DesignRevision，不能只改数据库里一个向量。"""
    axis: list[float]
    note: str | None = None
    approved_by: str | None = None

class LandmarksRequest(BaseModel):
    patient_points: list[list[float]]
    template_points: list[list[float]]
    morph_points_patient: list[list[float]] = Field(default_factory=list)
    morph_points_template: list[list[float]] = Field(default_factory=list)
    morph_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    #: A1：地标驱动的外形生成已降为显式 legacy/研究路径，必须由调用方明确开启
    legacy_landmarks: bool = False

class AutoOuterRequest(BaseModel):
    margin_points: list[list[float]]
    insertion_axis: list[float]

class FinalizeRequest(BaseModel):
    model_config = {"extra": "forbid"}
    expected_input_digest: str = Field(min_length=64,max_length=64)
    margin_revision_id: str
    axis_revision_id: str
    profile_revision_id: str

class OcclusalAdaptRequest(BaseModel):
    """咬合适配请求：全部有默认值（前端一键即可用）。"""
    model_config = {"extra": "forbid"}
    target_mm: float | None = Field(default=None,gt=0,le=2)
    min_wall_mm: float = Field(default=0.5,gt=0,le=3)
    rebuild_final: bool = True
