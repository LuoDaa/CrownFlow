"""CrownOuter 的数据模型与配置。

**配置纪律（方案 §12.2）**：所有数值型算法参数放进 ``CrownOuterConfig``，
不允许散落在函数体里。真实回归完成前全部标记 ``PROVISIONAL``，
不包装成临床常量。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .template import CrownTemplate  # noqa: F401  (re-export)

#: CrownOuter v0 的状态（fail-loud，见方案 §10.4）。
STATUS_OK = "OK"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
STATUS_FAILED = "FAILED"


@dataclass
class CrownOuterConfig:
    """CrownOuter v0 的全部可调参数。**真实回归前都是 PROVISIONAL。**

    权重不在方案文档里伪造，由合成网格 + 真实 FDI46 回归确定；
    改任何一项都必须在回归结果里留下版本痕迹。
    """

    #: 闭环重采样工作分辨率（方案 §8.1，首版 256）。
    resample_n: int = 256

    #: H.1 Crown Body Scale Decoupling。默认保留 H 的均匀缩放，保证旧结果
    #: 可复现；18 例对照验证后再翻转默认值。
    #: 效果：margin 周长比只负责颈部适配（颈环处 w≡1），随顶点到颈环的
    #: 距离线性衰减（轴壁中段 w≈0.5），咬合面保持牙库原始尺寸（w=0）。
    body_scale_decoupling: bool = False

    #: 缩放衰减特征长度（mm）。``None`` = 用模板冠高。
    scale_falloff_mm: float | None = None  # PROVISIONAL

    #: 牙颈环 / 根部是**硬约束**（行替换，见 laplacian.py 文档），
    #: 没有权重可调；这里保留字段只为配置快照的稳定性。
    weight_margin: float = -1.0        # 硬约束（行替换），无权重
    weight_root_anchor: float = -1.0   # 硬约束（行替换），无权重

    #: 咬合高位区**弱锚**相对权重（相对 delta 行 O(1) 的量级）。
    #: 真实 FDI46 回归的实测依据：0 太软（内部 11mm 过冲、39% 翻转），
    #: 0.3 时 0% 翻转、内部位移衰减正常。
    weight_occlusal_anchor: float = 0.3  # PROVISIONAL

    #: **H.2.1** proximal clearance 约束权重（低于 margin 的 weight_hard 1e4，
    #: 保证"margin 最高优先级"）。仅当调用方传入 active set 时才生效。
    weight_proximal_clearance: float = 300.0  # PROVISIONAL

    #: 咬合锚区定义：冠部 z 高于该分位的顶点视为咬合区。
    occlusal_anchor_percentile: float = 75.0  # PROVISIONAL

    #: 硬约束残差上限（mm）。CrownOuter 路径实测 0.00188，0.01 合适；
    #: I.1 精修约束顶点多（~1300）且带弱锚超定，lsqr 折中位到 0.04，
    #: 那里放宽到 0.05。**不做成一刀切**：两条路径的规模差一个量级。
    max_constraint_residual_mm: float = 0.01

    #: 区域带宽：距牙颈环顶点小于该值算 cervical 区（PROVISIONAL）。
    cervical_band_mm: float = 2.0

    #: 走向判定用分位（模板 +Z 核对，见 template._orientation_check）。
    orientation_band_percentile: float = 5.0

    #: QC 阈值（PROVISIONAL，真实回归后修订）：
    #: 颈缘贴合 p95 超过它 → REVIEW_REQUIRED。
    qc_margin_p95_mm: float = 0.30
    #: 咬合区位移 p95 相对颈区 p95 的最大比值，超过 → REVIEW_REQUIRED。
    qc_occlusal_motion_ratio: float = 0.5
    #: 法向翻转面占比超过该值 → REVIEW_REQUIRED（>5% → FAILED）。
    qc_flipped_faces_ratio: float = 0.01

    def as_dict(self) -> dict:
        out = {}
        for k in sorted(self.__dataclass_fields__):
            out[k] = getattr(self, k)
        out["_provisional"] = True
        return out


@dataclass
class PatientCrownContext:
    """患者牙冠上下文（方案 §4.2）。

    **坐标系要求**：Margin、Axis、Final Crown GT 必须来自同一 canonical
    geometry frame。正式几何 GT 只认 constructionInfo（见
    ``app/groundtruth/models.CANONICAL_DESIGN_SOURCE``），
    禁止把 modelInfo 的 Margin/Axis 与导出冠混拼。
    """

    case_uuid: str
    fdi: int
    margin_points: np.ndarray          # (M, 3) 有序闭环
    insertion_axis: np.ndarray         # (3,) 将被归一化
    center: np.ndarray | None = None
    final_crown_gt_path: str | None = None
    mesial_neighbor=None               # 本轮不做接触
    distal_neighbor=None
    antagonist=None

    def __post_init__(self) -> None:
        self.margin_points = np.asarray(self.margin_points, dtype=np.float64)
        if self.margin_points.ndim != 2 or self.margin_points.shape[1] != 3:
            raise ValueError("margin_points 必须是 (M, 3)")
        if len(self.margin_points) < 8:
            raise ValueError("margin_points 少于 8 个点，不构成可用闭环")
        self.insertion_axis = np.asarray(self.insertion_axis, dtype=np.float64)
        n = float(np.linalg.norm(self.insertion_axis))
        if n < 1e-9:
            raise ValueError("insertion_axis 是零向量")
        self.insertion_axis = self.insertion_axis / n


@dataclass
class CrownOuterResult:
    """CrownOuter v0 的输出（方案 §4.3）。"""

    mesh=None                          # trimesh.Trimesh，冠部外表面（含颈环）
    placement_transform: np.ndarray | None = None   # 4x4
    uniform_scale: float | None = None
    boundary_shift: int | None = None
    boundary_reversed: bool | None = None
    margin_error_median_mm: float | None = None
    margin_error_p95_mm: float | None = None
    margin_error_max_mm: float | None = None
    displacement_by_region: dict = field(default_factory=dict)
    mesh_qc: dict = field(default_factory=dict)
    gt_metrics: dict | None = None
    status: str = STATUS_FAILED
    notes: list = field(default_factory=list)
    #: I.1 精修的逐轮历史（accepted / 几何量），用于审计与调参。
    history: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "status": self.status,
            "uniform_scale": (round(self.uniform_scale, 4)
                              if self.uniform_scale else None),
            "boundary_shift": self.boundary_shift,
            "boundary_reversed": self.boundary_reversed,
            "margin_error_mm": {
                "median": round(self.margin_error_median_mm, 4),
                "p95": round(self.margin_error_p95_mm, 4),
                "max": round(self.margin_error_max_mm, 4),
            } if self.margin_error_median_mm is not None else None,
            "displacement_by_region": self.displacement_by_region,
            "mesh_qc": self.mesh_qc,
            "gt_metrics": self.gt_metrics,
            "notes": self.notes,
        }
