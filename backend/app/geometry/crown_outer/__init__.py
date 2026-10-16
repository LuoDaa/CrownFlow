"""Crown Outer：从标准牙库模板生成患者个性化牙冠外表面。

链路（方案 docx v1.0 / 用户规格的 CrownOuter v0）：

    标准模板 → 模板局部系 → 摆到患者牙位 → 整体均匀缩放
    → 模板牙颈边界 → 患者 Margin（弧长对应）→ Laplacian 保形形变 → Crown Outer v0

本轮暂时**不处理**：邻牙接触、对颌咬合、厚度、Intaglio、AI（方案 §2.2）。
先把一个问题单独做对：**标准 46 牙冠能不能自动变成适合这个患者 Margin 的 46 牙冠。**

提交计划对应：A 牙库集成（importers/tooth_library）、B 模板标准化（template.py）、
C 患者系+摆位（patient_frame.py / placement.py）、D 闭环对应（boundary.py）、
E Laplacian v0.1（laplacian.py）、F 生成器+QC（generator.py / quality.py）、
G GT 回归（gt_compare.py，acceptance-only）。
"""
from __future__ import annotations

from .contact_refine import (
    RefineParams,
    refine_contacts,
)
from .contact import (
    ContactError,
    ContactParameters,
    ContactResult,
    NeighborPatch,
    SideContactReport,
    optimize_contacts,
)
from .boundary import (
    BoundaryError,
    align_rings,
    ensure_closed_ring_order,
    resample_ring,
    ring_arc_params,
    targets_for_loop_vertices,
)
from .gt_compare import (
    GtExtractError,
    extract_outer_surface,
    sample_surface_points,
    surface_distance_metrics,
)
from .generator import generate_crown_outer_v0
from .laplacian import LaplacianError, cotangent_laplacian, solve_laplacian
from .models import (
    STATUS_FAILED,
    STATUS_OK,
    STATUS_REVIEW_REQUIRED,
    CrownOuterConfig,
    CrownOuterResult,
    PatientCrownContext,
)
from .patient_frame import PatientFrame, PatientFrameError, build_patient_frame
from .placement import Placement, place_template, patient_margin_perimeter
from .template import (
    CrownTemplate,
    TemplateError,
    component_labels,
    load_crown_template,
    split_crown_and_root,
)

__all__ = [
    # models
    "CrownOuterConfig", "CrownOuterResult", "PatientCrownContext",
    "STATUS_OK", "STATUS_REVIEW_REQUIRED", "STATUS_FAILED",
    # template
    "CrownTemplate", "TemplateError", "component_labels",
    "load_crown_template", "split_crown_and_root",
    # patient frame / placement
    "PatientFrame", "PatientFrameError", "build_patient_frame",
    "Placement", "place_template", "patient_margin_perimeter",
    # boundary
    "BoundaryError", "align_rings", "ensure_closed_ring_order",
    "resample_ring", "ring_arc_params", "targets_for_loop_vertices",
    # laplacian
    "LaplacianError", "cotangent_laplacian", "solve_laplacian",
    # contact
    "ContactError", "ContactParameters", "ContactResult", "NeighborPatch",
    "SideContactReport", "optimize_contacts",
    # contact refinement (I.1)
    "RefineParams", "refine_contacts",
    # generator / gt
    "generate_crown_outer_v0",
    "GtExtractError", "extract_outer_surface", "sample_surface_points",
    "surface_distance_metrics",
]

# CrownFlow 1.0 active proximal policy.
# Services import ``fit_proximal_envelopes`` directly from ``space_contact``.
# V9 keeps V8/V7 ownership and adds one final sub-0.2 mm contact "kiss" on the
# physical side with the larger remaining residual.  It never hard-codes camera
# left/right, and the opposite established contact remains protected.
from . import space_contact as _space_contact
from .proximal_v9 import fit_proximal_envelopes_v9 as _fit_proximal_envelopes_v9

_space_contact.fit_proximal_envelopes = _fit_proximal_envelopes_v9
