"""Crown outer geometry engines.

Two paths intentionally coexist:

* ``generate_crown_outer_v0`` keeps the legacy/library-template deformation path
  for preview, partial-context and regression compatibility.
* ``build_sectional_outer`` is Crown 1.0's patient-space path: exact Margin
  boundary (RED), broad neighbor/emergence envelope (GREEN), and antagonist-
  constrained smoothed occlusal cap (BLUE).  The final intaglio is still built
  by the existing blockout + gap + exact-boolean finalization pipeline.
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
from .sectional_v1 import SectionalOuterError, SectionalV1Config, build_sectional_outer
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
    # generators
    "generate_crown_outer_v0",
    "SectionalOuterError", "SectionalV1Config", "build_sectional_outer",
    # gt
    "GtExtractError", "extract_outer_surface", "sample_surface_points",
    "surface_distance_metrics",
]
