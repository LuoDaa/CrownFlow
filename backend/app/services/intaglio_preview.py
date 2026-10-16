"""Authoritative intaglio preview shared with outer-crown space fitting.

The final intaglio is determined by prepared support + insertion-direction blockout
+ confirmed gap parameters. CAD-1 needs that geometry before final stitching so
the outer embryo can be grown large enough to contain it.

For the 1.0 canvas this module returns the *fitting surface above the confirmed
3D Margin*, not the artificial closed cutter body. The closed cutter contains a
bottom closure used only for Boolean subtraction; rendering that closure made the
inner crown look like a triangular skirt below the Margin even though the real
fitting surface was correct.
"""
from __future__ import annotations

import numpy as np

from ..geometry.intaglio import BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout
from ..geometry.nonplanar_crown import MarginCurtain, _single_open_surface_above_margin


def build_intaglio_preview(fitting_support, margin_points, insertion_axis, params: dict):
    margin = np.asarray(margin_points, float)
    axis = np.asarray(insertion_axis, float)
    if margin.ndim != 2 or margin.shape[1] != 3 or len(margin) < 3:
        raise ValueError("Intaglio preview 需要已确认三维 Margin")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("Intaglio preview 需要有效插入方向")
    if params.get("cement_gap_mm") is None:
        raise ValueError("Intaglio preview 缺少 cement_gap_mm")

    policy = BlockoutPolicy(
        machine_axes=int(params.get("machine_axes") or 0),
        additional_blockout_after_gap=None,
        untouchable_zone_mm=float(params.get("untouchable_zone_mm") or 0.0),
        provenance={
            "source": "confirmed_design_profile",
            "machine_axes_explicit": params.get("machine_axes") is not None,
            "purpose": "pre_outer_space_constraint",
        },
    )
    blockout = build_blockout(fitting_support, margin, axis, policy)
    gap_params = GapFieldParams(
        cement_gap_mm=float(params["cement_gap_mm"]),
        cement_gap_start_mm=params.get("cement_gap_start_mm"),
        cement_gap_top_mm=params.get("cement_gap_top_mm"),
        additional_spacing_xyz_mm=None,
        additional_spacing_z_on_top_mm=None,
        provenance={"source": "confirmed_design_profile", "purpose": "pre_outer_space_constraint"},
    )
    closed_cutter = apply_gap_field(blockout, fitting_support, margin, axis, gap_params)
    vertices = np.asarray(closed_cutter.vertices, float)
    if not len(vertices) or not np.isfinite(vertices).all() or not len(closed_cutter.faces):
        raise ValueError("Intaglio preview 生成无效网格")

    # The Boolean cutter is deliberately closed below the preparation. That lower
    # cap is not crown intaglio and must never be presented as the inner crown.
    # Clip at transition=0 against the same non-planar Margin curtain used by the
    # final crown builder, leaving one open fitting surface whose cervical loop is
    # geometrically owned by the confirmed 3D Margin.
    preview = None
    extraction = None
    preview_error = None
    try:
        curtain = MarginCurtain(margin, axis)
        preview, extraction = _single_open_surface_above_margin(
            closed_cutter,
            curtain,
            0.0,
            "INTAGLIO_PREVIEW",
            invert=True,
        )
        pverts = np.asarray(preview.vertices, float)
        if not len(pverts) or not np.isfinite(pverts).all() or not len(preview.faces):
            raise ValueError("Margin 裁切后的内冠预览为空")
    except Exception as exc:
        # Do not make a display-only refinement capable of stopping generation.
        # The fallback remains traceable in the stage report.
        preview = closed_cutter.copy()
        preview_error = str(exc)

    report = {
        "status": "READY" if preview_error is None else "READY_WITH_DISPLAY_FALLBACK",
        "method": "DIRECTIONAL_BLOCKOUT_PLUS_CONFIRMED_GAP_FIELD_MARGIN_CLIPPED",
        "blockout": blockout.as_dict(),
        "closed_cutter_vertices": int(len(closed_cutter.vertices)),
        "closed_cutter_faces": int(len(closed_cutter.faces)),
        "preview_vertices": int(len(preview.vertices)),
        "preview_faces": int(len(preview.faces)),
        "cement_gap_mm": float(params["cement_gap_mm"]),
        "cement_gap_start_mm": params.get("cement_gap_start_mm"),
        "cement_gap_top_mm": params.get("cement_gap_top_mm"),
        "machine_axes": int(params.get("machine_axes") or 0),
        "scope": "PRE_FINALIZATION_FITTING_SURFACE_ABOVE_CONFIRMED_MARGIN",
        "margin_owned_cervical_boundary": preview_error is None,
        "artificial_bottom_closure_rendered": False if preview_error is None else True,
        "surface_extraction": extraction,
        "display_fallback_reason": preview_error,
    }
    return preview, blockout, report
