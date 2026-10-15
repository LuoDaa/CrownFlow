"""Authoritative intaglio preview shared with outer-crown space fitting.

The final intaglio is determined by prepared support + insertion-direction blockout
+ confirmed gap parameters.  CAD-1 needs that geometry *before* final stitching so
the outer embryo can be grown large enough to contain it.  This module constructs
the same engineering cutter semantics used by ``pipeline.finalize_crown`` without
performing a Boolean or cervical stitch.
"""
from __future__ import annotations

import numpy as np

from ..geometry.intaglio import BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout


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
    cutter = apply_gap_field(blockout, fitting_support, margin, axis, gap_params)
    vertices = np.asarray(cutter.vertices, float)
    if not len(vertices) or not np.isfinite(vertices).all() or not len(cutter.faces):
        raise ValueError("Intaglio preview 生成无效网格")
    report = {
        "status": "READY",
        "method": "DIRECTIONAL_BLOCKOUT_PLUS_CONFIRMED_GAP_FIELD",
        "blockout": blockout.as_dict(),
        "cutter_vertices": int(len(cutter.vertices)),
        "cutter_faces": int(len(cutter.faces)),
        "cement_gap_mm": float(params["cement_gap_mm"]),
        "cement_gap_start_mm": params.get("cement_gap_start_mm"),
        "cement_gap_top_mm": params.get("cement_gap_top_mm"),
        "machine_axes": int(params.get("machine_axes") or 0),
        "scope": "PRE_FINALIZATION_SPACE_ENVELOPE",
    }
    return cutter, blockout, report
