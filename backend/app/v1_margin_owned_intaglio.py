"""CrownFlow 1.0 outer-frozen, support-owned final intaglio assembly.

The contracts are intentionally one-way and independent:
- accepted outer owns external anatomy and is immutable during W3;
- PreparedSupport owns which cutter component is the clinical intaglio;
- ConfirmedMargin owns the final cervical boundary;
- the collar is generated topology only, never a reason to reshape the body.

Boolean remains an audit. It is never used to rediscover clinical surfaces.
"""
from __future__ import annotations

from contextvars import ContextVar
import numpy as np
from trimesh.proximity import closest_point

from .geometry import nonplanar_crown as _npc
from .geometry.crown_outer import space_morph as _space
from .geometry.canonical_intaglio import (
    IntaglioOwnershipError,
    build_canonical_margin_crown,
    cervical_feasibility,
    extract_support_owned_intaglio,
)
from .geometry.cervical_geodesic import inset_open_surface_geodesic


_FINALIZATION_OUTER_FROZEN: ContextVar[bool] = ContextVar(
    "crownflow_finalization_outer_frozen", default=False
)
_ORIGINAL_CLEARANCE_ENFORCER = _space.enforce_inner_envelope_clearance


def _audit_only_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs):
    """Finalization may measure inner feasibility but may not change accepted outer."""
    if not _FINALIZATION_OUTER_FROZEN.get():
        return _ORIGINAL_CLEARANCE_ENFORCER(
            mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs
        )
    audit = _space.audit_inner_envelope_clearance(
        mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs
    )
    return mesh.copy(), {
        "status": "AUDIT_ONLY_OUTER_FROZEN",
        "method": "ACCEPTED_OUTER_IMMUTABLE_DURING_FINAL_INTAGLIO",
        "required_outer_to_intaglio_mm": float(required_mm),
        "moved_vertices": 0,
        "max_push_per_pass_mm": 0.0,
        "audit_after": audit,
        "outer_modified": False,
        "intaglio_modified": False,
    }


_space.enforce_inner_envelope_clearance = _audit_only_clearance


def _boolean_audit(outer, cutter):
    report = {
        "role": "AUDIT_ONLY_NOT_SOURCE_SURFACE",
        "status": "NOT_RUN",
        "outer_unchanged": True,
        "cutter_unchanged": True,
    }
    try:
        boolean = _npc.crown_minus_cutter(outer, cutter)
        report.update({
            "status": "COMPUTED",
            "faces": int(len(boolean.faces)),
            "vertices": int(len(boolean.vertices)),
            "watertight": bool(boolean.is_watertight),
            "component_count": int(len(boolean.split(only_watertight=False))),
            "volume_mm3": float(abs(boolean.volume)) if boolean.is_volume else None,
        })
        try:
            from .geometry.boolean_healing import boolean_healing_stage
            healed, healing = boolean_healing_stage(boolean)
            report["healing"] = healing
            report["healed_faces"] = int(len(healed.faces))
        except Exception as exc:
            report["healing"] = {"status": "ERROR_IGNORED", "error": str(exc)[:200]}
    except Exception as exc:
        report.update({"status": "FAILED_NONBLOCKING", "error": str(exc)[:300]})
    return report


def _margin_owned_build(outer, cutter, support, margin, axis, *, transition_mm,
                        prepared_support=False, parameters=None, protected_surfaces=(),
                        quality_mode="DESIGN"):
    """Build final crown after outer is already accepted.

    `margin_band_mm` is interpreted as a surface/geodesic cervical width, not as
    an insertion-axis height. The exact Margin first separates supra/sub-margin
    geometry; then the source bodies are inset by geodesic distance without moving
    any accepted body vertex.
    """
    params = parameters or {}
    curtain = _npc.MarginCurtain(margin, axis)
    if not np.isfinite(transition_mm) or float(transition_mm) <= 0:
        raise _npc.MarginBuildError("三维颈部过渡高度 margin_band_mm 必须为正值")

    scan = _npc.audit_margin_on_scan(
        support, curtain.points, expected_margin_boundary=prepared_support
    )
    from .geometry.topology_validation import record_topology
    record_topology("01_input_outer", outer)
    record_topology("02_input_cutter", cutter)

    # OUTER: exact 3D Margin first, then constant surface-width transition.  The
    # accepted body is never deformed; only below-band source faces are discarded.
    outside_margin, outer_margin_extract = _npc._single_open_surface_above_margin(
        outer, curtain, 1e-6, "OUTER", invert=False
    )
    outside, outer_geodesic = inset_open_surface_geodesic(
        outside_margin, float(transition_mm), role="OUTER"
    )
    outer_extract = {
        "method": "EXACT_MARGIN_THEN_SURFACE_GEODESIC",
        "margin_clip": outer_margin_extract,
        "geodesic_transition": outer_geodesic,
    }
    record_topology("10_outer_selected", outside)

    # INTAGLIO: component ownership is proven on the full supramarginal cutter
    # before any collar-width trimming. Margin is boundary owner, not component owner.
    try:
        inside_margin, inner_ownership = extract_support_owned_intaglio(
            cutter, support, curtain, 1e-6, params
        )
    except IntaglioOwnershipError as exc:
        raise _npc.MarginBuildError(f"{exc.code}: {exc.report}") from exc
    if not inner_ownership.get("selected_hard_valid"):
        raise _npc.MarginBuildError(
            "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT: " + str(inner_ownership)
        )
    inside, inner_geodesic = inset_open_surface_geodesic(
        inside_margin, float(transition_mm), role="INTAGLIO"
    )
    inner_extract = {
        **inner_ownership,
        "geodesic_transition": inner_geodesic,
    }
    record_topology("29_intaglio_selected", inside)

    feasible, feasibility = cervical_feasibility(outside, inside, curtain.points)
    feasibility["intaglio_component_valid"] = True
    feasibility["transition_owner"] = "SURFACE_GEODESIC"
    if not feasible:
        raise _npc.MarginBuildError(
            "CERVICAL_ASSEMBLY_NOT_FEASIBLE: " + str(feasibility)
        )

    thickness_policy = None
    if params.get("enforce_minimum_thickness"):
        thickness_policy = {
            "status": "AUDIT_ONLY_OUTER_FROZEN",
            "requested_mm": float(params.get("minimum_thickness_mm") or 0.0),
            "geometry_changed": False,
        }

    final, outer_surface, inner_surface, collar = build_canonical_margin_crown(
        outside, inside, curtain.points
    )
    record_topology("40_outer_collar_final", outer_surface)
    record_topology("41_inner_collar_final", inner_surface)
    record_topology("50_final", final)

    outer_loop = _npc._boundary_loops(np.asarray(outside.faces, np.int64))[0]
    outer_ring = np.asarray(outside.vertices, float)[np.asarray(outer_loop, np.int64)]
    _cp, separation, _ = closest_point(inside, outer_ring)
    separation = np.asarray(separation, float)
    source_separation = {
        "status": "MEASURED_NO_GEOMETRY_CHANGE",
        "min_mm": float(separation.min()),
        "p01_mm": float(np.quantile(separation, .01)),
        "p05_mm": float(np.quantile(separation, .05)),
        "median_mm": float(np.median(separation)),
        "outer_modified": False,
        "intaglio_modified": False,
    }

    preservation = np.concatenate([
        closest_point(outer, np.asarray(outside.vertices)[i:i + 512])[1]
        for i in range(0, len(outside.vertices), 512)
    ]) if len(outside.vertices) else np.zeros(0, float)

    from .geometry.cervical_cleanup import mesh_diagnostics
    cervical_report = {
        "status": "GEODESIC_CANONICAL_SHARED_MARGIN_COLLAR",
        "outer_before": mesh_diagnostics(outside, curtain.points),
        "inner_before": mesh_diagnostics(inside, curtain.points),
        "scope": "CROWN_CERVICAL_REGION_ONLY",
        "margin_curve_untouched": True,
        "accepted_outer_body_untouched": True,
        "feasibility": feasibility,
        "canonical_collar": collar,
    }

    report = {
        "status": "ENGINEERING_ONLY",
        "method": "SUPPORT_OWNED_INTAGLIO_GEODESIC_CANONICAL_MARGIN_V3",
        "assembly_order": [
            "ACCEPTED_OUTER_FROZEN",
            "PATIENT_BLOCKOUT_GAP_CUTTER",
            "EXACT_3D_MARGIN_SUPRAMARGINAL_SPLIT",
            "PREPARED_SUPPORT_OWNS_INTAGLIO_COMPONENT",
            "SURFACE_GEODESIC_CERVICAL_TRANSITION",
            "CERVICAL_FEASIBILITY_GATE",
            "CANONICAL_MARGIN_ARCLENGTH_COLLAR",
            "BOOLEAN_AUDIT_ONLY",
        ],
        "contracts": {
            "outer_owner": "ACCEPTED_OUTER",
            "intaglio_owner": "PREPARED_SUPPORT",
            "cervical_boundary_owner": "CONFIRMED_MARGIN",
            "cervical_width_owner": "SURFACE_GEODESIC",
            "finalization_outer_mutability": "FROZEN",
        },
        "margin": _npc.margin_plane_diagnostics(curtain.points),
        "scan_adherence": scan,
        "transition_mm": float(transition_mm),
        "transition_semantics": "SURFACE_GEODESIC_DISTANCE_FROM_CONFIRMED_MARGIN",
        "outer_frozen": True,
        "body_vertices_moved": 0,
        "body_outer_vertex_count": int(len(outside.vertices)),
        "body_to_original_outer_p95_mm": float(np.quantile(preservation, .95)) if len(preservation) else 0.0,
        "body_to_original_outer_max_mm": float(preservation.max()) if len(preservation) else 0.0,
        "geometry_above_cervical_transition_preserved": True,
        "margin_profile": "SHARED_CONFIRMED_3D_MARGIN",
        "crown_border": None,
        "minimum_thickness_enforcement": None,
        "final_stage_thickness_policy": thickness_policy,
        "cervical_cleanup": cervical_report,
        "surface_extraction": {
            "method": "ROLE_SEPARATED_GEODESIC_SOURCE_EXTRACTION_V3",
            "outer": outer_extract,
            "intaglio": inner_extract,
            "cervical_separation": source_separation,
            "boolean_source_classification_used": False,
        },
        "boolean_audit": _boolean_audit(outer, cutter),
        "quality_mode": quality_mode,
        "design_mode_exception": None,
        "stitch_warnings": [],
    }
    return _npc.NonplanarCrownBuild(final, outer_surface, inner_surface, report)


_npc.build_nonplanar_crown = _margin_owned_build


from .services import pipeline as _pipeline
_ORIGINAL_FINALIZE = _pipeline.finalize_crown


def _finalize_outer_then_inner(*args, **kwargs):
    token = _FINALIZATION_OUTER_FROZEN.set(True)
    try:
        return _ORIGINAL_FINALIZE(*args, **kwargs)
    finally:
        _FINALIZATION_OUTER_FROZEN.reset(token)


_finalize_outer_then_inner._crownflow_outer_then_inner = True
_pipeline.finalize_crown = _finalize_outer_then_inner
