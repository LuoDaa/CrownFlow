"""CrownFlow 1.0 outer-frozen, support-owned final intaglio assembly.

Contracts are intentionally one-way:
- accepted outer owns external anatomy and is immutable during W3;
- PreparedSupport owns the clinical intaglio;
- ConfirmedMargin owns the final cervical seam;
- final collar generation may add local topology but may not reshape body anatomy.

V6 removes the previous 0.8 mm geodesic inset from the *final* collar source.
Case 63 proved that the geodesic ring can be much farther from Margin and can
have >1000 vertices, creating severe phase mismatch and self-intersection even
when topology is watertight.  Final collar assembly now starts from the already
validated exact-Margin local source loops and reduces dense loops progressively
to the exact shared Margin[128].
"""
from __future__ import annotations

from contextvars import ContextVar
import numpy as np
from trimesh.proximity import closest_point

from .geometry import nonplanar_crown as _npc
from .geometry.crown_outer import space_morph as _space
from .geometry.canonical_intaglio import (
    IntaglioOwnershipError,
    cervical_feasibility,
    extract_support_owned_intaglio,
)
from .geometry.fixed_margin_128 import build_fixed_128_margin_crown


_FINALIZATION_OUTER_FROZEN: ContextVar[bool] = ContextVar(
    "crownflow_finalization_outer_frozen", default=False
)
_ORIGINAL_CLEARANCE_ENFORCER = _space.enforce_inner_envelope_clearance


def _audit_only_clearance(mesh, inner_envelope_mesh, frame, margin, required_mm, *args, **kwargs):
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

    # Exact-Margin local outer source.  No additional 0.8 mm geodesic retreat is
    # used for final assembly: case 63 showed that retreat magnifies phase error.
    outside, outer_margin_extract = _npc._single_open_surface_above_margin(
        outer, curtain, 1e-6, "OUTER", invert=False
    )
    outer_extract = {
        "method": "EXACT_MARGIN_LOCAL_SOURCE_V1",
        "margin_clip": outer_margin_extract,
        "final_geodesic_inset_used": False,
    }
    record_topology("10_outer_exact_margin_source", outside)

    # PreparedSupport, not Margin proximity, proves the real fitting surface.
    try:
        inside, inner_ownership = extract_support_owned_intaglio(
            cutter, support, curtain, 1e-6, params
        )
    except IntaglioOwnershipError as exc:
        raise _npc.MarginBuildError(f"{exc.code}: {exc.report}") from exc
    if not inner_ownership.get("selected_hard_valid"):
        raise _npc.MarginBuildError(
            "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT: " + str(inner_ownership)
        )
    inner_extract = {
        **inner_ownership,
        "final_geodesic_inset_used": False,
        "final_source": "EXACT_MARGIN_LOCAL_PREPARED_SUPPORT_COMPONENT",
    }
    record_topology("29_intaglio_exact_margin_source", inside)

    feasible, feasibility = cervical_feasibility(outside, inside, curtain.points)
    feasibility["intaglio_component_valid"] = True
    feasibility["transition_owner"] = "EXACT_MARGIN_LOCAL_COLLAR"
    feasibility["final_mapping"] = "PROGRESSIVE_FULL_BOUNDARY_TO_FIXED_MARGIN_128"
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

    final, outer_surface, inner_surface, collar = build_fixed_128_margin_crown(
        outside, inside, curtain.points
    )
    record_topology("40_outer_progressive_margin128", outer_surface)
    record_topology("41_inner_progressive_margin128", inner_surface)
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
        "status": "EXACT_MARGIN_PROGRESSIVE_128_COLLAR",
        "outer_before": mesh_diagnostics(outside, curtain.points),
        "inner_before": mesh_diagnostics(inside, curtain.points),
        "scope": "CROWN_CERVICAL_REGION_ONLY",
        "margin_curve_untouched": True,
        "accepted_outer_body_untouched": True,
        "feasibility": feasibility,
        "fixed_128_collar": collar,
    }

    stitch_warnings = []
    if collar.get("self_intersections_review_required"):
        stitch_warnings.append("CERVICAL_COLLAR_SELF_INTERSECTION_REVIEW_REQUIRED")

    report = {
        "status": "ENGINEERING_ONLY",
        "method": "SUPPORT_OWNED_INTAGLIO_EXACT_MARGIN_PROGRESSIVE_128_V6",
        "assembly_order": [
            "ACCEPTED_OUTER_FROZEN",
            "PATIENT_BLOCKOUT_GAP_CUTTER",
            "EXACT_3D_MARGIN_OUTER_SPLIT",
            "PREPARED_SUPPORT_OWNS_INTAGLIO_COMPONENT",
            "EXACT_MARGIN_LOCAL_INTAGLIO_SOURCE",
            "CERVICAL_FEASIBILITY_GATE",
            "OUTER_FULL_SOURCE_BOUNDARY_PRESERVED",
            "INTAGLIO_FULL_SOURCE_BOUNDARY_PRESERVED",
            "PROGRESSIVE_CARDINALITY_REDUCTION",
            "EXACT_CONFIRMED_MARGIN_128",
            "BOOLEAN_AUDIT_ONLY",
        ],
        "contracts": {
            "outer_owner": "ACCEPTED_OUTER",
            "intaglio_owner": "PREPARED_SUPPORT",
            "cervical_boundary_owner": "CONFIRMED_MARGIN",
            "cervical_topology_owner": "PROGRESSIVE_FULL_BOUNDARY_TO_FIXED_128_MARGIN",
            "cervical_width_owner": "EXACT_MARGIN_LOCAL_GENERATED_COLLAR",
            "finalization_outer_mutability": "FROZEN",
        },
        "margin": _npc.margin_plane_diagnostics(curtain.points),
        "scan_adherence": scan,
        "transition_mm": float(transition_mm),
        "transition_semantics": "PARAMETER_RETAINED_FOR_COMPATIBILITY_NOT_FINAL_COLLAR_HEIGHT",
        "margin_topology_samples": 128,
        "outer_frozen": True,
        "body_vertices_moved": 0,
        "body_outer_vertex_count": int(len(outside.vertices)),
        "body_to_original_outer_p95_mm": float(np.quantile(preservation, .95)) if len(preservation) else 0.0,
        "body_to_original_outer_max_mm": float(preservation.max()) if len(preservation) else 0.0,
        "geometry_above_cervical_transition_preserved": True,
        "margin_profile": "SHARED_CONFIRMED_3D_MARGIN_128",
        "crown_border": None,
        "minimum_thickness_enforcement": None,
        "final_stage_thickness_policy": thickness_policy,
        "cervical_cleanup": cervical_report,
        "surface_extraction": {
            "method": "ROLE_SEPARATED_EXACT_MARGIN_PROGRESSIVE_128_V6",
            "outer": outer_extract,
            "intaglio": inner_extract,
            "cervical_separation": source_separation,
            "boolean_source_classification_used": False,
        },
        "boolean_audit": _boolean_audit(outer, cutter),
        "quality_mode": quality_mode,
        "design_mode_exception": None,
        "stitch_warnings": stitch_warnings,
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
