"""CrownFlow 1.0 outer-frozen, support-owned final intaglio assembly.

Contracts are intentionally one-way:
- accepted outer owns external anatomy and is immutable during W3;
- PreparedSupport owns the clinical intaglio;
- ConfirmedMargin owns the final cervical seam;
- final collar generation may add local topology but may not reshape body anatomy.

V7 starts from the validated exact-Margin sources, then performs a read-only
outer/intaglio cross-surface audit before collar generation.  If the two source
skins cross above the knife-edge zone, W3 retreats *both* source rings by the
smallest surface-geodesic distance that removes the crossing.  No accepted outer
vertex is moved: only the generated cervical collar becomes slightly deeper.
This preserves the good anatomy while enforcing Support < Intaglio < Outer in the
region where case 63 previously produced real OUTER x INNER intersections.
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
from .geometry.cervical_geodesic import inset_open_surface_geodesic
from .geometry.crown_outer.thickness import self_intersections
from .geometry.fixed_margin_128 import build_fixed_128_margin_crown


_FINALIZATION_OUTER_FROZEN: ContextVar[bool] = ContextVar(
    "crownflow_finalization_outer_frozen", default=False
)
_ORIGINAL_CLEARANCE_ENFORCER = _space.enforce_inner_envelope_clearance

# Exact Margin is always tried first.  Retreat is a fallback only for a proven
# outer/intaglio body crossing, and uses the smallest successful distance.
_CERVICAL_RETREAT_CANDIDATES_MM = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20, 1.40, 1.60)
_CROSS_AUDIT_MIN_HEIGHT_MM = 0.12
_CROSS_AUDIT_MAX_HEIGHT_MM = 3.0


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


def _cross_surface_audit(outer, inner, curtain):
    """Find true OUTER x INTAGLIO intersections above the knife-edge seam zone.

    Each individual surface may legitimately be open.  The only condition this
    gate cares about is a triangle from the frozen anatomical body crossing a
    triangle from the PreparedSupport-owned fitting body.  Faces very close to
    exact Margin are excluded because the two generated collars are supposed to
    converge there.
    """
    ov = np.asarray(outer.vertices, float)
    iv = np.asarray(inner.vertices, float)
    of = np.asarray(outer.faces, np.int64)
    inf = np.asarray(inner.faces, np.int64)
    if not len(of) or not len(inf):
        return False, {"status": "INVALID", "reason": "EMPTY_OUTER_OR_INTAGLIO"}

    offset = len(ov)
    vertices = np.vstack([ov, iv])
    faces = np.vstack([of, inf + offset])
    outer_centroids = ov[of].mean(axis=1)
    height = np.asarray(curtain.axial_above(outer_centroids), float)
    focus_outer = (
        np.isfinite(height)
        & (height >= _CROSS_AUDIT_MIN_HEIGHT_MM)
        & (height <= _CROSS_AUDIT_MAX_HEIGHT_MM)
    )
    focus = np.r_[focus_outer, np.zeros(len(inf), dtype=bool)]
    raw = self_intersections(
        vertices,
        faces,
        neighbor_k=16,
        max_pairs=1_000_000,
        focus_faces=focus,
        return_pairs=True,
    )
    pairs = np.asarray(raw.pop("self_intersections_pairs", np.zeros((0, 2), np.int64)), np.int64)
    if pairs.ndim != 2:
        pairs = np.zeros((0, 2), np.int64)
    n_outer = len(of)
    cross = pairs[
        ((pairs[:, 0] < n_outer) & (pairs[:, 1] >= n_outer))
        | ((pairs[:, 1] < n_outer) & (pairs[:, 0] >= n_outer))
    ] if len(pairs) else np.zeros((0, 2), np.int64)

    samples = []
    for a, b in cross[:20]:
        if a >= n_outer:
            a, b = b, a
        samples.append({
            "outer_face": int(a),
            "intaglio_face": int(b - n_outer),
        })
    report = {
        "status": "PASS_ENGINEERING" if len(cross) == 0 else "REVIEW_REQUIRED",
        "method": "FOCUSED_EXACT_TRIANGLE_OUTER_INTAGLIO_INTERSECTION_V1",
        "focus_outer_faces": int(np.sum(focus_outer)),
        "minimum_height_mm": float(_CROSS_AUDIT_MIN_HEIGHT_MM),
        "maximum_height_mm": float(_CROSS_AUDIT_MAX_HEIGHT_MM),
        "cross_intersection_count": int(len(cross)),
        "sample_pairs": samples,
        "candidate_pair_tests": int(raw.get("pair_tests") or 0),
        "candidate_screen_truncated": bool(raw.get("self_intersections_truncated")),
    }
    return len(cross) == 0, report


def _choose_non_crossing_cervical_sources(outside_exact, inside_exact, curtain):
    """Choose the shallowest generated collar start that separates both bodies."""
    attempts = []
    best = None
    for width in _CERVICAL_RETREAT_CANDIDATES_MM:
        try:
            if width <= 0.0:
                outside = outside_exact
                inside = inside_exact
                outer_inset = {"method": "EXACT_MARGIN_LOCAL_SOURCE", "width_mm": 0.0}
                inner_inset = {"method": "EXACT_MARGIN_LOCAL_SOURCE", "width_mm": 0.0}
            else:
                outside, outer_inset = inset_open_surface_geodesic(
                    outside_exact, float(width), role="OUTER"
                )
                inside, inner_inset = inset_open_surface_geodesic(
                    inside_exact, float(width), role="INTAGLIO"
                )

            feasible, feasibility = cervical_feasibility(
                outside, inside, curtain.points
            )
            crossing_ok, crossing = _cross_surface_audit(outside, inside, curtain)
            attempt = {
                "width_mm": float(width),
                "feasible": bool(feasible),
                "crossing_free": bool(crossing_ok),
                "cross_intersection_count": int(crossing.get("cross_intersection_count") or 0),
                "feasibility": feasibility,
                "cross_surface": crossing,
                "outer_inset": outer_inset,
                "inner_inset": inner_inset,
            }
            attempts.append(attempt)
            if feasible:
                rank = (attempt["cross_intersection_count"], float(width))
                if best is None or rank < best[0]:
                    best = (rank, outside, inside, attempt)
            if feasible and crossing_ok:
                return outside, inside, {
                    "status": "PASS_ENGINEERING",
                    "method": "MINIMUM_GEODESIC_RETREAT_TO_NONCROSSING_BODIES_V1",
                    "selected_width_mm": float(width),
                    "attempts": attempts,
                    "crossing_eliminated": True,
                }
        except Exception as exc:
            attempts.append({
                "width_mm": float(width),
                "feasible": False,
                "crossing_free": False,
                "error": str(exc)[:300],
            })

    # 1.0 is fail-open for review geometry.  If no width removes every crossing,
    # retain the feasible attempt with the fewest crossings rather than returning
    # an empty crown; the warning remains explicit and manufacturing is not passed.
    if best is not None:
        _rank, outside, inside, attempt = best
        return outside, inside, {
            "status": "REVIEW_REQUIRED",
            "method": "MINIMUM_GEODESIC_RETREAT_TO_NONCROSSING_BODIES_V1",
            "selected_width_mm": float(attempt["width_mm"]),
            "attempts": attempts,
            "crossing_eliminated": False,
            "remaining_cross_intersections": int(attempt["cross_intersection_count"]),
        }
    raise _npc.MarginBuildError(
        "CERVICAL_SOURCE_SELECTION_FAILED: " + str(attempts)
    )


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

    # Exact-Margin local sources are the first choice.  A later read-only crossing
    # audit decides whether a shallow geodesic retreat is actually necessary.
    outside_exact, outer_margin_extract = _npc._single_open_surface_above_margin(
        outer, curtain, 1e-6, "OUTER", invert=False
    )
    record_topology("10_outer_exact_margin_source", outside_exact)

    try:
        inside_exact, inner_ownership = extract_support_owned_intaglio(
            cutter, support, curtain, 1e-6, params
        )
    except IntaglioOwnershipError as exc:
        raise _npc.MarginBuildError(f"{exc.code}: {exc.report}") from exc
    if not inner_ownership.get("selected_hard_valid"):
        raise _npc.MarginBuildError(
            "INTAGLIO_COMPONENT_NOT_PROVEN_BY_SUPPORT: " + str(inner_ownership)
        )
    record_topology("29_intaglio_exact_margin_source", inside_exact)

    outside, inside, source_selection = _choose_non_crossing_cervical_sources(
        outside_exact, inside_exact, curtain
    )
    selected_width = float(source_selection.get("selected_width_mm") or 0.0)
    if selected_width > 0:
        record_topology("30_outer_adaptive_cervical_source", outside)
        record_topology("31_intaglio_adaptive_cervical_source", inside)

    feasible, feasibility = cervical_feasibility(outside, inside, curtain.points)
    feasibility["intaglio_component_valid"] = True
    feasibility["transition_owner"] = (
        "EXACT_MARGIN_LOCAL_COLLAR" if selected_width <= 0
        else "ADAPTIVE_SURFACE_GEODESIC_COLLAR_START"
    )
    feasibility["final_mapping"] = "PROGRESSIVE_FULL_BOUNDARY_TO_FIXED_MARGIN_128"
    feasibility["selected_cervical_source_width_mm"] = selected_width
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
        "status": "ADAPTIVE_EXACT_MARGIN_PROGRESSIVE_128_COLLAR",
        "outer_before": mesh_diagnostics(outside, curtain.points),
        "inner_before": mesh_diagnostics(inside, curtain.points),
        "scope": "CROWN_CERVICAL_REGION_ONLY",
        "margin_curve_untouched": True,
        "accepted_outer_body_untouched": True,
        "source_selection": source_selection,
        "feasibility": feasibility,
        "fixed_128_collar": collar,
    }

    stitch_warnings = []
    if not source_selection.get("crossing_eliminated", True):
        stitch_warnings.append("OUTER_INTAGLIO_BODY_CROSSING_REVIEW_REQUIRED")
    if collar.get("self_intersections_review_required"):
        stitch_warnings.append("CERVICAL_COLLAR_SELF_INTERSECTION_REVIEW_REQUIRED")

    report = {
        "status": "ENGINEERING_ONLY",
        "method": "SUPPORT_OWNED_INTAGLIO_ADAPTIVE_MARGIN_PROGRESSIVE_128_V7",
        "assembly_order": [
            "ACCEPTED_OUTER_FROZEN",
            "PATIENT_BLOCKOUT_GAP_CUTTER",
            "EXACT_3D_MARGIN_OUTER_SPLIT",
            "PREPARED_SUPPORT_OWNS_INTAGLIO_COMPONENT",
            "EXACT_MARGIN_LOCAL_INTAGLIO_SOURCE",
            "OUTER_INTAGLIO_CROSS_SURFACE_AUDIT",
            "MINIMUM_GEODESIC_RETREAT_IF_REQUIRED",
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
            "cervical_width_owner": "MINIMUM_NONCROSSING_GENERATED_COLLAR_DEPTH",
            "finalization_outer_mutability": "FROZEN",
        },
        "margin": _npc.margin_plane_diagnostics(curtain.points),
        "scan_adherence": scan,
        "transition_mm": float(transition_mm),
        "transition_semantics": "ADAPTIVE_GEODESIC_ONLY_WHEN_BODY_CROSSING_REQUIRES_RETREAT",
        "selected_cervical_source_width_mm": selected_width,
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
            "method": "ROLE_SEPARATED_ADAPTIVE_MARGIN_PROGRESSIVE_128_V7",
            "outer": {
                "method": "EXACT_MARGIN_THEN_MINIMUM_RETREAT_IF_REQUIRED",
                "margin_clip": outer_margin_extract,
            },
            "intaglio": {
                **inner_ownership,
                "final_source": "PREPARED_SUPPORT_COMPONENT_THEN_MINIMUM_RETREAT_IF_REQUIRED",
            },
            "source_selection": source_selection,
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
