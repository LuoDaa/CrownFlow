"""CrownFlow 1.0 W3 guard for artificial template basal closures.

The accepted outer remains immutable.  Some crown libraries are delivered as a
closed morphology prior: their otherwise anatomical outer shell is sealed by a
single high-valence fan vertex across the basal opening.  That artificial cap is
useful for some outer-only operations, but it is not a clinical crown surface.
If W3 clips/geodesically insets that closed mesh directly, the cap can win the
cervical component selection and a short partial loop is then stretched over the
full confirmed Margin.  The result is watertight but visually wrong and can hide
or intersect the real intaglio.

This module is intentionally narrow:
* accepted ``outer_path`` is never changed;
* only a derived W3 source copy may have a basal fan removed;
* removal requires several independent geometric proofs;
* the existing PreparedSupport-owned intaglio and confirmed 3D Margin stay the
  authorities for fit and final seam;
* a second gate rejects partial outer transition loops before collar assembly.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .geometry import nonplanar_crown as _npc
from .geometry.canonical_intaglio import _resample_closed_polyline
from .geometry.margin_seam import _polyline_distances
from . import v1_margin_owned_intaglio as _policy


_MIN_CAP_APEX_VALENCE = 24
_MIN_VALENCE_TO_MEDIAN_RATIO = 4.0
_MIN_VALENCE_TO_SECOND_RATIO = 3.0
_MAX_APEX_TO_MARGIN_MM = 3.0
_MIN_OPPOSING_NORMAL_FRACTION = 0.75
_MAX_CAP_AREA_FRACTION = 0.30
_POST_LOOP_PERIMETER_RATIO = (0.75, 2.00)
_POST_LOOP_TO_MARGIN_P95_MM = 3.0
_POST_LOOP_TO_MARGIN_MAX_MM = 4.0

_TRANSITION_PERIMETER_RATIO = (0.78, 2.25)
_TRANSITION_PHASE_MAX_FRACTION = 0.22
_TRANSITION_PHASE_P95_FRACTION = 0.18
_TRANSITION_PHASE_MAX_FLOOR_MM = 3.5
_TRANSITION_PHASE_P95_FLOOR_MM = 3.0


def _perimeter(points) -> float:
    p = np.asarray(points, float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())


def _boundary_metrics(mesh, margin):
    loops = _npc._boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(loops) != 1:
        return None
    ring = np.asarray(mesh.vertices, float)[np.asarray(loops[0], np.int64)]
    margin = np.asarray(margin, float)
    distance = _polyline_distances(ring, margin)
    perimeter = _perimeter(ring)
    margin_perimeter = _perimeter(margin)
    return {
        "loop_vertices": int(len(ring)),
        "perimeter_mm": perimeter,
        "margin_perimeter_mm": margin_perimeter,
        "perimeter_ratio": float(perimeter / max(margin_perimeter, 1e-12)),
        "to_margin_median_mm": float(np.median(distance)) if len(distance) else None,
        "to_margin_p95_mm": float(np.quantile(distance, .95)) if len(distance) else None,
        "to_margin_max_mm": float(np.max(distance)) if len(distance) else None,
    }


def _phase_metrics(ring, margin):
    """Measure whether one complete transition loop follows the full Margin cycle."""
    ring128 = _resample_closed_polyline(np.asarray(ring, float), 128)
    margin128 = _resample_closed_polyline(np.asarray(margin, float), 128)
    base = np.arange(128, dtype=np.int64)
    best = None
    for reverse in (False, True):
        order0 = base[::-1] if reverse else base
        for shift in range(128):
            order = np.roll(order0, -shift)
            d = np.linalg.norm(ring128 - margin128[order], axis=1)
            key = (float(np.mean(d * d)), float(np.max(d)))
            if best is None or key < best[0]:
                best = (key, d, reverse, shift)
    d = best[1]
    return {
        "mse_mm2": float(best[0][0]),
        "median_mm": float(np.median(d)),
        "p95_mm": float(np.quantile(d, .95)),
        "max_mm": float(np.max(d)),
        "reversed": bool(best[2]),
        "shift": int(best[3]),
        "search": "EXHAUSTIVE_128_X_2",
    }


def strip_proven_artificial_basal_closure(source, curtain):
    """Return a W3-only open outer skin when a basal fan is unambiguously proven."""
    source = source.copy()
    loops_before = _npc._boundary_loops(np.asarray(source.faces, np.int64))
    base_report = {
        "method": "PROVEN_HIGH_VALENCE_BASAL_FAN_REMOVAL_V1",
        "source_faces": int(len(source.faces)),
        "source_vertices": int(len(source.vertices)),
        "source_boundary_loop_count": int(len(loops_before)),
        "accepted_outer_modified": False,
    }
    if len(loops_before) == 1:
        return source, {**base_report, "status": "UNCHANGED_ALREADY_OPEN"}
    if len(loops_before) != 0:
        return source, {
            **base_report,
            "status": "UNCHANGED_UNPROVEN",
            "reason": "SOURCE_HAS_MULTIPLE_OPEN_BOUNDARIES",
        }

    faces = np.asarray(source.faces, np.int64)
    vertices = np.asarray(source.vertices, float)
    valence = np.bincount(faces.reshape(-1), minlength=len(vertices))
    order = np.argsort(valence)[::-1]
    apex = int(order[0])
    apex_valence = int(valence[apex])
    second_valence = int(valence[order[1]]) if len(order) > 1 else 0
    positive = valence[valence > 0]
    median_valence = float(np.median(positive)) if len(positive) else 0.0
    apex_margin_distance = float(
        _polyline_distances(vertices[[apex]], curtain.points)[0]
    )
    incident = np.any(faces == apex, axis=1)
    incident_count = int(np.sum(incident))
    dots = np.asarray(source.face_normals, float)[incident] @ np.asarray(curtain.frame.z, float)
    opposing_fraction = float(np.mean(dots < 0.0)) if len(dots) else 0.0
    removed_area = float(np.asarray(source.area_faces, float)[incident].sum())
    removed_area_fraction = float(removed_area / max(float(source.area), 1e-12))

    proof = {
        "candidate_apex_vertex": apex,
        "candidate_apex_valence": apex_valence,
        "second_highest_valence": second_valence,
        "median_valence": median_valence,
        "valence_to_median_ratio": float(apex_valence / max(median_valence, 1e-12)),
        "valence_to_second_ratio": float(apex_valence / max(second_valence, 1)),
        "apex_to_margin_mm": apex_margin_distance,
        "incident_faces": incident_count,
        "opposing_axis_normal_fraction": opposing_fraction,
        "removed_area_mm2": removed_area,
        "removed_area_fraction": removed_area_fraction,
    }
    proven = bool(
        apex_valence >= _MIN_CAP_APEX_VALENCE
        and apex_valence >= _MIN_VALENCE_TO_MEDIAN_RATIO * max(median_valence, 1.0)
        and apex_valence >= _MIN_VALENCE_TO_SECOND_RATIO * max(second_valence, 1)
        and apex_margin_distance <= _MAX_APEX_TO_MARGIN_MM
        and opposing_fraction >= _MIN_OPPOSING_NORMAL_FRACTION
        and removed_area_fraction <= _MAX_CAP_AREA_FRACTION
    )
    if not proven:
        return source, {
            **base_report,
            **proof,
            "status": "UNCHANGED_UNPROVEN",
            "reason": "HIGH_VALENCE_BASAL_FAN_PROOF_FAILED",
        }

    derived = trimesh.Trimesh(
        vertices.copy(), faces[~incident].copy(), process=False
    )
    derived.remove_unreferenced_vertices()
    components = [p for p in derived.split(only_watertight=False) if len(p.faces) >= 3]
    loops_after = _npc._boundary_loops(np.asarray(derived.faces, np.int64))
    metrics = _boundary_metrics(derived, curtain.points)
    post_valid = bool(
        len(components) == 1
        and len(loops_after) == 1
        and metrics is not None
        and _POST_LOOP_PERIMETER_RATIO[0] <= metrics["perimeter_ratio"] <= _POST_LOOP_PERIMETER_RATIO[1]
        and metrics["to_margin_p95_mm"] <= _POST_LOOP_TO_MARGIN_P95_MM
        and metrics["to_margin_max_mm"] <= _POST_LOOP_TO_MARGIN_MAX_MM
    )
    if not post_valid:
        return source, {
            **base_report,
            **proof,
            "status": "UNCHANGED_UNPROVEN",
            "reason": "REMOVAL_DID_NOT_PRODUCE_ONE_CLINICAL_CERVICAL_LOOP",
            "derived_component_count": int(len(components)),
            "derived_boundary_loop_count": int(len(loops_after)),
            "derived_boundary": metrics,
        }

    return derived, {
        **base_report,
        **proof,
        "status": "REMOVED_DERIVED_W3_ONLY",
        "derived_component_count": 1,
        "derived_boundary_loop_count": 1,
        "derived_boundary": metrics,
        "removed_face_count": incident_count,
        "clinical_meaning": "ARTIFICIAL_TEMPLATE_BASAL_CLOSURE_NOT_INTAGLIO",
    }


_ORIGINAL_SINGLE_OPEN = _npc._single_open_surface_above_margin


def _outer_clinical_skin_single_open(source, curtain, transition_mm, role, *, invert=False):
    if str(role).upper() != "OUTER":
        return _ORIGINAL_SINGLE_OPEN(source, curtain, transition_mm, role, invert=invert)
    clinical, guard = strip_proven_artificial_basal_closure(source, curtain)
    surface, report = _ORIGINAL_SINGLE_OPEN(
        clinical, curtain, transition_mm, role, invert=invert
    )
    report = dict(report)
    report["outer_clinical_skin_guard"] = guard
    report["source_for_w3_faces"] = int(len(clinical.faces))
    report["accepted_outer_faces"] = int(len(source.faces))
    return surface, report


_npc._single_open_surface_above_margin = _outer_clinical_skin_single_open


_ORIGINAL_FEASIBILITY = _policy.cervical_feasibility


def _full_margin_cycle_feasibility(outer, inner, margin_points):
    feasible, report = _ORIGINAL_FEASIBILITY(outer, inner, margin_points)
    report = dict(report)
    if not feasible:
        return feasible, report

    loops = _npc._boundary_loops(np.asarray(outer.faces, np.int64))
    if len(loops) != 1:
        return feasible, report
    ring = np.asarray(outer.vertices, float)[np.asarray(loops[0], np.int64)]
    margin = np.asarray(margin_points, float)
    ring_perimeter = _perimeter(ring)
    margin_perimeter = _perimeter(margin)
    ratio = float(ring_perimeter / max(margin_perimeter, 1e-12))
    phase = _phase_metrics(ring, margin)
    max_limit = max(
        _TRANSITION_PHASE_MAX_FLOOR_MM,
        _TRANSITION_PHASE_MAX_FRACTION * margin_perimeter,
    )
    p95_limit = max(
        _TRANSITION_PHASE_P95_FLOOR_MM,
        _TRANSITION_PHASE_P95_FRACTION * margin_perimeter,
    )
    cycle = {
        "status": "PASS_ENGINEERING",
        "outer_ring_vertices": int(len(ring)),
        "outer_ring_perimeter_mm": ring_perimeter,
        "margin_perimeter_mm": margin_perimeter,
        "perimeter_ratio": ratio,
        "phase": phase,
        "phase_max_limit_mm": float(max_limit),
        "phase_p95_limit_mm": float(p95_limit),
    }
    valid = bool(
        _TRANSITION_PERIMETER_RATIO[0] <= ratio <= _TRANSITION_PERIMETER_RATIO[1]
        and phase["max_mm"] <= max_limit
        and phase["p95_mm"] <= p95_limit
    )
    if not valid:
        cycle.update(
            status="REVIEW_REQUIRED",
            reason="OUTER_TRANSITION_IS_NOT_A_FULL_MARGIN_CYCLE",
        )
        report["outer_margin_cycle"] = cycle
        report.update(
            status="REVIEW_REQUIRED",
            reason="OUTER_TRANSITION_IS_NOT_A_FULL_MARGIN_CYCLE",
        )
        return False, report

    report["outer_margin_cycle"] = cycle
    return True, report


# v1_margin_owned_intaglio imported the function into its module namespace, so
# patch that exact binding rather than only patching canonical_intaglio.
_policy.cervical_feasibility = _full_margin_cycle_feasibility
