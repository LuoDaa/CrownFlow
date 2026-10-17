"""CrownFlow 1.0 intaglio support guard.

Real IOS abutment scans can produce a *second tiny boundary loop on the confirmed
Margin itself* after the non-planar Margin clip. That loop is a derived clip sliver,
not a clinical support hole. This module recovers only that narrowly proven case.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .geometry import abutment_support as _support
from .geometry.crown_outer.collar_body_outer import _boundary_loops

SLIVER_LOOP_TO_MARGIN_P95_MM = 0.02
SLIVER_LOOP_TO_MARGIN_MAX_MM = 0.03
SLIVER_MAX_MARGIN_COVERAGE = 0.20
SLIVER_MAX_LENGTH_RATIO = 0.18
SLIVER_MAX_PERIMETER_MM = 2.50
SLIVER_MATCH_P95_MM = 0.02

_ORIGINAL_EXTRACT_ABUTMENT_SUPPORT = _support.extract_abutment_support


def _is_margin_split_sliver_metrics(metrics: dict) -> bool:
    try:
        return bool(
            not metrics.get("eligible")
            and float(metrics.get("loop_to_margin_p95_mm", np.inf)) <= SLIVER_LOOP_TO_MARGIN_P95_MM
            and float(metrics.get("loop_to_margin_max_mm", np.inf)) <= SLIVER_LOOP_TO_MARGIN_MAX_MM
            and float(metrics.get("coverage_fraction", 1.0)) <= SLIVER_MAX_MARGIN_COVERAGE
            and float(metrics.get("length_ratio", 1.0)) <= SLIVER_MAX_LENGTH_RATIO
            and float(metrics.get("loop_length_mm", np.inf)) <= SLIVER_MAX_PERIMETER_MM
        )
    except (TypeError, ValueError):
        return False


def _loop_id(points) -> str:
    return sha256(np.round(np.sort(np.asarray(points, float), axis=0), 6).tobytes()).hexdigest()[:20]


def _loop_match_score(current_points, expected_points):
    """Geometry match independent of boundary start index/hash drift."""
    current = np.asarray(current_points, float)
    expected = np.asarray(expected_points, float)
    if len(current) < 3 or len(expected) < 3:
        return np.inf
    a = cKDTree(expected).query(current)[0]
    b = cKDTree(current).query(expected)[0]
    values = np.r_[a, b]
    return float(np.quantile(values, .95)) if len(values) else np.inf


def _find_current_loop(mesh, entry):
    """Find the reported sliver in the current derived mesh robustly."""
    wanted_id = str(entry.get("loop_id") or "")
    expected = np.asarray(entry.get("points") or [], float)
    best = None
    best_score = np.inf
    for loop in _boundary_loops(np.asarray(mesh.faces, np.int64)):
        arr = np.asarray(loop, int)
        points = np.asarray(mesh.vertices, float)[arr]
        if wanted_id and _loop_id(points) == wanted_id:
            return arr, 0.0, "EXACT_LOOP_ID"
        score = _loop_match_score(points, expected)
        if score < best_score:
            best, best_score = arr, score
    if best is not None and best_score <= SLIVER_MATCH_P95_MM:
        return best, float(best_score), "GEOMETRIC_P95_MATCH"
    return None, float(best_score), "NO_MATCH"


def _fan_cap(mesh: trimesh.Trimesh, loop):
    """Close a tiny proven sliver with a robust center fan.

    Ear clipping on this highly non-planar ~1 mm Margin fragment was the remaining
    failure point in FDI36. A center fan uses the already ordered boundary and is
    deterministic. It is applied only to a sliver that has already passed all
    Margin-proximity/size gates, and the new faces remain tagged as virtual caps.
    """
    loop = np.asarray(loop, np.int64).reshape(-1)
    if len(loop) < 3:
        raise ValueError("margin split sliver has fewer than 3 vertices")
    vertices = np.asarray(mesh.vertices, float).copy()
    center_id = len(vertices)
    center = np.asarray(vertices[loop], float).mean(axis=0)
    vertices = np.vstack((vertices, center))
    triangles = np.asarray([[int(loop[i]), int(loop[(i + 1) % len(loop)]), center_id]
                            for i in range(len(loop))], np.int64)

    u, w = map(int, loop[:2])
    existing = next((face for face in np.asarray(mesh.faces, int) if u in face and w in face), None)
    if existing is None:
        raise ValueError("margin split sliver boundary has no adjacent source face")
    source_direction = any(existing[k] == u and existing[(k + 1) % 3] == w for k in range(3))
    cap_direction = triangles[0, 0] == u and triangles[0, 1] == w
    if source_direction == cap_direction:
        triangles = triangles[:, [1, 0, 2]]

    start = len(mesh.faces)
    out = trimesh.Trimesh(vertices=vertices,
                          faces=np.vstack((np.asarray(mesh.faces, np.int64), triangles)),
                          process=False)
    _support._manifold(out)
    face_ids = list(range(start, len(out.faces)))
    area = float(np.asarray(out.area_faces)[face_ids].sum()) if face_ids else 0.0
    return out, face_ids, area


def _recover_margin_split_slivers(prepared, margin):
    report = prepared.report
    if report.get("rejection_code") != "UNKNOWN_SUPPORT_HOLE":
        return prepared

    classified = list(report.get("loops") or [])
    recovery_report = {
        "status": "NOT_RECOVERED",
        "reason": None,
        "input_boundary_loop_count": int(report.get("boundary_loop_count") or 0),
    }
    report["margin_split_sliver_recovery"] = recovery_report

    pending_screw = [entry for entry in classified
                     if entry.get("classification") == "SCREW_CHANNEL_CANDIDATE"
                     and not entry.get("confirmed")]
    if pending_screw:
        recovery_report["reason"] = "UNCONFIRMED_SCREW_CHANNEL_PRESENT"
        return prepared

    unknown = [entry for entry in classified if entry.get("classification") == "UNKNOWN_HOLE"]
    if not unknown:
        recovery_report["reason"] = "NO_UNKNOWN_HOLE"
        return prepared

    curve = np.asarray(margin, float)
    candidates = []
    for entry in unknown:
        points = np.asarray(entry.get("points") or [], float)
        metrics = _support._margin_loop_metrics(points, curve)
        entry["margin_split_sliver_metrics"] = metrics
        if _is_margin_split_sliver_metrics(metrics):
            candidates.append((entry, metrics))

    recovery_report["unknown_loop_count"] = len(unknown)
    recovery_report["proven_sliver_count"] = len(candidates)
    if len(candidates) != len(unknown):
        recovery_report["reason"] = "NOT_EVERY_UNKNOWN_LOOP_IS_A_PROVEN_MARGIN_SLIVER"
        return prepared

    mesh = prepared.mesh.copy()
    operations = list(report.get("virtual_cap_operations") or [])
    cap_faces = list(report.get("virtual_cap_face_ids") or [])
    recovered = []

    for entry, metrics in candidates:
        match, match_p95, match_method = _find_current_loop(mesh, entry)
        if match is None:
            recovery_report.update(reason="CURRENT_BOUNDARY_MATCH_FAILED",
                                   boundary_match_p95_mm=match_p95,
                                   boundary_match_method=match_method)
            return prepared
        before = _support.mesh_digest(mesh)
        try:
            mesh, new_faces, area = _fan_cap(mesh, match)
        except Exception as exc:
            recovery_report.update(reason="VIRTUAL_SLIVER_CAP_FAILED", error=str(exc)[:240],
                                   boundary_match_p95_mm=match_p95,
                                   boundary_match_method=match_method)
            return prepared
        after = _support.mesh_digest(mesh)
        cap_faces.extend(new_faces)
        operation = {
            "operation": "VIRTUAL_MARGIN_SPLIT_SLIVER_CAP",
            "loop_id": entry.get("loop_id"),
            "before_sha256": before,
            "after_sha256": after,
            "cap_face_ids": new_faces,
            "area_mm2": area,
            "reason": "DERIVED_NONPLANAR_MARGIN_CLIP_SLIVER_NOT_SOURCE_HOLE",
            "metrics": metrics,
            "boundary_match_method": match_method,
            "boundary_match_p95_mm": match_p95,
            "triangulation": "ORDERED_BOUNDARY_CENTER_FAN",
        }
        operations.append(operation)
        recovered.append(operation)
        entry["classification"] = "MARGIN_SPLIT_SLIVER_RECOVERED"
        entry["confirmed"] = True

    remaining = _boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(remaining) != 1:
        recovery_report.update(reason="RECOVERY_DID_NOT_LEAVE_ONE_MARGIN_LOOP",
                               remaining_boundary_loop_count=len(remaining))
        return prepared
    final_points = np.asarray(mesh.vertices, float)[np.asarray(remaining[0], int)]
    final_margin = _support._margin_loop_metrics(final_points, curve)
    if not final_margin.get("eligible"):
        recovery_report.update(reason="REMAINING_LOOP_DOES_NOT_REPRODUCE_FULL_MARGIN",
                               final_margin_loop=final_margin)
        return prepared

    prepared.mesh = mesh
    report["loops"] = classified
    report["virtual_cap_operations"] = operations
    report["virtual_cap_face_ids"] = cap_faces
    report["capped_region_area_mm2"] = float(sum(float(op.get("area_mm2") or 0.0) for op in operations))
    report["output_face_source_ids"] = list(report.get("output_face_source_ids") or []) + [-1] * sum(
        len(op.get("cap_face_ids") or []) for op in recovered)
    report["unresolved_support_holes"] = 0
    report["conditioned_support_sha256"] = _support.mesh_digest(mesh)
    report["readiness"] = "ABUTMENT_SUPPORT_READY"
    report["rejection_code"] = "READY"
    report["boundary_loop_count"] = 1
    recovery_report.update({
        "status": "RECOVERED",
        "reason": None,
        "recovered_loop_count": len(recovered),
        "remaining_boundary_loop_count": 1,
        "final_margin_loop": final_margin,
        "source_support_modified": False,
        "policy": "CAP_ONLY_PROVEN_DERIVED_MARGIN_SLIVERS_NEVER_ARBITRARY_HOLES",
    })
    report.setdefault("warnings", []).append(
        "检测到非平面 Margin 裁切产生的微小重复边界；仅在派生 fitting support 上虚拟封闭，原始基台扫描未修改。"
    )
    return prepared


def extract_abutment_support_v1(*args, **kwargs):
    prepared = _ORIGINAL_EXTRACT_ABUTMENT_SUPPORT(*args, **kwargs)
    margin = args[1] if len(args) > 1 else kwargs.get("margin")
    if prepared is None or margin is None:
        return prepared
    return _recover_margin_split_slivers(prepared, margin)


_support.extract_abutment_support = extract_abutment_support_v1
