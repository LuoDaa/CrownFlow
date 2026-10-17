"""CrownFlow 1.0 intaglio support guard.

Real IOS abutment scans can produce a *second tiny boundary loop on the confirmed
Margin itself* after the non-planar Margin clip.  That loop is not a real support
hole: it is a derived clip sliver.  Treating it as ``UNKNOWN_SUPPORT_HOLE`` blocks
intaglio generation even when the authoritative Margin matches the abutment at
micron scale.

This override keeps the safety policy for real holes unchanged.  It only recovers
a loop when all of the following are proven geometrically:

* the whole tiny loop lies essentially on the confirmed Margin;
* it covers only a small fraction of the full Margin;
* its perimeter is a small fraction of the authoritative Margin perimeter;
* there is no unconfirmed screw-channel candidate.

Only that derived sliver is virtually capped.  The source support is never
modified, arbitrary scan holes are never filled, and the operation is recorded in
the prepared-support provenance.  After recovery the prepared fitting support
must have exactly one remaining boundary loop and that loop must still reproduce
the entire confirmed Margin before readiness is upgraded.
"""
from __future__ import annotations

from hashlib import sha256

import numpy as np
import trimesh

from .geometry import abutment_support as _support
from .geometry.crown_outer.collar_body_outer import _boundary_loops


# Deliberately strict: the FDI36 failing case had a ~1.15 mm loop sitting at
# numerical-zero distance from Margin while the true Margin perimeter was
# ~19.53 mm.  These gates recover that topology artifact without relaxing the
# policy for genuine support holes.
SLIVER_LOOP_TO_MARGIN_P95_MM = 0.02
SLIVER_LOOP_TO_MARGIN_MAX_MM = 0.03
SLIVER_MAX_MARGIN_COVERAGE = 0.20
SLIVER_MAX_LENGTH_RATIO = 0.18
SLIVER_MAX_PERIMETER_MM = 2.50

_ORIGINAL_EXTRACT_ABUTMENT_SUPPORT = _support.extract_abutment_support


def _is_margin_split_sliver_metrics(metrics: dict) -> bool:
    """Return True only for a tiny derived loop proven to lie on Margin."""
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


def _triangulate_cap(mesh: trimesh.Trimesh, loop) -> tuple[trimesh.Trimesh, list[int], float]:
    """Cap one proven derived sliver while preserving all existing vertices/faces."""
    import mapbox_earcut

    loop = np.asarray(loop, dtype=np.int64).reshape(-1)
    if len(loop) < 3:
        raise ValueError("margin split sliver has fewer than 3 vertices")
    points = np.asarray(mesh.vertices, float)[loop]
    center = points.mean(axis=0)
    _u, _s, basis = np.linalg.svd(points - center, full_matrices=False)
    xy = np.ascontiguousarray((points - center) @ basis[:2].T, dtype=np.float64)
    indices = mapbox_earcut.triangulate_float64(xy, np.array([len(loop)], np.uint32))
    if not len(indices):
        raise ValueError("margin split sliver triangulation is empty")
    triangles = loop[np.asarray(indices, np.int64).reshape(-1, 3)]

    # Match the orientation convention already used by the confirmed screw-hole
    # cap path: the new cap must traverse the shared boundary opposite to the
    # existing side face.
    u, w = map(int, loop[:2])
    existing = next((face for face in np.asarray(mesh.faces, int) if u in face and w in face), None)
    if existing is None:
        raise ValueError("margin split sliver boundary has no adjacent source face")
    source_direction = any(existing[k] == u and existing[(k + 1) % 3] == w for k in range(3))
    cap_direction = any(face[k] == u and face[(k + 1) % 3] == w for face in triangles for k in range(3))
    if source_direction == cap_direction:
        triangles = triangles[:, ::-1]

    start = len(mesh.faces)
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, float).copy(),
        faces=np.vstack((np.asarray(mesh.faces, np.int64), triangles)),
        process=False,
    )
    _support._manifold(out)
    face_ids = list(range(start, len(out.faces)))
    area = float(np.asarray(out.area_faces)[face_ids].sum()) if face_ids else 0.0
    return out, face_ids, area


def _recover_margin_split_slivers(prepared, margin):
    report = prepared.report
    if report.get("rejection_code") != "UNKNOWN_SUPPORT_HOLE":
        return prepared

    classified = list(report.get("loops") or [])
    pending_screw = [
        entry for entry in classified
        if entry.get("classification") == "SCREW_CHANNEL_CANDIDATE" and not entry.get("confirmed")
    ]
    if pending_screw:
        return prepared

    unknown = [entry for entry in classified if entry.get("classification") == "UNKNOWN_HOLE"]
    if not unknown:
        return prepared

    curve = np.asarray(margin, float)
    candidates = {}
    for entry in unknown:
        points = np.asarray(entry.get("points") or [], float)
        metrics = _support._margin_loop_metrics(points, curve)
        entry["margin_split_sliver_metrics"] = metrics
        if _is_margin_split_sliver_metrics(metrics):
            candidates[str(entry.get("loop_id"))] = (entry, metrics)

    # Never turn a genuine unknown hole into a warning.  Every UNKNOWN_HOLE must
    # independently satisfy the strict sliver proof before recovery proceeds.
    if len(candidates) != len(unknown):
        return prepared

    mesh = prepared.mesh.copy()
    operations = list(report.get("virtual_cap_operations") or [])
    cap_faces = list(report.get("virtual_cap_face_ids") or [])
    recovered = []

    # Re-identify each current boundary by the same stable loop hash used by the
    # extraction module.  Recompute after every cap because the boundary set has
    # changed, but existing vertex indices remain untouched.
    for wanted_id, (entry, metrics) in candidates.items():
        match = None
        for loop in _boundary_loops(np.asarray(mesh.faces, np.int64)):
            points = np.asarray(mesh.vertices, float)[np.asarray(loop, int)]
            if _loop_id(points) == wanted_id:
                match = np.asarray(loop, int)
                break
        if match is None:
            return prepared
        before = _support.mesh_digest(mesh)
        try:
            mesh, new_faces, area = _triangulate_cap(mesh, match)
        except Exception:
            return prepared
        after = _support.mesh_digest(mesh)
        cap_faces.extend(new_faces)
        operation = {
            "operation": "VIRTUAL_MARGIN_SPLIT_SLIVER_CAP",
            "loop_id": wanted_id,
            "before_sha256": before,
            "after_sha256": after,
            "cap_face_ids": new_faces,
            "area_mm2": area,
            "reason": "DERIVED_NONPLANAR_MARGIN_CLIP_SLIVER_NOT_SOURCE_HOLE",
            "metrics": metrics,
        }
        operations.append(operation)
        recovered.append(operation)
        entry["classification"] = "MARGIN_SPLIT_SLIVER_RECOVERED"
        entry["confirmed"] = True

    remaining = _boundary_loops(np.asarray(mesh.faces, np.int64))
    if len(remaining) != 1:
        return prepared
    final_points = np.asarray(mesh.vertices, float)[np.asarray(remaining[0], int)]
    final_margin = _support._margin_loop_metrics(final_points, curve)
    if not final_margin.get("eligible"):
        return prepared

    # This is now a valid single-loop fitting surface.  Keep the recovery visible
    # in provenance; do not pretend the input scan itself was repaired.
    prepared.mesh = mesh
    report["loops"] = classified
    report["virtual_cap_operations"] = operations
    report["virtual_cap_face_ids"] = cap_faces
    report["capped_region_area_mm2"] = float(sum(float(op.get("area_mm2") or 0.0) for op in operations))
    report["output_face_source_ids"] = list(report.get("output_face_source_ids") or []) + [-1] * sum(
        len(op.get("cap_face_ids") or []) for op in recovered
    )
    report["unresolved_support_holes"] = 0
    report["conditioned_support_sha256"] = _support.mesh_digest(mesh)
    report["readiness"] = "ABUTMENT_SUPPORT_READY"
    report["rejection_code"] = "READY"
    report["boundary_loop_count"] = 1
    report["margin_split_sliver_recovery"] = {
        "status": "RECOVERED",
        "recovered_loop_count": len(recovered),
        "remaining_boundary_loop_count": 1,
        "final_margin_loop": final_margin,
        "source_support_modified": False,
        "policy": "CAP_ONLY_PROVEN_DERIVED_MARGIN_SLIVERS_NEVER_ARBITRARY_HOLES",
    }
    report.setdefault("warnings", []).append(
        "检测到由非平面 Margin 裁切产生的微小重复边界，已仅在派生 fitting support 上虚拟封闭；原始基台扫描未修改。"
    )
    return prepared


def extract_abutment_support_v1(*args, **kwargs):
    prepared = _ORIGINAL_EXTRACT_ABUTMENT_SUPPORT(*args, **kwargs)
    margin = args[1] if len(args) > 1 else kwargs.get("margin")
    if prepared is None or margin is None:
        return prepared
    return _recover_margin_split_slivers(prepared, margin)


# Imported before services.abutment_support binds the symbol, so all existing
# generation/finalization code consumes the guarded recovery without another pass.
_support.extract_abutment_support = extract_abutment_support_v1
