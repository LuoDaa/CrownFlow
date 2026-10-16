"""Live adapter for the RED/GREEN/BLUE Sectional Crown 1.0 geometry.

This service is deliberately thin: it resolves only *confirmed* patient context
from the design, calls the pure geometry builder, persists the candidate baseline,
and emits provenance.  Fine signed-distance correction and W3 intaglio preflight
remain in ``services.auto_fit.propose`` so the existing QC truth sources stay in
one place.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR
from ..geometry.crown_outer.sectional_v1 import SectionalV1Config, build_sectional_outer
from ..geometry.io import load_mesh, save_mesh
from .design_dependencies import file_digest


def _asset(d, kind):
    # Local import avoids a module cycle during API startup.
    from ..api.routes import asset
    return asset(d["case_id"], kind)


def _confirmed_neighbor_meshes(d, frame_ref):
    neighbors = json.loads(d.get("neighbors_json") or "{}")
    meshes = {}
    provenance = {}
    for side in ("mesial", "distal"):
        entry = neighbors.get(side, {})
        status = entry.get("confirmation_status", "UNKNOWN")
        provenance[side] = {
            "confirmation_status": status,
            "revision_id": entry.get("revision_id"),
            "frame_revision_id": entry.get("frame_revision_id"),
        }
        if status == "ABSENT":
            meshes[side] = None
            provenance[side]["status"] = "ABSENT_CONFIRMED"
            continue
        if status != "CONFIRMED" or entry.get("frame_revision_id") != frame_ref.get("revision_id"):
            raise ValueError("SECTIONAL_V1_NEIGHBOR_UNVERIFIED: %s" % side)
        path = entry.get("path")
        source = entry.get("source") or {}
        current = _asset(d, source.get("kind")) if source.get("kind") else None
        if (not current or current.get("id") != source.get("asset_id") or not path
                or not Path(path).is_file()
                or file_digest(current["path"]) != source.get("sha256")
                or file_digest(path) != entry.get("region_sha256")):
            raise ValueError("SECTIONAL_V1_NEIGHBOR_SOURCE_CHANGED: %s" % side)
        meshes[side] = load_mesh(Path(path))
        provenance[side].update(
            status="CONFIRMED",
            source=source,
            region_sha256=entry.get("region_sha256"),
            outward_normal=entry.get("outward_normal"),
        )
    return meshes, provenance


def _confirmed_antagonist(d, frame_ref):
    bite = json.loads(d.get("bite_json") or "{}")
    if not (bite.get("revision_id") and bite.get("pose_confirmed") and bite.get("transform")
            and bite.get("frame_revision_id") == frame_ref.get("revision_id")):
        raise ValueError("SECTIONAL_V1_OCCLUSION_REGISTRATION_UNVERIFIED")
    transform = np.asarray(bite["transform"], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("SECTIONAL_V1_BITE_TRANSFORM_INVALID")
    for source in bite.get("scans", []):
        current = _asset(d, source.get("kind"))
        if not current or current.get("id") != source.get("asset_id") or file_digest(current["path"]) != source.get("sha256"):
            raise ValueError("SECTIONAL_V1_BITE_SOURCE_CHANGED")
    antagonist_kind = "upper" if int(d["target_fdi"]) >= 31 else "lower"
    ref = _asset(d, antagonist_kind)
    if not ref:
        raise ValueError("SECTIONAL_V1_ANTAGONIST_MISSING")
    antagonist = load_mesh(Path(ref["path"]))
    antagonist.apply_transform(transform)
    return antagonist, {
        "revision_id": bite.get("revision_id"),
        "frame_revision_id": bite.get("frame_revision_id"),
        "source_method": bite.get("source_method"),
        "antagonist_kind": antagonist_kind,
        "asset_id": ref.get("id"),
        "asset_sha256": file_digest(ref["path"]),
    }


def build_sectional_baseline(d, frame, frame_ref, support_mesh, margin_points):
    """Create and persist a Sectional Crown 1.0 baseline for ``AUTO/BOTH``.

    The geometry builder already enforces the user's cross-section contract:
    RED prep/min-wall, GREEN broad proximal envelope, BLUE antagonist cap.
    This function does not run final intaglio boolean; the caller immediately
    hands the baseline to ``auto_fit.propose`` which performs the established
    fine correction and W3/finalize preflight.
    """
    params = json.loads(d.get("parameters_json") or "{}")
    proximal = params.get("proximal_target_mm")
    occlusal = params.get("occlusal_target_mm")
    if proximal is None or occlusal is None:
        raise ValueError("SECTIONAL_V1_REQUIRES_CONFIRMED_PROXIMAL_AND_OCCLUSAL_TARGETS")
    minimum = params.get("minimum_thickness_mm")
    if minimum is None:
        minimum = 0.50

    neighbors, neighbor_provenance = _confirmed_neighbor_meshes(d, frame_ref)
    antagonist, bite_provenance = _confirmed_antagonist(d, frame_ref)
    config = SectionalV1Config(
        min_wall_mm=float(minimum),
        proximal_gap_mm=float(proximal),
        occlusal_gap_mm=float(occlusal),
    )
    mesh, geometry = build_sectional_outer(
        margin_points,
        frame,
        support_mesh,
        neighbors=neighbors,
        antagonist_mesh=antagonist,
        fdi=int(d["target_fdi"]),
        config=config,
    )

    prefix = CASE_DIR / d["case_id"] / ("%s_sectional_v1_%s" % (d["id"], uuid4().hex))
    outer_path = prefix.with_suffix(".stl")
    report_path = prefix.with_name(prefix.name + "_report.json")
    save_mesh(mesh, outer_path)

    confirmed_sides = [side for side, value in neighbor_provenance.items()
                       if value.get("status") == "CONFIRMED"]
    green_statuses = [geometry["green_lateral"][side].get("status") for side in confirmed_sides]
    lateral_status = ("PASS_ENGINEERING" if green_statuses
                      and all(value == "PASS_ENGINEERING" for value in green_statuses)
                      else "REVIEW_REQUIRED" if green_statuses else "NOT_ASSESSED")
    report = {
        "type": "sectional_crown_v1",
        "candidate_kind": "AUTO_FIT_CANDIDATE",
        "generator": "SECTIONAL_RED_GREEN_BLUE_V1",
        "status": geometry["status"],
        "solver_version": "sectional-envelope/1 + local-band-fine-fit/1",
        "stages": [
            {"stage": "MARGIN", "status": "PASS_ENGINEERING",
             "max_error_mm": geometry["margin"]["max_error_mm"], "hard_boundary": True},
            {"stage": "PROXIMAL", "status": lateral_status,
             "per_side": geometry["green_lateral"]},
            {"stage": "OCCLUSAL", "status": geometry["blue_occlusal"]["status"],
             "details": geometry["blue_occlusal"]},
            {"stage": "INTAGLIO", "status": "DEFERRED_TO_W3_PREFLIGHT",
             "mechanism": "prepared_support -> blockout -> gap field -> exact boolean"},
        ],
        "sectional_geometry": geometry,
        "neighbor_provenance": neighbor_provenance,
        "bite_provenance": bite_provenance,
        "patient_frame": {
            "revision_id": frame_ref.get("revision_id"),
            "source": "CONFIRMED",
            "x": np.asarray(frame.x, float).tolist(),
            "y": np.asarray(frame.y, float).tolist(),
            "z": np.asarray(frame.z, float).tolist(),
        },
        "placement_path": "SECTIONAL_RED_GREEN_BLUE_V1",
        "template_source": {
            "source_kind": "WEAK_ANATOMY_NOT_USED_FOR_ENVELOPE",
            "note": "Crown 1.0 envelope is patient-geometry driven; template is not the size/height truth source",
        },
        "candidate_outer_sha256": file_digest(outer_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = {"outer": str(outer_path), "sectional_report": str(report_path)}
    report["artifact_sha256"] = {key: file_digest(path) for key, path in artifacts.items()}
    return mesh, report, artifacts
