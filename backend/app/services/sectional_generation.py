"""Live adapter for the RED/GREEN/BLUE Sectional Crown 1.0 geometry.

This service resolves only *confirmed* patient context, creates the Sectional
baseline, then applies a GREEN-only neighbour-facing surface pass.  The pass is
strictly isolated from the hard Margin, BLUE occlusal rim/cap and the downstream
W3 intaglio path, so Crown 1.0 can improve neighbour contact morphology without
regressing the already-working inner crown or occlusion.
"""
from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import CASE_DIR
from ..geometry.crown_outer.proximal_rounding import (
    ProximalRoundingConfig,
    round_proximal_contacts,
)
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
    """Create and persist the Crown 1.0 patient-geometry baseline for AUTO/BOTH.

    RED  : exact confirmed Margin + prep/minimum-wall safety.
    GREEN: neighbour inward-facing patch -> broad contact band -> rounded BL loft.
    BLUE : unchanged antagonist ceiling / static occlusal capability.

    Final intaglio is intentionally still delegated to the existing W3 blockout +
    variable-gap + boolean path; this function never changes inner-crown policy.
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

    # The Sectional baseline still provides RED/BLUE hard geometry and a safe
    # initial GREEN envelope.  The following patch-surface stage owns the final
    # proximal shape, so these values are intentionally permissive rather than
    # trying to solve contact with isolated rays.
    config = SectionalV1Config(
        min_wall_mm=float(minimum),
        proximal_gap_mm=float(proximal),
        occlusal_gap_mm=float(occlusal),
        contact_height_ratio=0.62,
        top_inset_mm=0.20,
        neighbor_sector_cos=0.32,
        neighbor_ray_band_mm=2.00,
        max_proximal_extension_mm=3.00,
        side_levels=(0.0, 0.10, 0.22, 0.36, 0.50, 0.62, 0.73, 0.83, 0.92, 1.0),
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

    # GREEN-only surface stage.  Real IOS neighbour patches can be dense, sparse,
    # or locally uneven depending on the click region.  A ~2.8-mm Y/Z query radius
    # is deliberately wider than the old ray band, but the actual movement is
    # still bounded by the measured facing X surface and max_proximal_extension.
    proximal_config = ProximalRoundingConfig(
        facing_query_radius_mm=2.80,
        facing_k=20,
        contact_band_pad_mm=0.30,
        bl_width_scale=0.94,
        bl_strength=0.72,
        smoothing_passes=4,
        smoothing_alpha=0.16,
    )
    mesh, proximal_surface = round_proximal_contacts(
        mesh,
        margin_points,
        frame,
        neighbors=neighbors,
        sectional_config=config,
        config=proximal_config,
    )
    geometry["green_surface"] = proximal_surface
    # Compatibility key retained for existing reports/UI consumers.
    geometry["green_rounding"] = proximal_surface
    for side in ("mesial", "distal"):
        if side in geometry.get("green_lateral", {}):
            geometry["green_lateral"][side]["post_rounding"] = (
                proximal_surface.get("per_side", {}).get(side, {}))

    prefix = CASE_DIR / d["case_id"] / ("%s_sectional_v1_%s" % (d["id"], uuid4().hex))
    outer_path = prefix.with_suffix(".stl")
    report_path = prefix.with_name(prefix.name + "_report.json")
    save_mesh(mesh, outer_path)

    confirmed_sides = [side for side, value in neighbor_provenance.items()
                       if value.get("status") == "CONFIRMED"]
    proximal_statuses = [
        proximal_surface.get("per_side", {}).get(side, {}).get("status")
        for side in confirmed_sides
    ]
    lateral_status = (
        "PASS_ENGINEERING" if proximal_statuses
        and all(value == "PASS_ENGINEERING" for value in proximal_statuses)
        else "REVIEW_REQUIRED" if proximal_statuses else "NOT_ASSESSED"
    )

    report = {
        "type": "sectional_crown_v1",
        "candidate_kind": "AUTO_FIT_CANDIDATE",
        "generator": "SECTIONAL_RED_GREEN_BLUE_V1",
        "status": geometry["status"],
        "solver_version": "sectional-envelope/1 + proximal-facing-surface/2 + local-band-fine-fit/1",
        "stages": [
            {"stage": "MARGIN", "status": "PASS_ENGINEERING",
             "max_error_mm": geometry["margin"]["max_error_mm"], "hard_boundary": True},
            {"stage": "PROXIMAL", "status": lateral_status,
             "per_side": geometry["green_lateral"],
             "surface": proximal_surface},
            {"stage": "OCCLUSAL", "status": geometry["blue_occlusal"]["status"],
             "details": geometry["blue_occlusal"],
             "preserved_by_proximal_surface": bool(
                 proximal_surface.get("blue_occlusal_vertices_preserved")),
             "axial_heights_preserved": bool(
                 proximal_surface.get("axial_heights_preserved"))},
            {"stage": "INTAGLIO", "status": "DEFERRED_TO_W3_PREFLIGHT",
             "mechanism": "prepared_support -> blockout -> gap field -> exact boolean",
             "policy_changed_by_proximal_surface": False},
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
