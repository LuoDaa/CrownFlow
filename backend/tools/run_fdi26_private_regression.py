"""Private FDI26 Crown 1.0 regression runner.

The raw scanner STLs stay local.  The runner now exercises the same RED/GREEN/BLUE
geometry used by live AUTO generation:

    RED   = prep / minimum-wall safety envelope
    GREEN = broad mesial/distal envelope from the same-jaw arch
    BLUE  = antagonist-constrained occlusal cap

It also exports the prep-driven blockout + gap cutter used as the future intaglio.
This is regression tooling only; raw-arch neighbor crops are explicitly NOT a
clinical substitute for confirmed neighbor patches in the application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.geometry.axis import suggest_insertion_axis  # noqa: E402
from app.geometry.crown_outer.patient_frame import (  # noqa: E402
    build_patient_frame,
    confirmed_patient_frame,
)
from app.geometry.crown_outer.sectional_v1 import (  # noqa: E402
    SectionalOuterError,
    SectionalV1Config,
    build_sectional_outer,
)
from app.geometry.intaglio import (  # noqa: E402
    BlockoutPolicy,
    GapFieldParams,
    apply_gap_field,
    build_blockout,
)
from app.geometry.nonplanar_crown import fitting_support_above_margin  # noqa: E402
from app.geometry.scan_pair_triage import (  # noqa: E402
    major_components,
    pair_report,
    select_separate_support,
)
from app.services.pipeline import auto_margin_m1  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path):
    mesh = trimesh.load(str(path), force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("没有读到三角网格: %s" % path)
    return mesh


def _mesh_info(mesh) -> dict:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "extents_mm": [float(v) for v in np.asarray(mesh.extents, float)],
    }


def _frame_seed(path: Path | None) -> tuple[dict | None, np.ndarray | None]:
    if path is None:
        return None, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if all(key in value for key in ("x", "y", "z")):
        seed = {
            "mesial": value["x"],
            "buccal": value["y"],
            "insertion": value["z"],
            "revision_id": value.get("revision_id"),
        }
    elif all(key in value for key in ("mesial", "buccal", "insertion")):
        seed = {
            "mesial": value["mesial"],
            "buccal": value["buccal"],
            "insertion": value["insertion"],
            "revision_id": value.get("revision_id"),
        }
    else:
        raise ValueError("frame-json 必须包含 x/y/z 或 mesial/buccal/insertion")
    axis = np.asarray(seed["insertion"], float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("frame-json 的 insertion/z 必须是有限非零三维向量")
    axis /= np.linalg.norm(axis)
    return seed, axis


def _local(points, frame):
    return (np.asarray(points, float) - np.asarray(frame.origin, float)) @ frame.matrix()


def _private_arch_neighbor_patches(arch, frame, margin_points, support):
    """Engineering-only neighbor crops for the private two-STL fixture.

    Live CrownFlow still requires human-confirmed neighbor patches.  This helper
    exists only so the supplied FDI26 raw scan can exercise GREEN broad-contact
    geometry without uploading extra patient files.
    """
    centers = np.asarray(arch.triangles_center, float)
    local = _local(centers, frame)
    margin_local = _local(np.asarray(margin_points, float), frame)
    support_local = _local(np.asarray(support.vertices, float), frame)

    x_lo, x_hi = np.quantile(margin_local[:, 0], [0.02, 0.98])
    y_radius = float(np.quantile(np.abs(margin_local[:, 1]), 0.98))
    prep_top = float(np.quantile(support_local[:, 2], 0.995))
    z_floor = float(np.quantile(margin_local[:, 2], 0.25) + 0.20)
    z_ceil = prep_top + 6.0

    patches = {}
    report = {}
    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        if sign > 0:
            md = (local[:, 0] > x_hi + 0.15) & (local[:, 0] < x_hi + 9.0)
        else:
            md = (local[:, 0] < x_lo - 0.15) & (local[:, 0] > x_lo - 9.0)
        mask = (
            md
            & (np.abs(local[:, 1]) < y_radius + 4.0)
            & (local[:, 2] > z_floor)
            & (local[:, 2] < z_ceil)
        )
        ids = np.flatnonzero(mask)
        if len(ids) < 40:
            patches[side] = None
            report[side] = {
                "status": "NOT_ASSESSED",
                "reason": "RAW_ARCH_CROP_TOO_SMALL",
                "face_count": int(len(ids)),
            }
            continue
        patch = arch.submesh([ids], append=True, repair=False)
        patch.remove_unreferenced_vertices()
        patches[side] = patch
        report[side] = {
            "status": "PRIVATE_ENGINEERING_CROP",
            "face_count": int(len(patch.faces)),
            "source": "largest_same_jaw_component",
            "clinical_confirmation": False,
            "warning": "仅用于私有回归；live workflow 必须使用人工确认 neighbor patch",
        }
    return patches, report


def _write_report(out: Path, report: dict) -> None:
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", required=True, type=Path)
    ap.add_argument("--lower", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fdi", type=int, default=26)
    ap.add_argument("--frame-json", type=Path, default=None)
    ap.add_argument("--margin-mode", choices=("SUPRA", "SUB"), default="SUPRA")
    ap.add_argument("--margin-radius-mm", type=float, default=1.0)
    ap.add_argument("--min-thickness-mm", type=float, default=0.50)
    ap.add_argument("--proximal-target-mm", type=float, default=0.02)
    ap.add_argument("--occlusal-target-mm", type=float, default=0.05)
    ap.add_argument("--cement-gap-mm", type=float, default=0.06)
    ap.add_argument("--cement-gap-start-mm", type=float, default=0.30)
    ap.add_argument("--cement-gap-top-mm", type=float, default=0.10)
    args = ap.parse_args()

    if int(args.fdi) != 26:
        raise SystemExit("这个 runner 专门用于 FDI26 私有回归")

    args.out.mkdir(parents=True, exist_ok=True)
    frame_seed, confirmed_axis = _frame_seed(args.frame_json)
    report = {
        "runner": "run_fdi26_private_regression/3-section-red-green-blue",
        "fdi": 26,
        "privacy": "raw_stl_local_only_not_committed",
        "release_status": "NOT_PRODUCTION_APPROVED",
        "inputs": {
            "upper": {"name": args.upper.name, "sha256": _sha256(args.upper)},
            "lower": {"name": args.lower.name, "sha256": _sha256(args.lower)},
            "frame": (
                {"name": args.frame_json.name, "sha256": _sha256(args.frame_json)}
                if args.frame_json
                else None
            ),
        },
        "stages": {},
        "artifacts": {},
    }

    upper = _load(args.upper)
    lower = _load(args.lower)
    report["scan_pair"] = pair_report(upper, lower)

    support, support_report = select_separate_support(upper)
    report["stages"]["support"] = support_report
    if support is None:
        report.update(status="BLOCKED", reason="DETACHED_SUPPORT_NOT_FOUND")
        _write_report(args.out, report)
        return 2

    upper_major, _upper_all, _upper_threshold = major_components(upper)
    lower_major, _lower_all, _lower_threshold = major_components(lower)
    arch = upper_major[0]
    antagonist = lower_major[0] if lower_major else lower
    report["stages"]["same_jaw_arch"] = _mesh_info(arch)
    report["stages"]["antagonist"] = {
        **_mesh_info(antagonist),
        "registration": "RAW_SCAN_PAIR_STATIC_POSE_ASSUMED_FOR_PRIVATE_REGRESSION",
    }

    support_path = args.out / "fdi26_support_candidate.stl"
    support.export(str(support_path))
    report["artifacts"]["support_candidate"] = str(support_path)
    report["stages"]["support"]["mesh"] = _mesh_info(support)

    if confirmed_axis is not None:
        axis = confirmed_axis
        report["stages"]["axis"] = {
            "status": "CONFIRMED_INPUT",
            "source": "frame_json",
            "axis": axis.tolist(),
        }
    else:
        axis = np.asarray(suggest_insertion_axis(support, None), float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        report["stages"]["axis"] = {
            "status": "PROVISIONAL",
            "source": "sidewall_fit_without_confirmed_margin",
            "axis": axis.tolist(),
        }

    margin = auto_margin_m1(
        support_path,
        axis=axis.tolist(),
        mode=args.margin_mode,
        radius_mm=float(args.margin_radius_mm),
        apply_snap=True,
    )
    report["stages"]["margin"] = margin
    if not margin.get("points"):
        report.update(status="REVIEW_REQUIRED", reason="AUTO_MARGIN_NOT_ACCEPTED")
        _write_report(args.out, report)
        return 3
    margin_points = np.asarray(margin["points"], float)

    if frame_seed is not None:
        frame = confirmed_patient_frame(
            margin_points.mean(axis=0),
            frame_seed["mesial"],
            frame_seed["buccal"],
            frame_seed["insertion"],
            revision_id=frame_seed.get("revision_id") or "private-regression-frame",
        )
        frame_status, frame_source = "CONFIRMED_INPUT", "frame_json"
    else:
        frame = build_patient_frame(margin_points, axis)
        frame_status, frame_source = "PROVISIONAL", "margin_pca"
    report["stages"]["patient_frame"] = {
        "status": frame_status,
        "source": frame_source,
        "revision_id": getattr(frame, "revision_id", None),
        "origin": np.asarray(frame.origin, float).tolist(),
        "x": np.asarray(frame.x, float).tolist(),
        "y": np.asarray(frame.y, float).tolist(),
        "z": np.asarray(frame.z, float).tolist(),
    }

    neighbors, neighbor_report = _private_arch_neighbor_patches(
        arch, frame, margin_points, support
    )
    report["stages"]["neighbors"] = neighbor_report
    for side, mesh in neighbors.items():
        if mesh is not None:
            path = args.out / ("fdi26_private_%s_patch.stl" % side)
            mesh.export(str(path))
            report["artifacts"]["%s_patch" % side] = str(path)

    config = SectionalV1Config(
        min_wall_mm=float(args.min_thickness_mm),
        proximal_gap_mm=float(args.proximal_target_mm),
        occlusal_gap_mm=float(args.occlusal_target_mm),
    )
    try:
        outer, sectional = build_sectional_outer(
            margin_points,
            frame,
            support,
            neighbors=neighbors,
            antagonist_mesh=antagonist,
            fdi=26,
            config=config,
        )
    except SectionalOuterError as exc:
        report["stages"]["sectional_outer"] = {
            "status": "BLOCKED",
            "reason": str(exc),
        }
        report.update(status="REVIEW_REQUIRED", reason="SECTIONAL_OUTER_BLOCKED")
        _write_report(args.out, report)
        return 4

    outer_path = args.out / "fdi26_sectional_outer.stl"
    outer.export(str(outer_path))
    report["artifacts"]["sectional_outer"] = str(outer_path)
    report["stages"]["sectional_outer"] = sectional

    # Requirement 4: prep-driven inner crown/cutter.
    try:
        fitting_support, fitting_report = fitting_support_above_margin(
            support, margin_points, axis
        )
        policy = BlockoutPolicy(
            machine_axes=3,
            additional_blockout_after_gap=0.0,
            untouchable_zone_mm=0.0,
            provenance={"source": "private_regression_explicit_three_axis"},
        )
        blockout = build_blockout(fitting_support, margin_points, axis, policy)
        gap = GapFieldParams(
            cement_gap_mm=float(args.cement_gap_mm),
            cement_gap_start_mm=float(args.cement_gap_start_mm),
            cement_gap_top_mm=float(args.cement_gap_top_mm),
            additional_spacing_xyz_mm=None,
            additional_spacing_z_on_top_mm=None,
            provenance={"source": "private_regression_cli"},
        )
        cutter = apply_gap_field(blockout, fitting_support, margin_points, axis, gap)
        cutter_path = args.out / "fdi26_intaglio_cutter.stl"
        cutter.export(str(cutter_path))
        report["artifacts"]["intaglio_cutter"] = str(cutter_path)
        report["stages"]["intaglio"] = {
            "status": (
                "PASS_ENGINEERING"
                if blockout.status.value == "OK"
                else "REVIEW_REQUIRED"
            ),
            "mechanism": "prep -> directional blockout -> variable gap cutter",
            "fitting_support": fitting_report,
            "blockout": blockout.as_dict(),
            "gap": gap.as_dict(),
            "cutter": _mesh_info(cutter),
        }
    except (ValueError, TypeError) as exc:
        report["stages"]["intaglio"] = {
            "status": "REVIEW_REQUIRED",
            "reason": str(exc),
        }

    red = sectional["margin"]["hard_boundary"] and sectional["margin"]["max_error_mm"] < 0.05
    green = all(
        sectional["green_lateral"].get(side, {}).get("status") == "PASS_ENGINEERING"
        for side in ("mesial", "distal")
        if neighbors.get(side) is not None
    )
    blue = sectional["blue_occlusal"].get("status") == "PASS_ENGINEERING"
    inner = report["stages"]["intaglio"].get("status") == "PASS_ENGINEERING"
    report["acceptance"] = {
        "margin_hard_fit": bool(red),
        "broad_proximal_shape": bool(green),
        "antagonist_occlusal_cap": bool(blue),
        "prep_driven_intaglio": bool(inner),
    }
    all_ok = all(report["acceptance"].values())
    if all_ok and frame_seed is not None:
        report["status"] = "PASS_PRIVATE_REGRESSION"
    elif all_ok:
        report["status"] = "PASS_PRIVATE_GEOMETRY_REGRESSION"
        report["reason"] = "PATIENT_FRAME_NOT_CONFIRMED"
    else:
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = "ONE_OR_MORE_CROWN_V1_REQUIREMENTS_NOT_MET"

    _write_report(args.out, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "acceptance": report["acceptance"],
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"].startswith("PASS_") else 6


if __name__ == "__main__":
    raise SystemExit(main())
