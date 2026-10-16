"""Private FDI26 Crown 1.0 regression runner.

Raw scanner STLs stay local. Margin source order is explicit:
1) --margin-json (confirmed live Margin),
2) the 128-ray prep/abutment shoulder detector used by the live abutment flow,
3) legacy M1 as an engineering fallback.

Every automatic candidate must pass the prep-relative Margin contract before
RED/GREEN/BLUE geometry is built. Automatic candidates remain engineering
evidence only; they never become production-confirmed Margin truth.
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

from app.geometry.abutment_margin import extract_abutment_margin  # noqa: E402
from app.geometry.axis import suggest_insertion_axis  # noqa: E402
from app.geometry.crown_outer.margin_contract import (  # noqa: E402
    MarginContractError,
    assess_margin_against_support,
    sanitize_margin_loop,
)
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


def _frame_seed(path: Path | None):
    if path is None:
        return None, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if all(key in value for key in ("x", "y", "z")):
        seed = {"mesial": value["x"], "buccal": value["y"],
                "insertion": value["z"], "revision_id": value.get("revision_id")}
    elif all(key in value for key in ("mesial", "buccal", "insertion")):
        seed = {"mesial": value["mesial"], "buccal": value["buccal"],
                "insertion": value["insertion"], "revision_id": value.get("revision_id")}
    else:
        raise ValueError("frame-json 必须包含 x/y/z 或 mesial/buccal/insertion")
    axis = np.asarray(seed["insertion"], float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("frame-json 的 insertion/z 必须是有限非零三维向量")
    axis /= np.linalg.norm(axis)
    return seed, axis


def _margin_json_points(path: Path) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    points = None
    if isinstance(value, list):
        points = value
    elif isinstance(value, dict):
        points = value.get("points") or value.get("margin_points")
        if points is None and isinstance(value.get("margin"), dict):
            points = value["margin"].get("points")
        if points is None and isinstance(value.get("payload"), dict):
            points = value["payload"].get("points")
    if points is None:
        raise ValueError("margin-json 必须包含 Nx3 points")
    result = np.asarray(points, float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("margin-json points 必须是 Nx3")
    return result


def _contract_candidate(points, support, axis, *, source: str, diagnostics=None):
    stage = {"source": source, "status": "CANDIDATE",
             "confirmed": source == "margin_json", "diagnostics": diagnostics or {}}
    try:
        clean, cleanup = sanitize_margin_loop(points)
        plausibility = assess_margin_against_support(
            clean, np.asarray(support.vertices, float), axis)
    except MarginContractError as exc:
        stage.update(status="REJECTED", reason=str(exc),
                     contract={"status": "REVIEW_REQUIRED", "reason": str(exc)})
        return None, stage
    stage["contract"] = {"cleanup": cleanup, "plausibility": plausibility}
    stage["point_count"] = int(len(clean))
    stage["perimeter_3d_mm"] = float(
        np.linalg.norm(np.roll(clean, -1, axis=0) - clean, axis=1).sum())
    if not plausibility["accepted"]:
        stage.update(status="REJECTED", reason=plausibility["reason"])
        return None, stage
    stage["status"] = "CONFIRMED_INPUT" if stage["confirmed"] else "PASS_ENGINEERING"
    stage["points"] = clean.tolist()
    return clean, stage


def _select_margin(args, support, support_path, axis):
    attempts = []
    if args.margin_json is not None:
        clean, stage = _contract_candidate(
            _margin_json_points(args.margin_json), support, axis, source="margin_json",
            diagnostics={"source_sha256": _sha256(args.margin_json)})
        attempts.append(stage)
        return clean, {**stage, "attempts": attempts}

    # The live CROWN_ON_EXISTING_ABUTMENT margin session already uses this detector.
    # Prefer it for the detached FDI26 engineering fixture so M1 cannot promote a
    # tiny local ring into the Crown 1.0 boundary.
    try:
        points, quality = extract_abutment_margin(support, axis)
        clean, stage = _contract_candidate(
            points, support, axis, source="abutment_shoulder_candidate",
            diagnostics=quality)
        attempts.append(stage)
        if clean is not None:
            return clean, {**stage, "attempts": attempts}
    except (ValueError, RuntimeError) as exc:
        attempts.append({"source": "abutment_shoulder_candidate", "status": "REJECTED",
                         "confirmed": False, "reason": str(exc)})

    m1 = auto_margin_m1(support_path, axis=axis.tolist(), mode=args.margin_mode,
                        radius_mm=float(args.margin_radius_mm), apply_snap=True)
    if m1.get("points"):
        clean, stage = _contract_candidate(
            np.asarray(m1["points"], float), support, axis, source="auto_margin_m1",
            diagnostics={k: v for k, v in m1.items() if k != "points"})
        attempts.append(stage)
        if clean is not None:
            return clean, {**stage, "attempts": attempts}
    else:
        attempts.append({"source": "auto_margin_m1", "status": "REJECTED",
                         "confirmed": False,
                         "reason": m1.get("reason") or m1.get("status") or "AUTO_MARGIN_NOT_ACCEPTED",
                         "diagnostics": m1})
    return None, {"status": "REVIEW_REQUIRED", "reason": "NO_PLAUSIBLE_MARGIN_CANDIDATE",
                  "attempts": attempts}


def _local(points, frame):
    return (np.asarray(points, float) - np.asarray(frame.origin, float)) @ frame.matrix()


def _private_arch_neighbor_patches(arch, frame, margin_points, support):
    centers = np.asarray(arch.triangles_center, float)
    local = _local(centers, frame)
    margin_local = _local(np.asarray(margin_points, float), frame)
    support_local = _local(np.asarray(support.vertices, float), frame)
    x_lo, x_hi = np.quantile(margin_local[:, 0], [0.02, 0.98])
    y_radius = float(np.quantile(np.abs(margin_local[:, 1]), 0.98))
    prep_top = float(np.quantile(support_local[:, 2], 0.995))
    z_floor = float(np.quantile(margin_local[:, 2], 0.25) + 0.20)
    z_ceil = prep_top + 6.0
    patches, report = {}, {}
    for side, sign in (("mesial", 1.0), ("distal", -1.0)):
        if sign > 0:
            md = (local[:, 0] > x_hi + 0.15) & (local[:, 0] < x_hi + 9.0)
        else:
            md = (local[:, 0] < x_lo - 0.15) & (local[:, 0] > x_lo - 9.0)
        mask = md & (np.abs(local[:, 1]) < y_radius + 4.0) \
            & (local[:, 2] > z_floor) & (local[:, 2] < z_ceil)
        ids = np.flatnonzero(mask)
        if len(ids) < 40:
            patches[side] = None
            report[side] = {"status": "NOT_ASSESSED", "reason": "RAW_ARCH_CROP_TOO_SMALL",
                            "face_count": int(len(ids))}
            continue
        patch = arch.submesh([ids], append=True, repair=False)
        patch.remove_unreferenced_vertices()
        patches[side] = patch
        report[side] = {"status": "PRIVATE_ENGINEERING_CROP",
                        "face_count": int(len(patch.faces)),
                        "source": "largest_same_jaw_component",
                        "clinical_confirmation": False}
    return patches, report


def _write_report(out: Path, report: dict):
    (out / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", required=True, type=Path)
    ap.add_argument("--lower", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fdi", type=int, default=26)
    ap.add_argument("--frame-json", type=Path, default=None)
    ap.add_argument("--margin-json", type=Path, default=None,
                    help="优先使用 live 已确认 Margin；否则先跑 128-ray shoulder，再回退 M1")
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
    report = {"runner": "run_fdi26_private_regression/5-shoulder-first", "fdi": 26,
              "privacy": "raw_stl_local_only_not_committed",
              "release_status": "NOT_PRODUCTION_APPROVED",
              "inputs": {"upper": {"name": args.upper.name, "sha256": _sha256(args.upper)},
                         "lower": {"name": args.lower.name, "sha256": _sha256(args.lower)},
                         "frame": ({"name": args.frame_json.name, "sha256": _sha256(args.frame_json)}
                                   if args.frame_json else None),
                         "margin": ({"name": args.margin_json.name, "sha256": _sha256(args.margin_json)}
                                    if args.margin_json else None)},
              "stages": {}, "artifacts": {}}

    upper, lower = _load(args.upper), _load(args.lower)
    report["scan_pair"] = pair_report(upper, lower)
    support, support_report = select_separate_support(upper)
    report["stages"]["support"] = support_report
    if support is None:
        report.update(status="BLOCKED", reason="DETACHED_SUPPORT_NOT_FOUND")
        _write_report(args.out, report)
        return 2

    upper_major, *_ = major_components(upper)
    lower_major, *_ = major_components(lower)
    arch = upper_major[0]
    antagonist = lower_major[0] if lower_major else lower
    report["stages"]["same_jaw_arch"] = _mesh_info(arch)
    report["stages"]["antagonist"] = {**_mesh_info(antagonist),
        "registration": "RAW_SCAN_PAIR_STATIC_POSE_ASSUMED_FOR_PRIVATE_REGRESSION"}
    support_path = args.out / "fdi26_support_candidate.stl"
    support.export(str(support_path))
    report["artifacts"]["support_candidate"] = str(support_path)
    report["stages"]["support"]["mesh"] = _mesh_info(support)

    if confirmed_axis is not None:
        axis = confirmed_axis
        axis_status, axis_source = "CONFIRMED_INPUT", "frame_json"
    else:
        axis = np.asarray(suggest_insertion_axis(support, None), float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        axis_status, axis_source = "PROVISIONAL", "sidewall_fit_without_confirmed_margin"
    report["stages"]["axis"] = {"status": axis_status, "source": axis_source,
                                   "axis": axis.tolist()}

    margin_points, margin_stage = _select_margin(args, support, support_path, axis)
    report["stages"]["margin"] = margin_stage
    if margin_points is None:
        report.update(status="REVIEW_REQUIRED", reason="NO_PLAUSIBLE_MARGIN_CANDIDATE")
        _write_report(args.out, report)
        return 3

    if frame_seed is not None:
        frame = confirmed_patient_frame(
            margin_points.mean(axis=0), frame_seed["mesial"], frame_seed["buccal"],
            frame_seed["insertion"],
            revision_id=frame_seed.get("revision_id") or "private-regression-frame")
        frame_status, frame_source = "CONFIRMED_INPUT", "frame_json"
    else:
        frame = build_patient_frame(margin_points, axis)
        frame_status, frame_source = "PROVISIONAL", "margin_pca"
    report["stages"]["patient_frame"] = {
        "status": frame_status, "source": frame_source,
        "revision_id": getattr(frame, "revision_id", None),
        "origin": np.asarray(frame.origin, float).tolist(),
        "x": np.asarray(frame.x, float).tolist(),
        "y": np.asarray(frame.y, float).tolist(),
        "z": np.asarray(frame.z, float).tolist()}

    neighbors, neighbor_report = _private_arch_neighbor_patches(
        arch, frame, margin_points, support)
    report["stages"]["neighbors"] = neighbor_report
    for side, mesh in neighbors.items():
        if mesh is not None:
            path = args.out / ("fdi26_private_%s_patch.stl" % side)
            mesh.export(str(path))
            report["artifacts"]["%s_patch" % side] = str(path)

    config = SectionalV1Config(min_wall_mm=float(args.min_thickness_mm),
        proximal_gap_mm=float(args.proximal_target_mm),
        occlusal_gap_mm=float(args.occlusal_target_mm))
    try:
        outer, sectional = build_sectional_outer(
            margin_points, frame, support, neighbors=neighbors,
            antagonist_mesh=antagonist, fdi=26, config=config)
    except SectionalOuterError as exc:
        report["stages"]["sectional_outer"] = {"status": "BLOCKED", "reason": str(exc)}
        report.update(status="REVIEW_REQUIRED", reason="SECTIONAL_OUTER_BLOCKED")
        _write_report(args.out, report)
        return 4
    outer_path = args.out / "fdi26_sectional_outer.stl"
    outer.export(str(outer_path))
    report["artifacts"]["sectional_outer"] = str(outer_path)
    report["stages"]["sectional_outer"] = sectional

    try:
        fitting_support, fitting_report = fitting_support_above_margin(
            support, margin_points, axis)
        policy = BlockoutPolicy(machine_axes=3, additional_blockout_after_gap=0.0,
            untouchable_zone_mm=0.0,
            provenance={"source": "private_regression_explicit_three_axis"})
        blockout = build_blockout(fitting_support, margin_points, axis, policy)
        gap = GapFieldParams(cement_gap_mm=float(args.cement_gap_mm),
            cement_gap_start_mm=float(args.cement_gap_start_mm),
            cement_gap_top_mm=float(args.cement_gap_top_mm),
            additional_spacing_xyz_mm=None, additional_spacing_z_on_top_mm=None,
            provenance={"source": "private_regression_cli"})
        cutter = apply_gap_field(blockout, fitting_support, margin_points, axis, gap)
        cutter_path = args.out / "fdi26_intaglio_cutter.stl"
        cutter.export(str(cutter_path))
        report["artifacts"]["intaglio_cutter"] = str(cutter_path)
        report["stages"]["intaglio"] = {
            "status": "PASS_ENGINEERING" if blockout.status.value == "OK" else "REVIEW_REQUIRED",
            "mechanism": "prep -> directional blockout -> variable gap cutter",
            "fitting_support": fitting_report, "blockout": blockout.as_dict(),
            "gap": gap.as_dict(), "cutter": _mesh_info(cutter)}
    except (ValueError, TypeError) as exc:
        report["stages"]["intaglio"] = {"status": "REVIEW_REQUIRED", "reason": str(exc)}

    red = sectional["margin"]["hard_boundary"] and sectional["margin"]["max_error_mm"] < 0.05
    green = all(sectional["green_lateral"].get(side, {}).get("status") == "PASS_ENGINEERING"
                for side in ("mesial", "distal") if neighbors.get(side) is not None)
    blue = sectional["blue_occlusal"].get("status") == "PASS_ENGINEERING"
    inner = report["stages"]["intaglio"].get("status") == "PASS_ENGINEERING"
    margin_confirmed = bool(args.margin_json is not None)
    frame_confirmed = bool(frame_seed is not None)
    report["acceptance"] = {"margin_hard_fit": bool(red),
        "broad_proximal_shape": bool(green), "antagonist_occlusal_cap": bool(blue),
        "prep_driven_intaglio": bool(inner), "margin_confirmed": margin_confirmed,
        "patient_frame_confirmed": frame_confirmed}
    geometry_ok = all((red, green, blue, inner))
    unresolved = ([] if margin_confirmed else ["MARGIN_NOT_CONFIRMED"]) + \
        ([] if frame_confirmed else ["PATIENT_FRAME_NOT_CONFIRMED"])
    if geometry_ok and not unresolved:
        report["status"] = "PASS_PRIVATE_REGRESSION"
    elif geometry_ok:
        report["status"] = "PASS_PRIVATE_GEOMETRY_REGRESSION"
        report["reason"] = "+".join(unresolved)
    else:
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = "ONE_OR_MORE_CROWN_V1_REQUIREMENTS_NOT_MET"
    _write_report(args.out, report)
    print(json.dumps({"status": report["status"], "acceptance": report["acceptance"],
                      "margin_source": report["stages"]["margin"].get("source"),
                      "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"].startswith("PASS_") else 6


if __name__ == "__main__":
    raise SystemExit(main())
