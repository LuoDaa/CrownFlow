r"""Private FDI26 Crown 1.0 regression runner.

The scanner STLs stay on the local machine.  This script never uploads them and
never copies them into the repository.  It is intentionally fail-loud: every
stage writes a machine-readable status into ``report.json``.

Typical use::

    python backend/tools/run_fdi26_private_regression.py \
        --upper "/path/to/upper.stl" \
        --lower "/path/to/lower.stl" \
        --out "/tmp/crown-v1-fdi26"

Stages
------
1. split raw scans and identify a detached prep/die candidate;
2. propose a provisional insertion axis and M1 margin on that candidate;
3. if a compatible FDI26 tooth library exists, generate a margin-constrained
   CrownOuter draft;
4. use the already aligned antagonist scan as a private-regression occlusal
   surface, evaluate the axial envelope and apply the Crown 1.0 axial plan;
5. write artifacts and a JSON report.  Proximal contacts remain NOT_ASSESSED
   unless confirmed neighbor patches exist in the normal application workflow.

This is regression tooling, not a clinical automatic-design endpoint.
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
from app.geometry.crown_outer import (  # noqa: E402
    CrownOuterConfig,
    generate_crown_outer_v0,
    load_crown_template,
)
from app.geometry.crown_outer.auto_fit_adapter import FitSurface  # noqa: E402
from app.geometry.scan_pair_triage import (  # noqa: E402
    major_components,
    pair_report,
    select_separate_support,
)
from app.importers.tooth_library import discover_libraries  # noqa: E402
from app.services.axial_placement import (  # noqa: E402
    apply_axial_placement,
    occlusal_envelope,
    plan_axial_placement,
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


def _choose_library(name: str | None, fdi: int):
    libraries = {lib.name: lib for lib in discover_libraries()}
    if not libraries:
        return None, {"status": "NOT_ASSESSED", "reason": "NO_TOOTH_LIBRARY_DISCOVERED"}
    ordered = ([name] if name else []) + [key for key in sorted(libraries) if key != name]
    errors = []
    for key in ordered:
        if key not in libraries:
            continue
        try:
            template = load_crown_template(libraries[key], int(fdi))
            return template, {"status": "READY", "library": key, "template": template.summary()}
        except Exception as exc:  # regression runner: keep searching and record why
            errors.append({"library": key, "error": str(exc)[:240]})
    return None, {"status": "NOT_ASSESSED", "reason": "NO_COMPATIBLE_FDI_TEMPLATE",
                  "attempts": errors}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upper", required=True, type=Path)
    ap.add_argument("--lower", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fdi", type=int, default=26)
    ap.add_argument("--library", default=None)
    ap.add_argument("--margin-mode", choices=("SUPRA", "SUB"), default="SUPRA")
    ap.add_argument("--margin-radius-mm", type=float, default=1.0)
    ap.add_argument("--min-thickness-mm", type=float, default=0.5)
    ap.add_argument("--occlusal-target-mm", type=float, default=0.05)
    args = ap.parse_args()

    if int(args.fdi) != 26:
        raise SystemExit("这个 runner 专门用于 FDI26 私有回归；其他牙位请复制病例包走通用 runner")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "runner": "run_fdi26_private_regression/1",
        "fdi": int(args.fdi),
        "privacy": "raw_stl_local_only_not_committed",
        "inputs": {
            "upper": {"name": args.upper.name, "sha256": _sha256(args.upper)},
            "lower": {"name": args.lower.name, "sha256": _sha256(args.lower)},
        },
        "stages": {},
    }

    upper = _load(args.upper)
    lower = _load(args.lower)
    report["scan_pair"] = pair_report(upper, lower)

    support, support_report = select_separate_support(upper)
    report["stages"]["support"] = support_report
    if support is None:
        report["status"] = "BLOCKED"
        report["reason"] = "DETACHED_SUPPORT_NOT_FOUND"
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        return 2

    support_path = args.out / "fdi26_support_candidate.stl"
    support.export(str(support_path))
    report["artifacts"] = {"support_candidate": str(support_path)}
    report["stages"]["support"]["mesh"] = _mesh_info(support)

    axis = np.asarray(suggest_insertion_axis(support, None), float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    report["stages"]["axis"] = {
        "status": "PROVISIONAL",
        "source": "sidewall_fit_without_confirmed_margin",
        "axis": [float(v) for v in axis],
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
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = "AUTO_MARGIN_NOT_ACCEPTED"
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        return 3
    margin_points = np.asarray(margin["points"], float)

    template, template_report = _choose_library(args.library, args.fdi)
    report["stages"]["template"] = template_report
    if template is None:
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = "FDI26_TEMPLATE_UNAVAILABLE"
        (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
        return 4

    config = CrownOuterConfig(body_scale_decoupling=True)
    crown = generate_crown_outer_v0(template, margin_points, axis, config=config)
    draft_path = args.out / "fdi26_crown_margin_fitted.stl"
    crown.mesh.export(str(draft_path))
    report["artifacts"]["margin_fitted_crown"] = str(draft_path)
    report["stages"]["outer"] = crown.summary()

    lower_major, _lower_all, _threshold = major_components(lower)
    antagonist = lower_major[0] if lower_major else lower
    # FDI26 is maxillary; the antagonist outward direction expected by the live
    # path is opposite the target insertion axis.
    surface = FitSurface(antagonist, (-axis).tolist())
    dummy_design = {"target_fdi": 26, "support_type": "natural"}
    envelope = occlusal_envelope(
        dummy_design,
        margin_points,
        axis,
        {},
        surface=surface,
        prep_path=support_path,
        crown_mesh=crown.mesh,
        min_thickness_mm=float(args.min_thickness_mm),
        target_clearance_mm=float(args.occlusal_target_mm),
    )
    plan = (plan_axial_placement(envelope, crown_top_mm=envelope.get("crown_top_mm") or 0.0)
            if envelope.get("status") != "NOT_ASSESSED" else None)
    report["stages"]["axial_placement"] = {"envelope": envelope, "plan": plan}

    final_mesh = crown.mesh
    if plan and plan.get("feasible") and plan.get("mechanism") == "AXIAL_SCALE":
        placed, applied = apply_axial_placement(final_mesh, margin_points, axis, plan)
        report["stages"]["axial_placement"]["applied"] = applied
        if placed is None:
            report["status"] = "REVIEW_REQUIRED"
            report["reason"] = "AXIAL_DEFORMATION_GATE_REJECTED"
            (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
            return 5
        final_mesh = placed
    elif plan and not plan.get("feasible"):
        report["status"] = "REVIEW_REQUIRED"
        report["reason"] = plan.get("mechanism") or "AXIAL_PLAN_NOT_FEASIBLE"

    axial_path = args.out / "fdi26_crown_axially_placed.stl"
    final_mesh.export(str(axial_path))
    report["artifacts"]["axially_placed_crown"] = str(axial_path)

    report["stages"]["proximal"] = {
        "status": "NOT_ASSESSED",
        "reason": "RAW_ARCH_HAS_NO_CONFIRMED_MESIAL_DISTAL_NEIGHBOR_PATCHES",
        "note": "近远中微调仍应在正常工作流中消费已确认邻牙 patch；本 runner 不从原始 arch 猜牙位。",
    }
    if "status" not in report:
        report["status"] = "PASS_PRIVATE_REGRESSION"
    report["release_status"] = "NOT_PRODUCTION_APPROVED"
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS_PRIVATE_REGRESSION" else 6


if __name__ == "__main__":
    raise SystemExit(main())
