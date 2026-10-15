"""B1 BASELINE（只报告，不断言期望）：
Canonical fixture + **当前未改动**的 solver + **当前未改动**的 QC。

目的：在动 Pair Quality / QC / solver 之前，先把"当前实现到底给出什么"钉下来。
明确区分两项：
    POSE_CORRECT        —— 位姿恢复是否对（与 Canonical World 的独立帧 GT 比）
    QC_VERDICT_CORRECT  —— QC 的档位判定是否符合该 case 的期望

⚠️ 本文件**故意不做期望断言**（B1 是基线，不是 gate）。B2 才是"修正 QC 后重跑"。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_bite_canonical_world import (  # noqa: E402
    band_of, canonical_world, jaw_solid, scanner_inputs, _rigid,
)
from app.geometry.bite_registration import register  # noqa: E402

#: 位姿判定容差（只看"是否恢复"这一件事，与 QC 阈值无关）
POSE_TOL_TRANSLATION_MM = 0.50
POSE_TOL_ROTATION_DEG = 1.00


def _errors(estimate, ground_truth):
    delta = np.asarray(estimate, float) @ np.linalg.inv(np.asarray(ground_truth, float))
    return (float(np.linalg.norm(delta[:3, 3])),
            float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))))


def _jaw_pose_point_rmse(estimate, ground_truth, mesh, samples=2000):
    """GT 度量（只存在 benchmark/test）：同一点用两个位姿的位移 RMSE。"""
    pts = mesh.sample(samples)
    a = trimesh.transform_points(pts, np.asarray(estimate, float))
    b = trimesh.transform_points(pts, np.asarray(ground_truth, float))
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _jitter(mesh, sigma, seed):
    rng = np.random.default_rng(seed)
    out = mesh.copy()
    out.vertices = np.asarray(out.vertices, float) + rng.normal(0, sigma, np.asarray(out.vertices).shape)
    return out


# ------------------------------------------------------------------ 13 类 case
def build_cases():
    """全部从**同一个 World truth**派生（不改 World，只改帧/扰动/输入内容）。"""
    cases = {}

    def base(A_upper=np.eye(4), A_lower=np.eye(4), A_bite=np.eye(4), **world_kw):
        world = canonical_world(**world_kw)
        inputs, gts = scanner_inputs(world, A_upper=A_upper, A_lower=A_lower, A_bite=A_bite)
        return inputs, gts, world

    cases["identity"] = (*base(), "PASS")
    cases["translation"] = (*base(A_lower=_rigid([0, 0, 1], 0, [0, 0, 2.5])), "PASS")
    cases["rotation"] = (*base(A_lower=_rigid([1, 0, 0], 6, [0, 0, 0])), "PASS")
    cases["rigid"] = (*base(A_lower=_rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0])), "PASS")
    cases["large_initial_misalignment"] = (*base(A_bite=_rigid([0.3, 0.6, 0.7], 40.0, [15.0, -8.0, 6.0])), "PASS")

    cases["partial"] = (*base(band=(-1.2, 1.2)), "PASS/REVIEW")
    inputs, gts, world = base()
    inputs["bite"] = _jitter(inputs["bite"], 0.04, 7)
    cases["noise"] = (inputs, gts, world, "PASS/REVIEW")
    inputs, gts, world = base()
    spike = trimesh.creation.icosphere(subdivisions=2, radius=1.2)
    spike.apply_translation([0, 0, 26])
    inputs["bite"] = trimesh.util.concatenate([inputs["bite"], spike])
    cases["outlier"] = (inputs, gts, world, "PASS/REVIEW")

    inputs, gts, world = base()
    here = band_of(inputs["upper"], -2.5, 2.5)
    inputs["bite"] = trimesh.util.concatenate([here.copy(), here.copy()])
    cases["ambiguous_side"] = (inputs, gts, world, "REVIEW_REQUIRED")

    # wrong bite：内容来自**另一个 World**（跨病例）
    inputs, gts, world = base()
    other = canonical_world(upper_seed=99, lower_seed=98, band=(-2.5, 2.5))
    inputs["bite"] = trimesh.util.concatenate([
        trimesh.Trimesh(np.asarray(other["upper_patch_world"].vertices, float) + np.array([40.0, 0, 0]),
                        np.asarray(other["upper_patch_world"].faces, int), process=True),
        trimesh.Trimesh(np.asarray(other["lower_patch_world"].vertices, float) + np.array([40.0, 0, 0]),
                        np.asarray(other["lower_patch_world"].faces, int), process=True)])
    cases["wrong_bite_cross_case"] = (inputs, gts, world, "FAILED")

    # wrong bite：镜像（非刚体，含反射）
    inputs, gts, world = base()
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    mirrored = inputs["bite"].copy()
    mirrored.apply_transform(mirror)
    inputs["bite"] = mirrored
    cases["wrong_bite_mirrored_or_unrelated"] = (inputs, gts, world, "FAILED")

    cases["gross_penetration"] = (*base(lower_overlap_mm=6.0), "FAILED")
    cases["missing_bite"] = ({"bite": None, "upper": None, "lower": None}, {"T_lower_to_upper": np.eye(4)}, None, "FAILED")
    inputs, _gts, world = base(band=(-0.25, 0.25))
    cases["low_information_patch"] = (inputs, _gts, world, "REVIEW/FAILED")
    return cases


def _run(name, payload):
    inputs, gts, world, expected = payload
    if name == "missing_bite":
        world_ = canonical_world()
        inputs, gts = scanner_inputs(world_, with_bite=False)
        result = register(None, inputs["upper"], inputs["lower"])
    else:
        result = register(inputs["bite"], inputs["upper"], inputs["lower"])
    est = result.get("T_lower_to_upper")
    if est is None:
        pose_line = "POSE=n/a"
        pose_ok, jaw_rmse = False, float("nan")
    else:
        te, re = _errors(est, gts["T_lower_to_upper"])
        jaw_rmse = _jaw_pose_point_rmse(est, gts["T_lower_to_upper"], inputs["lower"])
        pose_ok = (te <= POSE_TOL_TRANSLATION_MM and re <= POSE_TOL_ROTATION_DEG)
        pose_line = ("ROT_ERR=%.4f deg TRANS_ERR=%.4f mm JAW_POINT_RMSE=%.4f mm"
                     % (re, te, jaw_rmse))
    sides = result.get("per_side") or {}
    up, lo = sides.get("upper") or {}, sides.get("lower") or {}

    def fmt(block):
        if not block or block.get("rmse_mm") is None:
            return "rmse=n/a src_cov=n/a whole_tgt_cov=n/a"
        return ("rmse=%.4f src_cov=%.3f whole_tgt_cov=%.3f"
                % (block["rmse_mm"], block.get("source_coverage_ratio") or -1,
                   block.get("target_coverage_ratio") or -1))
    qc_ok = (result["status"] == expected) or (expected == "PASS/REVIEW" and result["status"] in ("PASS", "REVIEW_REQUIRED"))
    print("[B1] %-34s pose_correct=%-5s qc_correct=%-5s | %s | upper[%s] lower[%s] | %s / %s"
          % (name, pose_ok, qc_ok, pose_line, fmt(up), fmt(lo), result["status"], result["reason"]))
    return {"name": name, "pose_correct": pose_ok, "qc_correct": qc_ok,
            "status": result["status"], "reason": result["reason"],
            "jaw_pose_point_rmse_mm": jaw_rmse}


def test_b1_baseline_report_only():
    """B1：只报告，不断言。用于在改动 QC/solver 前钉住当前行为。"""
    rows = [_run(name, payload) for name, payload in build_cases().items()]
    print("[B1 SUMMARY] pose_correct=%d/%d qc_correct=%d/%d"
          % (sum(r["pose_correct"] for r in rows), len(rows),
             sum(r["qc_correct"] for r in rows), len(rows)))
    assert len(rows) == 13, "必须是 13 类 case"


if __name__ == "__main__":
    test_b1_baseline_report_only()
