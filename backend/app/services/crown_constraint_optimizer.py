"""`CrownConstraintOptimizer`：把三个 solver 组织成**分阶段迭代**编排（本批 P0 第 7 项）。

刻意**不做**单个巨大的非线性优化器。第一版就是本文件：

    placement/scale（调用方已完成）
      → repeated up to MAX_ITER:
            proximal solve
            occlusal solve（仅在咬合位姿已确认时）
            minimum thickness enforce
            re-measure proximal / occlusal / thickness
      → 最终 PASS / REVIEW_REQUIRED / NOT_ASSESSED

纪律：
- **禁止"部分失败仍标 PASS"**：只要有任何阶段不是 PASS，整体就不是 PASS（`_verdict`）。
- 没有确认咬合位姿时，咬合阶段记 `OCCLUSION_NOT_ASSESSED / BITE_POSE_NOT_CONFIRMED`，
  并且**不算失败**（`NOT_ASSESSED` 是如实状态），但会让整体退化为 `REVIEW_REQUIRED`，
  因为"未评估"不等于"通过"。
- 阶段顺序固定，且 Intaglio 不在本编排内（规格 §11：内冠必须在自动外形适配之后）。
"""

from __future__ import annotations

import numpy as np

from .crown_constraint_solvers import (
    STATUS_CONFLICT,
    STATUS_NOT_ASSESSED,
    STATUS_PASS,
    STATUS_REVIEW,
    enforce_minimum_thickness,
    solve_occlusion,
    solve_proximal,
)

#: 规格 §10 建议的迭代上限
MAX_ITER = 3
OPTIMIZER_VERSION = "crown_constraint_optimizer/staged/1"

#: 阶段名（顺序即执行顺序，前端进度条按它渲染）
STAGES = ("PROXIMAL", "OCCLUSAL", "THICKNESS", "RECHECK")


def _verdict(stages: list) -> tuple:
    """把各阶段状态汇总成整体判定。PASS 必须**所有**执行过的阶段都是 PASS。"""
    executed = [s for s in stages if s.get("status") != "SKIPPED"]
    if not executed:
        return STATUS_NOT_ASSESSED, ["NO_STAGE_EXECUTED"]
    if any(s["status"] == STATUS_CONFLICT for s in executed):
        return STATUS_CONFLICT, ["CONSTRAINT_CONFLICT"]
    if all(s["status"] == STATUS_PASS for s in executed):
        return STATUS_PASS, []
    if all(s["status"] == STATUS_NOT_ASSESSED for s in executed):
        return STATUS_NOT_ASSESSED, sorted({s.get("reason", "NOT_ASSESSED") for s in executed})
    return STATUS_REVIEW, sorted({s.get("reason") or s["status"] for s in executed if s["status"] != STATUS_PASS})


def _skip(stage: str, reason: str) -> dict:
    """跳过的阶段也必须是**有名有姓**的一条记录（否则汇总/进度条无法对齐）。"""
    return {"solver": stage, "stage": stage, "status": "SKIPPED", "reason": reason}


def optimize_crown(crown, frame, margin, profile, *,
                   proximal_patches=None,
                   antagonist=None,
                   bite_pose=None,
                   baseline=None,
                   inner=None,
                   protected_surfaces=(),
                   max_iter: int = MAX_ITER) -> dict:
    """分阶段约束优化。返回 ``(mesh, report)``。

    ``crown``：自动放置/缩放后的外冠（placement 由调用方完成，本编排不碰）。
    ``inner``：内冠/腔体（仅用于壁厚测量；不给则壁厚阶段记 NOT_ASSESSED）。
    """
    current = crown
    history = []
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        stages = []
        # 1) 近远中
        if proximal_patches:
            proximal = solve_proximal(current, frame, margin, proximal_patches, profile,
                                      baseline=baseline)
            current = _apply(current, proximal)
            stages.append({**proximal, "stage": "PROXIMAL"})
        else:
            stages.append(_skip("PROXIMAL", "NO_CONFIRMED_NEIGHBOR"))
        # 2) 静态咬合（没有确认位姿就是 NOT_ASSESSED，不是失败）
        if antagonist is not None or bite_pose is not None:
            occlusal = solve_occlusion(current, frame, margin, antagonist, profile,
                                       bite_pose=bite_pose, patches=proximal_patches,
                                       baseline=baseline)
            current = _apply(current, occlusal)
            stages.append({**occlusal, "stage": "OCCLUSAL"})
        else:
            stages.append(_skip("OCCLUSAL", "NO_ANTAGONIST_INPUT"))
        # 3) 最小厚度
        if inner is not None:
            thickness = enforce_minimum_thickness(current, inner, profile,
                                                  protected_surfaces=protected_surfaces)
            current = thickness.pop("result", current) if thickness.get("geometry_changed") else current
            stages.append({**thickness, "stage": "THICKNESS"})
        else:
            stages.append(_skip("THICKNESS", "NO_INTAGLIO_GEOMETRY"))
        # 4) 复测（每一轮都要重新量，不看 solver 自称）
        recheck = _recheck(current, frame, margin, profile, proximal_patches, antagonist, bite_pose)
        stages.append({**recheck, "stage": "RECHECK"})

        verdict, reasons = _verdict(stages)
        history.append({"iteration": iteration, "verdict": verdict, "reasons": reasons,
                        "stage_status": {s["stage"]: s["status"] for s in stages}})
        if verdict == STATUS_PASS:
            break
        if any(s["status"] in (STATUS_CONFLICT, "BLOCKED") for s in stages):
            break                       # 冲突就别再迭代（避免用迭代掩盖冲突）
        if all(s["status"] in (STATUS_NOT_ASSESSED, "SKIPPED") for s in stages):
            break

    final_stages = stages
    verdict, reasons = _verdict(final_stages)
    return current, {
        "optimizer": "CrownConstraintOptimizer", "algorithm_version": OPTIMIZER_VERSION,
        "strategy": "staged_iterative_solver", "max_iter": int(max_iter), "iterations": iterations,
        "status": verdict, "reasons": reasons,
        "stage_order": list(STAGES),
        "stages": final_stages,
        "history": history,
        "partial_failure_never_passes": True,
        "intaglio_stage_in_optimizer": False,
        "note": ("内冠（Intaglio/Blockout/Gap）不属本编排：必须在自动外形适配之后按固定顺序构建"
                 "（规格 §11），本编排只产出外冠与约束报告。"),
    }


def _apply(mesh, report: dict):
    """solver 报告里带出的几何只有在它**自称改过**时才采用（否则保持原样）。"""
    candidate = report.get("geometry")
    if candidate is None:
        return mesh
    if not report.get("geometry_changed", True):
        return mesh
    return candidate


def _recheck(mesh, frame, margin, profile, patches, antagonist, bite_pose) -> dict:
    """独立复测：不采信任何 solver 的自我描述，重新量一遍。"""
    from .contact_measurement import KIND_OCCLUSAL, KIND_PROXIMAL, measure_surface
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains

    out = {"solver": "Recheck", "status": STATUS_PASS, "reason": None, "measurements": {}}
    target, source = profile.value("proximal.target_distance_mm")
    if patches and source != "MISSING" and target is not None:
        masks, _ = fit_domains(mesh, frame, margin, patches, "PROXIMAL")
        out["measurements"]["proximal"] = {}
        for side, surface in patches.items():
            report, _ = measure_surface(surface, np.asarray(mesh.vertices), masks[side], float(target),
                                        kind=KIND_PROXIMAL, side=side)
            out["measurements"]["proximal"][side] = {
                "status": report.get("status"), "target_error": report.get("target_error"),
                "signed_min_mm": report.get("signed_min_mm"),
                "penetration_max_mm": report.get("penetration_max_mm")}
    occl_target, occl_source = profile.value("occlusion.target_clearance_mm")
    if antagonist is not None and occl_source != "MISSING" and occl_target is not None:
        pose = dict(bite_pose or {})
        if pose.get("pose_confirmed") and pose.get("transform"):
            masks, _ = fit_domains(mesh, frame, margin, {"occlusal": antagonist}, "OCCLUSAL")
            report, _ = measure_surface(antagonist, np.asarray(mesh.vertices), masks["occlusal"],
                                        float(occl_target), kind=KIND_OCCLUSAL, side="occlusal")
            out["measurements"]["occlusal"] = {"status": report.get("status"),
                                               "deficit": report.get("deficit"),
                                               "signed_min_mm": report.get("signed_min_mm")}
        else:
            out["measurements"]["occlusal"] = {"status": "NOT_ASSESSED", "reason": "BITE_POSE_NOT_CONFIRMED"}
    if not out["measurements"]:
        out.update(status=STATUS_NOT_ASSESSED, reason="NO_CONFIRMED_REFERENCE")
    return out
