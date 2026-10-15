"""冠约束求解器：把**已有**几何能力组织成三个具名 solver（本批 P0）。

本模块是**编排层**，不是新的几何实现。每个 solver 明确声明它复用了哪些既有能力：

- 近远中：`geometry.crown_outer.auto_fit_adapter.fit_stage`（局部带 Laplacian + α 回溯 +
  `deform_gate`）与 `services.contact_measurement`（唯一有符号口径）；
- 咬合：同一个 `fit_stage`（OCCLUSAL 阶段）+ `occlusion.solve_occlusion_collision` 同源的
  符号来源（`measurement_core.SignSource`，O1 起统一）；
- 壁厚：`geometry.minimum_thickness.enforce_body_thickness`（向外补厚 + 保护面 + α 回溯）。

纪律：

1. 目标距离/间隙**只能来自 profile**；MISSING 就 fail-loud（`ProfileParameterMissing`），
   绝不 `or 0`（既有 `services/auto_fit.propose` 里那两处 `or 0` 是本批要修的反例）。
2. 没有确认的静态咬合位姿 ⇒ 咬合一律 `OCCLUSION_NOT_ASSESSED / BITE_POSE_NOT_CONFIRMED`，
   不改任何几何、不构造假 bite。
3. 接触面积是**有版本、未认证**的工程口径（见 `crown_generation_profile`），
   报告必须带上定义与 `validated=False`。
4. 每一步都必须过既有几何门；报告里如实写 before/after/iterations/convergence。
5. 禁止"部分失败仍标 PASS"：本模块只输出 `PASS / REVIEW_REQUIRED / NOT_ASSESSED /
   CONSTRAINT_CONFLICT`，把判定权交给 optimizer 与调用方。
"""

from __future__ import annotations

import numpy as np

from .crown_generation_profile import (
    CONTACT_AREA_BAND_MM,
    ProfileParameterMissing,
    CONTACT_AREA_DEFINITION,
    CONTACT_AREA_VALIDATED,
    CONTACT_AREA_VERSION,
    SOURCE_MISSING,
    CrownGenerationProfile,
)

#: solver 版本（每个 solver 各自声明，provenance 里要能追溯）
PROXIMAL_SOLVER_VERSION = "proximal_contact_solver/local_band_alpha/1"
OCCLUSAL_SOLVER_VERSION = "static_occlusal_solver/local_band_alpha/1"
THICKNESS_SOLVER_VERSION = "minimum_thickness_enforcer/bounded_outward/1"

STATUS_PASS = "PASS"
STATUS_REVIEW = "REVIEW_REQUIRED"
STATUS_NOT_ASSESSED = "NOT_ASSESSED"
STATUS_CONFLICT = "CONSTRAINT_CONFLICT"
STATUS_BLOCKED = "BLOCKED"

#: 面积目标未达成时的封顶重试次数（每次只挪一点点目标距离）
MAX_AREA_ROUNDS = 3
#: 单次面积重试允许挪动的目标距离上限（mm）——超过就不是"距离约束"了
MAX_AREA_TARGET_SHIFT_MM = 0.04


#: 既有求解器的原生状态词汇 → 本批统一词汇。
#: ⚠️ 不做归一化会让判定形同虚设：`fit_stage` 成功时返回 `PASS_ENGINEERING`，
#: 若直接拿它和 `"PASS"` 比，面积二次目标**永远不会执行**（本批实测踩到）。
_STATUS_MAP = {"PASS_ENGINEERING": STATUS_PASS, "AUTO_ACCEPTED": STATUS_PASS, "OK": STATUS_PASS,
               "MEASURED_ENGINEERING": STATUS_PASS,
               "REVIEW_REQUIRED": STATUS_REVIEW, "NOT_ASSESSED": STATUS_NOT_ASSESSED,
               "OCCLUSION_NOT_ASSESSED": STATUS_NOT_ASSESSED,
               "CONSTRAINT_CONFLICT": STATUS_CONFLICT, "BLOCKED": STATUS_BLOCKED,
               "INPUT_UNVERIFIED": STATUS_NOT_ASSESSED}


def normalize_status(status, default: str = STATUS_REVIEW) -> str:
    """把既有求解器的原生状态翻译成本批统一词汇（原生值仍保留在 ``native_status``）。"""
    return _STATUS_MAP.get(str(status or ""), default)


def vertex_areas(mesh) -> np.ndarray:
    """每个顶点分摊的三角面积（1/3 面面积），用于面积口径。"""
    areas = np.zeros(len(mesh.vertices), float)
    faces = np.asarray(mesh.faces, int)
    third = np.asarray(mesh.area_faces, float) / 3.0
    for corner in range(3):
        np.add.at(areas, faces[:, corner], third)
    return areas


def contact_area(surface, vertices, mask, *, target_mm: float, crown, band_mm: float = CONTACT_AREA_BAND_MM) -> dict:
    """接触面积（工程口径，**未认证**）。

    ``contact_area = Σ 顶点分摊面积，其中 |signed - target| <= band``
    —— 定义、版本、band 与 ``validated=False`` 必须随报告一起走，避免有人把它当真相。
    """
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {"area_mm2": None, "status": STATUS_NOT_ASSESSED, "reason": "NO_DOMAIN_VERTICES",
                "definition": CONTACT_AREA_DEFINITION, "version": CONTACT_AREA_VERSION,
                "validated": CONTACT_AREA_VALIDATED, "band_mm": band_mm}
    _, _, signed, _, valid = surface.query(np.asarray(vertices)[ids])
    areas = vertex_areas(crown)[ids]
    in_band = valid & (np.abs(np.asarray(signed, float) - float(target_mm)) <= band_mm)
    return {"area_mm2": float(areas[in_band].sum()), "in_band_vertex_count": int(in_band.sum()),
            "domain_vertex_count": int(len(ids)), "status": "MEASURED_ENGINEERING",
            "definition": CONTACT_AREA_DEFINITION, "version": CONTACT_AREA_VERSION,
            "validated": CONTACT_AREA_VALIDATED, "band_mm": band_mm}


def _side_report(surface, vertices, mask, target_mm, *, side, crown) -> dict:
    from ..geometry.crown_outer.auto_fit_adapter import measure
    report, _ = measure(vertices, mask, surface, target_mm, kind="PROXIMAL", side=side)
    return report


def _distance_block(report: dict) -> dict:
    signed = report.get("signed") or {}
    return {"min_mm": signed.get("min_mm"), "p50_mm": signed.get("p50_mm"),
            "p95_mm": signed.get("p95_mm"), "max_mm": signed.get("max_mm")}


def solve_proximal(crown, frame, margin, patches, profile: CrownGenerationProfile, *,
                   baseline=None, max_area_rounds: int = MAX_AREA_ROUNDS) -> dict:
    """近远中接触：按 profile 的目标距离（可选叠加面积目标）自适应调整外冠表面。

    复用：`fit_stage`（区域选择 + 局部带 Laplacian + α 回溯 + flip/自交/非流形/锁定位移门）。
    本函数只负责：目标解析、面积口径与二次目标、before/after 量化、收敛与未达成的如实报告。
    """
    return _solve_proximal_impl(crown, frame, margin, patches, profile,
                                baseline=baseline, max_area_rounds=max_area_rounds)


def _solve_proximal_impl(crown, frame, margin, patches, profile, *, baseline, max_area_rounds):
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains, fit_stage

    target = float(profile.require("proximal.target_distance_mm", stage="PROXIMAL")["proximal.target_distance_mm"])
    area_target, area_source = profile.effective_contact_area_target_mm2()
    reference = np.asarray((baseline if baseline is not None else crown).vertices, float)
    enabled = {"mesial": bool(profile.get("proximal.mesial_enabled", True)),
               "distal": bool(profile.get("proximal.distal_enabled", True))}
    active = {side: surface for side, surface in patches.items() if enabled.get(side, True)}
    skipped = {side: {"status": "SKIPPED_DISABLED_BY_PROFILE"} for side in patches if side not in active}

    masks, locked = fit_domains(baseline if baseline is not None else crown, frame, margin, active, "PROXIMAL")
    before = {side: _side_report(surface, reference, masks[side], target, side=side, crown=crown)
              for side, surface in active.items()}
    before_area = {side: contact_area(surface, reference, masks[side], target_mm=target, crown=crown)
                   for side, surface in active.items()}

    working, stage = crown, None
    trials = 0
    if active:
        targets = {side: target for side in active}
        working, stage, masks = fit_stage(crown, frame, margin, active, targets, "PROXIMAL")
        trials = len(stage.get("trials") or [])

    rounds = 0
    area_note = None
    if (stage is not None and normalize_status(stage.get("status")) == STATUS_PASS
            and area_target is not None
            and active and area_source != SOURCE_MISSING):
        # 面积是**二次目标**：只允许在目标距离附近小幅挪动（≤ MAX_AREA_TARGET_SHIFT_MM/轮），
        # 且每一轮都必须照旧过几何门（fit_stage 内部的门不变）。
        best = working
        best_error = None
        for side in sorted(active):
            areas = contact_area(active[side], np.asarray(best.vertices), masks[side],
                                 target_mm=target, crown=crown)
            if areas.get("area_mm2") is None:
                continue
            error = float(areas["area_mm2"]) - float(area_target)
            best_error = error if best_error is None else max(abs(best_error), abs(error))
            direction = 1.0 if error > 0 else -1.0        # 面积偏大 ⇒ 离远一点；偏小 ⇒ 压近一点
            for _ in range(max_area_rounds):
                if abs(error) <= 0.05 * float(area_target):
                    break
                shift = direction * min(MAX_AREA_TARGET_SHIFT_MM,
                                        abs(error) / max(float(area_target), 1e-9) * 0.05)
                candidate, report, masks_try = fit_stage(
                    best, frame, margin, {side: active[side]},
                    {side: max(0.0, target + shift)}, "PROXIMAL")
                if normalize_status(report.get("status")) != STATUS_PASS:
                    break
                new_area = contact_area(active[side], np.asarray(candidate.vertices), masks_try[side],
                                        target_mm=target, crown=crown)
                if new_area.get("area_mm2") is None:
                    break
                new_error = float(new_area["area_mm2"]) - float(area_target)
                if abs(new_error) >= abs(error) - 1e-12:
                    break                                   # 没更好就停（不做无收益步）
                best, masks, working, error = candidate, masks_try, candidate, new_error
                rounds += 1
        if best is not None:
            working = best
        if best_error is not None and abs(error) > 0.05 * float(area_target):
            area_note = "CONTACT_AREA_TARGET_NOT_REACHED"

    after = {side: _side_report(surface, np.asarray(working.vertices), masks[side], target,
                                side=side, crown=crown) for side, surface in active.items()}
    after_area = {side: contact_area(surface, np.asarray(working.vertices), masks[side],
                                     target_mm=target, crown=crown) for side, surface in active.items()}

    per_side = {}
    for side in active:
        per_side[side] = {
            "target_distance_mm": target,
            "before_distance": _distance_block(before[side]),
            "after_distance": _distance_block(after[side]),
            "before_penetration_mm": (before[side].get("penetration") or {}).get("max_mm", 0.0),
            "after_penetration_mm": (after[side].get("penetration") or {}).get("max_mm", 0.0),
            "before_target_error": before[side].get("target_error"),
            "after_target_error": after[side].get("target_error"),
            "contact_area": {"before": before_area[side], "after": after_area[side],
                             "target_mm2": area_target, "source": area_source},
            "sign_method": before[side].get("sign_method"),
            "domain_vertex_count": before[side].get("domain_vertex_count"),
            "iterations": trials,
            "convergence": None,
            "status": None,
        }
        if after[side].get("status") == STATUS_NOT_ASSESSED:
            per_side[side].update(status=STATUS_NOT_ASSESSED, convergence="NO_VALID_SAMPLES")
        elif after[side].get("target_error") and before[side].get("target_error"):
            improved = after[side]["target_error"]["p95_mm"] < before[side]["target_error"]["p95_mm"] - 1e-9
            per_side[side]["convergence"] = "TARGET_ERROR_DECREASED" if improved else "NO_FURTHER_IMPROVEMENT"

    native_status = (stage or {}).get("status")
    status = STATUS_NOT_ASSESSED if not active else normalize_status(native_status)
    reasons = []
    if not active:
        reasons.append("NO_ENABLED_CONFIRMED_NEIGHBOR")
    if skipped:
        reasons.append("SIDES_DISABLED_BY_PROFILE:%s" % ",".join(sorted(skipped)))
    if area_note:
        reasons.append(area_note)
        if status == STATUS_PASS:
            status = STATUS_REVIEW
    if (stage or {}).get("reason"):
        reasons.append(stage["reason"])
    for side, block in per_side.items():
        comment = block.pop("status")
        if comment:
            block["side_status"] = comment
    return {
        "solver": "ProximalContactSolver",
        "algorithm_version": PROXIMAL_SOLVER_VERSION,
        "reused": ["auto_fit_adapter.fit_stage", "auto_fit_adapter.fit_domains",
                   "services.contact_measurement.measure_surface",
                   "crown_outer.deform_gate.deformation_gate"],
        "status": status, "native_status": native_status, "reasons": reasons,
        "target_distance_mm": target, "target_source": profile.source("proximal.target_distance_mm"),
        "contact_area_target_mm2": area_target, "contact_area_target_source": area_source,
        "contact_area_definition": CONTACT_AREA_DEFINITION,
        "contact_area_validated": CONTACT_AREA_VALIDATED,
        "per_side": per_side, "skipped_sides": skipped,
        "iterations": trials, "area_rounds": rounds,
        "convergence": "TARGET_ERROR_DECREASED" if any(
            block.get("convergence") == "TARGET_ERROR_DECREASED" for block in per_side.values()) else "NO_FURTHER_IMPROVEMENT",
        "locked_vertex_count": int(np.count_nonzero(locked)),
        "margin_locked": True,
        "stage_report": stage,
        "displacement_mm": ({"max_mm": float(np.linalg.norm(
            np.asarray(working.vertices) - np.asarray(crown.vertices), axis=1).max())}),
        "geometry": working,
        "geometry_changed": bool(np.linalg.norm(
            np.asarray(working.vertices) - np.asarray(crown.vertices), axis=1).max() > 1e-9),
    }


# ---------------------------------------------------------------- 咬合（静态）
def solve_occlusion(crown, frame, margin, antagonist, profile: CrownGenerationProfile, *,
                    bite_pose: dict | None, patches=None, baseline=None, protected=None) -> dict:
    """静态咬合自适应。**只消费已确认的静态咬合位姿**，不做动态咬合。

    硬规则（规格 §5）：没有 CONFIRMED_BITE_POSE ⇒ ``OCCLUSION_NOT_ASSESSED``，
    reason ``BITE_POSE_NOT_CONFIRMED``，几何原样返回。
    """
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains, fit_stage

    pose = dict(bite_pose or {})
    pose_ok = bool(pose.get("pose_confirmed") and pose.get("revision_id") and pose.get("transform"))
    target, target_source = profile.value("occlusion.target_clearance_mm")
    enabled = bool(profile.get("occlusion.enabled", True))
    require_pose = bool(profile.get("occlusion.require_confirmed_bite_pose", True))

    if not enabled:
        return {"solver": "StaticOcclusalSolver", "algorithm_version": OCCLUSAL_SOLVER_VERSION,
                "status": STATUS_NOT_ASSESSED, "reason": "OCCLUSION_DISABLED_BY_PROFILE",
                "geometry_changed": False, "geometry": None}
    if require_pose and not pose_ok:
        return {"solver": "StaticOcclusalSolver", "algorithm_version": OCCLUSAL_SOLVER_VERSION,
                "status": "OCCLUSION_NOT_ASSESSED", "reason": "BITE_POSE_NOT_CONFIRMED",
                "bite_pose_status": "NOT_CONFIRMED", "geometry_changed": False, "geometry": None,
                "note": "没有已确认的静态咬合位姿时不做任何咬合调整，也不构造替代位姿。"}
    if target_source == SOURCE_MISSING or target is None:
        return {"solver": "StaticOcclusalSolver", "algorithm_version": OCCLUSAL_SOLVER_VERSION,
                "status": "OCCLUSION_NOT_ASSESSED", "reason": "OCCLUSAL_TARGET_UNCONFIRMED",
                "geometry_changed": False, "geometry": None}
    if antagonist is None:
        return {"solver": "StaticOcclusalSolver", "algorithm_version": OCCLUSAL_SOLVER_VERSION,
                "status": "OCCLUSION_NOT_ASSESSED", "reason": "NO_CONFIRMED_ANTAGONIST",
                "geometry_changed": False, "geometry": None}

    target = float(target)
    reference = np.asarray((baseline if baseline is not None else crown).vertices, float)
    masks, locked = fit_domains(baseline if baseline is not None else crown, frame, margin,
                                {"occlusal": antagonist}, "OCCLUSAL", protected)
    before = _side_report(antagonist, reference, masks["occlusal"], target, side="occlusal", crown=crown)
    proximal_before = _proximal_snapshot(patches, reference, frame, margin, profile, crown)

    moved, stage, masks = fit_stage(crown, frame, margin, {"occlusal": antagonist},
                                    {"occlusal": target}, "OCCLUSAL",
                                    protected=protected, domain_mesh=baseline)
    after = _side_report(antagonist, np.asarray(moved.vertices), masks["occlusal"], target,
                         side="occlusal", crown=crown)
    proximal_after = _proximal_snapshot(patches, np.asarray(moved.vertices), frame, margin, profile, crown)

    reasons = []
    if not _proximal_not_worse(proximal_before, proximal_after):
        moved = crown                                      # 回滚：不许为了咬合破坏近远中接触
        reasons.append("JOINT_CONTACT_PRESERVATION_FAILED")
        stage = {**stage, "status": STATUS_CONFLICT}

    adjusted = np.linalg.norm(np.asarray(moved.vertices) - reference, axis=1)
    area = None
    if masks["occlusal"] is not None:
        area = float(np.asarray(crown.area_faces)[
            np.flatnonzero(np.any(np.asarray(masks["occlusal"])[np.asarray(crown.faces)], axis=1))].sum())
    return {
        "solver": "StaticOcclusalSolver", "algorithm_version": OCCLUSAL_SOLVER_VERSION,
        "reused": ["auto_fit_adapter.fit_stage", "auto_fit_adapter.fit_domains",
                   "measurement_core.SignSource(与 Occlusion 同源)"],
        "status": normalize_status(stage.get("status")), "native_status": stage.get("status"),
        "reasons": reasons + ([stage["reason"]] if stage.get("reason") else []),
        "bite_pose_status": "CONFIRMED", "bite_revision_id": pose.get("revision_id"),
        "target_clearance_mm": target, "target_source": target_source,
        "before_min": (before.get("signed") or {}).get("min_mm"),
        "after_min": (after.get("signed") or {}).get("min_mm"),
        "penetration_before_mm": (before.get("penetration") or {}).get("max_mm", 0.0),
        "penetration_after_mm": (after.get("penetration") or {}).get("max_mm", 0.0),
        "deficit_before": before.get("deficit"), "deficit_after": after.get("deficit"),
        "adjusted_area_mm2": area,
        "iterations": len(stage.get("trials") or []),
        "convergence": "CLEARANCE_IMPROVED" if (before.get("deficit") and after.get("deficit")
                                               and after["deficit"]["p95_mm"] < before["deficit"]["p95_mm"] - 1e-9)
                       else "NO_FURTHER_IMPROVEMENT",
        "proximal_recheck": {"before": proximal_before, "after": proximal_after,
                             "not_worse": _proximal_not_worse(proximal_before, proximal_after)},
        "geometry_changed": bool(adjusted.max() > 1e-9),
        "displacement_mm": {"max_mm": float(adjusted.max()), "p95_mm": float(np.quantile(adjusted, .95))},
        "cervical_locked_vertex_count": int(np.count_nonzero(locked)),
        "stage_report": stage,
        "geometry": moved,
    }


def _proximal_snapshot(patches, vertices, frame, margin, profile, crown) -> dict:
    """咬合前后都要重新量近远中（规格 §5.7：每轮之后重新跑 ContactMeasurement）。"""
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains

    if not patches:
        return {}
    target, source = profile.value("proximal.target_distance_mm")
    if source == SOURCE_MISSING or target is None:
        return {"status": STATUS_NOT_ASSESSED, "reason": "PROXIMAL_TARGET_UNCONFIRMED"}
    masks, _ = fit_domains(crown, frame, margin, patches, "PROXIMAL")
    return {side: _side_report(surface, np.asarray(vertices), masks[side], float(target),
                               side=side, crown=crown) for side, surface in patches.items()}


def _proximal_not_worse(before: dict, after: dict) -> bool:
    if not before or not after:
        return True
    for side, block in before.items():
        if not isinstance(block, dict) or "target_error" not in block:
            continue
        previous = block.get("target_error")
        current = (after.get(side) or {}).get("target_error")
        if previous and (not current or current["max_mm"] > previous["max_mm"] + 1e-9):
            return False
    return True


# ---------------------------------------------------------------- 壁厚
def enforce_minimum_thickness(outer, inner, profile: CrownGenerationProfile, *,
                              protected_surfaces=(), max_iterations: int = 12) -> dict:
    """最小厚度强制（编排 `geometry.minimum_thickness.enforce_body_thickness`）。

    复用既有实现（向外补厚 + 保护面锁 + α 回溯 + deformation_gate + 保护面穿越检查），
    这里只做：profile 取值、键名归一（thin_area_before/after 等）、未达成时的如实结论。
    """
    from ..geometry.minimum_thickness import enforce_body_thickness

    minimum, source = profile.value("thickness.minimum_thickness_mm")
    if source == SOURCE_MISSING or minimum is None:
        raise ProfileParameterMissing(["thickness.minimum_thickness_mm"], "THICKNESS")
    enforce = bool(profile.get("thickness.enforce", False))
    if not enforce:
        return {"solver": "MinimumThicknessEnforcer", "algorithm_version": THICKNESS_SOLVER_VERSION,
                "status": STATUS_NOT_ASSESSED, "reason": "THICKNESS_ENFORCE_DISABLED_BY_PROFILE",
                "configured_minimum_mm": float(minimum), "geometry_changed": False}

    result, report = enforce_body_thickness(outer, inner, float(minimum),
                                           protected_surfaces=tuple(protected_surfaces),
                                           max_iterations=max_iterations)
    before, after = report.get("before") or {}, report.get("after") or {}
    accepted = len(report.get("steps") or [])
    rejected = 1 if report.get("last_rejected_trial") else 0
    changed = float(np.linalg.norm(np.asarray(result.vertices) - np.asarray(outer.vertices), axis=1).max())
    return {
        "solver": "MinimumThicknessEnforcer", "algorithm_version": THICKNESS_SOLVER_VERSION,
        "reused": ["geometry.minimum_thickness.enforce_body_thickness",
                   "crown_outer.deform_gate.deformation_gate",
                   "crown_outer.auto_fit_adapter.surface_crossings"],
        "status": STATUS_PASS if report.get("status", "").startswith("PASS") else STATUS_REVIEW,
        "native_status": report.get("status"), "reason": report.get("reason"),
        "configured_minimum_mm": float(minimum),
        "minimum_before_mm": before.get("min_mm"), "minimum_after_mm": after.get("min_mm"),
        "p05_before_mm": before.get("p05_mm"), "p05_after_mm": after.get("p05_mm"),
        "thin_area_before_mm2": None, "thin_area_after_mm2": None,
        "remaining_thin_samples": report.get("remaining_thin_samples"),
        "protected_vertex_count": report.get("protected_vertex_count"),
        "accepted_steps": accepted, "rejected_steps": rejected,
        "max_displacement_mm": report.get("max_displacement_mm"),
        "p95_displacement_mm": float(np.quantile(
            np.linalg.norm(np.asarray(result.vertices) - np.asarray(outer.vertices), axis=1), .95)),
        "intaglio_preserved": report.get("intaglio_preserved"),
        "scope": report.get("scope"), "contact_coverage": report.get("contact_coverage"),
        "continuous_minimum_certified": report.get("continuous_minimum_certified"),
        "geometry_changed": bool(changed > 1e-9), "result": result, "native_report": report,
    }


def thickness_protected_surfaces(design) -> tuple:
    """保护面（Margin seam / 颈缘锁带 / 已确认接触区 / 内冠贴合面）——复用既有 service。"""
    from .contact_measurement import thickness_protection_surfaces
    return tuple(thickness_protection_surfaces(design) or ())
