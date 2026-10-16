"""轴向初放置约束（template initial placement / axial scale）。

**为什么需要它**：`initialize_from_margin` 横向按 Margin 拟合并对齐模板颈缘，
但并不根据对颌决定冠高。冠高必须在当前病例的确认位姿下测量，
不能混用其他设计的模板尺寸或咬合结果。

本模块把这件事**前置**：在 `rebuild-cervical` 之前，先算咬合可行区间，再把放置好的
冠体轴向放进区间；放不进去就明确报 ``INSUFFICIENT_OCCLUSAL_SPACE``，**不让下游去削半颗牙**。

口径（三个高度都沿**就位轴**，零点 = 已确认 Margin 平面，与 `MarginCurtain` 一致）
--------------------------------------------------------------------------------
- ``prep_top_mm``：预备体（支撑件 / 腔体）最高点的高度；
- ``antagonist_ceiling_mm``：对颌在每一柱（0.4mm）内**面向冠的最低点**；
- ``available_space_mm``：逐柱 ``天花板 − 预备体表面``（冠的咬合壁就长在这段空间里）；
- ``minimum_required_space_mm`` = ``min_thickness + target_clearance``；
- 全局恒高区间：``lower = prep_top + min_thickness``、``upper = min(天花板) − target_clearance``；
  若该区间为空，仍需查看逐柱空间，不能直接断言局部冠形不可行。

Crown 1.0 增补：只要对颌证据充分且空间可行，冠顶不能因为“已经落在区间内”就停在半空。
当冠顶距离 ``upper`` 仍大于 deadband 时，以颈缘带为固定锚点做**有限轴向增高**，先把模板
解剖面送到对颌附近，再交给现有 occlusal adaptation 按对颌局部表面削出基础凹凸。

**不修改**：margin detection、cervical reconstruction、boolean、stitch、QC gate。
只做"把牙放在正确的高度和尺度上"这一件事，且只动**颈缘带以上**的顶点（颈缘位移精确为 0）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry.crown_outer.deform_gate import deformation_gate
from ..geometry.io import load_mesh
from ..geometry.nonplanar_crown import MarginCurtain

CELL_MM = 0.4                  # 与咬合适配共用同一逐柱天花板
MIN_OCCLUSAL_THICKNESS_MM = 0.5
TARGET_CLEARANCE_MM = 0.05
AXIAL_SCALE_MIN = 0.75         # 允许的最小轴向缩放（低于此值不算"合理范围内的缩放"）
AXIAL_SCALE_MAX = 1.35         # 1.0：只允许有限增高；再高说明模板/牙位空间本身不匹配
CONTACT_DEADBAND_MM = 0.08     # 距对颌目标上界 ≤ 此值视为已到位，避免反复微缩放
CERVICAL_ANCHOR_MM = 0.4       # 颈缘带（≤ 此高度）完全不动；缩放从锚点向冠顶线性展开
OCCLUSAL_NORMAL_MIN = 0.35
IMPLEMENTATION = "services.axial_placement/occlusal_envelope/3"


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _cells(points, origin, basis, cell_mm=CELL_MM):
    local = (np.asarray(points, float) - origin) @ basis
    return np.floor(local[:, :2] / cell_mm).astype(np.int64)


def occlusal_envelope(design, margin, axis, frame, *, surface=None, prep_path=None,
                      crown_mesh=None, min_thickness_mm=MIN_OCCLUSAL_THICKNESS_MM,
                      target_clearance_mm=TARGET_CLEARANCE_MM):
    """算这位病例的咬合可行区间与逐柱可用空间（只读，不改任何几何）。

    ``surface``：已确认的对颌参考面（`FitSurface`）；缺省时自己用
    ``confirmed_antagonist_surface(strict=False)`` 取，取不到就如实标记 ``NOT_ASSESSED``。
    """
    from .contact_measurement import confirmed_antagonist_surface
    from .occlusal_adaptation import build_ceiling

    axis = _unit(axis)
    margin = np.asarray(margin, float)
    curtain = MarginCurtain(margin, axis)
    basis = curtain.frame.matrix()
    origin = np.asarray(curtain.frame.origin, float)
    report = {"implementation": IMPLEMENTATION, "status": "NOT_RUN", "reasons": [],
              "min_thickness_mm": float(min_thickness_mm),
              "target_clearance_mm": float(target_clearance_mm),
              "minimum_required_space_mm": float(min_thickness_mm + target_clearance_mm),
              "required_min_space_mm": float(min_thickness_mm + target_clearance_mm),
              "signed_convention": "ceiling_minus_surface_height; positive=space, negative=penetration",
              "search_direction": "along_confirmed_insertion_axis",
              "coverage": None, "valid_ratio": 0.0,
              "cell_mm": CELL_MM, "axis": axis.tolist()}

    # 模板**原始**高度（文件自身 +Z 就位方向）—— 与"放置后冠体高度"对照，用来判断
    # 是"放矮了/放高了"还是"模板本身就过大"。
    try:
        from .template_source import resolve_template_source
        src = resolve_template_source(design)
        if getattr(src, "library_obj", None) is not None:
            from .auto_fit_templates import canonical_library_crown
            template_mesh, _meta = canonical_library_crown(src.library_obj, int(design["target_fdi"]))
        else:
            template_mesh = load_mesh(Path(src.path))
        extent = np.asarray(template_mesh.bounds[1] - template_mesh.bounds[0], float)
        report["template_source"] = {"source_kind": getattr(src, "source_kind", None),
                                     "path_name": Path(getattr(src, "path", "")).name if getattr(src, "path", None) else None,
                                     "fdi": design.get("target_fdi")}
        report["template_bounds_mm"] = [float(v) for v in extent]
        report["template_height_mm"] = float(extent[2])
    except Exception as exc:      # 模板取不到不影响区间判定，但如实记录
        report["template_height_mm"] = None
        report["template_height_error"] = str(exc)[:160]

    if crown_mesh is not None:
        h = curtain.axial_above(np.asarray(crown_mesh.vertices, float))
        report["placed_crown_height_mm"] = float(h.max() - min(0.0, float(h.min())))
        report["margin_to_top_mm"] = float(h.max())
        report["crown_top_mm"] = float(h.max())

    # ── 预备体（腔体优先：它才是"冠必须盖住"的高度） ──
    prep_points = None
    source = None
    for label, path in (("support", prep_path), ("cutter_path", design.get("cutter_path"))):
        if path and Path(path).is_file():
            prep_points = np.asarray(load_mesh(Path(path)).vertices, float)
            source = label
            break
    if prep_points is None:
        report.update(status="NOT_ASSESSED", reason="NO_PREP_GEOMETRY",
                      reasons=["NO_PREP_GEOMETRY"])
        return report
    report["prep_source"] = source
    prep_h = curtain.axial_above(prep_points)
    report["prep_top_mm"] = float(prep_h.max())
    prep_top_region = prep_h >= max(1.0, 0.5 * float(prep_h.max()))
    report["lower_bound_mm"] = float(prep_h.max() + min_thickness_mm)

    if surface is None:
        surface, _bite, error = confirmed_antagonist_surface(design, _frame_ref(frame), strict=False)
        if surface is None:
            report.update(status="NOT_ASSESSED",
                          reason=(error or {}).get("reason") or "OCCLUSAL_REFERENCE_UNAVAILABLE",
                          reasons=["OCCLUSAL_REFERENCE_UNAVAILABLE"])
            return report
    ant = surface.mesh
    ant_points = np.vstack((np.asarray(ant.vertices, float), np.asarray(ant.triangles_center, float)))
    ant_h = curtain.axial_above(ant_points)
    ant_nz = np.concatenate((np.asarray(ant.vertex_normals, float) @ axis,
                             np.asarray(ant.face_normals, float) @ axis))
    lookup, ceiling_stats = build_ceiling(((ant_points - origin) @ basis)[:, :2], ant_h, ant_nz,
                                          cell_mm=CELL_MM)
    report["ceiling_columns"] = ceiling_stats(((ant_points - origin) @ basis)[:1])["cell_count"]

    # ── 逐柱可用空间：冠的咬合足印上，天花板 − 预备体表面 ──
    if crown_mesh is not None:
        crown_v = np.asarray(crown_mesh.vertices, float)
        crown_h = curtain.axial_above(crown_v)
        nz = np.asarray(crown_mesh.vertex_normals, float) @ axis
        top = float(crown_h.max())
        window_lo = top - max((top - min(0.0, float(crown_h.min()))) / 3.0, 2.0)
        cap = (crown_h >= window_lo) & (nz > OCCLUSAL_NORMAL_MIN)
        cap_keys = _cells(crown_v[cap], origin, basis)
    else:
        cap_keys = _cells(prep_points[prep_top_region], origin, basis)
    prep_local = (prep_points - origin) @ basis
    prep_keys = np.floor(prep_local[:, :2] / CELL_MM).astype(np.int64)
    cap_keys = np.asarray(cap_keys, np.int64)
    if not len(cap_keys):
        report.update(status="NOT_ASSESSED", reason="EMPTY_TARGET_TOOTH_ROI", reasons=["EMPTY_TARGET_TOOTH_ROI"])
        return report
    roi_lo, roi_hi = cap_keys.min(axis=0) - 1, cap_keys.max(axis=0) + 1
    in_roi = np.all((prep_keys >= roi_lo) & (prep_keys <= roi_hi), axis=1)
    top_samples = prep_top_region & in_roi
    if not top_samples.any():
        report.update(status="NOT_ASSESSED", reason="NO_PREP_TOP_IN_TARGET_ROI", reasons=["NO_PREP_TOP_IN_TARGET_ROI"])
        return report
    prep_ceiling = lookup(prep_local[top_samples])
    report["prep_top_mm"] = float(prep_h[top_samples].max())
    report["lower_bound_mm"] = float(report["prep_top_mm"] + min_thickness_mm)
    valid = np.isfinite(prep_ceiling)
    report["coverage"] = {"prep_requested": int(top_samples.sum()), "prep_valid": int(valid.sum()),
                          "crown": ceiling_stats((np.asarray(crown_mesh.vertices)[cap] - origin) @ basis)
                          if crown_mesh is not None else None}
    report["valid_ratio"] = float(valid.mean())
    if not valid.any() or report["valid_ratio"] < 0.8:
        report.update(status="NOT_ASSESSED", reason="NO_MEASURABLE_COLUMNS",
                      reasons=["NO_MEASURABLE_COLUMNS"])
        return report
    ceiling_all = prep_ceiling[valid]
    spaces = ceiling_all - prep_h[top_samples][valid]
    report["measured_columns"] = int(len(spaces))
    report["antagonist_ceiling_mm"] = {
        "min": float(ceiling_all.min()), "p05": float(np.quantile(ceiling_all, 0.05)),
        "p50": float(np.quantile(ceiling_all, 0.5))}
    report["available_space_mm"] = {
        "min": float(spaces.min()), "p05": float(np.quantile(spaces, 0.05)),
        "p50": float(np.quantile(spaces, 0.5)), "max": float(spaces.max())}
    required = float(min_thickness_mm + target_clearance_mm)
    short = spaces < required
    report["columns_below_required"] = int(short.sum())
    report["worst_shortfall_mm"] = float(max(0.0, required - spaces.min()))
    report["upper_bound_mm"] = float(ceiling_all.min() - target_clearance_mm)

    # ── 判定 ──
    reasons = []
    if np.quantile(spaces, .05) < required:
        reasons.append("INSUFFICIENT_OCCLUSAL_SPACE")
    if short.any():
        reasons.append("MIN_THICKNESS_NOT_SATISFIABLE_IN_%d_COLUMNS" % int(short.sum()))
    report["reasons"] = reasons
    if report["lower_bound_mm"] > report["upper_bound_mm"]:
        reasons.append("GLOBAL_HEIGHT_INTERVAL_EMPTY_LOCAL_SHAPE_MAY_FIT")
    report["status"] = ("INSUFFICIENT_OCCLUSAL_SPACE" if "INSUFFICIENT_OCCLUSAL_SPACE" in reasons
                        else "FEASIBLE" if not reasons else "MARGINAL")
    report["verdict"] = ("可行区间 [%.2f, %.2f] mm（lower=预备体顶+壁厚，upper=最低天花板−间隙）；"
                         "逐柱可用空间 min %.2f / p05 %.2f / p50 %.2f mm，需要 %.2f mm"
                         % (report["lower_bound_mm"], report["upper_bound_mm"],
                            spaces.min(), np.quantile(spaces, 0.05), np.quantile(spaces, 0.5), required))
    if "INSUFFICIENT_OCCLUSAL_SPACE" in reasons:
        report["verdict"] += "；当前确认位姿下至少 5% 目标柱位的局部材料空间不足，需人工复核"
    return report


def _frame_ref(frame):
    """把 frame 对象转成 `confirmed_antagonist_surface` 要的 ref 形状。"""
    if isinstance(frame, dict):
        return frame
    return {"revision_id": getattr(frame, "revision_id", None), "status": "CONFIRMED"}


def plan_axial_placement(envelope, *, crown_top_mm, crown_bottom_mm=None,
                         max_scale_down=AXIAL_SCALE_MIN, max_scale_up=AXIAL_SCALE_MAX,
                         contact_deadband_mm=CONTACT_DEADBAND_MM):
    """在区间内给出颈缘锁定的轴向放置方案（只算方案，不动几何）。

    Crown 1.0 约束：
    - 太高：缩到 ``upper``；
    - 太低或虽然可行但仍离对颌明显：在不动颈缘带的前提下增高到 ``upper``；
    - 只有距 ``upper`` 已进入 deadband 才不再动。

    这样初始 crown 不会停在可行区间下半部而“咬不到”；随后现有 occlusal adaptation
    只需要处理局部穿透/凹凸，不必凭空把整颗牙往上长。
    """
    if envelope.get("status") == "NOT_ASSESSED":
        return {"mechanism": "NOT_ASSESSED", "feasible": False, "axial_scale": None,
                "target_top_mm": None, "reason": envelope.get("reason")}
    lower, upper = float(envelope["lower_bound_mm"]), float(envelope["upper_bound_mm"])
    if lower > upper:
        hard = envelope.get("status") == "INSUFFICIENT_OCCLUSAL_SPACE"
        return {"mechanism": "INSUFFICIENT_OCCLUSAL_SPACE" if hard else "LOCAL_SHAPE_REQUIRED",
                "feasible": False, "axial_scale": None,
                "target_top_mm": None, "lower_bound_mm": lower, "upper_bound_mm": upper,
                "reason": ("逐柱空间不足" if hard else "全局恒定冠高不可行；逐柱空间允许局部冠形调整")}

    top = float(crown_top_mm)
    anchor = CERVICAL_ANCHOR_MM

    # 冠太高：保留原有安全缩短逻辑。
    if top > upper:
        if crown_bottom_mm is not None and crown_bottom_mm > upper:
            return {"mechanism": "REQUIRED_AXIAL_SCALE_OUT_OF_RANGE", "feasible": False,
                    "axial_scale": None, "target_top_mm": upper,
                    "reason": "连冠底都在上界之上，属整体过大而非高度问题"}
        span = max(top - anchor, 1e-6)
        scale = (upper - anchor) / span
        if scale < max_scale_down:
            return {"mechanism": "REQUIRED_AXIAL_SCALE_OUT_OF_RANGE", "feasible": False,
                    "axial_scale": float(scale), "target_top_mm": upper,
                    "reason": ("需要把冠体压到 %.0f%%（< 允许下限 %.0f%%）：模板牙体本身过大，"
                               "应先换更短的模板，而不是硬压" % (100 * scale, 100 * max_scale_down)),
                    "lower_bound_mm": lower, "upper_bound_mm": upper, "crown_top_mm": top}
        return {"mechanism": "AXIAL_SCALE", "feasible": True, "axial_scale": float(scale),
                "target_top_mm": float(upper), "anchor_mm": anchor,
                "reason": "颈缘被钉住：把颈缘带以上的冠体轴向压到对颌安全上界"}

    # 已经真正到达对颌目标附近：不为几十微米反复改形。
    gap_to_target = upper - top
    if gap_to_target <= float(contact_deadband_mm):
        return {"mechanism": "TRANSLATE", "feasible": True, "axial_scale": 1.0,
                "target_top_mm": top, "lower_bound_mm": lower, "upper_bound_mm": upper,
                "gap_to_target_mm": float(gap_to_target),
                "reason": "冠顶已进入对颌目标 deadband，不需要轴向调整"}

    # 冠太矮/悬空：从颈缘固定带向冠顶有限增高到 upper。
    span = top - anchor
    target_span = upper - anchor
    if span <= 1e-6 or target_span <= 1e-6:
        return {"mechanism": "CROWN_TOO_SHORT_FOR_PREP", "feasible": False,
                "axial_scale": None, "target_top_mm": upper,
                "lower_bound_mm": lower, "upper_bound_mm": upper,
                "reason": "冠顶未高出颈缘固定带，无法在不移动 Margin 的前提下增高"}
    scale = target_span / span
    if scale > max_scale_up:
        return {"mechanism": "REQUIRED_AXIAL_SCALE_OUT_OF_RANGE", "feasible": False,
                "axial_scale": float(scale), "target_top_mm": upper,
                "lower_bound_mm": lower, "upper_bound_mm": upper, "crown_top_mm": top,
                "reason": ("需要把冠体增高到 %.0f%%（> 允许上限 %.0f%%）：模板过短或牙位/模板不匹配，"
                           "应换模板或进入局部形态求解，不能硬拉" % (100 * scale, 100 * max_scale_up))}
    return {"mechanism": "AXIAL_SCALE", "feasible": True, "axial_scale": float(scale),
            "target_top_mm": float(upper), "anchor_mm": anchor,
            "lower_bound_mm": lower, "upper_bound_mm": upper,
            "gap_to_target_mm": float(gap_to_target),
            "reason": "Crown 1.0：保持颈缘不动，将冠顶有限增高到对颌安全上界"}


def apply_axial_placement(mesh, margin, axis, plan):
    """按方案把**颈缘带以上**的冠体轴向缩放（颈缘带位移精确为 0）。返回 ``(mesh|None, report)``。"""
    axis = _unit(axis)
    curtain = MarginCurtain(np.asarray(margin, float), axis)
    report = {"implementation": IMPLEMENTATION, "mechanism": plan.get("mechanism"),
              "target_top_mm": plan.get("target_top_mm"), "axial_scale": plan.get("axial_scale")}
    if not plan.get("feasible"):
        report["applied"] = False
        report["reason"] = plan.get("reason")
        return None, report
    scale = float(plan["axial_scale"])
    if abs(scale - 1.0) < 1e-9:
        report.update(applied=False, reason="已在区间内")
        return mesh.copy(), report
    anchor = float(plan.get("anchor_mm", CERVICAL_ANCHOR_MM))
    vertices = np.asarray(mesh.vertices, float)
    h = curtain.axial_above(vertices)
    upper = h > anchor
    new_h = h.copy()
    new_h[upper] = anchor + (h[upper] - anchor) * scale
    delta = new_h - h
    placed = mesh.copy()
    placed.vertices = vertices + axis[None, :] * delta[:, None]
    faces = np.asarray(mesh.faces, np.int64)
    moved = np.abs(delta) > 1e-12
    gate = deformation_gate(vertices, np.asarray(placed.vertices, float), faces, scale_mm=0.1,
                            locked_ids=np.flatnonzero(~moved),
                            focus_faces=np.flatnonzero(np.any(moved[faces], axis=1)))
    h_new = curtain.axial_above(np.asarray(placed.vertices, float))
    report.update(applied=bool(gate.ok), gate_ok=bool(gate.ok), violations=list(gate.violations),
                  gate=dict(gate.metrics),
                  crown_top_before_mm=float(h.max()), crown_top_after_mm=float(h_new.max()),
                  cervical_band_displacement_max_mm=float(np.abs(delta[h <= anchor]).max()) if (h <= anchor).any() else 0.0,
                  moved_vertex_count=int(moved.sum()), vertex_count=int(len(vertices)),
                  watertight=bool(placed.is_watertight))
    if not gate.ok:
        report["reason"] = "轴向缩放未通过形变硬门：%s" % list(gate.violations)
        return None, report
    return placed, report


def coarse_height_fit(design, mesh, margin, axis, frame, envelope, *, target_mm=0.05):
    """Use only cervical-anchored Z scale to enter the existing fine solver's range."""
    from .contact_measurement import confirmed_antagonist_surface
    from .occlusal_adaptation import build_ceiling

    report = {'implementation': 'services.axial_placement/coarse_height_fit/1',
              'status': 'NOT_APPLIED', 'md_scale': 1.0, 'bl_scale': 1.0,
              'translation_mm': [0.0, 0.0, 0.0], 'z_scale': 1.0,
              'reason': None, 'trials': []}
    params = json.loads(design.get('parameters_json') or '{}')
    anchor = max(CERVICAL_ANCHOR_MM, float(params.get('margin_band_mm') or .8) + .4)
    report['cervical_anchor_mm'] = anchor
    if envelope['status'] == 'INSUFFICIENT_OCCLUSAL_SPACE':
        report['reason'] = 'PREP_ANTAGONIST_SPACE_INSUFFICIENT'
        return None, report
    if envelope.get('valid_ratio', 0) < .8:
        report['reason'] = 'OCCLUSAL_COVERAGE_INSUFFICIENT'
        return None, report
    surface, _, error = confirmed_antagonist_surface(design, _frame_ref(frame), strict=False)
    if surface is None:
        report['reason'] = (error or {}).get('reason', 'ANTAGONIST_UNAVAILABLE')
        return None, report
    curtain = MarginCurtain(np.asarray(margin, float), axis)
    basis, origin = curtain.frame.matrix(), curtain.frame.origin
    ant = surface.mesh
    ant_points = np.vstack((ant.vertices, ant.triangles_center))
    lookup, _ = build_ceiling(((ant_points-origin) @ basis)[:, :2],
        curtain.axial_above(ant_points),
        np.r_[ant.vertex_normals @ axis, ant.face_normals @ axis], cell_mm=CELL_MM)

    def clearance(candidate):
        h = curtain.axial_above(candidate.vertices)
        top = float(h.max())
        window_lo = top - max((top - min(0.0, float(h.min()))) / 3.0, 2.0)
        mask = (h >= window_lo) & (np.asarray(candidate.vertex_normals) @ axis > OCCLUSAL_NORMAL_MIN)
        if not mask.any():
            return None, 0.0
        ceiling = lookup((np.asarray(candidate.vertices)[mask] - origin) @ basis)
        valid = np.isfinite(ceiling)
        return (float(np.min(ceiling[valid] - h[mask][valid])) if valid.any() else None,
                float(valid.mean()))

    before, coverage = clearance(mesh)
    report.update(before_min_clearance_mm=before, crown_valid_ratio=coverage)
    if before is None or coverage < .8:
        report['reason'] = 'CROWN_CEILING_COVERAGE_INSUFFICIENT'
        return None, report
    fine_range = -(float(target_mm) + .15)
    if before >= fine_range:
        report.update(status='NOT_NEEDED', reason='INITIAL_CLEARANCE_IN_FINE_RANGE')
        return mesh.copy(), report
    # shortcut: a target+0.15 mm residual is handed to the existing fine solver; revisit after more real cases.
    for scale in (.95, .90, .85, .80, .75):
        candidate, gate = apply_axial_placement(mesh, margin, axis,
            {'feasible': True, 'mechanism': 'AXIAL_SCALE', 'axial_scale': scale, 'anchor_mm': anchor})
        after, valid_ratio = clearance(candidate) if candidate is not None else (None, 0.0)
        report['trials'].append({'z_scale': scale, 'gate_ok': candidate is not None,
                                 'min_clearance_mm': after, 'valid_ratio': valid_ratio})
        if candidate is not None and after is not None and valid_ratio >= .8 and after >= fine_range:
            report.update(status='APPLIED', z_scale=scale, after_min_clearance_mm=after,
                          crown_top_before_mm=gate['crown_top_before_mm'],
                          crown_top_after_mm=gate['crown_top_after_mm'],
                          margin_displacement_mm=gate['cervical_band_displacement_max_mm'])
            return candidate, report
    report['reason'] = 'NO_GATE_PASSING_Z_SCALE_IN_FINE_RANGE'
    return None, report
