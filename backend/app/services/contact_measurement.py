"""唯一 ContactMeasurement（B 阶段）：四条路径共用同一**有符号**接触口径。

**为什么需要它**（审计 #5）：放置/AutoFit/Final QC 之间已经是有符号的
（`crown_outer.contact.NeighborPatch.clearance_detail` + `auto_fit_adapter.FitSurface` 的实体内外判定），
但 Candidate Review（`api/auto_fit.candidate_analysis`）与手工工作台（`api/designer.outer_analysis`）
仍只有 `trimesh.proximity.closest_point` 的**无符号**距离，报告里那句
"无符号三角面距离，不证明接触或穿透"就是断口 —— 同一份几何在不同页面给出互相矛盾的结论。

**口径**（沿用 `contact.py` 2026-10-09 决策，不新造；`clearance_detail` 仍是唯一幅值/有效性来源）::

    signed > 0   远离参考面（安全间隙）
    signed < 0   穿进参考面（穿透）
    |signed|     = 到参考面的欧氏距离（连续，热力图用它）
    penetration  = max(0, -signed)          正值 = 穿透深度
    deficit      = max(0, target - signed)   距目标间隙还差多少
    target_error = |signed - target|

符号来源只有两种，一律如实标注，不冒充全局内外：

- ``GLOBAL_INSIDE_OUTSIDE``：参考体是闭合有向实体 → 用实体内外判定（可信）；
- ``REVIEWED_LOCAL_NORMAL``：参考面是开放区域（扫描裁片等）→ 用人工确认的外向法向，
  并把歧义面（边界/折痕）剔出决策集。

**纪律**：任何新页面要显示"接触/穿透"，都必须调本模块，不允许再直接 `closest_point`。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KIND_PROXIMAL = "PROXIMAL"
KIND_OCCLUSAL = "OCCLUSAL"

SIGN_GLOBAL = "GLOBAL_INSIDE_OUTSIDE"
SIGN_REVIEWED_LOCAL = "REVIEWED_LOCAL_NORMAL"
#: 参考面连外向法向都没有（例如旧的球面 ROI 邻牙选区）⇒ 只能给幅值，**明确拒绝**穿透结论
SIGN_UNSIGNED_ONLY = "UNSIGNED_ONLY"

STATUS_MEASURED = "MEASURED"
STATUS_NOT_ASSESSED = "NOT_ASSESSED"

IMPLEMENTATION = "crown_outer.auto_fit_adapter.FitSurface(query)+contact.clearance_detail"
SIGN_SEMANTICS = "positive=gap_to_reference, negative=penetration_into_reference, |signed|=euclidean_distance"


def sign_method(surface) -> str:
    """符号是怎么来的（闭实体 vs 人工确认外向法向），报告里必须写清楚。"""
    return SIGN_GLOBAL if getattr(surface, "solid", False) else SIGN_REVIEWED_LOCAL


def _stats(values) -> dict | None:
    """p50/p95/max —— 与既有 fit 报告形状保持一致（不许改字段名，求解器在消费）。"""
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {"p50_mm": float(np.quantile(values, .5)),
            "p95_mm": float(np.quantile(values, .95)),
            "max_mm": float(values.max())}


def report_from_query(query, target_mm, *, kind, side=None, reference=None, crown=None, domain=None):
    """把一次 surface.query() 变成**唯一**接触报告（四条路径共用这一个形状）。"""
    ids, cp, fids, signed, distance, valid = query
    signed = np.asarray(signed, float)
    distance = np.asarray(distance, float)
    valid = np.asarray(valid, bool)
    target = float(target_mm or 0.0)
    deficit = np.maximum(0, target - signed)
    penalty = np.abs(signed - target)
    report = {
        "status": STATUS_MEASURED if valid.any() else STATUS_NOT_ASSESSED,
        "kind": kind, "side": side,
        "valid_count": int(valid.sum()), "ambiguous_count": int((~valid).sum()),
        "unsigned": _stats(distance),
        "signed_valid": _stats(signed[valid]),
        "penetration": _stats(np.maximum(0, -signed[valid])),
        "deficit": _stats(deficit[valid]), "target_error": _stats(penalty[valid]),
        "effective_target_gap_mm": target, "target_mm": target,
        "sign_method": sign_method(reference) if reference is not None else None,
        "sign_semantics": SIGN_SEMANTICS, "implementation": IMPLEMENTATION}
    if len(signed[valid]):
        report["signed"] = {"min_mm": float(signed[valid].min()), **_stats(signed[valid])}
        report["signed_min_mm"] = float(signed[valid].min())
        report["penetration_max_mm"] = float(max(0.0, -signed[valid].min()))
    if reference is not None:
        report["reference_sha256"] = getattr(reference, "sha256", None)
    if crown is not None:
        report["crown_sha256"] = crown
    if domain is not None:
        report["domain_vertex_count"] = int(np.count_nonzero(domain))
    return report


def measure_surface(surface, vertices, mask, target_mm, *, kind, side=None, crown=None, domain=None):
    """掩码内顶点 → (报告, 原始查询元组)。空掩码返回 ``(NOT_ASSESSED 报告, None)``。

    查询元组固定为 ``(ids, cp, fids, signed, distance, valid)``（求解器在消费这个形状）。
    """
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {"status": STATUS_NOT_ASSESSED, "kind": kind, "side": side,
                "valid_count": 0, "ambiguous_count": 0, "implementation": IMPLEMENTATION,
                "sign_semantics": SIGN_SEMANTICS}, None
    cp, fids, signed, distance, valid = surface.query(np.asarray(vertices)[ids])
    query = (ids, cp, fids, signed, distance, valid)
    report = report_from_query(query, target_mm, kind=kind, side=side, reference=surface,
                               crown=crown, domain=mask)
    return report, query


# ---------------------------------------------------------------- 参考面来源（已确认输入）
def _asset(design, kind):
    from ..db import db
    with db() as conn:
        row = conn.execute("SELECT * FROM assets WHERE case_id=? AND kind=? "
                           "ORDER BY created_at DESC LIMIT 1", (design["case_id"], kind)).fetchone()
    return dict(row) if row else None


def confirmed_neighbor_surfaces(design, frame_ref, *, strict=True):
    """已确认的邻牙面（逐条校验来源与区域字节），返回 ``side → (FitSurface, entry)``。

    ⚠️ 与 `services/auto_fit.propose` / `audit_final_context` 用同样的校验：来源资产未变、
    裁片字节未变、帧版本一致。``strict=True``（审核页面）不一致就 409；
    ``strict=False``（放置快照）只把该侧标成 NOT_ASSESSED，绝不"用别的邻牙凑一个数"。
    """
    from fastapi import HTTPException
    from .design_dependencies import file_digest
    from ..geometry.io import load_mesh
    from ..geometry.crown_outer.auto_fit_adapter import FitSurface
    choices = json.loads(design.get("neighbors_json") or "{}")
    surfaces, skipped = {}, {}
    for side in ("mesial", "distal"):
        entry = choices.get(side, {})
        if entry.get("confirmation_status") == "ABSENT":
            skipped[side] = {"status": "SKIPPED_ABSENT"}
            continue
        unverified = (entry.get("confirmation_status") != "CONFIRMED"
                      or entry.get("frame_revision_id") != frame_ref.get("revision_id"))
        if unverified:
            if strict:
                raise HTTPException(409, "邻牙（%s）尚未在当前坐标帧下确认，请先确认或标记缺牙" % side)
            skipped[side] = {"status": STATUS_NOT_ASSESSED, "reason": "NEIGHBOR_UNVERIFIED"}
            continue
        source = entry.get("source", {})
        current = _asset(design, source.get("kind"))
        path = entry.get("path")
        if (not current or current["id"] != source.get("asset_id") or file_digest(current["path"]) != source.get("sha256")
                or not path or not Path(path).is_file() or file_digest(path) != entry.get("region_sha256")):
            if strict:
                raise HTTPException(409, "邻牙（%s）来源或区域已变化，请重新确认" % side)
            skipped[side] = {"status": STATUS_NOT_ASSESSED, "reason": "NEIGHBOR_SOURCE_CHANGED"}
            continue
        surfaces[side] = (FitSurface(load_mesh(Path(path)), entry.get("outward_normal")), entry)
    return surfaces, skipped


def neighbor_patch_diagnostics(mesh, region_map, neighbors):
    """NeighborPatchValidity: source confirmation alone does not make a contact patch usable."""
    vertices = np.asarray(mesh.vertices, float)
    result = {}
    for side, (surface, entry) in neighbors.items():
        patch = surface.mesh
        centroid = np.average(patch.triangles_center, axis=0, weights=patch.area_faces)
        direction = vertices.mean(axis=0) - centroid
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        outward = surface.outward
        if surface.solid:
            _, closest_faces, _, _, _ = surface.query(vertices.mean(axis=0)[None, :])
            normal = np.asarray(patch.face_normals[int(closest_faces[0])], float)
        else:
            normal = np.asarray(outward if outward is not None else np.average(
                patch.face_normals, axis=0, weights=patch.area_faces), float)
        normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
        mask = (np.asarray(region_map.weights[side]) > .5) & (np.asarray(region_map.protection) > 0)
        _, _, signed, _, valid = surface.query(vertices[mask]) if mask.any() else (
            None, None, np.empty(0), np.empty(0), np.zeros(0, bool))
        _, _, _, all_distance, _ = surface.query(vertices)
        axis = np.asarray(region_map.axes['occlusal'], float)
        md = np.asarray(region_map.axes['mesial'], float)
        crown_axial = vertices @ axis
        patch_axial = np.asarray(patch.vertices) @ axis
        overlap = max(0.0, min(float(crown_axial.max()), float(patch_axial.max()))
                      - max(float(crown_axial.min()), float(patch_axial.min())))
        side_offset = float((centroid - vertices.mean(axis=0)) @ md)
        expected_side = (side_offset > 0 if side == 'mesial' else side_offset < 0)
        reasons = []
        if not expected_side: reasons.append('WRONG_SIDE')
        if len(patch.faces) < 3: reasons.append('PATCH_TOO_SMALL')
        if float(normal @ direction) < .25: reasons.append('NOT_FACING_CROWN')
        if overlap <= .25: reasons.append('NO_AXIAL_OVERLAP')
        if not np.isfinite(all_distance).all(): reasons.append('NONFINITE_DISTANCE')
        if not np.any(valid): reasons.append('NO_VALID_SIGNED_SAMPLES')
        result[side] = {
            'side': side, 'expected_side': bool(expected_side), 'patch_faces': int(len(patch.faces)),
            'patch_centroid': centroid.tolist(),
            'distance_to_current_crown': float(np.min(all_distance)),
            'facing_score': float(normal @ direction),
            'axial_overlap_mm': overlap,
            'region_vertex_count': int(mask.sum()),
            'valid_signed_samples': int(np.asarray(valid, bool).sum()),
            'signed_min_mm': float(np.min(np.asarray(signed)[valid])) if np.any(valid) else None,
            'source_confirmed': entry.get('confirmation_status') == 'CONFIRMED',
            'pollution_reviewed': bool(entry.get('pollution_reviewed')),
            'status': 'VALID' if not reasons else 'INVALID', 'reasons': reasons,
        }
    return result


def confirmed_antagonist_surface(design, frame_ref, *, strict=True):
    """已确认静态咬合位姿下的对颌参考面（逐条校验扫描字节）。"""
    from fastapi import HTTPException
    from .design_dependencies import file_digest
    from ..geometry.io import load_mesh
    from ..geometry.crown_outer.auto_fit_adapter import FitSurface
    bite = json.loads(design.get("bite_json") or "{}")
    if not (bite.get("pose_confirmed") and bite.get("revision_id") and bite.get("transform")
            and bite.get("frame_revision_id") == frame_ref.get("revision_id")):
        if strict:
            raise HTTPException(409, "静态咬合位姿尚未在当前坐标帧下确认")
        return None, bite, {"status": STATUS_NOT_ASSESSED, "reason": "OCCLUSION_REGISTRATION_UNVERIFIED"}
    for scan in bite.get("scans", []):
        current = _asset(design, scan.get("kind"))
        if not current or current["id"] != scan.get("asset_id") or file_digest(current["path"]) != scan.get("sha256"):
            if strict:
                raise HTTPException(409, "咬合来源扫描已变化，请重新确认")
            return None, bite, {"status": STATUS_NOT_ASSESSED, "reason": "OCCLUSION_SOURCE_CHANGED"}
    # 对颌 = **对侧颌**：目标牙在 31–48（下颌）⇒ 对颌取上颌扫描，反之取下颌。
    # 口径与 `services/auto_fit.py`、`designer._legacy_reference`、`confirm_neighbor`（工作颌）一致；
    # 之前这里写反了（取成了工作颌本身），实测表现为"对颌穿透与真实对颌的 z 位移无关"。
    kind = "upper" if int(design["target_fdi"]) >= 31 else "lower"
    asset = _asset(design, kind)
    if not asset:
        if strict:
            raise HTTPException(409, "缺少对颌扫描")
        return None, bite, {"status": STATUS_NOT_ASSESSED, "reason": "ANTAGONIST_ASSET_MISSING"}
    ant = load_mesh(Path(asset["path"]))
    ant.apply_transform(np.asarray(bite["transform"]))
    frame_z = np.asarray(json.loads(design["axis_json"]), float)
    frame_z /= np.linalg.norm(frame_z)
    return FitSurface(ant, (-frame_z).tolist()), bite, None


def thickness_protection_surfaces(design):
    """Reuse confirmed references; stale declared contacts must not lose protection."""
    from fastapi import HTTPException
    if not design.get('autofit_frame_json'):return []
    from ..api.auto_fit import frame_from
    _,ref=frame_from(design)
    neighbors,skipped=confirmed_neighbor_surfaces(design,ref,strict=False)
    entries=json.loads(design.get('neighbors_json') or '{}')
    if any(entries.get(side,{}).get('confirmation_status')=='CONFIRMED' for side in skipped):
        raise HTTPException(409,'补厚保护的邻牙确认已过期，请重新确认')
    ant,bite,error=confirmed_antagonist_surface(design,ref,strict=False)
    if error and bite.get('pose_confirmed'):raise HTTPException(409,'补厚保护的咬合确认已过期，请重新确认')
    parameters=json.loads(design['parameters_json']);surfaces=[]
    if neighbors and parameters.get('proximal_target_mm') is None:
        raise HTTPException(409,'请确认邻接目标间隙后执行受约束补厚')
    surfaces.extend((surface,parameters['proximal_target_mm']) for surface,_ in neighbors.values())
    if ant:
        if parameters.get('occlusal_target_mm') is None:raise HTTPException(409,'请确认咬合目标间隙后执行受约束补厚')
        surfaces.append((ant,parameters['occlusal_target_mm']))
    return surfaces


def frame_from_design(design, frame_ref):
    """由已确认帧值构造 PatientFrame（与 `api.auto_fit.frame_from` 同一构造，不含校验）。"""
    from ..geometry.crown_outer.patient_frame import PatientFrame
    margin = np.asarray(json.loads(design["margin_json"]), float)
    return PatientFrame(margin.mean(0), np.asarray(frame_ref["z"], float),
                        np.asarray(frame_ref["x"], float), np.asarray(frame_ref["y"], float),
                        revision_id=frame_ref.get("revision_id"))


def per_vertex_contact(surface, vertices, mask=None):
    """逐顶点接触数组（热力图用）：``signed`` 有符号、``magnitude`` = |signed|。

    与 `distance_mm`（无符号距离）的差别是**符号**：负值代表穿进参考面。
    """
    vertices = np.asarray(vertices, float)
    ids = np.arange(len(vertices)) if mask is None else np.flatnonzero(mask)
    cp, fids, signed, distance, valid = surface.query(vertices[ids])
    signed = np.asarray(signed, float)
    valid = np.asarray(valid, bool)
    return {"ids": ids.tolist(), "signed_mm": signed.tolist(),
            "distance_mm": np.asarray(distance, float).tolist(),
            "valid": valid.tolist(), "ambiguous_count": int((~valid).sum()),
            "sign_method": sign_method(surface)}


def combine_per_vertex(items):
    """把多张参考面的逐顶点结果合成一份热力图数据。

    ``signed_mm`` 取**最不利**（最小）的有效有符号值；某顶点对所有参考面都无效时，
    退回幅值（正数）并置 ``valid=False`` —— 热力图需要有限数，但结论必须标成不可用。
    """
    keys = list(items)
    if not keys:
        return None
    size = len(items[keys[0]]["ids"])
    signed = np.full(size, np.nan)
    distance = np.full(size, np.inf)
    valid = np.zeros(size, bool)
    for key in keys:
        entry = items[key]
        s = np.asarray(entry["signed_mm"], float)
        d = np.asarray(entry["distance_mm"], float)
        v = np.asarray(entry["valid"], bool)
        distance = np.minimum(distance, d)
        better = v & (np.isnan(signed) | (s < signed))
        signed[better] = s[better]
        valid |= v
    magnitude = np.where(np.isfinite(distance), distance, np.nan)
    signed = np.where(np.isnan(signed), magnitude, signed)
    return {"signed_mm": signed.tolist(), "distance_mm": magnitude.tolist(), "valid": valid.tolist(),
            "ambiguous_count": int((~valid).sum()), "per_side": {key: items[key] for key in keys},
            "sign_method": sign_method_for_items(items), "combine_rule": "min_signed_over_available_references"}


def sign_method_for_items(items):
    methods = {value.get("sign_method") for value in items.values()}
    if methods == {SIGN_GLOBAL}:
        return SIGN_GLOBAL
    if methods == {SIGN_REVIEWED_LOCAL}:
        return SIGN_REVIEWED_LOCAL
    return "MIXED_SIGN_SOURCES"


def unsigned_reference_distance(vertices, reference_mesh, *, reason, chunk=1024):
    """**显式降级**：参考面无可信符号来源时只报幅值，并给出机器可读的拒绝理由。

    这不是"另一个口径"，而是唯一模块里对"符号不可得"的正式回答；
    调用方不得把它当成接触/穿透结论（``sign_method=UNSIGNED_ONLY``、``signed_mm=None``）。
    """
    from trimesh.proximity import closest_point
    vertices = np.asarray(vertices, float)
    values = np.concatenate([closest_point(reference_mesh, vertices[i:i + chunk])[1]
                             for i in range(0, len(vertices), chunk)])
    return {"signed_mm": None, "distance_mm": np.asarray(values, float).tolist(),
            "valid": [False] * len(vertices), "ambiguous_count": int(len(vertices)),
            "min_mm": float(values.min()), "p05_mm": float(np.quantile(values, .05)),
            "sign_method": SIGN_UNSIGNED_ONLY, "status": STATUS_NOT_ASSESSED, "reason": reason,
            "implementation": IMPLEMENTATION,
            "sign_semantics": "magnitude_only_no_penetration_claim"}


def measure_confirmed_inputs(design, frame_ref, crown_path, *, target_proximal=None, target_occlusal=None,
                             strict=True):
    """用**当前已确认输入**测一颗冠（放置 baseline 或适配候选）的接触：四条路径共用这一段。

    返回 ``{'proximal': {side: 报告}, 'occlusal': 报告|None, 'skipped': {...},
    'status': MEASURED|NOT_ASSESSED, 'surfaces': {...}, 'domains': {...}}``。
    ``surfaces``/``domains`` 是**进程内对象**（FitSurface / 布尔掩码），给同一次调用的热力图复用，
    不要序列化进 HTTP 响应。放置路径传 ``strict=False``（未确认输入只标未评估），审核路径传
    ``strict=True``（来源变了必须 409）。
    """
    from ..geometry.io import load_mesh
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains
    from .design_dependencies import file_digest

    if not (frame_ref or {}).get("z"):
        return {"proximal": {}, "occlusal": None, "skipped": {}, "status": STATUS_NOT_ASSESSED,
                "reason": "FRAME_UNCONFIRMED", "implementation": IMPLEMENTATION,
                "sign_semantics": SIGN_SEMANTICS}
    frame = frame_from_design(design, frame_ref)
    margin = np.asarray(json.loads(design["margin_json"]), float)
    mesh = load_mesh(Path(crown_path))
    crown_sha = file_digest(crown_path)
    result = {"proximal": {}, "occlusal": None, "skipped": {}, "status": STATUS_NOT_ASSESSED,
              "crown_sha256": crown_sha, "implementation": IMPLEMENTATION,
              "sign_semantics": SIGN_SEMANTICS, "frame_revision_id": frame_ref.get("revision_id"),
              "surfaces": {}, "domains": {},
              "scope": "confirmed_inputs_sampled_crown_not_continuous_certification"}

    # 只构造**这次要测的**参考面：邻牙分析不必先确认咬合，咬合分析也不必先确认邻牙。
    neighbors, skipped = (confirmed_neighbor_surfaces(design, frame_ref, strict=strict)
                          if target_proximal is not None else ({}, {}))
    result["skipped"].update(skipped)
    if neighbors and target_proximal is not None:
        surfaces = {side: surface for side, (surface, _) in neighbors.items()}
        domains = fit_domains(mesh, frame, margin, surfaces, KIND_PROXIMAL)[0]
        for side, (surface, entry) in neighbors.items():
            report, _ = measure_surface(surface, mesh.vertices, domains[side], target_proximal,
                                        kind=KIND_PROXIMAL, side=side, crown=crown_sha, domain=domains[side])
            report["neighbor_revision_id"] = entry.get("revision_id")
            result["proximal"][side] = report
        result["surfaces"].update(surfaces)
        result["domains"].update(domains)
    elif neighbors:
        for side in neighbors:
            result["skipped"][side] = {"status": STATUS_NOT_ASSESSED, "reason": "TARGET_UNCONFIRMED"}

    if target_occlusal is not None:
        surface, bite, reason = confirmed_antagonist_surface(design, frame_ref, strict=strict)
    else:
        surface, bite, reason = None, None, None
    if surface is not None and target_occlusal is not None:
        mask = fit_domains(mesh, frame, margin, {"occlusal": surface}, KIND_OCCLUSAL)[0]["occlusal"]
        report, _ = measure_surface(surface, mesh.vertices, mask, target_occlusal,
                                    kind=KIND_OCCLUSAL, side=None, crown=crown_sha, domain=mask)
        report["bite_revision_id"] = bite.get("revision_id")
        result["occlusal"] = report
        result["surfaces"]["occlusal"] = surface
        result["domains"]["occlusal"] = mask
    elif surface is not None:
        result["skipped"]["occlusal"] = {"status": STATUS_NOT_ASSESSED, "reason": "TARGET_UNCONFIRMED"}
    elif reason is not None:
        result["skipped"]["occlusal"] = reason

    measured = [value for value in list(result["proximal"].values()) + [result["occlusal"]] if value]
    result["status"] = STATUS_MEASURED if measured else STATUS_NOT_ASSESSED
    return result


def serializable_measurement(measurement):
    """去掉进程内对象（FitSurface / 掩码），得到可直接进 HTTP 响应/修订链的副本。"""
    return {key: value for key, value in measurement.items() if key not in ("surfaces", "domains")}
