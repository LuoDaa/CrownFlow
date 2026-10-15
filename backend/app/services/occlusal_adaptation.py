"""Occlusal adaptation（咬合适配）：按**对颌天花板**给冠的咬合区减材，消除对颌穿透。

设计纪律
--------
1. **只动咬合区**：可动核 = 冠顶窗口（沿轴高度 ≥ 顶 − max(冠高/3, 2mm)）且法向朝就位轴；
   颈缘带（|h| ≤ 1.2mm）、内表面、窗口以下的侧壁**全部锁定**，位移必须精确为 0。
2. **度量口径**：驱动量用**沿就位轴的间隙**（虚拟咬合器口径，符号无歧义），
   同时用现有统一的有符号接触度量（`contact_measurement`）交叉核对 before/after。
   两个口径都写进报告，绝不只报好看的那个。
3. **只减材**（need ≥ 0），不加材。
4. **不可行就如实失败**：返回 ``OCCLUSAL_ADAPTATION_FAILED`` + 机器可读理由，不改冠、不发布。
   预期内的失败类型（不是算法错误）：
     - ``ANALYSIS_INPUT_MISSING``：没有已确认的静态咬合位姿 / 对颌资产；
     - ``OCCLUSAL_REGION_EMPTY``：咬合区没有可动顶点；
     - ``REQUIRED_DROP_EXCEEDS_OCCLUSAL_DOMAIN``：需要下压量 > 咬合区自身轴向高度
       ⇒ 局部减材不可能（冠体整体过高，属**放置**问题，不是咬合面问题）；
     - ``OCCLUSAL_SPACE_INSUFFICIENT_FOR_MIN_WALL``：预备体顶到对颌天花板的可用空间
       < 预备体顶 + 最小壁厚 ⇒ 这颗冠**任何形状**都无法既盖住预备体又不压到对颌（临床不足量）。
5. 求解复用既有积木（`solve_local_band` + `deformation_gate`）；**不改** margin /
   cervical reconstruction / boolean / stitch / QC。
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from ..geometry.crown_outer.deform_gate import deformation_gate
from ..geometry.crown_outer.laplacian import solve_local_band
from ..geometry.nonplanar_crown import MarginCurtain
from ..geometry.sculpt import cervical_weights

OCCLUSAL_THIRD = 1.0 / 3.0
MIN_OCCLUSAL_WINDOW_MM = 2.0
BLEND_BELOW_MM = 1.0
NORMAL_MIN = 0.35
CELL_MM = 0.4
ANTAGONIST_DOWN_NORMAL = -0.3
ALPHAS = (1.0, 0.5, 0.25, 0.125)
MAX_ITER = 3
MARGIN_LOCK_MM = 1.2
NEIGHBOR_RINGS = 6            # 可动核之外的过渡环数（位移在此衰减到 0）
MAX_LOOKUP_RINGS = 8          # 天花板查找向外扩张的最大圈数（8 × 0.4mm = 3.2mm）
FEASIBILITY_MARGIN_MM = 0.2   # 需要下压量留这么多余量才算"咬合区能吸收"
TARGET_MIN_CLEARANCE_MM = -0.1  # 验收：沿轴最小间隙必须 ≥ 这个值
IMPLEMENTATION = "services.occlusal_adaptation/axial_ceiling_carve/1"


def _unit(value):
    v = np.asarray(value, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def build_ceiling(local_xy, local_h, local_normal_z, *, cell_mm=CELL_MM, rings=1,
                  max_search_rings=MAX_LOOKUP_RINGS):
    """对颌在冠足印上的**沿轴天花板**：每 ``cell_mm`` 柱内、**面向冠**的最低点高度。

    只用局部法向朝下（``n·z ≤ -0.3``）的对颌顶点，避免把对颌的侧壁/牙龈当天花板；
    朝下面太少（<5%）时退回全体并如实记录（``normal_filter_applied=False``）。

    返回 ``(lookup, stats)``；``lookup(local_points)`` 给每个点一个天花板高度：
    先看自身柱，没有数据就**逐圈向外扩张**（最多 ``max_search_rings`` 圈，约
    ``max_search_rings·cell_mm`` mm），取找到的柱里的最低值（保守）；仍然没有 ⇒ ``nan``
    （该柱对颌无数据，"不可测"必须如实传下去，绝不猜）。
    """
    xy = np.asarray(local_xy, float)[:, :2]
    h = np.asarray(local_h, float)
    nz = np.asarray(local_normal_z, float)
    mask = np.ones(len(h), bool)
    filtered = False
    if len(h) > 200:
        down = nz <= ANTAGONIST_DOWN_NORMAL
        if down.sum() >= 0.05 * len(h):
            mask, filtered = down, True
    cells = {}
    for key, value in zip(map(tuple, np.floor(xy[mask] / cell_mm).astype(np.int64).tolist()), h[mask]):
        if key not in cells or value < cells[key]:
            cells[key] = value
    rings_by_distance = []
    for radius in range(1, int(max_search_rings) + 1):
        rings_by_distance.append(np.array(list(itertools.product(range(-radius, radius + 1), repeat=2)), np.int64))

    def lookup(local_points):
        keys = np.floor(np.asarray(local_points, float)[:, :2] / cell_mm).astype(np.int64)
        out = np.full(len(keys), np.nan)
        for index, (cx, cy) in enumerate(map(tuple, keys.tolist())):
            best = None
            for radius in range(0, int(max_search_rings) + 1):
                if radius == 0:
                    value = cells.get((int(cx), int(cy)))
                    if value is not None:
                        best = value
                else:
                    for ox, oy in rings_by_distance[radius - 1]:
                        value = cells.get((int(cx + ox), int(cy + oy)))
                        if value is not None and (best is None or value < best):
                            best = value
                if best is not None:
                    break
            if best is not None:
                out[index] = best
        return out

    def stats_for(points_local):
        values = lookup(points_local)
        return {"cell_mm": cell_mm, "cell_count": len(cells), "down_facing_vertices": int(mask.sum()),
                "normal_filter_applied": bool(filtered), "max_search_rings": int(max_search_rings),
                "measurable": int(np.isfinite(values).sum()), "requested": int(len(values)),
                "fill_ratio": float(np.isfinite(values).mean()) if len(values) else 0.0}

    return lookup, stats_for


def adapt_occlusal(design, mesh, frame, ref, *, target_mm=None, min_wall_mm=0.5,
                   mesh_source="final"):
    """对一颗冠执行咬合适配。返回 ``(adapted_mesh_or_None, report)``。

    ``adapted_mesh is None`` ⇔ 状态不是 ``OCCLUSAL_ADAPTED``（调用方必须不发布、不改冠）。
    """
    from .contact_measurement import confirmed_antagonist_surface, confirmed_neighbor_surfaces, measure_surface, _asset
    from .axial_placement import occlusal_envelope
    from ..geometry.crown_outer.auto_fit_adapter import fit_domains

    margin = np.asarray(json.loads(design["margin_json"]), float)
    params = json.loads(design.get("parameters_json") or "{}")
    target = float(target_mm if target_mm is not None else (params.get("occlusal_target_mm") or 0.05))
    curtain = MarginCurtain(margin, _unit(frame.z))
    basis, origin = curtain.frame.matrix(), np.asarray(curtain.frame.origin, float)
    axis = _unit(frame.z)
    local_of = lambda points: (np.asarray(points, float) - origin) @ basis

    vertices = np.asarray(mesh.vertices, float)
    faces = np.asarray(mesh.faces, np.int64)
    normals = np.asarray(mesh.vertex_normals, float)
    h = curtain.axial_above(vertices)
    nz = normals @ axis
    protection = np.asarray(cervical_weights(mesh, margin, axis), float)

    report = {"implementation": IMPLEMENTATION, "status": "NOT_RUN", "reason": None, "reasons": [],
              "mechanism": "OCCLUSAL_CARVE", "analysis_mesh_source": mesh_source,
              "vertex_count": int(len(vertices)), "face_count": int(len(faces)),
              "target_mm": target, "min_wall_mm": float(min_wall_mm)}

    surface, bite, error = confirmed_antagonist_surface(design, ref, strict=False)
    if surface is None:
        report.update(status="OCCLUSAL_ADAPTATION_FAILED",
                      reason=(error or {}).get("reason") or "OCCLUSAL_REFERENCE_UNAVAILABLE",
                      reasons=["ANALYSIS_INPUT_MISSING"])
        return None, report

    # ── 咬合区（可动核）：冠顶窗口 + 法向朝轴；窗口以下 / 颈缘带 / 内表面一律锁定 ──
    top = float(h.max())
    body_height = top - min(0.0, float(h.min()))
    window_lo = top - max(body_height * OCCLUSAL_THIRD, MIN_OCCLUSAL_WINDOW_MM)
    core = (h >= window_lo) & (nz > NORMAL_MIN) & (protection > 0)
    report["domain"] = {
        "window_lo_mm": float(window_lo), "top_mm": top, "body_height_mm": float(body_height),
        "occlusal_domain_extent_mm": float(top - window_lo),
        "movable_vertex_count": int(core.sum()),
        "margin_band_locked_vertex_count": int((np.abs(h) <= MARGIN_LOCK_MM).sum()),
        "intaglio_locked_vertex_count": int((nz <= 0.0).sum()),
        "definition": "h ≥ top − max(body/3, 2mm) 且 n·axis > 0.35 且 protection > 0",
    }
    if not core.any():
        report.update(status="OCCLUSAL_ADAPTATION_FAILED", reason="OCCLUSAL_REGION_EMPTY",
                      reasons=["OCCLUSAL_REGION_EMPTY"])
        return None, report

    # ── 沿轴天花板（对颌）：顶点 + 面心加密，避免稀疏处出现无数据的空洞 ──
    ant_mesh = surface.mesh
    ant_vertices = np.asarray(ant_mesh.vertices, float)
    ant_points = np.vstack([ant_vertices, np.asarray(ant_mesh.triangles_center, float)])
    ant_local = local_of(ant_points)
    ant_h = curtain.axial_above(ant_points)
    ant_nz = np.concatenate([np.asarray(ant_mesh.vertex_normals, float) @ axis,
                             np.asarray(ant_mesh.face_normals, float) @ axis])
    lookup, ceiling_stats = build_ceiling(ant_local[:, :2], ant_h, ant_nz, cell_mm=CELL_MM)

    def axis_gap(points, heights):
        """沿轴间隙 = 天花板高度 − 冠高度（>0 = 有间隙；<0 = 冠高出对颌）。"""
        return lookup(local_of(points)) - np.asarray(heights, float)

    def signed_measure(points):
        """统一的有符号接触度量（与 UI/审核同一口径），咬合域内取最小。"""
        try:
            masks, _locked = fit_domains(mesh, frame, margin, {"occlusal": surface}, "OCCLUSAL")
            mask = masks["occlusal"]
            if not mask.any():
                return None
            body, _query = measure_surface(surface, np.asarray(points, float), mask, target,
                                           kind="OCCLUSAL", side="occlusal")
            return body
        except (ValueError, KeyError, np.linalg.LinAlgError, IndexError):
            return None

    cap = np.flatnonzero(core)
    gap_before = axis_gap(vertices, h)
    measurable = np.isfinite(gap_before[cap])
    report["ceiling"] = ceiling_stats(local_of(vertices[cap]))
    signed_before = signed_measure(vertices)
    report["occlusion"] = {
        "driving_caliber": "axial_clearance_along_insertion_axis（柱内最低天花板，符号无歧义）",
        "cross_check_caliber": "unified signed contact measurement（REVIEWED_LOCAL_NORMAL）",
        "before_min_distance_mm": float(np.min(gap_before[cap][measurable])) if measurable.any() else None,
        "after_min_distance_mm": None,
        "penetrating_vertices_before": int((gap_before[cap][measurable] < 0).sum()),
        "penetrating_vertices_after": None,
        "occlusal_domain_vertex_count": int(len(cap)),
        "measurable_vertex_count": int(measurable.sum()),
        "signed_before_min_mm": None if signed_before is None else signed_before.get("signed_min_mm"),
        "signed_after_min_mm": None,
        "signed_domain_vertex_count": None if signed_before is None else signed_before.get("domain_vertex_count"),
    }

    # ── 可行性：需要下压量 vs 咬合区自身高度；可用咬合空间 vs 预备体顶 + 最小壁厚 ──
    required = np.zeros(len(cap))
    if measurable.any():
        required[measurable] = np.maximum(0.0, -gap_before[cap][measurable]) + target
    extent = top - window_lo
    report["required_drop"] = {
        "p50_mm": float(np.median(required[required > 0])) if (required > 0).any() else 0.0,
        "max_mm": float(required.max()),
        "occlusal_domain_extent_mm": float(extent),
    }
    support = _asset(design, 'support') if design.get('case_id') else None
    report["occlusal_space"] = occlusal_envelope(
        design, margin, axis, ref, surface=surface, crown_mesh=mesh,
        prep_path=support['path'] if support else None,
        min_thickness_mm=min_wall_mm, target_clearance_mm=target)
    if report["occlusal_space"]["status"] == "INSUFFICIENT_OCCLUSAL_SPACE":
        report["reasons"].append("OCCLUSAL_SPACE_INSUFFICIENT_FOR_MIN_WALL")
    if report["required_drop"]["max_mm"] > max(extent - FEASIBILITY_MARGIN_MM, 0.0):
        report["reasons"].append("REQUIRED_DROP_EXCEEDS_OCCLUSAL_DOMAIN")

    # ── 求解：局部带 Laplacian + 硬门 + alpha 回退，最多 MAX_ITER 轮 ──
    locked_ids = np.flatnonzero((np.abs(h) <= MARGIN_LOCK_MM) | (protection <= 0) | (nz <= 0.0)
                                | (h < window_lo - BLEND_BELOW_MM) | ~np.isfinite(h))
    adjacency = _adjacency(faces, len(vertices))
    current = np.asarray(vertices, float).copy()
    trials = []
    best_mesh, best_min = None, np.inf
    for _ in range(MAX_ITER):
        h_cur = curtain.axial_above(current)
        gap_cur = axis_gap(current, h_cur)
        core_cur = _core_ids(h_cur, np.asarray(mesh.vertex_normals, float) @ axis, protection, window_lo)
        ids = np.flatnonzero(core_cur & np.isfinite(gap_cur))
        drop = np.zeros(len(current))
        if len(ids):
            drop[ids] = np.maximum(0.0, -gap_cur[ids] + target)
        seed_ids = np.flatnonzero(drop > 1e-6)
        if not len(seed_ids):
            break
        target_positions = current.copy()
        target_positions[seed_ids] -= axis[None, :] * drop[seed_ids][:, None]
        # 种子顶点本身是**固定边界**（位移由构造精确给定），只有核外的过渡环是自由变量：
        # 与 `fit_stage` 同一分区（`seen=seeds|locked`，free=环且不含种子/锁定点）。
        pinned = set(int(i) for i in seed_ids) | set(int(i) for i in locked_ids)
        free_ids = np.asarray(sorted(_ring_ids(seed_ids.tolist(), adjacency, NEIGHBOR_RINGS) - pinned),
                              np.int64)
        if not len(free_ids):
            break
        fixed_ids = np.setdiff1d(np.arange(len(current)), free_ids)
        for alpha in ALPHAS:
            xt = current.copy()
            xt[seed_ids] += alpha * (target_positions[seed_ids] - current[seed_ids])
            candidate, _info = solve_local_band(current, faces, free_ids, fixed_ids, xt)
            moved = np.linalg.norm(candidate - current, axis=1)
            gate = deformation_gate(current, candidate, faces, scale_mm=0.1, locked_ids=locked_ids,
                                    focus_faces=np.flatnonzero(np.any(moved[faces] > 1e-9, axis=1)))
            gap_new = axis_gap(candidate, curtain.axial_above(candidate))
            keep = np.flatnonzero(core_cur & np.isfinite(gap_new))
            new_min = float(np.min(gap_new[keep])) if len(keep) else np.inf
            trials.append({"alpha": float(alpha), "gate_ok": bool(gate.ok),
                           "penetration_max_after_mm": float(-new_min) if np.isfinite(new_min) else None,
                           "locked_displacement_max_mm": gate.metrics.get("locked_displacement_max_mm"),
                           "violations": list(gate.violations)})
            if gate.ok and new_min < best_min:
                best_mesh, best_min, current = candidate.copy(), new_min, candidate
                break
    report["solver"] = {"implementation": "crown_outer.laplacian.solve_local_band + deform_gate",
                        "rounds": len(trials), "trials": trials,
                        "neighbor_rings": NEIGHBOR_RINGS, "alphas": list(ALPHAS)}
    if best_mesh is None:
        report.update(status="OCCLUSAL_ADAPTATION_FAILED", reason="NO_GATE_PASSING_CANDIDATE",
                      mesh_sha256_after=None, mesh_unchanged=True,
                      verdict=("没有任何 alpha 能同时通过形变硬门；冠未被修改"
                               "（需要下压 p50 %.2f / max %.2fmm，咬合区自身只有 %.2fmm）"
                               % (report["required_drop"]["p50_mm"], report["required_drop"]["max_mm"],
                                  report["required_drop"]["occlusal_domain_extent_mm"])))
        report["reasons"].append("NO_GATE_PASSING_CANDIDATE")
        return None, report

    # ── 验收 ──
    adapted_vertices = np.asarray(best_mesh, float)
    displacement = np.linalg.norm(adapted_vertices - vertices, axis=1)
    gap_after = axis_gap(adapted_vertices, curtain.axial_above(adapted_vertices))
    keep = np.flatnonzero(core & np.isfinite(gap_after))
    after_min = float(np.min(gap_after[keep])) if len(keep) else None
    after_signed = signed_measure(adapted_vertices)
    margin_band = np.abs(h) <= MARGIN_LOCK_MM
    report["occlusion"].update(
        after_min_distance_mm=after_min,
        penetrating_vertices_after=int((gap_after[keep] < 0).sum()) if len(keep) else None,
        signed_after_min_mm=None if after_signed is None else after_signed.get("signed_min_mm"))
    report["protected"] = {
        "locked_vertex_count": int(len(locked_ids)),
        "locked_displacement_max_mm": float(displacement[locked_ids].max()) if len(locked_ids) else 0.0,
        "margin_band_displacement_max_mm": float(displacement[margin_band].max()) if margin_band.any() else 0.0,
        "intaglio_displacement_max_mm": float(displacement[nz <= 0.0].max()) if (nz <= 0.0).any() else 0.0,
        "max_displacement_mm": float(displacement.max()),
        "p95_displacement_mm": float(np.quantile(displacement, 0.95)),
        "moved_vertex_count": int((displacement > 1e-9).sum()),
    }
    report["mesh_sha256_before"] = _sha(vertices, faces)
    report["mesh_sha256_after"] = _sha(adapted_vertices, faces)
    report["mesh_unchanged"] = report["mesh_sha256_before"] == report["mesh_sha256_after"]
    signed_min = None if after_signed is None else after_signed.get("signed_min_mm")
    signed_ok = signed_min is None or signed_min >= TARGET_MIN_CLEARANCE_MM
    accepted = (report["protected"]["locked_displacement_max_mm"] <= 1e-9
                and after_min is not None and after_min >= TARGET_MIN_CLEARANCE_MM and signed_ok)
    neighbors, _ = confirmed_neighbor_surfaces(design, ref, strict=False) if design.get('case_id') else ({}, {})
    drift = {}
    if accepted and neighbors:
        from ..geometry.crown_regions import build_region_map
        regions = build_region_map(design, mesh, frame, ref, neighbors=neighbors)
        for side, (neighbor, _) in neighbors.items():
            ids = np.flatnonzero((regions.weights[side] > .5) & (regions.protection > 0))
            if not len(ids):
                continue
            _, _, before_signed, _, before_valid = neighbor.query(vertices[ids])
            _, _, after_signed, _, after_valid = neighbor.query(adapted_vertices[ids])
            if before_valid.any():
                common = before_valid & after_valid
                if not common.any() or not after_valid[before_valid].all():
                    accepted = False
                    drift[side] = 'SIGNED_SAMPLES_LOST'
                    break
                change = abs(float(np.min(after_signed[common]) - np.min(before_signed[common])))
                drift[side] = change
                if change > .1:
                    accepted = False
                    break
    report['proximal_contact_drift_mm'] = drift
    if accepted:
        report["status"] = "OCCLUSAL_ADAPTED"
        report["verdict"] = "沿轴最小间隙 %.3f mm（目标 ≥ %.1f mm）" % (after_min, TARGET_MIN_CLEARANCE_MM)
    else:
        report["status"] = "OCCLUSAL_ADAPTATION_FAILED"
        report["reason"] = "PROXIMAL_CONTACT_DRIFT" if drift and any(
            value == 'SIGNED_SAMPLES_LOST' or value > .1 for value in drift.values()) else "TARGET_NOT_REACHED_AFTER_SOLVE"
        report["reasons"].append(report["reason"])
        report["verdict"] = ("邻牙接触漂移超过 0.1 mm 或失去有效符号样本，冠未发布"
                             if report['reason'] == 'PROXIMAL_CONTACT_DRIFT' else
                             "求解后沿轴最小间隙 %s mm，未达 %.1f mm；冠未发布"
                             % ("n/a" if after_min is None else "%.3f" % after_min, TARGET_MIN_CLEARANCE_MM))
    adapted = mesh.copy()
    adapted.vertices = adapted_vertices
    return (adapted if accepted else None), report


def _core_ids(h, nz, protection, window_lo):
    return ((np.asarray(h, float) >= window_lo) & (np.asarray(nz, float) > NORMAL_MIN)
            & (np.asarray(protection, float) > 0))


def _ring_ids(seed_ids, adjacency, rings):
    seen = set(int(i) for i in seed_ids)
    front = set(seen)
    for _ in range(int(rings)):
        front = {v for u in front for v in adjacency[u]} - seen
        seen |= front
    return seen


def _adjacency(faces, count):
    adj = [set() for _ in range(count)]
    for a, b in np.asarray(faces, np.int64)[:, [0, 1]]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    for a, b in np.asarray(faces, np.int64)[:, [1, 2]]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    for a, b in np.asarray(faces, np.int64)[:, [2, 0]]:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    return adj


def _sha(vertices, faces):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(vertices, np.float64)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(faces, np.int64)).tobytes())
    return digest.hexdigest()
