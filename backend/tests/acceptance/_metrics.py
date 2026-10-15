"""FDI36 基准病例的几何度量。

这些度量的全部意义是：**用与真实病例真值的定量误差替代"STL 成功生成"作为完成标准。**

坐标约定（已核实）：该病例中咬合方向为 **+Z**。
- 基台 z 3.49–10.82，真值最终冠 z 5.63–12.50，对颌扫描 z 10.4–25.5
因此"正确的就位方向"= 该病例坐标系下的 +Z。这是由对颌几何位置独立佐证的，
不是假定。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from app.geometry.axis import suggest_insertion_axis
from app.geometry.boolean import crown_minus_cutter, boolean_backend_name
from app.geometry.io import load_mesh
from app.geometry.margin_seam import margin_plane_diagnostics
from app.geometry.outer import trim_outer_at_margin

#: 该病例坐标系中的咬合方向 / 正确就位方向。由对颌几何位置佐证。
OCCLUSAL_AXIS = np.array([0.0, 0.0, 1.0])


def load_fdi36(case_dir):
    """加载 FDI36 基准病例。"""
    from pathlib import Path

    case_dir = Path(case_dir)
    main_dir = case_dir / "01_主要病例模型"
    return {
        "case_dir": case_dir,
        "support": load_mesh(main_dir / "1629-王煜 0.stl"),
        "gt_crown": load_mesh(main_dir / "1629-王煜 1.stl"),
        "lower": load_mesh(main_dir / "Unsectioned_LowerJaw.stl"),
        "upper": load_mesh(main_dir / "Unsectioned_UpperJaw.stl"),
    }


def gt_cervical_seam(crown: trimesh.Trimesh, support: trimesh.Trimesh,
                     axis=None) -> tuple[np.ndarray, dict]:
    """从真值冠提取 intaglio ↔ exterior 的真实颈缘缝（**当前验收使用的 Margin 真值**）。

    见 :mod:`app.geometry.gt_margin`。实测（FDI36）：

        周长 21.76 mm   方位角覆盖 1.000   包围 support 截面 1.000
        spacing_cv 0.127   最佳拟合平面残差 mean 0.010 / max 0.043 mm

    对比旧的 ``cervical_margin_legacy``：方位角覆盖只有 0.667（16/24，残弧），
    spacing_cv 1.89。旧方法产出的**不是闭环**。

    返回 ``(points, diagnostics)``。
    """
    from app.geometry.gt_margin import extract_gt_crown_margin

    axis = OCCLUSAL_AXIS if axis is None else np.asarray(axis, dtype=float)
    result = extract_gt_crown_margin(crown, support, axis)
    if not len(result.points):
        raise ValueError(
            "真值颈缘提取失败：" + "；".join(result.diagnostics.get("reasons", [])))
    return np.asarray(result.points, dtype=float), result.as_dict()


def cervical_margin_legacy(gt_crown: trimesh.Trimesh, n_points: int = 80) -> np.ndarray:
    """【旧方法，仅用于记录 legacy 度量】取真值冠最低 0.4 mm 带的顶点。

    **已证明不可用**：该带只覆盖 16/24 方位角，是**残弧不是闭环**；
    spacing_cv 1.89。改用 :func:`gt_cervical_seam`。

    保留此函数是为了让 ``baseline.json`` 里的 ``legacy_measurement`` 仍可复现 ——
    那些旧数值与新数值**不可直接比较**（度量定义已变）。
    """
    verts = np.asarray(gt_crown.vertices, dtype=float)
    z_min = verts[:, 2].min()
    band = verts[verts[:, 2] < z_min + 0.4]
    center = band.mean(axis=0)
    order = np.argsort(np.arctan2(band[:, 1] - center[1], band[:, 0] - center[0]))
    band = band[order]
    idx = np.linspace(0, len(band) - 1, n_points).astype(int)
    return band[idx]


def angle_between_deg(a, b) -> float:
    a = np.asarray(a, dtype=float); a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = np.asarray(b, dtype=float); b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(a, b))), -1.0, 1.0))))


def _nearest_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cKDTree(dst).query(src, k=1, workers=-1)[0]


def measure_insertion_axis_error_deg(support, margin) -> float:
    """建议就位方向与该病例正确方向（+Z）的夹角。"""
    suggested = suggest_insertion_axis(support, margin)
    return angle_between_deg(suggested, OCCLUSAL_AXIS)


def measure_margin_band_clearance(support, cutter, *, z_margin: float,
                                  band_mm: float = 0.8, n_samples: int = 120_000) -> dict:
    """把内表面间隙按"距边缘线高度"分层统计。"""
    sup_pts, _ = trimesh.sample.sample_surface(support, n_samples, seed=6)
    cut_pts, _ = trimesh.sample.sample_surface(cutter, 220_000, seed=5)
    d = _nearest_distance(sup_pts, cut_pts)

    z = sup_pts[:, 2]
    supra = z > z_margin + 0.15
    in_band = supra & (z < z_margin + 0.15 + band_mm)

    return {
        "margin_band_median_mm": float(np.median(d[in_band])) if in_band.any() else None,
        "supra_margin_median_mm": float(np.median(d[supra])) if supra.any() else None,
        "supra_margin_p95_mm": float(np.percentile(d[supra], 95)) if supra.any() else None,
        "supra_margin_max_mm": float(d[supra].max()) if supra.any() else None,
        "n_supra_margin_samples": int(supra.sum()),
    }


def measure_boolean_roundtrip(gt_crown: trimesh.Trimesh, *, pitch: float = 0.10) -> dict:
    """把**未被裁减**的牙冠自身体素化再重建，测量布尔后端引入的固有误差。

    理想情况下（精确布尔）应接近 0；体素降级会显著偏离。
    """
    probe = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    probe.apply_translation(np.asarray(gt_crown.bounds[0]) + 0.6)
    out = crown_minus_cutter(gt_crown, probe, allow_voxel_fallback=True,
                             voxel_pitch_mm=pitch, validate=False)
    d = _nearest_distance(np.asarray(out.vertices), np.asarray(gt_crown.vertices))
    return {
        "backend": boolean_backend_name(),
        "volume_ratio": float(abs(out.volume) / abs(gt_crown.volume)),
        "faces": int(len(out.faces)),
        "input_faces": int(len(gt_crown.faces)),
        "deviation_median_mm": float(np.median(d)),
        "deviation_p95_mm": float(np.percentile(d, 95)),
        "deviation_max_mm": float(d.max()),
    }


def measure_insertion_axis(support, margin) -> dict:
    """就位方向求解器的完整诊断。

    **不要把这个病例的"+Z 误差"当作验收目标。** 实测依据：

    - 基台侧壁法向独立拟合出的轴本身就在 +Z 的 12.9° 外（以 +Z 和以 Margin 平面法向
      为参考分别拟合，两次结果一致到 0.01°，拟合质量 eigenvalue ratio ≈ 0.105），
      即该基台自身的轴线就是斜的 —— 种植基台相对颌骨咬合平面倾斜是常见情况。
    - 倒凹面积、射线不可达、blockout 代价这三类目标在该基台上跨轴变化都在 ±8% 以内
      （原始支持体全量统计），最小值并不落在 +Z。
    - 因此 +Z 不是可从数据独立确认的真值；把它当目标会激励"恰好把方向旋到 +Z"，
      这正是要避免的。

    真正可断言的是：与**该基台自身拟合轴**的一致性、相对平凡解 +Z 的指标改善、
    以及该报不确定时报不确定。精度断言放在合成几何上（test_insertion_axis.py）。
    """
    from app.geometry.insertion import angle_deg, estimate_wall_axis, solve_insertion_axis

    result = solve_insertion_axis(support, margin)
    # 独立几何参考：侧壁法向拟合轴（不依赖求解器内部路径）
    wall_axis, wall_quality, wall_faces, wall_ok = estimate_wall_axis(support, [0.0, 0.0, 1.0])

    return {
        "axis": [round(float(v), 6) for v in result.axis],
        "initial_axis": [round(float(v), 6) for v in result.initial_axis],
        "status": result.status.value,
        "confidence": round(float(result.confidence), 4),
        "score": round(float(result.score), 6),
        "undercut_area_ratio": round(float(result.undercut_area_ratio), 6),
        "undercut_area_mm2": round(float(result.undercut_area_mm2), 4),
        "max_undercut_depth_mm": round(float(result.max_undercut_depth_mm), 4),
        "mean_undercut_depth_mm": round(float(result.mean_undercut_depth_mm), 4),
        "margin_visibility_ratio": round(float(result.margin_visibility_ratio), 6),
        "deviation_deg": round(float(result.deviation_deg), 4),
        "objective_is_flat": bool(result.diagnostics.get("objective_is_flat", False)),
        "reasons": list(result.diagnostics.get("reasons", [])),
        # 与 +Z 的夹角：**仅作诊断记录**，不是目标
        "error_vs_plus_z_deg_DIAGNOSTIC_ONLY": round(
            angle_deg(result.axis, OCCLUSAL_AXIS), 4),
        # 独立参考轴
        "fitted_wall_axis": [round(float(v), 6) for v in wall_axis],
        "fitted_wall_axis_quality": round(float(wall_quality), 6),
        "fitted_wall_axis_faces": int(wall_faces),
        "fitted_wall_axis_deviation_vs_plus_z_deg": round(
            angle_deg(wall_axis, OCCLUSAL_AXIS), 4),
        "axis_vs_fitted_wall_axis_deg": round(angle_deg(result.axis, wall_axis), 4),
        "initial_axis_sources": result.diagnostics.get("initial_axis_sources"),
        "search": result.diagnostics.get("search"),
        "undercut_vs_plus_z": measure_undercut_vs_baseline(support, margin, result.axis),
    }


def measure_undercut_vs_baseline(support, margin, solver_axis) -> dict:
    """求解轴相对平凡解 +Z 的倒凹指标改善。"""
    from app.geometry.insertion import _undercut_metrics

    base = _undercut_metrics(support, OCCLUSAL_AXIS, margin)
    improved = _undercut_metrics(support, np.asarray(solver_axis, dtype=float), margin)

    def _rel(a, b):
        return None if not b else (a - b) / b

    return {
        "undercut_area_ratio_plus_z": round(float(base["undercut_area_ratio"]), 6),
        "undercut_area_ratio_solver": round(float(improved["undercut_area_ratio"]), 6),
        "undercut_area_ratio_rel_change": round(
            float(_rel(improved["undercut_area_ratio"], base["undercut_area_ratio"]) or 0.0), 6),
        "max_depth_plus_z_mm": round(float(base["max_undercut_depth_mm"]), 4),
        "max_depth_solver_mm": round(float(improved["max_undercut_depth_mm"]), 4),
        "max_depth_rel_change": round(
            float(_rel(improved["max_undercut_depth_mm"], base["max_undercut_depth_mm"]) or 0.0), 6),
    }


def _wire_edges(mesh: trimesh.Trimesh) -> dict:
    """导出后重读的网格上的边界边 / 非流形边。"""
    edges = np.asarray(mesh.edges_sorted)
    if not len(edges):
        return {"boundary_edges": 0, "nonmanifold_edges": 0}
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"boundary_edges": int((counts == 1).sum()), "nonmanifold_edges": int((counts > 2).sum())}


def _boundary_edges(mesh: trimesh.Trimesh) -> int:
    return _wire_edges(mesh)["boundary_edges"]


def margin_plane_status(margin) -> dict:
    """Margin 是否是数值平面（决定平面裁切这条路能不能走）。"""
    return margin_plane_diagnostics(margin)


def measure_plane_clip(mesh, margin, axis, *, extension_mm: float = 0.15) -> dict:
    """**平面裁切**（只在 Margin 数值平面时合法）是否产出水密结果。

    这是原始 P0 的保护项：裁切盖片需要 mapbox-earcut，缺失时 ``cap=False`` 会留下
    开放边界，最终得到真值 13% 体积的碎冠。非平面 Margin 走这条路会被守卫拒绝
    （见 ``test_nonplanar_gt_margin_is_refused_not_clipped``）。
    """
    trimmed = trim_outer_at_margin(mesh.copy(), margin, np.asarray(axis, float), extension_mm)
    return {"watertight": bool(trimmed.is_watertight),
            "boundary_edges": _boundary_edges(trimmed),
            "faces": int(len(trimmed.faces)),
            "margin_plane": margin_plane_status(margin)}


def measure_live_chain(outer, support, margin, axis, params) -> dict:
    """**sanctioned** 非平面链路的几何不变量（与 API ``finalize_crown`` 同一实现族）。

    为什么不再用 ``trim_outer_at_margin + enforce_clearance + 体素布尔``：那条链路
    (a) 对非平面 Margin 已被守卫禁止，(b) 里面的 ``enforce_clearance`` 会把 37% 顶点外推
    最多 1.63 mm，正是"外冠形态被内腔环节改写"的 P0 缺陷来源。live 链路不补厚，
    薄区由 ``audit_manufacturing`` 报告并阻断放行。

    断言口径：
    - ``watertight`` / ``boundary_edges`` / ``components``：碎冠与碎片回归；
    - ``seam``：共享颈缘缝的**导出后**双向误差与共享边校验（量化后仍成立）；
    - ``outer_deviation_p95_mm``：链路对外冠形态的改写量（``body_to_original_outer_p95_mm``）；
    - ``mass_balance_relative_error``：``final ≈ outer − 内腔``，与几何形状无关，
      能同时抓到"崩坏"和"内腔没被真正扣减"。
    """
    from app.geometry.intaglio import (BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout)
    from app.geometry.margin_seam import audit_margin_seam
    from app.geometry.nonplanar_crown import build_nonplanar_crown, fitting_support_above_margin

    margin = np.asarray(margin, float)
    axis = np.asarray(axis, float)
    fitting, fitting_report = fitting_support_above_margin(support, margin, axis)
    policy = BlockoutPolicy(machine_axes=int(params["machine_axes"]),
                            additional_blockout_after_gap=None,
                            untouchable_zone_mm=float(params.get("untouchable_zone_mm") or 0.0),
                            provenance={"source": "acceptance_live_chain"})
    blockout = build_blockout(fitting, margin, axis, policy)
    gap = GapFieldParams(cement_gap_mm=float(params["cement_gap_mm"]),
                         cement_gap_start_mm=params.get("cement_gap_start_mm"),
                         cement_gap_top_mm=params.get("cement_gap_top_mm"),
                         additional_spacing_xyz_mm=None, additional_spacing_z_on_top_mm=None,
                         provenance={"source": "acceptance_live_chain"})
    cutter = apply_gap_field(blockout, fitting, margin, axis, gap)
    built = build_nonplanar_crown(outer, cutter, support, margin, axis,
                                 transition_mm=float(params["transition_mm"]))
    final, report = built.final, built.report

    import tempfile
    from pathlib import Path
    from app.geometry.io import save_mesh
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "final.stl"
        save_mesh(final, path)
        persisted = load_mesh(path)
        ids = cKDTree(np.asarray(persisted.vertices)).query(margin, k=1)[1]
        seam = audit_margin_seam(np.asarray(persisted.vertices), np.asarray(persisted.faces), ids, margin)
        wire = _wire_edges(persisted)
    cavity = trimesh.boolean.intersection([outer, cutter], engine="manifold")
    cavity_volume = float(abs(cavity.volume)) if cavity is not None else float("nan")
    outer_volume = float(abs(outer.volume))
    return {
        "chain": "fitting_support > directional_blockout > variable_gap > nonplanar_shared_seam",
        "params": {k: params[k] for k in sorted(params) if not isinstance(params[k], dict)},
        "margin_plane": margin_plane_status(margin),
        "fitting_faces": int(len(fitting.faces)),
        "fitting_scan_adherence_max_mm": float(fitting_report["scan_adherence"]["scan_distance_max_mm"]),
        "cutter_watertight": bool(cutter.is_watertight),
        "watertight": bool(final.is_watertight),
        "boundary_edges": wire["boundary_edges"],
        "nonmanifold_edges": wire["nonmanifold_edges"],
        "components": int(len(final.split(only_watertight=False))),
        "seam_shared_edges_incidence_two": bool(seam["shared_seam_edges_incidence_two"]),
        "seam_margin_to_seam_max_mm": float(seam["margin_to_seam_max_mm"]),
        "seam_to_margin_max_mm": float(seam["seam_to_margin_max_mm"]),
        "outer_deviation_p95_mm": float(report["body_to_original_outer_p95_mm"]),
        "outer_deviation_max_mm": float(report["body_to_original_outer_max_mm"]),
        "volume_final_mm3": float(abs(final.volume)),
        "volume_outer_mm3": outer_volume,
        "volume_cavity_mm3": cavity_volume,
        "volume_ratio": float(abs(final.volume) / outer_volume) if outer_volume else float("nan"),
        "mass_balance_relative_error": float(abs(abs(final.volume) - (outer_volume - cavity_volume)) / outer_volume)
        if outer_volume and np.isfinite(cavity_volume) else float("nan"),
        "thickness_gate": "NOT_RUN (audit_manufacturing 在服务层做；这里只测链路几何)",
    }


# ------------------------------------------------------------ 间隙分布（三区域）

def _clearance_regions(support, margin, axis, *, band_mm: float = 0.8,
                       band_offset_mm: float = 0.15, top_band_mm: float = 1.0):
    """把 support 顶点按**沿轴相对 Margin 的位置**分成三区。

    ``margin_band`` / ``axial_fit`` / ``top_fit``。区域边界用轴向坐标定义，
    与既有的 ``margin_band_clearance_median_mm``（v2）保持一致，
    以便新旧数值可直接比较。
    """
    pts = np.asarray(support.vertices, dtype=float)
    a = np.asarray(axis, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    s = pts @ a
    m_s = float((np.asarray(margin, dtype=float) @ a).mean())
    top_ref = float(s.max())
    band = s <= m_s + band_offset_mm + band_mm
    top = (~band) & (s >= top_ref - top_band_mm)
    axial = (~band) & (~top)
    return {"margin_band": band, "axial_fit": axial, "top_fit": top}


def _clearance_regions_plane_relative(support, margin, axis, *,
                                      band_mm: float = 0.8,
                                      band_offset_mm: float = 0.15,
                                      top_band_mm: float = 1.0):
    """按 **Margin 平面法向距离**划分三区（v3 定义）。

    为什么需要它：FDI36 的真实颈缘平面偏离 +Z **11.67°**。用轴向坐标切"边缘带"
    会把倾斜一侧远低于 Margin 平面的区域也算进去，使该区的间隙被高估。
    平面相对距离在 Margin 倾斜时才是正确的分区方式。

    这是**新增指标**，不与 v2 的轴向版本混用 —— 后者保持不变以便新旧可比。
    """
    from app.geometry.insertion import _robust_plane

    pts = np.asarray(support.vertices, dtype=float)
    a = np.asarray(axis, dtype=float)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    m = np.asarray(margin, dtype=float)
    c, n, _ = _robust_plane(m)
    if float(np.dot(n, a)) < 0:
        n = -n
    d = (pts - c) @ n                    # 距 Margin 平面的有符号距离
    s = pts @ a
    top_ref = float(s.max())
    band = (d > band_offset_mm) & (d <= band_offset_mm + band_mm)
    top = (d > band_offset_mm + band_mm) & (s >= top_ref - top_band_mm)
    axial = (d > band_offset_mm + band_mm) & (~top)
    return {"margin_band": band, "axial_fit": axial, "top_fit": top}


def _dist_to_cutter(points: np.ndarray, cutter, *, n_samples: int = 220_000,
                    seed: int = 5) -> np.ndarray:
    """点到 cutter **表面**的最近距离（对 cutter 表面密集采样后 KD-tree 查询）。"""
    cp, _ = trimesh.sample.sample_surface(cutter, n_samples, seed=seed)
    return cKDTree(cp).query(points, k=1, workers=-1)[0]


def _distribution(values: np.ndarray) -> dict:
    """p05 / median / p95 / max。

    **必须保存整个分布，不能只保存中位数。** 否则很容易出现"一半区域 0、
    另一半 0.5，平均下来很漂亮"但实际是坏的情况。
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"p05": None, "median": None, "p95": None, "max": None, "n": 0}
    return {
        "p05": round(float(np.percentile(v, 5)), 6),
        "median": round(float(np.median(v)), 6),
        "p95": round(float(np.percentile(v, 95)), 6),
        "max": round(float(v.max()), 6),
        "n": int(v.size),
    }


def measure_clearance_distributions(support, cutter, margin, axis) -> dict:
    """三区域的间隙分布（轴向分区，v2 口径）。"""
    pts = np.asarray(support.vertices, dtype=float)
    d = _dist_to_cutter(pts, cutter)
    reg = _clearance_regions(support, margin, axis)
    return {name: _distribution(d[mask]) for name, mask in reg.items()}


def measure_clearance_distributions_plane_relative(support, cutter, margin, axis) -> dict:
    """三区域的间隙分布（**Margin 平面相对**分区，v3 新增口径）。"""
    pts = np.asarray(support.vertices, dtype=float)
    d = _dist_to_cutter(pts, cutter)
    reg = _clearance_regions_plane_relative(support, margin, axis)
    return {name: _distribution(d[mask]) for name, mask in reg.items()}


def measure_input_hashes(gt_crown, support, margin, axis) -> dict:
    """三个输入（Outer / Margin / Axis）的内容哈希。

    用途：证明某一轮的改善**不是靠改输入得来的**。
    Axis / Margin / Outer 的哈希不变 + 指标变化 ⇒ 变化来自被测算法。
    """
    import hashlib

    def h(arr):
        a = np.round(np.asarray(arr, dtype=float), 6)
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

    return {
        "outer_crown": h(np.asarray(gt_crown.vertices)),
        "support": h(np.asarray(support.vertices)),
        "margin_v2": h(np.asarray(margin, dtype=float)),
        "insertion_axis": h(np.asarray(axis, dtype=float)),
    }
