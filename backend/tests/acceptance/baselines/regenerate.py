"""生成 / 更新 ``tests/acceptance/baselines/baseline.json``。

    python -m tests.acceptance.baselines.regenerate            # 打印当前测量值
    python -m tests.acceptance.baselines.regenerate --write     # 写入 baseline.json

baseline.json 分两部分：

- ``measurements`` —— 当前代码在 FDI36 基准病例上的**实测值**。用于在 diff 里
  看到每次几何改动的 before/after。
- ``targets``      —— 验收阈值。测试断言的是 targets，不是 measurements。
  因此当前不达标的项会失败（并以 ``xfail(strict=True)`` 标记为已知缺陷），
  修好后 XPASS 会让 strict 模式报错，强制移除标记。目标：0 xfail。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from tests.acceptance import _metrics as M  # noqa: E402
from tests.acceptance.conftest import BASELINE_PATH, exocad_cases_dir, fdi36_dir  # noqa: E402

#: 验收阈值。每一项都要能回答"这条断言是为了拦住哪个具体失败模式"。
TARGETS: dict = {
    "boolean_backend": {
        "target": "manifold3d",
        "why": "体素降级在无裁减情况下也会把体积放大 10%、引入 0.089 mm 中位误差。",
        "tolerance": None,
    },
    "margin_trim_watertight": {
        "target": True,
        "why": "非水密外冠进入布尔后实测产出真值 13% 体积的碎冠。裁切盖片需要 mapbox-earcut。",
        "tolerance": None,
    },
    # 就位方向：**不以 +Z 为目标**（见 _metrics.measure_insertion_axis 的说明）。
    # 该基台自身轴线就偏 +Z 12.9°，三类倒凹目标在跨轴 ±8% 内无法区分，+Z 不是可独立
    # 确认的真值。可断言的是与基台自身拟合轴的一致性 + 相对平凡解的改善 + 该报不确定时报。
    "insertion_axis_vs_fitted_wall_axis_deg": {
        "target_max": 8.0,
        "why": "求解结果应与基台侧壁法向独立拟合出的轴一致（该拟合质量 eigenvalue ratio ≈ 0.105）。"
               "精度断言在合成几何上，这里只保证与该病例几何自洽。",
        "tolerance": None,
    },
    "insertion_axis_undercut_rel_improvement": {
        "target_max": -0.25,
        "why": "求解轴相对平凡解 +Z 的倒凹面积比必须显著改善（至少降低 25%），"
               "否则说明求解器没有带来实际收益。",
        "tolerance": None,
    },
    "insertion_axis_margin_visibility": {
        "target_min": 0.90,
        "why": "Margin 沿最优轴必须基本可见。",
        "tolerance": None,
    },
    "insertion_axis_status": {
        "target": "OK",
        "why": ("用**正确的**颈缘真值（v2）后，最优轴倒凹比 0.117 < 0.15、Margin 可见率 1.0、"
                "朝向场锐利（|cos|<0.1 占 0.7%）—— 三条 REVIEW 触发条件都不成立，故应为 OK。"),
        "tolerance": None,
    },
    "margin_band_clearance_median_mm": {
        "target_range": [0.015, 0.050],
        "why": "边缘带是精度最关键区域。exocad 真实参数 margin_gap 目标为 0.025（无量取时用 cement_gap）。",
        "tolerance": None,
    },
    "supra_margin_clearance_p95_mm": {
        "target_max": 0.100,
        "why": "配合面间隙上界。exocad 真实参数 cement_gap 多为 0.02–0.03，p95 超过 0.10 说明投影柱 blockout 把内腔撑大了。",
        "tolerance": None,
    },
    "boolean_roundtrip_deviation_p95_mm": {
        "target_max": 0.020,
        "why": "精确布尔的往返误差应接近 0。体素路径在此项会显著超标。",
        "tolerance": None,
    },
    "isolated_chain_outer_deviation_p95_mm": {
        "target_max": 0.050,
        "why": "以真值冠为输入的隔离链路偏差 = CAD 链路自身误差。牙冠边缘适合性预算约 50–100 µm。",
        "tolerance": None,
    },
    # ---- 本轮新增的目标（v3 口径）。与 v2 轴向口径并存，不互相替代。
    "margin_plane_band_median_mm": {
        "target_range": [0.015, 0.050],
        "why": ("按 **Margin 平面相对距离**划分边缘带的中位间隙。"
                "v2 的轴向口径在 Margin 倾斜 11.67° 的 FDI36 上会把倾斜一侧远低于"
                "Margin 平面的区域算进边缘带，那里 blockout 必须深填，使该区被高估。"
                "平面相对口径才是 Margin 倾斜时的正确分区。"),
        "tolerance": None,
    },
    "blockout_own_undercut_area_ratio": {
        "target_max": 1e-6,
        "why": ("方向阴影体沿就位轴必须完全单值 —— 这是『倒凹已被封闭』的直接证据。"
                "非 0 说明阴影体构造有误或底面盖片未排除。"),
        "tolerance": None,
    },
    "support_undercut_area_ratio": {
        "target_max": 0.15,
        "why": "原始 support 在就位面内的拔模负角面积比（blockout 需要处理的量）。",
        "tolerance": None,
    },
    "isolated_chain_volume_ratio": {
        "target_range": [0.90, 1.10],
        "why": "隔离链路（完美外冠输入）不应显著改变体积。最小厚度修正曾把 37% 顶点外推最多 1.63 mm。",
        "tolerance": None,
    },
    "final_watertight": {
        "target": True,
        "why": "CAM 可用性的底线。",
        "tolerance": None,
    },
    "final_components": {
        "target": 1,
        "why": "牙冠被切成多块说明布尔或厚度修正产生了自交/断裂。",
        "tolerance": None,
    },
}

#: 每个度量的定义版本。
#:
#: **必要**：Margin 真值的提取方式已从"最低 0.4 mm 带"改为"intaglio ↔ exterior
#: 真实交界缝"，因此任何用到 Margin 的度量，其定义都变了。
#: 旧值记录在 ``legacy_measurement``，与 ``measurements`` **不可直接比较** ——
#: 差异里混了 Margin 真值修正，不能归因给 blockout 改善。
METRIC_VERSIONS: dict[str, dict] = {
    "_default": {"version": "v1", "definition": "未变更"},
    "margin_band_clearance_median_mm": {
        "version": "v2_gt_cervical_seam",
        "definition": ("按 Margin 高度分层的内腔间隙中位。Margin 来源从"
                       "'最低 0.4 mm 带'（16/24 方位角残弧）改为"
                       "'intaglio↔exterior 真实交界缝'（24/24 闭环，周长 21.76 mm）。"),
    },
    "supra_margin_clearance_p95_mm": {
        "version": "v2_gt_cervical_seam",
        "definition": "同上，配合面（边缘带以上）间隙的 p95。",
    },
    "supra_margin_clearance_median_mm": {
        "version": "v2_gt_cervical_seam",
        "definition": "同上。",
    },
    "supra_margin_clearance_max_mm": {
        "version": "v2_gt_cervical_seam",
        "definition": "同上。",
    },
    "isolated_chain": {
        "version": "v2_gt_cervical_seam",
        "definition": "隔离链路（真值冠作完美外冠输入）的偏差与体积比。Margin 裁切平面随之改变。",
    },
    "isolated_chain_outer_deviation_p95_mm": {
        "version": "v2_gt_cervical_seam",
        "definition": "同上。",
    },
    # ---- 本轮新增的目标（v3 口径）。与 v2 轴向口径并存，不互相替代。
    "margin_plane_band_median_mm": {
        "target_range": [0.015, 0.050],
        "why": ("按 **Margin 平面相对距离**划分边缘带的中位间隙。"
                "v2 的轴向口径在 Margin 倾斜 11.67° 的 FDI36 上会把倾斜一侧远低于"
                "Margin 平面的区域算进边缘带，那里 blockout 必须深填，使该区被高估。"
                "平面相对口径才是 Margin 倾斜时的正确分区。"),
        "tolerance": None,
    },
    "blockout_own_undercut_area_ratio": {
        "target_max": 1e-6,
        "why": ("方向阴影体沿就位轴必须完全单值 —— 这是『倒凹已被封闭』的直接证据。"
                "非 0 说明阴影体构造有误或底面盖片未排除。"),
        "tolerance": None,
    },
    "support_undercut_area_ratio": {
        "target_max": 0.15,
        "why": "原始 support 在就位面内的拔模负角面积比（blockout 需要处理的量）。",
        "tolerance": None,
    },
    "isolated_chain_volume_ratio": {
        "version": "v2_gt_cervical_seam",
        "definition": "同上。",
    },
    "insertion_axis": {
        "version": "v2_gt_cervical_seam",
        "definition": "就位方向求解以新 Margin 为输入（用于初始轴与可见率评分）。",
    },
}

#: 被测算法链路的版本。**与 ``metric_version`` 是两个不同的东西：**
#:
#: - ``metric_version``  —— 指标**怎么算**。本轮 Margin/clearance 的测量定义没有再变，
#:   所以**保持 v2_gt_cervical_seam 不动**。
#: - ``pipeline_version`` —— 被测算法是什么。本轮 blockout/gap 重写，故升级。
#:
#: 这样 ``margin_band_clearance_median_mm: 0.254 -> 新值`` 才是**可直接比较**的
#: before/after，差异可归因给 blockout/gap —— 而不是混了度量定义变更。
#: （上一轮反过来：度量定义变了，所以必须升 metric_version 并冻结旧值为 legacy。）
PIPELINE_VERSION = "v3_directional_blockout_variable_gap"
PIPELINE_COMPONENTS = {
    "insertion_axis": "v2_cone_search_area_weighted_undercut",
    "margin": "v2_gt_cervical_seam",
    "blockout": "v3_directional_shadow_volume",
    "cement_gap": "v3_variable_gap_field",
    "thickness": "v1_unified_offset  (本轮未改动)",
    "boolean": "manifold3d_exact",
}

#: 上一版（Margin = 最低 0.4 mm 带）测得的数值。**只作历史记录，不参与断言。**
LEGACY_MEASUREMENT: dict = {
    "_note": ("这些数值的度量定义已过期（Margin 真值提取方式变更）。"
              "禁止把它们与 measurements 直接比较 —— 差异里混了度量定义变更，"
              "不能归因给几何算法改善。要比较必须用同一 metric_version 重跑。"),
    "_legacy_margin_definition": "真值冠最低 0.4 mm 带（16/24 方位角，残弧；spacing_cv 1.89）",
    "margin_band_clearance_median_mm": 0.515,
    "supra_margin_clearance_p95_mm": 0.965,
    "isolated_chain_outer_deviation_p95_mm": 0.976,
}

#: 上一轮的判据修正记录（以免重犯）。
RETRACTED_EXPECTATIONS: dict[str, str] = {
    "insertion_axis_reports_review_when_undercut_remains":
        "曾期望该病例就位方向必须报 REVIEW_REQUIRED，理由是『该基台有轴对称外扩 emergence "
        "profile，相对任何轴都是倒凹，OK 是错答案』。**该结论已被推翻。** "
        "它建立在错误的 Margin 真值上：旧的『最低 0.4 mm 带』给出一个几乎水平的面"
        "（偏离 +Z 0.80°），而真实颈缘缝的平面偏离 +Z 11.67° —— 与基台自身拟合轴"
        "（12.9°）一致，因为边缘平面本就垂直于基台轴线。用错平面切分『就位面』"
        "把倒凹统计的作用域放错了位置，虚高了倒凹比。",
}

#: 已知缺陷 -> 对应的 xfail 理由。修好后必须清空。
KNOWN_DEFECTS: dict[str, str] = {
    # 原 "insertion_axis_error_deg < 10°（vs +Z）" 项已删除，改为上面四项。
    # 原因：该目标不可达且判据本身不成立（+Z 不是真值，见 _metrics 说明）。
    # 强行保留一个以 +Z 为真值的 xfail 会激励"恰好把方向旋到 +Z"，
    # 而不是求出更合理的就位方向。
    "margin_band_clearance_median_mm":
        "P0：build_blockout_cutter 用投影柱充当内腔基面（填的是整个上部基台的投影而非局部截面），"
        "在边缘附近把内腔撑大 2–5 倍。待替换为沿就位方向的可见性 blockout。"
        "注：度量定义已升级为 v2_gt_cervical_seam，与 legacy 数值不可比。",
    "supra_margin_clearance_p95_mm":
        "P0：同上。配合面间隙 p95 实测 0.983 mm。",
    "isolated_chain_outer_deviation_p95_mm":
        "P0：enforce_clearance 用顶点到 cutter 顶点的距离当壁厚，误判并外推 37% 顶点最多 1.63 mm。"
        "待替换为符号距离场真壁厚 + 分区厚度场 t_min(p)。",
    "margin_band_clearance_median_mm_definition_limited":
        "该指标（v2 轴向口径）在本病例上受**度量定义**限制而非算法缺陷："
        "Margin 平面偏离 +Z 11.67°，轴向切带会把倾斜一侧远低于 Margin 平面的区域算进来，"
        "那里的 blockout 必然深填。同一实现按 Margin 平面相对口径测得的边缘带中位为 0.034 mm，"
        "已在目标 0.015–0.05 内（见 targets.margin_plane_band_median_mm）。"
        "**修好算法不改定义是修不好这个数的**；要修定义必须升 metric_version 并冻结旧值。",
}


def measure() -> dict:
    fdi = fdi36_dir()
    if fdi is None:
        raise SystemExit(
            "FDI36 基准病例不可用。请设置 CROWNFLOW_FDI36_DIR 指向病例目录。"
        )

    from app.geometry.boolean import boolean_backend_name
    from app.geometry.intaglio import build_blockout_cutter

    meshes = M.load_fdi36(fdi)
    support = meshes["support"]
    gt = meshes["gt_crown"]

    # 真值颈缘（v2 度量）。旧方法产出的残弧数值仍记录在 legacy_measurement，
    # 但**不可与 v2 直接比较** —— 度量定义已变。
    margin, seam_diag = M.gt_cervical_seam(gt, support)
    margin_z = float((margin @ M.OCCLUSAL_AXIS).mean())

    out: dict = {}
    out["_margin_source"] = "gt_cervical_seam_v2"
    out["_margin_quality"] = {
        "status": seam_diag["status"],
        "is_closed": seam_diag["is_closed"],
        "is_simple": seam_diag["is_simple"],
        "azimuth_coverage": seam_diag["azimuth_coverage"],
        "n_points": len(seam_diag["points"]),
        "perimeter_mm": seam_diag["perimeter_mm"],
        "spacing_cv": seam_diag["spacing_cv"],
        "n_self_intersections": max(
            (c["n_self_intersections"] for c in seam_diag["diagnostics"]["candidates"]),
            default=0),
        "best_fit_plane_residual_max_mm":
            seam_diag["diagnostics"]["best_fit_plane"]["residual_max_mm"],
        "best_fit_plane_tilt_vs_plus_z_deg": round(float(np.degrees(np.arccos(
            abs(np.dot(seam_diag["diagnostics"]["best_fit_plane"]["normal"],
                       M.OCCLUSAL_AXIS))))), 4),
        "n_candidate_loops": seam_diag["diagnostics"]["n_candidate_loops"],
        "genus": seam_diag["diagnostics"]["genus"],
        "confidence": seam_diag["confidence"],
    }
    out["boolean_backend"] = boolean_backend_name()
    out["margin_trim_watertight"] = M.measure_margin_trim(
        gt, margin, M.OCCLUSAL_AXIS)["watertight"]
    out["margin_trim_detail"] = M.measure_margin_trim(gt, margin, M.OCCLUSAL_AXIS)

    out["insertion_axis"] = M.measure_insertion_axis(support, margin)

    cutter = build_blockout_cutter(support, margin, M.OCCLUSAL_AXIS, cement_gap_mm=0.06)
    band = M.measure_margin_band_clearance(support, cutter, z_margin=margin_z)
    out["margin_band_clearance_median_mm"] = _round(band["margin_band_median_mm"])
    out["supra_margin_clearance_median_mm"] = _round(band["supra_margin_median_mm"])
    out["supra_margin_clearance_p95_mm"] = _round(band["supra_margin_p95_mm"])
    out["supra_margin_clearance_max_mm"] = _round(band["supra_margin_max_mm"])
    out["_margin_z_reference_mm"] = round(margin_z, 3)

    # ---- 本轮新增：方向 blockout 的度量 + 三区域间隙分布 + 输入哈希
    from app.geometry.intaglio import (
        BlockoutPolicy, GapFieldParams, apply_gap_field, build_blockout)
    from app.geometry.insertion import solve_insertion_axis

    solver = solve_insertion_axis(support, margin)
    solver_axis = np.asarray(solver.axis, dtype=float)

    policy = BlockoutPolicy(machine_axes=5, additional_blockout_after_gap=None,
                            untouchable_zone_mm=0.1,
                            provenance={"source": "acceptance_fixture_5axis"})
    gap_params = GapFieldParams(
        cement_gap_mm=0.025, cement_gap_start_mm=None, cement_gap_top_mm=None,
        additional_spacing_xyz_mm=None, additional_spacing_z_on_top_mm=None,
        provenance={"source": "acceptance_fixture_assumed",
                    "_note": ("FDI36 没有 .constructionInfo，其 cement gap 未知。"
                              "验收固定用证据数据集的众数 0.025 作为该 fixture 的取值，"
                              "仅用于让 clearance 可测量；不是该病例的真值。")})

    gap_diag = {}
    clearances = {}
    for label, ax in (("legacy_axis_plus_z", M.OCCLUSAL_AXIS), ("solver_axis", solver_axis)):
        bo = build_blockout(support, margin, ax, policy)
        if label == "solver_axis":
            out["_blockout_quality"] = {
                "status": bo.status.value,
                "watertight": bool(bo.blockout_mesh.is_watertight),
                "n_components": int(len(bo.blockout_mesh.split(only_watertight=False))),
                "volume_mm3": round(float(abs(bo.blockout_mesh.volume)), 4),
                "support_undercut_area_ratio": round(float(bo.undercut_area_ratio), 6),
                "blockout_undercut_area_ratio": round(float(bo.blockout_undercut_area_ratio), 6),
                "max_blockout_depth_mm": round(float(bo.max_blockout_depth_mm), 4),
                "margin_protected_area_mm2": round(float(bo.margin_protected_area_mm2), 4),
                "diagnostics": bo.diagnostics,
            }
        cutter = apply_gap_field(bo, support, margin, ax, gap_params)
        gap_diag[label] = {
            "cutter_watertight": bool(cutter.is_watertight),
            "clearance_axial": M.measure_clearance_distributions(support, cutter, margin, ax),
            "clearance_plane_relative": M.measure_clearance_distributions_plane_relative(
                support, cutter, margin, ax),
        }
    out["clearance_distributions"] = gap_diag
    out["input_hashes"] = M.measure_input_hashes(gt, support, margin, solver_axis)

    rt = M.measure_boolean_roundtrip(gt, pitch=0.10)
    out["boolean_roundtrip_deviation_p95_mm"] = _round(rt["deviation_p95_mm"])
    out["boolean_roundtrip_detail"] = {k: _round(v) if isinstance(v, float) else v
                                      for k, v in rt.items()}

    iso = M.measure_isolated_cad_chain(gt, support, margin)
    out["isolated_chain"] = {
        label: {k: (_round(v) if isinstance(v, float) else
                    ([_round(x) for x in v] if isinstance(v, list) else v))
                for k, v in data.items()}
        for label, data in iso.items()
    }
    # 取"用正确轴"的一档作为主指标
    out["isolated_chain_outer_deviation_p95_mm"] = out["isolated_chain"]["correct_z"]["deviation_p95_mm"]
    out["isolated_chain_volume_ratio"] = out["isolated_chain"]["correct_z"]["volume_ratio"]
    out["final_watertight"] = out["isolated_chain"]["correct_z"]["final_watertight"]
    out["final_components"] = out["isolated_chain"]["correct_z"]["final_components"]

    # 证据数据集规模（供参数验收引用）
    exo = exocad_cases_dir()
    if exo is not None:
        from app.importers.exocad.case_loader import discover_cases
        groups = discover_cases(exo)
        out["_evidence_dataset"] = {
            "n_case_dirs": len([d for ds in groups.values() for d in ds]),
            "n_order_groups": len(groups),
            "n_duplicate_dir_groups": sum(1 for ds in groups.values() if len(ds) > 1),
        }
    return out


def _producer_commit() -> str | None:
    """当前 HEAD 的提交哈希 —— 便于把一组 baseline 数值钉到具体代码状态。"""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _round(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="写入 baseline.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    measurements = measure()
    metric_versions = {
        key: METRIC_VERSIONS.get(key, METRIC_VERSIONS["_default"])
        for key in measurements
    }
    payload = {
        "schema_version": 3,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_components": PIPELINE_COMPONENTS,
        "producer_commit": _producer_commit(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": "FDI36 / 1629-王煜",
        "note": (
            "measurements 是当前实测值（用于 diff 追踪）；targets 是验收阈值（测试断言它）。"
            "KNOWN_DEFECTS 里的项当前应失败并被 xfail(strict=True) 标记，修好后必须移除标记。"
            "metric_versions 记录每个度量的定义版本 —— **跨版本比较无效**。"
        ),
        "metric_versions": metric_versions,
        "retracted_expectations": RETRACTED_EXPECTATIONS,
        "legacy_measurement": LEGACY_MEASUREMENT,
        "measurements": measurements,
        "targets": TARGETS,
        "known_defects": KNOWN_DEFECTS,
    }

    print(json.dumps(measurements, ensure_ascii=False, indent=2))

    if a.write:
        path = Path(a.out) if a.out else BASELINE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
        print(f"\n已写入 {path}")


if __name__ == "__main__":
    main()
