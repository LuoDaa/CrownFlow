"""参数敏感性（规格 §13/§14）：**参数必须真正影响几何**，否则不算实现。

这批测试就是本批的验收规则：A/B 两个取值，断言输出几何与对应 QC 有单调或明确变化。
只用合成几何（不读真实病例库），不构造假 bite 来让咬合测试"通过"。

覆盖（P0 范围）：
    test_proximal_target_changes_contact
    test_contact_area_mode_is_load_bearing
    test_occlusal_target_changes_clearance
    test_occlusal_requires_confirmed_bite_pose
    test_min_thickness_changes_outer_geometry
    test_cement_gap_changes_actual_gap
    test_profile_missing_is_never_zero
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import PatientFrame
from app.geometry.io import load_mesh
from app.services.crown_constraint_optimizer import MAX_ITER, optimize_crown
from app.services.crown_constraint_solvers import (
    STATUS_NOT_ASSESSED,
    STATUS_PASS,
    STATUS_REVIEW,
    enforce_minimum_thickness,
    solve_occlusion,
    solve_proximal,
)
from app.services.crown_generation_profile import (
    CONTACT_AREA_TARGETS_MM2,
    ProfileParameterMissing,
    from_parameters,
)

RADIUS = 3.0
SECTIONS = 64


# ---------------------------------------------------------------- 合成几何
def _frame(margin):
    origin = np.asarray(margin, float).mean(axis=0)
    return PatientFrame(origin=origin, z=np.array([0.0, 0.0, 1.0]),
                        x=np.array([1.0, 0.0, 0.0]), y=np.array([0.0, 1.0, 0.0]),
                        revision_id="frame-test")


def _margin(amplitude=0.0):
    theta = np.arange(SECTIONS) * 2 * np.pi / SECTIONS
    return np.c_[RADIUS * np.cos(theta), RADIUS * np.sin(theta),
                 amplitude * np.sin(2 * theta)]


def _crown(radius=RADIUS + 1.0, height=4.0, sections=SECTIONS, max_edge=0.35):
    """外冠：闭合圆柱。

    ⚠️ **必须细分**：裸圆柱只有上下两圈顶点（+2 个盖心），于是"既非颈缘带也非咬合面"
    的近远中带**恒为空**，`fit_domains` 会把全部顶点锁死、近远中域一个点都不剩
    （实测 mask count = 0）。细分后侧壁才有顶点，近远中域才有意义。
    """
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.apply_translation([0, 0, height / 2.0])
    if max_edge:
        vertices, faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=max_edge)
        mesh = trimesh.Trimesh(vertices, faces, process=True)
    return mesh


def _neighbour(offset_mm: float, radius=1.6, height=4.0):
    """邻牙：闭合实体（⇒ FitSurface 走 SOLID_CONTAINMENT，符号无需人工声明）。"""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    mesh.apply_translation([RADIUS + 1.0 + radius + offset_mm, 0, height / 2.0])
    return mesh


def _antagonist(clearance_mm: float, radius=RADIUS + 1.4):
    """对颌：一块盖住冠顶的闭合圆盘，底面在 z = height + clearance。"""
    mesh = trimesh.creation.cylinder(radius=radius, height=1.0, sections=48)
    mesh.apply_translation([0, 0, 4.0 + clearance_mm + 0.5])
    return mesh


#: 与真实链路一致的"必须显式给出"的参数集合（缺任何一项都该 fail-loud）
BASE = {"proximal_target_mm": 0.05, "occlusal_target_mm": 0.15,
        "minimum_thickness_mm": 0.5, "cement_gap_mm": 0.06,
        "margin_band_mm": 0.8, "machine_axes": 5, "cement_gap_start_mm": 0.5,
        "undercut_policy": "REMOVE", "border_horizontal_mm": 0.25}


def _profile(**overrides):
    return from_parameters({**BASE, **overrides})


def _displacement(a, b):
    return np.linalg.norm(np.asarray(a.vertices) - np.asarray(b.vertices), axis=1)


# ---------------------------------------------------------------- 近远中
def test_proximal_target_changes_contact():
    """proximal_target 改变 → 最终近远中距离改变（且方向正确）。"""
    margin, frame = _margin(), _frame(_margin())
    crown = _crown()
    neighbour = _neighbour(offset_mm=0.60)

    near_gap = solve_proximal(crown, frame, margin, {"mesial": FitSurface(neighbour)},
                              _profile(proximal_target_mm=0.02))
    far_gap = solve_proximal(crown, frame, margin, {"mesial": FitSurface(neighbour)},
                             _profile(proximal_target_mm=0.25))

    assert near_gap["status"] == STATUS_PASS, near_gap["reasons"]
    assert far_gap["status"] == STATUS_PASS, far_gap["reasons"]

    near_after = near_gap["per_side"]["mesial"]["after_distance"]["min_mm"]
    far_after = far_gap["per_side"]["mesial"]["after_distance"]["min_mm"]
    assert near_after < far_after, "目标距离变大必须让最终距离变大（实测 %.4f vs %.4f）" % (near_after, far_after)
    # 参数必须改动几何：两个目标值下外冠顶点不同
    assert _displacement(near_gap["geometry"], far_gap["geometry"]).max() > 5e-3, \
        "不同 proximal_target 必须给出不同几何，否则参数只写在 JSON 里"
    assert near_gap["target_distance_mm"] == 0.02 and far_gap["target_distance_mm"] == 0.25
    # 双侧都要有 before/after 距离与穿透读数
    block = near_gap["per_side"]["mesial"]
    for key in ("before_distance", "after_distance", "before_penetration_mm",
                "after_penetration_mm", "contact_area", "convergence", "iterations"):
        assert key in block, "近远中报告缺少 %s" % key
    assert block["before_penetration_mm"] >= block["after_penetration_mm"] - 1e-9


def test_contact_area_mode_is_load_bearing():
    """contact_area_mode 不能只是 UI 常量：目标必须进入求解与报告，未达成要如实标 REVIEW。"""
    margin, frame = _margin(), _frame(_margin())
    crown = _crown()
    neighbour = _neighbour(offset_mm=0.60)

    small = solve_proximal(crown, frame, margin, {"mesial": FitSurface(neighbour)},
                           _profile(proximal_contact_area_mode="SMALL"))
    large = solve_proximal(crown, frame, margin, {"mesial": FitSurface(neighbour)},
                           _profile(proximal_contact_area_mode="LARGE"))

    assert small["contact_area_target_mm2"] == CONTACT_AREA_TARGETS_MM2["SMALL"]
    assert large["contact_area_target_mm2"] == CONTACT_AREA_TARGETS_MM2["LARGE"]
    assert small["contact_area_target_source"] == "DERIVED"
    # 面积口径必须是**有版本、未认证**的显式定义
    assert small["contact_area_definition"] and small["contact_area_validated"] is False
    assert small["per_side"]["mesial"]["contact_area"]["after"]["validated"] is False
    # 目标显著大于可达面积时，不许静默成功
    if large["status"] != STATUS_PASS:
        assert "CONTACT_AREA_TARGET_NOT_REACHED" in large["reasons"], large["reasons"]
    # 未达成时的判定不能是 PASS（禁止"部分失败仍标 PASS"）
    reached = (large["per_side"]["mesial"]["contact_area"]["after"].get("area_mm2") or 0.0) \
        >= 0.95 * CONTACT_AREA_TARGETS_MM2["LARGE"]
    assert reached or large["status"] == STATUS_REVIEW


# ---------------------------------------------------------------- 咬合
def test_occlusal_target_changes_clearance():
    """occlusal_target 改变 → 最终咬合间隙 min 改变（有已确认位姿时）。"""
    margin, frame = _margin(), _frame(_margin())
    crown = _crown()
    antagonist = _antagonist(clearance_mm=-0.20)          # 初始是穿透
    pose = {"pose_confirmed": True, "revision_id": "bite-test", "transform": np.eye(4).tolist()}

    tight = solve_occlusion(crown, frame, margin, FitSurface(antagonist),
                            _profile(occlusal_target_mm=0.05), bite_pose=pose)
    loose = solve_occlusion(crown, frame, margin, FitSurface(antagonist),
                            _profile(occlusal_target_mm=0.30), bite_pose=pose)

    assert tight["bite_pose_status"] == "CONFIRMED"
    assert tight["before_min"] is not None and tight["after_min"] is not None
    assert tight["penetration_before_mm"] > 0, "夹具应造出初始穿透，否则测不到修正"
    assert tight["after_min"] < loose["after_min"], \
        "目标间隙变大必须让最终间隙变大（实测 %.4f vs %.4f）" % (tight["after_min"], loose["after_min"])
    assert _displacement(tight["geometry"], loose["geometry"]).max() > 5e-3, \
        "不同 occlusal_target 必须给出不同几何"
    for key in ("target_clearance_mm", "before_min", "after_min", "penetration_before_mm",
                "penetration_after_mm", "adjusted_area_mm2", "iterations", "convergence"):
        assert key in tight, "咬合报告缺少 %s" % key


def test_occlusal_requires_confirmed_bite_pose():
    """没有确认咬合位姿 ⇒ OCCLUSION_NOT_ASSESSED / BITE_POSE_NOT_CONFIRMED，且不动几何。"""
    margin, frame = _margin(), _frame(_margin())
    crown = _crown()
    antagonist = _antagonist(clearance_mm=-0.20)

    for pose in (None, {}, {"pose_confirmed": False, "revision_id": "x", "transform": None}):
        report = solve_occlusion(crown, frame, margin, FitSurface(antagonist),
                                 _profile(), bite_pose=pose)
        assert report["status"] == "OCCLUSION_NOT_ASSESSED", report
        assert report["reason"] == "BITE_POSE_NOT_CONFIRMED"
        assert report["geometry_changed"] is False
        assert report["geometry"] is None or _displacement(crown, report["geometry"]).max() == 0.0


# ---------------------------------------------------------------- 壁厚
def _inner(crown, inset=0.12):
    """内腔：外冠向内缩一圈（贴合面），用于壁厚强制。"""
    mesh = crown.copy()
    center = np.asarray(crown.vertices).mean(axis=0)
    direction = np.asarray(crown.vertices) - center
    length = np.linalg.norm(direction, axis=1, keepdims=True)
    mesh.vertices = np.asarray(crown.vertices) - direction / np.maximum(length, 1e-9) * inset
    return mesh


def test_min_thickness_changes_outer_geometry():
    """minimum_thickness 改变 → 薄区与位移改变（向外补厚，内面不动）。"""
    crown = _crown()
    inner = _inner(crown, inset=0.12)
    thin = enforce_minimum_thickness(crown, inner, _profile(minimum_thickness_mm=0.20,
                                                           enforce_minimum_thickness=True))
    thick = enforce_minimum_thickness(crown, inner, _profile(minimum_thickness_mm=0.60,
                                                            enforce_minimum_thickness=True))
    # 诚实不变量：状态可以是 PASS 或 REVIEW，但**不许静默宣布成功**
    assert thin["status"] in (STATUS_PASS, STATUS_REVIEW), thin
    assert thick["status"] in (STATUS_PASS, STATUS_REVIEW), thick
    assert (thick["status"] == STATUS_PASS) == (thick["remaining_thin_samples"] == 0), \
        "PASS 必须等价于没有残余薄区（禁止部分失败标 PASS）：%s" % thick["status"]
    assert thick["max_displacement_mm"] > thin["max_displacement_mm"] > 0, \
        "更大的最小厚度必须产生更大的向外补厚（实测 %.4f vs %.4f）" \
        % (thick["max_displacement_mm"], thin["max_displacement_mm"])
    assert _displacement(thin["result"], thick["result"]).max() > 1e-3
    assert thin["intaglio_preserved"] is True
    for key in ("minimum_before_mm", "minimum_after_mm", "accepted_steps", "rejected_steps",
                "max_displacement_mm", "p95_displacement_mm"):
        assert key in thin, "壁厚报告缺少 %s" % key
    # 未开开关时必须 NOT_ASSESSED 且不动几何
    off = enforce_minimum_thickness(crown, inner, _profile(enforce_minimum_thickness=False))
    assert off["status"] == STATUS_NOT_ASSESSED and off["geometry_changed"] is False


# ---------------------------------------------------------------- 内冠间隙（真实链路）
def test_cement_gap_changes_actual_gap():
    """cement_gap_mm 改变 → **实际内腔间隙**改变（走真实 finalize 链路）。

    ⚠️ 量的是 **cutter（腔体实体）表面 → 预备体表面**距离：cutter 的边界就是冠的内腔面。
    拿 `trimmed`（外冠）去量只会得到外冠到预备体的距离（≈1.0 mm，与 gap 无关）——
    第一版就是这么写错的，实测两个 gap 都是 1.0014 mm。

    夹具用 `prep_case` 自带的 **gt_crown + SYNTHETIC_PARAMS**（只有 `cement_gap_mm` 做 A/B）：
    第一版用自造圆柱 + 自选 border 参数组合，被颈缘缝合门如实拒绝
    （`Cervical stitch geometry failed: self_intersections 40`）——那是夹具组合的问题，
    不是参数链的问题，所以这里改用已被验收链路证明可用的参数集。
    """
    from app.services.pipeline import finalize_crown
    from tests.acceptance.synthetic_prep import prep_case

    case = prep_case(amplitude_mm=0.0, sections=48)
    support, margin, axis = case["support"], case["margin"], case["axis"]
    crown = case["gt_crown"]
    import tempfile
    _tmpdir = Path(tempfile.mkdtemp(prefix="crownflow_sens_"))
    # `finalize_crown` 收的是**路径**（不是 Trimesh 对象）：直接传对象会在 load_mesh 里炸。
    crown_path, support_path = _tmpdir / "crown.stl", _tmpdir / "support.stl"
    crown.export(crown_path)
    support.export(support_path)
    cutters, finals = {}, {}
    for name, gap in (("small", 0.03), ("large", 0.12)):
        params = dict(case["params"])
        params.update(cement_gap_mm=gap)          # A/B 只动这一个参数
        tmp = _tmpdir / ("sens_%s" % name)
        tmp.mkdir(parents=True, exist_ok=True)
        trimmed, cutter, final, blockout = finalize_crown(
            crown_path, support_path, tmp / "final.stl", tmp / "cutter.stl", margin, axis, params)
        cutters[name] = load_mesh(tmp / "cutter.stl")      # 资产重读，不看内存里的中间结果
        finals[name] = final

    def _axial_wall_gap(cutter, support):
        """**轴壁段**腔体面 → 预备体表面的最近距离中位数（mm）。

        只看轴壁：紧邻 Margin 那一段本来就有 CementGapStart 的零间隙带
        （夹具 `cement_gap_start_mm=0.5` + `margin_band_mm=0.8`），把它算进来会稀释信号。
        """
        from app.geometry.manufacturing_qc import _closest
        centers = np.asarray(cutter.triangles_center) + 1e-9
        wall = centers[centers[:, 2] >= 1.6]      # 轴 = Z，1.3 = 0.5 + 0.8 之上
        assert len(wall), "轴壁段没有采样点，夹具失效"
        _, distance, _ = _closest(support, wall)
        return float(np.median(distance))

    small_gap = _axial_wall_gap(cutters["small"], support)
    large_gap = _axial_wall_gap(cutters["large"], support)
    assert large_gap > small_gap + 0.02, \
        "cement_gap 变大必须让实际内腔间隙变大（实测 %.4f vs %.4f）" % (small_gap, large_gap)
    assert 0.0 < small_gap < large_gap < 0.40, \
        "间隙量级应落在配置附近（实测 %.4f/%.4f）" % (small_gap, large_gap)
    # 腔体几何确实变了：用**形状无关**的判据（两个 cutter 的面数/顶点数不同，逐顶点比较会广播失败）。
    # 方向别搞反：cutter 是**腔体实体**（被减去的负形体），cement gap 越大 ⇒ 腔体越大。
    # 第一版按"间隙大 ⇒ 实体小"断言，实测 136.27 vs 144.06 mm³ 直接把这个错方向钉了出来。
    assert cutters["large"].volume > cutters["small"].volume, \
        "间隙变大时腔体必须变大（实测 %.4f vs %.4f mm³）" \
        % (cutters["small"].volume, cutters["large"].volume)


# ---------------------------------------------------------------- 缺失参数
def test_profile_missing_is_never_zero():
    """MISSING 不许静默变 0；需要它的阶段必须 fail-loud。"""
    profile = from_parameters({"cement_gap_mm": 0.06})
    assert "proximal.target_distance_mm" in profile.missing()
    assert profile.get("proximal.target_distance_mm") is None
    assert profile.source("proximal.target_distance_mm") == "MISSING"
    with pytest.raises(ProfileParameterMissing):
        profile.require("proximal.target_distance_mm", stage="PROXIMAL")
    # 显式 0 与"没填"必须可区分
    assert from_parameters({"proximal_target_mm": 0.0}).get("proximal.target_distance_mm") == 0.0
    assert from_parameters({"proximal_target_mm": 0.0}).source("proximal.target_distance_mm") == "USER"
    # 面积：只给 mode 时目标是派生的；都不给就是 MISSING（不是 0）
    mode_only = from_parameters({"proximal_contact_area_mode": "MEDIUM"})
    assert mode_only.effective_contact_area_target_mm2() == (CONTACT_AREA_TARGETS_MM2["MEDIUM"], "DERIVED")
    assert from_parameters({}).effective_contact_area_target_mm2() == (None, "MISSING")
    # 策略默认必须显式标注为 DEFAULT_POLICY，而不是混成 USER
    assert from_parameters({}).source("occlusion.require_confirmed_bite_pose") == "DEFAULT_POLICY"
    assert from_parameters({}).get("occlusion.require_confirmed_bite_pose") is True
    # 有未知键时如实登记（不猜用途）
    assert any(note.startswith("UNKNOWN_PARAMETER") for note in from_parameters({"nonsense": 1}).notes)


# ---------------------------------------------------------------- 编排器
def test_optimizer_never_reports_partial_pass():
    """编排：没有确认咬合时整体不得 PASS；NOT_ASSESSED 不等于通过。"""
    margin, frame = _margin(), _frame(_margin())
    crown = _crown()
    neighbour = _neighbour(offset_mm=0.60)
    mesh, report = optimize_crown(crown, frame, margin, _profile(),
                                  proximal_patches={"mesial": FitSurface(neighbour)},
                                  antagonist=FitSurface(_antagonist(clearance_mm=-0.2)),
                                  bite_pose=None, baseline=crown, max_iter=2)
    assert report["iterations"] <= 2
    stages = {s["stage"]: s for s in report["stages"]}
    assert stages["OCCLUSAL"]["status"] == "OCCLUSION_NOT_ASSESSED"
    assert stages["OCCLUSAL"]["reason"] == "BITE_POSE_NOT_CONFIRMED"
    assert report["status"] != STATUS_PASS, "存在未评估阶段时整体不能是 PASS"
    assert report["status"] in (STATUS_REVIEW, "CONSTRAINT_CONFLICT")
    assert report["max_iter"] == MAX_ITER or report["max_iter"] == 2
    assert report["intaglio_stage_in_optimizer"] is False
    assert report["partial_failure_never_passes"] is True
    # 近远中确实被执行并且几何变了
    assert stages["PROXIMAL"]["status"] == STATUS_PASS
    assert _displacement(mesh, crown).max() > 1e-3
