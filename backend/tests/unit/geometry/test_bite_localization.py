"""D1 之后的**定位**测试（不下结论，只定位）：
1) pair-level lower registration  —— 绕过 assign_sides / register / QC
2) side assignment provenance     —— 三帧非单位时有没有 patch swap / crop 错
3) SE(3) equivariance (≥20 组)    —— 公共刚体 C 下是否 conjugation 一致

判读规则（用户给的）：
  A. pair-level fail → 修 pair solver / coarse basin
  B. pair-level pass 但 full register fail → 修 side assignment / transform orchestration
  C. equivariance fail → 查 global-Z / normal orientation / PCA sign / axis-based cropping
"""

from __future__ import annotations

import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_bite_canonical_world import canonical_world, scanner_inputs, _rigid  # noqa: E402
from app.geometry.bite_registration import (  # noqa: E402
    BiteRegistrationProfile, assign_sides, preprocess, register, register_pair, rigid_transform_report,
)

TOL_MM, TOL_DEG = 0.50, 1.00


def _pose_error(estimate, ground_truth):
    delta = np.asarray(estimate, float) @ np.linalg.inv(np.asarray(ground_truth, float))
    return (float(np.linalg.norm(delta[:3, 3])),
            float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))))


def _moved(mesh, A):
    out = mesh.copy()
    out.apply_transform(np.asarray(A, float))
    return out


# ---------------------------------------------------------------- 1) pair-level
def test_pair_level_lower_registration_three_frames():
    """直接测 pair solver：lower_patch ↔ lower_jaw（不经过 assign_sides/register/QC）。"""
    profile = BiteRegistrationProfile()
    world = canonical_world()
    A_upper, A_lower, A_bite = (_rigid([1, 0, 0], -4, [0.5, 0.5, 0]),
                               _rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0]),
                               _rigid([0, 1, 0], 9, [1, 0, -1]))
    lower_jaw = _moved(world["lower_world"], A_lower)
    lower_patch_input = _moved(world["lower_patch_world"], A_bite)
    gt = A_bite @ np.linalg.inv(A_lower)                    # T_lower_to_bite

    pair = register_pair(preprocess(lower_patch_input, profile), preprocess(lower_jaw, profile), profile)
    estimate = np.linalg.inv(np.asarray(pair["matrix"], float))   # 契约方向
    te, re = _pose_error(estimate, gt)
    print("[pair-level] lower  T_lower_to_bite: %.4f mm / %.4f deg | rmse=%.4f inlier=%.3f"
          % (te, re, pair["rmse_mm"] or -1, pair["inlier_ratio"] or -1))
    assert rigid_transform_report(estimate)["rigid_pose_valid"]
    assert te <= TOL_MM and re <= TOL_DEG, \
        "A 类：pair-level lower 自己就配错（%.4f mm / %.4f deg）⇒ 修 pair solver/coarse basin" % (te, re)


# ---------------------------------------------------------------- 2) provenance
def test_side_assignment_provenance_three_frames():
    """三帧非单位时，确认没有 patch swap / crop 错：选中的 lower patch 必须**就是** lower 那条。"""
    profile = BiteRegistrationProfile()
    world = canonical_world()
    A_upper = _rigid([1, 0, 0], -4, [0.5, 0.5, 0])
    A_lower = _rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0])
    A_bite = _rigid([0, 1, 0], 9, [1, 0, -1])
    upper_jaw = preprocess(_moved(world["upper_world"], A_upper), profile)
    lower_jaw = preprocess(_moved(world["lower_world"], A_lower), profile)
    upper_patch_true = _moved(world["upper_patch_world"], A_bite)
    lower_patch_true = _moved(world["lower_patch_world"], A_bite)
    bite = preprocess(trimesh.util.concatenate([upper_patch_true, lower_patch_true]), profile)

    sides = assign_sides(bite, upper_jaw, lower_jaw, profile)
    assert not sides.get("ambiguous"), "三帧非单位时被判歧义：%s" % sides.get("scores")
    chosen_upper, chosen_lower = sides["upper"], sides["lower"]
    if chosen_upper is None or chosen_lower is None:
        raise AssertionError("B 类：某一侧没有选中 patch ⇒ side assignment 错")
    # pair report 里没有 .mesh；lower 侧用 sides["lower_patch"] 与真值比 centroid，
    # upper 侧只能靠位姿误差判定（这条就是上一版 harness 写错的地方）。
    chosen_lower_patch = sides.get("lower_patch")
    assert chosen_lower_patch is not None, "B 类：side assignment 没带出 lower patch"
    c_chosen = np.asarray(chosen_lower_patch.vertices, float).mean(axis=0)
    c_truth = np.asarray(lower_patch_true.vertices, float).mean(axis=0)
    distance = float(np.linalg.norm(c_chosen - c_truth))
    print("[provenance] lower patch centroid offset = %.4f mm (chosen=%s)"
          % (distance, sides.get("chosen")))
    assert distance <= 1e-6, "B 类：lower 侧发生 patch swap/crop（centroid 偏 %.4f mm）" % distance
    # 上侧也必须把 upper patch 配到 upper jaw：用 GT 位姿误差判定（无 mesh 可取）
    te_u, re_u = _pose_error(np.linalg.inv(np.asarray(chosen_upper["matrix"], float)),
                             A_bite @ np.linalg.inv(A_upper))
    print("[provenance] upper side pose %.4f mm / %.4f deg" % (te_u, re_u))
    assert te_u <= TOL_MM and re_u <= TOL_DEG, "B 类：upper 侧配错（%.4f/%.4f）" % (te_u, re_u)
    # 且两侧各自的位姿误差都要达标
    for name, pair, gt in (("upper", chosen_upper, A_bite @ np.linalg.inv(A_upper)),
                           ("lower", chosen_lower, A_bite @ np.linalg.inv(A_lower))):
        te, re = _pose_error(np.linalg.inv(np.asarray(pair["matrix"], float)), gt)
        print("[provenance] %s pose %.4f mm / %.4f deg" % (name, te, re))
        assert te <= TOL_MM and re <= TOL_DEG, "B 类：%s 侧配准错（%.4f/%.4f）" % (name, te, re)


# ---------------------------------------------------------------- 3) equivariance
def test_se3_equivariance_common_rigid(n=20):
    """对三个输入同时施加随机公共刚体 C：结果必须满足 T' = C @ T @ inv(C)。"""
    profile = BiteRegistrationProfile()
    world = canonical_world()
    base_inputs, _ = scanner_inputs(world, A_upper=_rigid([1, 0, 0], -4, [0.5, 0.5, 0]),
                                    A_lower=_rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0]),
                                    A_bite=_rigid([0, 1, 0], 9, [1, 0, -1]))
    base = register(base_inputs["bite"], base_inputs["upper"], base_inputs["lower"], profile)
    base_t = base.get("T_lower_to_upper")
    assert base_t is not None, "基线没给出结果：%s/%s" % (base["status"], base["reason"])

    rng = np.random.default_rng(2026)
    worst = (0.0, 0.0)
    for index in range(n):
        axis = rng.normal(size=3)
        C = _rigid(axis, rng.uniform(-60, 60), rng.uniform(-25, 25, 3))
        case = register(_moved(base_inputs["bite"], C), _moved(base_inputs["upper"], C),
                        _moved(base_inputs["lower"], C), profile)
        est = case.get("T_lower_to_upper")
        if est is None:
            print("[equivariance] #%02d 无结果 %s/%s" % (index, case["status"], case["reason"]))
            worst = (float("inf"), float("inf"))
            continue
        expected = C @ np.asarray(base_t, float) @ np.linalg.inv(C)
        te, re = _pose_error(est, expected)
        worst = (max(worst[0], te), max(worst[1], re))
    print("[equivariance] C-rigid ×%d  worst: %.4f mm / %.4f deg" % (n, worst[0], worst[1]))
    assert worst[1] <= TOL_DEG and worst[0] <= TOL_MM, \
        "C 类：SE(3) 等变失败（worst %.4f mm / %.4f deg）⇒ 查 global-Z / normal orientation / PCA sign / axis crop" \
        % (worst[0], worst[1])
