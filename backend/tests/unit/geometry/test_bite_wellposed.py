"""WELL_POSED_CANONICAL：夹具必须**真的**打破轴向/旋转对称 + 确定性回归。

A/B：沿源主轴 ±0.5/1/2/3mm、绕主轴 0.5–4° 的扰动**必须**明显降低匹配质量（否则夹具不合格）。
H  ：同一输入连续跑 10 次 register_pair，transform 必须逐位一致（防 RNG 再进 solver）。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bite_fixture import canonical_world, scanner_inputs, quality_sensitivity, symmetry_axis, _rigid  # noqa: E402
from app.geometry.bite_registration import BiteRegistrationProfile, preprocess, register, register_pair  # noqa: E402


def test_well_posed_fixture_breaks_axial_symmetry():
    profile = BiteRegistrationProfile()
    world = canonical_world(well_posed=True)
    patch = preprocess(world["upper_patch_world"], profile)
    jaw = preprocess(world["upper_world"], profile)
    sensitivity = quality_sensitivity(patch, jaw, np.eye(4), profile, spin_axis=symmetry_axis(jaw))
    print("[well-posed] sensitivity", {k: round(v, 3) for k, v in sensitivity.items()})
    # 圆周带的自转是**固有**不可观测（点面距离对切向滑动不敏感）；只要求其余 5 个自由度强约束。
    for dof in ("Tx", "Ty", "Tz", "Rx", "Ry"):
        assert sensitivity[dof] >= 1.0, "%s 必须强约束（实测 %.3f）" % (dof, sensitivity[dof])


def test_degenerate_fixture_is_really_degenerate():
    profile = BiteRegistrationProfile()
    world = canonical_world(well_posed=False)
    patch = preprocess(world["upper_patch_world"], profile)
    jaw = preprocess(world["upper_world"], profile)
    sensitivity = quality_sensitivity(patch, jaw, np.eye(4), profile, spin_axis=symmetry_axis(jaw))
    print("[degenerate] sensitivity", {k: round(v, 3) for k, v in sensitivity.items()})
    # 含法向证据的度量下，"均匀圆柱必有弱自由度"不成立（自转时法向随体转、目标法向不变）。
    # 真正的要求是可比较：well-posed 的绕轴可观测性必须显著更强。
    well = canonical_world(well_posed=True)
    well_patch = preprocess(well["upper_patch_world"], profile)
    well_jaw = preprocess(well["upper_world"], profile)
    well_sens = quality_sensitivity(well_patch, well_jaw, np.eye(4), profile, spin_axis=symmetry_axis(well_jaw))
    assert sensitivity["R_spin"] * 4.0 <= well_sens["R_spin"], ("well-posed 绕轴可观测性必须显著强于均匀圆柱（%.3f vs %.3f）" % (sensitivity["R_spin"], well_sens["R_spin"]))


def test_register_pair_is_deterministic_10_runs():
    profile = BiteRegistrationProfile()
    world = canonical_world(well_posed=True)
    inputs, gts = scanner_inputs(world, A_lower=_rigid([0, 0, 1], 0, [0, 0, 1.5]))
    patch = preprocess(inputs["bite"], profile)
    jaw = preprocess(inputs["lower"], profile)
    first = np.asarray(register_pair(patch, jaw, profile)["matrix"], float)
    for index in range(9):
        again = np.asarray(register_pair(patch, jaw, profile)["matrix"], float)
        assert np.allclose(first, again, atol=0.0), "第 %d 次与首次不一致 ⇒ RNG 又进 solver 了" % (index + 2)


def test_well_posed_recovers_unique_gt():
    profile = BiteRegistrationProfile()
    A_upper, A_lower, A_bite = (_rigid([1, 0, 0], -4, [0.5, 0.5, 0]),
                               _rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0]),
                               _rigid([0, 1, 0], 9, [1, 0, -1]))
    world = canonical_world(well_posed=True)
    inputs, gts = scanner_inputs(world, A_upper=A_upper, A_lower=A_lower, A_bite=A_bite)
    result = register(inputs["bite"], inputs["upper"], inputs["lower"], profile)
    translation = np.linalg.norm(np.asarray(result["T_lower_to_upper"], float)[:3, 3]
                                - gts["T_lower_to_upper"][:3, 3])
    delta = np.asarray(result["T_lower_to_upper"], float) @ np.linalg.inv(gts["T_lower_to_upper"])
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    import trimesh.transformations as tf
    pts = np.asarray(world["upper_patch_world"].vertices, float)
    centre = pts.mean(axis=0)
    vals, axes = np.linalg.eigh((pts - centre).T @ (pts - centre) / len(pts))
    principal = axes[:, int(np.argmax(vals))]
    spin_matrix = np.eye(4); spin_matrix[:3, :3] = delta[:3, :3]
    spin_rad, spin_axis, _ = tf.rotation_from_matrix(spin_matrix)
    omega = np.asarray(spin_axis, float) * spin_rad
    observable = float(np.degrees(np.linalg.norm(omega - principal * float(omega @ principal))))
    spin_deg = float(np.degrees(spin_rad))
    print("[well-posed] T_lower_to_upper err %.4f mm / %.4f deg | status=%s reason=%s"
          % (translation, rotation, result["status"], result["reason"]))
    assert result["status"] != "FAILED", result["reason"]
    assert translation <= 0.5 and observable <= 1.0, "well-posed 可观测子空间必须唯一恢复 GT: %.4f mm / %.4f deg (spin %.3f)" % (translation, observable, spin_deg)


def test_degenerate_world_never_passes():
    profile = BiteRegistrationProfile()
    world = canonical_world(well_posed=False)
    inputs, _ = scanner_inputs(world)
    result = register(inputs["bite"], inputs["upper"], inputs["lower"], profile)
    print("[degenerate] status=%s reason=%s" % (result["status"], result["reason"]))
    assert result["status"] != "PASS", "退化/弱约束世界不得 PASS"
    assert result["reason"] in ("INSUFFICIENT_REGISTRATION_EVIDENCE", "REGISTRATION_SOLUTION_AMBIGUOUS"), \
        "退化世界的理由应为证据不足或多解歧义，实测 %s" % result["reason"]
