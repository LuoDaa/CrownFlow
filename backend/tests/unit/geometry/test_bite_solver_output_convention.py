"""§2 solver-output convention tests：证明 **solver 真正输出正确方向**（不只是公式对）。

之前的 convention tests 用的是 GT 矩阵代公式，只能证明"公式自洽"。
本文件用 Canonical World 的独立帧 GT，逐项比对 solver 的三条**输出**：

    T_upper_to_bite_est  vs  GT_T_upper_to_bite = A_bite @ inv(A_upper)
    T_lower_to_bite_est  vs  GT_T_lower_to_bite = A_bite @ inv(A_lower)
    T_lower_to_upper_est vs  GT_T_lower_to_upper = A_upper @ inv(A_lower)

并同时校验刚性契约（RᵀR≈I / det≈+1 / bottom row / 无 scale·shear）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_bite_canonical_world import canonical_world, scanner_inputs, _rigid  # noqa: E402
from app.geometry.bite_registration import register, rigid_transform_report  # noqa: E402

#: 位姿容差（实测 B1.5 在 0.015mm / 0.36° 量级，这里留足余量）
TOL_TRANSLATION_MM = 0.50
TOL_ROTATION_DEG = 1.00


def _pose_error(estimate, ground_truth):
    delta = np.asarray(estimate, float) @ np.linalg.inv(np.asarray(ground_truth, float))
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1))))
    return translation, rotation


def _check(name, result, gts):
    assert result["status"] != "FAILED" or result["T_lower_to_upper"] is not None, \
        "%s：solver 未给出结果（%s / %s）" % (name, result["status"], result["reason"])
    pairs = (("T_upper_to_bite", "T_upper_to_bite"),
             ("T_lower_to_bite", "T_lower_to_bite"),
             ("T_lower_to_upper", "T_lower_to_upper"))
    lines = []
    for key, gt_key in pairs:
        estimate = result.get(key)
        assert estimate is not None, "%s：结果缺 %s" % (name, key)
        te, re = _pose_error(estimate, gts[gt_key])
        report = rigid_transform_report(estimate)
        assert report["rigid_pose_valid"], "%s：%s 违反刚性契约 %s" % (name, key, report)
        assert te <= TOL_TRANSLATION_MM and re <= TOL_ROTATION_DEG, \
            "%s：%s 方向/数值错误 —— translation=%.4f mm rotation=%.4f deg（GT 比对）" % (name, key, te, re)
        lines.append("%s: %.4fmm/%.4fdeg" % (key, te, re))
    print("[solver-output] %-26s %s" % (name, " | ".join(lines)))


def test_solver_output_identity():
    world = canonical_world()
    inputs, gts = scanner_inputs(world)
    _check("identity", register(inputs["bite"], inputs["upper"], inputs["lower"]), gts)


def test_solver_output_translation():
    world = canonical_world()
    inputs, gts = scanner_inputs(world, A_lower=_rigid([0, 0, 1], 0, [0, 0, 2.5]))
    _check("translation", register(inputs["bite"], inputs["upper"], inputs["lower"]), gts)


def test_solver_output_rotation():
    world = canonical_world()
    inputs, gts = scanner_inputs(world, A_lower=_rigid([1, 0, 0], 6, [0, 0, 0]))
    _check("rotation", register(inputs["bite"], inputs["upper"], inputs["lower"]), gts)


def test_solver_output_rigid():
    world = canonical_world()
    inputs, gts = scanner_inputs(world, A_lower=_rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0]),
                                 A_upper=_rigid([1, 0, 0], -4, [0.5, 0.5, 0]), A_bite=_rigid([0, 1, 0], 9, [1, 0, -1]))
    _check("rigid(both-frames)", register(inputs["bite"], inputs["upper"], inputs["lower"]), gts)


def test_solver_output_direction_not_inverted():
    """专门钉住 D1：若每侧矩阵少一次求逆，误差会恰好≈GT 的两倍。"""
    world = canonical_world()
    A_lower = _rigid([0, 0, 1], 0, [0, 0, 2.5])
    inputs, gts = scanner_inputs(world, A_lower=A_lower)
    result = register(inputs["bite"], inputs["upper"], inputs["lower"])
    te, re = _pose_error(result["T_lower_to_upper"], gts["T_lower_to_upper"])
    doubled = float(np.linalg.norm(2.0 * np.asarray(A_lower, float)[:3, 3]))
    assert te < doubled * 0.2, \
        "疑似方向反了（误差 %.4f 接近两倍 GT %.4f）" % (te, doubled)
