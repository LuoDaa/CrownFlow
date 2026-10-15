"""A0 回归：**唯一 Patient Frame Truth**（审计 P0-1）。

三件事必须成立：

1. 确认帧必须右手正交，退化/非正交/左手系一律**拒绝**（不许静默正交化）；
2. 传了 ``patient_frame`` 的放置**全局旋转等变**：``place(R·X) ≈ R·place(X)``；
3. **不传帧**的 legacy 路径**不等变** —— 这条测试正是"world-X 推断"被禁用的理由。
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from app.geometry.crown_outer.patient_frame import (
    PatientFrame, PatientFrameError, confirmed_patient_frame, validate_patient_frame,
)
from app.geometry.registration import initialize_from_margin


def _rot(axis, deg):
    return trimesh.transformations.rotation_matrix(math.radians(deg), axis)[:3, :3]


def _template_and_margin():
    """非对称模板（长方体）+ 偏心椭圆 Margin，便于暴露横向轴的方向错误。

    Margin 跨度与模板跨度的比值必须落在 ``initialize_from_margin`` 的
    0.4–2.5 系数门内，否则会被尺寸门先拦掉（那是另一条路径的检查）。
    """
    template = trimesh.creation.box(extents=(3.0, 2.0, 2.0))
    theta = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    margin = np.c_[2.6 * np.cos(theta), 1.6 * np.sin(theta), np.zeros(48)]
    return template, margin


def _frame(mesial=(1.0, 0.0, 0.0), origin=(0.0, 0.0, 0.0), revision="rev-1"):
    return confirmed_patient_frame(origin, mesial, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                                   revision_id=revision)


# ------------------------------------------------------------ 帧校验

def test_confirmed_frame_accepts_orthonormal_right_handed():
    f = _frame()
    validate_patient_frame(f)
    assert f.revision_id == "rev-1"
    assert np.allclose(f.matrix() @ f.matrix().T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("bad", [
    ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),   # mesial ∥ buccal → 不正交
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.6, 0.8, 0.0)),   # 轴与横向不垂直
    ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),  # 左手系（x×y=-z）
])
def test_confirmed_frame_rejects_degenerate_or_left_handed(bad):
    mesial, buccal, insertion = bad
    with pytest.raises(PatientFrameError):
        confirmed_patient_frame((0, 0, 0), mesial, buccal, insertion)


def test_placement_rejects_frame_whose_axis_differs_from_insertion_axis():
    template, margin = _template_and_margin()
    f = _frame()                                   # 帧 Z = +Z
    with pytest.raises(ValueError, match="patient_frame"):
        initialize_from_margin(template, margin, [1.0, 0.0, 0.0], patient_frame=f)


# ------------------------------------------------------------ 等变性

def test_placement_with_confirmed_frame_is_rotation_equivariant():
    """place(R·X) ≈ R·place(X) —— 确认帧下的放置不含任何 world 轴依赖。"""
    template, margin = _template_and_margin()
    axis = np.array([0.0, 0.0, 1.0])
    frame = _frame()
    base = initialize_from_margin(template, margin, axis, patient_frame=frame)

    R = _rot([0.0, 0.0, 1.0], 37.0)                 # 绕世界 Z 转 37°（换一个 world-X 方向）
    t2 = template.copy(); t2.vertices = np.asarray(t2.vertices) @ R.T
    m2 = np.asarray(margin) @ R.T
    a2 = R @ axis
    f2 = confirmed_patient_frame(R @ np.asarray(frame.origin), R @ frame.x, R @ frame.y, R @ frame.z)
    rotated = initialize_from_margin(t2, m2, a2, patient_frame=f2)

    assert np.allclose(np.asarray(rotated.vertices), np.asarray(base.vertices) @ R.T, atol=1e-6), \
        "确认帧下放置应整体随场景旋转"


def test_legacy_world_x_path_is_not_rotation_equivariant():
    """legacy（不传帧）路径在全局旋转下不等变 —— 这就是 live 必须传帧的原因。"""
    template, margin = _template_and_margin()
    axis = np.array([0.0, 0.0, 1.0])
    with pytest.warns(DeprecationWarning):
        base = initialize_from_margin(template, margin, axis)

    R = _rot([0.0, 0.0, 1.0], 37.0)
    t2 = template.copy(); t2.vertices = np.asarray(t2.vertices) @ R.T
    m2 = np.asarray(margin) @ R.T
    with pytest.warns(DeprecationWarning):
        rotated = initialize_from_margin(t2, m2, R @ axis)

    drift = float(np.abs(np.asarray(rotated.vertices) - np.asarray(base.vertices) @ R.T).max())
    assert drift > 1e-3, "legacy world-X 路径本就应当不等变（漂移 %.6f）" % drift
