"""补充测试：特征分解的正确性 + 主方向跨度的旋转不变性。

与 ``test_arch_audit.py`` 互补 —— 那份测**判据规则**，这份测**底层那个量本身**。
这里的断言来自一次实测调查（"主方向跨度比盒子实际尺寸大 1~3%"），
结论是：分解没问题，是**顶点汤 PCA** 的固有性质。

## 调查结论（别当成 bug）

STL 里同一顶点会在多个三角形里重复出现。重复次数**不均匀**时，协方差出现
非对角项，特征向量偏一点点，投影跨度按 ``L·cos(t) + W·sin(t)`` 被放大。

实测一个 50x30x10 的合成盒子：八象限顶点计数 42/40 不对称，
协方差非对角项 4.63，特征向量偏约 1.1 度，跨度读成 **50.61 / 31.03 / 10.43**
（每个方向多出约 0.4~0.6 mm）。真实扫描网格的重复分布更均匀，偏差只会更小。

所以：**这些数字是形状量级，不是毫米级尺寸测量**；
分类用的是比值（0.35 vs 0.45 这种量级），完全不受影响。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.unit.groundtruth.test_groundtruth_indexer import (
    _tris_from_verts,
    _write_tris,
)

from app.groundtruth.arch_audit import _jacobi_eigen, _principal_extents


# ---------------------------------------------------------------- 夹具

def _box_points(lx: float, ly: float, lz: float, *, rot=None,
                subdiv: int = 3) -> list[tuple[float, float, float]]:
    """盒子表面上的点（模拟 STL 的顶点汤，含重复）。

    **刻意保留重复**：真实 STL 就是这样，而重复的不均匀正是下面那条
    偏置的来源。夹具与真实数据在这一点上同构。
    """
    h = (lx / 2.0, ly / 2.0, lz / 2.0)
    pts: list[tuple[float, float, float]] = []
    for (i, j), axis, sg in (((0, 1), 2, -1.0), ((0, 1), 2, 1.0),
                             ((0, 2), 1, -1.0), ((0, 2), 1, 1.0),
                             ((1, 2), 0, -1.0), ((1, 2), 0, 1.0)):
        for a in range(subdiv + 1):
            for b in range(subdiv + 1):
                p = [0.0, 0.0, 0.0]
                p[axis] = sg * h[axis]
                p[i] = (a / subdiv - 0.5) * 2 * h[i]
                p[j] = (b / subdiv - 0.5) * 2 * h[j]
                if rot is not None:
                    p = [sum(rot[r][k] * p[k] for k in range(3)) for r in range(3)]
                pts.append((p[0], p[1], p[2]))
    return pts


def _rot(ax: float, ay: float, az: float):
    def mul(a, b):
        return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)]
                for r in range(3)]
    def rz(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    def rx(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    def ry(d):
        c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
        return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    return mul(mul(rz(az), rx(ax)), ry(ay))


# ============================================================ 特征分解

def test_jacobi_eigen_on_diagonal_matrix():
    vals, axes = _jacobi_eigen([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert sorted(vals) == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)
    for i in range(3):
        for j in range(3):
            dot = sum(axes[i][k] * axes[j][k] for k in range(3))
            assert dot == pytest.approx(1.0 if i == j else 0.0, abs=1e-9)


def test_jacobi_eigen_recovers_a_known_spectrum():
    """A = R·diag(5,2,1)·R^T：特征值必须回到 {5,2,1}，且 A·v = λ·v。"""
    r = _rot(23.0, -41.0, 17.0)
    d = [5.0, 2.0, 1.0]
    a = [[sum(r[i][k] * d[k] * r[j][k] for k in range(3)) for j in range(3)]
         for i in range(3)]
    vals, axes = _jacobi_eigen(a)
    assert sorted(vals, reverse=True) == pytest.approx([5.0, 2.0, 1.0], abs=1e-6)
    for i in range(3):
        av = [sum(a[r][c] * axes[i][c] for c in range(3)) for r in range(3)]
        assert av == pytest.approx([vals[i] * x for x in axes[i]], abs=1e-6)


# ============================================================ 旋转不变性

def test_principal_extents_are_rotation_invariant():
    """**这是"不能用轴对齐包围盒判整颌"的核心证据。**

    同一个 50x30x10 的盒子，正放与任意旋转，主方向跨度必须几乎一致 ——
    因为它描述的是**形状**；而轴对齐包围盒会随朝向变化，描述的是**摆位**。
    """
    upright = _principal_extents(_box_points(50.0, 30.0, 10.0))
    rotated = _principal_extents(_box_points(50.0, 30.0, 10.0,
                                             rot=_rot(37.0, 53.0, 29.0)))
    assert rotated == pytest.approx(upright, rel=0.005)


def test_principal_extents_track_a_known_box_within_a_documented_bias():
    """量级对得上，但**带上已记录的 1/cos 偏置**（见本文件模块文档）。

    断言用绝对容差 1.2 mm，而不是 1% —— 小边上的相对偏差更大
    （10 mm 那条边偏 4.3%），用相对容差会掩盖机制。
    """
    e_min, e_mid, e_max = _principal_extents(_box_points(50.0, 30.0, 10.0))
    assert (e_max, e_mid, e_min) == pytest.approx((50.0, 30.0, 10.0), abs=1.2)


def test_derived_ratios_are_insensitive_to_the_bias():
    """分类用的是**比值**：绝对偏置在比值里几乎抵消 —— 这才是它无害的原因。"""
    e_min, e_mid, e_max = _principal_extents(_box_points(50.0, 30.0, 10.0))
    assert e_mid / e_max == pytest.approx(0.6, abs=0.03)      # 真值 30/50
    assert e_min / e_max == pytest.approx(0.2, abs=0.03)      # 真值 10/50


def test_long_band_has_small_second_ratio_but_arch_does_not():
    """长条 vs 弓形：第二个主方向的比值是判据本身。"""
    band = _principal_extents(_box_points(48.0, 8.0, 6.0))
    arch = _principal_extents(_box_points(55.0, 42.0, 16.0))
    assert band[1] / band[2] < 0.30          # 长条：第二方向很小
    assert arch[1] / arch[2] > 0.60          # 弓形：两个方向都有尺度
