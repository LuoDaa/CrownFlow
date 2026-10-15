"""环向平滑 + 重采样（纯几何，不用 GT）。

用**环向移动平均 + 均匀重采样**而不是样条拟合：prep 网格本身有扫描噪声，
M1 v1 先把"点的位置"做对再谈曲线光顺度。Catmull-Rom 等留待 M1 迭代
（本文件即落点）。
"""

from __future__ import annotations

import math


def circular_smooth(points, window):
    """环向移动平均（首尾相接），window 为奇数。"""
    n = len(points)
    if n < 3 or window <= 1:
        return list(points)
    w = min(window, n - 1)
    if w % 2 == 0:
        w -= 1
    half = w // 2
    out = []
    for i in range(n):
        sx = sy = sz = 0.0
        for k in range(-half, half + 1):
            p = points[(i + k) % n]
            sx += p[0]
            sy += p[1]
            sz += p[2]
        c = w
        out.append((sx / c, sy / c, sz / c))
    return out


def resample_closed(points, n_out):
    """按弧长均匀重采样成 n_out 个点（闭环）。"""
    n = len(points)
    if n < 3:
        return list(points)
    seg = []
    total = 0.0
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        L = math.dist(a, b)
        seg.append(L)
        total += L
    if total <= 1e-9:
        return list(points)
    step = total / n_out
    out = []
    i = 0
    acc = 0.0
    for k in range(n_out):
        target = k * step
        while acc + seg[i] < target and i < n - 1:
            acc += seg[i]
            i += 1
        t = (target - acc) / seg[i] if seg[i] > 1e-12 else 0.0
        a, b = points[i], points[(i + 1) % n]
        out.append((a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t,
                    a[2] + (b[2]-a[2])*t))
    return out
