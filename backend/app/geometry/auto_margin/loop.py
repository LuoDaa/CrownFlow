"""闭环：按角向排序、检查闭环/自交/环长。

角向扇区法的最大好处是**闭环与不自交由构造保证**（每扇区一个点、按 θ 排序），
所以这里的工作是**核验**而不是修复 —— 核验不过就报 REVIEW/FAILED，不硬凑。
"""

from __future__ import annotations

import math


def order_by_theta(points, frame):
    """按角向排序（闭环顺序），返回 [(theta, p), ...] 排序后的点列。"""
    with_th = [(frame.coords(p)[2], p) for p in points]
    with_th.sort(key=lambda x: x[0])
    return [p for _t, p in with_th]


def theta_gaps(points, frame):
    """相邻点的角向间隔（度）。闭环要求所有间隔都小且均匀。"""
    ths = sorted(frame.coords(p)[2] for p in points)
    n = len(ths)
    if n < 3:
        return []
    gaps = []
    for i in range(n):
        d = ths[(i + 1) % n] - ths[i]
        if d < 0:
            d += 2.0 * math.pi
        gaps.append(math.degrees(d))
    return gaps


def is_closed(points, frame, max_gap_deg):
    gaps = theta_gaps(points, frame)
    return bool(gaps) and max(gaps) <= max_gap_deg


def self_intersections(points, frame):
    """自交检测：把环投到 ⊥ 轴的平面上，检查多边形边是否两两相交。

    只检查**不相邻**的边对；返回相交对数（0 = 干净）。O(N²)，N = 240 时可接受。
    """
    n = len(points)
    if n < 4:
        return 0
    uv = []
    for p in points:
        d = (p[0]-frame.origin[0], p[1]-frame.origin[1], p[2]-frame.origin[2])
        uv.append((d[0]*frame.e1[0]+d[1]*frame.e1[1]+d[2]*frame.e1[2],
                   d[0]*frame.e2[0]+d[1]*frame.e2[1]+d[2]*frame.e2[2]))

    def seg_int(a, b, c, d):
        def cross(o, p, q):
            return (p[0]-o[0])*(q[1]-o[1]) - (p[1]-o[1])*(q[0]-o[0])
        d1, d2 = cross(c, d, a), cross(c, d, b)
        d3, d4 = cross(a, b, c), cross(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

    cnt = 0
    for i in range(n):
        a, b = uv[i], uv[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = uv[j], uv[(j + 1) % n]
            if seg_int(a, b, c, d):
                cnt += 1
    return cnt


def loop_length(points):
    n = len(points)
    if n < 2:
        return 0.0
    tot = 0.0
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        tot += math.dist(a, b)
    return tot
