"""局部几何特征：面法向、面邻接、脊线锐度、轴向/径向/角向统计。

全部纯几何（只用网格本身 + 就位轴），**不接触 GT 标签**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FaceStat:
    fid: int
    centroid: tuple
    normal: tuple          # 单位面法向（按面顶点序朝向未知 → 统一取“与轴同侧”的朝向）
    area: float
    h: float               # 轴向坐标（相对 frame.origin）
    r: float               # 径向距离
    theta: float           # 角向
    nx: float              # |dot(normal, axis)| ∈ [0,1]：1=垂直于轴（顶/底），0=平行于轴（竖壁）
    sharp_deg: float       # 最大二面角（与相邻面法向夹角，度）
    radial_slope: float    # d(半径)/d(轴向) 的局部估计（用邻居面心符号）


def face_normal(tri):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    u = (bx-ax, by-ay, bz-az)
    v = (cx-ax, cy-ay, cz-az)
    n = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    L = math.sqrt(sum(c*c for c in n)) or 1e-12
    return (n[0]/L, n[1]/L, n[2]/L)


def face_area(tri):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = tri
    u = (bx-ax, by-ay, bz-az)
    v = (cx-ax, cy-ay, cz-az)
    cr = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
    return 0.5 * math.sqrt(sum(c*c for c in cr))


def build_adjacency(tris):
    """面邻接（共享顶点的面视为相邻；网格是三角面片，两三角共边即相邻）。"""
    vid2f = {}
    for fid, t in enumerate(tris):
        for v in t:
            vid2f.setdefault((round(v[0], 5), round(v[1], 5), round(v[2], 5)),
                             []).append(fid)
    adj = {}
    for fid, t in enumerate(tris):
        s = set()
        for v in t:
            for g in vid2f[(round(v[0], 5), round(v[1], 5), round(v[2], 5))]:
                if g != fid:
                    s.add(g)
        adj[fid] = s
    return adj


def build_edge_adjacency(tris):
    """**按边**相邻（共享 2 个顶点）。

    二面角必须按边算：共顶点但只交一点的邻居（例如底盖扇形与壁面）会把
    壁面误标成 84° 锐脊 —— 实测合成件上 `sharp` 因此把整条底盖棱选成了
    Margin。共边才是真正的邻面关系。
    """
    e2f = {}
    for fid, t in enumerate(tris):
        for i in range(3):
            a, b = t[i], t[(i + 1) % 3]
            ka = (round(a[0], 5), round(a[1], 5), round(a[2], 5))
            kb = (round(b[0], 5), round(b[1], 5), round(b[2], 5))
            k = (ka, kb) if ka <= kb else (kb, ka)
            e2f.setdefault(k, []).append(fid)
    adj = {}
    for fids in e2f.values():
        if len(fids) == 2 and fids[0] != fids[1]:
            adj.setdefault(fids[0], set()).add(fids[1])
            adj.setdefault(fids[1], set()).add(fids[0])
    return adj


def face_scale_mm(stats):
    """网格的"面尺度"（中位面面积的等效边长，mm）。

    候选是**面**，面质心位置天然带 ±face_scale/2 的量化；因此 v2 的跳变硬门
    不能只取 GT 的 p99（0.215mm），必须加上这条离散度，否则密集网格上
    "任何环都不可行"（实测 62% 回退）。
    """
    if not stats:
        return 0.0
    a = sorted(s.area for s in stats)
    med = a[len(a) // 2]
    return (2.0 * med) ** 0.5


def face_stats(tris, frame):
    """一次算齐所有面的特征。O(F)·邻接。"""
    adj = build_adjacency(tris)
    eadj = build_edge_adjacency(tris)
    normals = [face_normal(t) for t in tris]
    areas = [face_area(t) for t in tris]
    cents = [((t[0][0]+t[1][0]+t[2][0])/3.0,
              (t[0][1]+t[1][1]+t[2][1])/3.0,
              (t[0][2]+t[1][2]+t[2][2])/3.0) for t in tris]
    stats = []
    for fid in range(len(tris)):
        cen = cents[fid]
        n = normals[fid]
        h, r, th = frame.coords(cen)
        # 二面角：取**共边**邻面法向夹角的最大值（锐边）
        sharp = 0.0
        for g in eadj.get(fid, ()):
            d = abs(sum(n[k]*normals[g][k] for k in range(3)))
            d = 1.0 if d > 1.0 else d
            ang = math.degrees(math.acos(d))
            if ang > sharp:
                sharp = ang
        # 径向斜率留到第二趟（需要全部 h 已知）
        stats.append(FaceStat(fid=fid, centroid=cen, normal=n,
                              area=areas[fid], h=h, r=r, theta=th,
                              nx=abs(sum(n[k]*frame.axis[k] for k in range(3))),
                              sharp_deg=sharp, radial_slope=0.0))
    # 径向斜率需要全部 h 已知后再算一遍（两趟）
    h_of = [s.h for s in stats]
    r_of = [s.r for s in stats]
    for s in stats:
        lo = [(h_of[g], r_of[g]) for g in adj[s.fid] if h_of[g] < s.h - 1e-9]
        if not lo:
            s.radial_slope = 0.0
            continue
        # 往下（h 更小）看：半径是变大还是变小
        dh = sum(s.h - x[0] for x in lo) / len(lo)
        dr = sum(x[1] - s.r for x in lo) / len(lo)
        s.radial_slope = (dr / dh) if dh > 1e-9 else 0.0
    return stats, adj
