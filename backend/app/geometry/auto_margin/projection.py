"""把预测点投回网格表面。

与 ``tools/build_auto_margin_gt.py`` 里的 ``closest_on_tri`` **算法相同**
（Ericson §5.1.5，含退化三角形保护）。这里独立实现一份，是因为 ``tools/``
不是包、且 M1 要被引擎调用；两处行为由合成用例逐点比对钉住（见单测）。
"""

from __future__ import annotations


def _seg_nearest(p, a, b):
    """点到线段最近点 → (点, 距离²)。退化三角形与兜底都用它。"""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    d2 = ab[0]*ab[0] + ab[1]*ab[1] + ab[2]*ab[2]
    if d2 < 1e-20:
        return a, (p[0]-a[0])**2 + (p[1]-a[1])**2 + (p[2]-a[2])**2
    t = ((p[0]-a[0])*ab[0] + (p[1]-a[1])*ab[1] + (p[2]-a[2])*ab[2]) / d2
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    q = (a[0]+ab[0]*t, a[1]+ab[1]*t, a[2]+ab[2]*t)
    return q, (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2


def closest_on_tri(p, a, b, c):
    """点到三角形最近点 → (点, 距离²)。退化三角形退化为“投影到最长边”。"""
    ab = (b[0]-a[0], b[1]-a[1], b[2]-a[2])
    ac = (c[0]-a[0], c[1]-a[1], c[2]-a[2])
    nn = (ab[1]*ac[2]-ab[2]*ac[1], ab[2]*ac[0]-ab[0]*ac[2],
          ab[0]*ac[1]-ab[1]*ac[0])
    if nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2] < 1e-18:
        return min((_seg_nearest(p, a, b), _seg_nearest(p, b, c),
                    _seg_nearest(p, c, a)), key=lambda x: x[1])
    ap = (p[0]-a[0], p[1]-a[1], p[2]-a[2])
    d1 = ab[0]*ap[0] + ab[1]*ap[1] + ab[2]*ap[2]
    d2 = ac[0]*ap[0] + ac[1]*ap[1] + ac[2]*ap[2]
    if d1 <= 0.0 and d2 <= 0.0:
        return a, ap[0]*ap[0] + ap[1]*ap[1] + ap[2]*ap[2]
    bp = (p[0]-b[0], p[1]-b[1], p[2]-b[2])
    d3 = ab[0]*bp[0] + ab[1]*bp[1] + ab[2]*bp[2]
    d4 = ac[0]*bp[0] + ac[1]*bp[1] + ac[2]*bp[2]
    if d3 >= 0.0 and d4 <= d3:
        return b, bp[0]*bp[0] + bp[1]*bp[1] + bp[2]*bp[2]
    vc = d1*d4 - d3*d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        den = d1 - d3
        t = (d1/den) if abs(den) > 1e-20 else 0.0
        q = (a[0]+ab[0]*t, a[1]+ab[1]*t, a[2]+ab[2]*t)
        return q, (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
    cp = (p[0]-c[0], p[1]-c[1], p[2]-c[2])
    d5 = ab[0]*cp[0] + ab[1]*cp[1] + ab[2]*cp[2]
    d6 = ac[0]*cp[0] + ac[1]*cp[1] + ac[2]*cp[2]
    if d6 >= 0.0 and d5 <= d6:
        return c, cp[0]*cp[0] + cp[1]*cp[1] + cp[2]*cp[2]
    vb = d5*d2 - d1*d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        den = d2 - d6
        t = (d2/den) if abs(den) > 1e-20 else 0.0
        q = (a[0]+ac[0]*t, a[1]+ac[1]*t, a[2]+ac[2]*t)
        return q, (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
    va = d3*d6 - d5*d4
    if va <= 0.0 and (d4-d3) >= 0.0 and (d5-d6) >= 0.0:
        den = (d4-d3) + (d5-d6)
        t = ((d4-d3)/den) if abs(den) > 1e-20 else 0.0
        q = (b[0]+(c[0]-b[0])*t, b[1]+(c[1]-b[1])*t, b[2]+(c[2]-b[2])*t)
        return q, (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
    s = va + vb + vc
    if abs(s) < 1e-20:
        return min((_seg_nearest(p, a, b), _seg_nearest(p, b, c),
                    _seg_nearest(p, c, a)), key=lambda x: x[1])
    q = ((a[0]*va + b[0]*vb + c[0]*vc)/s,
         (a[1]*va + b[1]*vb + c[1]*vc)/s,
         (a[2]*va + b[2]*vb + c[2]*vc)/s)
    return q, (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2


class MeshIndex:
    """均匀网格加速的最近点查询（纯 stdlib）。

    M1 每例要投影 ~240 个点，而每点都要在 2~4 万面里找最近点；均匀网格把
    O(N) 降到"邻格内搜索"。若某圈已找到点且格距超过当前最优距离即停。
    """

    def __init__(self, tris, cell_mm=None):
        self.tris = tris
        xs = [v[0] for t in tris for v in t]
        ys = [v[1] for t in tris for v in t]
        zs = [v[2] for t in tris for v in t]
        self.lo = (min(xs), min(ys), min(zs))
        diag = ((max(xs)-self.lo[0])**2 + (max(ys)-self.lo[1])**2
                + (max(zs)-self.lo[2])**2) ** 0.5
        self.cell = float(cell_mm or max(0.5, diag / 40.0))
        self.bins = {}
        for fid, t in enumerate(tris):
            c = (sum(v[0] for v in t) / 3.0, sum(v[1] for v in t) / 3.0,
                 sum(v[2] for v in t) / 3.0)
            self.bins.setdefault(self._key(c), []).append(fid)

    def _key(self, p):
        return (int((p[0]-self.lo[0]) // self.cell),
                int((p[1]-self.lo[1]) // self.cell),
                int((p[2]-self.lo[2]) // self.cell))

    def nearest(self, p, max_rings=8):
        """→ (最近点, 距离, 面 id)。"""
        k0 = self._key(p)
        best = (None, float("inf"), None)
        for ring in range(max_rings):
            for dx in range(-ring, ring + 1):
                for dy in range(-ring, ring + 1):
                    for dz in range(-ring, ring + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != ring:
                            continue
                        for fid in self.bins.get((k0[0]+dx, k0[1]+dy,
                                                  k0[2]+dz), ()):
                            t = self.tris[fid]
                            q, d2 = closest_on_tri(p, t[0], t[1], t[2])
                            if d2 < best[1]:
                                best = (q, d2, fid)
            if best[2] is not None and ring * self.cell > best[1] ** 0.5:
                break
        return best[0], best[1] ** 0.5, best[2]


def project_points(points, index):
    """把点列投回网格 → (投影点列, 每点位移)。"""
    out, disp = [], []
    for p in points:
        q, d, _fid = index.nearest(p)
        if q is None:
            out.append(p)
            disp.append(0.0)
        else:
            out.append(q)
            disp.append(d)
    return out, disp


def point_to_polyline_dist(p, poly, closed=True):
    """点到折线（默认闭环）的最短距离。"""
    n = len(poly)
    if n < 2:
        return float("inf")
    best = float("inf")
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n] if closed else poly[min(i + 1, n - 1)]
        _q, d2 = _seg_nearest(p, a, b)
        if d2 < best:
            best = d2
    return best ** 0.5
