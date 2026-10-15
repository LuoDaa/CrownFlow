"""代型局部坐标系：就位轴 + 轴向/径向/角向坐标。

轴由调用方给出（来自 constructionInfo / modelInfo 的 InsertionAxis）——
**不做“猜测方向”**（用户规格：轴是输入，不是待估计量）。原点取网格质心，
两个径向基由轴正交化得到。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


@dataclass(frozen=True)
class DieFrame:
    origin: tuple
    axis: tuple
    e1: tuple
    e2: tuple

    def coords(self, p):
        """→ (轴向 h, 径向 r, 角向 theta)"""
        d = (p[0] - self.origin[0], p[1] - self.origin[1], p[2] - self.origin[2])
        h = d[0]*self.axis[0] + d[1]*self.axis[1] + d[2]*self.axis[2]
        u = d[0]*self.e1[0] + d[1]*self.e1[1] + d[2]*self.e1[2]
        v = d[0]*self.e2[0] + d[1]*self.e2[1] + d[2]*self.e2[2]
        r = math.hypot(u, v)
        th = math.atan2(v, u) % (2.0 * math.pi)
        return h, r, th


def build_frame(centroids, axis) -> DieFrame:
    """由面质心集合与轴建立坐标系（原点 = 质心均值，纯几何，无 GT）。"""
    ax = _unit(axis)
    n = len(centroids)
    o = (sum(c[0] for c in centroids) / n,
         sum(c[1] for c in centroids) / n,
         sum(c[2] for c in centroids) / n)
    # 取与轴不平行的参考方向做正交化
    ref = (1.0, 0.0, 0.0) if abs(ax[0]) < 0.9 else (0.0, 1.0, 0.0)
    dot = ref[0]*ax[0] + ref[1]*ax[1] + ref[2]*ax[2]
    e1 = _unit((ref[0] - dot*ax[0], ref[1] - dot*ax[1], ref[2] - dot*ax[2]))
    e2 = (ax[1]*e1[2] - ax[2]*e1[1],
          ax[2]*e1[0] - ax[0]*e1[2],
          ax[0]*e1[1] - ax[1]*e1[0])
    return DieFrame(origin=o, axis=ax, e1=e1, e2=e2)
