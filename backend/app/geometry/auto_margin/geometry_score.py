"""候选打分函数 —— M1 要回答的第一个问题："哪个局部几何特征真的指向 Margin？"

Margin 的几何本质：**代型外壁（略呈锥面、法向近径向）与冠方预备体表面交界处的
一条锐脊**。所以候选特征围绕"锐度 / 法向转向 / 半径突变"三类：

    sharp          最大二面角（纯锐度）
    sharp_axis     锐度 × 竖壁权重（1 − nx）　　nx=|dot(n,axis)|
    sharp_down     锐度，但只认"上方比下方更内向"的脊（冠方凸起）
    radius_break   |径向斜率| 的突变强度（外壁→预备体的折点）
    normal_turn    法向与轴夹角的变化率（nx 的局部落差）

把这些都在 **train** 上跑一遍、在 **val** 上比较，是 M1 第一版的核心产出之一；
选中的那个进 `AutoMarginConfig.score` 并冻结。**不许逐例换特征。**
"""

from __future__ import annotations

import math

SCORES = ("sharp", "sharp_axis", "sharp_down", "radius_break", "normal_turn")


def _neighbour_mean(stats, adj, fid, attr):
    ns = adj[fid]
    if not ns:
        return 0.0
    return sum(getattr(stats[g], attr) for g in ns) / len(ns)


def score_of(name):
    if name not in SCORES:
        raise ValueError("unknown score %r（可选 %s）" % (name, SCORES))
    return _IMPL[name]


def _sharp(st, stats, adj):
    return st.sharp_deg


def _sharp_axis(st, stats, adj):
    # 竖壁（nx 小）上的锐脊更可能是 Margin；顶/底盖上的锐边（nx≈1）不是
    return st.sharp_deg * (1.0 - min(1.0, st.nx))


def _sharp_down(st, stats, adj):
    # "上方比下方更内向"—— 预备体在外壁之上收窄；用邻居 h 分层看半径
    dh = 0.6
    up = [stats[g] for g in adj[st.fid] if stats[g].h > st.h + dh]
    dn = [stats[g] for g in adj[st.fid] if stats[g].h < st.h - dh]
    if not up or not dn:
        return 0.0
    r_up = sum(x.r for x in up) / len(up)
    r_dn = sum(x.r for x in dn) / len(dn)
    inward = 1.0 if r_up < r_dn else 0.0
    return st.sharp_deg * inward


def _radius_break(st, stats, adj):
    # 径向斜率的局部落差（折点强度），乘竖壁权重
    m = _neighbour_mean(stats, adj, st.fid, "radial_slope")
    return abs(st.radial_slope - m) * (1.0 - min(1.0, st.nx))


def _normal_turn(st, stats, adj):
    # nx 的局部落差：从"接近平行于轴"过渡到"更斜"的地方
    m = _neighbour_mean(stats, adj, st.fid, "nx")
    return abs(m - st.nx) * 100.0


_IMPL = {
    "sharp": _sharp,
    "sharp_axis": _sharp_axis,
    "sharp_down": _sharp_down,
    "radius_break": _radius_break,
    "normal_turn": _normal_turn,
}
