"""QC：闭环性、半径突变、自交、投影位移、环长。

**这些判据都不看 GT**：它们回答"这条预测曲线本身像不像一条 Margin"，而不是
"它离 GT 多近"（后者只在 evaluate 里出现）。用户规格里"错得离谱的 Margin 绝不
能被自动接受"就靠这里 + 后续 confidence 模块。
"""

from __future__ import annotations

import math

from . import loop as loop_mod
from .models import STATUS_FAILED, STATUS_OK, STATUS_REVIEW


def assess(points, frame, config, raw_points=None, disp=None):
    """→ (status, quality dict, notes)

    ``points`` 为投影+重采样后的闭环点列；``raw_points`` 是平滑前（用于报告
    扇区覆盖率）；``disp`` 是投影位移。
    """
    q = {}
    notes = []
    n = len(points)
    q["n_points"] = n
    if n < 12:
        return STATUS_FAILED, q, ["too_few_points=%d" % n]
    # 扇区覆盖（用平滑前的点算，反映搜索是否每扇区都有候选）
    if raw_points:
        q["n_sectors_filled"] = len(raw_points)
        q["filled_sector_frac"] = round(
            len(raw_points) / float(config.n_sectors), 4)
        if q["filled_sector_frac"] < config.min_filled_sector_frac:
            notes.append("sparse_sectors=%.2f" % q["filled_sector_frac"])
    # 闭环：角向间隔不应超过名义间隔的 3 倍
    nominal = 360.0 / config.n_sectors
    gaps = loop_mod.theta_gaps(points, frame)
    q["max_gap_deg"] = round(max(gaps), 3) if gaps else None
    closed = bool(gaps) and max(gaps) <= 3.0 * nominal
    q["closed"] = closed
    if not closed:
        notes.append("loop_gap=%.2fdeg" % (max(gaps) if gaps else -1))
    # 半径突变（相邻重采样点）
    rs = [frame.coords(p)[1] for p in points]
    jumps = [abs(rs[i] - rs[(i + 1) % n]) for i in range(n)]
    q["max_radius_jump_mm"] = round(max(jumps), 4)
    if max(jumps) > config.max_radius_jump_mm:
        notes.append("radius_jump=%.2fmm" % max(jumps))
    # 自交
    n_int = loop_mod.self_intersections(points, frame)
    q["n_self_intersections"] = n_int
    if n_int:
        notes.append("self_intersections=%d" % n_int)
    # 投影位移（预测点被投回表面的距离，反映"曲线是否贴着网格"）
    if disp:
        q["proj_disp_p95_mm"] = round(
            sorted(disp)[min(len(disp) - 1, int(.95 * (len(disp) - 1)))], 4)
        q["proj_disp_max_mm"] = round(max(disp), 4)
    q["length_mm"] = round(loop_mod.loop_length(points), 3)
    q["radius_min_mm"] = round(min(rs), 3)
    q["radius_max_mm"] = round(max(rs), 3)
    status = STATUS_OK if not notes else STATUS_REVIEW
    return status, q, notes
