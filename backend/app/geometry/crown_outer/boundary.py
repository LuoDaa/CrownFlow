"""闭环对应（Boundary Correspondence，方案 §8）。

两条闭环（模板牙颈环 / 患者 Margin）点数不同、没有天然"第 0 点"、
方向也可能相反。三步：

1. **保序**：输入点序若不是沿环走的（相邻段跳变），先用贪近邻链重排 ——
   constructionInfo 的 Margin 通常是有序的，但这一步是防线的最低成本。
2. **弧长重采样**：按累计弧长归一化到 [0,1)，重采样到固定 N 点
   （首版 256，只是算法工作分辨率，不覆盖原始几何）。
3. **起点与方向**：同时测试 forward / reversed × 全部 cyclic shift，
   取初始匹配误差最小者：

       E(k, dir) = mean( ||T_i − M_((i+k) mod N)||² )

   v0 用稳定的 arc-length + 循环穷举；切向一致性/曲率项留给第二版。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BoundaryError(ValueError):
    pass


# ---------------------------------------------------------------- 保序

def ensure_closed_ring_order(points: np.ndarray, *, copy: bool = True
                             ) -> np.ndarray:
    """若点序不是沿环走的，用贪近邻链重排成沿环的序。

    判据：有序闭环的相邻段长应与周长/N 同量级；若 max(段长) > 0.25×周长，
    认为序被打乱（随机序的相邻跳变会接近环直径）。贪近邻从第 0 点出发，
    对凸/近似凸的颈缘足够；不完美处由重采样+形变吸收。
    """
    p = np.asarray(points, dtype=np.float64)
    if copy:
        p = p.copy()
    if len(p) < 8:
        raise BoundaryError("闭环点数少于 8")
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    per = float(seg.sum())
    if per < 1e-9:
        raise BoundaryError("闭环周长为 0（重合点？）")
    if seg.max() <= 0.25 * per:
        return p                       # 已经沿环
    # 贪近邻重排
    n = len(p)
    used = np.zeros(n, dtype=bool)
    order = [0]
    used[0] = True
    for _ in range(n - 1):
        cur = p[order[-1]]
        d = np.linalg.norm(p - cur, axis=1)
        d[used] = np.inf
        nxt = int(np.argmin(d))
        order.append(nxt)
        used[nxt] = True
    return p[order]


# ---------------------------------------------------------------- 重采样

def ring_arc_params(points: np.ndarray) -> np.ndarray:
    """有序闭环各点的归一化弧长参数 ∈ [0,1)（第 0 点为 0）。"""
    p = np.asarray(points, dtype=np.float64)
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    per = float(seg.sum())
    if per < 1e-9:
        raise BoundaryError("闭环周长为 0")
    cum = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    return cum / per


def resample_ring(points: np.ndarray, n: int) -> np.ndarray:
    """有序闭环按弧长等距重采样到 n 点。"""
    p = ensure_closed_ring_order(points)
    params = ring_arc_params(p)
    u = np.arange(n) / n
    out = np.empty((n, 3))
    m = len(p)
    for dim in range(3):
        # 周期插值：末段从 p[-1] 回到 p[0]
        out[:, dim] = np.interp(
            u, np.append(params, 1.0),
            np.append(p[:, dim], p[0, dim]))
    return out


# ---------------------------------------------------------------- 对应

@dataclass
class Correspondence:
    shift: int                 # 患者 resampled 环的循环移位
    reversed_: bool            # 是否取了反向
    mean_error_mm2: float      # 最优 (shift, dir) 的初始匹配均方误差
    template_params: np.ndarray   # (N,) 模板重采样点参数
    patient_targets: np.ndarray   # (N,3) **已对齐**的患者重采样点


def align_rings(template_ring: np.ndarray, patient_ring: np.ndarray, n: int
                ) -> Correspondence:
    """两条闭环 → 重采样 + 最优 (方向, shift)，返回已对齐的目标。

    ``patient_targets[i]`` 就是模板环上参数 i/N 处应该去的点。
    """
    t = resample_ring(template_ring, n)
    p0 = resample_ring(patient_ring, n)
    best = None
    for rev in (False, True):
        p = p0[::-1].copy() if rev else p0
        # E(k) = mean_i ||t_i − p_((i+k) mod N)||²，对全部 k 穷举。
        # N=256 时 O(N²·3)≈20 万次乘加，直接算比 FFT 卷积更透明且不易错位。
        sum_t = float(np.sum(t * t))
        sum_p = float(np.sum(p * p))
        for k in range(n):
            err = (sum_t + sum_p
                   - 2.0 * float(np.sum(t * np.roll(p, -k, axis=0)))) / n
            if best is None or err < best.mean_error_mm2:
                best = Correspondence(
                    shift=k, reversed_=rev, mean_error_mm2=err,
                    template_params=np.arange(n) / n,
                    patient_targets=np.roll(p, -k, axis=0))
    return best


def targets_for_loop_vertices(loop_ordered_points: np.ndarray,
                              corr: Correspondence) -> np.ndarray:
    """把对齐后的重采样目标插值回**原始环顶点**（方案 §8.3 的"重采样目标点"）。

    每个原始环顶点按其弧长参数在已对齐的目标环上线性插值，
    得到逐顶点的 3D 约束目标 —— 避免离散点一一锁死造成的局部抖动。
    """
    params = ring_arc_params(np.asarray(loop_ordered_points, dtype=np.float64))
    n = len(corr.patient_targets)
    out = np.empty((len(params), 3))
    for dim in range(3):
        out[:, dim] = np.interp(params, corr.template_params,
                                corr.patient_targets[:, dim])
    return out
