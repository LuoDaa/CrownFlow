"""就位方向的兼容入口。

真正的实现在 :mod:`app.geometry.insertion`（``solve_insertion_axis``）。
本模块保留旧的 ``suggest_insertion_axis`` 签名，使既有调用点自动获得修复。

旧实现直接对整块 support 顶点做 PCA 取主轴，实测在真实颈缘上稳定偏 60.9–64.2°：
底部、颈缘与扫描裁切形状都会把主轴拉歪，而且顶点分布疏密不均会进一步偏置质心。
新实现改用「Margin 拟合平面法向 + 侧壁法向拟合轴」并用锥内搜索优化，详见该模块文档。
"""

from __future__ import annotations

from .insertion import (  # noqa: F401
    DEFAULT_MAX_DEVIATION_DEG,
    AxisStatus,
    InsertionAxisResult,
    angle_deg,
    estimate_wall_axis,
    solve_insertion_axis,
    suggest_insertion_axis,
    validate_margin,
)

__all__ = [
    "suggest_insertion_axis",
    "solve_insertion_axis",
    "InsertionAxisResult",
    "AxisStatus",
    "validate_margin",
    "estimate_wall_axis",
    "angle_deg",
    "DEFAULT_MAX_DEVIATION_DEG",
]
