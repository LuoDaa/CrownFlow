"""合成"预备体 + 真值外冠"夹具：给 **live 非平面链路**一个它能接受的输入。

**为什么需要它**（Q0 结论，2026-10-12）：验收套件原先用 FDI36 走
``trim_outer_at_margin``（平面裁切）+ ``enforce_clearance``（自动补厚）+ 体素布尔的
**退役链路**。现在：

- 平面裁切对非平面 Margin 已被守卫禁止（``MarginReconstructionRequired``，位置正确：
  非平面 Margin 用平面裁切会静默产出错误几何）；
- 而 sanctioned 的 live 链路（``fitting_support_above_margin`` → ``build_nonplanar_crown``）
  在现有真实数据上**跑不起来**，且拒绝理由是对的：
  * FDI36（种植基台，``1629-王煜 0.stl``）：Margin 以上不是单一完整就位面（螺丝通道开口）；
  * FDI46 病例包 ``cf_b1b73bae221211b1``：``prep.stl`` 的开放边界跨过 Margin；
    ``die.stl`` 与 FDI36 同因（非单一完整就位面）。

所以"链路不变量"（水密 / 单连通 / 共享颈缘缝 / 外冠形态不被改写）需要一个
**闭合预备体 + 真值外冠**的可执行输入。这就是本模块。

**它不冒充病例精度**：这里只断言链路自身的几何不变量与体积量级；
真实病例精度断言仍留在 FDI36（就位方向、倒角/间隙分布等）与各自的验收项里。
"""

from __future__ import annotations

import numpy as np
import trimesh

#: 合成链路的工程参数（ENGINEERING_ONLY：未经制造放行，只用于几何不变量）
SYNTHETIC_PARAMS: dict = {
    "machine_axes": 5,
    "cement_gap_mm": 0.06,
    "cement_gap_start_mm": 0.5,
    "cement_gap_top_mm": None,
    "margin_band_mm": 0.8,
    "transition_mm": 0.8,
    "untouchable_zone_mm": 0.0,
    "minimum_thickness_mm": 0.6,
}


def prep_case(*, amplitude_mm: float = 0.6, sections: int = 96, radius_mm: float = 3.0):
    """返回 ``dict(support, gt_crown, margin, axis, params, describe)``。

    - ``support``：闭合圆柱预备体（单一完整就位面，无开放边界）——live 链路能接受；
    - ``margin``：贴在预备体侧壁上的**非平面**闭环（``amplitude_mm`` 的双波起伏），
      使链路走的是三维颈缘缝而不是平面裁切；
    - ``gt_crown``：包住 Margin 以上预备体的"真值外冠"（闭合、单连通），
      其下缘略低于 Margin —— 与真实 GT 冠（下缘即 Margin）同构，
      所以"裁掉的部分"与 FDI36 指标里的量级可比。
    """
    axis = np.array([0.0, 0.0, 1.0])
    support = trimesh.creation.cylinder(radius=radius_mm, height=5.5, sections=sections)
    support.apply_translation([0, 0, 0.75])          # z ∈ [-2.0, 4.75]
    theta = np.arange(sections) * 2 * np.pi / sections
    margin = np.c_[radius_mm * np.cos(theta), radius_mm * np.sin(theta),
                   amplitude_mm * np.sin(2 * theta)]

    # 外冠必须**高于**内腔（否则内腔会从顶面开出第二道边界，布尔主体无法分成两个单颈环面）。
    # 下缘贴在 Margin 附近：这样"被裁掉的量"与真实 GT 冠（下缘即 Margin）量级可比。
    height = 5.65
    crown = trimesh.creation.cylinder(radius=radius_mm + 1.0, height=height, sections=sections)
    crown.apply_translation([0, 0, height / 2 - 0.05])   # z ∈ [-0.05, 5.60]
    return {
        "support": support,
        "gt_crown": crown,
        "margin": margin,
        "axis": axis,
        "params": dict(SYNTHETIC_PARAMS),
        "describe": {
            "kind": "synthetic_prep_nonplanar_margin",
            "amplitude_mm": float(amplitude_mm),
            "support_faces": int(len(support.faces)),
            "gt_crown_faces": int(len(crown.faces)),
            "margin_points": int(len(margin)),
            "margin_plane_deviation_max_mm": float(amplitude_mm),
            "note": "合成几何：只用于链路不变量（水密/单连通/共享颈缘缝/形态保持/体积量级）。",
        },
    }
