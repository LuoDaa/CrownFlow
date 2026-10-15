"""M1 Geometry Auto Margin：只用几何，不训练模型，不读 GT 生成。

用户 2026-09-30 规格：
    输入 = 代型网格 + 就位轴
    流程 = 识别冠方表面 → 局部几何特征 → 候选带 → 网格上搜索闭环
           → 样条/重采样 → 投影回网格 → 预测 Margin
    GT   = 只用于 evaluate（median/p95/max、Chamfer、Hausdorff、
           within 0.10/0.20mm、闭环成功率、自交）
"""
