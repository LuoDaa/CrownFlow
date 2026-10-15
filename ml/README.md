# CrownFlow AI Training

这里没有任何伪造权重。

AI 模式的真实数据最少应包含：
- upper.stl
- lower.stl
- support.stl（Prep 或 Abutment）
- crown_outer_gt.stl（经确认的牙冠外表面；不能直接用包含内面的最终冠 STL）
- meta.json（至少包含 FDI）

流程：

```bash
pip install -r requirements.txt
python build_dataset.py --raw D:/real_cases --templates D:/templates --out D:/dataset_npz
python train.py --data D:/dataset_npz --out checkpoints/crown_deformer.pt --epochs 200
python export_torchscript.py --checkpoint checkpoints/crown_deformer.pt --out checkpoints/crown_deformer.ts
```

然后后端安装 AI runtime：

```bash
pip install -r backend/requirements-ai.txt
```

设置：

```powershell
$env:CROWNFLOW_AI_CHECKPOINT="D:\...\crown_deformer.ts"
```

重启后端，前端会自动开放 AI Generator。

> 这套网络是可运行的条件模板形变基线，不应被描述为达到 exocad/3Shape 商用品质。真实性能取决于病例量、标注/审核质量、模板一致性、坐标规范化、损失函数和临床验证。
