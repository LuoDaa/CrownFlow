# 真实数据与 AI 训练

> 数据实际能干什么、不能干什么 → **`DATASET.md`**（必读，避免高估现有数据）
> 参数真值的来源与建模 → `PARAMETERS.md`
> exocad 文件格式 → `EXOCAD_IMPORTER.md`

---

## 1. 不提供预训练假权重

AI 模式默认关闭，只有设置真实 TorchScript checkpoint 后才开放。
`ml/checkpoints/` 为空。`/api/system/capabilities` 会如实报告
`ai_generator_available: false`。

---

## 2. 当前训练数据状态：0

代码里直接定义（避免"有 15 例数据"这种模糊表述）：

```
PARAMETER_REGRESSION_READY  = true     8 个有效独立病例
GEOMETRY_REGRESSION_LIMITED = true     约 2 例
CROWN_AI_TRAINING_READY     = false    0 例单冠
```

**15 个完整病例里没有任何一例是独立单冠**（最少 2 单位，最多 14 单位），
且最终冠 STL 含内表面、不能作为 `crown_outer_gt` 标签。
详见 `DATASET.md` §3.4。

所以 AI Generator 现在**无法训练**，不是"数据不太够"。

---

## 3. 训练样本规范

每个病例一个目录：

```text
raw/
  case_x/
    upper.stl
    lower.stl
    support.stl
    crown_outer_gt.stl
    margin.json          # 可选但强烈建议：技师确认的 finish line
    meta.json            # {"fdi": 46, "material": "...", "source_order_id": "..."}
```

**`crown_outer_gt.stl` 必须是技师/医生确认的外表面**，
包含内面的最终冠不能直接用作训练目标。

### 采集时必须遵守的三条

1. **按 `source_order_id` 分组**，不按目录。同一订单的多份数据必须整体进入
   训练或验证的同一侧。现有数据里已有一个现成的坑：
   `82404020415薛文爱` 与 `薛文爱` 是同一订单。
   `app/importers/exocad/case_loader.discover_cases()` 已按订单分组。
2. **脱敏在 Import 阶段完成**，不是"读进系统再清理"。
   PHI 分布六层（目录名 / STL 名 / `.matrix4` 名 / `.3ox` 字段 / Notes / `import.log`），
   见 `EXOCAD_IMPORTER.md` §7。
3. **外表面标签要从原软件单独导出**。`.dentalCAD` 是二进制容器、无解析器；
   `.DCM` 全部 CE 加密读不出。

---

## 4. 基线模型

`ml/model.py` 是条件模板形变网络：

**输入**
- 上下颌 + support 点云 context
- FDI 基础模板
- FDI 条件 embedding

**输出**
- 与模板拓扑一致的个性化 Crown Outer 顶点

**当前损失**：Chamfer distance + 模板位移正则

### 为什么用固定拓扑形变而不是直接生成 mesh

固定 topology 带来四个下游必需的性质：

```
vertex ID 不变      cusp ID 不变
region ID 不变      boundary ID 不变
```

于是 AI 输出后，Contact / Occlusion / Intaglio / Thickness 四个引擎可以无缝继续工作。
若 AI 每次吐一个不同拓扑的 STL，后面的 CAD 会非常难接。

这个判断与商业产品的做法一致：AI 只负责 anatomy，牙冠内部交给确定性 CAD。

### 建议继续增加的损失

- normal loss
- curvature loss
- **margin-aware loss**（Margin 是真值里最关键的一项，且必须与 anatomy 耦合）
- contact / antagonist collision loss
- minimum-thickness-aware constraint
- surface smoothness / local feature losses

### 样本规模的现实预期

公开研究中，Margin 线 AI 已使用 1113 份 prepared-tooth surface scans
（1035 训练 / 78 测试，覆盖所有牙位）。Crown Generator 的样本需求不低于此。

建议按牙位、修复类型、材料、扫描设备、技师风格**分层覆盖**，而不是只看"多少病例"：

- PoC：先聚焦 1–2 个后牙牙位（36 / 46 或 16 / 26），建立数百到数千个严格对齐病例
- 扩展：第一/第二磨牙 → 前磨牙 → 前牙；每个牙位维护独立难例与失败集
- 切分：必须按患者/订单切分（见 §3）

---

## 5. 训练流程

```bash
cd ml
pip install -r requirements.txt
python build_dataset.py --raw D:/real_cases --templates D:/templates --out D:/dataset_npz
python train.py --data D:/dataset_npz --out checkpoints/crown_deformer.pt --epochs 200
python export_torchscript.py --checkpoint checkpoints/crown_deformer.pt --out checkpoints/crown_deformer.ts
```

后端启用 AI runtime：

```bash
pip install -r backend/requirements-optional.txt    # 只有 torch 是可选依赖
$env:CROWNFLOW_AI_CHECKPOINT="D:\...\crown_deformer.ts"
```

重启后端，前端会自动开放 AI Generator。

> 这套网络是可运行的条件模板形变基线，不应被描述为达到 exocad/3Shape 商用品质。
> 真实性能取决于病例量、标注/审核质量、模板一致性、坐标规范化、损失函数和临床验证。

---

## 6. 模板库的问题（与 AI 无关但同样阻塞）

`32牙位基础解剖模板库/` 里的 `FDI_XX.stl` **是带牙根的完整牙**，
不是牙冠外形模板。剖面实测 `FDI_36.stl` 高 12.82 mm（真实冠高 6.87 mm）。

`data/templates/` 是空的，README 与 UI 都正确警告过这些文件含牙根。
但 `config.ANATOMY_DIR` 正指向该文件夹、`/api/anatomy` 对外提供它、
文件夹名叫"基础解剖模板库" —— 误用只差一次没有点击标签。

**需要**：导入期模板校验（高度、根部锥度）＋ 把解剖库改名或重新标注，
切断误用路径。现在 `create_design` 只检查模板文件存在，
**从不校验它是不是纯牙冠**。

来源是 Dental Merlin / 3PointX，商用前必须单独确认授权范围。
