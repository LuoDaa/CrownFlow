# 系统架构

## 1. 总体链路

```text
exocad 证据数据（.constructionInfo / .dentalProject / .3ox / .matrix4）
        |
        v
  exocad Importer ──→ 四态参数模型 ──→ Lab Profile（由数据推导分类）
        |                                        |
        |                                        v
        |                              Parameter Resolver（固定优先级）
        |                                        |
        v                                        v
Upper / Lower STL  +  Prep / Abutment Support  +  Target FDI  +  解析后的参数集
        |                                        |
        v                                        |
   Support / Margin                              |
        |                                        |
        v                                        |
  Insertion Axis Solver  ──→ 状态机（OK / REVIEW_REQUIRED / FAILED）
        |                                        |
        v                                        |
  DesignRevision 链（不可变，人工结果不被静默覆盖）
        |                                        |
        v                                        v
  Crown Outer Generator  ──────────────────→  Margin Trim
  |- Template + Kabsch + RBF                      |
  `- AI conditional generator（真实 checkpoint）   |
        |                                        v
        |                                 Support Blockout        🔴 已知缺陷
        |                                        |
        |                                        v
        |                                 Cement Gap / Margin Gap 🔶 参数已修正，分区未修正
        |                                        |
        |                                        v
        |                                 Intaglio Cutter
        |                                        |
        v                                        v
        └────────→ Minimum Thickness Correction 🔴 已知缺陷
                                |
                                v
                     Boolean Difference（有校验的操作）
                                |
                                v
                          Final Crown STL
                                |
                                v
                  Geometry QC + Design Package + revisions.json
                                |
                                v
                    验收套件（quantitative gate）
```

🔴 = 已知缺陷，挂在 `baseline.known_defects`；🔶 = 部分完成。详见 `ALGORITHMS.md`。

---

## 2. 分层

### 2.1 证据层（新增）

```
app/importers/exocad/
  construction_info.py   XML 解析（禁止正则）+ 零值语义注册表
  dental_project.py      病例级标注 → 牙位 / 修复体 / 材料 / OrderID
  three_ox.py            UTF-16 XML → OrderID / ScanSource / PHI
  matrix4.py             角色判定（内容一致性 → 固定坐标系变换）
  sanitizer.py           PHI 扫描 + 导入阶段脱敏
  case_loader.py         聚合 + 按订单分组（防训练/验证泄露）
```

这一层的职责是**把 exocad 的真实参数变成可溯源的数值**，并明确它不是什么
（`.matrix4` 不是咬合配准；`.DCM` 全部读不出；`.dentalCAD` 无解析器）。
详见 `EXOCAD_IMPORTER.md` 与 `DATASET.md`。

同一层还有 3Shape 侧：

```
app/importers/three_shape/
  aliases.py          AliasFilenames 还原逻辑名（物理名是 GUID、不带扩展名）
  order.py            订单 XML → 处方 / 牙位 / PHI（descendants，不是 children）
  materials.py        Materials.xml → 材料记录 + CAD 参数（按材质+子类型收窄）
  hps_scan.py         HPS 三态判定：DECODED / SUPPORTED_BUT_ENCRYPTED / UNSUPPORTED
  geometry_source.py  companion STL 三档绑定（绝不位置绑定）+ 双 provenance
  validator.py        验收闸门（面向用户的措辞在这里）
```

它比 exocad 侧多两件事：**几何可能拿不到**（CE 加密，且不绕过厂商密钥），
以及**导入必须与设计解耦**（闸门不过就 409，不建病例）。
详见 `THREESHAPE_IMPORT.md`。

### 2.2 参数层（新增）

```
app/parameters/
  models.py       ParameterCell（四态）/ ResolvedParameter / ResolvedParameterSet
  classify.py     由数据推导 LAB_GLOBAL_PRESET / RESTORATION_PRESET /
                  LAB_PRESET_VARIANT / CASE_VARIABLE
  derived.py      派生规则（功能依赖 ≠ 经验值）
  profiles.py     Lab Profile 读写 + PHI 校验
  resolver.py     固定优先级：Tooth → Case/Order → Restoration → Lab → System
  lab_profiles/   由证据数据生成的 Profile（可入 Git，无 PHI）
```

核心约束：**进入几何计算的每个数值都必须可溯源**。代码里不允许出现
`minimum_thickness = 0.8` 这类字面量。详见 `PARAMETERS.md`。

### 2.3 修订层（新增）

```
app/revisions/
  models.py       DesignRevision / RevisionKind / MarginSource（不可变）
  service.py      追加 / 采纳 / 候选消费语义 / 链摘要
```

不变量：修订不可变；自动提案永不静默覆盖人工结果；
候选被采纳后不再要求决策；审计能回答"为什么接受这个方向"。详见 `REVISIONS.md`。

### 2.4 几何层

```
app/geometry/
  insertion.py    就位方向求解器（锥内搜索 + 面积加权倒凹 + 状态机）
  axis.py         旧签名兼容入口 → insertion.solve_insertion_axis
  registration.py Kabsch + RBF
  margin.py       Margin 边界 / 平面拟合
  outer.py        Margin 裁切（含水密断言）+ 最小厚度修正
  intaglio.py     Blockout + cement gap
  boolean.py      精确布尔 + fail-loud 校验 + 体素降级（默认关闭）
  qc.py           几何 QC 测量
  io.py / roi.py
```

### 2.5 制造层（新增，部分完成）

```
app/manufacturing/
  models.py       ManufacturingProfile（3 轴 / 5 轴分离，ORDER 维度）
  reachability.py candidate_directions / split_outer_intaglio / visibility_fraction
                  evaluate_reachability → NOT_IMPLEMENTED（不发布假闸门）
```

### 2.6 服务层

```
app/services/
  pipeline.py               总流程编排
  ai_runtime.py             TorchScript 推理（真实 checkpoint 才开放）
  three_shape_import.py     3Shape 导入：preview / commit / 面板 / next_steps
```

`three_shape_import.py` 里有一个刻意的分层：**判定只在 `validator` 一处**，
`build_panel` 只是它的呈现层，不复制判定逻辑 —— 否则会出现"界面显示 ✓
而闸门实际是 FAIL"这种最坏的错位。

### 2.7 验收层（新增）

```
tests/
  ├─ unit（几何/参数单元测试）
  ├─ integration/          修订链不变量
  └─ acceptance/
     ├─ parameter/         参数回归
     ├─ geometry/          合成几何精度 + FDI36 真实病例量化守门
     └─ baselines/         baseline.json（入 Git）
```

**完成标准不是"STL 成功生成"，而是"与真实病例真值的定量误差下降"。**
详见 `ACCEPTANCE.md`。

---

## 3. 服务划分

### Frontend
- React + Three.js / React Three Fiber
- 真实 STL 可视化；Support / Template / Margin / landmark 3D 拾取
- 就位方向求解结果卡片（状态徽章 / 置信度 / 倒凹指标 / 理由列表）
- Margin 与就位方向的候选决策界面（替换 / 保留人工）
- 修订链浏览

### Backend
- FastAPI + SQLite（`cases` / `assets` / `designs` / `design_revisions`）
- trimesh / NumPy / SciPy / scikit-image
- 文件系统保存病例资产
- **精确布尔与裁切盖片是功能依赖**：`manifold3d`、`mapbox-earcut`、`rtree`
  全部钉版本在 `requirements.txt`

### AI（未启动）
- PyTorch + TorchScript
- 条件模板形变网络（固定 topology，保留 vertex / cusp / region / boundary ID，
  使下游 CAD 可无缝继续）
- **训练数据当前为 0**，`ml/checkpoints/` 为空。见 `DATASET.md` §3.4

---

## 4. 三个架构判断

### 4.1 AI 与 CAD 分开

AI 负责"牙冠外部解剖形态应该长什么样"；CAD 内核负责"怎么套在 Prep/Abutment 上、
怎么留 cement gap、怎么处理倒凹和最小厚度"。

**内腔绝不能交给 AI。** 3Shape 官方也说明 Automate 只设计 anatomy，
牙冠内部由 Dental System 的 CAD 算法完成。这与我方实测一致：
内腔误差在 0.5 mm 量级时，任何形态质量都没有意义。

### 4.2 参数与算法分开

参数来自证据数据（可溯源、可重新推导），算法只消费解析后的参数集。
这样"换一个实验室"或"参数被更新"不需要改代码。

反例（本项目已发生）：把 `minimum_thickness = 0.8` 写进算法，
导致真实冠的 33.4% 被判过薄。

### 4.3 每个模块都必须能表达"我不确定"

```
输入 → 算法 → QC → 不通过就 STOP
```

不是"算法返回一个 Mesh 就认为成功"。已落地的形态：

| 模块 | 表达不确定的方式 |
|---|---|
| 参数解析 | 四态；`MISSING` 不跌落成 0；`EXPLICIT_BUT_UNRESOLVED` 阻断自动设计 |
| 就位方向 | `OK / REVIEW_REQUIRED / FAILED` + `objective_is_flat` |
| Margin | `validate_margin()` 四项校验；不通过即 REVIEW_REQUIRED |
| 布尔 | 前置水密断言 + 后置七项校验；不通过不保存 |
| 制造可达性 | `NOT_IMPLEMENTED` + `machinable = None`（不发布假闸门） |
| 验收 | `xfail(strict=True)` 单调烧除，目标 0 xfail |
