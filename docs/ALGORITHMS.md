# 算法实现说明

**状态标注**：本文件区分**已修正**与**仍为旧实现**的算法。旧实现部分的实测缺陷已记录，
不要当成可用能力。

> 分项详述见：
> - 就位方向 → `INSERTION_AXIS.md`
> - 参数系统 → `PARAMETERS.md`
> - 验收与量化守门 → `ACCEPTANCE.md`

---

## 1. Crown Outer 配准 ✅ 已可用

### Kabsch

使用至少三对非共线对应点，最小化刚体变换后的平方误差：计算质心 → SVD 求最优旋转
→ 计算平移 → 得到 4×4 变换矩阵。

实现已核对：`u, _, vt = svd(a.T @ b)`，`r = vt.T @ u.T`（即 `V Uᵀ`），
`det(r) < 0` 时反转 `vt[-1]` 消除反射。校验见 `tests/test_registration.py`。

### RBF 非刚性形变

提供至少 4 对额外控制点时，用 thin-plate-spline RBF 对刚体配准后的模板顶点做连续形变。

用途：把基础牙位模板向患者邻牙、牙弓空间和外形控制点适配。

### 仍有问题的一环：模板来源

`initialize_from_margin()` 把模板最小投影放到 Margin 平面。
**但 `32牙位基础解剖模板库/` 里的 `FDI_XX.stl` 是带牙根的完整牙** ——
剖面实测 `FDI_36.stl` 高 12.82 mm，截面从底部 39.9 mm²（根尖）增至赤道 91.7 mm²
再收至 11.8 mm²（咬合面）；同牙位真实 exocad 冠高 6.87 mm。

用它当牙冠模板时，下颌模板的牙根会被放在 Margin 平面上、牙冠浮在上方
（实测输出 648.7 mm³ = 真值的 308%，即一根完整的牙连根带冠扣在基台上）。

`data/templates/` 是空的，README 与 UI 都正确警告过这些文件含牙根。
但 `config.ANATOMY_DIR` 正指向该文件夹、`/api/anatomy` 对外提供它、
文件夹名叫"基础解剖模板库" —— **误用只差一次没有标签的点击**，
且 `create_design` 只检查模板文件存在，**从不校验它是不是纯牙冠**。
需要加导入期模板校验（高度、根部锥度）。

---

## 2. Margin 🔶 部分可用

支持三种方式：

- 人工 3D 点序列
- 对开放 Support 网格寻找开放边界 loop 的候选
- 从真值冠颈缘取参考点（仅验收用）

**`boundary_loops()` 在真实扫描上会产出退化环。** 它用 `choices[0]` 贪心游走，
遇到边界图上级数 > 2 的顶点（T 型连接、非流形边、扫掠裁剪孔）会静默跳环。
实测：

| 文件 | 检出"环" | 实际情况 |
|---|---|---|
| `PreparationScan.stl` | 1 个，周长 118.90 mm，z 0.8→14.6 | 走遍整颗牙的高度，坏环 |
| `AbutmentAlignmentScan.stl` | 周长 208.87 mm，z −0.8→13.6 | 坏环 |
| `AntagonistScan.stl` | 周长 124.88 mm，z 12.3→25.5 | 坏环 |

然后 `suggest_margin_from_open_boundary()` 挑**周长最大**的环 —— 正好挑中坏的那个。
API 只拦 `extracted_` 前缀，把 `PreparationScan.stl` 当 support 上传会绕过拦截。

**没有任何健全性校验**：平面残差、闭环性、点是否在 support 面上。

新求解器已自带 `validate_margin()`（点数 / 闭环性 / 平面性 / 采样均匀性，见
`INSERTION_AXIS.md` §5），但 `suggest_margin_from_open_boundary` 本身还没改用
正确的环追踪。

**真值提取已修复**：`app/geometry/gt_margin.py` 的 `extract_gt_crown_margin()`
从已完工的最终冠提取 intaglio ↔ exterior 真实交界缝（朝向场符号 + 包围判据），
实测闭环、简单、自交 0、方位角覆盖 1.000。详见 `GT_MARGIN.md`。

**但 `suggest_margin_from_open_boundary` 本身仍未改用正确的环追踪** ——
患者 Prep 侧的 Margin 仍是不可靠的（上表的退化环）。两件事要分开。

**当前 Final Crown 的 cervical trim 使用 Margin 的最佳拟合平面**，是工程基线，
不等同于商业 CAD 中沿复杂三维 finish line 的高精度曲线裁切。

关于平面假设：FDI36 的真实颈缘缝接近平面（残差 max 0.043 mm），
但**旧方法测出的"残差 max 3.83 mm"是坏提取（残弧）的产物，不是颈缘的性质**。
"不要假设近平面"这个设计决定仍然正确，理由要改成"平面与否是病例的性质"，
而不是"本病例不平面"。

---

## 3. Insertion Axis ✅ 已重写

旧实现：对整块 support 顶点做 PCA 取主轴；有 Margin 时在
`|dot(toward, plane_n)| < 0.35` 时才回退到平面法向（条件写反了）。
实测在真实颈缘上稳定偏 **60.9–64.2°**。

新实现 `solve_insertion_axis()`：

```
初始轴 = Margin 稳健拟合平面法向 (A) + 侧壁法向拟合轴 (B)
      → 锥内搜索（粗 5°×12 + 细 1°）
      → 面积加权倒凹 + Margin 可见率 + 偏角 逐候选评分
      → 状态机 OK / REVIEW_REQUIRED / FAILED
```

关键设计（详见 `INSERTION_AXIS.md`）：

- 倒凹用**拔模角** `n·a < -sin(tol)` 判定，不用射线命中（后者对平行侧壁数值退化）
- 倒凹统计**严格限定在 Margin 平面之上**（否则底面圆盘贡献 16.17% 假倒凹）
- **刚性等变**：所有指标在刚体变换下严格不变
- `objective_is_flat`：候选集评分极差过小时置位，明示轴未被目标函数定住
- 必须允许人工反转和修正 → 走修订链（`REVISIONS.md`）

---

## 4. Undercut Blockout 🔴 已知缺陷（待重写）

当前实现把 Support 旋转到 insertion axis = +Z，体素化后对每个 XY 列
从底部填充到最高占用 voxel，得到 blockout envelope。

**这个做法是错的，实测证据：**

它填的是**整个上部基台的投影**，不是局部截面。在 FDI36 基台上：

| 高度 | 基台截面 | cutter 截面 | 倍数 |
|---|---|---|---|
| z=5.0 | 7.10 mm² | 34.98 mm² | 5× |
| z=6.0 | 17.15 mm² | 35.20 mm² | 2× |
| z=7.0 | 29.99 mm² | 33.49 mm² | 1.1× |
| z=8.0 | 29.69 mm² | 31.35 mm² | 1.1× |

即**边缘附近被撑大 2–5 倍**，颈部缩窄的 emergence profile 被抹掉。

实测内腔间隙（正确 +Z 轴，边缘线以上）：

| 区域 | 实测中位 | 目标 |
|---|---|---|
| 边缘带 0.8 mm 内 | **0.523 mm** | 0.025–0.060 |
| 边缘带以上 | 0.082 mm | 0.060（合格） |
| 全配合面 | 30.9% 面积超差 >0.15 mm，p95 0.983 mm，max 1.571 mm | — |

对照：一次普通的 0.06 mm 顶点法向偏移给出中位 0.061 mm（这是目标行为）。

**另外还有半个 pitch 的偏移**：`z_coords = origin[2] + k*pitch` 而
`origin = vox.transform[:3,3] - pad*pitch`，但 trimesh 的体素栅格 transform 落在
`min - pitch/2`，所以边缘带整体偏了 0.05 mm（在 0.5 mm 的误差面前是次要因素）。

**修复方向**：blockout 只应消倒凹，不应做全列棱柱。按就位方向求可见性包络，
倒凹区单独填充，再把 cutter 限制在 Margin 平面内侧。
验收阈值为 `margin_band_clearance_median_mm ∈ [0.015, 0.05]` 与
`supra_margin_clearance_p95_mm < 0.10`，当前挂在 `baseline.known_defects`。

---

## 5. Cement Gap 🔶 参数已修正，分区未修正

当前对 blockout Support 使用欧式距离变换扩张：Margin band 内用 `margin_gap_mm`，
其他区域用 `cement_gap_mm`。

**参数默认值已修正**（原来写死 0.06，实测 exocad 真实值见 `PARAMETERS.md`）：

| 参数 | 代码旧默认 | exocad 实测 |
|---|---|---|
| `cement_gap_mm` | 0.06 | 众数 0.025（`CASE_VARIABLE`，**无默认值**） |
| `CementGapTopThickness`（咬合面间隙） | 无 | 0.01（25/25，恒为轴向的一半） |

**分区仍不完整**：缺少咬合面/轴向/边缘带的三区 gap field。
exocad 的参数命名已给出结构（`CementGapThickness` / `CementGapTopThickness` /
`CrownAdditionalSpacingX/Y/Z`），应改成沿"距 Margin 距离"变化的 `g(p)` 函数。

---

## 6. Minimum Thickness 🔴 已知缺陷（待重写）

当前用 **cutter 顶点 KD-tree** 估计 Outer 顶点的内外壳距离，低于最小厚度的区域
沿 Outer 顶点法线向外迭代移动（系数 0.72、迭代 3 次），同时保护 Margin 附近。

**两个问题：**

**问题一：顶点到顶点距离不是壁厚。** 它比对的是"最近 cutter 顶点"距离，
而厚度应是沿法线穿壁的距离。颈部附近最近顶点在 cutter 颈缘上，不在垂直穿壁方向。
实测它标记 7248 个顶点过薄，其中 **2454 个（34%）其实离 cutter 表面还有 >0.25 mm**。

**问题二：默认值 0.8 是错的，且模型是一刀切。** 用 Möller–Trumbore 光线求交
实测真实可加工牙冠的壁厚：

```
min 0.035   p01 0.314   p05 0.449   median 0.909   p95 1.268   max 1.483 mm
颈部三分之一（z 5.6–8.0）: median 0.654–0.780, p05 0.33–0.48
```

**`minimum_thickness_mm = 0.8` 会把真实冠的 33.4% 判为过薄。** 而 exocad 真实参数是
分区的：`MinThickness = 0.6`（25/25）、`MinThicknessNearPreparationMargin = 0.2`、
`MinThicknessFullDistanceToPreparationMargin = 0.8`（过渡距离）。

**后果**：实测 `enforce_clearance` 移动 **37% 的顶点、最大 1.63 mm**，
把整个颈部三分之一外推，改变邻接与咬合。

**修复方向**：改成空间变化厚度场 `t_min(p)`（由距 Margin 距离决定），
用符号距离场求真实壁厚，只做局部修复 + Laplacian 平滑；
只向外膨胀 Outer，**绝不改内腔**（否则破坏就位）；优先减小 cement gap 而不是外推外冠。

验收阈值 `isolated_chain_outer_deviation_p95_mm < 0.05`，当前实测 0.976。

---

## 7. Boolean ✅ 已改为有校验的操作

**精确布尔与裁切盖片是功能依赖，不是可选依赖。**

```python
crown_minus_cutter(outer, cutter, allow_voxel_fallback=False)
```

前置条件：`outer` 与 `cutter` 必须水密（`assert_watertight`）。
后置校验：`validate_boolean_result()` 检查体积、连通分量、包围盒包含关系、
面数、质心位移。任一项不通过即抛异常，**不保存结果**。

`trim_outer_at_margin()` 不再吞掉 `cap=True` 失败 —— 那是实测 13% 碎冠的直接原因。

实测教训（默认安装，缺 `manifold3d` 与 `mapbox-earcut`）：

```
Outer(非水密) - Cutter  ->  watertight=True, volume=27.6 mm³
真值 210.8 mm³           ->  只有 13%
```

因果链：`cap=True` 需要 `mapbox-earcut` 三角化裁切面，缺失时抛异常 →
旧实现静默退化到 `cap=False` → 留下 651 条开放边界边 →
`trimesh.repair.fill_holes()` 也失败（同一个引擎缺失）→
`voxelized().fill()` 对开放壳只得到一层表面（40.2 mm³ 而非 232.2 mm³）。

**注意：mesh 合法 ≠ 结果正确。** 该结果 `watertight=True`，
QC 不报任何异常。

### 体素降级：仅调试用

`allow_voxel_fallback=True` 才启用，默认关闭。实测它**即使不做任何裁减**也会：

| pitch | 面数 | 体积 | 偏差 中位 / p95 / max |
|---|---|---|---|
| 0.10（旧默认） | 96,568（原 40,406） | 232.1 / 210.8 mm³ = **+10%** | 0.089 / 0.132 / 0.424 mm |
| 0.05 | 382,868 | 221.4 mm³ | 0.063 / 0.112 / 0.424 mm |

装上 `manifold3d` 后**布尔往返偏差为 0.0**，体积比 1.0000。

---

## 8. QC

当前包括：

- mesh 信息（顶点 / 面 / 包围盒 / 连通分量 / 水密 / 体积）
- Outer 到 Cutter 的 shell clearance estimate
- Final Crown 到 Support 的最近距离
- 过滤 Support 附近后的同颌环境距离
- Final Crown 到 antagonist 的最近距离

这些是几何测量值，**不自动替代临床阈值判断**。

### 已知缺陷

`build_qc()` 的 `shell_clearance_estimate` 用 `nearest_stats(outer, cutter)`，
即顶点到顶点最近距离 —— 与 §6 同一问题，不是壁厚。

`nearest_stats` 会把两侧网格顶点都降采样到 60000 点（顶点密集侧被稀释），
在密集区域低估距离。

---

## 9. 制造约束 🔶 只交了构件

`app/manufacturing/`：

- `ManufacturingProfile`：3 轴 / 5 轴分离；`MillingDiameter` 与 `MillingMachineAxes`
  的作用域是 **ORDER**，不属于材料 profile（实测刀具直径跨 2.4 倍、机轴数有 3 与 5）。
- `candidate_directions()`：3 轴 → 1 个方向；5 轴 → 19 个方向（锥内采样，默认 20°）。
- `split_outer_intaglio()`：外/内表面分离（真实冠 29858 / 10568 片）。
- `visibility_fraction()`：单向可见性测量。

**完整的可加工性判定尚未实现**，`evaluate_reachability()` 返回
`status = NOT_IMPLEMENTED` 且 `machinable = None`。

原因（实测于真实 exocad 最终冠）：

1. 牙冠是**壳**，外表面与内表面的可加工路径完全不同（内表面只能从颈缘开口进入）。
   用一个固定方向测全部表面会把内表面全部判为不可达 —— 那是错的。
2. 固定轴的可见率在真实冠上只有约 54%（外表面沿 +Z 遮挡 45.9%），
   **但这不代表不可加工**：真实 3 轴加工会把牙冠按优化角度装夹，
   正确的问题是"是否存在一个装夹方向使全部外表面可达"。
3. 刀体-特征干涉需要"口袋宽度分析"；"局部曲率半径 > 刀具半径"只是必要条件。

**在这些完成前 `machinable` 保持 `None`。**
发布一个会把合格设计判为"切不出来"的假闸门，比不发布更糟。
