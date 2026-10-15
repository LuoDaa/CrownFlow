# 就位方向求解器

`app/geometry/insertion.py` — `solve_insertion_axis()`

---

## 1. 旧实现错在哪

旧 `suggest_insertion_axis` 直接对整块 support 顶点做 PCA 取主轴：

```python
centered = verts - centroid
cov = np.cov(centered.T)
vals, vecs = np.linalg.eigh(cov)
axis = vecs[:, np.argmax(vals)]      # 主轴
if margin_points is not None:
    toward = centroid - margin_center
    # 只在 |dot(toward, plane_n)| < 0.35 时才回退到平面法向
```

两个问题：

1. **顶点 PCA 受非几何因素影响。** 底部、颈缘与扫描裁切形状都会把主轴拉歪；
   顶点分布疏密不均（边缘密、平面疏）还会偏置质心。
2. **回退条件写反了。** 只有 `toward` 近乎**垂直**于 Margin 平面时才回退到平面法向；
   而近平面环状 Margin 的 `|dot|` 约 0.7，于是保留了倾斜的 `toward`。

实测：在真实颈缘上稳定偏 **60.9–64.2°**（40/80/160 个 Margin 点、
0.2/0.4/0.8 mm 取带宽度都一致）—— 不是噪声，是系统性错误。

---

## 2. 新流程

```
Margin 稳健拟合平面法向 (A)  ┐
                             ├─→ 初始轴 ─→ 锥内搜索（粗+细）─→ 状态判定
侧壁法向拟合轴       (B)  ┘

逐候选评分：E = w₁·R_undercut + w₂·D_depth + w₃·(1−V_margin) + w₄·D_axis
```

### 初始轴（两个独立来源）

**A = Margin 稳健拟合平面法向**：迭代剔离群点的 SVD 拟合（`_robust_plane`），
再按"support 主体位于法向正侧"定向为 cervical → occlusal。

**B = 侧壁法向拟合轴**：面积加权法向协方差 `Σ wᵢ nᵢnᵢᵀ` 的**最小特征值**方向。
对圆柱侧壁，所有法向都垂直于轴，因此这是该轴的最小二乘估计。迭代选面
（以 A 为临时轴取 `|n·a| < sin(60°)` 的面）。

拟合质量 = `最小特征值 / 次小特征值`。实测 FDI36 基台为 **0.105**（越小越像圆柱），
意味着该参考可用；若 > 0.3 则参考不可用，验收里会显式检查这一点。

**A 与 B 分歧 > 35° 时不硬平均**，直接报 `REVIEW_REQUIRED`。

### 锥内搜索

粗层 5° × 12 方位，细层 1° × 8 方位（围绕粗层最优解）。
候选集的基取自 **Margin 平面内主轴**，使候选集随几何一同旋转 ——
这是刚性等变性的前提。

---

## 3. 三个判据选择（都有实测依据，勿改回）

### 3.1 倒凹用拔模角，不用射线命中

```
倒凹面 = n·a < -sin(tol)
```

**不要改成"从点沿 a 发射线是否命中"。** 射线法对**与就位轴平行的侧壁**数值退化：
掠射几乎必然误命中某个相邻三角形，而侧壁恰是最需要精确判定的区域。

实测（FDI36 基台，全支持体）：

| 判据 | 沿 +Z 的"不可达"比例 |
|---|---|
| 射线法 | 56.8% |
| 拔模法 | 45.6% |

两者差异主要来自掠射噪声。拔模法在 `n·a ≈ 0`（垂直侧壁）处是稳定的。

对拔模合格但仍被上方几何遮挡的面，另用射线补一个"阴影深度"统计 ——
此时射线与表面不平行，数值可靠。

### 3.2 射线命中必须过滤掠射

`_first_blocker` 只把满足两条的命中算作遮挡：

1. 命中距离 > `min_distance_mm`（排除起点附近的自身接触）
2. 命中面法向与射线方向显著相对（`n·dir < -0.2`）

**条件 2 是必需的。** 没有它时，垂直侧壁会被自己的掠射命中误判为遮挡。实测后果：
在纯圆柱预备体上，Margin 位于垂直壁底部，沿 +Z 的射线掠射 →
`margin_visibility` 被算成 0 → 其罚项（权重 0.8）压过倒凹项 →
求解器被推向一个 **17° 倾斜、倒凹 41.6%** 的错误轴（正确轴倒凹为 0）。

### 3.3 倒凹统计严格限定在 Margin 平面之上

```python
above = (centers - center) @ normal > ABOVE_MARGIN_TOLERANCE_MM   # = 0.05 mm
```

容差必须 **> 0**。若把 Margin 平面附近也算入，合成圆台的**底面圆盘**（法向朝下）
会被计入倒凹，贡献恰好等于其面积占比的**假倒凹 16.17%**
（= πr² / (侧面积+底面+顶面)）。真实病例里 Margin 平面以下同样只是
扫描裁切边界或牙龈，不参与就位。

### 3.4 面积加权

逐面判定 + 按面积折算阴影，**不用顶点计数**。网格疏密不均时两者实测差 **0.32**
（面积加权 0.886 vs 采样计数 0.567）。

---

## 4. 刚性等变性（这是一个硬要求）

所有指标必须在刚体变换下**严格不变**：`axis(T·M) == R·axis(M)`，
且所有标量指标完全一致。

这条要求抓到一个真 bug：`estimate_wall_axis` 曾用

```python
if axis[2] < 0: axis = -axis      # ❌ 世界坐标符号约定
```

轴被翻转后拔模角 `n·a` 全部反号，倒凹比在刚体变换下从 **0.844 跳到 0.156**
（两者互补），而**轴夹角仍显示 0.000°** —— 用 `abs(dot)` 算夹角会完全掩盖方向翻转。

**方向必须用几何约定确定**：从 Margin 平面指向 support 主体
（`dot(centroid − plane_origin, axis) > 0`）。

---

## 5. 状态机

| 状态 | 触发条件 |
|---|---|
| `FAILED` | Margin 结构性缺陷（点数 < 3、坐标非有限、点重合） |
| `REVIEW_REQUIRED` | Margin 非闭环 / 严重非平面 / 采样严重不均；A/B 分歧 > 35°；最优轴倒凹比 > 0.15；Margin 可见率 < 0.90；**`objective_is_flat`** |
| `OK` | 以上都不触发 |

**`objective_is_flat` 的用途**：候选集上评分极差小于 `FLATNESS_THRESHOLD = 0.02` 时置位，
明示"该病例的就位方向未被目标函数定住，argmin 不具备可信度"。

这是必要的：倒凹面积、射线不可达、blockout 代价这三类目标，在**紧凑且轴对称的基台**上
区分度很低（实测 FDI36 基台上跨轴变化都在 ±8% 以内）。此时一个平坦目标上的 argmin
不能冒充可信结论 —— 不允许"挑最不坏的一个然后声称 OK"。

---

## 6. FDI36 病例：为什么不用"+Z 误差"当验收判据

**这是本项目修正掉的一个错误判据，记录在此以免重犯。**

最初把"与 +Z 的夹角 < 10°"作为目标（理由：该病例咬合方向是 +Z，由对颌几何位置佐证）。
实测后放弃，三条独立证据：

**证据一：+Z 不是真值。** 基台侧壁法向独立拟合出的轴**本身就在 +Z 的 12.9° 外**。
以 +Z 与以 Margin 平面法向为参考各拟合一次，两次结果一致到 0.01°，
拟合质量 eigenvalue ratio ≈ 0.105（7322 个面）。种植基台相对颌骨咬合平面倾斜是常见情况。

**证据二：三类目标都无法区分。** 倒凹面积、射线不可达、blockout 代价在该基台上
跨轴变化都在 ±8% 以内，且最小值不落在 +Z。blockout 代价（= cutter 体积 − 基台体积）：

| 就位轴 | 代价 mm³ |
|---|---|
| +Z | 183.2 |
| 旧 PCA 建议轴 | 177.1 |
| 倾斜 20° | 188.2 |
| 倾斜 45° | 182.4 |
| 倾斜 61° | 187.3 |

分不开。原因是该基台是个紧凑块体，投影体积不随方向显著变化。

**证据三（已被推翻，见下方"撤回"）**：曾称该基台有一个轴对称外扩的 emergence
profile，相对任何轴都是倒凹，因此"OK"本身是错答案。

> **撤回。** 该结论建立在错误的 Margin 真值上。旧的"最低 0.4 mm 带"给出一个几乎
> 水平的就位面边界（偏离 +Z **0.80°**），而真实颈缘缝的平面偏离 +Z **11.67°**，
> 与基台自身拟合轴（**12.9°**）一致 —— 边缘平面本就垂直于基台轴线。
> 用错平面切分"就位面"会把倒凹统计的作用域放错位置，虚高倒凹比。
>
> 用正确 Margin 重算：最优轴倒凹比 **0.117 < 0.15**，求解器报 **OK**。
> 记录在 `baseline.json` 的 `retracted_expectations`。

**因此真实病例上只断言三件事**（精度断言放在合成几何上）：

| 断言 | 实测 | 阈值 |
|---|---|---|
| 与基台拟合壁轴一致 | **2.27°** | ≤ 8° |
| 倒凹面积比相对平凡解 +Z 的改善 | 0.328 → 0.117 = **−64.3%** | ≤ −25% |
| Margin 沿最优轴可见率 | **1.00** | ≥ 0.90 |
| 状态 | **`OK`** | 与 Margin 质量一致 |

"求解器必须能表达没有好方向"这一 fail-loud 性质**不依赖本病例** ——
由合成几何的 `test_solver_never_silently_returns_worst_of_bad_set` 与
`test_bulge_is_flagged_not_hidden` 断言。这样避免了"为了烧掉 xfail 而把病例
强行判为 REVIEW"这种反向迎合。

与 +Z 的夹角降级为 baseline 的 `error_vs_plus_z_deg_DIAGNOSTIC_ONLY = 11.08`，
**仅作追踪，不作断言**。

---

## 7. 结果字段

```python
InsertionAxisResult(
    axis, initial_axis, score,
    undercut_area_mm2, undercut_area_ratio,
    max_undercut_depth_mm, mean_undercut_depth_mm,
    margin_visibility_ratio,
    deviation_deg,
    confidence,
    status,               # OK | REVIEW_REQUIRED | FAILED
    diagnostics,          # 见下
)
```

`diagnostics` 关键字段：

| 字段 | 用途 |
|---|---|
| `margin_validation` | 点数 / 闭环性 / 平面性 / 采样均匀性 + 拒绝理由 |
| `initial_axis_sources` | A、B 两个轴、拟合质量、分歧角 |
| `search` | 候选数、评分极值、极差 |
| `objective_is_flat` | 目标是否被定住 |
| `best_axis_metrics` | 最优轴的完整指标与权重 |
| `max_deviation_deg_source` | **20° 的出处**（Lab Profile 证据，非 exocad 全局规律） |
| `neighbors_note` | 邻牙是否参与评分 |
| `reasons` | 状态判定理由 |

`as_dict()` 会把非有限浮点净化成 `null` —— NaN 不是合法 JSON，
FastAPI 序列化会直接抛 `ValueError: Out of range float values`。

---

## 8. `max_deviation_deg = 20°` 的出处

来自当前 Lab Profile 的实测证据：`CrownBottomInsertionAngleBetweenTeethLimit`
在 25/25 条 exocad 记录上为 20。

**这是该实验室的病例级证据，不是 exocad 全局规律。** 上传的 exocad
*Material Configuration Guide* 并未给出该技术字段的定义。换实验室需重新取值。
这个说明随每次求解结果一起返回（`diagnostics.max_deviation_deg_source`）。

`neighbors` 当前只记录不参与评分。缺少可靠邻牙分割时，邻牙碰撞项会把求解器
拖向不可靠的结论；`neighbors=None` 必须能完整运行。

---

## 9. 验收

**合成几何**（`tests/acceptance/geometry/test_insertion_axis.py`，24 项）——
真值可构造，精度断言在这里：

| 类别 | 实测 |
|---|---|
| A 刚性等变 | 轴偏差 **0.0000°**；Δundercut / Δvis / Δscore **严格为 0**；方向不翻转 |
| B 完美锥台 | 轴误差 **0.000°**，倒凹比 < 1e-6，`status = OK` |
| C 倾斜预备体 | tilt 13 / 25 / −18 / 32° 全部 **0.000°** |
| D 人工倒凹 | 求解轴评分优于 +Z 与 15 / 30 / 45° 全部替代方向 |
| E 退化 Margin | 点数不足 → `FAILED`；非闭环 / 非平面 / 采样不均 → `REVIEW_REQUIRED` |
| F 分辨率鲁棒性 | 40 / 120 / 360 段之间轴差 **0.0000°** |

合成 fixture 在 `tests/acceptance/_synthetic.py`。自建圆台而非用
`trimesh.creation.cone` —— 后者的底面是带中心点的圆盘，会让 Margin 校验失败，
而这与求解器无关。

**真实病例**：`test_fdi36_golden.py` 的四项（见 §6）。

---

## 10. API 与前端

- `POST /api/designs/{id}/suggest-axis` → 完整诊断。状态非 OK 时前端**必须**让用户确认。
- `POST /api/designs/{id}/axis` → 人工确认/修改，写 `axis_accepted` 修订并记录确认人。
  不允许只更新数据库里一个向量。
- 前端显示：状态徽章 + 置信度 + 倒凹面积比 / 最大深度 / Margin 可见率 / 偏角 + 理由列表
  + 「确认方向（记入修订）」按钮。

详见 `REVISIONS.md`。
