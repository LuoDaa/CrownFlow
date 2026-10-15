# Localization v3.2 二轮 QA 与**路线终止**判定

**日期**：2026-09-28。**这是 v3.2 的最后一轮，也是"自有 exocad 数据能否
用于目标牙定位"这个问题的终审。** 结论：**不能**，该路线终止，不再做
v3.3/v4；`exocad_gt_v1` 随即冻结（定位集合只保留人工确认的 A）。

前置文档：[`LOCALIZATION_QA.md`](LOCALIZATION_QA.md)（v1/v2 判据被证伪）、
[`ARCH_ASSET_AUDIT.md`](ARCH_ASSET_AUDIT.md)（弓资产分类）。

---

## 1. 这一轮到底测什么

v2 判据"`<Center>` 附近有表面"被 QA1 证伪（precision ≈ 0.10）后，假设是
**判据太弱**：只看"附近有没有表面"，分不清预备体、洞、天然牙尖、底座。
于是做了 v3.0→v3.2：

* v3.0：中心/肩台/外环三带 + 凸起量 `prominence`；
* v3.1：符号无关特征（`rel_pos` / `prominence_abs` / `radial_coverage`
  / `shoulder_contrast_abs` / `spread`），并修掉 `MISSING_LOCAL_FRAME`
  （帧来源优先级 Tooth Frame → Margin PCA → `FEATURE_UNAVAILABLE`）；
* v3.2：**子中心格搜索**（±6mm、1mm 步长、13×13）+ NMS + 三个同时条件
  （中心凸起、肩台环形结构、连续覆盖、与 Margin 的关系）。

如果 v3.2 能稳定把峰落在真预备体上，就把高置信、人工确认的行放进
`LocalizationGTSet`；否则终止这条路线。

## 2. 样本设计（55 例，分层，**刻意过采样工作点**）

从 `lattice.csv` 与 `features.csv` 的 join 结果里按分层抽 55 例（上限 55，
用户规格 50～60）。关键分层是 **`in_poly`**：`peak1_in_polygon == True`
正是 v3.2 唯一会判"这里是 Prep"的那批 —— 它的精度就是要测的东西。

| 分层 | 定义 | 抽到 | 其中 A |
|---|---|---:|---:|
| `A_high` | WITH_TARGET 按分数降序 top 10 | 10 | 1 |
| **`in_poly`** | **WITH_TARGET 且峰在目标牙 margin 轮廓内，按分数降序 top 14** | **12** | **4** |
| `border` | WITH_TARGET 分数排名 13–26 | 12 | 2 |
| `mid` | WITH_TARGET 分数排名 31–42 | 11 | 0 |
| `v2pos_weak` | WITH_TARGET 且 `prom < 1.0` | **0** | – |
| `miss_strong` | `FULL_ARCH_TARGET_MISSING` 按分数降序 | 1 | 0 |
| `unk_strong` | `UNKNOWN`/`PARTIAL_ARCH` 按分数降序 top 14 | 9 | 0 |
| `meta` | `classification_basis` 以 METADATA 开头 | 0（去重后并入上列） | – |

**两个分层本身是空的，这是本轮的一个独立发现**：

* `v2pos_weak`（低分正类）**一例都没有** —— 在整个 join 结果里，
  WITH_TARGET 行的 `peak1_prom` 没有一例 < 1.0。**这个分数不会"安静下来"。**
* `miss_strong` 全语料只有 **1** 行可渲染（`FULL_ARCH_TARGET_MISSING`
  里 `feature_status == OK` 的只有 1 行）。

因此 55 例而不是按容量算的 68 例，**不是去重或偷懒造成的，是数据本身没有
那两批样本**。渲染脚本的运行记录同时给出
`QA_RENDERED = 55`、`QA_ASSET_MISSING = 0`、`QA_HASH_MISMATCH = 0`
（55 例全部成功渲染，没有因为资产丢失或哈希不符而丢样本）。

渲染方式：`app/groundtruth/qa_render.py` 的纯 stdlib z-buffer，
每例三栏 —— **俯视（沿就位方向，叠**目标牙自身 margin 轮廓**点线 +
峰位方块标记）/ 侧视 / 径向高度曲线（含 r=4、6mm 参考线）**。
轮廓与几何用同一个 `in_plane_basis(zdir)`，所以"峰是否落在目标牙轮廓内"
可以直接目视核对，不必相信我的判据实现。11 张拼图 `sheet_0..10.png`。

标注口径（用户规格，只标 A/B/C/U）：

* **A** = 预备体确实在可观察几何里，且峰落在它上面；
* **B** = 代型被抽走，只剩洞/沟槽；**B 不算 A** —— 洞的存在恰恰说明预备体已被移除；
* **C** = 峰落在其它牙（天然完整牙、邻牙牙尖）、模型底座、或扫描伪影上；
* **U** = 看不清。

## 3. 判定结果

**A = 7 / B = 2 / C = 42 / U = 4（n = 55）。**

`A` 的 7 例：`cf_ee381ad3e0e04b45(47)`、`cf_fe645cc532c36cda(16)`、
`cf_bf0e61da23a8a34b(45)`、`cf_b71e6f7def991fa9(13)`、
`cf_a11a60c891cbd2de(37)`、`cf_a9fa2afa422ec952(36)`、
`cf_dac42eb1fc1a6192(37)`。

判定表（完整 55 行；`prom` 为 `peak1_prom`，`d_marg` 为峰到 margin 轮廓的
距离 mm，`surf_d` 为 `<Center>` 到最近扫描表面的距离 mm）：

| idx | 分层 | case_uuid | FDI | v2 类 | prom | rel_pos | d_marg | in_poly | surf_d | 人工 |
|---:|---|---|---:|---|---:|---:|---:|---|---:|---|
| 0 | A_high | cf_9593a3da3b1e3409 | 36 | FA_WITH_TARGET | 8.78 | 0.960 | 0.003 | True | 0.537 | **C** |
| 1 | A_high | cf_3e365fd609d5a85a | 21 | FA_WITH_TARGET | 8.76 | 0.992 | 3.209 | False | 0.738 | **C** |
| 2 | A_high | cf_ee381ad3e0e04b45 | 47 | FA_WITH_TARGET | 8.69 | 0.144 | 0.383 | False | 1.447 | **A** |
| 3 | A_high | cf_ffb6f0cb356ca848 | 32 | FA_WITH_TARGET | 8.38 | 0.116 | 2.733 | False | 0.064 | **U** |
| 4 | A_high | cf_9e45cb360d8409b4 | 15 | FA_WITH_TARGET | 8.19 | 0.019 | 0.8 | False | 0.896 | **C** |
| 5 | A_high | cf_a243254eadea0b85 | 32 | FA_WITH_TARGET | 8.16 | 0.0 | 4.24 | False | 1.199 | **C** |
| 6 | A_high | cf_3e9478dd4181472c | 27 | FA_WITH_TARGET | 8.15 | 0.045 | 0.082 | True | 0.383 | **C** |
| 7 | A_high | cf_35d32e1b7a731ad0 | 21 | FA_WITH_TARGET | 8.14 | 0.055 | 1.081 | False | 1.32 | **C** |
| 8 | A_high | cf_9e45cb360d8409b4 | 44 | FA_WITH_TARGET | 8.01 | 0.083 | 3.691 | False | 1.044 | **C** |
| 9 | A_high | cf_4eb14580a61b071e | 14 | FA_WITH_TARGET | 7.99 | 0.002 | 2.96 | False | 3.079 | **C** |
| 10 | in_poly | cf_fe645cc532c36cda | 16 | FA_WITH_TARGET | 7.85 | 0.057 | 1.95 | True | 1.548 | **A** |
| 11 | in_poly | cf_bf0e61da23a8a34b | 45 | FA_WITH_TARGET | 7.81 | 0.099 | 1.175 | True | 1.219 | **A** |
| 12 | in_poly | cf_918e4e9948fc232e | 22 | FA_WITH_TARGET | 7.61 | 0.976 | 1.036 | True | 1.573 | **C** |
| 13 | in_poly | cf_2fe820adf9b440fd | 46 | FA_WITH_TARGET | 7.06 | 0.03 | 3.651 | True | 0.668 | **C** |
| 14 | in_poly | cf_15afe00929c040f8 | 14 | FA_WITH_TARGET | 6.91 | 0.023 | 0.539 | True | 0.219 | **C** |
| 15 | in_poly | cf_4d1162d079648847 | 16 | FA_WITH_TARGET | 6.78 | 0.837 | 0.206 | True | 1.056 | **C** |
| 16 | in_poly | cf_73813c4a97693bfe | 46 | FA_WITH_TARGET | 6.41 | 0.983 | 1.587 | True | 0.264 | **B** |
| 17 | in_poly | cf_b71e6f7def991fa9 | 13 | FA_WITH_TARGET | 6.32 | 0.923 | 2.966 | True | 2.876 | **A** |
| 18 | in_poly | cf_eb4a25c302e5ae2a | 11 | FA_WITH_TARGET | 6.30 | 0.050 | 0.447 | True | 1.404 | **C** |
| 19 | in_poly | cf_a11a60c891cbd2de | 37 | FA_WITH_TARGET | 6.24 | 0.059 | 2.309 | True | 2.15 | **A** |
| 20 | in_poly | cf_913e0d4ebd3b4561 | 36 | FA_WITH_TARGET | 7.43 | 0.082 | 4.198 | True | 1.552 | **C** |
| 21 | in_poly | cf_bb24c6a1779cc561 | 46 | FA_WITH_TARGET | 6.16 | 0.976 | 3.397 | True | 2.179 | **C** |
| 22 | border | cf_a94d455ccde32e79 | 15 | FA_WITH_TARGET | 7.75 | 0.054 | 3.348 | False | 0.505 | **C** |
| 23 | border | cf_c533f356eb4cef0c | 22 | FA_WITH_TARGET | 7.64 | 0.039 | 3.175 | False | 0.883 | **C** |
| 24 | border | cf_5a6f18cc1349a978 | 35 | FA_WITH_TARGET | 7.62 | 0.045 | 3.402 | False | 2.152 | **C** |
| 25 | border | cf_9f3a7d2d4e5760e7 | 45 | FA_WITH_TARGET | 7.61 | 0.021 | 3.093 | False | 2.168 | **C** |
| 26 | border | cf_a9fa2afa422ec952 | 36 | FA_WITH_TARGET | 7.59 | 0.014 | 3.774 | False | 1.728 | **A** |
| 27 | border | cf_f547b232c173b216 | 45 | FA_WITH_TARGET | 7.49 | 0.056 | 0.476 | False | 1.674 | **U** |
| 28 | border | cf_c6ac4c93adb9905c | 45 | FA_WITH_TARGET | 7.48 | 0.008 | 2.7 | False | 0.192 | **C** |
| 29 | border | cf_84fec1df0576d32b | 33 | FA_WITH_TARGET | 7.45 | 0.027 | 0.289 | False | 0.487 | **C** |
| 30 | border | cf_dac42eb1fc1a6192 | 37 | FA_WITH_TARGET | 7.43 | 0.928 | 2.19 | False | 1.889 | **A** |
| 31 | border | cf_3e3bf04cc11b75a3 | 14 | FA_WITH_TARGET | 7.42 | 0.029 | 3.586 | False | 1.625 | **C** |
| 32 | border | cf_637255ed37cb5989 | 36 | FA_WITH_TARGET | 7.41 | 0.095 | 4.844 | False | 0.55 | **C** |
| 33 | border | cf_14611ff0f46877f3 | 44 | FA_WITH_TARGET | 7.41 | 0.064 | 0.053 | False | 0.422 | **C** |
| 34 | mid | cf_485c9dea34ec8517 | 43 | FA_WITH_TARGET | 7.21 | 0.972 | – | – | 0.422 | **U** |
| 35 | mid | cf_5c84b6532ae6264c | 43 | FA_WITH_TARGET | 7.17 | 0.026 | 3.225 | False | 2.262 | **C** |
| 36 | mid | cf_8be39f199d7d7af5 | 24 | FA_WITH_TARGET | 7.14 | 0.032 | 0.581 | False | 0.201 | **C** |
| 37 | mid | cf_d8ac811ee84cf465 | 21 | FA_WITH_TARGET | 7.11 | 0.135 | 2.746 | False | 0.065 | **C** |
| 38 | mid | cf_005dc7b22ddbc1ca | 11 | FA_WITH_TARGET | 7.09 | 0.040 | 2.575 | False | 0.403 | **C** |
| 39 | mid | cf_a94d455ccde32e79 | 12 | FA_WITH_TARGET | 7.08 | 0.03 | 3.638 | False | 1.622 | **C** |
| 40 | mid | cf_b1e30086277b0b09 | 24 | FA_WITH_TARGET | 7.06 | 0.026 | 0.051 | False | 1.299 | **C** |
| 41 | mid | cf_d4b1cc2ba960f982 | 23 | FA_WITH_TARGET | 7.04 | 0.903 | 3.226 | False | 0.367 | **C** |
| 42 | mid | cf_106a04fc5bb24ee5 | 45 | FA_WITH_TARGET | 7.03 | 0.014 | 3.395 | False | 2.302 | **U** |
| 43 | mid | cf_6ce79eff7ba0a36c | 13 | FA_WITH_TARGET | 6.99 | 0.027 | 3.133 | False | 2.254 | **C** |
| 44 | mid | cf_5c884747e8d680ff | 15 | FA_WITH_TARGET | 6.96 | 0.039 | 0.907 | False | 0.562 | **C** |
| 45 | miss_strong | cf_758022c0d9ed2fb3 | 16 | FA_TARGET_MISSING | 2.31 | 0.063 | 0.008 | True | 6.213 | **C** |
| 46 | unk_strong | cf_23af5ce6b0e60a57 | 12 | PARTIAL_ARCH | 8.40 | 0.933 | 0.599 | False | 2.417 | **C** |
| 47 | unk_strong | cf_910d96981137a2fc | 11 | PARTIAL_ARCH | 8.15 | 0.052 | 0.521 | True | 3.814 | **C** |
| 48 | unk_strong | cf_5ba6ba8039266c0c | 46 | PARTIAL_ARCH | 7.00 | 0.919 | 1.296 | False | 1.82 | **C** |
| 49 | unk_strong | cf_85d34e3ad1df2166 | 23 | PARTIAL_ARCH | 6.97 | 0.063 | 1.334 | False | 1.311 | **C** |
| 50 | unk_strong | cf_4fbba56bce3c2d9f | 33 | PARTIAL_ARCH | 6.90 | 0.982 | 4.192 | False | 0.277 | **B** |
| 51 | unk_strong | cf_d5cdd15dfb61cbfa | 27 | UNKNOWN | 6.45 | 0.053 | 4.528 | True | 3.713 | **C** |
| 52 | unk_strong | cf_6f5d0c42b3f2e3c7 | 46 | PARTIAL_ARCH | 6.41 | 0.952 | 0.145 | False | 0.282 | **C** |
| 53 | unk_strong | cf_106a04fc5bb24ee5 | 47 | UNKNOWN | 6.23 | 0.083 | 1.763 | False | 3.725 | **C** |
| 54 | unk_strong | cf_224f72506d86c43b | 16 | PARTIAL_ARCH | 6.11 | 0.965 | 0.057 | True | 0.145 | **C** |

## 4. 混淆矩阵

两个"工作点"必须分开说，不能混成一个"精度"：

| 判据（预测正类） | 触发 | 命中 A | precision |
|---|---:|---:|---:|
| **raw**：取最高峰即判 Prep | 55 / 55 | 7 | **0.127** |
| **in_poly**：再要求峰落在目标牙 margin 轮廓内 | 18 / 55 | 4 | **0.222** |

`in_poly` 的完整混淆矩阵（54 例有 in_poly 值且非 U）：

```
                  人工 A   人工 非 A
in_poly = True        4        14      → precision 0.222
in_poly = False       3        33      → recall(样本内) 0.571，accuracy 37/54 = 0.685
```

**结论：v3.2 的 margin 轮廓判据 precision 0.222，与它要取代的 v2
判据（QA1 测得 ≈ 0.10，本轮 QA1 复核计数 A3/B4/C19/U5）在统计上
不可区分。** 加了一层格搜索、一层符号无关特征、一层 margin 几何关系，
精度没有实质提升 —— 因为**"凸起"在牙列里到处都是**（见第 5 节）。

召回率 0.571 只是样本统计量，**不是总体估计**：这 7 个 A 是被分层选进来的，
不是从总体随机抽到的。

## 5. 三条独立的证伪（不依赖我的目检）

目检有主观性，所以另外做了三条纯计算的检验，全部独立支持终止结论。

### 5.1 全语料：v3.2 的最高峰 **79% 落在目标牙自己的 margin 轮廓之外**

`lattice.csv` 1611 行（470 行有峰）：

| v2 类 | 有峰行数 | 峰在轮廓内 | 比例 |
|---|---:|---:|---:|
| `FULL_ARCH_WITH_TARGET` | 382 | 73 | **19%** |
| `UNKNOWN` | 68 | 20 | 29% |
| `PARTIAL_ARCH` | 19 | 5 | 26% |
| **合计** | **470** | **99** | **21%** |

**唯一一个最强的峰（也就是 v3.2 会输出的那个答案）有 79% 连目标牙自己的
轮廓都不在。** 这不是"分类器不够准"，是"峰的落点在四分之三的情况下根本
不是那颗牙"。渲染图上可以看到它的样子：峰落在邻牙的天然牙尖上、
落在完整牙列的牙尖上、落在模型底座上。

### 5.2 用户要求的附加条件**没有区分度**

用户明确要求"不要直接选最高峰，要同时检查中心凸起 + 肩台环形结构 +
连续覆盖 + Margin 关系"。实测（落错牙 vs 落在本牙）：

| 附加条件 | 落错牙（n=275）中位 | 落在本牙（n=73）中位 |
|---|---:|---:|
| `shoulder_contrast`（肩台对比度） | **2.167** | 1.854 |
| `radial_coverage`（连续覆盖） | **1.000** | 1.000 |
| `spread`（环带离散度） | 1.995 | 2.548 |
| `dist_to_margin` | 2.672 | 1.320 |

**肩台对比度在"落错牙"上反而更高**，覆盖率两边都是满值 1.0。
也就是说：这两条被寄予希望的条件，在真实数据上不区分正负类。这是对
"再加条件就能救回来"这个假设的直接证伪。

### 5.3 目标牙位完全没有几何时，照样出强峰

`<Center>` 到最近扫描表面的距离 ≥ 6mm（该位置**没有任何几何**，不可能
有预备体）的行里，仍有 2 行给出 `prom ≥ 2.0` 的强峰，**且峰都在目标牙
margin 轮廓内**。其中一例就是判定表的 idx 45
（`cf_758022c0d9ed2fb3`，FDI 16，`surf_d = 6.213`，`prom = 2.31`，
`in_poly = True`，人工判 **C**）。

**在一个 6.2mm 内没有几何的位置上、在目标牙自己的轮廓内报出"预备体凸起"，
是逻辑上的不可能，不是判断上的分歧。**

## 6. 一个规模性的旁证：70% 的行连特征都算不出来

同一批数据里，`<Center>` 有 **1108 / 1611 行**（69%）在 7mm 邻域内找不到
局部点（`NO_LOCAL_POINTS`），其中 960 行的 `target_surface_distance ≥ 6mm`、
737 行 **> 9mm**。两条独立的代码（`arch_audit` 的距离计算与 v3 的
`local_points`）对同一 Center 给出一致的结论，所以不是读取假象。

一颗牙的尺寸只有 8～10mm，`<Center>` 离最近表面超过 9mm 意味着它基本不在
模型上。**即便局部形态判据完美，"设计坐标系 ↔ 扫描坐标系"这一层也没对齐**，
自动定位的可覆盖范围先被砍掉约 70%。这一层与 5.1 的 79% 是同一个成因的
两种表现。

## 7. 实测局限（必须一起读）

* 目检基于 200×200 的 z-buffer 灰度图。**"平滑的天然牙合面"与"浅
  预备体"在个别例子上确实难以区分**，所以有 4 例记 U。换一个评审人可能
  在 A↔C 之间有若干例摆动。
* **但这个摆动不改变结论**：把 4 个 U 全部最宽松地当成 A，
  `in_poly` 工作点的 precision 也只是 8/22 = 0.36；而 5.1 的 79%、
  5.2 的"无区分度"、5.3 的逻辑矛盾，全都**不依赖我的目检**。
* 跨轮次一致性只做了一次：QA1 与 QA2 样本有 1 个牙位重叠
  （`cf_005dc7b22dd…` FDI 11），两次独立标注都是 C。样本太小，只能算
  一个正向信号，不构成可靠性证明。
* 召回率不可估（见第 4 节）。

## 8. 判定与落地

### 8.1 终止自有数据 Target Locator 路线

```python
# app/groundtruth/human_labels.py
LOCALIZATION_ROUTE_STATUS = "TERMINATED_GEOMETRY_SELF_DATA"
```

**不再做 v3.3/v4。** 依据就是第 4、5 节：v3.2 相对 v2 无实质增益，
附加条件无区分度，且存在逻辑上不可能的假阳性。

生产路线改为（与本轮结论一致）：

* **3Shape 病例** → 直接读牙位 + 3D Origin（牙位坐标系）→ 定位 Prep；
* **普通 STL** → Target Locator 给出候选 → **不确定时由用户点一下目标牙**。

也就是说 **Target Locator 不是主项目阻塞项**：用户点选是合法且可靠的兜底。

### 8.2 `LocalizationGTSet` 改为人工准入

`arch_asset_class == FULL_ARCH_WITH_TARGET` 从"唯一准入值"降级为
**必要不充分的自动预筛**；准入的最终判据是**人工判定 == A**：

```python
# app/groundtruth/models.py
LOCALIZATION_REQUIRES_HUMAN_PREP = True
```

排除原因也相应细分：`excluded_no_human_verdict` / `excluded_human_label_B`
/ `excluded_human_label_C` / `excluded_human_label_U`。
`EndToEndGTSet` 同样要求人工 A（它内含定位环节，不能声称一个已被证伪的
环节是通的）。

两轮人工判定合计 **10** 个牙位被确认为 A（QA1 三个 + QA2 七个），这就是
`LocalizationGTSet` 的规模上限；具体数量由 `freeze` 按索引实际算出来
（写进 `dataset_manifest.json`，不预填数字）。

### 8.3 这批数据**仍然可用**的部分（下一主线的输入）

定位路线死了，但**与定位无关的真值全部不受影响** —— 它们来自
`.constructionInfo` 的设计几何与导出的 `*_cad.stl`，不依赖从扫描里找目标牙：

* `MarginGTSet` / `AxisGTSet` —— 颈缘线与就位方向；
* `SingleAnatomicCrownGTSet` —— 单单位解剖全冠的完整真值；
* `OuterCrownGTSet` —— 导出冠几何（**正是"模板 → Crown Outer"需要的回归目标**）。

数据清洗阶段到此结束，主线转入 **32 牙位基础模板 → 依据 Prep / Margin /
左右邻牙 / 对颌牙生成 Crown Outer**。

## 9. 复现

```bash
# 服务器（/tmp/gt 为 app 包副本；数据根目录只读）
python3 -m app.groundtruth.v3_features --stage features --out /tmp/gt_v31
python3 -m app.groundtruth.v3_features --stage lattice  --out /tmp/gt_v31   # 64 min
python3 -m app.groundtruth.qa2_render \
        --lattice /tmp/gt_v31/lattice.csv --features /tmp/gt_v31/features.csv \
        --root <all_data> --root <EXO> --out /tmp/qa2                        # 55 例
python3 /tmp/gt_v31/v32_verdict.py    # 全语料判定
python3 /tmp/gt_v31/v32_diag.py       # 自洽性诊断
```

冻结：

```bash
python3 -m app.groundtruth.indexer --root <all_data> --root <EXO> --out <index>
python3 -m app.groundtruth.arch_audit --out <audit>       # 本轮的 audit 用 v2 那一次
python3 -m app.groundtruth.freeze --index <index> --audit <audit> \
        --out <out> --dataset-version exocad_gt_v1
```

人工判定表是**唯一来源** `app/groundtruth/human_labels.py`，由
`tests/unit/groundtruth/test_human_labels.py` 把本文档的数字钉住
（标签分布、分层容量、A 计数、混淆矩阵内部一致性），改标签而不同步改
文档/常量会直接测试失败。

---

## 10. 冻结结果：`exocad_gt_v1`

索引 **6132 行 / 2136 例 / 2133 个分组**（分组键 = 病例，用于训练/验证拆分）。
完整 manifest（PHI-free：只有计数与政策，无 case_uuid / 订单号 / 姓名）随仓库
冻结在 [`exocad_gt_v1_manifest.json`](exocad_gt_v1_manifest.json)；真实索引与
成员表含订单号，**只留在服务器** `/tmp/gt_freeze_v1/`。

| GT 集合 | 行数 | 病例数 |
|---|---:|---:|
| `MarginGTSet` | 3498 | 1863 |
| `AxisGTSet` | 3498 | 1863 |
| **`LocalizationGTSet`** | **4** | **4** |
| `SupportGTSet` | 2028 | 1406 |
| `OuterCrownGTSet` | 686 | 385 |
| `SingleAnatomicCrownGTSet` | 177 | 163 |
| `EndToEndGTSet` | **0** | 0 |

`EndToEndGTSet` 为 0 是**诚实的结果**：它要求端到端每一环都齐，而定位这一环
现在只能靠人工确认的 4 例，且还要求同病例有对颌牙 —— 没有任何一例同时满足。
这不影响 `SingleAnatomicCrownGTSet`（177 行）作为下一主线的回归目标。

### 10.1 人工 A 有 10 个，为什么集合只有 4 个

逐个查过（`frozen_tooth_index.csv`），差额全部有明确原因：

| 原因 | 数量 | 说明 |
|---|---:|---|
| 通过全部门槛 | **4** | margin/axis 有效 + 有整颌候选 + `WITH_TARGET` + `CONSTRUCTION_INFO` |
| 设计快照只有 `.modelInfo` | 5 | 冻结政策：几何 GT 只认 constructionInfo（见 `models.CANONICAL_DESIGN_SOURCE`）。这 5 例的 Center/Axis 出自 modelInfo，坐标帧未经验证 |
| 颈缘无效 | 1 | `cf_0335f160d84…` FDI 36 是 `AnatomicPontic`（桥体），无有效颈缘 |

也就是说 **这 4 个不是"数据里只有 4 个真预备体"，而是"只有 4 个同时满足
人工确认 + 快照来源合规 + 其余几何门槛"**。

### 10.2 还有 151 行是"没被人工看过"，不是"被判为假"

排除原因里 `excluded_no_human_verdict = 151`：这些行通过了自动预筛
（`WITH_TARGET` 且其余几何门槛都过），但**没有被抽进任何一轮 QA**，
所以既没被准入、也没被判定为假。这是人工审阅量的上限，不是数据的上限 ——
如果将来要扩 `LocalizationGTSet`，唯一的正当途径是**继续人工审阅这 151 行**，
而不是放宽判据。

### 10.3 排除原因全表（合计 = 6132，可与索引行数对账）

```
excluded_not_margin_gt            2239
excluded_no_arch_candidate        2181
excluded_unknown                   635
excluded_target_missing            461
excluded_not_geometric_gt_source   395   <- 本轮修复补上的门（见下）
excluded_no_human_verdict          151
excluded_partial_arch               37
excluded_human_label_C              20
excluded_antagonist                  4
admitted                             4
excluded_human_label_U               3
excluded_human_label_B               2
```

**修复记录**：第一次冻结时 `admitted` 报 9 而集合实际 4 —— 原因是排除原因
链漏了 `is_geometric_gt_source` 这道门，`.modelInfo` 行被算成 admitted。
已加 `excluded_not_geometric_gt_source` 桶并补了一条**对账式**回归测试
（`admitted == len(LocalizationGTSet)`，而不是只校验分母求和）。
冻结是这种"报告与集合必须对账"最容易漏的环节，所以写进了测试。
