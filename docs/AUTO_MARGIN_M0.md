# M0 AutoMarginGTSet —— 数据集构建（只做数据集与 QC）

**日期**：2026-09-29。代码 `backend/tools/build_auto_margin_gt.py`（纯 stdlib，
服务器跑）。**本轮不做**：不训练模型、不进入 M1、不改 Contact（用户规格）。

## 为什么值得做

`MarginGTSet`（3498 行 / 1863 例，**全部 canonical constructionInfo**）是项目
现成、且最值钱的数据资产。把它变成可训练、可审计的 face-level 分割真值，
比继续做局部几何调参对主线的推动更大。

## 标签语义（用户规格 2）

**face-level 分割**，不是"Margin 顶点 = 1"（极不平衡）：

```
Class 1 = Margin 冠方 / 咬合方区域（预备体被牙冠覆盖的那部分表面）
Class 0 = Margin 颈方区域（代型本体 / 模型段）
Margin  = Class0 与 Class1 的边界
```

## 漏斗（前四层已在 16 例探针上验证）

| 层 | 数量 | 说明 |
|---|---:|---|
| `MarginGTSet` | 3498 行 / 1863 例 | 全部 CONSTRUCTION_INFO |
| 有代型且**坐标系已验证**（`modeldie_same_frame`） | **2546 行 / 1595 例** | Prep mesh 明确 |
| 投影可读（审计阶段） | 待全量 | |
| 拓扑切分成功 | 待全量 | 探针 13/16 |

## Prep mesh 的来源与投影实测

Prep mesh = `.constructionInfo` 里该牙位的 **modeldie**（代型），按
`modeldie_asset_id` 的**内容哈希**在病例目录里定位（不按文件名猜）。

投影实测（已用暴力最近点交叉验证，索引与暴力差 0.00e+00）：

* **Margin 点精确落在代型表面上**（距离 ≈ 0）—— exocad 代型由设计生成，
  Margin 是设计的一部分；
* 但**只有约 77% 精确落在网格顶点上**（413 点里 320 个），其余在三角形内部
  （到"面"距离同样为 0）。这个细节决定了切断算法必须按**面**而不是按顶点。

## 切断 adjacency（用户规格 4）

切割带 = 每个 Margin 点**所在面** + 该面三个顶点的 incident faces，整体从面
邻接图移除。

**踩的坑**：只按"匹配到的网格顶点"切断会留下断口 —— 顶点环的度分布是
`{2:192, 1:114, 0:14}`（不是闭合环），切完网格仍然连通 → 误报 bypass。
改成面基后：切割带 2249 面 / 3.6%，代型分成 `52727 + 7698 + 2 个孤立单面`。

## 冠方/颈方判定（三次踩坑，已固化）

判据不能用这三个（都实测反过）：

1. ~~取最大分量~~ —— 代型/模型段的基础部分常占多数面（fdi=14: 52721 vs 7698）；
2. ~~取轴向最高面~~ —— 基座/邻牙外壁可以比预备体顶还高（58.75 > 55.53）；
3. ~~带邻面轴向偏移符号~~ —— 切割带 2~4 层厚，两侧带邻面都在 Margin 外侧，
   偏移**都为正**（10/14 变成 side_ambiguous）。

**采用**：`Class0 = 含全局轴向最低面的分量`（代型/模型段的底面必然在轴向
最低处，与"哪边面多 / 哪边更高"无关）。
**交叉验证**：`Class1`（冠方 cap）不得延伸到 Margin 环轴向最低值以下
（cap 坐在 Margin 上面），超差 `MARGIN_CAP_TOL_MM=0.5` → fail-loud REJECT。

16 例探针结果：**13 OK / 3 REJECT**（2 个 `margin_not_separating` +
1 个审计拒绝）。OK 例的语义正确（fdi=14: c1=7700 小 cap / c0=52721 底座；
fdi=11/12 同样）。

## fail-loud 清单（用户规格 10 的重点）

| 情形 | 处理 |
|---|---|
| 切完后**仍连通**（能绕过 Margin） | `margin_not_separating(bypass_detected)` REJECT |
| 主区域数 ≠ 2 | `too_many_regions` REJECT |
| 无 InsertionAxis | `no_insertion_axis` REJECT（不猜方向） |
| cap 延伸到 Margin 最低值以下 | `cap_below_margin` REJECT |
| Margin 点落不到网格上 | `margin_off_mesh` REJECT |
| 投影超阈值 | `proj_p95/p_max_over_threshold` REJECT |

**绝不强行生成标签** —— 出问题的进 reject manifest，附原因。

## 产物（每例）

```
labels/<case>_fdi<N>_labels.bin     uint8/face：0=Class0, 1=Class1, 255=cut/未标
labels/<case>_fdi<N>_curves.json    原 GT curve + projected curve + 逐点误差
auto_margin_gt.csv                   case_uuid/group_id/fdi/qc_status/
                                     reject_reason/projection_metrics/
                                     topology_metrics（fragment_ratio 等）
split_manifest.json                  case/group-level 固定 70/15/15
funnel.json                          漏斗 + 每层 reject 原因分布
```

无 PHI：case_uuid 是目录名的哈希，原始文件名与路径不入库。

## 当前状态与下一步

* 全量审计（2546 行）在跑，约 3.4h；之后链式自动 build → funnel → split
  （链式任务已挂，无人值守）。
* **阈值不拍脑袋**：等审计的 p50/p90/p95/p99/max 分布出来再冻结
  `--max-proj-p95` / `--max-proj-max`（当前默认不设阈值 = 只记录）。
* M0 跑完后按用户要求：**先看 20~30 个随机病例的标签可视化**，
  再谈训练。标签错了，后面再强的模型也没意义。


---

## 追加：冠方/颈方判定的第四次修正（符号无关 + 三态）

前三个判据都实测反过之后，换成两个**与轴符号无关**的判据并互相验证：

* **A 测地跨度**：从切割带出发在分量内 BFS，面跳数最大者 = 颈方
  （代型本体/模型段要延伸到基座，cap 只是一只浅碟）；
* **B 面积**：面积小者 = 冠方 cap。

16 例实测：**7 OK / 6 REVIEW / 3 REJECT**。两判据一致的进 OK；
**不一致进 REVIEW**（标签按测地跨度写盘，但排除出训练集、留待看图）——
这正好落在用户要的三态（AUTO_ACCEPTED / REVIEW_REQUIRED / FAILED）上。

### 一次被自己拦下的错误尝试：Final Crown 当参考

曾想用 "Final Crown 内表面坐在预备体上" 作**定义性判据**。但实测该字段
指向的是 ``*_coping_cad.stl``（**内冠**），其表面同时靠近两个分量
（投票 23240/14095 = 62%/38%，不决定性）。**已整块摘除**，并在代码里
写明启用前必须验证的前提（内表面只落在 Class1 一侧 + "距 die < 0.3mm
才算贴附"的距离门限）。在验证前绝不允许它参与标签侧向。

### 踩坑清单（全部已固化进代码注释）

| # | 判据 | 为什么反了 |
|---|---|---|
| 1 | 取最大分量当冠方 | 代型基座常占多数面（fdi=14: 52721 vs 7698） |
| 2 | 取轴向最高面 | 基座/邻牙外壁可高于预备体顶（58.75 > 55.53） |
| 3 | 带邻面轴向偏移符号 | 切割带 2~4 层厚，两侧带邻面都在 Margin 外侧，偏移都为正 |
| 4 | 按匹配到的**顶点**切断 | 只有 ~77% 的 Margin 点落在顶点上，环不闭合 → 误报 bypass |
| 5 | Final Crown 当参考 | 该字段是内冠，两分量都靠近它（已摘除） |

## 可视化工具（M0 验收用，已就绪）

``render_auto_margin_qa.py``：随机抽 N 例，按面着色渲染两个视角
（沿 InsertionAxis 俯视 + 横向侧视）——**Class1 红 / Class0 蓝 /
切割带黄 / Margin 曲线绿**。绿线必须正好落在红蓝交界上，这是标签正确性
最直接的目视判据。已在小样本上跑通出图。

## 当前状态

* 全量审计（2546 行）进行中；链式任务（脚本 ``am_pipeline.sh``，已改为
  独立脚本文件避免 heredoc 吃换行）会依次跑 build → funnel → split。
* **待办**：① 审计分布出来后冻结投影阈值；② 侧向判据在 REVIEW 例上的
  人工确认（这批是当前最大的不确定性）；③ 看 20~30 例随机标签可视化。
* **未触碰**：Contact（按用户指示本轮不改）。
