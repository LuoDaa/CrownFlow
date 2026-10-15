# Blind v1 —— Final Blind 结果（M1_V2_PRODUCTION_CANDIDATE）

**数据集**：91 个全新独立病例 / 198 牙位（从 2068 个未使用病例筛出，见
`docs/BLIND_V1.md`）。**算法全程未改**：`git_commit 420823d`、
`config_id faces_K24|ridge6|sharp|dp|gate0.301/0.219x1.00|lam1.00/1.00/0.60|hub0.35|sec120|res240|band0.08|sm5`、
runner 字节 `d7a5e3f6…eee7f`（服务器实跑字节与仓库提交一致，已核对）。
生成 198/198 零崩溃，未使用逐行异常隔离；seal 锁住输入清单哈希 + 预测哈希 +
git commit + config_id + runner 字节哈希 + n_expected/n_written。

## 四层结果

### 第一层 Coverage（198 条）

| 状态 | 条数 | 占比 |
|---|---|---|
| AUTO_ACCEPTED | **155** | **78.3%** |
| REVIEW_REQUIRED | 43 | 21.7% |
| FAILED | 0 | 0% |

43 条 REVIEW **全部**是 `NO_FEASIBLE_CYCLE`（无可行闭环 → 不输出曲线）。

### 第二层 Accuracy given AUTO_ACCEPT（行权）

| 指标 | 值 |
|---|---|
| median | **0.056 mm** |
| p95 | **0.109 mm** |
| Hausdorff（中位） | **0.158 mm** |
| 点级 ≤0.20 / ≤0.50 占比 | 0.89 / 0.90 |
| 行级 p95≤0.20 | 128/155（83%） |
| 行级 p95≤0.50 | 140/155（90%） |

### 第三层 Case-weighted（病例内先取 median 再聚合）

91 病例中 78 例有 AUTO_ACCEPT：病例 median(median) **0.055**、
median(p95) **0.107**、p90(p95) **0.171**；病例 p95≤0.20 共 71/78、
p95≤0.50 共 75/78。

### 第四层 Safety（最关键）

> **AUTO_ACCEPT 中有 15 条 p95 > 1mm（占 AUTO_ACCEPT 的 9.7%）**，
> p95 达 9.7~12.7mm、Hausdorff 最大 15.8mm。

## 结论：**不能升级为 production candidate**

对照用户事先定下的冻结标准：

| 标准 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 可解率（AUTO_ACCEPT） | ≥90%（最好 >95%） | 78.3% | ❌ |
| 可解病例 p95 median | ≤0.20 | 0.109 | ✅ |
| 病例 p95≤0.50 | ≥90% | 96%（75/78） | ✅ |
| **AUTO_ACCEPT 中的 catastrophic miss** | **≈0** | **15 条（9.7%）** | ❌❌ |

**核心问题不是平均精度，而是安全性**：算法会在**自信**（全部 QC 通过）的状态下
输出一条离真实 Margin 约 10mm 的闭环。开发集上一条都没有（65/67 + 19/19 +
35/36 全部 p95≤0.5mm）——**blind cohort 暴露了开发集里不存在的失效模式**。

**适用范围（必须写清）**：91 个病例是从 2068 个未使用病例里，经"存在真实
modeldie + GT 可验证 + margin 同帧投影通过"筛出来的（**4.4%**）。所以本结论只在
"符合当前 modeldie 输入准入条件"的病例上成立，**不能**推广成"所有天然基牙病例
都能达到这个结果"。2068 → 91 这一步本身就是产品 coverage 的另一个维度：
真实产品还要回答"没有独立 modeldie 时怎么办"。

## 关键诊断：GT-free 置信度确实能区分这个失效模式

用**已经落盘、且不含 GT** 的置信度字段分组（15 catastrophic vs 140 good）：

| 组 | n | `chosen_mean`（选中候选平均排名） | `energy_gap` | `second_separation_mm` |
|---|---|---|---|---|
| catastrophic (p95>1mm) | 15 | **3.89** | **0.186** | **10.60** |
| good (p95≤0.5) | 140 | **1.88** | **0.496** | **2.81** |

三条读法：

1. **`energy_gap` 就是你坚持要我按"屏蔽 P1 候选邻域后重解"定义的那个量** ——
   catastrophic 组只有 0.186（第二解与最优解能量几乎相同 → **病例本身有歧义**），
   好组 0.496。**用户 §5/§13 的假设在 blind 上被实证确认。**
2. `chosen_mean` 3.89 vs 1.88：跑偏的环是**由深排名（低分）候选搭起来的**。
3. `second_separation` 10.60 vs 2.81：catastrophic 病例存在**另一条几何上极远、
   能量却几乎相同**的环 —— 算法在其中挑错了那一条。

`chosen_max` 单独**不能**区分（13 个好行也有 chosen_max≥20），所以要用
`chosen_mean` / `energy_gap` / `second_separation` 这类**分布性**信号。

## 下一步（按用户纪律，不在本次 blind 上调任何阈值）

1. **不做阈值调参**：本 blind 已被看过，任何阈值都必须用**新的 blind set** 验证。
2. 方向明确：**v4 = 把"歧义/置信"做成 GT-free 的拒绝规则**
   （`energy_gap` + `chosen_mean` + `second_separation` 的组合），
   把 15 条 catastrophic 路由到 REVIEW。**宁可 REVIEW，不可自信地错** ——
   AUTO_ACCEPT 率降到 70% 以下也是正确的交换。
3. 该失效模式在开发集里不存在（开发集 0/171 条），**因此开发集不能用来调这个阈值**
   —— 这正是需要新 blind set 的原因；下一轮扩容量应优先保证**出现这类歧义病例**。
4. **M2（learned candidate ranker）暂不启动**：几何侧已经"看得见"Margin
   （可解行 median 0.056mm），缺的是"知道自己在两条环之间犹豫"。

---

## 追加诊断（用户要求的首要一步）：15 条 catastrophic 的正确候选在不在？

**结论：候选生成无罪；失败拆成两种，其中一种是"正确环被硬门禁掉"。**

15 条全部逐扇区检查（池子 = 冻结配置下的全部候选，排名按冻结 score `sharp_deg`）：

| 检查 | 结果 |
|---|---|
| GT 候选召回 @0.10 / 0.20 / 0.50 mm | **15/15 全部 100%**（每个扇区都有正确候选） |
| GT 候选的分数排名 | p10=8 / **p50=11** / p90=13 / max=14 |
| GT 候选能组成**可行闭环**（过冻结硬门） | **9/15** |
| 被冻结硬门挡住 | **6/15**（GT 路径相邻 Δh 达 0.66~0.97mm，门仅 0.44mm） |

逐条（示例）：

```
cf_53b80b8d1d5 fdi=11  召回 1.00/1.00/1.00  排名中位 9   硬门 maxΔh 0.717/0.439 → 挡 ❌
cf_8a418105cb4 fdi=12  召回 1.00/1.00/1.00  排名中位 10  硬门 maxΔh 0.112/0.515 → 可行 ✅
cf_e5ee8547964 fdi=43  召回 1.00/1.00/1.00  排名中位 13  硬门 maxΔh 0.332/0.431 → 可行 ✅
```

### 因此 v4 必须同时做两件事（不是二选一）

1. **Safety / Ambiguity Gate**（对 9/15 纯选错有效）：
   用已落盘的 GT-free 信号 `energy_gap`（catastrophic 0.186 vs good 0.496）、
   `chosen_mean`（3.89 vs 1.88）、`second_separation`（10.60 vs 2.81）做**单调风险分数**，
   把可疑环路由到 REVIEW。**宁可 REVIEW，不可自信地错。**
2. **修 transition 硬门**（对 6/15 有效）：正确候选自身的相邻步长会超过"由 GT 点分布
   推出的门 + face_scale"。这正是用户 §7 的 MESH_AWARE 场景 —— 用 mesh 连通性
   （相邻候选在面图上是否局部连通 / 局部最短路长度）替代/放宽常数 h/r 门。
   不修它，这 6 条永远无法被**做对**，只能被拒绝。

### 仍然不需要 M2

正确候选在 **100% 的扇区、全部 15 条**里都存在（@0.10mm 召回都是 1.00），
且排名中位仅 11 —— 属于"存在但排得深"，是**排序/置信**问题，不是"几何看不见"。
所以顺序仍是：先 Safety Gate + transition，M2 继续挂条件（若 Blind v2 上仍出现
正确候选不存在的失效，再启动 Learned Candidate Ranker）。

### 阈值的处理纪律

本 blind 已被看过 → 只做**分析**，不调阈值。必须画
`risk threshold → AUTO_ACCEPT coverage → catastrophic recall → good-case false reject`
的安全-覆盖曲线（并对 `energy_gap` / `chosen_mean` / `second_separation` 看
min/p10/p25/p50/p75/p90/max 与重叠度，而不是只看组中位），任何冻结的阈值都要在
**Blind v2** 上验证。Blind v2 需按**纯 GT-free 风险分层**抽样，保证 cohort 里
有足够的高风险（多环歧义）病例，否则"没遇到危险病例"会被误读成"gate 成功"。

---

## ② transition 三模式验收（MESH_AWARE / HYBRID）

范围锁死：只替换"是否允许连接 + transition cost"；Candidate（faces Top-24）、
Score（sharp_deg）、Cyclic DP 主体、GT、Safety Gate、M2 全部未动；
`V2_FROZEN` 永久保留为 Champion baseline。

实现：`graph_search.SurfaceDistance`（face-dual 图上的**局部**最短表面路径，
半径截断 + `detour_ratio` 上限）+ `transition_cost` 统一入口。
夹具 4 条（`tests/unit/geometry/test_transition_modes.py`）：表面连续但面质心跳动
→ V2 禁 / MESH+HYBRID 允；隔缝相邻只能绕远（detour 16.7× > 上限 6）→ 必禁；
局部半径外不可达；HYBRID 仍挡 10mm 级跳跃。

### 验收第一项：6 条 GT-infeasible 的 GT 路径现在可行了吗 → **超出成功线**

15 条 catastrophic 的完整归类：

| 归类 | 数量 | 说明 |
|---|---|---|
| **A：transition 就是阻塞原因、已修且 DP 选对** | **3** | 见下表 |
| **B：transition 已修但 DP 仍选另一条环** | 3 | 归 ranking / Safety Gate |
| **B′：正确环本来就可行、纯选错（与 transition 无关）** | **9** | 同上 |
| C：GT 路径仍不可行 | **0** | transition 表达无剩余缺陷 |

| 行 | V2_FROZEN | MESH_AWARE / HYBRID |
|---|---|---|
| cf_53b80b8d1d5 fdi=11 | 不可行（**8 段被挡**）p95 **10.109** | 可行 → p95 **0.126** |
| cf_53b80b8d1d5 fdi=17 | 不可行（7 段被挡）p95 **12.700** | 可行 → p95 **0.168** |
| cf_53b80b8d1d5 fdi=27 | 不可行（1 段被挡）p95 **12.690** | 可行 → p95 **0.151** |
| cf_e5ee8547964 fdi=43 | 路径可行但 DP 选错 p95 10.510 | DP **改选正确环** → p95 **0.121** |

⇒ **4 条从 catastrophic 恢复到 0.12~0.17mm**；`MESH_AWARE` 与 `HYBRID` 在本批
数据上结论一致（均 0 段被挡），HYBRID 的额外保护由夹具锁住。

### 验收第二项：有没有把稳定好例放开到错误结构上（4 分片仍在跑，早期证据）

前几行样本：V2 p95 0.085~0.111 的行，MESH/HYBRID 给 0.059~0.224（一条略升，
仍远低于 0.50），**未出现新的错误结构**。另有若干 GOOD 行在 V2 下 GT 路径也有
1 段被挡（说明好行的 DP 走了别的候选），改成 MESH_AWARE 后 p95 反而更低。

### 结论与下一步

- **transition 修复正中根因**：class C = 0，且救回 4 条；剩余 12 条全部属于
  "正确环可行却被选错" → 下半程只需解决 **ranking / ambiguity**，不要再动 transition。
- **① 的数值全部作废、只保留形态**：P1/P2 变了 → `energy_gap` /
  `second_separation` 必须在新 transition 下**重新计算**，再由 Calibration Set
  冻结阈值，Final Blind v2 保持封存。
- 数据：这批历史语料的 modeldie 已被 Blind v1 吃干（2068 → 91，全用过），
  Calibration 与 Final Blind v2 必须来自新数据；若引入 modelbase，
  必须作为独立 domain 分层报告。

---

## ①′ 结果（2026-10-02）：transition 冻结为 HYBRID；Hard Dev 198 行全量重算

### 顺序（按用户锁死）：GOOD 全量回归 → 选 Champion → 对账 → ①′

**Transition Champion = `M1_V4_TRANSITION = HYBRID`**（models.py 常量）：

| 验收项 | 结果 |
|---|---|
| 6 条 V2 GT-infeasible 恢复 | **6/6**（逐条账见下；fdi=21 在 MESH/HYBRID 下 0 段被挡 → class C=0 成立） |
| GOOD 140 行回归 | **140/140 AUTO_ACCEPT、0 条新 catastrophic**；p95 中位 0.106 → **0.076** |
| REVIEW 变化 | 无新增 |
| MESH vs HYBRID | 全量**逐位一致** → 按用户倾向取 HYBRID（多一层宽松 sanity guard） |

6 条逐条账：`cf_53b80b8d1d5` fdi=11/17/21/27（V2 挡 8/7/9/1 段）与
`cf_e5ee8547964` fdi=33/42（挡 1/1 段）——在 MESH 与 HYBRID 下**全部 0 段被挡**。

### ①′-1 迁移矩阵（Blind v1 198 行，V2 → v4/HYBRID）

| V2 → v4 | n |
|---|---:|
| GOOD → GOOD | 140 |
| **GOOD → CATA** | **0** ✅（不合格线 = 0） |
| CATA → GOOD | 4 |
| CATA → CATA | 11 |
| REVIEW → GOOD | 1 |
| REVIEW → MID | 2 |
| **REVIEW → CATA** | **1** ⚠️ |
| REVIEW → REVIEW | 39 |

v4 总体：**GOOD 145 / MID 2 / CATA 12 / REVIEW 39**（REVIEW 里 31 条
NO_FEASIBLE_CYCLE + 8 条被 QC 规则拦下：transition 异常/未闭环/自交）。
accepted-only median 0.080。

⚠️ 一条 REVIEW → CATA：V2 下无可行环的行，HYBRID 打开可行性后 DP 挑了错环 ——
**这正是 Safety Gate 必须接住的类型**（可行性放开会放进来新错环）。

### ①′-2 三信号在新分组下的分布（**分离度变化 → 证明不冻结阈值是对的**）

| 信号 | GOOD (n=145) | CATA (n=12) | 判读 |
|---|---|---|---|
| `energy_gap` | p25 0.674 / p50 0.874（min 0.003） | 0.064~0.462（p90 0.452） | 主体分离、有重叠 |
| `chosen_mean` | p50 3.15（1.26 也有） | p50 3.90（1.26~4.54） | **严重重叠，单独无用** |
| `second_separation` | p50 2.477 / p90 2.665 / **max 12.734** | 8.171~12.303 | **①时的零重叠消失了** |

**关键**：① 时 `second_separation` 完美分离（GOOD max 2.95 vs CATA min 8.09），
换成 HYBRID transition 后 GOOD 里出现一条 12.73 的——**如果当时冻结了阈值就错了**。
用户"先冻结 transition 再重算 confidence"的顺序判断被实证确认。

### ①′-3 Safety-Coverage curve（阈值取 GOOD 分位，**未冻结**）

| rule | AUTO% | good保 | cat漏 |
|---|---:|---:|---:|
| none（v4 现状） | 80.3 | 145 | **12** |
| sep only（τ=GOOD q90） | 71.7 | 139 | 1 |
| **gap AND sep（τ=0.630/2.584）** | **72.2** | **140/145** | **1** |
| gap only（τ=GOOD q20） | 20.2 | 40 | 0（coverage 崩，不取） |

**Pareto 最优点：`gap AND sep` → cat漏 1、good 保留 140/145（96.6%）、AUTO 72.2%**。
从 12 条 catastrophic 压到 1 条，代价是 coverage 80.3% → 72.2%——完全符合
"宁可 REVIEW 不可自信地错"的优先级。

### ①′ 的结论

- **Gate 形态冻结（仅逻辑结构）**：`ambiguous = second_best_exists AND
  energy_gap ≤ τ_gap AND second_separation ≥ τ_sep`；`chosen_mean` 移出规则、
  只作诊断项（两组严重重叠，加进来只损失 coverage）。
- **τ_gap / τ_sep 不冻结**（本数据已开 GT）；且 transition 改变会重塑信号分布，
  阈值必须在新 transition + Calibration Set 上定。
- **v4 算法结构就此冻结**：faces/Top-24 + sharp + HYBRID transition + cyclic DP +
  structural second-best + Ambiguity Gate（形态）+ fail-loud QC。只剩两个阈值等
  **新数据**：Calibration 40%（开 GT 定值）/ Final Blind v2 40%（封 GT 一次性评估）/
  Future Hold 20%（留给 v5，不再把数据全部吃完）。
- 顺带修复 provenance 漏洞：`config_id` 指纹补入 `transition_mode`（此前
  V2_FROZEN 与 HYBRID 两次运行共用同一指纹）。
- 流程纠错记录：中途曾把开发语料 122 行的 HYBRID 重跑误当迁移矩阵（那里没有
  catastrophic，得出"0 CATA"的假象）；正确语料是 Blind v1 的 198 行。
