# J.3 Constraint-Integrated CrownOuter —— 约束可行域架构（vNext 主线）

用户 2026-10-09 决策 ②：**批准立项**，把 Neighbor Space 从"后处理形变"提升为
**构造时的可行域**，与 MinThickness Envelope 同级。

## 为什么必须改架构（本轮实测证据）

```
H.2.1 后 mesial 0.9950  →  J.2 后 mesial 1.0842     （run h22l_chain）
B+H.2.1 0.9676          →  J.2 1.0921               （上一链，独立复现）
```

两次独立复现：**J.2 的 minimum-thickness envelope 不知道邻牙空间**，出形后按
自己的厚度下限把已收窄的 proximal 重新撑进邻牙。只把 Neighbor Constraint 放进
Anatomy Prior 不够 —— 它会被随后的 envelope 撤销。

## 目标模型

```
          Anatomy Prior
                │
                ▼
┌────────────────────────────────┐
│      Feasible Crown Volume      │
│  Intaglio / MinThickness（下界）  │
│  Neighbor Space（上界）           │
│  Antagonist Space（上界）         │
│  Cervical / Margin（边界条件）     │
└────────────────────────────────┘
                ↓
           Final Outer
```

* ``MinThicknessVolume`` = **下界**（必须至少这么大）；
* ``AllowedNeighborSpace / AllowedAntagonistSpace`` = **上界 / 禁止体**；
* 二者无交集 → ``CONSTRAINT_CONFLICT`` —— 这才是当前 FDI46 真正应该被发现的状态，
  而不是"先生成 → 再反复 deformation → 修不动 → 最后才发现空间不够"。

## 第一件（已完成）：Feasibility Audit —— 生成之前的秒级判定

`app/geometry/crown_outer/feasibility_audit.py` + `tools/run_j3_feasibility_audit.py`

逐点判两个界：

```
min_required_outer(p) = intaglio(p) + T_min · n̂(p)     （沿内表面外法向，t_ray 同定义）
max_allowed_outer(p)  : neighbor/antagonist 带符号间隙 ≥ 目标间隙
min_required 侵入允许空间（signed < clearance） → LOCAL_CONSTRAINT_CONFLICT
```

* 阈值全部来自病例 `die_ref.json`（`.modelInfo` 设计参数）：
  MinThickness / CementGapThickness / DistanceToNeighbor / DistanceToAntagonist；
  读不到**报错不猜**（引擎不写临床默认值）。
* 外法向定向：winding 壳外/壳内探针（项目自己的 `winding_number`），固定种子
  抽样子集（≤800 面）；残差面用凸性兜底并如实计数（歧义面数进报告）。
* 固定随机种子 → 确定性；只测不生成不求解。

### FDI46 实测（第一个应该被秒级识别出的病例）

```
病例 cf_b1b73bae221211b1 | MinThickness=0.60 | DistanceToNeighbor=0.00 | DistanceToAntagonist=0.10
== J.3 Feasibility Audit（7.7s，n_samples=6000）==
| 空间        | 查询点 | 冲突点 | 超噪冲突 | 侵入 p50/p95/max (mm) |
| 邻牙 mesial | 6000  | 689  | 666     | 0.622 / 1.177 / 1.314 |
| 邻牙 distal | 6000  | 630  | 601     | 0.427 / 1.113 / 1.453 |
| 对颌        | 6000  | 966  | 928     | 0.523 / 1.298 / 1.594 |
判定：CONSTRAINT_CONFLICT
```

**7.7 秒**（抽稀探针前 137s），判定与全量探针逐位一致（666/601/928 点、
p50/p95/max 三位小数相同）—— 快速路径可信的直接证据。对比此前要跑完
H.2.1 → J.2 → K0 → Contact 整链约 40 分钟才知道无解。

### 与当前链的关系（如实说明）

* 邻牙冲突 = 本案 SPACE_CONSTRAINT_CONFLICT 的根因，也是 J.3 的判据核心；
* 对颌冲突在**当前**管线里属 Occlusion 阶段的 handoff 项（Occlusion v1 专门修
  咬合），J.3 里它与邻牙空间同级纳管；现阶段审计两者都报，读报告时按空间分列。
* 内表面来自 `intaglio_from_prep(prep, CementGapThickness)`（与厚度引擎同源）；
  本工具默认读病例包的 `_intaglio_for_chain.stl`（链路同一份）。

## 后续（本文件是 J.3 主文档，逐步补）

1. ✅ Feasibility Audit（生成前判定，秒级）；
2. ⬜ Feasible Crown Volume 构造：`MinThicknessVolume ∩ AllowedNeighborSpace
   ∩ AllowedAntagonistSpace + Anatomy Prior → Final feasible anatomy`；
3. ⬜ 与 J.2 的关系：最低厚度 envelope 与邻牙上界在**同一层**求交（而不是先后
   相乘再回修）；
4. ⬜ 病例级状态机接入：`CONSTRAINT_CONFLICT`（生成前）与
   `REVIEW_CONTACT / SPACE_CONSTRAINT_CONFLICT`（生成后）分开记。

## 与现有冻结件的关系（纪律）

* 未改 H.2.1 / K0 / J.2 / Contact Gate 的任何验收判据；
* J.3 是**新增层**，不替换已冻结的 J.2 验收 —— 冻结件继续按原验收跑；
* 本轮之前的 E2E 主线（Contact → Occlusion → Thickness → Stitch）继续用冻结件，
  J.3 成熟后再决定替换点。

---

## ⚠️ 重大更正（2026-10-09 晚）：patch 作用域污染 —— "SPACE_CONSTRAINT_CONFLICT" 被推翻

**发现**：bundle 的邻牙 patch 由 `grow_neighbor_patch.py` 做**半径 9mm 的整段牙弓
区域生长**（实测 u 跨 −16..+15mm），把**目标牙自身位置/颈缘**也包含进去。用病例
自己的**临床已交付设计冠**（`final_crown.stl`，GT control）一测就露馅：

| 参照 | GT 设计冠的"穿透" |
|---|---|
| 原 patch（整段牙弓） | mesial p95 **1.109** / distal **1.065** mm（> 0.2 允许值） |
| **近接大陆**（\|lat\|≤1.5，距 MD 远端 1.2mm 内） | **0.000**（间隙 +1.07 / +1.83mm） |
| **wall-only patch**（只留邻牙朝向本牙的近接壁） | **p95 0.000**（min 间隙 +1.19 / +0.74mm） |

**同一测量下我们生成的 H.1/J.2 也不穿透**（wall-only：H.1 最小间隙 +1.19mm、
J.2 p95 0.000）。⇒ 该案此前从 K0 → H.2 → H.2.1 → H.2.2 → Contact 一路量的
"~1.1mm 近中穿透"**是 patch 污染伪值**：GT 冠自己都过不了那把尺。

**J.3 可行性审计（同一工具，换参照）**：

```
原 patch:    邻牙 mesial 666 点 / distal 601 点冲突（p95 1.18/1.11）→ "CONSTRAINT_CONFLICT"
wall-only:   邻牙 0 点 / 0 点冲突；只剩对颌 928 点（p95 1.30）= 咬合过突，
            属 Occlusion 阶段 handoff 项，不是空间冲突
```

⇒ **J.3 的机制成立（秒级判定 ✓），但前提是"上界"必须来自正确裁剪的邻牙表面。**

### 新增前置 QC（硬前置）

`tools/audit_neighbor_patch_scope.py`：用病例自己的 GT 冠作对照量 patch ——
GT 冠读数 p95 > 0.2mm（= Contact Gate 的允许值）⇒ `PATCH_SCOPE_CONTAMINATED`，
patch 不能作为接触参照；`run_h2_2b_chain.py` 已把它接成**链首前置门**
（不通过则不下游、写 `blocked_at=PATCH_SCOPE_QC`，几秒出结果）。

### 对架构的直接影响

* `AllowedNeighborSpace`（上界）**必须**由"邻牙朝向本牙的近接壁"构成 ——
  整段牙弓 patch 会把目标牙自身的原牙/临时冠/颈缘算成"禁止体"，
  于是任何合法冠都被判穿透。**这是 J.3 的第一条输入纪律。**
* 对颌 patch 也要做同样的作用域 QC（本轮未做，如实记录）。
* 结论修正：FDI46 的 `REVIEW_CONTACT` **不是因为患者空间不足**，
  而是**病例包 patch 作用域缺陷**；`SPACE_CONSTRAINT_CONFLICT` 标注为未证实。

### 下一步（明确任务）：重新生长 patch，而不是本地裁剪

本轮试过两条本地裁剪规则都**不成立**（如实记录）：

1. "去掉 GT 冠体内的面"：只去掉 11–13%，GT 仍读 p95 1.32/1.29mm ——
   污染面在**冠体外但与冠面近平行**（目标牙自身的扫描表面/颈缘）。
2. "只留 u ≥ GT 极端 − 1.0mm 的邻牙近接壁"：内边界留 1mm 时 mesial 侧仍剩
   一簇深穿（GT p95 0.799、49 点，22% 最近点在 GT 体内）—— 因为在真实口内
   **邻牙壁与目标牙自身的扫描面相邻**，一个平面切不开；裁到贴边又会让
   最近点落进 patch 边界安全带（AMBIGUOUS_PATCH_EDGE）把所有证据剔空。

⇒ 正确做法是**在源牙弓上重新生长**（服务器侧 `grow_neighbor_patch.py` 的
作用域修复）：种子选在**邻牙朝向本牙的近接区**（而非"离冠最近的三角形"——
那可能落在目标牙自身表面），生长时排除目标牙侧（按到 margin 环/代型的距离
或按扫描牙列的分齿连通域），并且**保留足够内伸**让接触区远离 patch 边界。
验收：病例自己的 GT 冠在该 patch 上读数 p95 ≤ 0.2mm（`audit_neighbor_patch_scope.py`）。

---

## P1 进展（2026-10-09 第二轮）：作用域规则 + 前置 Gate + 操作侧提取工具

按用户锁定的新主线（M1 = 可靠病例输入 → Contact PASS），本轮落地：

### 规则（`app/geometry/crown_outer/patch_scope.py`，numpy）

参考 = **代型/内表面**（永远可得，**不依赖 GT**；GT 只作验收对照）。
逐面：① 距参考表面 ∈ [0.05, 4.0]mm；② 颈缘平面以上 ≥ 0.5mm；
③ 本侧参考极值 ±（内留 0.6mm）；④ 朝向就位轴（dot > 0.10）；
⑤ 冠体内排除（给了 `--crown` 才做 clamp）；⑥ 取最大连通块。

### 前置 Gate（`verify_patch()`）

GT 冠在 patch 上 **穿透 p95 ≤ 0.2mm** 且**近接大陆被覆盖**（顶点距离判覆盖，
不受开放 patch 边界安全带影响）→ `PATCH_OK`；否则
`PATCH_SCOPE_CONTAMINATED` / `PATCH_TRUNCATED` / `PATCH_UNUSABLE`。

### 操作侧提取工具（`tools/rescope_patch.py`，纯 stdlib，服务器/本地两用）

A组（工作颌段）→ 由规则生长出 mesial/distal patch；B组 + 冠体 → antagonist patch。
与 numpy 版规则同参，在 FDI46 上**交叉验证一致**（mesial 1198 vs 1193 面；distal 276 = 276）。

### FDI46 验证（候选 = 现有牙弓派生 patch；源牙弓已不在语料，见下）

| 对照 | mesial | distal |
|---|---|---|
| **GT 设计冠** | **PATCH_OK**（穿透 p95 0.000，大陆覆盖 602 点） | **PATCH_OK**（p95 0.000，覆盖 456 点） |
| H.1 | PATCH_OK（零穿透，覆盖 166） | 无有效证据（patch 只盖到 GT 的壁） |
| J.2 | PATCH_OK（零穿透，覆盖 386） | 无有效证据 |

⇒ 干净 patch 下 **GT 与我们生成冠在近接大陆都是零穿透** ——
`SPACE_CONSTRAINT_CONFLICT` 正式撤销，改记 `PATCH_SCOPE_CONTAMINATED`（历史）。
distal 想让 H.1/J.2 也有有效证据，把 `d_max` 放宽到 5–6mm 再跑（下一轮）。

### 语料侦察（服务器，P3 用）

* FDI46（`82403260671`）源目录现在只剩设计冠 + `.dentalCAD` 等容器 ——
  **工作颌 A组 已不在语料**；`.dentalCAD` 是二进制私有容器（按纪律**不解析**）。
  ⇒ FDI46 的"从源牙弓重新提取"暂不可做；两条本地裁剪规则也都不成立（已记录）。
  该案按"受污染 patch 影响"归档即可，不再投入。
* **源资产完整、可用于主病例的两个病例**（都含 A组 工作颌 + FDI46 设计冠）：
  * `cf_3af96f22b1a79b5c` → `exo/EXO/82404140632/`（`A组 (1)(2)(3)` +
    `all_data/…-150778-A组-A3-46-crown_cad.stl` + constructionInfo）
  * `cf_bb24c6a1779cc561` → `exo/EXO/82404090822/`（`A组 (1)(2)(3)` +
    `all_data/82404090822-46-crown_cad.stl`）
* 服务器依赖 `/tmp/gt`（`app.groundtruth.assets` 等）**已被清理** ⇒ 跑 stage_prep/
  align_prep 前需先把 `backend/app` 上传回 `/tmp/gt`。

### P1 剩余（下一轮）

1. 上传 `backend/app` → `/tmp/gt`；对 `82404140632` 跑：stage_crown_outer_case →
   stage_prep_case → align_prep_to_design（得到设计系代型）→ `rescope_patch.py`
   从 A组 提取双侧 patch（d_max 5–6mm 试）→ **对颌同规则提取** →
   本地 `audit_neighbor_patch_scope.py`（GT control + 覆盖性）通过后入库。
2. 把该病例包作为 E2E 主病例：`GT → Patch QC → H.1/J.2 → Contact Audit`（快验），
   再按锁序 Contact → Occlusion → Thickness → Final Stitch。
