# M0 资产可用性审计：`modeldie` 不是代型

> **后续（已落地）**：几何选代型 + AutoMarginGTSet v1 冻结已完成，见
> `docs/AUTO_MARGIN_GT_V1.md` 与 `docs/auto_margin_gt_v1_manifest.json`。
> 结论：候选池 329 行 / 139 病例 → 几何命中 130 行 / 48 病例 → 构建出
> **122 行 / 44 病例**的 v1 核心集（全部 DIE_LIKE + ABOVE_FRAC，proj_p95 全为 0，
> 40 例目视验收通过）。本文档保留为**根因证据链**，第 9 节的待决策项均已有结论。

**结论（一句话）**：`frozen_tooth_index.csv` 里 `has_modeldie / modeldie_asset_id`
对 **94.9%** 的 eligible 行来自 indexer 的**回退启发式**，实际指向的是**冠/桥
修复体 CAD 实体（冠壳 / 实心冠体）**，不是代型。在冠壳上按 Margin 切分，
分出的两侧是**外表面 / 内表面**，不是 M0 规格要求的 **冠方 / 颈方**。

→ 因此 **M0 的 build 阶段已停**（链式进程 `am_pipeline.sh` 已终止，全量审计继续，
它的数据本身有效）。在资产问题解决前，不产出任何标签 —— 否则就是把
"外表面 vs 内表面"当成"冠方 vs 颈方"喂进训练集。

本文档记录全部证据与判定方法，供决策。

---

## 1. 触发点：侧别判据四次反复矛盾

M0 构建器的"哪一侧是冠方"判据改了四版（测地跨度 / 面积 / 带相邻有符号偏移 /
Final Crown 参考），仍在 16 例探针里有 6 例 `REVIEW`。矛盾不是判据不够聪明，
而是**被切的网格根本不是代型** —— 前提错了。

## 2. 根因：indexer 的 modeldie 回退

`backend/app/groundtruth/indexer.py:343-366`：

```python
die_path = None
for p in singles:
    if "modeldie" in os.path.basename(p).lower() and tooth.tooth_fdi in fdis_in_name(p):
        die_path = p; break
if die_path is None and row.center is not None:      # ← 回退
    best = None; best_d = 1e9
    for p in singles:                                # 取"bbox 中心离设计中心最近"
        d = math.dist(probes[p].center, row.center)
        if d < best_d: best_d, best = d, p
    if best is not None and best_d <= 10.0:
        die_path = best
...
row.modeldie_same_frame = d <= 10.0                  # ← 因而恒真
```

两个问题：

1. **回退把修复体当成了代型**。exo2 病例目录里只有编号 STL（见 §4），没有
   `*modeldie*` 命名的文件，回退就挑"包围盒中心离牙位设计中心最近"的那个 ——
   而冠壳的 bbox 中心本来就离牙位中心 1~3 mm，必然入选。
   这正是"包里第一个 STL = Prep"那类规则的变体。
2. **`modeldie_same_frame` 在回退路径上是恒真条件**：`d <= 10.0` 与回退的入选
   条件是同一条式子，所以它**不携带任何坐标系信息**，却被下游当成"同坐标系"
   的证据（M0 的 eligible 过滤里有它）。

### 量化（`audit_modeldie_evidence.py`，只对命名含 modeldie 的文件做哈希）

| `modeldie` 来源 | 行数 | 占比 |
|---|---|---|
| **回退产物（无任何文件名证据）** | **2417** | **94.9%** |
| 指向真实的 `*modeldie*` 文件 | 129 | 5.1% |

回退挑中的文件实例（抽样）：

```
cf_529112a300a8e  fdi=14  →  -MY0089779-16-15-14-bridge_slm_cad.stl
cf_cedc4df646fe5  fdi=27  →  .82412200034-170062  3M2.stl
cf_20d419b898e07  fdi=36  →  .82412250320-36-crown_cad.stl
cf_6628c21c35299  fdi=46  →  111996-250513698401-46-coping_cad.stl
cf_b7b46065390e5  fdi=17  →  121338-250408556501-3m2.stl
```

全是修复体（`bridge_slm_cad` / `crown_cad` / `coping_cad` / 编号导出），没有一个是代型。

## 3. 几何证据：这些网格是闭合冠壳

对 `cf_0bfa2756db8a301f`(fdi=13) 与 `cf_486db93116ef079b`(fdi=17) 做逐项测量：

| 测量 | 结果 | 含义 |
|---|---|---|
| 开放边界边 | **0**（闭合） | 不是开放曲面 |
| 有符号体积 | 179.9 / 303.4 mm³（bbox 7.7×8.7×9.4） | 体积远小于同 bbox 实心块 → 薄壳 |
| 绕数 w(margin质心 ±1mm·轴) | **全部 ≈ 0** | Margin 处**零厚度**，两侧都不是材料 |
| 分量 #0 平均(法向·径向) | **−0.830 / −0.550** | 法向指向牙体轴 → **内表面（贴预备体那面）** |
| 分量 #1 平均(法向·径向) | **+0.788 / +0.555** | 法向背离轴 → **外表面** |

即：**外表面 + 内表面在 Margin 处接缝形成闭合壳**，这正是修复体的形态。
（`cf_0bfa2756db8a301f` 轴上 +3mm 处绕数=1，与"咬合面处内外表面收敛成实体"
一致，进一步支持壳的解释。）

## 4. 数据树里到底有什么（文件名普查）

`find /data1/welo/dentistry/exo -name "*.stl"` 的名称模式统计：

| 模式 | 文件数 |
|---|---|
| 编号导出（`N-N (N).stl` 等） | **2452** |
| `*-modeldie.stl` | 1074 |
| `*-modelbase.stl` | 524 |
| `*-crown_cad.stl` | 113 |
| `*-telescope_cad.stl` | 122 |
| `*-coping_cad.stl` | 33 |
| `*-bridge_slm_cad.stl` | 27 |

但按 **eligible 行的病例目录**统计，只有 **167 行**的目录里存在 `*modeldie.stl`
（68 个病例），**1708 行（67%）的目录里只有编号 STL**。也就是说：按名字可辨的
代型文件在这批 eligible 行里基本不存在。

另外 `exo2` 病例的 `.constructionInfo` 里 `ScanFiles` 声明的是
`UnsegmentedJaw` / `Antagonist`，但**这些扫描文件在导出目录里没有**（目录里
只有修复体壳）。所以"预备体所在的牙弓扫描"这批数据不提供。

## 5. 牙弓（full_arch）路线也不可用

| 指标 | 值 |
|---|---|
| eligible 行有 `has_full_arch_candidate=True` | 1145 |
| `target_center_surface_distance_mm` | p50 = **8.53**，p90 = 15.2，max 25.0 |
| `arch_asset_class` | UNKNOWN 1826 / TARGET_MISSING 396 / **WITH_TARGET 277** |
| `target_dist < 0.5 mm` 的行 | 47 |
| `target_dist < 2.0 mm` 的行 | 191 |

牙弓资产存在（748 个唯一），但**与 CI/Margin 不同坐标系**（中位偏差 8.5mm）。
这正是此前判定为 **GT leakage** 的那条恢复路线的根源（用 Final Crown GT 去把
牙弓配准回来，再用同一个 GT 评价 Contact Engine）—— 已降级，不重启。

## 6. 已实施的处置（M0 内部，未动冻结索引）

1. **`--stage kind` 资产形态审计**（新增）：逐行判定
   `DIE_LIKE / CROWN_SHELL_INTAGLIO / UNCLEAR` 并写
   `asset_kind_audit.csv`（含 `has_filename_evidence`）。
2. **构建时的 fail-loud 门**：`_build_one` 在切分后调用 `asset_kind()`，
   只要存在"内表面"分量就 **REJECT**，理由
   `asset_is_crown_shell(radial=[...],above=[...])`。
3. **判定规则抽成纯函数 `_classify_kind(stats)`** 并加单测
   （`tests/unit/geometry/test_asset_kind.py`，5 例），规则：
   - 有分量 `radial < -0.30` → **冠壳**（内表面存在）
   - 否则要求：冠方 `above_frac ≥ 0.80` **且** 颈方 `radial ≥ 0.35`（是环壁，
     不是顶/底盖）**且** 颈方 `ax_min ≤ −1.5mm`（真的向下延伸）→ **代型**
   - 其余 → **UNCLEAR**（排除出训练集）
   **保守偏置**：不确定一律 UNCLEAR，绝不把冠壳判成代型。
4. **切割逻辑抽成 `_prepare()`**，build 与 kind 共用同一份实现（切割带逻辑
   踩过 4 次坑，不允许两套实现分叉）。

### 形态审计首批结果（24 行，收紧规则前）

| 形态 | 行数 |
|---|---|
| `CROWN_SHELL_INTAGLIO` | 17 |
| `DIE_LIKE` | 4 |
| `UNCLEAR` | 2 |
| `REJECT` | 1 |

那 4 行 `DIE_LIKE` 经复查**全是假阳**（`bridge_slm_cad` 等实心冠体：两分量
radial 都 ≈ +0.1，是顶/底盖不是环壁）→ 已加"环壁 + 下探"判据修正（§6.3）。
收紧后同一批全样本：**0 例假阳**（原 4 例全落到 UNCLEAR）。

### 关键：真正的 `*modeldie*` 文件是教科书级代型（正样本 18/18）

对"资产哈希 = 目录里 `*modeldie*` 文件哈希"的 129 行跑 `--only-evidence`：

| 病例 | fdi | 形态 | radial[冠方, 颈方] | above[冠方, 颈方] | 文件 |
|---|---|---|---|---|---|
| cf_d5cdd15dfb61cbfa | 25 | `DIE_LIKE` | [0.723, 0.679] | **[0.952, 0.026]** | `82403200651-25-modeldie.stl` |
| cf_6fc88119791c22ed | 12 | `DIE_LIKE` | [0.834, 0.590] | **[0.914, 0.021]** | `82403200734-12-modeldie.stl` |
| cf_6fc88119791c22ed | 11 | `DIE_LIKE` | [0.860, 0.526] | **[0.932, 0.017]** | `82403200734-11-modeldie.stl` |
| cf_813609a295707193 | 21 | `DIE_LIKE` | [0.706, 0.753] | **[0.905, 0.020]** | `贺裕华…B组11-modeldie (5).stl` |
| cf_813609a295707193 | 43 | `DIE_LIKE` | [0.842, 0.800] | **[0.893, 0.024]** | `贺裕华…B组11-modeldie (15).stl` |

**18/18 全为 `DIE_LIKE`**，且数值形态高度一致：

- 冠方分量 **above_frac 0.885~0.974**（几乎全在 Margin 平面之上 → 预备体表面）
- 颈方分量 **above_frac 0.007~0.036**（几乎全在之下 → 代型颈部侧壁）
- 两分量 radial 都 **+0.45~+0.86**（都是环绕牙体、法向朝外的壁）

**这正是 M0 规格要的语义**：按 Margin 切开后，一侧是冠方（预备体），另一侧是颈方
（代型侧壁）。§6.3 里那两条"待验证假设"（`PREP_ABOVE_FRAC=0.80`、
`CERVICAL_WALL_RADIAL=0.35`）现在由 18 个正样本支持。

### 可用规模（`count_real_dies.py`，全索引 6132 行）

| 项 | 值 |
|---|---|
| 目录里有 `*modeldie*` 文件的病例 | 139 |
| 行级：modeldie 文件名含该 fdi | 236 |
| **其中且有 margin** | **234 行 / 131 病例** |
| 其中且有 constructionInfo + 有效轴 | 234 / 234 |
| 各病例贡献牙位数 | 1 牙位 86 例、2 牙位 19、3 牙位 10、4 牙位 8 … 最多 8 牙位 |

→ **AutoMarginGTSet 的已验证核心 = 234 行 / 131 病例。**
低于目标的 500~1000 例；且这是**本数据集走"代型"路线的上限**：其余 ~1500 病例的
目录里**只有修复体壳**（编号导出），根本没有导出代型 —— 不是索引挑错了，
是数据里没有。要扩量只能换资产类型或换数据源（见 §9）。

## 7. 另一个独立修复（顺带）

`closest_on_tri`（Ericson §5.1.5）在**退化三角形**上除零
（`t = d1/(d1-d3)`），真实扫描网格里会命中，会导致整行误判 REJECT。
已修为：先判面积，退化时退化为"投影到最长边"；其余除法加分母保护。
本地验证：共线 / 三点重合 / 正常 / 内部 / 钝角外侧 五例数值正确。

## 8. 审计数值真实性（在冻结阈值之前做的核验）

`projection_audit.csv`（全量 2516 行）形态：

- `proj_p50 / p95` 中位数 **0.0000**（字面 `0.0` 的行 1146/1192）
- `proj_p95 > 0.02mm` 的只有 **37 行（3.1%）**，且**坏例一坏就是 2~7mm**，
  全部 margin 点都在 0.2mm 外（`n_beyond_0p2 == n_margin`）
- 抖动核验（`verify_proj.py`）：原始距离 ~1e-6mm（float32 精度极限）；
  沿轴外移 0.05mm → 返回 p50=0.0428 / p95=0.0498；侧向外移 0.05mm →
  p50=0.0118（沿面滑动，几何正确）

**结论**：投影距离是真的（margin 点确实贴在网格面上），阈值只需做**粗差过滤**；
它在 0.05~0.5mm 之间取任何值结果几乎相同（33 vs 23 行）。
**投影阈值不能当质量过滤器** —— 真正的质量体现在拓扑/形态门。

## 8b. M0 已完成的部分（本轮的最终结果）

### 全量审计（2546 行，已完成）

`find_ci` 接受 `.modelInfo` 之后（原来只认 `.constructionInfo`），审计从
**238 个假拒绝**变成 **2545/2546 通过**（唯一真拒绝：`margin_missing_in_ci`）。

投影分布：`proj_p95` p50/p90/p95 = 0，p99 = 4.06，max = 10.53 —— 双峰：绝大多数
margin 精确贴在面上，少数（3%）差 2~10mm（帧不符/取错牙）。

### 真正的代型核心：234 行 → 92 行 OK

用 `--only-evidence`（资产 = 目录里 fdi 匹配的 `*modeldie*` 文件，不哈希）
构建 234 行：

| QC | 行数 | 说明 |
|---|---|---|
| **OK** | **92** | `asset_kind=DIE_LIKE`，**`proj_p95` 全为 0.0000**（margin 精确落在代型上） |
| REJECT `margin_off_mesh` | 121 | margin 完全不在该代型附近（取错牙/帧不符）→ fail-loud 拦下 |
| REJECT `margin_not_separating` | 12 | 切不开（bypass） |
| REJECT `proj_p95_over_threshold` | 6 | 投影超阈（9.8mm，回退冠壳资产） |
| REJECT 其他 | 3 | `geodesic_span_tie` / `asset_is_crown_shell` / `side_criteria_disagree` |

**92 行 / 39 病例**，切分 train 25 / val 8 / test 6（按病例分组，`split_manifest.json`）。

### 侧别判据的根因修复（这是前四版反复矛盾的真解）

代型上用 **`above_frac`**（哪一侧在 Margin 平面之上）做决定性判据：

- 18 例正样本：冠方 0.885~0.974 vs 颈方 0.007~0.036（差 1~2 个数量级）
- 92 行 OK 全部 `side_method=ABOVE_FRAC`
- **`geo_criteria_agree = False` 92/92** —— 也就是说旧的"测地跨度/面积"判据
  在这 92 行上**每一行都会判反**。它不是"不够聪明"，是在代型这个形态上
  **方向就是错的**（它是为冠壳设计的）。四版判据的反复矛盾到此结束。

### 目视验收（用户规格：M0 后先看 20~30 例可视化）

`render_auto_margin_qa.py --n 24` → 4 张图册（`/tmp/automargin/qa_render/`）。
每例两视角：沿 InsertionAxis 的俯视 + 侧视。

**检视结论（已逐张看）**：红（Class1=冠方）为**预备体**（俯视图中居中），
蓝（Class0=颈方）为**代型外壁**（环绕成环），**绿线（Margin）在每一例都正好
落在红蓝交界上** —— 符合验收标准。23/24 例渲染成功，1 例渲染器报
`list index out of range`（渲染器 bug，非标签问题）。

## 8c. 121 行 `margin_off_mesh` 的根因已查明：按文件名匹配 fdi 不可靠

`diag_offmesh.py` 抽样 60 行（判据：margin 质心 vs 代型网格质心的距离 / bbox 包含）：

| 判定 | 行数 |
|---|---|
| **A 取错牙（margin 在 die bbox 之外）** | **60 / 60** |
| B 框内但投影不到（坐标系/尺度不符） | 0 |

质心距离 **20.8 ~ 49.4 mm**（邻牙只有 ~8mm，所以不是"取了邻牙"，是取了**别的牙位
或别的坐标系**的代型）。这些 die 网格本身是正常单颗代型（bbox ≈ 5~8 × 5~8 × 15mm）。

**根因**：`evidence_rows` 按 **文件名里是否含该 fdi** 选代型，而一批病例的文件名
长这样：

```
贺裕华82403200782-160439- B组11-modeldie (5).stl
贺裕华82403200782-160439- B组11-modeldie (12).stl
```

病历号 `82403200782`、订单号 `160439`、组名 `B组11`、括号序号 `(5)/(12)` 混在一起，
**同一病例的多颗代型靠文件名区分不开** —— 于是不同牙位落到同一个文件、或落到
别的牙位的文件。这正是用户警告过的"不要靠文件名猜"的又一个实例。

**正确修法（已验证方向，未实施）**：在病例目录的候选代型里**用几何选** ——
把该牙位的 CI margin 投影到每个候选 `*modeldie*` 文件上，取投影误差最小者；
若最小误差仍 > 阈值则 REJECT。这条规则自带验证（投影误差就是判据），不依赖命名。
预期效果：把可用核心从 92 行推向 234 行候选的上限，且顺带能救回那些名字里
根本没有 fdi 的病例。

## 9. 待决策（需要你定）

**已由数据回答的部分**：真正的 `*modeldie*` 文件**是**代型（18/18 DIE_LIKE +
92 行 proj_p95=0 + 目视通过），所以 (a) 可行：**M0 的可建核心 = 234 行候选 /
92 行 OK / 39 病例**。`.modelInfo` 的解析路径也已补上（`find_ci` 接受两种后缀）。

剩下的决策：

1. **资产源扩容**：
   - (a) 接受 **92 行 / 39 病例**作为 M0 起步集（已验证正确，但远低于
     "500~1000 个干净病例"）—— 并先把 121 行 `margin_off_mesh` 的原因查清
     （取错牙还是帧不符？取错牙的话是 fdi 名称匹配的边界问题，可修）；
   - (b) 修 indexer：回退路径**不得**写 `modeldie_*`，改为 `nearest_asset_*`，
     并给每个资产加 `asset_kind`（闭合性/内表面/环壁），下游按 kind 过滤；
     这会**改冻结产物 `exocad_gt_v1`**，需要重新 freeze；
   - (c) 换数据源（例如真机扫描的`UnsegmentedJaw`牙弓 + 已终止的 Target Locator
     路线，或 3Shape 直读牙位）。
2. **M0 是否改建在牙弓上**：191 行 `target_dist < 2.0mm`（47 行 < 0.5mm）里的牙弓
   若确认"预备体确实在扫描面上"，那才是 AutoMargin 的真实输入分布；
   但需要先解决配准（且不能引入 GT 泄漏）。
3. **投影阈值冻结值**：建议 `--max-proj-p95 0.20 --max-proj-max 0.60`
   （分布上 0.05~0.5 无差别；取 0.2 丢掉 25 行明显坏例）。
4. **是否接受 39 病例作为 M0 起步规模**：这是走代型路线在本数据集上的上限
   （234 行候选里 92 行过关）。若要更多，需要换资产类型（`modelbase` 524 个
   文件 —— 模型基座通常不含预备体表面，待核）或换数据源。用户规格 9 是
   "质量优先于数量，500–1000 个干净病例足够"，39 例明显偏少，但**每一例都
   经过几何 + 目视双重验证**。M1 几何基线可以先在这 39 例上跑出方法学结论，
   规模问题与数据源决策并行。

## 10. 复核命令

```bash
# 审计进度
tail -1 /tmp/am_audit.log

# 形态审计（含 fail-loud 门）
python3 /tmp/gt/build_auto_margin_gt.py --stage kind --out /tmp/automargin --limit 40

# 投影距离真实性（抖动核验）
python3 /tmp/gt/verify_proj.py --any 0

# 单案例目录里所有 STL 的形态
python3 /tmp/gt/probe_dir.py cf_0bfa2756db8a3 13

# 只看"资产=真实 *modeldie* 文件"的行（正样本路线）
python3 /tmp/gt/build_auto_margin_gt.py --stage kind --out /tmp/automargin     --only-evidence --limit 18

# 可用规模清点
python3 /tmp/gt/count_real_dies.py
```
