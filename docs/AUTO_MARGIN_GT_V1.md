# AutoMarginGTSet v1（已冻结）

**规模：122 行 / 44 病例**，切分 train 27 / val 8 / test 9（group-level）。

标签是**面级二分类**，做在**真代型**网格上：`Class1 = Margin 之上（冠方，预备体表面）`、
`Class0 = Margin 之下（颈方，代型外壁）`、`255 = 切割带与碎片`（不参与训练）。
冻结 manifest：`docs/auto_margin_gt_v1_manifest.json`（PHI-free，只含 case_uuid 哈希与计数）。

## 这一版为什么推翻了之前的做法

上一版走完全程才发现**前置假设是错的**：索引里的 `modeldie_asset_id` 有
**2417/2546 = 94.9%** 来自 indexer 的回退启发式（bbox 中心离设计中心最近的 STL），
实际拿到的是**修复体 CAD 壳**（crown / coping / bridge_slm）。在冠壳上按 Margin
切割，分出来的两侧是"外表面 / 内表面"，**不是**规格要的"冠方 / 颈方"。

这也是"侧别判据改了四版仍在 16 例里矛盾 6 例"的真正原因 —— **不是判据不聪明，
是被切的网格不是代型**。全部证据见 `docs/AUTO_MARGIN_M0_ASSET_AUDIT.md`。

## 资产：几何选代型（不再靠文件名）

```
候选池   : 病例目录里有 *-modeldie.stl 的病例 → 329 行 / 139 病例
选择规则 : 对同病例所有候选代型（按内容哈希去重）逐个投影该牙位 Margin，
           取投影误差 p95 最小者；p95 <= 0.20mm 才算命中
命中     : 130 行 / 48 病例（err_p95: p50 = 0、p90 = 0、max = 0.0273 mm）
```

**为什么必须用几何而不是文件名**：文件名里病历号/订单号/组名/括号序号混在一起
（`贺裕华82403200782-160439- B组11-modeldie (5).stl`），同病例多颗代型区分不开 ——
实测按文件名匹配的 121 行 `margin_off_mesh` 抽查 60/60 全是"取错牙"（Margin 到
代型距离 20.8~49.4mm）。投影误差自带验证，不依赖命名。

同一份选择表也交给**渲染器**（`--die-selection`），保证"标签在哪颗网格上"与构建
逐字一致（此前踩过"标签在代型、网格画冠壳"）。

## 冻结的阈值

```
max_proj_p95 = 0.20 mm
max_proj_max = 0.60 mm
```

这两个是**粗差门控**（分布双峰：96.9% 的样本投影精确为 0，坏样本直接跳到 2~10mm，
0.05~0.5 之间几乎没有灰区）。真正决定数据质量的是拓扑 + 语义 QC，不是这两个数。

## QC（全部冻结在 manifest 里）

| 项 | 值 |
|---|---|
| `asset_kind` | **DIE_LIKE 122/122**（无一个冠壳；形态门是 fail-loud 的） |
| `side_method` | **ABOVE_FRAC 122/122** |
| `proj_p95` | p50 = p90 = **max = 0.0000 mm** |
| `fragment_ratio` | p50 0.1035 / p90 0.1402 / max 0.1798 |
| 未标面占比 | 224865 / 2966204 = **7.6%**（切割带 + 碎片，如实上报） |
| Class1/Class0 面积比 | p50 **0.38**（预备体表面 < 代型外壁，符合解剖） |

### 标签正确性的四重独立验证

1. **above_frac**：Class1 中位 0.928 vs Class0 0.011（差 1~2 个数量级）
2. **面积**：116/116 行 Class1 面积更小（预备体 ~85~120 mm²，代型外壁 ~190~340 mm²）
   —— 注意**不能用面数**判：预备体是扫描面、致密，面数反而是代型外壁的 3 倍
3. **极值**：沿轴最低点（代型基座）**60/60 在 Class0**；沿轴最高点 **60/60 在 Class1**
4. **目视**：40 例渲染（`qa_render/sheet_0..4.png`，8 例/张），
   **绿线（Margin）精确落在红蓝交界上**

> 渲染缩略图里"红色在下、蓝色在上"是**图像朝向**问题（exocad `<Axis>` 符号逐例不同），
> 不是标注错误 —— 上面 2、3 两条与朝向无关，已把语义钉死。

## 不在这 122 行里的（如实列出）

| 类别 | 行数 | 原因 |
|---|---|---|
| `BEST_STILL_OFF` | 101 | 该病例目录里**没有**这颗牙的代型（最好的候选 p95 仍 1.0~12.0mm） |
| `no_candidate_projects` | 98 | 没有候选代型能接受该 Margin 的一半以上点 |
| `asset_is_crown_shell` | 4 | 拿到的仍是修复体壳 → 形态门拦下 |
| `geodesic_span_tie` / `side_criteria_disagree` | 2 | 非代型资产上的旧判据问题 |

即：**本数据集走"代型"路线的上限就是 ~130 行 / 48 病例**，122 行是其中的干净部分。
要再扩量只能换资产类型或换数据源 —— 不对 `modelbase`（524 文件）下手，按用户指示
留作以后支线调查。

## 复现命令（服务器）

```bash
cd /tmp/gt
# 1. 几何选代型（4 分片并行，各写自己的 CSV）
for i in 1 2 3 4; do python3 build_auto_margin_gt.py --stage select \
    --out /tmp/automargin --shard $i/4 & done; wait
python3 merge_select.py            # 合并分片（自写脚本）

# 2. 构建（冻结阈值；用几何选择表）
python3 build_auto_margin_gt.py --stage build --out /tmp/automargin \
    --die-selection /tmp/automargin/die_selection.csv \
    --max-proj-p95 0.20 --max-proj-max 0.60

# 3. 切分 + 冻结 manifest
python3 build_auto_margin_gt.py --stage split --out /tmp/automargin
python3 freeze_v1.py

# 4. 人工验收（40 例图册）
python3 render_auto_margin_qa.py --out /tmp/automargin --n 40 --sheet 8 \
    --seed 11 --die-selection /tmp/automargin/die_selection.csv

# 5. 标签正确性复核（与朝向无关）
python3 check_base.py              # 基座/顶点极值判据
python3 check_areas.py             # 面积判据（不是面数）
```



## ⚠️ 需要拍板的政策问题：v1 的 Margin 全部来自 `.modelInfo`

| Margin 来源 | 行数 |
|---|---|
| `.modelInfo`（**模型坐标系**快照） | **122 / 122** |
| `.constructionInfo`（扫描坐标系） | 0 |

冻结的 GT 政策里 `is_geometric_gt_source()` **只认 `constructionInfo`**，所以这
122 行**不会**进入 `MarginGTSet` / `AxisGTSet` —— 那两个集合在 v1.1 索引里仍是
3498 行，与本集合不相交。

**为什么仍然认为这些行可用（但需要你确认）**：

1. **几何证据是直接证明**：这 122 行的 Margin 投影到代型上 **p95 = 0.0000mm**
   （最大 0.0273mm）。如果 Margin 来自"另一个设计阶段"（旧政策里 modelInfo 的
   依据是 3566 对拟合、rigid 残差中位 1.69mm），**不可能精确落在代型表面上**。
   而且抖动核验（把 Margin 点外移 0.05/0.20mm，投影距离同步变成 0.0428/0.1947）
   已排除"投影卡在 0"的假象。
2. **同帧是构造上的必然**：`.modelInfo` 写的是**模型坐标系**，而 `*-modeldie.stl`
   正是**模型的导出** —— 两者同属模型帧。旧政策的分歧（1.69mm）来自拿 modelInfo
   去比 constructionInfo（两个不同坐标系），对本集合不适用。
3. 每条记录都带 `margin_method` 字段，来源可追溯、可随时剔除。

**选项**：
- **(a) 接受**（推荐）：把这 122 行当作 v1，并在 manifest 里如实标注来源为
  `MODEL_INFO`。M0 有 122 行可用。
- **(b) 不接受**：只认 `constructionInfo` → 本集合**为空**，M0 需要换数据源
  （等 3Shape 直读或牙弓扫描），M1 无法开始。

这一条也解释了为什么 v1.1 索引里几个集合的规模变化很大：
`SupportGTSet` 2028 → **10**、`SingleAnatomicCrownGTSet` 177 → **3** ——
因为它们要求 `has_modeldie`，而代型"有证据"的门槛提高后，能过关的行大多住在
`.modelInfo` 病例里，被 `is_geometric_gt_source` 二次排除。**这不是回归，是
此前 94.9% 的假代型被清掉后的真实规模。**

## 下一步

M0 到此为止，**按用户定的顺序**：下一步不是训练，而是 **M1 几何 Auto Margin
baseline**（曲率/法向/高度/候选带/图搜索/样条，在这 122 行上回归）。
M2 AI 训练要等 M1 说明几何方法差在哪之后再说。
