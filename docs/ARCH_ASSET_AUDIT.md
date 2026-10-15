# 提交 2.1 · 弓段/整颌资产审计（Arch Asset Audit）

**范围**：只做资产分类与诊断。**不碰** Target Locator、不碰任何牙冠生成算法，
也**不改** GT Set 规则 —— 那些要等 2.2 拿到真实分布之后再定。

对应代码：`backend/app/groundtruth/arch_audit.py`（纯 stdlib，可在数据服务器上就地跑）。
测试：`backend/tests/unit/groundtruth/test_arch_audit.py`。

---

## 1. 为什么不能再用 `bbox_max_extent >= 40` 判整颌

提交 2 用「最长边 ≥ `ARCH_MIN_MM = 40`」筛出 672 个"整颌候选"，但：

* 一座 3–4 单位长桥也能到 40–50 mm；
* 代型被抽走的 sectioned modelbase 同样在整颌尺度上；
* 于是 `LocalizationGTSet`（整颌 → 找目标牙的正样本）到底有多少真样本，**未知**。

而 `LocalizationGTSet` 有没有训练集，正是 Target Locator 路线能不能走的前提。
所以先把这件事验死：给每个资产一组**不依赖世界坐标**的几何描述，让数据自己分层。

## 2. 每个 STL 量什么

| 量 | 怎么算 | 为什么 |
|---|---|---|
| `triangle_count` / `size_bytes` | 文件头 + 文件大小一致性校正 | 基本盘；也用于判断文件是否被截断 |
| `bbox_extent_1..3` | 全部顶点包围盒，**升序** | 与旧判据可直接对照 |
| `principal_extent_1..3` | 顶点抽样 → 协方差 3×3 → Jacobi 特征分解 → 各主方向上的展布（max−min），升序 | **核心**：长桥是细长条，牙弓是马蹄形 |
| `second_plane_ratio` | `principal_extent_2 / principal_extent_3` | 马蹄形 ≳ 0.5；细长条 ≲ 0.2（合成件实测 0.11 vs 0.55+） |
| `third_plane_ratio` | `principal_extent_1 / principal_extent_3` | 牙弓"厚度方向"应明显小 |
| `surface_area` | 全部三角形 `\|u×v\|/2` 求和 | 判断是否只是薄壳/碎片的旁证 |
| `bbox_volume` | 三个边长之积 | 与面积一起看"实心感" |
| `components` / `largest_component_ratio` | 顶点并查集（面数 ≤ 预算时才算） | 牙弓 + 独立代型被合成一个文件时会有多个分量 |
| `target_surface_distance` | 目标牙 `<Center>` 到该资产**顶点**的最近距离 | 旧口径，保留以对照 |
| `near_dist_min / p10 / median`、`near_count_r2/r4/r6`、`near_coverage` | 以 `<Center>` 为心，半径内表面点的距离分位数与计数 | 见第 3 节 |

**面数超预算时不假装算过**：`topology_skipped=1`、`components` 留空。
（纯 Python 并查集在 0.3M 面上约 4 秒；真实整颌 0.3M~0.76M 面。）

## 3. 中心邻域占据度：两种"中心"可分

单看"最近距离"不够 —— 预备体还在、和"代型被抽走但中心挨着洞边"，都可能很近。
所以同时看**距离分位数**与**半径内顶点数**：

```text
预备体存在      → 中心附近有密集表面         → near_occupancy = surface_present
代型被抽走      → 中心空、外围一圈有模型面   → near_occupancy = void
3.5 ~ 6 mm 之间 → 不硬判                     → near_occupancy = ambiguous
```

半径（12 mm）与 4 mm 计数窗口都是**审计特征**，不是临床规则；`near_*` 全部写进 CSV，
阈值等 2.2 用分布定。

## 4. 元数据优先（关键）

CI 的 `ScanFiles/ScanFile/PartType` 是**真值**，实测取值：

```text
UnsegmentedJaw   ← 未分割整颌（我们要找的东西）
Antagonist       ← 对颌
BiteRegister / SituScan / MergedBridgeSLM
```

所以角色判定顺序是：

```text
1) ScanFiles.FileName 匹配到这个 STL        → metadata_role = <PartType>
2) 元数据里有但指向 .DCM（服务器读不了几何）  → 只给引用 + 上/下颌提示
3) 都没有                                    → 几何启发式（并在 basis 里标明）
```

**对颌（antagonist）按这条原则修**：不再"取最远的弓候选"，而是
`METADATA_ANTAGONIST` → 退到 `METADATA_ANTAGONIST_REFERENCE`（给出文件引用与侧别）
→ 最后才是 `GEOMETRY_FARTHEST_ARCH`。

> **实测的坑**：真值里 `UnsegmentedJaw` / `Antagonist` 指向的是 `*-lowerjaw.DCM` /
> `*-upperjaw.DCM`，而 DCM 需要 `hpsdecode + numpy + trimesh`，**数据服务器上没有**
> （见 `assets.py` 开头）。因此这一版把元数据当**标签**用，几何只对 STL 做。
> 换句话说：**"这个病例的整颌在哪"能由元数据回答；"它是不是包含目标预备体"目前只能靠 STL 几何回答。**

## 5. 资产语义（取代 sectioned/unsectioned/unknown 三态）

```text
FULL_ARCH_WITH_TARGET      完整牙弓 + 目标位置确有牙面   → 定位正样本
FULL_ARCH_TARGET_MISSING   完整牙弓 + 目标位置是空洞     → 不能做 Support 分割，但仍可用于
                                                          FDI 空间先验 / 邻牙与牙弓分析
PARTIAL_ARCH               弓段（一个方向为主）
MULTI_UNIT_RESTORATION     多单位桥 / 连冠
SINGLE_TOOTH               单牙
ANTAGONIST                 元数据明确标为对颌
UNKNOWN
```

每一行都必须带：

* `classification_basis`：`METADATA_UNSEGMENTED_JAW` / `METADATA_ANTAGONIST` /
  `GEOMETRY_FULL_ARCH` / `GEOMETRY_PARTIAL_ARCH` / `GEOMETRY_MULTI_UNIT` /
  `GEOMETRY_SINGLE_TOOTH` / `UNKNOWN`
* `classification_confidence`：离阈值 15% 以上算 `high`，落在带内算 `low`
* `classification_evidence`：ASCII 证据串（`max_extent=..;second_ratio=..;center=..`），
  只含 token 与数字 —— 索引里**不允许出现非 ASCII**，避免 PHI 混进来。

**缺 Center 的弓形尺度资产一律 `UNKNOWN`**，不写成 `FULL_ARCH_WITH_TARGET`：
承重假设必须保守。

## 6. 阈值：这一版是临时的，2.2 用数据换掉

`THRESHOLDS` 集中一处，并在输出里明确标注为 provisional：

```python
single_tooth_max_mm = 25.0        # 沿用提交 1/2 实测（单牙 7~17 mm）
arch_min_mm         = 40.0        # 同上（弓段 50~65、整颌 61~87）
arch_second_plane_min_ratio = 0.35   # 观测前的猜测，只为把分布切开
arch_third_plane_max_ratio  = 0.45
confident_margin            = 0.15
```

`arch_asset_summary.json` 里给 2.2 准备了两样东西：

* `per_class` / `all_assets_feature_quantiles`：各量的 p05/p25/p50/p75/p95/max
* `threshold_sweep`：把 `second_plane_min_ratio` 从 0.20 扫到 0.50，
  看"达弓形尺度且两平面都有展布"的资产数怎么变

**不要**现在拍 `e1 > 55, e2 > 30` 这种数字：先看分布。

## 7. 用法与实测性能

```bash
python -m app.groundtruth.arch_audit \
    --root /data1/welo/dentistry/exo/EXO \
    --root /data1/welo/dentistry/exo/all_data \
    --out /data1/welo/gt_index
```

常用开关：`--limit N`（调试）、`--max-assets-per-case 3`（每病例最多审计几个，
按体积从大到小）、`--no-topology`（跳过连通分量，大整颌上快很多）、
`--all-assets`（连单牙也量，看分布用）。

本地 15 例真实样本实测（单文件 0.3M~0.76M 面）：

| 模式 | 单资产耗时 |
|---|---|
| 单趟（包围盒+面积+主方向+网格，不建拓扑） | **0.8 ~ 1.8 秒** |
| 再加连通分量（0.33M 面） | +4 秒左右（超 0.4M 面自动跳过） |

优化前是 4.5 秒/资产（探包围盒、量特征、建网格各读一遍文件）；现在合成一趟读盘。

## 8. 输出

* `arch_asset_audit.csv` —— 一行一个候选资产，`AUDIT_COLUMNS` 就是表头（只追加，不改名）
* `arch_asset_summary.json` —— 分类/依据/置信度计数、各特征分位数、阈值扫描、临时阈值表

写文件前过 `assert_phi_free`：任何字段出现非 ASCII 直接**拒绝写出**。

## 9. 这一版**没有**做的事

* 不动 `models.py` 的 GT Set 谓词（`LocalizationGTSet` 仍按旧口径算）；
* 不读 DCM 几何（服务器无重库）；
* 不给 `LocalizationGTSet` 下结论 —— 那要等 2.2 全量跑完的真实数字。

## 10. 后续（已结案）

第 1 节提的"`LocalizationGTSet` 到底有多少真样本，未知"已有答案：
**它不能按本文档的 `FULL_ARCH_WITH_TARGET` 自动准入。** 两轮人工 QA
（[`LOCALIZATION_QA.md`](LOCALIZATION_QA.md) /
[`LOCALIZATION_QA2.md`](LOCALIZATION_QA2.md)）测得该类的实际精度只有
0.13～0.22（后者用 v3.2 格搜索 + margin 轮廓约束，仍为 0.222），因此
`FULL_ARCH_WITH_TARGET` 已降级为**必要不充分的预筛**，准入改由人工判定
（`app/groundtruth/human_labels.py`）决定。

**本文档的分类学仍然有效，而且是本轮结论的基础**：`FULL_ARCH_TARGET_MISSING`
（911 行）与 `FULL_ARCH_ANTAGONIST`（19 行）被正确地排除掉了，假阳性没有一例
来自这两类；问题全部落在 `FULL_ARCH_WITH_TARGET`（480 行）内部 ——
"整颌且目标牙在"并不等于"该处是预备体"。这也说明阈值/判据再怎么调都救不了：
**要区分的是"天然牙尖 vs 预备体"，而这需要的信息不在这个类的定义里。**
