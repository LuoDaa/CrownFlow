# P1：Prep Surface Contract / Conditioning 编排（2026-10-13）

范围（用户冻结）：**只做已有积木的编排与契约化** —— 不新增几何算法、不自动封洞/缝螺丝孔/
智能删组件/重建 prep。产出"哪些真实病例属于 1.0、哪些输入满足 crown-bottom chain 契约"的
**机器可判定结论 + 可信分母**。

## 1. 消费的现有积木（不复制判据）

| 用途 | 现有实现 |
|---|---|
| **权威契约谓词**（决定 `readiness`） | `geometry.nonplanar_crown.fitting_support_above_margin`（单一完整就位面 + 单一颈环；内部跑 `audit_margin_on_scan` 拒"Margin 跨开放边界"） |
| 只读诊断（组件数 / 边界环 / 只读源码面 id / Margin 到扫描距离） | `geometry.prep_margin_diagnostics.diagnose`（**从不删岛、不补洞**） |
| 资产选取原语（不在本批新增） | `geometry.roi.spherical_roi`、`geometry.implant_candidate.component_at_point` |

`readiness` **只由那个谓词决定**；本模块只把它的失败消息映射成拒绝码。**没有第二份判据** ——
否则半年后又会"diagnostics 说能用、intaglio 又拒绝"。

## 2. 契约对象与两条正交轴

`PreparedSurfaceContract`（`app/services/prep_conditioning.py`）：identity / geometry /
diagnostics / operations / classification / provenance，其中

```text
readiness = PREP_SURFACE_READY | REVIEW_REQUIRED | UNSUPPORTED_INDICATION | NO_SUPPORT_ASSET
rejection_class = READY | ASSET_PREPARATION_REQUIRED | REVIEW_REQUIRED | UNSUPPORTED_INDICATION
```

**关键设计**：`indication`（是否属于 1.0 范围）与 `readiness`（输入是否满足契约）是**两条正交轴**。
只有"明确不支持"才短路几何；`indication` 未判定时**照样给几何结论**（否则会掩盖真实拒绝码）。
`operations` 恒为 `NONE（本批不做任何几何修复）`，并断言输入网格字节不变。

1.0 支持范围冻结为 **Natural-tooth single anatomical crown**；以下一律 `UNSUPPORTED_INDICATION`：
`AnatomicPontic` / `ReducedPontic` / `Coping-only` / `Implant abutment`（含螺丝通道）/ `Bridge unit` /
`Antagonist design artifact` / `HealthyTooth` / `Unknown indication`。

## 3. 真实语料跑出来的数字（`数据/prep_conditioning_report.json`）

```
总病例 15 · 总单位 76 · eligible（天然牙游离单冠）13
by_indication: 天然牙单冠 13 / 桥体单元 16 / 解剖桥体 14 / 对冠设计稿 23 / 内冠 6 / 减径桥体 2 / 健康牙 2
契约尝试：19 个病例包里只有 2 个带 prep.stl（17 个无支撑资产）
契约结果：2/2 = REVIEW_REQUIRED · MARGIN_CROSSES_OPEN_BOUNDARY（B 类：资产准备）
eligible 但无资产：29 → 13 + 17 个包（含 3 例无元数据病例）
```

三条结论：

1. **1.0 的有效分母不是 15 例，也不是 76 个牙位，而是 13 个游离天然牙单冠单位**（上界；
   `bridge_rule` 只判"与桥体直接相邻"，仅间接相邻者可能仍被计入）。
2. **合格单位 100% 卡在"没有可分段的支撑资产"**：15 例的设计元数据里有 Margin（含逐牙位点数）
   与 Center/Axis，但目录里**没有按牙位分好的预备体网格**。这是**流程缺口，不是数据缺口** ——
   这些病例的 `.DCM`/模型数据里本来就有代型，缺的是那个提取器（此前的 `基台测试集_v1` 产物就是
   这么来的，而该目录现已被清空）。
3. **唯二有 `prep.stl` 的病例包都被契约正确拒绝**，拒绝码是 `MARGIN_CROSSES_OPEN_BOUNDARY`
   （ROI/裁片边界跨过边缘线）—— B 类，人工重裁/重选组件后可再试，不是 indication 外。

另外：3 例（`82404110766` / `82404140676张寿山` / `薛文爱`）**没有 constructionInfo/modelInfo**，
设计元数据只在二进制 `.dentalCAD` 快照里 ⇒ 无法参与分诊（报表 note 如实写明）。

## 4. 只读接线

`GET /api/designs/{id}/prep-surface-contract`：返回契约报告，**不改设计、不落库、不做修复**。
顺序是 indication → 输入确认 → 资产 → 几何，因此范围外的病例不会浪费在几何上，
未确认 Margin/轴时会直接说 `MARGIN_OR_AXIS_UNCONFIRMED`。

## 5. 测试

| 文件 | 覆盖 |
|---|---|
| `tests/unit/services/test_prep_conditioning.py`（10 条） | 两条正交轴；干净单一就位面 → `PREP_SURFACE_READY`；`MARGIN_CROSSES_OPEN_BOUNDARY`（真实病例的失败形态，用"无盖管壁"复现）；多组件 → B 类 + island 污染嫌疑；失败消息映射是**全映射**（未映射的必须显式标 `UNMAPPED`，不静默当 READY）；逐牙位分类；桥体相邻规则（含跨象限不算相邻）；设计行评估的三段顺序；报表字段齐全可序列化；**契约不得改输入网格** |
| `tests/integration/test_prep_conditioning_corpus.py`（3 条） | 真实语料报表跑完且自洽（indication 之和 = 单位总数、eligible 与 indication 一致、readiness 落在允许档内、`contract_attempts` 与包数一致）；契约尝试过的单位**不许静默通过**（拒绝必带理由）；端点只读性（对比调用前后 `design_revision`/`margin_json`/产物路径） |

## 6. 本批运行

本机全量 `pytest tests`：**608 passed / 10 skipped / 2 xfailed / 0 failed**
（P1 前 595 passed；新增 13 条：契约/分类单测 10 + 真实语料报表 3）。
本批零几何改动、零算法改动、零 schema 改动（唯一新增的 API 是只读自检端点）。

## 7. 更正（2026-10-13，C2 审计时发现）

两处必须更正，都是"数字看起来对、其实不成立"的类型：

### 7.1 契约缺一道"同坐标系健全性门"（已修）

原契约只做拓扑判据（`fitting_support_above_margin`）。实测发现：当 margin 与支撑网格**不在同一坐标系**
（30–45 mm 量级）时，"margin 不落在开放边界上"会**平凡成立**，于是产出**假 `PREP_SURFACE_READY`**。
修复：先过同坐标系门 —— `margin → 支撑表面最大距离 ≤ 1.0 mm`（`MARGIN_ON_SURFACE_MAX_MM`），
否则 `REVIEW_REQUIRED / MARGIN_NOT_ON_SUPPORT_SURFACE`，理由里点明"两者很可能不在同一坐标系"。
依据：真实病例里 margin 就贴在被测表面上（1629 实测 0.24 mm 量级）。

### 7.2 "合格单位 100% 无支撑资产"这句要收窄

P1 报告当时只查了病例包里的 `prep.stl/die.stl`。C2 审计把目录里**所有 STL** 过了一遍：

- 真实数据里确实有 **3 个逐牙位代型**（`82404110733` 的 FDI 24/25/27）+ 2 个模型底座；
- 但它们**跨坐标系**（代型在模型/打印系、margin 在设计系，31–46 mm）⇒ 修复后的契约给
  `REVIEW_REQUIRED / MARGIN_NOT_ON_SUPPORT_SURFACE`，**不是 READY**；
- 另有 **63 个未归类 STL**（命名不含 modeldie/modelbase/cad 且不在 `construction_files` 里），
  归属未知；`ConstructionFileList` 只在 2/15 例存在 ⇒ 其余病例的 CAD 文件无法用元数据映射牙位。

所以正确说法是：**合格单位的支撑资产不是"不存在"，而是"存在但坐标系不通 / 归属未定"**，
缺的是 **asset production（含 design→model 变换）**，而不是几何分类器。详见
`docs/C2_BITE_REFERENCE_AUDIT_2026-10-13.md` §4。

## 8. 未做（明确边界）

1. **不做任何几何修复**（用户明确要求）：没有自动封洞、缝螺丝孔、智能删组件、重建 prep。
   下一批要不要做，取决于本报表给出的拒绝分布 —— 现在只有 B 类一种真实拒绝码，
   样本量 2，**不足以决定**先做哪种修复。
2. **没有产生支撑资产的流程**：把 15 例的 `.DCM`/模型变成按牙位分段的 prep 网格，
   需要那个提取器（曾是 `基台测试集_v1` 的来源）。它是 Prep Conditioning 真正的工作量所在，
   但已经超出"编排现有积木"的范围，应当单独一批。
3. **桥体相邻规则是上界**：只判直接相邻（同象限 ±1 牙位），间接相邻者可能仍计入 eligible。
4. 3 例无元数据病例需要读 `.dentalCAD` 二进制才能分诊 —— 已知该容器非明文浮点，
   本批不碰。
