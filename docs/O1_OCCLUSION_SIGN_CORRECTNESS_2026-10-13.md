# O1：Occlusion 符号来源正确性（2026-10-13）

范围：**只修"符号来源是否可信"**。不动 occlusion 的目标间隙、Gate 参数、统计口径
（`deficit = max(target_gap − signed, 0)` 原样保留）。

## 1. 改之前查到的事实（B1 的遗留）

`occlusion.py` 用**一次性质心投票**给符号：`normal_sign = sign(Σ (冠质心 − 面质心)·n)`。
两个后果：

1. **闭合对颌可能反号**：对颌把冠包住时，侧面成对抵消、上下两面占优 ⇒ 投票选出
   相反方向 ⇒ 报告"无干涉"，而 Contact 的实体内外判定看出真穿透（B1 已钉成测试）；
2. **开放裁片没有已复核朝向也照样给结论**：猜出来的符号被当成穿透结论使用。

## 2. 冻结的 8 条验收标准（逐条落地）

| # | 标准 | 落点 |
|---|---|---|
| 1 | closed mesh 不再使用 centroid voting | `occlusion` 里的投票代码**已删除** |
| 2 | closed mesh 与 Contact 共享同一 solid sign truth | 新增 `measurement_core.SignSource`；`FitSurface.query` 委托给它，`occlusion` 也用它 —— **实体内外判定只有一份实现** |
| 3 | reviewed open patch 保留 local signed | `SignSource.for_mesh(mesh, outward)` → `REVIEWED_NORMAL_LOCAL`；`occlusion(..., antagonist_outward=)` 或直接传 `FitSurface` |
| 4 | 无 normal 的 open patch 永远不能产生 penetration 结论 | 无声明 ⇒ `UNSIGNED_ONLY`：`signed` 全 `nan`、`penetration_mm = None`、`status = NOT_ASSESSED`、**不动几何** |
| 5 | `deficit` 保持 Occlusion 自己的 domain semantics | `deficit = max(target_gap − signed, 0)`（历史 `p50/p95/max/sum` 键名与含义未动） |
| 6 | surface crossing 与 signed-distance 明确分责 | 契约表 + 两条测试（见下）；本模块仍**不**做相交审计 |
| 7 | 6 场景 characterization 后再改实现 | 5 场景 B1 已有；本批先补第 6 场景（crossing 且无负顶点）与"未复核裁片当前会猜"两条，跑通后才改代码 |
| 8 | 改完再跑 K/J.2 相关 tests，确认只是 sign correctness | 见 §5 证据 |

## 3. 实现

### 3.1 `measurement_core.SignSource`（唯一符号 primitive）

```
SOLID_CONTAINMENT        闭合可定向实体：-d（内部）/ +d（外部）   ← 与 Contact 共享
REVIEWED_NORMAL_LOCAL    开放裁片 + 调用方声明的外向法向（局部可信）← 与 Contact 共享
UNSIGNED_ONLY            两者都没有：signed 全 nan（**不是 0**，0 会被读成"贴面"）
```

- `query(points) -> (closest, face_ids, signed, distance, valid)`，与 `clearance_detail` /
  `FitSurface.query` 同形状；
- **ε 歧义带**：`d < epsilon_mm`（默认 1e-6 mm）判为歧义（`valid=False`），
  避免接触点出现 ±1e-12 级别的随机翻号；
- 词汇表收敛：`SOLID_CONTAINMENT` / `REVIEWED_NORMAL_LOCAL` / `UNSIGNED_ONLY`
  （`GLOBAL_INSIDE_OUTSIDE` / `REVIEWED_LOCAL_NORMAL` 保留为**别名**，不再新增名字）；
  `ONE_SHOT_CENTROID_VOTE` 已从契约表移除。

### 3.2 报告结构：把两个概念拆开

每个测量条目新增：

```text
penetration_mm          {"max_mm","p95_mm","count"} 或 None   ← 真穿透深度
clearance_shortfall_mm  {"max_mm","p95_mm","count"}           ← max(target − signed, 0)
sign_method / signed_available / signed_min_mm
```

`UNSIGNED_ONLY` 下：`penetration_mm = None`、`signed_min_mm = None`、
`clearance_shortfall_mm` 由 **unsigned 幅值**算（`max(target − unsigned, 0)`），
状态 `NOT_ASSESSED`，`modified_vertex_count == 0`。

### 3.3 调用方必须**声明**方向（而不是让求解器猜）

- `crown_outer_pipeline.stage_occlusion`：新增 `_declare_outward()`，把原先的质心投票
  变成**显式声明**并写进 stage 报告：
  `outward_declaration = "ALGORITHMIC_CENTROID_ASSUMPTION_NOT_HUMAN_REVIEWED"` ——
  算法假设可以，但要标出来（人工确认的邻牙面走 `confirm_neighbor` 存的 `outward_normal`）。
- 测试夹具同理由显式声明（`test_occlusion_v1` 的对颌板与两块邻牙面）。

## 4. 验收矩阵（`tests/unit/geometry/test_occlusion_sign_matrix.py`，7 条全绿）

| 场景 | 断言 |
|---|---|
| watertight 不相交 | `SOLID_CONTAINMENT`；`signed_min > 0`；`penetration` 0；`shortfall` 0；不动 |
| watertight 轻微穿透（0.1） | `signed_min < 0`；`penetration.max ≈ 0.1`；`shortfall ≈ 0.15` |
| 冠顶点在对颌实体内 | `signed_min < 0`；`penetration.max ≈ 0.2`；**`count > 0`（不再"无干涉"）**；触发修正 |
| 开放裁片 + 已声明法向 | `REVIEWED_NORMAL_LOCAL`；`penetration.max ≈ 0.4`；与 `FitSurface` 结果一致到 1e-9 |
| 开放裁片无声明法向 | `UNSIGNED_ONLY`；`signed`/`penetration` 为 None；`shortfall` 用幅值；`NOT_ASSESSED`；不动几何 |
| ε 边界 | 贴面点判歧义（`valid=False`），不许随机翻号 |
| 职责分离 | `signed_min ≥ 0` **不能推出** `crossings == 0`（blade 穿板反例） |

## 5. 证据（item 8：这是 sign correctness，不是偷偷改 Gate）

| 检查 | 结果 |
|---|---|
| K/J.2 相关门测试（occlusion_v1 / contact_engine / contact_v2_local / contact_refine / contact_i2 / sign_matrix / nonplanar_crown_build） | **45 passed** |
| J1 / CrownOuter 管线门 | **20 passed** |
| B 阶段 parity（Contact 四处 `signed_min_mm` 完全一致） | **4 passed** —— Contact 数值未被 O1 改动 |
| 全量 `pytest tests` | 见 §6 |

唯一的行为变化是**符号不可信时报 NOT_ASSESSED**（旧行为是猜一个符号继续算）：
这在 `test_occlusion_v1` 里表现为"夹具必须声明方向"，已按真实形态（已复核邻牙面）声明。

## 6. 本批运行

本机全量 `pytest tests`：**595 passed / 10 skipped / 2 xfailed / 0 failed**
（O1 前 585 passed / 2 xfailed；新增 10 条：sign matrix 7 + 特征补 2 + 契约 1）。
`test_occlusion_v1` 的两条测试改的是**夹具声明方式**（补上外向法向），
断言本身（穿透 p95 单调下降、锁定区不动、翻面受控、Contact 保持性有账）一条没放松。

## 7. 未做 / 边界

1. **`crown_outer_pipeline` 的对颌朝向仍是算法假设**（只是现在标出来了）。要让它是
   人工确认的，需要走 app 的 `confirm_neighbor` 路径或引入"对颌朝向复核"步骤 ——
   属于 C2 Bite Registration 的前置。
2. **`thickness` 未动**（用户明确）：`flipped` 仍是 diagnostic，先统计
   `P(flipped)`、`P(flipped|thin)`、`P(flipped|margin)`、`P(flipped|高曲率)`、
   `P(flipped|invalid ray)`；没有 false-pass 证据就不进 Gate。
3. **`occlusion` 仍不做相交审计**（职责分离），相交结论仍由
   `thickness.self_intersections` / `services.surface_crossings` 给。
4. `occlusion.py` 里还有 `math` / `cKDTree` / `I2Params` 三个未使用 import（他人遗留），
   本批未清。
