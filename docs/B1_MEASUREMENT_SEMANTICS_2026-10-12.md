# B1：几何测量语义分叉清理（2026-10-12）

范围：把"决策路径"上剩下的 **occlusion.py / thickness.py** 两套符号与有效性约定显式化、
共用 primitive，**但保留各自的领域对象**（不合并成一个 DTO）。本批**不改变任何数值行为**
（特征测试逐条钉住旧行为），也不碰任何阈值。

## 1. 现状：三个域本来就是三套口径（查证结论）

| 维度 | Contact | Occlusion | Thickness |
|---|---|---|---|
| 实现 | `contact.clearance_detail` + `auto_fit_adapter.FitSurface` | `contact.closest_with_validity`（raw） | 自带 `SurfaceQuery` |
| 符号来源 | 闭实体实体内外 / 人工外向法向 | **一次性质心定向**（全局投票） | **平滑法向**（最近面三顶点面积加权法向均值） |
| validity | 边界 + 折痕（`AMBIGUOUS_CREASE`） | 仅边界安全 | 仅边界安全 |
| 额外如实计数 | `ambiguous_count` | `ambiguous` | **`flipped`**（单面法向与平滑法向反向） |
| 穿透量语义 | `max(0, −signed)`：真穿透深度 | `max(target_gap − signed, 0)`：**目标相对亏欠** | `signed − target`：相对约束的余量 |
| 相交审计 | 服务层 `surface_crossings` | **不负责** | **本域负责**（`self_intersections` / `input_qc`） |
| 输入契约 | 任意网格 | 冠网格必须有可识别**颈环** | 外表面必须有开放**颈缘** |

结论：**"顺手统一成一个 DTO" 会把三件不同的事写成一件**。B1 因此采用"共用 primitive +
各自领域对象 + 显式标注"。

## 2. 本批实现

### 2.1 新模块 `app/geometry/measurement_core.py`

- 口径词汇表：`SIGN_GLOBAL_INSIDE_OUTSIDE` / `SIGN_REVIEWED_LOCAL_NORMAL` /
  `SIGN_ONE_SHOT_CENTROID_VOTE` / `SIGN_SMOOTHED_VERTEX_NORMAL` / `SIGN_UNSIGNED_ONLY`；
  `VALIDITY_BOUNDARY_AND_CREASE` / `VALIDITY_BOUNDARY_ONLY` / `VALIDITY_NONE`。
- 共用 primitive：`stats()`（p50/p95/max，空输入 → `None`，**不假装有数**）、`validity_counts()`。
- **领域契约表** `DOMAIN_CONTRACTS`（`DomainContract`）：域名 / 符号来源 / 语义 / 有效性 /
  穿透含义 / **是否负责相交审计** / 输入契约 / 实现位置 + `domain_report()` 标注。

### 2.2 接线（各自 DTO 保留）

- `occlusion.OcclusionResult` 新增 `domain_report`，并在 `notes` 里写明
  "目标相对亏欠，不是 Contact 的穿透深度；相交审计不属本域"；内部那处局部 lambda 统计
  改为共用 `stats()`。
- `thickness.ThicknessResult` 新增 `domain_report`，`notes` 写明"符号用平滑法向、
  有效性仅边界安全、相交审计属本域"。
- `services/contact_measurement` 的语义串与契约表**同一字符串**（由测试强制一致）。

## 3. 特征测试（先钉旧行为，再动结构）

| 文件 | 条数 | 钉住的场景（用户列的 6 类） |
|---|---|---|
| `tests/unit/geometry/test_occlusion_characterization.py` | 8 | ①闭合对颌的符号来源（并在第二条里钉住"闭合对颌可能被全局投票判成无干涉"这一**真实分歧**，附 Contact 实体内外判定的对照）；②开放裁片 validity=边界安全；③无效采样不进统计但计入 `ambiguous`；④缺颈环必须抛错；⑤相交审计**不属本域**（职责边界）；⑥零间隙不算干涉（`signed<0` 严格）＋目标相对亏欠口径 |
| `tests/unit/geometry/test_thickness_characterization.py` | 7 | ①符号用平滑法向（翻一个三角形不整体翻符号 + `flipped` 如实计数 + `normal_sign` 被尊重）；②validity 仅边界安全（闭合网格全有效）；③**相交审计属本域**；④外表面必须有开放颈缘（否则报错）；⑤薄判定与"未验冲突"如实记录；⑥归因用 `signed − target < margin`（近 ⇒ 接触冲突，远 ⇒ 可修）；⑦薄判定不作用于无效采样（用恒等式断言，不写死实现细节条数） |
| `tests/unit/geometry/test_measurement_core_contracts.py` | 7 | 契约表与代码实际行为一致：相交审计归属（thickness 有 `self_intersections`，occlusion 没有）、Contact 语义串与服务模块逐字一致、三种有效性约定不被悄悄改成同一种、两个域的结果都带 `domain_report` |

**数值零变化**：本批没有改任何几何计算分支；`audit_thickness` / `solve_occlusion_collision`
的判定路径未动（只加了报告字段与注释），既有 `test_occlusion_v1` / `test_contact_*` 全绿。

**本批运行**（本机，全量 `pytest tests`）：**576 passed / 10 skipped / 2 xfailed / 0 failed**
（Q0 前为 548 passed / 3 failed / 3 xfailed）。剩 2 个 xfail 都是投影柱 blockout 撑大边缘带内腔
这一真实缺陷，有 owner、有文字理由、有替代方向。

## 4. 本批发现的、需要决策的事（未擅自改）

1. **Occlusion 的符号对闭合对颌不可靠**：全局质心投票在"对颌把冠包住"时可能选出相反符号，
   于是报告"无干涉"；而 Contact 的实体内外判定会报真穿透。已钉成测试
   （`test_closed_antagonist_can_be_voted_to_no_interference`）。
   要改的话是把 occlusion 的符号切到实体内外判定 —— 会影响 K/J.2 门，需单独决策。
2. **thickness 的 validity 只到边界安全**，`flipped` 只计数、不参与判定。
   是否让 `flipped` 进判定集（例如强制该点判无效）也是行为变更，未擅自做。
3. `occlusion.py` 里还留着 `math` / `cKDTree` / `I2Params` 三个未使用 import（他人代码遗留），
   本批未清，避免与别人的未提交改动冲突。
