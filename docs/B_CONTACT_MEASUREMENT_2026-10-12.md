# B：唯一 ContactMeasurement（有符号接触口径统一，2026-10-12）

范围：**只做审计 #5**（"Placement 有 signed penetration，Candidate Review 仍只有 unsigned"）——
把四条路径并到同一个测量实现上，并落一条 parity 回归。未改任何阈值、未改求解器迭代逻辑、
未做 PHI 政策（并行治理任务）。

## 1. 审计指出的断口（本批前的源码事实）

| 路径 | 本批前 | 符号 |
|---|---|---|
| 放置（`initialize_draft_outer` / `workflow/generate`） | **完全没有**接触测量（只产出几何） | — |
| AutoFit（`services/auto_fit.propose` + `fit_stage`） | `crown_outer.auto_fit_adapter.measure` → `contact.clearance_detail` | ✅ 有符号 |
| 候选审核（`api/auto_fit.candidate_analysis`、`api/designer.outer_analysis`） | `trimesh.proximity.closest_point` | ❌ **无符号**，报告写"不证明接触或穿透" |
| 最终 QC（`services/auto_fit.audit_final_context` → `qc['ContactQC']`） | 同 AutoFit（`measure`） | ✅ 有符号 |

⇒ 同一颗冠在"候选审核"页面显示的距离无法判定穿透，而求解器与最终 QC 却在用有符号 clearance 下结论。

## 2. 唯一实现：`services/contact_measurement.py`

口径沿用 `crown_outer/contact.py` 2026-10-09 决策（`clearance_detail` 仍是唯一幅值/有效性来源），
**不新造第二套语义**：

```
signed > 0   远离参考面（安全间隙）
signed < 0   穿进参考面（穿透）
|signed|     = 到参考面的欧氏距离（连续，热力图用它）
penetration  = max(0, -signed)          正值 = 穿透深度
deficit      = max(0, target - signed)   距目标间隙还差多少
```

符号来源只有两种，报告里如实标注（`sign_method`）：

| 值 | 何时 | 可信度 |
|---|---|---|
| `GLOBAL_INSIDE_OUTSIDE` | 参考体闭合、可定向、体积>0 → 用实体内外判定 | 可信 |
| `REVIEWED_LOCAL_NORMAL` | 参考面是开放裁片（邻牙选区） → 人工确认的外向法向 + 歧义面剔除 | 局部可信 |
| `UNSIGNED_ONLY` | 参考面**连外向法向都没有**（旧的球面 ROI 选区） | **拒绝穿透结论**，只给幅值 |

公开 API：

- `measure_surface(surface, vertices, mask, target, *, kind, side, crown, domain)` → `(报告, 查询元组)`；
- `report_from_query(...)`：**唯一**报告形状（求解器消费的字段名一个都没改，只新增字段）；
- `per_vertex_contact(surface, vertices)`：逐顶点 `signed_mm` / `distance_mm` / `valid`（热力图用）；
- `combine_per_vertex({side: ...})`：多参考面按**最不利（最小）有符号值**合成，全无效顶点退回幅值并标 `valid=False`；
- `measure_confirmed_inputs(design, frame_ref, crown_path, *, target_proximal, target_occlusal, strict)`：
  用**已确认输入**（邻牙面/咬合位姿，逐条校验来源与区域字节）测一颗冠；
- `confirmed_neighbor_surfaces` / `confirmed_antagonist_surface`：已确认参考面（`strict=False` 时只标未评估）；
- `unsigned_reference_distance(...)`：符号不可得时的**正式降级**（带机器可读 reason）；
- `serializable_measurement(...)`：剥掉进程内对象（FitSurface / 掩码）供 HTTP/修订链使用。

## 3. 四条路径的接线

| 路径 | 落点 | 说明 |
|---|---|---|
| 放置 | `services/generation_workflow.generate_candidate` → 报告新增 `placement_contact` | `strict=False`：未确认的输入标 `NOT_ASSESSED` + reason，绝不猜符号；快照失败不阻断候选生成 |
| AutoFit | `auto_fit_adapter.measure` 变成**薄封装**转发到 service（求解器返回形状 `(report, query)` 不变）；11 处调用点补上 `kind=`/`side=` 标签 | 数值路径零改动（只加标签） |
| 候选审核 | `api/auto_fit.candidate_analysis`、`api/designer.outer_analysis` 全部改走 service | 返回 `signed_mm`（有符号）+ `distance_mm`（幅值，热力图）+ `stats` + `sign_method` + `note`（"负 = 穿进参考面"，不再说"不证明接触或穿透"） |
| 最终 QC | `audit_final_context` 不变（本来就走 `measure`） | 现在与放置/审核是同一条实现 |

**热力图口径**：`per_vertex_contact` 覆盖**全部冠顶点**（前端 `DistanceModel` 按 index 上色），
统计量取**审核接触域**（`fit_domains`）；两者在报告里分别是 `coloring_scope` 与 `stats.scope`。

## 4. Parity 回归（`tests/integration/test_b_contact_parity.py`）

同一颗冠 + 同一份已确认邻牙，四处数字必须一致（不是"接近"）：

| 断言 | 结果 |
|---|---|
| 基线：AutoFit(`metrics_before`) ↔ 放置(`placement_contact`) ↔ 最终 QC（`audit_final_context` 传同一文件）在 `valid_count`/`ambiguous_count` 上**精确相等**，5 个有符号标量 `abs≤1e-9` | ✅ |
| 候选：候选审核 ↔ 最终 QC（同一份候选文件、同一接触域）逐字段精确相等 | ✅ |
| AutoFit 的 `after` 用的是**基线冻结域**（before/after 可比），审核/QC 用该文件自身域 ⇒ 只比决策量（`signed_min_mm`、`penetration_max_mm`）与实现标识，域大小差异如实暴露在 `domain_vertex_count` | ✅ |
| 四条路径都给出有符号结果、`sign_method` 一致、`sign_semantics` 一致 | ✅ |
| 无符号-only 拒绝：`unsigned_reference_distance` 返回 `signed_mm=None` + reason；开放面缺外向法向时 `FitSurface` 直接 `ValueError` | ✅ |
| 求解器报告形状仍是唯一实现产出（15 个字段 + 查询元组 6 元组） | ✅ |

服务级单测（`tests/unit/services/test_b_contact_measurement.py`，8 条）：穿进邻牙时 `signed_min_mm<0`、
`penetration_max_mm == -signed_min_mm`、幅值 == |有符号|、空掩码 NOT_ASSESSED、
`combine_per_vertex` 的最不利规则与全无效顶点退回幅值、无帧时 `FRAME_UNCONFIRMED`。

实测数字（合成夹具，圆柱冠 + 球状邻牙，目标间隙 0.05mm）：四条路径
`signed_min_mm = 0.47763821624694536`（完全一致），`penetration_max_mm = 0.0`。

**本批运行**（本机，全量 `pytest tests`）：**548 passed / 10 skipped / 3 xfailed / 3 failed**；
3 条失败仍是工作树里**他人未提交**的 `app/geometry/outer.py` 非平面守卫（`_metrics.py:21` 直接调
`trim_outer_at_margin`），与 A1/B 路径无关。前端 `npm run build` 通过。

## 5. 前端

`CandidateReview.jsx` 新增"有符号接触（负 = 穿进参考面）"表：放置后/适配后的**最小间隙**与**最大穿透**，
并标注符号来源与共用实现；模板初稿（无 before/after）用 `placement_contact` 当"放置后"；
放置快照非 MEASURED 时显示 `status`/`reason` 与各侧的未评估原因。
`npm run build` 通过。

## 6. 已知残留（诚实清单）

1. **`occlusion.py` 与 `thickness.py` 仍调用 raw `closest_with_validity`/`closest`**（未经 `clearance_detail`
   的去边界/折痕过滤）——与 2026-10-09 "决策路径只允许 clearance_detail" 的约定不符。本批未动它们
   （属咬合求解与壁厚归因，改口径要先有对应回归）。
2. **`manufacturing_qc.audit_manufacturing` 里的 `ContactQC` 仍是占位**（`NOT_ASSESSED /
   confirmed_instance_and_signed_scope_contract_not_connected`），真正的数字由 `audit_final_context`
   在构造 W3 之后填入。占位保持原样是**有意的**：没有已确认实例时它就该说"没有"。
3. **旧球面 ROI 邻牙选区（`status='SELECTED'`，无 `outward_normal`）只能拿幅值**：
   `sign_method=UNSIGNED_ONLY` + 明确 reason，不再假装能判穿透。要在这些病例上看有符号结论，
   需先走 AutoFit 面板的"确认邻牙实例面区"。
4. **`combine_per_vertex` 的最不利合成**只用于多参考面热力图；判定用的仍是每侧各自的域统计
   （`stats.per_side`），不存在"把两侧混成一个数"的结论。
5. **PHI 三级政策**（默认 `PSEUDONYMIZED`）仍未落地——按你的裁定是并行治理任务。
