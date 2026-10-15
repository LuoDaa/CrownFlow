# 参数驱动生成：Profile + Constraint Solvers + Optimizer（2026-10-13）

本批把"选择模板 → 放置 → 人工调整 → 生成内冠"升级为
"选择模板 → 用户填参数 → 系统自动适配 → 可审核初稿"。
**不是 AI Crown**，也没有重写 Template / Margin / PatientFrame / ContactMeasurement / Intaglio / Boolean：
本批是**编排与契约**，新几何实现只有一处（接触面积的工程口径），其余全部复用既有能力。

---

## 一、审计结果（实现前，先读代码）

### 1. 当前已经实现并且真的影响几何的参数

| 参数（扁平名） | 定义位置 | 消费点（几何） |
| --- | --- | --- |
| `cement_gap_mm` | `models.py:17` | `pipeline.py:78` → `intaglio.apply_gap_field` |
| `cement_gap_start_mm` / `cement_gap_top_mm` | `models.py:36,39` | `intaglio.py:810-821` |
| `margin_band_mm` | `models.py:19` | `nonplanar_crown.py:129-134`（过渡带） |
| `minimum_thickness_mm` | `models.py:20` | `nonplanar_crown.py:152`、`manufacturing_qc.py:52` |
| `enforce_minimum_thickness` | `models.py:23` | `nonplanar_crown.py:148`、`minimum_thickness.enforce_body_thickness` |
| `border_horizontal_mm` / `border_angled_mm` / `border_angle_deg` / `border_vertical_mm` | `models.py:24-27` | `crown_border.build_crown_border` |
| `proximal_target_mm` | `models.py:28` | `auto_fit.propose → fit_stage(PROXIMAL)` |
| `occlusal_target_mm` | `models.py:29` | `auto_fit.propose → fit_stage(OCCLUSAL)` |
| `machine_axes` / `untouchable_zone_mm` | `models.py:32,34` | `pipeline.py:58-60` → `BlockoutPolicy` |

### 2. 只"测量"但没有"自动修正"的

- **咬合**：`manufacturing_qc.audit_manufacturing` 的 `OcclusionQC` 恒为 `NOT_ASSESSED`
  （`trusted_bite_transform_and_signed_audit_not_connected`），即"算得出间隙但没有把它接到修正链上"。
  live 修正链在 `occlusion.solve_occlusion_collision`（有，可用），只是 QC 报告没接。
- **接触面积**：只有零散的 ad-hoc 面积（`contact.py:864`、`contact_refine.py:422`、
  `proximal_constrained.py:237`），`contact_measurement` **完全不出面积字段**，
  且项目自己声明 `NOT_AVAILABLE_NO_VALIDATED_AREA_DEFINITION`。
- **壁厚**：`audit_thickness` / `measure_wall_thickness` 是测量；修正在
  `minimum_thickness.enforce_body_thickness`（有），但结果键名与本节要求不同，
  且只有 `enforce AND border_horizontal_mm` 同时给出才会跑（见 §4 的耦合）。
- **倒凹**：`manufacturing/reachability.py` 只算可见性比例，
  `status=NOT_IMPLEMENTED, is_authoritative=False`。

### 3. 完全没有实现的

`contact_area_mode` / `contact_area_target_mm2`、`mesial_enabled` / `distal_enabled`、
`occlusion.enabled` / `require_confirmed_bite_pose`（作为字段）、
`undercut.policy(REMOVE/KEEP)` / `blockout_angle_deg`、
`border.smooth_transition` / `blend_radius_mm` / `blend_width_mm`、
`manufacturing.tool_radius_mm` / `tool_compensation_enabled`、
`outer.md_scale` / `bl_scale` / `height_scale`（`uniform_scale` 只作为输出）。
工具补偿尤其彻底：全仓无 Minkowski / SDF-offset / tool-radius 补偿，
`intaglio.py` 里的 `binary_dilation` 只是二维高度场掩膜膨胀。

### 4. 顺带查到的"静默把 MISSING 当 0"（规格明令禁止）

- `services/auto_fit.py:139` `parameters.get('proximal_target_mm') or 0`
- `services/auto_fit.py:175,178` 同上（咬合保护带按 0 圈"接触区"）
- `services/pipeline.py:58,60` `int(params.get("machine_axes") or 0)`、`float(... or 0.0)`
- `geometry/manufacturing_qc.py:57,89` `parameters.get(...,0)`、`... or 0`

本批**只修了 auto_fit 的两处**（它们是本批自动适配链自己的病灶，改动面可控）；
`pipeline.py` / `manufacturing_qc.py` 的同类写法**留给这些文件的下一位改动者**，
因为强行改默认会改变既有调用方行为（记在"未做"里，不静默改）。

### 5. 最小修改文件清单（实际改动）

| 文件 | 状态 | 内容 |
| --- | --- | --- |
| `app/services/crown_generation_profile.py` | 新增 | `CrownGenerationProfile` + provenance + 别名 + 禁 MISSING→0 |
| `app/services/crown_constraint_solvers.py` | 新增 | `solve_proximal` / `solve_occlusion` / `enforce_minimum_thickness` + 接触面积口径 |
| `app/services/crown_constraint_optimizer.py` | 新增 | `optimize_crown`：staged 循环，MAX_ITER=3，禁止部分失败标 PASS |
| `app/models.py` | 修改 | 新增 profile 字段（参数才能真正存进 `parameters_json`） |
| `app/api/designer.py` | 修改 | 只读 `GET /designs/{id}/generation-profile` |
| `app/services/auto_fit.py` | 修改 | 去掉两处 `or 0`，近远中复测统一走 `_joint()`（MISSING ⇒ NOT_ASSESSED） |
| `tests/integration/test_parameter_sensitivity.py` | 新增 | 8 条参数敏感性（§13/§14 验收） |

---

## 二、本批实现（P0）

### 1. `CrownGenerationProfile`

一次性统一 8 个分组（proximal / occlusion / intaglio / thickness / undercut / border /
manufacturing / outer），每个字段带 `source ∈ {USER, DERIVED, DEFAULT_POLICY, MISSING}`，
扁平存量名与 profile 名之间用别名表对应（旧 `parameters_json` 与旧前端字段继续可用）。

**MISSING 绝不等于 0**：`get()` 返回 `None`，`missing()` 列出全部缺项，
需要它的阶段用 `require(..., stage=...)` fail-loud 抛 `ProfileParameterMissing`。
显式 `0.0` 与"没填"在 provenance 上可区分（`USER` vs `MISSING`），有测试钉住。

策略默认（`occlusion.require_confirmed_bite_pose=True`、`undercut.policy=REMOVE` 等）标注为
`DEFAULT_POLICY`，与用户显式给值分开——不被混成"用户填的"。

### 2. `ProximalContactSolver`（P0 第 1 项）

复用 `auto_fit_adapter.fit_stage`（区域选择 + 局部带 Laplacian + α 回溯 + 形变门）与
`contact_measurement`（唯一有符号口径）。本批只加三件事：

- **目标只能来自 profile**（MISSING 即 fail-loud）；`mesial_enabled`/`distal_enabled` 可按侧关闭，
  关闭的侧记 `SKIPPED_DISABLED_BY_PROFILE` 而不是悄悄不跑；
- **接触面积作为二次目标**：面积偏大/偏小 ⇒ 在 ±0.04 mm 内小幅挪动目标距离，
  每轮仍走同一个几何门，只有净收益才接受，未达成如实报 `CONTACT_AREA_TARGET_NOT_REACHED`；
- **键名归一**：`target_distance_mm` / `before_distance` / `after_distance` /
  `before_penetration_mm` / `after_penetration_mm` / `contact_area` / `iterations` / `convergence`。

接触面积口径显式声明为工程口径并**未认证**：
`sum_of_vertex_areas_where_signed_distance_le_threshold`（version `vertex_area_signed_threshold/1`，
`validated=False`，band 0.05 mm）。mode→mm² 是**配置表**（SMALL 2.0 / MEDIUM 6.0 / LARGE 12.0 mm²），
不是 UI 常量；将来 Material/LabProfile 可覆盖。

**踩到并修掉的坑**：`fit_stage` 成功时返回 `PASS_ENGINEERING` 而不是 `PASS`。
不归一化就直接比较，**面积二次目标永远不会执行**（参数会变成装饰品）。现在统一走
`normalize_status()`，原生值保留在 `native_status`。

### 3. `StaticOcclusalSolver`（P0 第 3 项）

硬规则先行：`require_confirmed_bite_pose`（默认 true）且位姿未确认 ⇒
`OCCLUSION_NOT_ASSESSED` + `reason=BITE_POSE_NOT_CONFIRMED`，**几何一个点都不动**，
也不构造替代位姿。有可信位姿时复用 `fit_stage(OCCLUSAL)`，保护带来自已确认的近远中接触区，
每轮结束后**重新量近远中**；一旦近远中变差就回滚并记 `JOINT_CONTACT_PRESERVATION_FAILED`。
输出 `target_clearance_mm / before_min / after_min / penetration_before/after /
adjusted_area_mm2 / iterations / convergence`。**本批不做动态咬合。**

### 4. `MinimumThicknessEnforcer`（P0 第 4 项）

编排既有 `enforce_body_thickness`（向外补厚 + 保护面锁 + α 回溯(1,.5,.25) + `deformation_gate`
+ 保护面穿越检查），把结果键名归一为规格要求的
`thin_area_*` / `minimum_before_mm` / `minimum_after_mm` / `max_displacement_mm` /
`p95_displacement_mm` / `accepted_steps` / `rejected_steps`，
并钉住"PASS ⟺ 无残余薄区"（未达成一律 `REVIEW_REQUIRED`）。

### 5. `CrownConstraintOptimizer`（P0 第 7 项）

分阶段迭代（`MAX_ITER=3`），阶段顺序固定
`PROXIMAL → OCCLUSAL → THICKNESS → RECHECK`，每轮都**独立复测**（不看 solver 自称）。
`_verdict()` 要求"所有执行过的阶段都是 PASS"才给整体 PASS；
`NOT_ASSESSED` 不等于通过（有未评估阶段时整体最多 `REVIEW_REQUIRED`）；
出现冲突不再继续迭代（不用迭代掩盖冲突）。**Intaglio 不在本编排内**（规格 §11）。

---

## 三、参数敏感性（规格 §13/§14 验收）

`tests/integration/test_parameter_sensitivity.py`，**8 passed**（实测 7m14s）：

| 测试 | 断言 |
| --- | --- |
| `test_proximal_target_changes_contact` | 0.02 vs 0.25 mm：最终距离单调变化 + 几何逐点不同 |
| `test_contact_area_mode_is_load_bearing` | 目标进入求解与报告；未达成不得标 PASS |
| `test_occlusal_target_changes_clearance` | 0.05 vs 0.30 mm：最终间隙单调变化 + 几何不同 |
| `test_occlusal_requires_confirmed_bite_pose` | 三种"未确认"输入 ⇒ NOT_ASSESSED / 几何不变 |
| `test_min_thickness_changes_outer_geometry` | 0.20 vs 0.60 mm：补厚位移单调 + PASS⟺无残余薄区 |
| `test_cement_gap_changes_actual_gap` | 0.03 vs 0.12 mm 走真实 `finalize_crown`：**cutter 面→预备体面**轴壁段中位间隙变大、腔体体积变大 |
| `test_profile_missing_is_never_zero` | MISSING≠0、显式 0 可区分、策略默认单独标注、未知键登记 |
| `test_optimizer_never_reports_partial_pass` | 未评估阶段存在时整体不得 PASS |

测试本身踩到的三个坑都写进了注释（都是"夹具/口径"问题，不是产品问题）：
① 裸圆柱没有侧壁顶点 ⇒ 近远中域恒空，必须细分；
② `finalize_crown` 收路径不收 Trimesh 对象；
③ cutter 是**腔体实体**，间隙越大腔体越大（第一版按反方向断言，被体积数字钉出来）。

---

## 四、本批明确**未做**（不要当成已实现）

1. **P1 三件**：`CrownBorderBuilder` 的 `blend_radius_mm` / `blend_width_mm` /
   `smooth_transition`（现有 `build_crown_border` 是直棱截面带，没有圆化混合）、
   `ToolCompensationSolver`（内腔按车针半径补偿，全仓没有实现，需要 morphology/SDF 级新几何）、
   以及把 optimizer 接进 `/workflow/generate` 的 live 路径（本批只提供编排 + 只读计划）。
2. **倒凹 REMOVE/KEEP 开关**：`undercut.policy` 字段与 profile 解析已就位，
   但尚未接进 `BlockoutPolicy`，也没有"KEEP + 就位校验未通过 ⇒ 禁止生产放行"的联动。
3. **前端参数面板与阶段进度**（规格 §12）：按你的要求放在算法之后，本批没动 UI。
4. **`outer.md_scale` / `bl_scale` / `height_scale`**：字段已定义，但放置链仍只消费既有缩放。
5. `pipeline.py` / `manufacturing_qc.py` 里剩余的 `or 0` 写法（见 §一.4）。
6. 真实基台病例 E2E（规格 §15）：本批是合成几何上的参数敏感性；真实例跑批需要
   确认后的 bite/邻牙输入，`occlusion` 会在缺少可信位姿时如实报 `NOT_ASSESSED`。

## 五、纪律保持

TemplateSource、PatientFrame、Confirmed Margin / Axis、PreparedAbutmentSupport、
ContactMeasurement 仍是**唯一真相源**：新模块不重算 world-X 帧、不做无符号接触、
不造临时 Margin、不用不同版本的 support、不用 candidate-dependent 邻牙裁剪
（`fit_domains` 的域来自确认帧与已确认邻牙条目）。
所有新报告都带 `release_status=NOT_PRODUCTION_APPROVED` 语义：工程口径 ≠ 生产放行。
