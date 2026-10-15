# Q0：恢复真绿基线（acceptance 三项失败，2026-10-12）

目标：**`pytest tests` 全绿，且没有任何"known failure"** —— 从这一轮起，任何失败都应当意味着
"新改动破坏了东西"。不 xfail、不放宽阈值。

## 1. 三项失败的真因（查证结论）

三项失败**不是回归**，也不是守卫太严，而是**验收套件在测一条已退役的链路**：

| 事实 | 证据 |
|---|---|
| 失败来自 `app/geometry/outer.py` 的守卫 | `trim_outer_at_margin` 在 `Margin` 非数值平面时抛 `MarginReconstructionRequired` |
| 本病例真值 Margin 确实非平面 | `plane_deviation_max_mm = 0.043315`（p95 0.0408），`numerically_planar = False` |
| 三项测试用的链路是**退役链路** | `_metrics.measure_margin_trim` / `measure_isolated_cad_chain` = 平面裁切 + `enforce_clearance` + 体素布尔；`enforce_clearance` 只被这个验收度量调用（`grep` 全仓：`app/` 内零调用） |
| sanctioned 链路在现有真实数据上**被正确拒绝** | ① FDI36（种植基台 `1629-王煜 0.stl`）→ "Margin 以上基牙不是单一完整就位面"（螺丝通道开口）；② FDI46 病例包 `cf_b1b73bae221211b1` 的 `prep.stl` → "Margin 跨扫描开放边界或缺口"；③ 同包 `die.stl` → 与 ① 同因。另试 4 组 ROI 支撑（下颌扫描裁剪）同样被拒（邻牙/牙龈导致多组件） |
| 退役链路对真实的 `enforce_clearance` 缺陷无代表性 | 该函数把"顶点到 cutter 的距离"当壁厚，实测外推 5788/约 25000 个顶点、**最大 1.64 mm**（与 `known_defects` 记录的 37%/1.63 mm 一致） |

⇒ 原来的"绿"只能由**被禁止的平面裁切链路**提供。正确做法是：
**守卫该钉住、链路不变量改在链路真能接受的输入上测、并把这次口径变更按项目自己的
retraction 格式记录在案。**

## 2. 本批实现

### 2.1 新增合成可执行输入 `tests/acceptance/synthetic_prep.py`

闭合圆柱预备体（单一完整就位面、无开放边界）+ 双波非平面 Margin（振幅 0.6 mm）
+ "真值外冠"（下缘贴 Margin、高于内腔 —— 否则内腔会从顶面开第二道边界，
布尔主体无法分成两个单颈环面）。

明确写进 docstring：**它不冒充病例精度**，只提供链路不变量的可执行输入。

### 2.2 `_metrics`：删掉退役链路的两个度量，换成 sanctioned 链路

| 旧 | 新 |
|---|---|
| `measure_margin_trim(gt_crown, margin, axis)`（平面裁切） | `measure_plane_clip(mesh, margin, axis)`：**只在数值平面 Margin 上**合法；另加 `margin_plane_status()` |
| `measure_isolated_cad_chain(...)`（裁切 + `enforce_clearance` + 体素布尔） | `measure_live_chain(outer, support, margin, axis, params)`：`fitting_support_above_margin → build_blockout → apply_gap_field → build_nonplanar_crown`（与 API `finalize_crown` 同一实现族） |

`measure_live_chain` 的输出（都在**导出重读**之后计算）：水密 / 开放边界 / 非流形边 / 组件数 /
共享颈缘缝双向误差与共享边校验 / `body_to_original_outer_p95_mm`（外冠形态改写量）/
体积三元组（final、outer、内腔）与 `mass_balance_relative_error`。

### 2.3 测试改写（断言没有放松，覆盖面有增）

| 旧测试 | 新测试 | 断言 |
|---|---|---|
| `test_margin_trim_is_watertight`（真值 Margin 上裁切） | `test_plane_clip_is_watertight_on_planar_margin` | **平面** Margin 上裁切必须水密、0 开放边界（原始 P0：`cap=False` 留下开放边界 → 13% 碎冠） |
| — | `test_nonplanar_gt_margin_is_refused_not_clipped` | 真值 Margin 非平面（偏离 0.0433 mm）且裁切必须 `MARGIN_RECONSTRUCTION_REQUIRED` + 三条缺件适配器如实列出（**钉住守卫**，此前无任何测试覆盖） |
| `test_final_mesh_is_usable` | `test_live_chain_final_mesh_is_usable_and_balanced` | sanctioned 链路：水密、组件数 1、0 开放边界、0 非流形边、共享缝成立、缝误差 ≤ 1e-6 mm、质量守恒 ≤ 6% |
| `test_isolated_chain_volume_ratio` | `test_live_chain_mass_balance` | `final ≈ outer − 内腔`（相对误差目标，与几何形状无关） |
| `test_isolated_chain_preserves_outer_anatomy`（**xfail**） | `test_live_chain_preserves_outer_anatomy` | p95 ≤ 0.05 mm，**实测 1.13e-7 mm 通过** |
| — | `test_live_chain_refuses_abutment_support_with_reason` | 真实基台病例必须被拒且理由是"单一完整就位面"（钉住输入契约，防止以后被当 bug"修好"） |

### 2.4 baseline.json 按项目自身格式更新

- `measurements`：退役链路的 FDI36 数值（`margin_trim_*` / `isolated_chain*` / `final_*`）移除，
  新增 `live_chain`（完整报告）+ `live_chain_watertight` / `live_chain_components` /
  `live_chain_outer_deviation_p95_mm` / `live_chain_mass_balance_relative_error` /
  `margin_plane_gt_max_deviation_mm`；
- `targets`：新增 `live_chain_components=1`、`live_chain_boundary_edges=0`、
  `live_chain_seam_max_mm=1e-6`、`live_chain_mass_balance_relative_error=0.06`；
  删除 `isolated_chain_volume_ratio`（区间隐含"外冠即薄壳冠"的输入假设，换输入即失效）；
  `isolated_chain_outer_deviation_p95_mm` **保留同一个目标值 0.05**，只更新 `why` 与定义版本；
- `known_defects`：**删除** `isolated_chain_outer_deviation_p95_mm`（缺陷载体是退役链路的
  `enforce_clearance`，它已不被任何链路调用）；conftest 的 xfail 映射同步删除；
- `retracted_expectations`：新增三条（隔离链路偏差指标、体积比指标、FDI36 平面裁切链路），
  逐条说明"为什么撤、替代断言在哪、不是放宽阈值"；
- `metric_versions`：`live_chain` / `isolated_chain_outer_deviation_p95_mm` /
  `margin_trim_watertight` 升到 `v4_live_nonplanar_chain`（定义确实变了 ⇒ 必须升版本），
  clearance 两项保持 `v2_gt_cervical_seam`（定义未变 ⇒ 不许升，守卫测试仍在）；
- `pipeline_version = v4_nonplanar_shared_seam`；`pipeline_components` 新增
  `cervical = v4_nonplanar_shared_seam_stitch`、`thickness = audit_only`（不做自动补厚）。

## 3. 结果

```
tests/acceptance        118 passed, 2 xfailed      （原 112 passed / 3 failed / 3 xfailed）
```

剩 2 个 xfail 是**真实已知缺陷**（投影柱 blockout 撑大边缘带内腔），有 owner、有文字理由、
有明确替代方向 —— 符合"可以 xfail"的唯一条件。

## 4. 本批暴露、需要你决策的两件事

1. **真实病例的 sanctioned 链路验收仍是空的**。现有三个真实输入（FDI36 基台、FDI46 包
   prep/die）都被输入契约正确拒绝，而契约要的是"**分段好的单一预备体面**"
   （无开放边界、Margin 以上无第二个开口）。要补真实病例验收，需要其中之一：
   - 提供一个满足契约的真实病例（扫描 + 分段预备体 + GT 冠）；或
   - 实现"螺丝通道/扫描杆开口的封闭"适配器（基台病例当前走不通）；或
   - 用 ROI 裁剪 + 单一连通分量提取把下颌扫描变成合法支撑（邻牙/牙龈要排除干净）。
   本批**没有**替这条做任何近似替代 —— 只把"契约与拒绝理由"钉成了测试。
2. **`outer.enforce_clearance` 已成孤儿**（只被已删除的验收度量调用）。它留/删由你定：
   本批不动 `outer.py`（该文件同时还有他人未提交的守卫改动）。
