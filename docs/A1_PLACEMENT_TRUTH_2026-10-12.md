# A1：统一 placement 入口 + 原子修订（2026-10-12，第二批）

范围：**只做 A1（placement 入口收敛）+ 原子修订（atomic revisions）**。
PHI 三级政策按你的裁定留作并行治理任务，本批**未改任何 PHI 行为**；未接触 B 阶段的 Contact 语义统一。

## 1. 本批关闭的 A0 残留（A0 文档"未做项"2、3）

| A0 残留 | 现状（本批前） | 本批处理 |
|---|---|---|
| 原子修订 | `edit-outer` / `restore-outer` / `/outer`（legacy）/ `/outer/auto` / `finalize` 都是"先 publish、再 append_revision"两步 | 五处全部改为**单事务** `publish_artifacts(..., revision=...)` |
| legacy 端点 | `/outer`、`/outer/auto` 仍可 world-X 回退（`frame_source='WORLD_X_FALLBACK'`） | `/outer/auto`（live）**无确认帧直接 409**；`/outer` 的地标分支降为**必须显式开启**的 research/legacy 路径 |
| truth source 只到报告 | 报告里有，产物上没有 | `template_source` / `patient_frame` / `placement` 同时写进**候选报告**、**修订 payload**、**产物旁 provenance sidecar** |

## 2. 实现

### 2.1 唯一 placement 入口，live 必须给确认帧

`api/routes.py::initialize_draft_outer(d, p, *, allow_world_frame=False)`：

- 模板：一律 `services.template_source.resolve_template_source`（A0 已建），本批无例外；
- 帧：`api.auto_fit.frame_from(d)`（确认的 mesial/buccal/axis）；**拿不到就 409** ——
  live 生成不再用世界轴猜横向方向。只有显式 `allow_world_frame=True`（研究/回归）才回退，
  并标 `frame_source='WORLD_X_FALLBACK_EXPLICIT'`（旧值 `WORLD_X_FALLBACK` 已废弃）；
- 库件仍**只放置一次**（`placed_library_crown` 内含颈缘 footprint 拟合）；
- 返回/产物带 `template_source`、`frame_source`、`placement_path`、`provenance_path`。

`/designs/{id}/outer/auto` 因此要求"已确认坐标帧"，与 exocad 的顺序一致
（**边缘线 → 就位方向 → 局部方向 → 冠底**）。

### 2.2 帧必须绑定当时的边缘线版本

确认帧的原点是**边缘线质心**（`PatientFrame(origin=margin.mean(0), ...)`），此前帧不记录边缘线版本
⇒ 改了 Margin 后帧会自动漂移到新质心而"看起来仍然合法"。本批：

- `confirm_frame` 写入 `margin_revision_id`；
- `frame_from` 校验它等于 `confirmed_margin_revision_id`，不等就 409「边缘线已修改，请重新确认…」。

副作用（有意）：**改 Margin 后必须重新确认方向**。测试里所有"先改 Margin 再预览"的用例都已按新顺序写。

### 2.3 legacy `/outer` 的语义

- 模板同样走 `resolve_template_source`（不再直接读 `TEMPLATE_DIR/FDI_x.stl`）；
- `generator_mode=='template'`（地标 kabsch+RBF）**必须** `legacy_landmarks=true` 才执行，否则 409，
  并在响应/修订里记 `legacy_flags=['LEGACY_LANDMARK_KABSCH_RBF']`；
- AI 分支记 `legacy_flags=['RESEARCH_AI_CHECKPOINT']`、`placement_path='AI_PREDICTED'`；
- **顺手修的真相覆盖 bug（两处）**：旧代码把 `landmarks_json` 整表覆盖成
  `p.model_dump()`（legacy `/outer`）或 `{'mode':'auto_initialization'}`（`/outer/auto`），
  会抹掉 `_selected_template` 与 `_body_dependency_digest` ⇒ "用户确认过的牙库件在下一次初稿时被静默丢弃"。
  现在两处都只做 merge。

### 2.4 原子修订（五个入口）

| 入口 | 之前的断点 | 现在 |
|---|---|---|
| `POST /designs/{id}/outer`（legacy/AI） | publish → 另一次 INSERT revision | `publish_artifacts(..., revision=)` |
| `POST /designs/{id}/outer/auto` | 同上 | 同上 |
| `POST /designs/{id}/edit-outer` | 同上（且 digest 取自旧行） | 同上（digest 用 `{**d, **updates}` 预构造 payload） |
| `POST /designs/{id}/restore-outer` | 同上 | 同上 |
| `POST /designs/{id}/finalize` | 同上 | 同上（FINAL revision 与产物同事务；payload 带 provenance） |

判据：进程死在两步之间时，不再出现"当前几何已换、修订链没记"或反向的不一致。
`publish_artifacts` 的 `revision=` 路径在 `BEGIN IMMEDIATE` 内分配 `revision_no` 并 INSERT，
与 `UPDATE designs` 同一事务（A0 已有的契约，本批只是把调用点补齐）。

### 2.5 产物自证来源（"任何 STL 都能自证"）

`finalize`：

- `qc['provenance'] = {template_source{sha256, source_kind, selection_revision_id, ...},
  patient_frame{revision_id, source}, placement{implementation, outer_path, outer_sha256},
  consumed_revisions, input_dependency_digest, algorithm_version}`；
- sidecar `<final_stem>_provenance.json`（修订 `artifacts['provenance']` 指过去）。

`initialize_draft_outer`（初稿/baseline）与 legacy `/outer`：同样写 `<outer_stem>_provenance.json`。

### 2.6 摘要（dependency digest）覆盖真正消费的字节

- **确认帧参与摘要**：`dependency_digest` 现在总是包含帧的**几何投影**
  （`x/y/z/buccal/units/axis_revision_id/margin_revision_id/prep_sha256`，不含每次确认都会换的
  `revision_id`）。此前只在"选了牙库"时才计入 ⇒ 改方向后旧外形仍算 `current`；而纳入整份 JSON
  又会让"原样再确认一次"误判过期，故取几何投影。
- **回退模板也进摘要**：没有人工确认时，若回退到 template_studio 候选或牙库件，
  其文件字节也进 `sha256` 表 ⇒ 牙库文件被换掉会让已发布外形失效（`FALLBACK_DESIGN_DIR` 早已覆盖，跳过）。

### 2.7 阻断与提示（fail loud，不猜）

- `workflow/eligibility`：无确认帧 → `blockers += {stage:'FRAME'}`，`template_allowed=False`；
- `/status`：`generator_mode=='template'` 且输入已确认但无帧 → `{code:'FRAME_UNCONFIRMED'}`，
  于是 `next_actions` 不含 `generate_outer`（前端"生成"按钮因此不可用，并给出"先确认局部方向"入口）。
  **AI 模式不加这条**：AI 推理不以帧为输入，加了会把研究路径一起锁死。
- 候选报告新增 `template_source` / `template_source_warning` / `frame_source` / `patient_frame` /
  `placement_path` / `baseline_sha256`；前端"候选形态"页新增"本次放置的来源（自证）"三行表。

## 3. 回归与证据

新增 `tests/integration/test_a1_placement_parity.py`（5 条）：

| 断言 | 结果 |
|---|---|
| `/auto-fit/template-preview` 的放置结果与 `/workflow/generate` 的 baseline **逐字节相同**（真牙库 `洋溢-长夜同在`，eoff 通过 `placed_library_crown` + 确认帧） | ✅ 本机不 skip |
| `/outer/auto` 与 baseline **逐字节相同**（同一模板、同一帧、同一实现） | ✅ |
| 候选报告带 `template_source.sha256` / `patient_frame.revision_id` / `placement_path` / `baseline_sha256` | ✅ |
| 采纳牙库件后再走初稿：`_selected_template` 不被抹掉，且初稿仍用该牙库、与 baseline 逐字节相同 | ✅ |
| 无确认帧：`/outer/auto` 409；`eligibility` 唯一 blocker 是 `FRAME`；`/workflow/generate` 409 且 blockers 含 FRAME | ✅ |

改动测试（按新 live 顺序重写，非放宽）：
`test_auto_fit.ready` / `test_revision_finalize_integrity.ready_design` / `test_revision_chain`（先确认配置与帧）、
`test_a0_draft_template_identity`（新增"无帧 409"断言 + 显式 legacy 才走 world-X）、
`test_designer_v1.test_save_restore_and_boolean_rebuild`（legacy 路径显式开启）。

**本批运行**（本机）：**全量 `pytest tests` = 535 passed / 10 skipped / 3 xfailed / 3 failed**；其中
`tests/unit + tests/integration` = **413 passed / 10 skipped**，那 3 条失败是工作树里**他人未提交**的
`app/geometry/outer.py` 非平面守卫（`MarginReconstructionRequired`，`_metrics.py:21` 直接调用
`trim_outer_at_margin`），与 A1 路径无关（A1 未触碰 `geometry/outer.py` / `intaglio.py` / `boolean.py`）。
前端 `npm run build` 通过。

顺手修的测试隔离漏洞：`test_revision_chain` 的 fixture 只 patched `config.TEMPLATE_DIR`，
上传模板实际写进了**真实运行目录** `data/templates/FDI_36.stl`（已删除该污染文件）；
现已同时 patched 模块别名。注意：`resolve_template_source` 在调用时从 `config` 取路径，
所以"只 patched config 而不 patched 各模块已绑定的别名"这类隔离会再次骗过测试。

## 4. 已知残留（诚实清单）

1. **帧的 origin 是派生量**（Margin 质心）。本批用 `margin_revision_id` 绑定版本解决漂移，
   但"帧 = 方向（3 向量） + 绑定版本"仍是两段式；更彻底的形态是让帧直接存 origin 并由 Margin 修订驱动。
2. **AI 模式的 `/outer` 仍是研究路径**（`predict_outer` 无帧输入），只标 `RESEARCH_AI_CHECKPOINT`，
   未纳入"确认帧"纪律；产品化时要么给 AI 也传帧，要么明确它不在 live。
3. **`/auto-fit/template-preview` 仍限天然基牙**（`checked(natural_only=True)`）；基台的模板预览走
   `workflow/generate` 的 TEMPLATE_ONLY。
4. **基台病例要靠接口层确认帧**，而"确认帧"要求已确认配置（`confirmed_inputs`），
   所以基台也不允许"没参数就先放一个"。
5. **候选采纳（`auto-fit/{id}/accept`）写 outer 修订时**只校验候选字节，不再补一次 provenance sidecar；
   provenance 在候选 payload 里（`template_source`/`source_template_path`）。
6. **B 阶段未开始**：`ContactMeasurement` 四路 signed 语义统一（审计 #5）仍是下一步。

## 5. 下一步（B）→ 已完成

定义唯一 `ContactMeasurement`（实现源 `crown_outer/auto_fit_adapter.FitSurface`），
让 Placement / AutoFit / Candidate Review / Final QC 共用同一 signed 语义，并落审计 #5 的 parity 回归。
**已完成**：见 `B_CONTACT_MEASUREMENT_2026-10-12.md`（`services/contact_measurement.py` +
`tests/integration/test_b_contact_parity.py`：四条路径在同一颗冠上给出同一组有符号数字）。
PHI 三级政策（默认 `PSEUDONYMIZED`）作为并行治理任务落地。
