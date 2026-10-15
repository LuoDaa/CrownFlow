# Boundary Session：指征感知的边界候选（2026-10-13）

把 `MarginSession` 升级成**指征感知**的 `BoundarySession`。几何一行未改：M1 引擎、`margin_edit.py`
的全部算子、`_validated_margin_points` 校验器、非平面 Margin 守卫都不动；改的只是"这条边界是什么"
以及由此决定的放行/拒绝。

## 1. 为什么必须显式记录语义

同一条闭环在天然牙上是**预备体边缘线**，在已有实体基台上是**冠终止线**——曲线长得像，含义与后续
内冠生成、QC 口径都不同。以前只能从 `support_type` 事后反推，等于把语义寄存在一个与它无关的字段上。
现在语义是会话与修订的**一等字段**：

| indication | boundary_semantics | 中文标签 | 放行 |
| --- | --- | --- | --- |
| `NATURAL_PREP` / `NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN` | `NATURAL_PREP_MARGIN` | 天然牙边缘线 | ✅ 与原来一致 |
| `CROWN_ON_EXISTING_ABUTMENT` | `ABUTMENT_CROWN_FINISH_LINE` | 基台冠终止线 | ✅ **本批新放行** |
| `CUSTOM_IMPLANT_ABUTMENT`（含历史别名 `IMPLANT_ABUTMENT`、`IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL`，归一化到同一族） | — | — | ⛔ 409 `UNSUPPORTED_INDICATION` |
| `SCREW_RETAINED_CROWN` | 未定义 | — | ⛔ 409 `BOUNDARY_SEMANTICS_NOT_DEFINED` |
| 其他未知值 | 未定义 | — | ⛔ 409 `BOUNDARY_SEMANTICS_NOT_DEFINED` |

两点设计取舍：

- **只有 CUSTOM_IMPLANT_ABUTMENT 是 `UNSUPPORTED_INDICATION`**。历史别名被归一化进它，不新增第二类拒因。
- 螺丝固位冠**不是**被"默认当天然牙"处理，而是按名字拒绝：它是直接安装在植体上、另有螺丝通道
  工作流的另一种 construction，其**支撑/边界/螺丝通道语义**都不能沿用 `CROWN_ON_EXISTING_ABUTMENT`。
  注意措辞边界：这里只声明"没有定义相应 `BoundarySemantics`"，**不预先规定它对应哪一条几何线**
  （公开资料能确认的是"不使用 abutment 的另一种 construction + 自己的 screw-hole workflow"，
  不能据此断定边界一定是某个具体几何特征）。宁可退回 409 也不拿别的语义顶替（"自信地错"一票否决）。

## 2. 已有实体基台：能力完全相同，不是第二条代码路径

`detect / Top-K / add-anchor / replace-segment / move / smooth / confirm` 六个端点对
`CROWN_ON_EXISTING_ABUTMENT` 全部开放，走的就是同一套 `margin_edit` 算子与同一个 M1 自动检测。
唯一的差别是会话携带的 `boundary_semantics` 与标签。

"已有实体基台"必须先**被确认为实际修复基台**：支撑网格元数据里 `implant_candidate` 为真而
`restorative_abutment_confirmed` 为假时仍 409（扫描定位件不能当冠终止线来源）——这条守卫保持原样。

## 3. Confirm 写入修订

`record_manual_margin(..., boundary_semantics=...)` →
`margin_payload(..., boundary_semantics=...)`，落进 `payload.boundary_semantics`
（同时写进 `operator_actions[0].boundary_semantics`）。`dependency_digest` 只读 `payload['points']`，
所以新增字段不会让已生成的产物失效。`/margin/confirm` 的响应也回带
`indication / boundary_semantics / boundary_label`。

## 4. 前端

`MarginStep` 的措辞与放行**只认服务端语义**：`session.boundary_semantics` > `design.indication` >
`support_type` 依次回退，标签取自 `boundary_label`。基台病例显示"基台冠终止线 / Crown Finish Line"、
复选框写"已逐面检查实际修复基台的冠终止线"、确认按钮写"确认候选为正式基台冠终止线"，
**不再出现"天然牙边缘线"**。自定义种植基台/螺丝固位冠才显示 409 说明并禁用检测与纠偏点。

## 5. 测试

`backend/tests/integration/test_margin_session_api.py` 由 10 条扩到 15 条：

- `test_custom_implant_abutment_is_still_rejected`（原"基台必须 409"改名，参数改为显式 custom）
- `test_legacy_implant_abutment_aliases_normalize_to_the_same_rejection`（归一化到同一族）
- `test_indication_without_boundary_semantics_is_rejected_by_name`（螺丝固位冠）
- `test_existing_abutment_runs_the_full_boundary_session`（detect→anchor→magnetic→move→smooth→confirm 全通，
  且 revision 的 `payload.boundary_semantics == ABUTMENT_CROWN_FINISH_LINE`）
- `test_natural_tooth_confirm_records_natural_prep_margin_semantics`（天然牙保持原语义）
- `test_unconfirmed_scanned_implant_part_cannot_be_the_boundary_source`（未确认的扫描件仍 409）

## 6. 本批未做

- 未改 Contact / Intaglio / Bite / Material / M1 任何模块。
- 未给 `SCREW_RETAINED_CROWN` 定义种植平台边界语义（需要时另行开口）。
- 会话仍是进程内存储（多 worker 需粘性路由或外部存储）。
- **自动候选**（`record_auto_margin_candidate`，含基台 `draft-preparation` 那条路）仍未带 `boundary_semantics`：
  本批只按需求保证 `confirm` 写入。等那条路径也要自证语义时再一并加，避免与正在并行改动的
  `designer.py` / `routes.py` 冲突。

## 7. 验证结果与"全量绿"的真实状态

本批自身的测试与相邻契约测试全绿（在**当前工作树**上实测）：

- `tests/integration/test_margin_session_api.py` 15 passed
- `tests/integration/test_revision_finalize_integrity.py` 18 passed
- `tests/test_margin.py` + `test_margin_seam.py` + `test_draft_margin.py` 6 passed
- `tests/integration/test_margin_tools_api.py` + `test_revision_chain.py` +
  `test_a1_placement_parity.py` + `test_b_contact_parity.py` 17 passed
- `tests/acceptance`（整棵）118 passed / 2 xfailed / 0 failed / 0 errors
- 前端 `npm run build` 通过

`pytest tests` 全量**当前不是绿**，原因已定位且不在本批：同一工作树里有另一个写者在并行改
（实测 mtime：`services/prep_conditioning.py` 11:33、`geometry/abutment_support.py` 11:34、
`services/generation_workflow.py` 11:35、`integration/conftest.py` 11:46、`geometry/registration.py` 11:48、
`services/abutment_support.py` 12:08，以及 13:24–13:45 的 `geometry/nonplanar_crown.py`、`models.py`、
`services/auto_fit.py`、`services/contact_measurement.py`、`api/designer.py`、`api/routes.py`、
`services/pipeline.py`、`geometry/crown_border.py`、`geometry/manufacturing_qc.py`、
`geometry/minimum_thickness.py`，并新增 `tests/integration/test_existing_abutment.py`）。
两次全量跑得到**不同的失败集**（一次 3 failed/12 errors，一次 5 failed/2 errors），失败的用例全在
我没碰过的文件里，且单独跑即通过（`tests/acceptance` 整棵 118 passed；B contact parity 单独 2 passed）。

## 8. Dependency identity 审计（indication / boundary_semantics 是否参与）

结论：**已经覆盖，不要再加一遍**（实测 + 读码）。

1. `services/design_dependencies.py::dependency_digest` 的 keys 第一行就是
   `['target_fdi','support_type','indication','generator_mode','axis_json','current_axis_revision_id']` ——
   这三个语义字段里的 **`support_type` 与 `indication` 对 `outer` / `body` / `final` 三个阶段全都在**，
   且在 stage 条件之前，不会被任何分支裁掉。实测量化（合成 design dict，仅翻转一个字段）：

   | stage | 翻转 `indication` | 翻转 `support_type` |
   | --- | --- | --- |
   | outer | digest 变 | digest 变 |
   | body | digest 变 | digest 变 |
   | final | digest 变 | digest 变 |

   所以"同一组 points、语义从 `NATURAL_PREP_MARGIN` 变成 `ABUTMENT_CROWN_FINISH_LINE`"
   这条路径下，`dependency_digest` **本来就会失效**（不是靠新增字段才失效）。

2. `boundary_semantics` **不独立**：它是 `indication` 的纯函数（`BOUNDARY_SEMANTICS` 映射无第二自变量），
   把它塞进 digest 属于**不可达的重复**——`indication` 变它必变，`indication` 不变它不可能变。
   因此本批**故意不加**（`margin_payload` 里写它是为了修订自证，不是为了依赖失效）。

3. 顺带查到的两处（**都不在本批改动范围，仅登记**）：

   - `services/abutment_support.py::prepare_design_support` 的支撑条件确认新鲜度判据
     `basis = {source_abutment_sha256, margin_revision_id, axis_revision_id, patient_frame_revision_id}`
     **不含 indication**。当前 API 下 `indication` / `support_type` 都是**创建时一次写入**
     （`create_design` 里 INSERT + 同一条 `UPDATE ... indication=?`，之后无任何端点改写），
     所以这条**不是当前可达的 stale**，只是纵深防御候选；等 `abutment_support.py` 空闲时补一个
     `'indication': indication_of(design)` 进 `basis` 即可（一行）。
   - 同一文件的 `assess_abutment_surface` 在 provenance 里硬编码 `margin_semantics='CROWN_FINISH_LINE'`，
     与本批的 `ABUTMENT_CROWN_FINISH_LINE` 是**近义重复命名**（字段不同、不冲突）。
     统一命名留到下一次动那个文件时处理，避免现在与并行写者抢文件。

顺带修掉一个**与本批无关但真实**的陈旧测试桩：`test_revision_finalize_integrity.py` 的
`paused_geometry(outer,support,final,cutter,*args)` 缺 `**kwargs`，而 `_finalize` 早已把
`prepared_support=` / `support_contract=` 传给 `finalize_crown` —— 于是打桩函数在**调用期**就 TypeError，
请求提前 400 返回，`entered.wait(10)` 必然失败。补上 `**kwargs` 后该文件 18 条全绿。

## 9. 本批改动清单与 freeze 状态（供归属与合并使用）

| 文件 | git 状态 | 说明 |
| --- | --- | --- |
| `backend/app/services/margin_session.py` | untracked（本批新建于上一批） | `BoundarySession` + 语义表 + 拒绝分类 |
| `backend/app/api/margin.py` | untracked | 六端点守卫/语义透传 |
| `backend/app/revisions/models.py` | modified | `margin_payload(..., boundary_semantics=)` |
| `backend/app/revisions/service.py` | modified | `record_manual_margin(..., boundary_semantics=)` |
| `backend/tests/integration/test_margin_session_api.py` | untracked | 15 条 |
| `backend/tests/integration/test_revision_finalize_integrity.py` | untracked | 陈旧测试桩补 `**kwargs`（1 行） |
| `frontend/src/components/PrepMarginAxisStep.jsx` | untracked | 语义驱动措辞 + 只对 custom/无语义阻断 |
| `frontend/src/App.jsx` | modified | `indication` 透传 + 409 归类提示 |
| `docs/BOUNDARY_SESSION_INDICATION_2026-10-13.md` | new | 本文 |

⚠️ 归属提醒：`frontend/src/App.jsx` / `PrepMarginAxisStep.jsx` 在同期**也被并行写者改过**
（例如"基台冠终止线 / Crown Finish Line"、"肩台凸缘 / 台肩凹沟"那几处不是我写的），
仅靠 `git status` 无法把这两个文件干净地归给某一批；合并前需要按 hunk 归属。

**freeze 前置状态（2026-10-13 实测）**：仓库只有一个 worktree、分支 `main`、HEAD `73f8cb6`
（`feat(h2.2)`），而 A0→Boundary 全部批次与并行写者的工作**都还没提交**：
186 条未提交（57 个 tracked 文件 +2757/−813、40+ untracked，含 `api/margin.py`、`api/designer.py`、
`api/auto_fit.py`、`geometry/abutment_support.py` 等从未进过版本库的文件）。
⇒ `git worktree add ../x -b feature/y` 现在只会从 `73f8cb6` 拉出一个**缺掉全部近期工作**的旧树，
必须**先有一次 freeze commit**，再按 agent 拆 worktree。
