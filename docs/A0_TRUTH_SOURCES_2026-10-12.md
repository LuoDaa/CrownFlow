# A0：收敛三个 truth source（2026-10-12，第一批 PR）

范围：**只做 A0（模板来源 / 患者帧）+ 审计列出的回归测试 + 两处安全一行修复**。未碰任何算法参数、
未迁移 PHI 政策、未做事务统一（见"未做项"）。

## 1. 问题（审计 P0 的源码事实）

| 症状 | 源码事实 |
|---|---|
| 预览与实算两套帧语义 | `geometry/registration.py` 用 `x=world_X 投影到 ⊥axis` 重建横向轴；`api/routes.py::initialize_draft_outer` 的库分支**同样**重建（**两处**） |
| 报告哈希 ≠ 几何来源 | `initialize_draft_outer` 走 `TEMPLATE_DIR → template_studio → 第一个牙库`，**不读** `landmarks['_selected_template']`；而 `services/auto_fit.py` 的 report 优先写 `_selected_template.sha256` ⇒ `Hash_reported ≠ Source(Geometry)` |
| 路径穿越 | `services/crown_outer_runner.py` 用 `str(p).startswith(str(root))`（`r1` vs `r10` 前缀碰撞，已复现） |
| 目录名未校验 | `api/crown.py::register_bundle` 直接把 `case_uuid` 拼进路径 |

## 2. 本批实现

### 2.1 唯一 Template Source：`services/template_source.py`
- `ResolvedTemplate(path, sha256, fdi, source_kind, library, proposal_id, selection_revision_id, metadata_*, library_obj)`；
- `resolve_template_source(design)`：**有确认记录时**校验磁盘真实字节（存在 + sha256 一致 + 元数据一致），
  不一致一律 fail loud（`TEMPLATE_HASH_MISMATCH` / `TEMPLATE_SOURCE_MISSING` / `TEMPLATE_LIBRARY_UNAVAILABLE`），
  **绝不静默回退**；无确认记录才走回退链，并如实标注 `FALLBACK_DESIGN_DIR / FALLBACK_TEMPLATE_STUDIO / FALLBACK_LIBRARY`；
- 消费方已统一：`api/routes.py::initialize_draft_outer`、`services/auto_fit.py::propose`（report 里的
  `template_sha256` + `template_source` 现在就是**真正读入的字节**）。

### 2.2 唯一 Patient Frame
- `geometry/crown_outer/patient_frame.py`：新增 `revision_id` 字段、`validate_patient_frame()`
  （单位性 + 两两正交 + 右手性，**不静默正交化**）、`confirmed_patient_frame(origin, mesial, buccal, insertion)`；
- `geometry/registration.initialize_from_margin(..., patient_frame=None)`：传帧时**不再推断 X/Y**，
  且要求帧 Z 与 `insertion_axis` 一致（否则拒绝）；不传帧才走 legacy world-X，并发 `DeprecationWarning`；
- `services/pipeline.generate_initialized_outer(..., patient_frame=None)` 透传，返回值带 `frame_source`；
- `api/routes.py::initialize_draft_outer`：**库件只放置一次**（`placed_library_crown` 内含颈缘 footprint 拟合，
  不再对已放置网格二次 `initialize_from_margin`），且优先用 `api.auto_fit.frame_from` 的确认帧，
  未确认时回退 world-X 并标 `frame_source='WORLD_X_FALLBACK'`；返回 `template_source` / `placement_path`；
- `api/auto_fit.frame_from` 现在把 `revision_id` 写进帧并调用 `validate_patient_frame`；
- `services/auto_fit_templates.placed_library_crown` 显式传**规范局部帧**（+Z 就位、X=近中），
  不再触发 legacy 分支。

### 2.3 安全一行修复
- `crown_outer_runner.artifact_path`：`str.startswith` → **`p.is_relative_to(root)`**；
- `api/crown.register_bundle`：`case_uuid` 加白名单 `[A-Za-z0-9._-]{1,64}`（不再把路径安全寄托在 `case.json` 存在上）。

## 3. 回归测试（审计列的 6 条，本批覆盖 5 条）

| # | 审计要求 | 落地 | 结果 |
|---|---|---|---|
| 1 | Selected template identity（TEMPLATE_DIR 有诱饵 B 时，几何必须来自 A） | `tests/integration/test_a0_draft_template_identity.py`（Z 向高度作判据 + 报告 sha == 真实字节；篡改后 400 `TEMPLATE_HASH_MISMATCH`） | ✅ |
| 2 | Preview / Generate 帧一致 | 结构上统一：两条路径现在**同一个 helper + 同一个确认帧**（`placed_library_crown(frame_from(d))`），并补了 helper 级帧一致性断言 | ✅（结构等价） |
| 3 | Rotation invariance | `tests/unit/geometry/test_a0_patient_frame.py::test_placement_with_confirmed_frame_is_rotation_equivariant`（`place(R·X) ≈ R·place(X)`，atol 1e-6）；并加一条**反例**证明 legacy world-X 路径不等变（drift > 1e-3） | ✅ |
| 4 | Provenance geometry identity | 同 #1：`template_source.sha256` 由磁盘字节算出，且与 draft 实际消费的路径同源 | ✅ |
| 5 | Contact parity（四路 signed 一致） | **未做** —— 属 B 阶段（`ContactMeasurement` 服务化） | ⏳ |
| 6 | Artifact prefix collision 永久回归 | `tests/unit/services/test_a0_artifact_path.py`（`r1 + ../r10/secret.txt` 必拒；合法路径仍可读） | ✅ |

另：`tests/unit/services/test_a0_template_source.py` 覆盖模板来源四种语义（确认/篡改/缺失/回退标注）。
**本批运行**：A0 四组新测试 + 既有 `test_auto_fit` / `test_generation_workflow` / `test_margin_tools_api`
等共 **58 passed**；先前全量 510 passed（4 failed 均为工作树里他人未提交的 `outer.py` 守卫冲突与内存偶发，见 `AUDIT_VERIFICATION_2026-10-12.md`）。

## 4. 未做项（明确边界）

1. **PHI 三级政策**（IDENTIFIED / PSEUDONYMIZED / DEIDENTIFIED_EXPORT）需要你定默认档与"导出包禁什么"，
   本批未改任何 PHI 行为（`keep_source_package` 默认值未动）。
2. **原子修订迁移**（`publish_artifacts(..., revision=)` 统一到 edit/restore/legacy outer/finalize）未做。
3. ~~**Legacy 端点** `/designs/{id}/outer`、`/outer/auto` 仍走标注回退（`frame_source='WORLD_X_FALLBACK'`）~~ →
   **已由 A1 关闭**：`/outer/auto` 无确认帧直接 409；`/outer` 的地标分支必须显式 `legacy_landmarks=true`
   且标 `LEGACY_LANDMARK_KABSCH_RBF`（见 `A1_PLACEMENT_TRUTH_2026-10-12.md`）。
4. **Contact 语义统一**（审计 #5）= B 阶段。
5. ML canonicalization / representation / RNG（P2）未动。

> 另：A0 未做项 2「原子修订迁移」也已由 A1 关闭（`edit / restore / legacy outer / finalize` 统一到
> `publish_artifacts(..., revision=)`），且产物旁新增 provenance sidecar。

## 5. 下一步（A1 → B）

- ~~A1：把 placement 的其余入口全部收敛~~ → **已完成**（`A1_PLACEMENT_TRUTH_2026-10-12.md`）：
  legacy/AI `/outer` 与 `/outer/auto` 都走 `resolve_template_source` + 确认帧（无帧 409），
  候选报告/修订/产物 sidecar 三处带 `template_source`/`patient_frame`/`placement`，
  并加了逐字节 parity 回归（preview ↔ baseline、outer/auto ↔ baseline）。
- B：定义唯一 `ContactMeasurement`，placement / AutoFit / candidate review / final QC 共用
  （实现源用 `crown_outer/auto_fit_adapter.FitSurface`），并落审计 #5 的 parity 回归。
