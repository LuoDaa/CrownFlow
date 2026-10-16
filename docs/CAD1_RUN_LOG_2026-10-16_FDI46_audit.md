# CAD-1 快速基台流程实跑日志（二）— FDI46 审计版

- 日期：2026-10-16 02:19–02:25
- 代码：`e0c25ec`（生成审计 + 可复用几何缓存 + shell 决策 + Margin immutable band）
- 运行方式：按 UI 同序 API 跑（`run_gate.py`），最多 3 轮自动纠正，门禁判据与 `abutmentFast.js` 一致
- 病例：case `61d3a6ab…` / design `2ef7cf7d…`，**FDI46**，单侧邻牙（mesial ABSENT / distal CONFIRMED），模板/帧/咬合/邻牙/参数均已确认
- 脱敏：无患者姓名，只保留 uuid 与牙位
- 原始材料（本次一并入库）：
  - `docs/audit/CAD1_FDI46_2026-10-16_generation_audit.jsonl`（93 行事件）
  - `docs/audit/CAD1_FDI46_2026-10-16_gate_rounds.json`（每轮门禁判定 + 阶段状态）

> 对照：上一份 `CAD1_RUN_LOG_2026-10-16_FDI26.md` 记录的是 `76d00fd`+`bb67ad4`（三轮逐字节相同、单轮 ~7 分钟）。

## 结论速览

1. **提速达成了**：单轮从 ~420 s 降到 **63 / 75 / 87 s**（审计 `total_elapsed_s` 52.6 / 68.5 / 80.6），约 **5×**。
2. **跨轮反馈闭环成立了**：三轮 `outer` sha256 互不相同（`a1677493` → `ff74f1ec` → `5bbb7863`）⇒ 上一份日志的发现 #1（重试无几何反馈）确实修掉了。
3. **门禁仍然每轮失败**，但原因换成了三个可定位的问题：咬合阶段根本没跑（`mode=PROXIMAL_ONLY` ⇒ 前端读到 `Occlusion:None`）、Margin 判据在 refinement 轮读到 no-op 的陈旧 `after_mm`、Proximal 真未达标。
4. **Final 仍然 400**，自交 **2743**（A_margin 6 / B_cervical_offset 2652 / C_stitch_sweep 85）。

## 每轮数字

| | round 1（冷启） | round 2 | round 3 |
| --- | --- | --- | --- |
| 候选 | `4d948532` | `08972fc8` | `66e2a419` |
| 墙钟 | 63 s | 75 s | 87 s |
| 审计 total | 52.6 s | 68.5 s | 80.6 s |
| outer sha16 | `a1677493d9e9d973` | `ff74f1ec692856af` | `5bbb7863f4fb8ba0` |
| `mode` | PROXIMAL_ONLY | PROXIMAL_ONLY | PROXIMAL_ONLY |
| 门禁失败项 | Proximal / Occlusion:None | +Margin:P95 | +Margin:P95 |
| `CERVICAL_FIT` after_p95 | **0.01339**（before 1.152，真拟合） | 0.41993（before 0.41993，**no-op**） | 0.40733（同上） |
| 整圈反向覆盖 | 1.0 / reverse_p95 0.021538 | 同 | 同 |
| `SHELL_REPAIR_DECISION` | SKIPPED_SMALL_LOCAL_CHANGE | 同 | 同 |
| `INNER_SHELL_FINAL_AUDIT` | REVIEW_REQUIRED | REVIEW_REQUIRED | REVIEW_REQUIRED |

每阶段耗时（审计 `elapsed_s`，单位秒）：

| 阶段 | r1 | r2 | r3 |
| --- | --- | --- | --- |
| SUPPORT_PREPARE | 2.90 | 2.90 | 2.91 |
| INTAGLIO_SPACE_PREVIEW | 7.08 | 7.10 | 7.20 |
| GLOBAL_PREFIT | 12.73 | 20.59 | **27.36** |
| CERVICAL_FIT | 0.12 | 0.13 | 0.13 |
| PROXIMAL_SOLVE | 2.52 | 2.30 | 2.73 |
| INNER_SHELL_RECHECK | 20.36 | 27.21 | **31.08** |
| INNER_SHELL_FINAL_AUDIT | 5.43 | 6.99 | 7.82 |
| GEOMETRY_QC / PERSIST_AND_READBACK | 0.44 / 0.11 | 0.42 / 0.11 | 0.43 / 0.14 |

- **当前单轮耗时前两位**：`INNER_SHELL_RECHECK`（20→31 s）与 `GLOBAL_PREFIT`（13→27 s），两者合计占一轮 ~70%；且**两轮都随轮次变大**（同一病例、同一拓扑，outer 文件大小三轮都是 601,984 字节）。
- 缓存实测：`CACHE_NEIGHBOR_SURFACE` MISS → **HIT → HIT**（内存缓存生效 ✅，命中后该步 0.0014 s）；`CACHE_INTAGLIO_PREVIEW` **三轮全 MISS，且 cache_key 每轮都不同**（`7e710a95…` → `118d0e3c…` → `c0c2e43b…`）。

## 发现（按影响）

1. **咬合/冠高阶段根本没跑（最要命）**：三轮 payload 的 `mode` 都是 `PROXIMAL_ONLY`，阶段集合里没有 `OCCLUSAL_HEIGHT_PREFIT` / `OCCLUSAL` / `OCCLUSAL_RECHECK_AFTER_SHELL`，审计里也没有对应的 span；payload 连 `occlusal` 表面键都没有（只有 `mode / bite_revision_id / input_diagnostics / proximal_asset / placement_path`）。代码里这段是有的（`auto_fit.py:531/556`，`if mode in ('OCCLUSAL_ONLY','BOTH')`），所以是**咬合表面没被构建出来**导致 `mode` 退化成 `PROXIMAL_ONLY`。
   **影响**：① 结论 #3 的 "咬合必须 PASS" 在当前构建里**永远无法判定**（前端 `latestStageOf([...OCCLUSAL...])` 拿到 null ⇒ `Occlusion:None`）；② 冠高/咬合根本没被求解，这三轮做的只是"侧向 + 壳"。
2. **Margin 判据在 refinement 轮误报**：round 1 `CERVICAL_FIT` 是真拟合（after_p95 0.01339，before 1.152）⇒ 门禁没有报 Margin；round 2/3 是 no-op（before == after = 0.42）⇒ 报 `Margin:P95_EXCEEDS_0.05MM`。而同三轮的**整圈反向覆盖都是满的**（`within_0_05mm_fraction 1.0`、`reverse_p95 0.021538 mm`）。即：判据该看反向覆盖，或 no-op 轮跳过 `after_mm`（与上一份日志的发现 #2 同源，这次在另一个病例上复现，且能看出"第 1 轮过、后续轮误报"的规律）。
3. **`SHELL_REPAIR_DECISION` 的名字会误导**：三轮状态都写 `SKIPPED_SMALL_LOCAL_CHANGE`，但同一事件里 `moved_vertices` 1220 / 781 / 685（`moved_fraction` 20.3% / 13.0% / 11.4%，`max_move_mm` 1.53 / 0.75 / 0.74），`major_change: true`，阈值 `major_vertex_threshold 181 / major_move_threshold_mm 0.12`。另外 `proximal_domain_touched` 在 r1/r2 是 **true**、r3 才 false。也就是说决定重算与否的依据并不是"改动小"，而是 drift 标志；状态名与 `major_change` 字段自相矛盾，读审计的人会被带错。建议改名（如 `SKIPPED_NO_CONSTRAINT_DRIFT`）并把判定依据单列字段。
4. **Intaglio 预览缓存从未命中**：key 每轮都变。按设计 key 只该由 `support + Margin + Axis + 影响内冠的参数` 决定，这三轮这些都没变 ⇒ key 里混进了每轮都会变的东西（或哈希输入不稳定）。当前每轮白花 ~7.1 s，更关键是**它本该让"只改邻接/咬合不改内冠"时跳过这一步**，现在没起到作用。
5. **内冠壳审计从 PASS 退到 REVIEW_REQUIRED**：三轮都带警告 `INNER_SHELL_SPACE_CONSTRAINT_REVIEW_REQUIRED`，`INNER_SHELL_FINAL_AUDIT = REVIEW_REQUIRED`（`INNER_SHELL_POST_REPAIR_AUDIT` 则 SKIPPED，原因 `NO_CONTACT_REPAIR_GEOMETRY_CHANGE`）。而这一轮 shell 推挤是 11–20% 顶点 / 最大 0.74–1.53 mm。
6. **Final**：`rebuild-cervical` 400，`self_intersections 2743`（`A_margin 6 / B_cervical_offset 2652 / C_stitch_sweep 85`）。把历次数字排起来看是单调变差：576 → 800 → 1553 → **2743**，而 shell 推挤幅度也在变大 ⇒ 嫌疑集中在**壳推挤把外皮折了**（B_cervical_offset 占 97%）。
7. **Margin 作为根确实生效**：`CERVICAL_LOCK_AFTER_PROXIMAL` `restored 0 / displacement 0.0`；`CERVICAL_LOCK_AFTER_SHELL` `restored 73–78，max_pre_restore_displacement 0.577 → 0.390 → 0.106 mm` ⇒ 壳推挤后被移出颈带的顶点被拉回 Margin 对齐位置，且位移逐轮收敛（0.577→0.106）。这条按设计工作。
8. `SUPPORT_PREPARE` 每轮 2.9 s × 3 未缓存（你已列为下一步）。

## 建议的下一步（按性价比）

1. **修咬合表面构建**（否则"三个最低条件"里的第 3 条无法判定，冠高也没解）——这是本轮的 P0。
2. **Margin 门禁判据换成整圈反向覆盖**（或 no-op 轮跳过 `after_mm`）。
3. **修 Intaglio 缓存 key**：让 key 只由 support/Margin/Axis/内冠参数决定；顺带把 `SUPPORT_PREPARE` 也纳入缓存。
4. **`SHELL_REPAIR_DECISION` 改名 + 分字段**，并把"大改动但无 drift"这种情形在审计里说清楚。
5. **查 Final 的 2743**：先看是不是壳推挤造成外皮自交（可在推挤后加一次"外皮自交增量"审计，超阈值就回退到推挤前 + 报 REVIEW）。当前 `INNER_SHELL_FINAL_AUDIT` 已经 REVIEW_REQUIRED，本该在推挤这一步就拦住。
6. 性能：`INNER_SHELL_RECHECK`(20→31 s) 与 `GLOBAL_PREFIT`(13→27 s) 是仅剩的两个大头，且都逐轮变慢，值得按轮次分解一次。

## 复现

```bash
# 每阶段耗时 / 缓存 / 决策（本文件同源）
Get-Content .\data\cases\<case_id>\<design_id>_generation_audit.jsonl -Wait
```
本次已固化为 `docs/audit/CAD1_FDI46_2026-10-16_generation_audit.jsonl`（93 行）与 `…_gate_rounds.json`。
