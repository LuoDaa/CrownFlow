# CAD-1 快速基台流程实跑日志 — FDI26

- 日期：2026-10-16 01:31–01:58（另有一次 02:02 起的重跑，见文末）
- 代码：`76d00fd`（三条件硬门禁）+ `bb67ad4`（前端自动纠正三轮）
- 后端：`uvicorn app.main:app --host 0.0.0.0 --port 3277`
- 输入：用户从 UI 上传的上/下颌扫描（case `cbdb208d…`），牙位 FDI26，模板 `USER_SELECTED / 6.eoff`，库 `炼金王`，参数 targets 0.01 / 0.01 / machine_axes 5
- 脱敏：本日志不含患者姓名与原始扫描路径，只保留 case / design 的 uuid 与牙位
- design `4e65c3d5452e452b8f40fd9499bb1628`，最终 `status=outer_ready`、`final_path=None`

## 结论速览

1. **三轮自动纠正产出同一个网格。** rev13 / rev15 / rev17 的 `outer` 三个文件**逐字节相同**（723,084 B，sha256 `d8fcb9c558f8f106…`），三轮 payload 的 `baseline_outer_sha256` 也都是 `dadc469096e4…`、`baseline_outer_revision_id` 都是 `None`。也就是说"采纳候选 → 再生成"这一步**没有把上一轮外形变成下一轮基线**，校验结果没有驱动任何几何变化，只把耗时乘了 3。
2. **三门禁：Margin ✗（判据用错指标）/ Proximal ✗（真未达标）/ Occlusion ✓（但两处咬合数字互相矛盾）。**
3. **第 3 轮失败后的 publish 阶段 `accept` 被连续调用两次**（第二次 409 Conflict），前端会在抛错处中断，用户看不到"已自动求解 N 轮 · M 项仍需检查 / 保留当前最接近结果供 3D 复核"的收尾。

## 每轮数字（rev13 `98210a9e` / rev15 `77e01126` / rev17 `a3300105`）

三轮 payload 的下列字段**逐位相同**：

| 项 | 值 |
| --- | --- |
| Margin 最后一次 refit | `CERVICAL_REFIT_FINAL` `PASS_ENGINEERING`，rim 3086 / seam 1544 |
| Margin `after_mm`（= `before_mm`，说明这一步没动） | median 2.21101 / p95 **5.69018** / max 5.87493 mm |
| Margin 整圈反向覆盖 `full_loop_coverage` | `within_0_05mm_fraction 1.0`，`reverse_p95 0.010437 mm`，`reverse_max 0.013624 mm`，covered 128 / uncovered 0 |
| `transition_band_mm.after_p95` | 5.7937 mm |
| Proximal | `CONSTRAINT_CONFLICT` / `PROXIMAL_TARGET_NOT_REACHED` |
| Proximal mesial | `MEASURED` valid 734 / domain 1767，`physical_after.mesial.target_reached = False` |
| Proximal distal | `MEASURED` valid 615 / domain 1330，`physical_after.distal.target_reached = False` |
| Occlusal | `PASS_ENGINEERING`，`after.occlusal.signed_valid.p50 = +3.2808 mm`，`penetration.max = 0.0342 mm` |
| 高度闭环 `OCCLUSAL_HEIGHT_PREFIT` | `PASS_ENGINEERING`，mode `ONE_SIDED_AXIAL_CLOSED_LOOP_ANTAGONIST_HEIGHT_V3` |
| 内冠壳 `INNER_SHELL_FINAL_AUDIT` | `PASS_ENGINEERING`，min 0.79962 / p05 0.8 / deficit_max 0.0004 mm |
| warnings | `CROSSING_WORSENED_OR_NOT_ASSESSED`、`DEFORMATION_REVIEW_REQUIRED`、`OUTER 在 Margin 过渡面以上不能提取单一开放颈环面` |

## 每阶段耗时（第 1 轮 tag `c22a9d`，用阶段 STL 的 mtime 计算）

| 阶段 | 相对上一阶段 | 说明 |
| --- | --- | --- |
| diagnostic_crop | — | 起点 |
| intaglio_preview | +17 s | blockout + cement gap 场 |
| prefit | +40 s | 患者空间整体变形 |
| cervical | +0 s | 幂等 no-op |
| proximal | +16 s | |
| **height_prefit** | **+181 s** | 对颌高度闭环 |
| occlusal | +17 s | 局部牙尖/窝沟 |
| **shell_guard** | **+168 s** | 内冠壳推挤 + 其后的重解/refit |

第 2 轮 `cf36e2`：17 / 40 / 0 / 16 / **180** / 18 / **168** s；第 3 轮 `b83dce`：17 / 34 / 0 / 10 / **118** / 11 / **143** s。
→ 单轮 ≈ 7 分钟，其中 `height_prefit` + `shell_guard` ≈ 348 s，占 **~78%**；其余全部合计约 90 s。

## 发现（证据 → 影响）

1. **自动纠正没有反馈闭环。** 三轮 `outer` sha256 相同、`baseline_outer_sha256` 相同 → 采纳候选没有改变生成基线，三轮是同一计算的重复。**影响**：单病例耗时 ×3、且"继续自动纠正"的承诺不成立。修法方向：让 `workflow/generate` 接受并消费"上一轮已采纳外形"作为模板/基线（现在 `template_source` 三轮都是同一个 `6.eoff`）。
2. **Margin 门禁的 p95 子项读的是一个 no-op 的陈旧指标。** `CERVICAL_REFIT_FINAL` 这次是幂等 no-op（`before_mm` 与 `after_mm` 完全相同，rim 3086 / seam 1544 这个 seam 集合覆盖了大量远离终止线的顶点），`after_mm.p95 = 5.69 mm`；而**同一次调用的反向整圈覆盖是满的**（`within_0_05mm_fraction 1.0`、`reverse_p95 0.0104 mm`、uncovered 0）。前端 `abutmentFast.js` 用的是 `after_mm.p95 ≤ 0.05` → 必然判失败。**影响**：整圈明明对齐，Margin 条件仍报 `Margin:P95_EXCEEDS_0.05MM`。修法：no-op 轮改判 `full_loop_coverage`（`within_0_05mm_fraction`+`reverse_p95`），或在 stage 报 `idempotent/no-op` 时跳过 `after_mm` 子项。
3. **Proximal 是真的没达标，不是 schema 问题。** 已核对：前端读 `proximal.physical_after[side].target_reached`，与后端写入路径一致；本次两侧都是 `False`。**影响**：这是三门禁里唯一"真失败"的一项；`valid/domain`（734/1767、615/1330）说明新求解器确实在真实域上跑了，但没把目标侧的接触做到位。
4. **咬合两套数字互相矛盾。** `OCCLUSAL_HEIGHT_PREFIT` 第 1 轮 trial 记录：`before.penetration_max 4.466 mm / p95 3.836 mm / fraction 0.682`、`signed_median −0.488`，方向 `LOWER_TO_RELIEF_PENETRATION`、实际下移 −1.1 mm，`after.penetration_p95` 仍 **3.030 mm**；而 `OCCLUSAL` 阶段报 `signed_valid.p50 = +3.2808 mm`（间隙）、`penetration.max 0.0342 mm`。两个阶段对"咬合"给出相反符号、量级都很大。**影响**：门禁只看阶段状态（`PASS_ENGINEERING`）就放行，但真实几何并未收敛到可接受咬合；同时这两套度量至少有一套的参考面/采样域/符号有问题，需要先统一到"一个参考面 + 一个采样域"。
5. **publish 双 accept → 409。** 日志顺序：`generate 200` → `accept 98210a9e 200` → `generate 200` → `accept 77e01126 200` → `generate 200` → `accept a3300105 200` → **`accept a3300105 409`**。原因在 `abutmentFast.js`：循环内失败分支已经 `confirmCrownCandidate`（为下一轮设基线），循环结束后 publish 分支又对同一个 candidate 再 `confirmCrownCandidate` 一次。**影响**：三轮都失败时，前端在 publish 抛错，看不到保留外冠的收尾提示（外冠其实已经落到 `outer_path`）。
6. **耗时结构**：一轮 7 分钟里 `height_prefit` + `shell_guard` 占 78%，且两者跨轮完全重复计算。跨轮不变但每轮重算的还有 `intaglio_preview`（blockout + cement gap 场）、邻牙 patch、对颌采样。

## 怎么复现 / 继续观察

```bash
# 请求级进度（含每轮的 generate/accept 与 409）
tail -f backend/uvicorn_3277.log
# 阶段级进度：每落一个 *_autofit_<tag>_<stage>.stl 就是一个阶段过去
ls -t data/cases/<case_id> | head -8
```
数据面：`designs` / `design_revisions`（`payload_json.stages`）；本次 design `4e65c3d5…` 的候选修订 rev13 / rev15 / rev17。

## 建议的修法（按影响排序）

1. 让自动纠正真正闭环：第二轮起以"上一轮已采纳的 outer"作为模板/基线（`template_source` / `baseline_outer_*` 现在三轮不变）。
2. Margin 门禁判据换成整圈反向覆盖（或 no-op 轮跳过 `after_mm`）。
3. publish 前判断该 candidate 是否已是当前外形，避免二次 `accept` 409。
4. 统一咬合度量到一个参考面/采样域，并把"最大穿透 / 咬合域中位间隙"纳入门禁（别只看阶段状态）。
5. 性能：缓存跨轮不变量（intaglio 预览、邻牙 patch、对颌采样）+ 让 `shell_guard` 后的重解按"本轮实际位移"条件触发。

## 附：本次之外

同一 design 在 02:02:59 又起了一次重跑（tag `9ee03d`），到 02:04:21 处于 `proximal` 阶段；它与上面三轮的关系、结果未纳入本日志。
