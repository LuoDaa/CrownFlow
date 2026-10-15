# Margin 步骤重构：自动候选 + 少量纠偏点 + 局部磁性修线

日期：2026-10-13 ｜ 状态：已完成（后端 10 测试绿 + 前端 build 通过）

回归实测：`python -m pytest tests -q` → **626 passed, 10 skipped, 2 xfailed, 0 failed**（1048.52s）；
`npm run build` → 643 modules transformed，built in 3.37s。

目标：把 Margin 从"要么整圈手绘、要么全盘接受自动结果"改成接近 exocad / 3Shape 的工作流 ——
**用户只需要 1–4 个纠偏点，或一小段磁性修线，就能在几十秒内得到正确的边缘线**。

## 1. 不变量（本批最重要的部分）

1. **候选不落盘**。自动检测产出的 Top-K 候选只活在 `BoundarySession` 里（进程内、TTL 3600s），
   **只有 `/margin/confirm` 会写出正式 Margin 修订**。已确认 Margin 的 truth source 语义一字未改：
   仍然是 `revisions.record_manual_margin` + `_validated_margin_points`。
2. **M1 引擎未被修改**。候选来自既有 `predict_margin(..., return_candidates=True)` 的
   `topk/full/chosen/stats/mode`，加上特征模式变体；`/auto-margin-m1`、`/margin-snap`、
   `/section-preview` 三个老接口全部原样保留（M1 直出按钮仍在，只是要求显式选龈上/龈下）。
3. **解算器改动的是副本**。`geometry/margin_edit.py` 只吃投影后的点集与网格副本，
   不把任何中间状态写回 `design`。
4. **拒绝比假装成功好**。所有局部编辑要么产出一条通过 `_validated_margin_points` 的健康闭环，
   要么如实返回 `REVIEW_REQUIRED` 并**保留原候选**（见下"局部移动的歧义拒绝"）。

## 2. 后端结构

| 文件 | 职责 |
| --- | --- |
| `backend/app/geometry/margin_edit.py`（新） | 纯几何：投影、环参数化、锚点重描、局部段替换、沿表面移动、平滑、去重 |
| `backend/app/services/margin_session.py`（新） | `MarginProposal` / `BoundarySession`（原名 `MarginSession`，保留别名）/ 进程内 TTL 会话存储 / detect / add_anchors / replace_segment / move / smooth / confirm |
| `backend/app/api/margin.py`（新） | 六个 POST 端点 + 指征守卫（基台 → 409 `UNSUPPORTED_INDICATION`） |
| `backend/app/main.py` | 注册 `margin_router` |

### 2.1 会话对象字段

`session_id / design_id / source_mesh_sha256 / algorithm_version / seed_point / anchors /
feature_mode(AUTO|SUPRAGINGIVAL|SUBGINGIVAL) / proposals[] / active_proposal_id /
ambiguity_flags[] / status / operations[] / active_points / as_report()`

**`active_points` 是显式语义**：同分并列时调用方不必自己猜"哪一条才是编辑后的结果"。
`_promote()` 把 `active_proposal_id` 指向**刚被判定的新候选**（这条修掉了一个真实 bug：
分数排序后 active 停在 `proposals[0]`，导致刚做的编辑被旧候选遮住，锚点硬约束实测偏 1.15 mm）。

### 2.2 锚点重描（1–4 个硬约束）

锚点先投影到网格，按角度相邻的锚点两两连线，沿线在网格上重描并重采样；
**锚点最后写入**（在 snap/resample 之后），任何后续操作都不得移动它们。
`MIN_SPACING_MM = 0.03` 去重（保留锚点优先）。

### 2.3 局部编辑三档

- **编辑**：`move` = 沿局部表面的行走，单步 ≤0.05 mm，patch 半径 ≤0.6 mm，
  用面积加权的主导簇法向（先做符号对齐）；`dominant < 0.6` 时**拒绝**并回原候选
  （`EDGE_OR_CREASE_LOCAL_NORMAL_AMBIGUOUS` + `status=REVIEW_REQUIRED`）。
  整圈上下移动后重新投影到网格，**不是世界 Z 平移** —— 这条写进了操作 note，测试直接断言。
- **磁性**：`replace_segment` 只替换笔迹角度区间**覆盖的那一段**，接缝端点保留，
  返回 `untouched_index` 供前端验证"对侧没被动过"。
- **自由绘制**：兜底路径，沿用既有整圈手绘 + `rough_margin` 补闭环。

### 2.4 歧义与确认

`detect` 归一化正向分数 `1/(1+energy) + 0.5*(1-1/(1+gap))`；
当 `|score_a - score_b| <= 0.05*max(score)` **且**形状 p95 差 > 0.35 mm 时打
`STRONG_CANDIDATES_DISAGREE(...)`。存在歧义旗标且调用方没给 `proposal_id`/锚点时，
**`confirm` 被阻断**，必须先由人显式选一条。

## 3. 前端

- `frontend/src/api.js`：新增 `marginDetect / marginAddAnchor / marginReplaceSegment /
  marginMove / marginSmooth / marginConfirm` 六个调用。
- `frontend/src/components/PrepMarginAxisStep.jsx`：主工具行 = **自动检测 / 编辑 / 磁性 / 自由绘制**；
  辅助行 = **自动 / 龈上（凸）/ 龈下（凹）**（`marginFeatureMode` 默认改成 `AUTO`；
  `AUTO` 下 `按模式吸附` 与 `M1 直出` 显式禁用，避免把 AUTO 静默当 SUPRA 用）；
  候选 **1/2/3**（带分数）、**整圈上移/下移 ±0.1**、**平滑强度**滑块（纠偏点锁定）、
  鼠标跟随截面气泡、当前视角打光 —— 后两项沿用既有实现。
  并列强候选时，前端要求先点一条候选才允许确认（本地 `picked` 门）。
- `frontend/src/App.jsx`：新增会话状态与 `marginDetect / marginSelectProposal / marginAddAnchor /
  marginAdjust / marginSmoothRing / marginConfirmSession / pushMagneticStroke`；
  3D 视图里 `onMarginCurveChange` 的磁性路径把"改动前后曲线差 > 0.05 mm 的点"当笔迹
  交给 `replace-segment`，实现"只替换动过的那段"。

## 4. 测试（`backend/tests/integration/test_margin_session_api.py`，10 条全绿）

1. 自动候选**不覆盖**正式 Margin（confirmed margin / revision 数不变）
2. 多候选歧义 → `REVIEW_REQUIRED`，且不带 `proposal_id` 的 confirm 被拒
3. 锚点作为硬约束被结果通过（距离锚点 < 容差），且锚点数 ≤4
4. 磁性修线**动不了对侧**（远侧点位移为 0）
5. 只有 `confirm` 之后才产生新修订号
6. 种植基台进不了天然牙自动检测（409 `UNSUPPORTED_INDICATION`）
7. 局部移动：要么健康闭环，要么如实拒绝且不改候选（并断言"非纯世界 Z"）
8. 平滑：纠偏点锁定不被移动
9. 会话候选不写 `margin_json`
10. 缺轴/未确认轴时拒绝建会话（409）

## 5. 本批明确未做（不越界）

- 不修改 Contact / Intaglio / Bite / Material / AI Crown 任何模块。
- 不动 M1 几何算法、不动 `_validated_margin_points` 规则、不动非平面 Margin 守卫。
- 会话存储是进程内的，多 worker 部署下需要粘性路由或外部存储（记为部署注意事项，不在本批范围）。
