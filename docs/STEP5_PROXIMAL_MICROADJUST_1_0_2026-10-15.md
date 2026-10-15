# Step 5 / Proximal 1.0 本轮源码改动说明

> 基于用户提供的当前工程快照继续修改；保留快照中原有全部改动。本文件只描述本轮新增/调整部分。

## 目标

1. 修正 Case1 FDI46 邻牙适配“距离达到目标但冠体被拉成长尖/长桥”的问题。
2. 将 Step 5 从“按钮存在但编辑失败难定位”提升为可用的工程微调工作台：局部预览、后端原子验证、失败回滚、失败面高亮、接触/厚度状态、撤销/重做。
3. 不降低 Margin、Finalize 或制造 QC；不重写已经跑通的轴向 coarse fit 与 occlusal fine adapt。

## Proximal coarse → fine

### 旧问题

旧 `proximal_coarse_fit` 与 Fine Adapt 本质上都重复调用同一个局部 `fit_stage`。`fit_stage` 适合最后几十分之一毫米的局部接触调整，但对于约 1 mm 以上间隙，会把窄 seed patch 向邻牙最近点拉伸，形成细长 spike/bridge。

### 新 coarse

`backend/app/services/proximal_coarse.py`

- coarse 使用完整 proximal wall 的连续权重，而不是窄 seed；
- cervical / Margin 受保护；
- occlusal top 权重降低，避免整体冠高和 cusp 被横向拖走；
- 主位移方向限制在已确认 PatientFrame 的 MD 轴；
- coarse 只推进到 Fine Adapt 可达范围（默认目标 0.75 mm），不直接追 0.05 mm；
- 每个 line-search candidate 同时检查：
  - 原有 `deformation_gate`；
  - 新增相对 shape-quality gate；
  - Margin/锁定区；
  - crown height；
  - antagonist；
  - opposite side。

`backend/app/geometry/crown_outer/auto_fit_adapter.py`

- PROXIMAL Fine Adapt 过渡带从固定 3 rings 改为可配置，默认 7 rings；
- Fine proximal displacement 主要沿 MD 轴，仅保留少量正交残差，降低多顶点向同一最近点收尖的问题。

`backend/app/geometry/crown_shape_quality.py`

新增相对形变质量审计：

- symmetric edge stretch p95 / max；
- triangle quality p05 / min；
- local triangle area ratio p05 / p95。

该 gate 是 Step-5 工程形变保护，不替代 Finalize 制造 QC。

## Step 5 微调

### 后端

`backend/app/geometry/sculpt.py`

- `SculptEditRejected`：结构化失败报告；
- 失败报告包含：操作、受影响顶点、最大/p95 位移、锁定区位移、严重翻面、新增/基线自交、退化面、失败 face ids、失败区域、建议动作；
- 每笔增加相对 shape gate，防止笔刷产生明显 spike/sliver；
- Margin/cervical 继续硬保护；
- candidate 破坏 watertight 时拒绝并回滚；
- 已支持局部 `drag`，单步位移后端限制 <= 1 mm。

`backend/app/services/sculpt_commit.py`

- 厚度 gate 失败也统一为结构化 `REJECTED_AND_ROLLED_BACK`；
- 仍保持“全部验证通过才发布新 outer revision”的原子事务。

`backend/app/services/sculpt_session.py`

新增：

- `edit_session`：当前 outer hash / frame revision / contact / thickness 状态；
- `capabilities`：当前可用工具及 remove 禁用原因；
- `baseline_geometry`：vertices/faces/watertight/nonmanifold/degenerate；
- thickness 状态区分 `FINAL_REFERENCE_AVAILABLE` / `REBUILD_REQUIRED`。

`backend/app/api/designer.py`

- Step-5 编辑 422 可返回机器可读 JSON detail，而不是只返回长字符串。

### 前端

`frontend/src/app/pages/DesignPage.jsx`

- 明确显示 `READY / PREVIEW / VALIDATING / SAVED / REJECTED / STALE`；
- 失败报告不再只放 toast；显示失败区域、翻面数、新增自交、最大位移、建议动作；
- Contact 显示测量状态、mesh source 和 vertex count；
- 厚度参考缺失时明确禁用 Remove；
- 笔刷过程中异步采样接触/厚度预览；
- 新增“局部拖动”工具（快捷键 D）；
- 保留 Add / Remove / Smooth / Flatten / Move / Rotate / Measure / Undo / Redo / Reset。

`frontend/src/components/SculptCanvas.jsx`

- rejected faces 以红色 overlay 高亮；
- 新增局部 drag 预览/提交；
- brush ring 与失败高亮保持独立。

`frontend/src/sculptPreview.js`

- 新增 `localDragPreview`；
- preview 仍然只是可丢弃预览，正式几何以后端求解与 gate 为准。

## 本轮测试

已在当前容器完成：

- `python -m compileall -q backend/app`：通过；
- `backend/tests/unit/geometry/test_crown_shape_quality.py`：3 passed；
- `frontend/tests/*.test.js`：全部通过；
- JSX/ES module 使用 Babel parser 做过语法解析：通过。

当前容器无法完成完整后端集成回归：原工程依赖 `mapbox-earcut` / `rtree`，依赖已经正确写入 `backend/requirements.txt`，但当前离线容器未安装且不能联网安装。

当前容器也不能执行 Linux Vite build：上传项目自带的 `frontend/node_modules` 是 Windows 平台二进制依赖；源码和 Node 纯 JS 测试已通过，实际开发机请删除/重装 `node_modules` 后执行 `npm ci && npm run build`。

## 1.0 边界

本轮没有实现：

- exocad 级完整 semantic Paint & Pull；
- cusp / ridge / fissure 独立语义编辑器；
- dynamic articulation；
- 新的 FFD 系统；
- Case2/3 support 根因修复；
- 制造级 perfect boolean healing。

这些不应阻塞 Step-5 工程粗稿编辑和 Case1 的 1.0 主流程。
