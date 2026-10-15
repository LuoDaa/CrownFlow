# 边缘线步骤：接 M1 自动检测 + 凸/凹特征吸附 + 剖面气泡（2026-10-12）

对照对象：exocad 官方 `Detect / edit margins`（2023+ 版）与 `Matching the scan abutments`（2025+ 版）。
本次只做"能照搬的东西"——**全部落在工作台层，不改任何冻结基线**
（`auto_margin` 的 ded3ec3 配置、`CrownOuter J.2` e3503fe、`intaglio` 的 A/B 两段语义都未改动，
M1 只是被**调用**）。

## 一、本次新增

### 1. 工作台接上 M1 自动边缘检测（此前只是"截面启发式"）
- 新服务函数 `app/services/pipeline.py::auto_margin_m1(support_path, axis=None, mode, radius_mm, apply_snap)`
  → 调 `geometry/auto_margin/engine.predict_margin`（**冻结配置 v2**：faces/K24 + 环状 DP + fail-loud），
  引擎状态非 `AUTO_ACCEPTED` 时**不产出点**，并给出 `reason`。
- 轴来源：`axis=None` 时用 `suggest_insertion_axis(mesh, None)`（侧壁拟合临时轴），
  返回里标 `axis_source='provisional_sidewall_fit'` —— 边缘线步骤本来就在就位方向之前，
  与 exocad 顺序一致；**不冒充已确认轴**。
- 新接口 `POST /api/designs/{id}/auto-margin-m1`
  → 复用 `record_auto_margin_candidate`：**不覆盖人工边缘**（人工结果存在时只记候选修订，
  返回 `requires_decision`），算法版本写 `welooai/auto_margin_m1v2/<config_id>`。

### 2. 凸/凹"磁力"特征吸附（exocad Supragingival / Subgingival）
- 新模块 `app/geometry/margin_feature_snap.py`：
  - `face_concavity_deg(tris, ref_point)`：**有符号二面角（度）**，>0 凹（沟）/ <0 凸（棱）；
    用 `ref_point`（边缘线质心）做**绕向无关**的法向修正 —— 输入网格法向朝内/朝外不会把判据翻过来。
  - `snap_to_feature(...)`：每个边缘点在 `radius_mm` 内挑最强的同号特征面 → 把点贴到该面最近位置
    → 沿闭环平滑一次 → 再贴回网格。特征是"**面**"级别，粗网格上位移被面尺度限制，
    所以返回 `mean_move_mm / max_move_mm` 供调用方判断。
  - **不猜**：找不到特征面的点原样保留；特征面占比 <60% 时返回 `REVIEW_REQUIRED` 并给出原因。
- 新接口 `POST /api/designs/{id}/margin-snap`（`mode=SUPRA|SUB`）。

### 3. 悬停剖面气泡 + 从视角打光
- 新接口 `POST /api/designs/{id}/section-preview`：过给定表面点、**包含就位轴**的一张剖面，
  返回 3D 折线 + 2D 坐标（`u=就位轴`，`v=径向`）+ 平面/基向量；纯几何、无状态。
- 前端：边缘线面板加两个开关
  - **悬停剖面气泡**：鼠标在支撑体上移动（150 ms 节流）→ 调上面的接口 → 面板里画一张小剖面图
    （竖轴=就位方向，横轴=径向），帮助找正确的颈缘位置；
  - **从视角打光**：按下的那一刻把光源方向**定格在当前视角**，之后转到侧面看台肩时阴影显出颈缘
    （exocad `Adjust Light from view` 的等效做法）。

### 4. 前端交互
- `PrepMarginAxisStep.jsx`：新增 `龈上（凸）/ 龈下（凹）` 模式切换、`自动检测（M1）`、`按模式吸附`
  三个按钮 + 一行工具状态（显示 M1 状态/配置指纹/轴来源/吸附位移）。
- `api.js`：`autoMarginM1 / marginSnap / sectionPreview`。
- `Viewer.jsx`：支撑体 `onProbe(point, normal)` 悬停探测（节流）+ `headlight` 定向光。

## 二、与 exocad 的对照（本轮后）

| exocad 功能 | 状态 |
|---|---|
| 悬停 2D 剖面气泡 | ✅ 本轮实现（接口+面板小窗） |
| 单击启动自动检测 | ✅ 已有（M1 按钮；此前是 −3 mm 截面启发式） |
| 追加点引导重跑 / 删点 | ⏳ 未做（现有点选增删点，但"补点后重跑自动检测"未接） |
| Adjust Light from view | ✅ 本轮实现（从视角打光） |
| Subgingival / Supragingival | ✅ 本轮实现（SUPRA 凸 / SUB 凹） |
| Draw Free / Draw magnetic / Edit | ⚠️ 部分（已有曲面吸附手绘、拖点、局部重画；**"磁性画线吸最大曲率"未做**） |
| Move margin up/down | ✅ 已有（整圈 ±0.1 mm） |
| Smooth margin | ✅ 已有（平滑折线）；exocad 的"补偿扫描误差"部分未做 |
| Repair/Draw（扫面修复 + 4 参数） | ❌ 未做（几何要自研） |
| Auto-hide adjacents | ⚠️ 部分（有图层显隐，无"自动隐藏邻牙代型"） |
| 多步 Undo/Redo + 热键 | ⚠️ 部分（有撤销/重做；exocad 的 1/2/3/4、+/−、CTRL+X 未接） |
| Imported margins（导入边缘线） | ❌ 未做（**建议下一批做**：exocad/3Shape 病例的 `.modelInfo/.constructionInfo` 本来就带 `<Margin>`） |
| 种植病例的**扫描杆↔库匹配**定义终止线 | ❌ 未做（需要种植库数据 + 匹配算法；SIC 那套可直接起步） |

## 三、验证

- 单元（`tests/unit/geometry/test_margin_feature_snap.py`，6 passed）：
  球面全凸（中位 −2.8°，低于 8° 特征阈值 ⇒ 不吸附）；环面内圈 +11.2°/外圈 −11.2°；
  台阶代型外缘棱 −90°、内角 +90° 且**绕向翻转后符号不变**；
  细分到 0.15 mm 面后 **SUPRA 把点从 r=2.20 吸到 2.88（外缘棱）、SUB 停在 2.19（凹角）**。
- 集成（`tests/integration/test_margin_tools_api.py`，3 passed）：剖面接口返回对应的 3D/2D 折线；
  `margin-snap` 两模式行为不同；`auto-margin-m1` **fail-loud 契约**成立
  （无闭环时 `points` 为空且 `status != AUTO_ACCEPTED`，并如实标 `provisional_sidewall_fit`）。
- 前端：`npm run build` 通过（主包 1.25 MB 体积警告为既有项）；`node --test tests/*.test.*` 9 passed。
- 后端全量回归：**510 passed / 4 failed / 10 skipped / 3 xfailed**（684 s）。4 个失败**均非本批引入**：
  - `tests/acceptance/geometry/test_fdi36_golden.py` 3 例：工作树里**未提交**的 `geometry/outer.py`
    非平面守卫（`margin_seam.py` 为新增未跟踪文件）在真实病例 Margin 平面偏离 0.043 mm 时按设计
    抛 `MARGIN_RECONSTRUCTION_REQUIRED`，而该 golden 用例仍走旧的平面链路；这两个文件与本批改动无关
    （`git status`：`outer.py` 已被他人修改、测试文件未改）。
  - `tests/unit/geometry/test_outer_v2_j1.py::test_no_violation_at_all_is_noop`：全量跑时的
    内存/负载偶发（`ArrayMemoryError`），**单跑 7 passed**。

## 四、下一批（按价值排序）

1. **Import 边缘线**：导入 exocad/3Shape 病例自带 `<Margin>` 作为候选/真值（数据已在手，51/51 例有）。
2. **磁性画线**：在现有手绘基础上，把新增线段吸附到"最大曲率/最尖角"（exocad Draw magnetic）。
3. **补点重跑**：追加 1–4 个点后以这些点为约束重跑 M1（exocad 的"点几个点帮算法"）。
4. **种植库匹配**：`Detect Implant Position` 式库搜索 + 单点配对 + 质量色图 → 终止线由库件定义。
5. 热键与 Auto-hide adjacents；Repair/Draw 的扫面修复（几何自研，放最后）。
