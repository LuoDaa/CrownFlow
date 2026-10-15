# CrownFlow 1.0：已有实体基台上的粘接冠

日期：2026-10-13。状态：工程实现与审核版本；未提交代码，未重启现有 8010 服务。独立前端审核构建位于 `output/abutment-review/frontend`。

## 1. 实现范围

支持 `CROWN_ON_EXISTING_ABUTMENT`：从已确认的实体基台、冠终止线和插入轴提取支撑面，复用现有内冠链生成闭合牙冠。`CUSTOM_IMPLANT_ABUTMENT` 与 `SCREW_RETAINED_CROWN` 明确拒绝，不能借普通冠路径绕过。

基台 Margin 的语义是 **CROWN_FINISH_LINE（基台冠终止线）**。它不是种植体底部接口，也不是穿龈轮廓边界。肩台可成为候选；用户修正并确认的 Margin 才是生成依据。边界编辑复用现有候选、Top-K、锚点重描、磁性局部编辑、自由编辑和上下调整，不重写编辑器。

## 2. 使用流程

1. 导入扫描，选择目标 FDI 和已有基台修复类型；确认组件确为实际修复基台。扫描杆不能仅凭外形自动当作修复基台。
2. 在“基台冠终止线”页面选择候选或人工修正，检查整圈后确认。
3. 确认插入轴和 PatientFrame，设置间隙、厚度及冠形参数。
4. 进入生成页面的“检查基台支撑 / 确认螺丝孔”。绿色表示冠终止线，橙色表示待核对的顶部螺丝孔，红色表示未知孔。
5. 无额外孔时直接评估支撑；有螺丝孔时逐个人工确认。系统只在派生面上建立虚拟盖片，原始基台保持不变。未知孔不能继续。
6. 支撑就绪后生成和审核外冠，按需微调；再生成内冠、布尔差集、共享颈缘接缝，检查 QC 并导出 STL。

![螺丝孔复核界面](../output/abutment-review/screw-channel-review.png)

![确认后支撑就绪](../output/abutment-review/support-ready.png)

## 3. 几何实现与数据契约

链路：`PreparedAbutmentSupport → directional blockout → variable gap → CavitySolid → Outer − Cavity → 3D margin seam → closed crown → QC`。

提取器复用 MarginCurtain 切分，将确认曲线投影到目标网格边界并嵌入三角形；依据 Margin 的源面引用定位目标组件，再沿切分后的 face adjacency 提取冠终止线以上的连通支撑。可选 seed 用于验证目标组件。不以最大组件代替目标识别。排除线以下接口和无关组件，歧义直接拒绝。

当前切分适用范围是 **沿确认插入轴可单值表达的冠终止线**。边界嵌入允许的最大投影距离为 0.05 mm，属于工程容差；不改变用户确认的曲线，不支持任意内在曲面的复杂回折边界或自动牙龈分割。

输出沿用 PreparedSurfaceContract，补充以下数据：

| 数据 | 内容 |
|---|---|
| 身份与依据 | indication、support_type、FDI、Margin/Axis/PatientFrame revision |
| 来源 | source_abutment_sha256、conditioned_support_sha256 |
| 面映射 | source_face_ids、source→output mapping；虚拟面源标记为 -1 |
| 边界环 | CROWN_MARGIN、SCREW_CHANNEL_CANDIDATE、UNKNOWN_HOLE |
| 虚拟盖片 | loop id、源面、盖片面、面积、VIRTUAL_SCREW_CHANNEL_CAP、前后 SHA256 |
| 生成与审计 | intaglio 参数、gap profile、算法版本、派生 STL 路径及哈希 |

螺丝孔候选依照闭环位置、轴向、顶部高度和开口尺度筛选，必须人工确认。盖片仅参与支撑构造、blockout 和 cavity；actual-gap 测量使用同一份派生支撑的真实面，排除盖片区域并单独输出面积和排除采样数。不会重新对完整扫描测距。

Margin、Axis、PatientFrame 或支撑文件变化后，确认依据和后续结果失效。最终生成再次核对源文件及实际消费支撑的 SHA256。

## 4. 状态与拒绝

| 状态 | 含义 |
|---|---|
| ABUTMENT_SUPPORT_READY | 单一连通支撑、唯一确认冠边界、没有未解决孔、满足流形条件 |
| REVIEW_REQUIRED | 输入缺失、待确认孔或几何不合法，需要修正 |
| NO_SUPPORT_ASSET | 缺少可读取的支撑资产 |
| UNSUPPORTED_INDICATION | 自定义基台或螺丝固位冠不在范围内 |

拒绝码包括 `MARGIN_NOT_CONFIRMED`、`MARGIN_NOT_ON_SUPPORT`、`MULTIPLE_SUPPORT_COMPONENTS`、`SCREW_CHANNEL_CONFIRMATION_REQUIRED`、`UNKNOWN_SUPPORT_HOLE`、`NONMANIFOLD_SUPPORT`、`SUPPORT_EXTRACTION_AMBIGUOUS`、`CUSTOM_ABUTMENT_NOT_SUPPORTED`。没有静默填未知孔或自动选择歧义组件。

## 5. 真实病例 E2E

病例：`1.0测试数据/基台测试1-46`。经导入、Margin 确认、支撑提取、轴与局部坐标确认、牙库外冠、内冠、布尔、接缝、QC、STL 重读和导出哈希核验。

**组件身份是工程测试声明，未以厂商工程元数据证明它是最终修复基台。** 此例没有额外开放螺丝孔环；虚拟盖片另由合成孔病例验证。

此次外冠显式使用径向 1.15 倍微调，颈部过渡参数 `margin_band_mm=2.4`；默认外冠包络不足时没有自动掩盖失败。

| 指标 | 结果 |
|---|---|
| 支撑状态 | ABUTMENT_SUPPORT_READY |
| 原始数据字节 | 哈希前后不变 |
| 最终 STL 重读 | 闭合，water-tight |
| 共享颈缘接缝 | 边关联次数为 2，拓扑测量通过 |
| 实际 gap 采样 | 24,882 |
| gap p05 / 中位 / p95 | 0.03038 / 0.06057 / 0.06120 mm |
| gap 最小 / 最大 | 约 0 / 0.11961 mm；颈缘附近零间隙需结合区域检查 |
| 薄壁采样面积 | **34.60 mm²**，配置厚度 0.8 mm；尚需修正 |
| 连续就位认证 | false，尚未实现连续路径证明 |
| 生产状态 | ENGINEERING_ONLY / NOT_PRODUCTION_APPROVED |

交付证据：`output/abutment-review/real-test1-46/result.json`、`manufacturing_qc.json`、`prepared_support.stl`、`final.stl`。**全链成功不等于该牙冠已达到生产质量。**

## 6. 验证

- 新基台套件 10 项与 PatientFrame 7 项：17 passed。覆盖干净支撑、待确认螺丝孔、虚拟盖片、未知孔、错误 Margin、目标非最大组件、候选不是真值、原始字节、盖片统计排除、来源哈希、自定义基台拒绝及真实 E2E。
- 天然牙及原流程回归：此前组合运行 32 passed / 1 failed；已修复天然牙接缝审计应继续消费原始扫描的回归，失败项单独复跑通过。新增边界语义修改后，Margin session API 与支撑契约针对性回归 **26 passed**。
- 前端单元检查 9 passed；独立审核目录 Vite 构建通过。
- 浏览器验证了螺丝孔人工确认到支撑就绪。长时间浏览出现一次 LineMaterial shader error / WebGL context lost；不能据此声称长时间 3D 稳定性已通过。

## 7. 修改文件清单

以下仅列本批涉及项；仓库已有其他未提交改动，不作为本批新增成果。

**新增**：

- `backend/app/geometry/abutment_support.py`：提取、孔分类、虚拟盖片、面映射。
- `backend/app/services/abutment_support.py`：契约、来源校验、确认绑定。
- `backend/tests/integration/test_existing_abutment.py`：合成/API/真实链路验证。
- `docs/EXISTING_ABUTMENT_CROWN_1_0.md`：本说明。

**数据及编排**：`backend/app/models.py`、`db.py`、`revisions/models.py`、`services/prep_conditioning.py`、`design_dependencies.py`、`generation_workflow.py`、`auto_fit.py`、`pipeline.py`。

**几何/QC**：`backend/app/geometry/nonplanar_crown.py`、`manufacturing_qc.py`、`registration.py`。复用原内冠算法，仅补派生支撑接入、QC 真实面排除、模板颈缘参考定位及 cavity 人工下封面延伸；不改变实际 gap profile。

**API**：`backend/app/api/routes.py`、`designer.py`、`auto_fit.py`、`margin.py`。新增 `POST /designs/{id}/abutment-support/confirm`，读取复用 prep-contract；保留正在并行修改的 Margin session 语义。

**前端**：`frontend/src/api.js`、`components/AutoFitPanel.jsx`、`components/PrepMarginAxisStep.jsx`。复用原任务分页和 Margin 编辑器。

**回归与 fixture**：`backend/tests/integration/conftest.py`、`test_margin_session_api.py`、`tests/unit/services/test_prep_conditioning.py`。隔离测试数据路径，避免导入时缓存路径污染真实数据。

## 8. 未支持的边界

不包含自定义基台设计、种植库接口设计、穿龈轮廓、螺丝固位冠、AI 冠、咬合配准、新 offset/gap/Boolean 或铣刀补偿重构。闭合网格中的内部孔洞没有额外 boundary loop 时，不以此分类器声明识别出螺丝通道。

孔候选、基台身份、冠终止线都需人工核对；没有厂商身份真值或连续就位证明。当前真实示例仍有薄壁，不能直接用于生产。尚未做全仓库回归、全部真实病例批量验证或长时间 GPU 稳定性验收。
