# WelooAI牙冠自动设计

2026-10-10：五步流程、图形化牙位选择、AutoFit 候选、内嵌冠体微调及 QC 色图已统一到常驻 3D 工作区。操作与验证见 [前端流程集成记录](docs/FRONTEND_WORKFLOW_2026-10-10.md)。工程导出仍未制造放行。

2026-10-10：传统几何 AutoFit 工程链路已接入，包含真实邻牙面区确认、静态位姿确认、专业牙库放置候选、顺序适配与联合重测、显式采纳及 W3 预构冠。使用方式、模板核查和未验收项见 [AutoFit 实现记录](docs/AUTOFIT_IMPLEMENTATION_2026-10-10.md)。真实病例仍受输入拓扑阻断，保持 ENGINEERING_ONLY。

**2026-10-10：1.0 人工辅助单冠工作区已接入。** 操作、快捷键、验证与未验收边界见 [1.0 使用说明](docs/WELOOAI_1_0.md)。入口 http://127.0.0.1:8010/ 。当前提供初稿、Margin/插入轴编辑、冠体变换与塑形、布尔及保存导出；不代表全自动或临床/加工验收完成。下文研发记录中的旧状态应结合此说明阅读。

基于 **Prep / Abutment 条件约束** 的自动牙冠设计研发工程。

W3 已接入真实三维 Margin 颈部重建、实测间隙与薄区审核：[实现和验收边界](docs/W3_NONPLANAR_MANUFACTURING_QC.md)。当前仍为工程测试版；真实病例输入探测尚未通过，不能生产放行。

> 这个仓库不内置任何假病例、假 STL、假 AI 权重或假质检结果。
> 所有病例、模板、AI checkpoint 都必须由你们自己导入真实数据后才能产生结果。

**开发方式已经切换**：从"按经验设计一个类似 exocad/3Shape 的系统"，
改为**以真实 exocad 病例参数与几何结果做回归驱动开发**。
完成标准是"与真实病例真值的定量误差下降"，不是"STL 成功生成"。

---

## 0. 文档索引

| 文档 | 内容 |
|---|---|
| `docs/PARAMETERS.md` | 参数系统：四态模型、数据驱动分类、固定优先级解析链、派生规则 |
| `docs/EXOCAD_IMPORTER.md` | exocad 文件格式、禁止正则的原因、`.matrix4` 角色判定、PHI 脱敏 |
| `docs/THREESHAPE_IMPORT.md` | **3Shape 订单包导入**：三态扫描判定、companion STL 三档绑定、验收闸门与面板 |
| `docs/TOOTH_LOCATION.md` | **目标牙定位 + Prep/Support 分割**：凹陷代价测地线、自动模式为何未实现、已知未解决项 |
| `docs/LOCALIZATION_QA.md` | **Localization 分类器人工 QA**：precision≈0.17、假阳模式、凸起判据提案 |
| `docs/INSERTION_AXIS.md` | 就位方向求解器：判据选择、刚性等变要求、FDI36 判据修正 |
| `docs/GT_MARGIN.md` | **真值颈缘提取器**：朝向场符号判据、环选择、以及两个被推翻的结论 |
| `docs/REVISIONS.md` | 设计修订链：不可变性、人工结果保护、候选消费语义 |
| `docs/ACCEPTANCE.md` | 验收套件：baseline.json、xfail 烧除、如何增删判据 |
| `docs/DATASET.md` | **证据数据集清单**：什么能干什么不能干（避免高估数据） |
| `docs/ALGORITHMS.md` | 逐算法实现 + 已修正/已知缺陷的状态标注 |
| `docs/ARCHITECTURE.md` | 分层架构 + 三个架构判断 |
| `docs/VALIDATION.md` | 三层验证体系与各自完成度 |
| `docs/DATA_AND_AI.md` | AI 训练规范（当前训练数据为 0） |
| `docs/牙冠自动生成系统调研与技术实现报告_中文版.docx` | exocad/3Shape 公开资料调研与技术路线 |

---

## 1. 现在真正实现了什么

### 1.1 证据数据导入 ✅

- `.constructionInfo` 解析（**XML 解析器，禁止正则** —— 自闭合标签会让参数错位配对）
- 四态参数模型：`EXPLICIT` / `EXPLICIT_BUT_UNRESOLVED` /
  `MISSING_INTERNAL_DEFAULT_UNKNOWN` / `NO_CONSTRAINT`
- 病例身份只认 `source_order_id`（实测同一订单出现在两个目录）
- `.matrix4` 角色**判定**（实测 15 个文件内容逐字节一致 → 固定坐标系变换，不是咬合配准）
- PHI 扫描与导入阶段脱敏（六层来源）
- `.dentalCAD` 无解析器；`.DCM` 全部 CE 加密无法解码（见 `docs/DATASET.md`）

### 1.2 参数系统 ✅

- 参数分类**由数据推导**：`LAB_GLOBAL_PRESET` / `RESTORATION_PRESET` /
  `LAB_PRESET_VARIANT` / `CASE_VARIABLE`
- 解析优先级固定：Tooth → Case/Order → Restoration Preset → Lab Preset → System Default
- **未知不跌落成 0**；语义未定的值（如 `CementGapThickness = 0`）阻断自动设计
- 派生规则：某些参数是另一个参数的确定性函数（实测相关度 100%）
- Lab Profile 由真实证据生成，无 PHI，可入 Git

### 1.3 病例与数据 ✅

- 新建 / 删除病例
- 上传真实上下颚 STL 或可解码的 3Shape HPS DCM（上传时转为 STL）
- 上传独立的 Prep / Abutment STL 或可解码的 DCM
- 无独立 Support 时可在颌模型上点中心点提取局部 ROI
- STL 网格校验：顶点、三角面、尺寸、连通块、watertight、体积

### 1.4 32 牙位模板库 ✅（导入接口）

单颗 FDI STL 导入；Dental Merlin / Universal 1–32 ZIP 自动识别并映射为 FDI 11–48。

**注意**：`32牙位基础解剖模板库/` 里的 `FDI_XX.stl` 是**带牙根的完整牙**，
不是牙冠外形模板（实测 `FDI_36.stl` 高 12.82 mm，真实冠高 6.87 mm）。
生成牙冠必须另导纯牙冠模板。详见 `docs/DATA_AND_AI.md` §6。

### 1.5 Crown Outer ✅

两条真实执行路径：

1. **Template + CAD** — 3 对对应点 Kabsch 刚体配准；可选 ≥4 对控制点的 RBF 非刚性形变
2. **AI 条件生成** — 只有检测到真实 TorchScript checkpoint 后才开放；**当前权重为 0**

### 1.6 Margin / 就位方向 ✅（就位方向已重写）

- 人工在 3D Support 上逐点绘制 Margin，**保存即落库并记入修订链**
- **真值颈缘提取器**（验收用）：从完工冠提取 intaglio ↔ exterior 真实交界缝，
  无阈值朝向场符号 + 包围判据；实测闭环/简单/自交 0/方位角覆盖 1.000/刚性等变
- 对开放网格支持从最大开放边界自动提取 Margin 候选
- Margin 健全性校验：点数 / 闭环性 / 平面性 / 采样均匀性
- **就位方向求解器**：Margin 稳健拟合平面法向 + 侧壁法向拟合轴 → 锥内搜索
  （粗 5°×12 + 细 1°）→ 面积加权倒凹 + Margin 可见率 + 偏角评分 → 状态机
- 状态 `OK` / `REVIEW_REQUIRED` / `FAILED`，含 `objective_is_flat`
  明示"该病例的轴未被目标函数定住"
- 支持人工修改与反转（走修订链，不允许只改数据库里一个向量）

### 1.7 Intaglio / Cement Gap / Blockout 🔴 已知缺陷

- Support 沿就位轴对齐；基于体素列做 undercut blockout
- 支持 `cement_gap_mm` / `margin_gap_mm` / `margin_band_mm` / `voxel_pitch_mm`
- 生成真实 `intaglio_cutter.stl`

**该 blockout 已被实测证明是错的**：它填的是整个上部基台的投影而非局部截面，
在边缘附近把内腔撑大 2–5 倍（实测边缘带中位间隙 0.523 mm，目标 0.015–0.05）。
待替换为沿就位方向的可见性 blockout + 分区 gap field。详见 `docs/ALGORITHMS.md` §4。

### 1.8 Final Crown ✅（有校验）

- 外冠按 Margin 参考平面裁切，**裁切后断言水密**
- 根据 cutter 对 Outer 做最小厚度外扩修正 🔴（该修正本身是已知缺陷，见 §1.9）
- Outer − Intaglio cutter 精确布尔（`manifold3d`）
- **布尔是有前置条件与后置校验的操作**：输入必须水密，输出必须通过体积、
  连通分量、包围盒等七项校验，不通过即抛异常、不保存结果
- 输出真实 `FDI_xx_final_crown.stl`

### 1.9 Minimum Thickness 🔴 已知缺陷

用 cutter 顶点 KD-tree 估计内外壳距离，低于阈值沿法线外推。

**两个问题**：顶点到顶点距离不是壁厚（实测 34% 的"过薄"顶点是误判）；
默认值 0.8 一刀切会把真实可加工冠的 **33.4%** 判为过薄
（exocad 真实值是分区的：主体 0.6、边缘区 0.2）。
后果是移动 37% 的顶点、最大 1.63 mm，改变邻接与咬合。
详见 `docs/ALGORITHMS.md` §6。

### 1.10 制造约束 🔶 部分完成

- `ManufacturingProfile`：3 轴 / 5 轴分离；刀具与机台参数属 ORDER 维度
- 候选刀具方向（3 轴 1 个 / 5 轴 19 个）、外内表面分离、单向可见性
- **完整可加工性判定未实现**，`machinable = None` —— 不发布会把合格设计
  判为不可加工的假闸门

### 1.11 QC 与修订链 ✅

- 所有结果来自真实几何计算，不内置"通过/失败"的假结论
- 设计修订链：不可变、自动提案不覆盖人工结果、审计能回答"为什么接受这个方向"
- 导出完整设计 ZIP：输入 STL、参数、QC、Outer、Cutter、Final、`revisions.json`

### 1.12 3Shape 订单包导入 ✅

操作员在 3Shape 里导出订单包 ZIP，上传后自动读出处方（牙位 / 修复体 / 材料 /
上下颌）与 CAD 参数，并跑验收闸门。详见 `docs/THREESHAPE_IMPORT.md`。

- 两步流程：**preview 零副作用**（不建病例、不写网格、不落库）→ **import 建病例**
- **闸门不过就 409，不建病例** —— "元数据能解析"不等于"能进设计管线"
- 扫描三态判定：实测真实包是 `SUPPORTED_BUT_ENCRYPTED`（格式受支持，顶点 CE 加密）
- **companion STL 三档绑定**（订单引用 / 同目录同主名 / 不绑定），**绝不位置绑定**
- 双 provenance：`original_geometry_source` 与 `preferred_geometry_source` 都留
- 建病例顺带写 `three_shape_import.json` 凭据与源包，导入**不**延伸出设计
- 前端有导入面板，逐行显示解析结果与补救建议

**当前限制（如实说）**：没有合法密钥就读不出 HPS 顶点，所以真实包目前
需要操作员从 3Shape 另外导出 STL 放进包里（同目录同主名）才能建病例。
**不提取、不绕过 3Shape 的 CE 密钥**。另外真实包没有任何牙位坐标信息
（`ModelTransformation`、`ModelBoundingBoxMin/Max` 实测为空），
因此不能自动定位预备体 —— `support` 仍需一次 ROI 取冠。

### 1.13 目标牙定位 + Prep/Support 分割 🟡 部分完成

上游只给牙位编号（FDI46），不给它在扫描里的 XYZ。这一层把编号变成网格区域，
再切成能交给 Axis / Margin 的 support。详见 `docs/TOOTH_LOCATION.md`。

- **凹陷代价测地线**：穿过凹脊（邻接沟/龈沟）的行程代价被放大 + 硬屏障，
  区域自然停在凹谷处 —— 取代 `spherical_roi` 按直线距离取面
- 凹/凸判据纯几何（法向 × 邻面形心位移），**刚性变换等变**，有测试
- 双模式：`user_seed`（操作员点一下）**已实现**；**自动模式未实现**，
  返回 `REVIEW_REQUIRED` + 实测理由（多起点 ICP 前 5 名残差只差 10%，判别度不足）
- support 返回全部可复核的验收数字；不达标就 `REVIEW_REQUIRED`

**未解决**：真实病例上 support 补片是 **8 条边界环**（理想 1 条），
所以状态仍是 `REVIEW_REQUIRED`；近远中接触面的精确分离做不到
（平滑鞍部，纯局部几何切不开）。

---

## 2. 重要边界

这个项目是**可运行的研发/算法工程实现**，不是已经注册获批的医疗器械，
也不应声称已经达到 exocad / 3Shape 的商业临床成熟度。

**验收套件当前状态：158 passed / 3 xfailed。** 那 3 项就是 §1.7 与 §1.9 的缺陷，
用 `xfail(strict=True)` 标记，修好后必须转正（见 `docs/ACCEPTANCE.md`）。
（3Shape 导入的闸门本身是 PASS 的，但真实病例因几何加密停在
`CASE_NOT_READY` —— 这是如实结果，不是测试失败。）

仍需要真实数据和大量工程验证才能达到生产使用要求的部分：

- 自动牙齿实例分割 + FDI 识别
- 从完整颌模型高精度自动分离 Prep / Abutment
- 商用品质 Margin AI（真值颈缘已可提取，见 `docs/GT_MARGIN.md`；患者侧 Margin 算法仍不可靠）
- 曲线 Margin 的亚毫米高精度裁切内核
- 专业动态咬合器
- 高精度邻接接触 patch 自动优化
- 经临床验证的最小厚度和材料规则库
- ✅ **真实临床/加工参数** —— 这一项已从"待填"变为"已有真实数据"
- 达到商业质量的 AI Crown 权重（**训练数据当前为 0**）
- 医疗器械质量体系、审计、权限、病例脱敏和法规验证

---

## 3. 项目结构

```text
CrownFlow_Complete/
├─ backend/
│  ├─ app/
│  │  ├─ importers/exocad/   exocad 证据数据导入（四态模型、矩阵角色、PHI 脱敏）
│  │  ├─ parameters/         参数分类 / Lab Profile / 固定优先级解析链 / 派生规则
│  │  ├─ revisions/          不可变设计修订链
│  │  ├─ manufacturing/      加工档案（3轴/5轴）与可达性构件
│  │  ├─ geometry/
│  │  │  ├─ insertion.py     就位方向求解器（锥内搜索 + 面积加权倒凹）
│  │  │  ├─ axis.py          旧签名兼容入口
│  │  │  ├─ registration.py  Kabsch + RBF
│  │  │  ├─ margin.py        Margin 边界 / 平面拟合
│  │  │  ├─ outer.py         Margin 裁切（水密断言）+ 最小厚度修正
│  │  │  ├─ intaglio.py      Cement gap + blockout
│  │  │  ├─ boolean.py       精确布尔 + fail-loud 校验
│  │  │  └─ qc.py            几何 QC
│  │  ├─ services/           pipeline / ai_runtime
│  │  ├─ api/routes.py       FastAPI 接口
│  │  ├─ db.py               SQLite（含轻量迁移）
│  │  └─ models.py           Pydantic 请求模型
│  ├─ tests/
│  │  ├─ integration/        修订链不变量
│  │  └─ acceptance/         parameter / geometry / baselines
│  └─ requirements.txt       全部钉版本
├─ frontend/                 React + Three.js 3D 工作站（Crown Design 6 步工作流）
│  ├─ src/domain/            流程模型 steps.js + 唯一后端接缝 CrownDesignAdapter
│  ├─ src/app/               外壳 + 6 个页面（首页 / 病例信息 / Margin·Axis / 参数·模板 / 牙冠设计 / 检查·导出）
│  ├─ src/components/        复用能力：Viewer / ViewTools / MarginCurveEditor / SculptCanvas
│  └─ tests/                 前端流程模型与纯几何函数单测
├─ ml/                       数据集构建 / 模型 / 训练 / TorchScript 导出
├─ docs/                     见 §0 文档索引
├─ data/                     初始为空
├─ docker-compose.yml
└─ run_dev.ps1
```

---

## 4. Windows 本地启动

要求：Python 3.11+；首次构建前端需要 Node.js 20+。

```powershell
python -m pip install -r backend/requirements.txt
Set-ExecutionPolicy -Scope Process Bypass
.\run_dev.ps1
```

打开：

- 工作站：http://localhost:3277
- FastAPI：http://localhost:3277/docs

`run_dev.ps1` 在同一个 3277 端口运行前后端；首次运行若缺少前端构建，
会自动执行 `npm install` 和 `npm run build`。

### 依赖说明（重要）

`manifold3d`、`mapbox-earcut`、`rtree` 是**功能依赖，不是可选依赖**：

| 缺失 | 后果 |
|---|---|
| `manifold3d` | 只剩体素降级，它**未做任何裁减**也会把牙冠放大 10%、引入 0.089 mm 中位误差 |
| `mapbox-earcut` | `slice_mesh_plane(cap=True)` 抛异常 → 牙冠以非水密状态进入布尔 → **实测产出真值 13% 体积的碎冠** |

因此 `requirements.txt` 里全部钉版本。体素降级路径改为**显式 opt-in 的调试路径**
（`allow_voxel_fallback=True`），默认关闭。

`torch` 是唯一真正的可选依赖（只有启用 AI Generator 时需要），
在 `backend/requirements-optional.txt`。

---

## 5. Docker

```bash
docker compose up --build
```

---

## 6. 推荐真实使用顺序

0. **3Shape 用户**：上传订单包 ZIP（`00 导入 3Shape 订单包`）→ 看面板确认解析结果。
   若几何显示"已加密 / 无合法密钥"，先在 3Shape 里把扫描导出为 STL、
   与原扫描同目录同主名放进包里再上传（如 `Scans/Lower/PreparationScan.stl`）。
   导入成功即得到病例与上下颌几何、处方记录；`support` 仍需下一步的 ROI 取冠。
1. 新建病例
2. 上传已完成咬合配准的 Upper / Lower STL
3. 上传 Prep/Abutment STL；没有则从患者颌模型提取 Support ROI
4. 导入你们合法拥有授权的纯牙冠外形模板（**不要用带牙根的解剖库**）
5. 选择目标 FDI 和 Support 类型
6. 创建 Design
7. 标注 Margin 并在 3D 中确认 → **保存（立即落库）**
8. 求解就位方向 → **检查状态**：非 `OK` 时必须人工确认
9. Template CAD 模式：选 3 对刚体对应点，可选增加 RBF 控制点
10. 生成 Crown Outer
11. 设置参数（**应由 Lab Profile 解析，不要手填**）
12. 生成 Intaglio cutter + Final Crown
13. 查看真实 QC（并为已知缺陷留出余量）
14. 导出 Final STL 或完整设计 ZIP

---

## 7. 模板预处理

### Template Studio：从完整牙制作待审牙冠模板

```powershell
python ml/template_studio.py --source "D:\code\牙冠自动生成\32牙位基础解剖模板库\32位牙列-分" --out data/template_studio/candidates
python -m unittest ml.test_template_studio
```

每颗牙输出 `crown_template.stl`、`cervical_boundary.json`、
`local_frame.json`、`semantic_regions.json`、`landmarks.json` 和 `review.json`。
`manifest.json` 汇总 32 颗结果。需要调整某颗牙的候选平面时用
`--fdi 46 --cut-z -8.0` 重算该牙。

**所有输出保持 `REVIEW_REQUIRED`，不得直接导入 `data/templates`。** 原始
Dental Merlin STL 没有标注 CEJ；当前中段截面只提供可复核的切割候选，不是
已识别的真实牙颈线。颊舌/近远中区域与极值点也是几何候选，需设计师确认。
完整 32 牙位侧视检查图在 `data/template_studio/qa_all_32.png`。

---

## 8. 运行测试

```bash
cd backend
python -m pytest tests -q
```

验收套件需要仓库外的真实病例数据，通过环境变量指定：

```bash
export CROWNFLOW_FDI36_DIR="D:/code/牙冠自动生成/数据/1629-王煜-STL"
export CROWNFLOW_EXOCAD_CASES_DIR="D:/code/牙冠自动生成/数据/15个完整病例"
export CROWNFLOW_THREESHAPE_FIXTURE_DIR="D:/code/牙冠自动生成/数据/测试_胡昭"
```

找不到数据时会**带理由 skip**（不是静默通过）。详见 `docs/ACCEPTANCE.md`。

测试分两层，3Shape 导入尤其如此：

- `tests/unit/` —— 合成 fixture，**可提交 Git**，锁 parser 与 API 行为
- `tests/acceptance/` —— 真实病例，运行时在仓库外查找，不进 Git

刷新量化基线：

```bash
python -m tests.acceptance.baselines.regenerate --write
```

---

## 8. AI 训练

项目没有任何假 AI 权重，`ml/checkpoints/` 为空。

**当前训练数据为 0 例单冠** —— 15 个完整病例全是多单位（2–14 单位），
且最终冠 STL 含内表面、不能作为外表面标签。详见 `docs/DATASET.md` §3.4。

真实训练数据目录约定：

```text
raw/
  case_x/
    upper.stl  lower.stl  support.stl
    crown_outer_gt.stl     # 技师确认的外表面，不能含内面
    meta.json              # {"fdi": 46, "material": "...", "source_order_id": "..."}
```

流程见 `ml/README.md` 与 `docs/DATA_AND_AI.md`。

---

## 9. 数据合规

**患者数据不在本仓库范围内**：git 只初始化在 `CrownFlow_Complete/` 这一层，
上级的 `数据/` 与 `32牙位基础解剖模板库/` 在仓库之外；
`.gitignore` 另加一道防线（`数据/`、`*.DCM`、`*.dentalCAD`、`*deid_mapping*` 等）。

PHI 分布六层（目录名 / STL 名 / `.matrix4` 名 / `.3ox` 字段 / Notes / `import.log`），
脱敏工具已就绪但**尚未实际执行**。见 `docs/DATASET.md` §5。

---

## 10. 第三方牙库授权

Dental Merlin / 3PX、exocad、3Shape 等第三方牙库资产不属于本仓库。

本项目只实现导入接口；你们用于商业软件前必须自行确认牙库、历史病例
及训练数据的授权范围。
