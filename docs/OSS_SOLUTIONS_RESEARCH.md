# 开源方案调研报告 —— 对应"手动设计牙冠 8 步流程"的缺口

调研与实测日期：2026-10-10
调研环境：`ssh luoda@192.168.110.85`，工作目录 `/data/luoda/oss_crown_research/`（独立 venv，
未触碰他人目录与 docker 容器）
被测病例：`cf_b1b73bae221211b1`（FDI 46，含 GT 设计冠 / 代型 / 双侧邻牙 patch / 对颌 patch）
纪律：本次只调研与实测，**未修改 CrownFlow 任何代码**。

---

## 0. 摘要（TL;DR）

| 8 步缺口 | 可用性判定 | 结论要点 |
|---|---|---|
| 1 数据导入 | 半个可用 | hpsdecode（MIT）实测可识别我们的 HPS DCM，但文件是 **CE 加密**，无合法密钥不可解；CA/CC 可直接解 |
| 2 牙位自动定位 | 有真货，卡在权重 | TGNet / 3DTeethSAM 均为真论文真代码，但**权重都在 Google Drive**（服务器不可达），且两个仓库**都没有 LICENSE** |
| 3 真实颈缘 | 无开源可用 | 唯一相关项目（TrimLine）预处理/数据/权重全部 "Coming soon"；维护自研 Auto Margin 线 |
| 4 插入轴 | 不需要 | 现有求解器已覆盖，开源无增量 |
| 5 AI 外冠生成 | 最接近可用但**实测不够格** | ToothCraft 权重公开且已真跑通：恒等精度天花板 0.65mm，补洞 0.89mm/p95 1.30mm，牙桩→整冠 1.09mm/p95 2.14mm；且无 LICENSE |
| 6 内冠 Blockout | 只能参考思路 | C.B.GEN（MIT）代码仅 117 行且方向硬编码、非就位轴 blockout；odc_public 有倒凹填平但无 LICENSE |
| 7 接触咬合 | 开源不解决 | 关键仍是先修输入（邻牙 patch 从源牙弓重新提取） |
| 8 技师微调 | 有 MIT 原型可借鉴 | 251123_MeshSculpter（MIT，three.js 笔刷/撤销/OBJ 导出），与现前端同技术栈 |

**最重要的合规结论**：本批最相关的 AI 仓库（ToothCraft、TGNet、3DTeethSAM、CrownGen、DMC）
**全部没有 LICENSE 文件**。按版权默认规则等于"保留所有权利"，公开可见不等于可以商用；
若要用于产品，必须先向作者取得书面授权（包括我们已下载的 ToothCraft 权重）。

---

## 1. 调研方法（可复核）

1. 逐仓库克隆（GitHub 直连不通，经 `https://gh-proxy.com/https://github.com/...` 代理；
   一个仓库经 `gitclone.com` 兜底），得到源码本体而非 README 转述。
2. 逐个检查：是否存在、LICENSE 文件（含 README 内许可声明）、依赖清单、权重引用
   （grep `checkpoint/weights/.pth/drive.google/huggingface`）、按步骤核对输入输出格式。
3. 对权重可达的模型（ToothCraft）**在真实病例上实跑推理**并量化，而不是止步于"论文说"。
4. 实测纪律：先验 operand（SDF 符号/尺度/坐标系）再谈模型问题；每个疑似模型缺陷先做
   对照实验（恒等对照、尺度对照、层级扫描）。

### 服务器网络事实（决定了可行路径）

| 目标 | 可达性 | 用途 |
|---|---|---|
| github.com / raw.githubusercontent.com | ❌ 超时 | —— |
| huggingface.co / Google Drive | ❌ 超时 | TGNet/3DTeethSAM 权重不可直接取 |
| `gh-proxy.com` 前缀 | ✅ | 克隆 GitHub 仓库 |
| `hf-mirror.com`（HF_ENDPOINT） | ✅ | 下载 HF 权重（ToothCraft 实测成功） |
| pypi.org / download.pytorch.org / 清华镜像 | ✅ | 依赖安装 |
| gitee.com / gitclone.com | ✅ | 备用 |

GPU：RTX 5090 32GB（空闲），torch 2.11.0+cu128 实测可用（sm_120 正常）。

---

## 2. 候选仓库全表（全部经克隆核实）

| 仓库 | 对应步骤 | LICENSE | 权重 | 可跑性 | 判词 |
|---|---|---|---|---|---|
| [hpsdecode](https://github.com/HeadTriXz/hpsdecode) | 1 导入 | MIT | 无需 | ✅ 已装（PyPI 1.0.0） | 可用（需合法密钥解 CE） |
| [Open3SDCM](https://github.com/Nosenzor/Open3SDCM) | 1 导入 | BSL-1.0 | 无需 | 未构建（C++20/vcpkg） | 备选适配器 |
| [ToothGroupNetwork](https://github.com/limhoyeon/ToothGroupNetwork) | 2 定位 | **无** | Drive（挑战赛 ckpts） | 需移植（pointops CUDA 扩展，官方只测到 torch 1.7） | 代码真、权重不可达 |
| [3DTeethSAM](https://github.com/Crisitofy/3DTeethSAM) | 2 定位 | **无** | Drive（含 SAM2 权重） | 重（pytorch3d 源码编译 + SAM2） | 新的对照路线 |
| [CrownGen](https://github.com/baejustin/CrownGen) | 5 外形 | **无** | 未发布 | 依赖重（pytorch3d、torch-scatter、libigl） | 代码+样例数据可研究 |
| [VISAPP_ToothCraft](https://github.com/ikarus1211/VISAPP_ToothCraft) | 5 外形 | **无** | ✅ HF `DejvaX/ToothCraft`（镜像可下） | ✅ 已实跑（3 处小坑） | 唯一跑通的 AI 模型；精度不够格 |
| [DMC](https://github.com/Golriz-code/DMC) | 5 外形 | **无** | 无牙科权重 | 依赖极旧（open3d==0.9） | readme.txt 为空，不可用 |
| [AI-3D-Dental-Crown-Generation](https://github.com/stanveer/AI-3D-Dental-Crown-Generation) | 5 外形 | MIT | **无**（checkpoints/ 不存在） | —— | 不可用（README 自称有 17M 权重，仓库里没有） |
| [DCrownFormer](https://github.com/suyang93/DCrownFormer) | 5 外形 | 无 | —— | —— | 仓库只剩一份 readme：算法 2024-09 已转让，代码不再公开 |
| [C.B.GEN](https://github.com/ImaneChafi/C.B.GEN) | 6 内冠 | MIT | 无需（纯几何） | 可跑（pyvista/pymeshlab） | 只可借鉴结构，非就位轴 blockout |
| [odc_public](https://github.com/patmo141/odc_public) | 6 内冠 | **无** | 无需 | Blender 2.8 插件 | 倒凹检查/填平可读，不可抄代码 |
| [Dental-CAD-Designer](https://github.com/ZoliQua/Dental-CAD-Designer) | 3–8 全流程 | **无** | 无需 | TS/Node | 架构与 QC 门设计参考（仅 fixture 验收） |
| [Automatic-Trim-Line-Detection](https://github.com/supergengchen/Automatic-Trim-Line-Detection) | 3 颈缘 | **无** | 未发布 | 不可跑 | 数据/预处理 4 处 "Coming soon!" |
| [251123_MeshSculpter](https://github.com/ekimroyrp/251123_MeshSculpter) | 8 微调 | MIT | 无需 | ✅（three.js） | 可直接借鉴的前端雕刻原型 |
| [TSegFormer](https://github.com/huiminxiong/TSegFormer) | 2 定位 | **MIT** | **未发布**（README 只教自行训练） | 需重训（原环境 py3.7/torch1.9） | 授权路径最清晰的分割备选；无权重、无公开数据集 ⇒ 只能"自训"路线 |
| [zju-ai4h/Unsupervised-Pretraining-for-tooth-segmentation](https://github.com/zju-ai4h/Unsupervised-Pretraining-for-tooth-segmentation) | 2 定位 | **无** | 有调试权重 `finetune/pretrained_weights/checkpoint.pth`（README 注明 debug only） | 依赖 MinkowskiEngine 0.5.4 / open3d 0.9（现代 CUDA 难装） | 研究参考，不作推理候选 |
| [dental-guide-mesh-poc](https://github.com/Sandyyy123/dental-guide-mesh-poc) | 6 内冠 | 专有（All Rights Reserved） | —— | 禁止使用 | 连运行都不允许，只能当公开说明读 |

---

## 3. 分组详述

### 3.1 第 1 步：格式导入

**hpsdecode（MIT，PyPI 1.0.0）—— 实测记录**
- 已装入调研 venv；对真实文件 `刘金鸿-upperjaw.DCM`（20.7MB，`<HPS version="1.1">`）实测：
  `HPSEncryptionError: Encryption key not found`（说明：识别为 **CE schema**，格式受支持但加密）。
- 代码路径逐层解析到 `schemas/ce.py`，库自述"不提供密钥、也不提供获取途径"——与我们的纪律一致。
- 支持矩阵：CA ✅ / CC ✅ / CE ✅（需密钥）/ CB ❌（无样例数据，未实现）。
- 结论：一旦有**合法密钥**（例如客户在自己 3Shape 环境中导出时提供的密钥），这条路能无缝替代现在的
  "操作员另导 STL"折衷；没有密钥则维持现状。MIT 许可可商用。

**Open3SDCM（BSL-1.0）**：C++20 + nanobind Python 绑定，同样支持 CA/CB/CC/CE（CE 需密钥），
仓库内还带 HIMSA《Packed Scan Standard format 501.pdf》。未构建实测（需要 CMake/vcpkg），
定位为 hpsdecode 的备选。

### 3.2 第 2 步：牙位自动定位（最值得动的一步）

**ToothGroupNetwork（MICCAI 2022 挑战赛冠军方案）**
- 存在 `main` 与 **`challenge_branch`**（官方推荐做推理用），challenge 分支含完整推理脚本
  （`inference_mid.py` / `inference_final.py`）、输入约定（`casename_upper.obj` 命名 + 指定轴向）、
  输出 JSON（`jaw/instances/labels`，即 FDI 标签），`ckpts/` 目录**只有 dummy.txt**。
- 权重位置：README 指向 **Google Drive**（`drive.google.com/drive/folders/15oP0CZM...`），
  服务器不可达。grep 全仓库确认权重不在代码里。
- 依赖：`external_libs/pointops` 与 `pointnet2_utils`（自定义 CUDA 扩展），官方安装文档只测到
  **torch 1.7.1**；移植到现代 torch + sm_120（5090）需要实际工作量。
- LICENSE：无（README 无许可声明）。

**3DTeethSAM（AAAI 2026，arXiv 2512.11557）**
- 轻量适配器接在冻结的 SAM2（Hiera-L）上，多视角渲染 + 2D→3D 反投影 + Graph Cut，
  Teeth3DS 基准 95.48% OA / 91.90% T-mIoU；输出**逐顶点标签**，与我们网格链可直接衔接。
- 权重同样在 **Google Drive**（含 SAM2 权重），不可达。
- 依赖更重：`pytorch3d`（需源码编译）、PyMaxflow、pymeshfix、decord 等；预处理把 OBJ 转 npz/hdf5。
- LICENSE：无。

**可行的取权重方案**：找一台能访问 Google 的机器下载两份权重（TGNet 的 `ckpts(challenge).zip`、
3DTeethSAM 的 Drive 目录），上传到服务器 `/data/luoda/oss_crown_research/` 即可开始。

### 3.3 第 3 步：真实颈缘（无开源可用）

- TrimLine 仓库（TLP-Net/TLR-Net）README 中 "Coming soon!" 出现 4 次（预处理、数据、权重均未发布），
  无法运行；且任务目标是"牙-龈修剪线"而非修复体 finish line。
- Dental-CAD-Designer 的 `κ2 脊线自动检测`（曲率脊 + 全手工编辑）可作为算法思路参考（见 3.7），
  但它自述只在合成 fixture 上验收过，真实口扫未认证。
- 结论：继续走自研 Auto Margin 线；"候选线 + 局部路径优化 + **选择性自动接受**"的方向与我们
  Blind v1 的结论一致（重点是把"自信地错"压到零容忍，而不是继续压中位误差）。

### 3.4 第 5 步：AI 外冠生成（ToothCraft 实测专题见第 4 节）

- **CrownGen**（arXiv 2512.21890，HKUST）：真仓库（78 个 py 文件 + 4.6MB 样例 dentition 数据），
  输入是**按 FDI 编号的逐牙点云 + 圆柱边界 json**（即依赖第 2 步分割产物），两阶段
  （边界预测 + 点云扩散），重建水密网格另需 DPSR（Shape as Points）。**未发布训练权重**。
  依赖 pytorch3d 0.7.8、torch-scatter、libigl-python。LICENSE：无。
- **DMC**：仓库指向的正是 MICCAI 2023 "From Mesh Completion to AI Designed Crown" 的代码（建于
  Shape as Points 之上，含 `datasets/crowndataset.py`），但 `readme.txt` 为空、无牙科权重、
  依赖停在上古版本（open3d==0.9 / timm==0.4.5）、LICENSE 无。不可用。
- **AI-3D-Dental-Crown-Generation（stanveer）**：MIT 许可，但 README 声称的
  `checkpoints/best_model.pth`（17M 参数）**在仓库中不存在**，仅有一组示例输出；判为不可用。
- **DCrownFormer（MICCAI 2024）**：仓库只剩 readme——"2024 年 9 月算法转让给韩国公司，
  不再公开代码"。只能读论文。

### 3.5 第 6 步：内冠与倒凹（Blockout）

- **C.B.GEN（MIT，SPIE MI 2024）**：代码级核实——`scripts/cb_generation.py` 仅 117 行，
  做法是：margin 线点集 → `lines_from_points` → 沿硬编码方向 `(3,2,10)` 挤出 → 凸包 →
  `select_enclosed_points` 切出冠底 → 按到 margin 距离做平滑。参数里有
  `dist_margin_undercut / dist_margin_smooth`，但**没有沿就位轴的可达包络构造**。
  结论：结构可借鉴（margin 包围盒切分 + 分区平滑），不能原样替换我们的 Cutter。
- **odc_public（Open Dental CAD，Blender 2.8 插件）**：功能含 undercut surveying 与
  **undercut blockout**，是"传统几何 CAD 倒凹填平"的完整参考实现；但无 LICENSE 文件，
  且是老版本 Blender 插件，只能当思路参考。
- 建议仍按四层结构自研：真实备牙面 → 沿就位轴可达包络（只处理真正阻碍就位的局部倒凹）→
  分区 Cement Gap → 布尔与验收；几何库用现有 manifold3d + Open3D + libigl 即可。

### 3.6 第 7 步：接触与咬合

开源侧没有能替代 Contact v2 / Occlusion v1 的实现；核心问题仍是**输入几何**（邻牙 patch 从
源牙弓重新提取，见 J3 文档的 P1 任务）。第 2 步的分割模型（TGNet/3DTeethSAM）一旦跑通，
可直接用于把邻牙/对颌 patch 提取做实——这是两条线交汇的点。

### 3.7 第 8 步：技师式微调

- **251123_MeshSculpter（MIT，2025）**：three.js 单页原型，含 raycast 笔刷（Sculpt 法向位移 /
  Inflate-Deflate / Blend 平滑锐化 / Twist-Warp）、笔刷半径/强度/衰减、细分与面数护栏、
  撤销/重做栈、OBJ 导出、线框与统计面板。与我们的 React + three.js 前端同栈，是"推拉/膨胀/
  平滑"第一批操作最省事的起步参考（MIT 可商用）。
- **Dental-CAD-Designer（ZoliQua，匈牙利个人项目 "DQ Dental CAD"）**：功能清单几乎覆盖我们 8 步
  （κ2 边缘线、就位轴热力图、内冠/cement gap、RBF 形变、自由雕刻、BVH 三角对自交检测的 QC 门、
  服务端字节级复核、案例归档）。但：**无 LICENSE**、自述"真实口扫认证与技师 beta 未做"。
  仅作为**架构与 QC 门设计**的参考，不要复制代码。

---

## 4. ToothCraft 实测专题（本次核心实证）

### 4.1 被测对象与准备

- 模型：`ToothCraft_normal_diff_branch.pth` + `ToothCraft_normal_control_branch.pth`
  （HF `DejvaX/ToothCraft`，494MB，经 hf-mirror 下载成功；checkpoint 内 iteration=750k）。
- 配置：仓库 `configs/test/test_normal.yaml`（无 class conditioning，`y=None` 正确）。
- 病例：FDI 46 真实病例包（GT 冠 = 技师设计冠 `final_crown.stl`）。
- 推理：`p_sample_loop`（SpacedDiffusion，respacing=10 → 100 步二阶采样），
  5090 上约 6 秒/样本。

### 4.2 先验 operand：实测中排掉的四个输入侧陷阱

| # | 陷阱 | 现象 | 修正 |
|---|---|---|---|
| 1 | **GT 冠是薄壳** | `final_crown.stl` 体积 411.5mm³，体内最深 SDF 仅 -0.111（壳壁厚 ~0.5-1.5mm） | 需构造"实心牙体"真值（栅格化屏障+洪泛填充） |
| 2 | **坐标系错配** | `die.stl` 与牙冠不同帧（冠→die 中位 3.47mm）；`prep.stl` 才与牙冠同帧（内表面间隙中位 0.094mm/p95 0.150mm） | 全部几何统一用 `prep.stl` 判定内外 |
| 3 | **prep.stl 非水密** | 543 条开放边；fill_holes 无效；其 SDF 符号大面积泄漏（solid min=-0.75），模型因此产出巨大团块 | 放弃 mesh2sdf，改用体素屏障+洪泛填充构造实心域 |
| 4 | **归一化约定非 0.8 而是 ~1.6** | 按仓库代码推导为"牙占体积 0.8"时，**恒等对照**外表面差 2.52mm；改 1.6（`--scale2`）后降到 0.65mm | 以恒等对照为标尺实测标定 |

第 4 条是本轮最重要的**方法学发现**：按仓库代码推导的"牙占体积 0.8"约定下，恒等对照外表面差 2.52mm；
把目标牙放到 1.6 extent（`--scale2`）后降到 0.65mm，必须实测标定。

> ⚠️ **二轮复核修正（2026-10-10，见 §9）**：
> 1. `scale2` 的 1.6 extent 是**本病例上标定出的有效配置**，不是模型通用的官方输入尺度；
> 2. 恒等对照的 **0.65mm 是当前输入/推理配置下的重建误差，不是 64³ 模型在数学意义上的固定精度上限**；
> 3. 官方 README 的配方原文是："use a local context cutout that includes neighboring crowns and
>    spans roughly twice the size of the missing or damaged tooth. The cutout should then be
>    normalized to the unit sphere and approximately aligned with the ODD arches."
>    即 **带邻牙冠的局部牙弓截取（≈2× 缺损牙）→ 归一化**；结合代码（`process_ood`/`process_3ds`）
>    与训练配置（全部为 `train_64_odd_*`），OOD 管线的复合归一化 = 单位球（`v /= max_dist`，按顶点均值）
>    + 盒式 0.8（max extent→1.6），复合结果等价于"截取盒 max extent → 1.6"，**牙冠在 2× 截取盒里约占 0.8**。
>    因此单牙无上下文实验里 0.8 的失败，很可能就是"缺上下文"的假象，而不是官方约定错误。

### 4.3 三个实验（修正约定后；外表面误差 = GT 冠外表面点到预测网格的距离，mm）

| 实验 | 中位 | p95 | 最大 | 洞填满率 | 体积比 pred/gt |
|---|---|---|---|---|---|
| C 恒等对照（完好输入） | **0.65** | 0.84 | — | — | 1.28× |
| A 补洞（3 球洞，占体积 34%） | **0.89** | 1.30 | 2.15 | **100%** | 1.56× |
| B 牙桩 → 整冠（只给下部 38%） | **1.09** | 2.14 | 5.99 | 21% | 0.79× |

辅助证据：在错误约定下（0.8），层级扫描显示 level=-0.10 时体积比 0.97、误差 0.61mm，
level=0 时 2.38×、2.49mm——即"形状大致对、整体系统性外扩"；换到正确约定后该偏置基本消失。

### 4.4 结论

1. **形状"像"但精度不够格（限定在本轮测试条件）**：在"单牙无上下文 + 标定尺度"的配置下，连
   **原样复现**都要差 0.65mm 中位，真实任务 0.89–1.09mm 中位、p95 1.3–2.1mm、最差 6mm。
   64³ 体素（~0.25mm/格）与缺少官方上下文截取都是候选原因，**但这两条都不构成"模型上限"的证明**；
   0.65mm 只能表述为"当前输入/推理配置下的重建误差"。我们的接触/咬合 QC 要求是 0.1–0.2mm 量级，
   因此**在该配置下**差约一个量级。论文自称的"真实病例迁移"是"咬合干扰最小"的定性描述，与本次量化一致。
2. **补洞能力真实**（洞 100% 填满），但伴随体积外扩 1.56×——不能当"保持形态的修复"用。
3. **工程可用性一般**：官方代码开箱跑不起来（见下），且**无 LICENSE**，权重不能进产品。
4. 适用定位：研究对照 / 形态先验候选（若要产品化：需作者授权 + 用自己的 GT 对微调 +
   提高分辨率，且仍需我们的确定性 CAD 层施加 Margin/内冠/厚度/接触约束）。

### 4.5 释出代码的三处坑（复现者必读）

1. `configs/*/test_*.yaml` 里 `unet_activation:` 为空 → `getattr` 拿到 None（不走默认值）→
   `NotImplementedError: Unsupported activation: None`；需显式设 `SiLU`。
2. `p_sample_loop` 的 `model_kwargs` 必须包含 `noise_save_path: None`，否则 KeyError。
3. requirements 缺三个实际必需依赖：`rtree`（trimesh 最近点）、`shapely`
   （`slice_mesh_plane(cap=True)`）、`pymeshfix`（官方数据管线用）。

---

## 5. 许可证与合规红线

| 资源 | 许可 | 商业使用 |
|---|---|---|
| hpsdecode | MIT | ✅（CE 解码仍需合法密钥） |
| C.B.GEN | MIT | ✅ |
| 251123_MeshSculpter | MIT | ✅ |
| Open3SDCM | BSL-1.0 | ✅（宽松） |
| stanveer/AI-3D-Dental | MIT | ✅（但无权重） |
| **ToothCraft / TGNet / 3DTeethSAM / CrownGen / DMC / odc_public / Dental-CAD-Designer / TrimLine** | **无 LICENSE 文件** | ❌ 默认保留所有权利；需作者书面授权 |
| dental-guide-mesh-poc | 专有（All Rights Reserved） | ❌ 禁止运行/复制/修改/分发 |
| 3DTeethSeg 挑战赛数据集 | CC BY-NC-ND 4.0 | ❌ 不可直接用于商业训练 |

说明：上表的"无 LICENSE"为逐仓库实测结论（含检查 README 内的许可声明；CrownGen 的 README 里
只有对第三方依赖的许可标注，没有自身许可）。

---

## 6. 建议的行动顺序（按杠杆排序）

1. **搬权重 → 跑通第 2 步分割**：找能访问 Google 的机器下载 TGNet `ckpts(challenge).zip` 与
   3DTeethSAM 权重 → 上传服务器 → 先跑 TGNet（有完整推理脚本；主要工作是 pointops/pointnet2
   两个 CUDA 扩展移植到现代 torch）。产出 FDI 实例分割后，可直接解决第 7 步的
   **邻牙 patch 从源牙弓重新提取**（J3 文档 P1 任务）——这是当前整条链最大的卡点。
2. **第 7 步输入修复**仍按既有计划推进（源资产完整的两个病例 `82404140632`、`82404090822`），
   与第 1 条互相印证。
3. **第 5 步定策略**：生产端保持 Template CAD；AI 端把 ToothCraft 当对照基线（环境已就绪，
   可批量跑病例出统计）。若考虑微调，先盘点"原始扫描+备牙+技师 Margin+最终冠 STL"的成对数据量，
   并同步与作者谈授权。
4. **第 6 步自研 Blockout**（四层结构），C.B.GEN/odc_public 仅作思路参考；验收沿用既有
   回归基线（0.523mm 中位间隙作为失败基线，不得用调小 gap 掩盖）。
5. **第 8 步**：以 MeshSculpter（MIT）为起点做 three.js 笔刷，落实"Margin/最小厚度/锁定区
   自动保护 + 提交后重算 QC"两条硬规则。
6. **第 1 步**：hpsdecode 已装；与业务确认能否合法取得 CE 密钥——能则打通原始包，不能则维持
   操作员 STL 路径。

---

## 7. 复现资产与命令

服务器 `/data/luoda/oss_crown_research/`：

```
repos/                 16 个已克隆仓库（含 TGNet_challenge 分支）
venv/                  torch 2.11.0+cu128 / xformers 0.0.35 / hpsdecode 1.0.0 /
                       trimesh 5.1.1 / mesh2sdf / scikit-image / shapely / rtree / omegaconf / hydra-core
toothcraft_ckpt/       ToothCraft_normal_{diff,control}_branch.pth（494MB）
case_b1b73bae/         FDI46 病例包（final_crown/prep/die/邻牙与对颌 patch/case.json）
tc_run6.py             实测主脚本：--stage prep|run|metrics，--test A|B|C，--field edt，--scale2
tc_diag*.py           SDF 约定/帧对齐/实心域诊断脚本
hps_test.py            hpsdecode 对真实 DCM 的解码测试
tc_out/                各版本输入/输出 npz 与预测 STL（v6*_A/B/C、v6*res_*、v6*pred_*.stl）
digest.txt             全部仓库的许可/README/依赖/权重线索摘要
```

本机助手：`C:\Users\Administrator\.zcode\workspace\default\cf_ssh85.py`（paramiko，密码认证）。
常用命令模式：

```bash
# 克隆（GitHub 不可直连时）
git clone --depth 1 https://gh-proxy.com/https://github.com/OWNER/REPO.git
# 下载 HF 权重
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download DejvaX/ToothCraft --include "ToothCraft_normal*" --local-dir ./toothcraft_ckpt
# 复现 ToothCraft 实验（标定配置；官方配方与归一化推导见 §4.2 修正块）
./venv/bin/python tc_run6.py --stage prep --test A --field edt --scale2
./venv/bin/python tc_run6.py --stage run  --test A --field edt --scale2
# 带邻牙上下文（薄片）实验：tc_run8.py --stage prep|run --test B0|B1 [--scale2]
# 权重下载（HF 镜像 Xet 通道 401 时加 HF_HUB_DISABLE_XET=1）
```

---

## 8. 本次未验证 / 待办（如实记录）

- TGNet 与 3DTeethSAM **未实跑**（权重位于不可达的 Google Drive）；两仓库的现代 torch 移植
  工作量只有静态评估，未实测。
- ToothCraft 仅测 **1 个病例**（FDI 46）；`antag`（含对颌条件）与 `classfree` 两组权重未测；
  也未能在其官方数据管线上做交叉验证（其数据集同样在 Google Drive）。
- C.B.GEN 仅做代码级核实，未执行；Open3SDCM 未构建。
- ToothCraft 的"外扩"现象在正确约定下大幅缓解，但**根因未完全定论**（最可能是训练切片含
  牙弓/牙龈上下文而我们的输入只有单牙；也可能与域差异有关）。若继续推进，建议先补
  "带上下文输入"的对照实验。
  → 已补（§9.3），但用的是薄片 patch 而不是官方意义的牙弓截取，尚不能定论。

---

## 9. 第二轮：复核意见落实与补充实测（2026-10-10）

### 9.1 对 §4 的表述修正（已就地标注）

- `scale2` 的 1.6 extent 是**本病例标定出的有效配置**，不是模型通用的官方输入尺度；
- 恒等对照的 **0.65mm 是当前输入/推理配置下的重建误差，不是 64³ 模型在数学意义上的固定精度上限**；
- 官方 README 的配方原文（含归一化推导）已补入 §4.2 的修正块。

### 9.2 新增候选核实（本轮克隆实测）

| 项目 | LICENSE | 权重 | 环境 | 判词 |
|---|---|---|---|---|
| TSegFormer（MICCAI 2023，huiminxiong/TSegFormer） | **MIT**（LICENSE 实测存在） | **无**（README 只给训练脚本，`best_model.t7` 需自训产生） | python 3.7.11 / torch 1.9.0+cu111 | 授权路径最清晰的分割备选；**不能解决"现在马上推理"**，适合"自有数据重训"路线 |
| zju-ai4h / Unsupervised-Pretraining-for-tooth-segmentation（MIDL 2022） | **无** | 有 `finetune/pretrained_weights/checkpoint.pth`，README 原文注明 "For *debugging purpose only*" | MinkowskiEngine 0.5.4 / open3d 0.9.0（现代 CUDA 难装） | 研究参考，不作推理候选 |

**3DTeethSAM 设备硬编码（确认）**：`preprocess/preprocess.py` 第 2 行为
`os.environ["CUDA_VISIBLE_DEVICES"]="1"`（未注释）；`start_inference.py` 内为可配置的 gpu_id 逻辑，
`start_train.py` 内已注释。单卡机器直接跑 preprocess 会选错卡，移植时必须改为可配置。

### 9.3 实验 B（带邻牙上下文）实测结果

上下文构造：目标牙实心域 + 近中/远中 patch 栅格化薄片（±1 体素）；
固定随机种子（123）、同一病例、同一权重；指标含双向 Chamfer / HD95 / 体积偏差 / 近远中与咬合分区误差。

| 配置 | 恒等对照 B0（GT-outer→pred 中位/p95，mm） | 损坏 B1（同指标） | 体积偏差 |
|---|---|---|---|
| 单牙，截取盒口径（牙=0.8） | 2.52（v6 C） | 2.49（v6 A） | +140% 级 |
| 单牙，牙=1.6（scale2） | **0.65** | 0.89 | +28% / +56% |
| 牙+双邻牙薄片，牙=0.8（官方截取口径） | 1.96 | 1.92 | +47% / +48% |
| 牙+双邻牙薄片，牙=1.6 | 1.12 | 1.41 | +57% / +73% |

**结论（如实）**：我们手里的"上下文"只有近接壁 patch 薄片，**不是**官方要求的"包含邻牙冠的
局部牙弓截取"；模型会把薄片"增厚成牙"，连带把目标牙区拖偏——两种尺度下上下文都使结果变差。
因此：既**不能**据此判定"上下文无用"，也**不能**据此判定官方配方成立。
决定性实验（D：完整真实牙弓局部截取）必须使用真实牙弓资产，材料已定位（见 §9.4）。

### 9.4 权重与资产解封（已完成的准备）

- **antag 权重**：HF 镜像在 Xet 通道返回 401（`cas-server.xethub.hf.co`）；
  用 `HF_HUB_DISABLE_XET=1` 重试**成功**，已落盘 `toothcraft_ckpt_antag/`（540MB，含
  diff/control/antag 三支）→ 实验 C 的权重条件具备（但仍需真实对颌/牙弓上下文）。
- **语料服务器**（192.168.110.15，只读）确认 `exo/EXO/82404140632/` 有
  `82404140632-150778-A组 (1)(2)(3).stl`（真实牙弓）；另一例 `82404090822` 结构相同。
  这两份"源资产完整"病例即 P0（源牙弓→FDI46 邻牙提取）与实验 D 的材料。

### 9.5 二轮后的执行顺序（与复核意见一致，仅补依赖关系）

1. **P0 源牙弓 → FDI46 邻牙提取 → GT Contact Gate**：不等 AI 权重；先用现有
   user_seed + 测地线分割建立"人工复核的可信几何集"，作为后续自动分割的对照基准。
2. **P1 权重搬运与移植**：TGNet challenge 权重（Drive → 人工搬运）+ pointops/pointnet2 移植；
   3DTeethSAM / TSegFormer 作为对照（注意设备硬编码、授权与 MinkowskiEngine 依赖）。
3. **P2 Blockout 重构**；**ToothCraft D 实验**（真实牙弓截取 + 可选 antag 权重）→ 依据 D 的结果
   再决定是否值得补 A/B/C 的完整对照。
4. **P3 MeshSculpter 前端集成**。

> 注：本轮之后，§8 中"ToothCraft 未做上下文实验"一条已部分落实（薄片上下文），
> "决定性上下文实验"仍待真实牙弓资产；TGNet/3DTeethSAM 仍未实跑。

---

## 10. 后续：生成路线决策与 P0 真实牙弓实测（2026-10-11）

- **决策记录**：`docs/GENERATION_ROUTE_DECISION.md`（ToothCraft 降为研究基线；确定性 CAD 为主线；
  证据分级表；预先约定的停止条件已触发）。
- **P0 实测**（真实病例 82404140632）：资产搬运 SHA256 三处一致；坐标系由病例元数据恢复
  （margin-Kabsch 残差中位 0.011mm，GT 冠内表面→工作颌 0.031mm）；邻牙 patch 提取（GT 辅助 Oracle）
  与 GT Contact Gate 读数（p05 −0.04/−0.045mm，负例可判别）。
- **P0-5 结果**（统一 bounds 内口径）：最好配置 pred→GT 中位 1.08mm / p95 1.83mm；
  完整牙弓上下文与 ODD 式方向对齐均**未**改善；对颌条件（antag）在官方口径下无实质变化、
  在有效尺度下仅中位改善而尾部/体积劣化 → 停止条件触发。
- ⚠️ **更正**：§4.4/聊天中"比 J.2 margin p95 0.023mm 差 1–2 个数量级"的说法**已撤回**——
  颈缘指标与冠面 Chamfer 不可比；与确定性 CAD 的严格对照需统一指标后重做。
- 产物与可复跑脚本：服务器 `110.85:/data/luoda/oss_crown_research/p0/`（含 `p05/README.md`、`p05_table.md`、`record.json`）。
