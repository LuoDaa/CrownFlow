# 第三方源码审计的逐条核验（2026-10-12）

对象：一份外部（ChatGPT）源码审计报告的 P0/P1/P2 结论。方法：**只读源码 + 关键项实际复现**。
结论摘要：**5 条 P0 基本成立**（1 条已复现），**3 条被夸大或需限定范围**，**1 条不成立**。

## 一、确认成立的（附文件:行号）

| # | 结论 | 证据 | 我的复现 |
|---|---|---|---|
| P0-1 | **Placement 帧不一致**：预览用已确认 mesial/buccal/axis，实算重建 world-X 基 | `geometry/registration.py:69` `x=np.array([1.,0,0]);x-=axis*np.dot(x,axis)`；`api/designer.py::initialize_draft_outer` 牙库分支**同样**重建 world-X 基（**两处都要修**）；对照 `services/auto_fit_templates.py::placed_library_crown` 用的是 confirmed 帧 | 读源码确认 |
| P0-2 | **`_selected_template` 未被 draft 路径消费** | `initialize_draft_outer` 只按 `TEMPLATE_DIR/FDI_xx.stl` → `template_studio` 候选 → `discover_libraries()` 第一个成功者；`_selected_template` 只被 `services/design_dependencies.py:16`、`services/auto_fit.py:90` 消费 ⇒ "几何来自 B、报告记 A"可达（用户选择≠TEMPLATE_DIR 默认件时） | 读源码确认 |
| P0-3 | **审核口径无符号**：求解有 signed、审核只有 unsigned | `api/designer.py:248,292` 与 `candidate_analysis` 用 `closest_point`（≥0）；`crown_outer/auto_fit_adapter.py` 已有 `FitSurface`（闭合体 `mesh.contains` 定 inside/outside；开放 patch 要求 reviewed outward normal 只建局部语义）、`measure()` 出 penetration/ambiguous、`fit_stage` 门 + α 回溯、`surface_crossings` | 读源码确认 |
| P0-4 | **PHI 未真正脱敏持久化** | `services/three_shape_import.py:220 keep_source_package=True`、`:195/:316` 落 `phi_fields`、asset.`original_name`=包内 logical path；`tests/unit/api/test_three_shape_import_api.py:218` 断言 `source_package.zip` 存在 | 读源码确认 |
| P0-5 | **`/api/crown` artifact 路径穿越** | `services/crown_outer_runner.py:189` `if not str(p).startswith(str(root))` | ✅ **复现**：`run_id=r1` + `name=../r10/secret.txt` 读到兄弟 run 文件；`../secret`、`sub/../../secret` 被拒 → 现有测试没覆盖前缀碰撞。修复：`p.is_relative_to(root)`（仓库 `api/routes.py:856` 已用此惯用法） |
| P1-附 | `register_bundle` 对 `case_uuid` 无校验 | `api/crown.py:108` `d = CASES_ROOT / p.case_uuid`，仅靠 `case.json` 存在兜底 | 读源码确认（危害低于 artifact 穿越） |

## 二、被夸大或需限定范围的

| # | 审计说法 | 实际情况 |
|---|---|---|
| P1-1 | "manual edit/restore 的 publish 与 revision insert 不是同一事务" | 成立，且**范围更大**：`publish_artifacts` 只有 `api/auto_fit.py:342` 带 `revision=`；`api/designer.py:214/241`、`api/routes.py:634/690` 以及 **`routes.py:774`（finalize 后另 append FINAL 修订）** 都是"先提交产物、再记修订"两步 |
| P1-2 | "ParameterResolver 已存在但 Workbench 没真正用" | 成立：`app/parameters/`（resolver/lab_profiles/derived/classify/profiles/models）成熟且含 PHI 扫描与"禁 MISSING→0"，但 `api/designer.py::set_profile` 未 import resolver ⇒ 结论应是"已有规则引擎没接 live"，**不是"缺材料引擎"** |
| P2-1 | "`random_split()` 是文件级 ⇒ train/val 可能泄漏" | **当前不成立**：`ml/build_dataset.py` 是**一病例一个 npz**，按文件切分即病例级切分；只有未来一病例出多样本时才需按 patient/order 分组 |

## 三、ML 输入表示（成立，但属 P2）

`ml/build_dataset.py`：context = `(采样点 − support 质心)/scale`（**无解剖规范化**，保留扫描器朝向）；训练模板 = `sample(template,2048,rng)`（随机表面点），而 `services/ai_runtime.py::predict_outer` 用 `template.vertices`（拓扑顶点）⇒ **训练/推理表示不一致**；`sample(mesh,n,rng)` 内部 `trimesh.sample.sample_surface(mesh,n)` **丢弃 rng**（`default_rng(7)` 不生效）。
`ml/checkpoints/` 为空 ⇒ 该线未启用（与 README 自述一致）。

## 四、K0 归因（审计的"更正"与项目文档一致）

`docs/K0_PROXIMAL_FIT.md:680`《⚠️ 更正（2026-10-09 晚）：本节全部接触数字受 patch 作用域污染》明确撤回：
`uniform_scale 1.1mm` 根因、H.2/H.2.1/H.2.2 全部渗透读数、"J.2 再膨胀" 都是**在被污染 patch 上量的**，需重裁 patch 后重测；被推翻的是该案数字与归因，机制成果（冻结域/α 回溯/几何门/J.2 seam Phase 3/可行性审计）不受影响。
另核验：live 路径 `initialize_from_margin` 用的是**两轴独立 scale**（`factor=(target_extent+.4)/extent` 作用于 `local[:,:2]`），**不是**研究期等比 `uniform_scale`。
⇒ "邻牙感知放置要做"仍成立；但**不能再说"已证实 live 冠宽问题就是 uniform_scale=1.10218"**（本次也已更正本地记忆里过期的同一条表述）。

## 五、建议执行顺序（采纳 A→D）

- **A 让 live geometry 只有一种语义**：`resolve_template_source(design)`（唯一模板解析，吃 `_selected_template` 的 sha256）+ 统一 `PatientFrame`（用 confirmed mesial/buccal/axis）喂给 placement/baseline/provenance；`registration.initialize_from_margin` 与 `initialize_draft_outer` **两处**都要改。
- **B 把 signed contact 做成基础服务**：定义唯一 `ContactMeasurement`（signed/unsigned/penetration/target_error/p95/max/near_contact_area/penetration_area/patch_count/triangle_crossings/sign_scope/sign_confidence），供 placement、AutoFit、candidate review、final QC 共用；把 `auto_fit_adapter.FitSurface` 作为唯一实现来源。
- **C 再减人工输入**：instance segmentation/FDI、bite 刚体配准、research auto-margin 接 live（**已在本轮完成第一批**，见 `MARGIN_AUTODETECT_EXOCAD_PARITY.md`）——全部保留 revision/confirm Gate。
- **D 参数与制造规则进 live**：把 `app/parameters` 的 resolver 接进 Workbench Profile，而不是再建引擎。
- **部署前必修（与 A–D 并行）**：artifact 路径 `is_relative_to`、`case_uuid` 校验、PHI 政策（`keep_source_package` 默认值与 zip 处置、`phi_fields`/logical path 是否落库）。
- 之后才 SSM/CSM，最后生成式。

## 六、本次核验未改动任何源码

仅新增本记录与本地记忆条目；未修改被审计代码。上表所有条目可用给出的文件:行号复核。
