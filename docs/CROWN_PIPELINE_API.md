# CrownOuter 流水线封装 + 前端工作区

把已冻结的几何引擎（H.1 冠外形 → 内表面 → Contact v2/I.2 → Occlusion v1 →
Thickness v1 审计；J.1 / J.2.1 为实验阶段）接成**可调用的 API + 后台任务 + 前端工作区**，
UI 结构参考 3Shape Automate 的病例编辑页。

## 1. 分层

```
frontend/src/components/CrownWorkspace.jsx   病例工作区（阶段 + 参数 + 3D 分层 + QC）
frontend/src/api.js                          crown 端点
backend/app/api/crown.py                     HTTP 层（登记病例 / 起任务 / 轮询 / 产物）
backend/app/services/crown_outer_runner.py   后台任务（线程 + manifest 落盘 + 路径安全）
backend/app/services/crown_outer_pipeline.py 阶段编排（每个阶段调用一个已验收引擎）
backend/tools/run_crown_api_e2e.py           端到端验收（走前端同一条 API）
backend/tests/unit/services/test_crown_api.py 封装语义单测（快，不跑几何）
```

## 2. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/crown/stages` | 阶段清单 + 默认参数（带 provenance）+ 中文标签 |
| GET | `/api/crown/cases` | 服务器上的病例包（`crown_outer_cases/<case_uuid>/`，PHI-free） |
| POST | `/api/crown/cases/register` | 把病例包登记进 app 库（幂等，只用 case_uuid 命名） |
| POST | `/api/crown/runs` | 起后台任务 `{case_uuid, stages[], params{}}` → `run_id` |
| GET | `/api/crown/runs` | 历史任务（磁盘 manifest，最新在前） |
| GET | `/api/crown/runs/{id}` | 轮询：阶段状态 / 日志 / 结果（done 后带 QC、分区表、分层） |
| GET | `/api/crown/runs/{id}/artifact/{name}` | 下载 STL / GLB / JSON（**限定在 run 目录内**） |

阶段与依赖（依赖自动补齐、顺序固定）：

```
outer_h1 ─────────────────────────┐
intaglio ─────────────┬───────────┤
contact (← outer_h1)  │           ├─ thickness_audit (← outer_h1,intaglio,occlusion)
occlusion (← contact) ┘           │
envelope_j1 / collar_body (← outer_h1,intaglio)   实验阶段
```

## 3. 分层热力图（参考 3Shape 的"显示器"面板）

审计阶段直接产出顶点色 GLB，前端用 `GLTFLoader` 叠在 STL 上：

| 图层 | 域 | 颜色语义 |
|---|---|---|
| 平面 | — | 无着色（STL） |
| 薄区域 | **内表面** | 红=不达标 → 绿=达标（`t_primary / T_threshold`） |
| 邻牙接触 | 外表面 | 红=穿透 / 绿=有间隙（到两侧 patch 的带符号距离） |
| 咬合间隙 | 外表面 | 红=干涉 / 绿=离开（到对颌的带符号距离） |

**薄区域必然落在内表面**：壁厚是"内表面到外表面的距离"，着色在内表面上才有物理意义
（前端图例里写明了这一点）。

## 4. 起服务

```bash
cd backend
export CROWNFLOW_CROWN_CASES_DIR=<数据盘>/crown_outer_cases   # 真实病例包仓库
export CROWNFLOW_TOOTH_LIBRARY_ROOT=<数据盘>/牙齿数据库        # 模板牙库
python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
# 前端：npm run dev（开发）或 npm run build（产物由后端 / 挂载）
```

## 5. 端到端验收（真实例 cf_b1b73bae221211b1）

`python3 tools/run_crown_api_e2e.py` ——走前端同一条 API，2026-10-08 实测：

| 阶段 | 耗时 | 结果 |
|---|---:|---|
| outer_h1 | 10.6s | OK（scale 1.1022） |
| intaglio | 0.0s | — |
| contact | 53.1s | I.2 完成，颈缘位移 0 |
| occlusion | 12.6s | REVIEW_REQUIRED，7 轮，**颈缘/邻接核心位移 = 0.0**，翻面 224 |
| thickness_audit | 34.7s | **FAILED**（输入自交 442 —— 这是已知的 H.1→接触→咬合 折叠，审计如实拒绝） |

审计数字（与直接跑引擎一致）：负壁厚 733 点 / 14.71mm²、薄区 56.97mm²
（可修 15.63 / 接触冲突 6.26 / 咬合冲突 17.81 / 颈缘 17.26）；分层 GLB 三个均可下载
（`glTF` 魔数校验通过）。

**诚实边界**：主链的最终审计现在是 **FAILED**，因为 H.1→接触→咬合 这条链仍会把
网格折起来（自交 442）—— 这正是 J.2 系列要解决的上游问题。封装层**不掩盖**它：
任务 `status=done`（流程跑完）但每个阶段的 `status` 与审计结论原样透出，
前端 QC 面板第一行就是"壁厚审计：FAILED"。

## 6. PHI 与数据纪律（沿用仓库既有约定）

* 病例包只用 `case_uuid` 命名；接口与 manifest **不含**患者姓名/原文件名/内网路径；
* 真实病例包用环境变量指到数据盘，**不复制进仓库**（`data/*` 与 `数据/` 都在 .gitignore）；
* `frontend/dist/` 也在 .gitignore —— 前端要 `npm run build` 后再由后端挂载；
* 后台任务 manifest 落在 `CROWNFLOW_CROWN_RUNS_DIR`（默认 `data/crown_runs/`，不入库）。

## 7. 已知未完成

1. **病例包来源**：目前只支持"已暂存的病例包目录"（case.json + prep/die + 邻牙/对颌 patch）。
   从 app 内的扫描资产（upper/lower/support）自动生成 patch（区域生长）尚未接入 ——
   缺 patch 时阶段会**明确失败**并说明缺什么，不会静默跳过。
2. **J.1 / J.2.1 是实验阶段**：前端标注（实验），结果面板带 `experimental: true`；
   J.2.1 的 body 仍不干净（见 docs/OUTER_V2_J21.md），下一轮转 Boolean 路线。
3. **下游硬化未做**：Contact/Occlusion 的接受条件还没加 no-fold / no-self-intersection
   硬 QC（Occlusion 的 224 翻面就是这条欠账的直接证据）。
