# Contact Engine v1（提交 I）

**日期**：2026-09-29。代码 `backend/app/geometry/crown_outer/contact.py`；
合成测试 `backend/tests/unit/geometry/test_contact_engine.py`（8 项全过）；
真实链路 `backend/tools/run_contact_v1_case.py`。

```python
optimize_contacts(crown_mesh, patient_frame, mesial_neighbor_patch,
                  distal_neighbor_patch, contact_parameters) -> ContactResult
```

## 只解决一件事

**把牙冠的近中/远中邻接区调整到合理接触，同时不破坏 Margin、咬合面和整体解剖。**

| 区域 | 策略 | 真实病例实测 |
|---|---|---|
| Contact region（近/远中） | 可调整（硬约束） | 修正 326 / 391 顶点 |
| Margin / Cervical（颈环） | **LOCK** | 位移 **2.4e-13 mm** |
| Occlusal（咬合面） | 弱锚强限制 | 位移 p95 **1.2e-8 mm** |
| Buccal / Lingual | Laplacian 平滑传播 | 随接触区收敛 |

做法：接触区顶点沿"到邻牙最近表面点"的**带符号距离最小修正**
（`target = v + n̂·(target_gap − signed)`，位移 = `|target_gap − signed|`，
物理上就是技师在邻接面磨掉/补上的量），再用现有 Laplacian 行替换法传播
（颈环恒等 LOCK、咬合弱锚）。接触区外一圈**锚定在原位**，防止平滑把它拖进邻牙。

捕捉集是**并集**：①穿透集 `signed < target_gap` 且欧氏距离 ≤ 2mm（"真穿透"
口径，避免开放 patch 切口边的法向伪影被当成穿透）∪ ②邻近集
`dist ≤ dist_min + band`（邻牙在几 mm 外时把接触区拉过去）。

## DESIGN_TARGET vs REALIZED_GEOMETRY（永久区分，用户规格）

`ContactParameters.design_target_mm` 只**记录** XML 的 `DistanceToNeighbor`，
**不驱动求解**。求解目标是 `target_gap_mm`（PROVISIONAL，取 GT 反求区间中值
0.05mm）。验收比较的是与 REALIZED_GEOMETRY 的距离/面积/位置。

实测对照（唯一双侧可用病例 `cf_b1b73bae221211b1`）：

| | 设计参数（XML） | 技师实际做出（GT 反求） |
|---|---|---|
| mesial | `DistanceToNeighbor = 0.0` | min **0.045 mm** |
| distal | `DistanceToNeighbor = 0.0` | min **0.000 mm**（接触） |

**设计参数写 0 不等于最终几何 0** —— 目标值必须来自 GT 反求，不能照抄参数。

## 合成测试（8 项，算法正确性的主要依据）

用户规格的 5 类场景 + 不变量，每个 fixture 的真实距离预先已知，断言严格：

1. **邻牙太远** → 接触区向邻牙移动（平面 patch 在 6.0mm，冠面被拉到 0.3mm 外）
2. **穿透** → 自动退让消除穿透（局部 fin 侵入 0.5mm，退到 fin 面 +0.3mm）
3. **一侧正常一侧过远** → 只修改异常一侧（另一侧 `vertices_modified = 0`）
4. **两侧同时约束** → 两侧同时收敛
5. **不可满足**（邻牙 8.5mm 外，需 3.2mm > 上限）→ `DESIGN_INFEASIBLE`，
   **几何不得被半途修改**（fail-loud）
6. 不变量：Margin 位移 < 0.01mm、咬合面基本不动、无翻转面、无退化面
7. 接触指标（面积/质心）报告正确
8. patch 法向自动定向（翻转的 patch 必须被检出并纠正，否则穿透判定反向）

## 真实病例结果（cf_b1b73bae221211b1，H.1 冠 + 双侧邻牙 patch）

```
状态：REVIEW_REQUIRED（残留穿透 1.16 / 0.98 mm）
Margin 位移 2.4e-13 mm      ← LOCK 成立
咬合面位移 p95 1.2e-8 mm    ← 解剖零破坏
法向翻转 84 面（0.7%）
接触面积 mesial 11.0 / distal 21.2 mm²
```

分区带符号误差（正=偏胖），H.1 → v1：

| 区域 | H.1 | v1 | 判读 |
|---|---:|---:|---|
| **mesial** | −0.251 | **−0.205** | ✅ 朝 GT 改善 |
| **buccal** | −0.127 | **−0.079** | ✅ 朝 GT 改善 |
| distal | −0.602 | −0.613 | 基本不变 |
| lingual | −0.326 | −0.346 | 基本不变 |
| occlusal | −0.290 | −0.279 | 未动（符合"锁死咬合"） |
| cervical | −0.244 | −0.241 | 未动（符合"锁死颈缘"） |

## 已知未解决：残留穿透 + 朴素迭代发散

**残留穿透 1.16/0.98mm**，根因是**一阶线性化**：把顶点沿"当前最近面的法向"
移到目标间隙，邻牙是曲面，移动后最近面变了，单次解留下 1mm 级残留。

**朴素外层迭代（重测-重解 + 累积约束）在真实病例上发散**：3 轮后法向翻转
32%（3779 面）、lingual 位移 +4.97mm、cervical +3.75mm —— 每轮的膨胀锚点
钉在**已移动**的位置上，误差逐步累积。所以 `n_iterations` 默认 **1**，
残留穿透如实报 REVIEW_REQUIRED，**不假装收敛**。

下一步的正确做法（二选一）：
* 对邻牙建**符号距离场**后一次性投影（避免逐轮法向漂移）；
* 或加阻尼的外层迭代（每轮限制位移步长、锚点固定在**原始**位置）。

## 数据现状（承前）

Contact v1 的真实数据回归**只有 1 例**（双侧）。其余 17 例缺牙弓扫描，
按用户决定的优先级作为并行数据补齐任务，不阻塞算法开发。


---

# I.1 Contact Penetration Refinement（提交 I.1）

按用户规格的六条要点实现（`app/geometry/crown_outer/contact_refine.py`，
`refine_contacts()`）：

1. **固定 reference**：``V_ref`` = H.1 冠，永不改变；每轮只用当前 ``V_best``
   重新测最近面，Laplacian 求解始终以 ``V_ref`` 为基准（delta 来自 H.1 解剖）。
   测试 ``test_never_accumulates_across_iterations_reference_stays_fixed``
   断言两次独立运行结果逐位一致。
2. **Oriented Distance 而非封闭体 inside/outside**：``φ(x) = (x−q)·n``，
   每轮重新找 ``q, n``。
3. **patch 边界单独处理**：``NeighborPatch.closest_with_validity`` 返回
   ``valid``，最近点落在开放边界安全带内即 ``AMBIGUOUS_PATCH_EDGE``，不参与投影。
4. **阻尼 + trust region**：``Δx = α(g−φ)n``，α=0.4，单轮 ``max_step_mm=0.35``。
   测试 ``test_damping_limits_single_round_displacement`` 断言单轮位移不超上限。
5. **几何量收敛**：penetration_max/p95、contact_gap_error、翻转率不恶化、
   Margin/Occlusal ≈ 0；连续两轮无改善即停并报 REVIEW_REQUIRED。
6. **rollback**：候选先评估，只有字典序键 ``(penetration_max, gap_err_p95)``
   改善且翻转率未恶化才提交（``test_rollback_rejects_a_worsening_round``）。

**不上 IPC**：barrier potential / CCD 适合把 Contact+Occlusion 统一成严格碰撞
优化器，当前只是修局部接触，不引入（留待以后按需评估）。

## 顺带修掉的定向缺陷（用户指出的核心问题）

上一版 patch 法向用"多数查询点在法向正侧"投票定向 —— **对开放 patch 不成立**：
若冠整体陷入邻牙表面之后，多数票必然把法向翻到"看起来都在正侧"，
穿透检测直接失效（合成例实测报 penetration=0）。

改为 ``orient_against_crown``：用**冠自身的面积加权外法向**定向，判据是
"接触区里邻牙外法向与冠外法向应当相对（面对面）"。这不需要 patch 是封闭体，
也不受整体穿透影响 —— 冠的法向是自己网格的固有属性。

## 合成测试

`tests/unit/geometry/test_contact_refine.py` 9 项全过（连同 Contact v1 的 8 项，
geometry 共 52 项）。实测轨迹（邻牙在冠面外 0.5mm 穿透、目标间隙 0.3）：

```
it=0  pen=0.500  gap_err=0.800
it=1  pen=0.100  gap_err=0.400   accepted
it=2  pen=0.000  gap_err=0.200   accepted
it=3  pen=0.000  gap_err=0.294   REJECTED（rollback）
```

**穿透两轮清零**，Margin 位移 ~1e-11、咬合位移 0.03mm、零翻转。

## 真实病例：被 patch 质量卡住（本轮结论）

在同一病例上跑 I.1，活跃约束只剩 **1 个顶点**，穿透没被修。定位到根因，
**不是精修算法**，而是**邻牙 patch 的制备**：

审计脚本的 ROI 是 ``u ∈ [margin边 + 0.8, +13mm]`` —— 内边界正好切在
接触区起始处。于是冠的最近点几乎全部落在 **patch 的内切割边缘**上：

```
mesial patch: 12073 三角形 / 555 边界边 / 5 个面连通分量
              wide 内查询点 2621，其中 valid=1、invalid(边界歧义)=2620
              phi<target 的顶点 623，其中 valid 仅 1
```

边界歧义剔除（规格要点 3）**正确地工作了** —— 它告诉我们这个 patch 在
几乎整个接触区都不可用于有向距离。要修的是 patch：

* 用**区域生长**从"离冠最近的三角形"向外扩 N 环，保证接触区落在 patch 内部；
* 或直接取**整颗邻牙**（而不是盒子裁剪），接触区自然在内部；
* 或把 ROI 内边界外移一个安全带（但接触区本身就在内边界附近，效果有限）。

## 结论

I.1 的算法侧按规格完成并通过合成验证；真实例的下一步是**修 patch 制备**
（区域生长 / 整牙），然后 I.1 才能在真实例上把 1.18mm 的穿透压下去。
在此之前 Contact v1 的真实例结果保持 REVIEW_REQUIRED（残留穿透如实上报）。


---

## 邻牙 patch 制备修复（区域生长）——已做

新工具 `backend/tools/grow_neighbor_patch.py`：从"离冠最近的那个牙弓三角形"
出发做面邻接区域生长，只收面心到种子 ≤ R（默认 9mm）的三角形 —— 得到以
接触区为中心的圆盘状 patch，接触区落在内部、边界远离它。

修复前后（病例 `cf_b1b73bae221211b1`，用 I.1 的边界有效性判据度量）：

| | 边界边 | 查询点 | valid | 穿透顶点中被判有效的 |
|---|---:|---:|---:|---:|
| 盒子 ROI（旧） | 555 | 2621 | **1** | 1 |
| 区域生长（新） | 302 | 2621 | **1939** | 450 |

distal 侧同样：2990 / 3832 有效（旧版近 0）。**边界歧义剔除本身没错，
它正确地指出了旧 patch 在接触区不可用。**

## 用正确 patch 后才看得见的真实状况（诚实记录）

1. **真实穿透比之前测到的大**：v1 残留穿透 1.40 / 1.19mm（旧 patch 下只测到
   1.16 / 0.98，因为大部分穿透点落在边界安全带里没被计入）。
   H.1 的解剖模板冠在近远中面上确实比患者的邻间隙宽 1~2mm —— 这正是
   Contact Engine 要解决的问题，量级是对的。
2. **咬合被拖 1.08mm，违反规格的"Occlusal ≈ 0"**。根因是约束规模：
   正确 patch 下约束顶点从 ~700 涨到数千，咬合弱锚（权重 3 → 10 → 300，
   实测位移纹丝不动 0.992）在本尺度下无效 —— 硬约束行权重 1e5 完全压倒它。
   **试过把咬合也硬锁，但引入两个新问题**：合成例里咬合带与接触区重叠导致
   锁死无法修（penetration 恒 0.5）；真实例方阵路径变奇异（`splu`
   "Factor is exactly singular"），且去掉弱锚后超定/方阵两条路径的形状
   前提被打破。已回退，作为**已知未完成项**留待下一步（正确做法是让
   咬合锚随约束规模自适应，或对接触修正做区域上限）。
3. **I.1 在当前参数下推不动 2mm 级的修正**：α=0.4、单轮上限 0.35mm，
   两轮即无改善并被 rollback 正确拦下（pen 1.996 → 1.998，拒绝）。
   需要更大的单轮预算或更多轮，且必须先解决第 2 条的咬合拖动。
4. **求解器参数不能激进**：曾把硬行权重提到 1e5、lsqr 容差收到 1e-14 /
   20 万迭代，结果 **H.1 生成本身失败**（残差 1e90），而管线自己的 QC
   （`H.1 状态: FAILED` + 后续数字全乱）把它抓了出来。已回退到
   1e4 / 1e-12 / 60000。硬约束残差上限做成可配置
   （`CrownOuterConfig.max_constraint_residual_mm`：CrownOuter 路径 0.01
   实测 0.00188；I.1 路径 0.05，因其约束顶点多且带弱锚超定）。

## 当前状态

```
H.1                      OK
Contact v1               REVIEW_REQUIRED（咬合拖动 1.08mm + 残留穿透 1.40/1.19mm）
I.1 refinement           REVIEW_REQUIRED（两轮无改善，rollback 正确拦下）
```

**未达标项按规格是明确的**：Occlusal ≈ 0 与 penetration < 阈值都还没做到。
下一步按优先级：(a) 让对方块/硬约束规模自适应，把咬合位移压回 ≈0；
(b) 给 I.1 足够的单轮预算把 2mm 级修正做进去；(c) 之后再进 J Occlusion。

---

# Contact v2：局部带状求解（2026-10-02）

**目标（用户规格）**：邻接修复只能局部动 proximal surface，不能把 CrownOuter
尤其咬合面一起拖走。

## 根因链（本轮挖清的三层）

1. **注释与求解器语义不符**：`contact.py` 写"颈环 + 咬合区**都硬锁**"，但
   `solve_laplacian` 把 `occlusal_vertex_indices` 放进**弱锚块**（α=300 对硬行
   W=1e4 无效）—— 咬合从来没被锁住过。
2. **指标混淆**：真实例"咬合被拖 1.08mm"里有相当一部分是**接触修正本身**——
   磨牙邻接面在牙尖高度，按 z≥p75 定义的"咬合区"覆盖了邻接壁。修接触推这些
   顶点是合法的，被误读成拖动。
3. **咬合"表面"必须按法向定义**：`occ_table_mask = 高 z 且 |n̂·就位轴| ≥ 0.6`。
   径向法向的高 z 顶点是邻接壁，必须允许修正（否则接触区被锁死，残留穿透
   2.0mm 全在那里修不动）。

## 实现

- `laplacian.solve_local_band`：自由区 = 修改区外扩的过渡环（顶点 BFS）；
  颈环与咬合面是**固定边界**进右端项 —— 位移**由构造精确为 0**（不是弱锚）。
  方阵 splu 精确解，带对角正则。
- `contact.py` 主求解与 `contact_refine.py` 精修都默认走 local_band；
  v1 全局路径保留（`local_band=False`）做对照。
- **锁定区目标一律忽略**（fail-safe）：落在颈环/咬合面的修改顶点本迭代不动。
- 指标拆分：咬合**表面**位移（必须 0）与"接触修正波及的高 z 邻接壁"分开报。

## 真实例验收（cf_b1b73bae221211b1）

| 指标 | Contact v1 | **Contact v2** |
|---|---|---|
| 咬合表面位移 p95 | **1.08 mm**（规格违反） | **0.0（精确）** |
| 颈缘位移 | 1.56e-12 | **0.0（精确）** |
| 翻面 | 220 | 200 |
| 穿透残留 | ~2.0mm | ~2.0mm（未变，见下） |

## 已知未解（如实）

穿透残留 ~2.0mm 与 v1 相同 —— 这是**预先存在的 I.1 问题**（阻尼 + trust
region 0.35mm 推不动 2mm 级修正，两轮无改善即 rollback），本轮语义纯化后它
不再被咬合拖动混淆。消除它的候选路线（下个工作项 I.2）：
(a) 对邻牙 patch 建符号距离场后一次性投影（此前记录的路线①）；
(b) 迭代策略改为"步长自适应 + 收敛看穿透量"（现 rollback 规则在 2mm 级修正
下会过早停）；
(c) 检查穿透最深处是否在锁定区 —— 若在咬合面上，那是 Occlusion Engine 的
输入干涉，不是 Contact 能修的。

## 工程教训（本轮新踩）

- `set(np_bool_array)` 迭代出 `{True, False}` 而不是索引 —— 锁定排除静默失效
  （过渡环侵入颈环与咬合区，被 v2 专测抓住）。
- 把接触种子放进自由区又给它们设目标 —— 目标只作用于固定顶点，等于没有任何
  力把接触区拉向目标（穿透纹丝不动，合成测试抓到）。
- **修几何的补丁后必须看最上游阶段的状态**：I.1 的 occ=0 候选被拒，根因在
  Contact v1 主求解的指标/锁定语义，只看精修日志会误判为"精修失败"。

---

# I.2 归因诊断（2026-10-02）：2mm 残留穿透在哪里、为什么修不掉

**只诊断，未改求解器。** 工具 `tools/diag_contact_residual.py`，输出
`docs/contact_v2_residual_attribution.json`（逐顶点四分类，沿用 v2 的法向语义）。

## 归因表（Contact v2 输出网格，trust region = 0.35mm）

**mesial**（穿透顶点 2296，median 0.593 / p95 1.817 / max 1.950 mm）

| region | count | pen_sum | pen_max | req/trust p50 | p90 |
|---|---:|---:|---:|---:|---:|
| PROXIMAL_EDITABLE | 1568 | 838.96 | 1.948 | **0.82** | **4.52** |
| OCCLUSAL_LOCKED | 663 | 729.39 | 1.950 | 4.30 | 5.49 |
| CERVICAL_LOCKED | 65 | 35.63 | 1.940 | 0.73 | 4.37 |

**distal**（穿透顶点 2214，median 0.318 / p95 1.719 / max 1.950 mm）

| region | count | pen_sum | pen_max | req/trust p50 | p90 |
|---|---:|---:|---:|---:|---:|
| PROXIMAL_EDITABLE | 1845 | 1060.49 | 1.950 | **1.04** | **4.66** |
| OCCLUSAL_LOCKED | 329 | 214.32 | 1.931 | 1.74 | 4.71 |
| CERVICAL_LOCKED | 40 | 8.66 | 1.188 | 0.29 | 2.71 |

## 判定：B 为主（可修壁上步长永远不够），mesial 叠加显著的 A

- **B（CONTACT_SOLVER_LIMIT）成立**：distal 穿透总量的 **83%**（1060/1283）、
  mesial 的 53% 落在 PROXIMAL_EDITABLE；所需修正最大 **2.05mm = 5.9× trust
  step**，p90 达 4.5~4.7×。即使方向全对，0.35mm/轮 + "两轮无改善即 rollback"
  **数学上永远走不到** —— 与"方向错了"明确区分。
- **A（OCCLUSAL_CONFLICT）真实存在且偏侧**：mesial 有 663 个咬合面锁定顶点
  穿透（占该侧 pen_sum 46%），最深 1.95mm；distal 只有 17%。**这部分 Contact
  不许修** —— 应作为输入干涉移交 Occlusion Engine。
- **C（GEOMETRY/PATCH_ERROR）排除**：无 OTHER 类顶点，两侧 signed 距离有效、
  patch 法向定向正确。
- **CERVICAL_LOCKED 可忽略**（两侧合计 44mm·mm）。

## 一个必须上报的上游事实

穿透不是"小块接触干涉"：**约 2200 个顶点、整片近远中面中位 0.3~0.6mm、最深
1.95mm 地陷入邻牙** —— 即 H.1 冠的近远中宽度比邻间隙宽约 1~2mm（与早先
"H.1 模板冠近远中面比邻间隙宽 1~2mm"的观察一致）。这意味 Contact 要做的是
**整面 1~2mm 的近远中收窄**，属于大变形 —— I.2 的 SDF 一步投影正适合这个量级；
若最终仍越界，上游解法是让 CrownOuter 把邻间隙作为输入约束（设计决策，待定）。

## 对下一步的直接指令

1. **I.2 Solver 批准启动**（B 主导）：邻牙 patch 符号距离场 → 穿透顶点一次投影
   到 target_gap → 过渡环 Laplacian blend → **只接受 penetration 单调下降的
   step**（line search 替代固定 α/rollback）。
2. **A 部分移交**：mesial 咬合面干涉写入 Occlusion Engine 的输入清单
   （Contact 报告，不修）。
3. 硬不变量继续自动测试：咬合面/颈缘位移恒 0、锁定/自由不相交、翻面不恶化。

---

# I.2 Solver（2026-10-02）：SDF 投影 + local-band blend + 单调 line search

实现：`app/geometry/crown_outer/contact_i2.py`（`solve_proximal_projection`）。
范围照归因锁死：只修 PROXIMAL_EDITABLE；OCCLUSAL_LOCKED 记 handoff；颈环带永久锁定。

- SDF 定目标：editable 穿透顶点沿**邻牙 patch 法向**拉回 target_gap（不用冠自己
  的顶点法向）；**质心一次性定向**（冠质心必在邻牙体外，与穿透深度无关 ——
  I.1 时代的区域法向投票在"整面深陷"时必然翻错，实测首跑全 0 穿透的根因）。
- local-band Laplacian 只管形变传播/平滑；line search α ∈ {1, 1/2, …}，
  **接受条件 = editable p95 单调下降 + editable max 不恶化 + 锁定位移精确 0 +
  翻面不恶化**（第一个满足的最大 α）。
- 无固定 trust step（归因证明 0.35mm 永远走不到 2mm）；外层 ≤10 轮，
  停止 = 进 tolerance / 相对改善 < ε / 无 α 可行。
- 指标分账：editable / occlusal(handoff) / cervical 三本账分开报。

## 真实例验收（cf_b1b73bae221211b1，4 轮，α=0.5×4）

| 指标 | before | after |
|---|---:|---:|
| mesial editable p95 | 1.122 | **0.097** |
| distal editable p95 | 1.129 | **0.106** |
| mesial editable max | 1.249 | 0.137 |
| distal editable max | 1.287 | 0.221 |
| **occlusal locked（handoff）** | 1.387/1.353 | **1.387/1.353（原样保留）** |
| 咬合面/颈缘位移 | — | **0.0（精确）** |
| 翻面（相对 v2 输出） | — | 61（0.6%，集中接触区） |

handoff 账：mesial 257 顶点 / sum 179.3mm·mm；distal 209 / 155.0 —— **Occlusion
Engine 的输入干涉清单**。proximal_shrink：mesial max 0.83 / distal max 0.60mm
（大头收窄由 v2 的 local band 一步完成，I.2 收剩余）。

## Contact 冻结判定

剩余穿透**主体已是 OCCLUSAL_LOCKED**（editable 残留 0.10mm 级、接近 tolerance
0.05）—— 符合用户定下的冻结条件（"可编辑穿透基本消掉，剩余交给 Occlusion"）。
**Contact 到此冻结**：v2（局部带硬锁）+ I.2（SDF 投影 + line search）为最终形态。
已知项（不阻塞冻结，记录在案）：editable 残留 0.097/0.106 未进 0.05 tolerance
（line search 被翻面门 0.002 卡住，第 5 轮起全部 α 被拒）；翻面 61 面（0.6%）。

## 工程教训（本轮）

- **测量口径必须与 runner 一致**：我用 `trimesh.load(process=False)` 装邻牙 patch，
  STL 三角汤不焊接 → NeighborPatch 的边界检测按顶点索引计数 → 所有边都成"边界"
  → valid=0 → 穿透"全部消失"（假象）。runner 用 `force="mesh"`（默认焊接）。
  归因工具与 solver 必须共用同一加载口径。
- `g = wids; g[wids]` 是数组自索引（wids 长 177、值到 385）→ 越界且报错的
  "size 177" 指向 wids 自己 —— 与 editable 无关，极具误导性。
- 深穿透场景下法向定向不能靠区域投票，要用质心判据并**只做一次**。

---

# Occlusion v1 —— Collision Resolution Engine（2026-10-02）

实现：`app/geometry/crown_outer/occlusion.py`（`solve_occlusion_collision`）。
完全沿用 Contact v2/I.2 已验证的架构：对颌 SDF 定目标 → 只开放**法向判据的咬合面**
（occ_table）+ 面内过渡环 → local-band Laplacian → monotone line search。
**一切非咬合面顶点都在锁定集**（proximal core 由构造精确不动）。

- 对颌法向**质心一次性定向**（与 I.2 同款；逐轮投票会漂移）。
- **Contact 保持性复测**：每个被接受的步都重测近远中接触穿透 p95
  （只测不重跑 solver），恶化超容差（0.02mm）即拒绝该步。
- 空过渡环 = 刚性应用目标的退化情形（整片咬合面同时穿透），line search 兜底。

## 数据侧：antagonist patch 摘取（`tools/stage_antagonist_case.py`）

按 content hash 找 antagonist STL（361k 面）→ 同帧验证（GT 冠咬合区到对颌距离
p10=0.24 / p50=0.90 / p90=2.76mm，**PASS**——与 Contact handoff 的 ~1.4mm 干涉
吻合）→ ROI 裁剪（R=11.9mm，32836 面）→ 拉回本地。
踩坑：自写 STL writer 把头写成 84 字节零 + 面数（正确 = 80+4）→ 读取方读到
"0 面网格"。

## 真实例验收（cf_b1b73bae221211b1，链路 v2 → I.2 → Occlusion，9 轮）

| 层 | 结果 |
|---|---|
| **Occlusal** | 穿透顶点 **587 → 0**；p95 **1.187 → 0.000**；max 1.500 → 0.000 |
| Contact 保持 | mesial p95 0.779 → 0.751（改善）；distal 0.662 → 0.662（不动） |
| Hard locks | margin **0.0**、proximal core **0.0**（精确） |
| 位移 | 咬合面 p95 1.07 / max 1.59mm（咬合面下降量=干涉深度，合理） |
| Geometry | 翻面 247（**2.08%**，集中在新塑造的咬合区）；修改顶点 1064 |

## 已知项（如实）

1. **翻面 2.08%**：翻面门按 `base+0.002+0.004×it` 逐轮放宽（防止 line search
   卡死），9 轮累计到 2%。代价/收益明确：完全消除 1.5mm 咬合干涉 vs 2% 局部
   翻面。收紧会退回"消不完"。v1.1 候选方案：过渡环在咬合面内更宽 +
   翻面门固定 0.002 + 更多轮。
2. mesial contact gap min −1.321 → −0.886：咬合面下压时近中壁上部被
   transition 环轻微带动，**方向是改善**（穿透变浅），在容差内。

## 主线状态

```text
Contact v2 + I.2   ✅ 冻结
Occlusion v1       ✅ 碰撞消除达成（翻面细化 = v1.1 已知项）
↓ 下一步
Thickness Constraint Engine
↓ Outer + Intaglio（共享 cervical boundary）→ Final Stitch
↓ Watertight / Manifold QC → Final Crown STL
```

## Phase K —— Contact v2 / I.2 硬化（2026-10-08/09）

用户规格：**改接受规则，不改效果目标**。旧口径"penetration ↓ + 翻面别太多
（flip tolerance +0.2%）"作废，统一为：

```
ACCEPT(step) iff
  editable penetration p95 ↓  AND  max 不恶化
  AND 新增自交 = 0        （聚焦口径：只测本步可能动的面 ∪ 近邻，与形变前同口径相减）
  AND 非流形边不新增       （拓扑不变量；"绝对为 0"由阶段门负责）
  AND 翻面 = 0            （面积 ≥ (scale/2)² 才算 severe，与 J.2 的 micro 口径一致）
  AND margin 位移 = 0     AND 咬合锁定面位移 = 0（精确 0）
line search: α = 1, 1/2, 1/4, 1/8, 1/16；全失败 → CONTACT_CONSTRAINT_CONFLICT
```

落地：

* `deform_gate.py`（新）：形变步硬门，Contact/Occlusion 共用。
* `contact_i2.py`：α line search 保留，几何判据换成硬门；每步账进 `history`
  （含被拒绝的 trial 与原因），状态多一种 `CONSTRAINT_CONFLICT`。
* `contact.py`：新增 `harden_contacts()` —— v2 也走 α + 硬门；**冲突时返回原网格**
  （α=0 也是一次 Laplacian 求解，会漂 0.156mm，不能拿它当"没动"）。
* 管线 `stage_contact`：产出 `contact_hardened.stl` / `contact_report.json` /
  `geometry_qc_after_contact.json`；I.1 也要过不变量，不过就丢弃并记账。
* **Gate 纪律**：runner 里阶段给出 `gate.status != PASS` 就**停住**，下游一律不跑
  （`blocked_by_gate` + `blocked_at` + `blocked_reason` 落盘）。

### 真实例实测（`cf_b1b73bae221211b1`，2026-10-09）

| 项 | 值 |
|---|---|
| 几何 | 自交 0 / 非流形 0 / 翻面 0 / degenerate 0（**几何完全干净**） |
| margin 位移 | **0.0** |
| editable 残留 | p50 0.334 / **p95 1.025** / max 1.126 mm，**909 个顶点** |
| 残留去向 | occlusal handoff 0 / cervical handoff 0 / editable residual **909** |
| v2 硬化 | α=1:167 翻面+298 新自交；1/2:66+23；1/4:12+8；1/8:4+1；1/16: 门过但无改善 → **CONFLICT** |
| I.2 | 接受 α=0.125，0 翻面 |
| 咬合锁定面位移 | **0.0**（两条不变量都成立） |
| Gate | **CONTACT_CONSTRAINT_CONFLICT** → 下游 occluded / audit **未运行**（`blocked_by_gate`） |

**结论**：Contact v2 在这一例上"清穿透必然折叠"——H.1 解剖先验在近中/远中面
**整体宽出 ~1.0–1.1mm**（909 个可编辑顶点、p95 1.02mm），把 1mm 级收缩压在
单次局部带形变里，内部应变必然折叠（α=1 时 167 片）。这正是本项目自己在
`contact_i2` 里标注的 `UPSTREAM_CROWN_TOO_WIDTH` 情形。

下一步三选一（**需要决策，不自动选**）：
1. **上游修正**：在 H.1 放置/形变阶段就按邻牙间隙收缩（把 ~1mm 的收窄挪到
   anatomy prior 生成时，Contact 只做收尾）；
2. **改形变模型**：把"逐顶点沿各自法向"换成"接触区整体平移 + 带状混合"
   （区域内部零应变，只在过渡带应变）；
3. 放弃本阶段清干净，如实带 `CONTACT_CONSTRAINT_CONFLICT` 走 REVIEW 流程。

### 已定位并修掉的口径 bug（0.1559mm 假报警）

`occlusal table` 的判据依赖 **z 分位 + 法向**，两者都随形变移动 → 形变后重算的
掩码与形变前**不是同一套**。实测：`_zones(形变前)` 1297 个顶点、
`_zones(形变后)` **1305** 个（差 18 个），而那 18 个里有 13 个正是本轮被接触
修正动过的顶点 → 用"形变后集合"去量位移，就把它们算成"拖了咬合面"。

修法：报告与不变量都用**形变前**重算的锁定集。复算确认：形变前集合上的位移
**恒为 0.0000**（v2/i1/i2 每步开副本直测也全为 0）。现在 Gate 里两条位移不变量
`margin_preserved` / `occlusal_locked_preserved` 都是 **true**。

**教训**：这类"掩码随状态漂移"的判据，必须在**固定参考系**上算一次然后锁住——
这也是 `_solve_once` 里 `locked_ids` 要提前求好、以及各子步要传副本的原因。

### 遗留（如实记录）

* **I.1/I.2 仍会原地改写入参网格**：现已用副本隔离（各阶段独立 `_fresh`），
  但**源头未改**，后续接手时要注意。

---

## clearance 口径全项目统一（用户 2026-10-09 决策 ③）—— 测量 bugfix，不是 Gate 放宽

```
OLD:  raw signed query          残留 p95 <= 0.2
NEW:  robust unified signed     残留 p95 <= 0.2
```

**变的是测量 primitive；`0.2mm` 验收阈值一字未动** —— 不破坏冻结纪律。

唯一实现：`NeighborPatch.clearance_detail()`（`app/geometry/crown_outer/contact.py`）。
`NeighborClearanceQuery`（H.2.x / K0 / 工具链）只是它的薄包装；Contact v2（`_side_need`
+ verify）、I.1（`contact_refine._side_measure`）、I.2（`measure` 及其求解器全部
`measure` 调用）、Contact Gate 审计（`crown_outer_pipeline._contact_report`）
全部改走同一条。

口径定义：

* ``signed = sign(dot(p−c, n̂)) · euclid`` —— **幅值单规则用 euclid**
  （连续、单调）。`closest()` 返回的 dot 已含 ``normal_sign``（见其末行），
  绝不能二次乘 —— 合成夹具上双重翻转把 3.8mm 外侧读成 3.8mm 穿入（已修）。
* ``valid = 边界有效 & 法向可靠``：
  * ``AMBIGUOUS_PATCH_EDGE``：最近点落在开放 patch 边界安全带内（既有口径，0.6mm）；
  * ``AMBIGUOUS_CREASE``：``|dot|/euclid < 0.90`` —— 最近点落在三角形共享边/顶点，
    法向投影在 µm 级位移下可跳变 0.1mm 级（实测深穿 1.2mm 被读成 0.03mm，
    8.5µm 位移后跳到 −1.13mm）。
* **有效性掩码在固定参考几何上一次冻结**（用户纪律），不随形变漂移：
  H.2.x 冻在 H.1 基线；I.2/v2/Gate 审计冻在各阶段输入网格（`vertices_ref`）。

旧口径（raw dot）在真实数据上的偏差：H.1 的 mesial 穿透 p95 由
**1.0359 → 1.0827mm**（K0 审计口径，run `h22m_chain`）；冻结域 deficit p95
由 **0.9711 → 1.0728mm**。即此前读数**偏浅**。

