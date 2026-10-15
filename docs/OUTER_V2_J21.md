# J.2.1 —— Fixed Cervical Collar + Implicit Body + Conforming Stitch

**状态**：**collar ✓（margin 拓扑级精确）/ body ✗（仍被 d1 截断切成两块）**。
安全门与三项新拓扑 QC 如实 fail loud，没有把失败磨平。

用户 2026-10-05 规格（J.2 冻结为"融合语义✓ / 提取质量✗"之后）：
**A. fixed cervical collar（来自 H.1，margin 精确保留） + B. implicit body（Anatomy ∪
Envelope 的 level set），在 d_margin = d1 的稳定截面 conforming stitch。**

## 1. 场：按区域切换（不再让 T 硬降到 0 的薄楔进 MC）

```
d_margin <= d0 (0.30) :  F = φ_anatomy
d0 < d < d1 (0.60)    :  F = blend(φ_anatomy, min(φ_anatomy, s_int − T))
d_margin >= d1        :  F = min(φ_anatomy, s_int − T)
```

`d1` 的选取有一条**实证约束**：Audit 只判 `d_margin ≥ 0.6` 的采样点，而过渡带里
包络只按比例生效 —— `d1=1.0` 时 0.6~1.0 段只被部分修正，覆盖指标反而退回 72.8%
（实测）。故 `d1 = 0.6`（= Audit 的起判点）。

## 2. 本轮修掉的一个**共享代码真 bug**（J.2 也受影响）

`_signed_field` 的符号用"连通域标记"，但**被膨胀掩码排除的表面带（lab==0）默认
被判成"外部"** → 实体内表面附近出现一层"正号壳" → 窄带精确重算把这个错的符号
一起带进来 → 场里满是小孔洞 → marching cubes 吐出成百上千个碎片组件
（**实测 components=1753**）。修法：用 `distance_transform_edt(..., return_indices=True)`
把表面带的标签用**最近有标签体素**补满。

## 3. 结果（真实例 cf_b1b73bae221211b1，网格 117×122×109 @0.12mm）

### A. collar —— ✓ 达到规格要求

| 项 | 值 |
|---|---|
| collar 面数 / 新增顶点 | 1035 / 276（裁剪插值点） |
| 底环 = **prescribed margin 原顶点** | **true**（146 个原顶点） |
| 底环到 margin 曲线 p95 / max | **0.023 / 0.029 mm** |
| 截面环（d=d1）顶点数 | 325 |

即：**最终 Margin 边界就是原 anatomy 的 boundary 顶点**（不是"MC 生成一圈再吸附"），
用户要求的"拓扑上就是它"在 collar 这半边已经成立。

### B. body —— ✗ 仍需下一轮

| 项 | J.2 | J.2.1（本轮） |
|---|---:|---:|
| 体组件（原始） | — | **1753 → 98**（去垃圾后 1 主件 + 2 碎片） |
| 非流形边 | 40041 | **26** |
| 绕向不一致边 | 3469 | **1** |
| 自交 | 62 | 897 |
| 边界环 | 282 | 38 |
| 最终组件数 | — | **3**（collar + 主体 56245 面 + 2 面碎片） |

**关键问题**：`d_margin ≥ d1` 的**面级过滤**把 body 切成了**两块大块**
（56245 + 29012 面）；"只留最大组件"于是丢掉了 **152.7mm²** 的真实表面 ✗ ——
这也是覆盖指标从 J.2 的 92.9% 退到 81.1% 的原因（被丢掉的那块里就有包络接管区）。
**数字现在不可信，正如用户提醒的：clean mesh 之前不要解释这些百分比。**

### 缝合 —— 机制 ✓

body 截面环 483 个顶点 → 全部吸附到 collar 截面曲线（max 0.42mm ≤ tol 0.6），
三角形带已生成；只是它接上的是一个"缺了一块"的 body。

## 4. 三项新 QC（用户要求）

| 新 QC | 目标 | 实测 |
|---|---|---|
| connected components | 1 | **3** ❌ |
| boundary loops | 1 | **38** ❌ |
| prescribed-margin ↔ mesh boundary | 一一对应 | **collar 底环 = 146 个原顶点 ✓**（主边界环 325 顶点那个是 body 的截面环，不是 margin） |

## 5. 下一轮（只动 body 的裁切方式，不动融合公式/厚度场）

1. **把 `d_margin ≥ d1` 从"面级过滤"改成"提取前掩码"**：在 MC 之前令
   `F(d < d1) = +BIG`，让 level set 在腰带处**干净收口成一个环**（现在是在已提取的
   面上按面心切，切出了两条以上闭合曲线 → 两块大块 + 一堆杂环）；
2. 提取后再**只取含截面环的那个组件**（并按"是否含 d≈d1 的环"判定，而不是按面积），
   避免丢真实表面；
3. 重跑八项 + 三项；**在此之前不解释 coverage/valid-anatomy 的百分比**；
4. 若 1~3 后仍是 5~7% envelope violation，再谈场分辨率 / adaptive octree / 精确
   offset surface；若拓扑仍不干净，转用户提的 **J.2.1-B challenger**：
   `manifold3d` 精确 Boolean UNION + fixed cervical collar（环境里已装 manifold3d）。


---

## 6. 追加一轮（提取前掩码 + 按截面环选组件）—— 仍不干净，触发用户预设的切换条件

改动：`d_margin ≥ d1` 从"提取后面级过滤"改成"**提取前掩码** `F(d<d1)=+1e6`"；
组件选择从"按面积"改成"**按是否含截面环**"（用户上一轮指定的判据）。

| 指标 | 面级过滤 | **提取前掩码** |
|---|---:|---:|
| coverage `t≥T` | 81.1% | **88.5%**（仍不可信，见下） |
| body 原始组件 | 98 | 2608 |
| 保留组件的腰带顶点 | — | 4488（选对了：含腰带的那块） |
| 非流形边 | 26 | 24 |
| 自交 | 897 | **260** |
| 边界环 | 38 | **59** |
| body 截面环 | 483 | **6** ← 症状 |

**症状读数**：掩码把 `F` 在腰带处从负直接跳到 `+1e6`，于是 MC 提取到的是
**"碗 + 腰带墙"的闭合面**（不是开口补片）——所以 body 的边界环只剩 6 个顶点，
缝合成了一条垃圾带（6 顶点环 ↔ 325 顶点环）。**这是掩码法的结构性副作用**，
不是参数问题。

**判定（按用户上一轮预设的条件）**：连续两轮都在打扫同一件事——
**隐式提取出不干净的 body**（59 个边界环、260 处自交）——
用户明确写过："**如果连 clean topology 都做不到，就不要继续堆 voxel resolution，
直接转 `manifold3d` Boolean UNION + fixed cervical collar**"。条件已满足。

collar 侧不受影响（146 个原顶点、p95 0.023mm 稳定），
`manifold3d` 路线会**保留**它：Boolean 只负责 body 的 `Anatomy ∪ Envelope`，
collar 仍是裁剪出来的固定 patch，两者仍然在 `d=d1` 截面 conforming stitch。

---

## 7. J.2.1-B —— Closed Envelope + Boolean UNION（提交见 git log）

用户 2026-10-08 的调整（采纳并实现）：

* **开放 body 不再交给隐式提取**：隐式场只生成**完整闭合**的 Minimum-Envelope
  solid；开放 body 由 Boolean 之后**真三角裁剪**得到（避开上一轮
  `F(d<d1)=+1e6 → 碗+腰带墙`）。
* cervical 以下是**牺牲闭合区**：`T` 不再降到 0 的刀口，取
  `closure_floor_mm = 0.20` 把壳体封死；只要求 `d ≥ d1` 的 envelope 保持原 `T(x)`。
* 两个 operand **各自过 preflight**；seam 周围设 anatomy-only **guard band**；
  stitch 前做 **ring compatibility gate** 并按**弧长重采样**；每个面打
  **来源标签**（ANATOMY_RETAINED / ENVELOPE_DOMINATED / COLLAR_FIXED /
  BOOLEAN_SEAM）。

### 已通过的（这一段是真正的进展）

| 项 | 结果 |
|---|---|
| Envelope operand | **closed ✓ manifold ✓ 朝向一致 ✓ 自交 0 ✓ 体积 +160.59mm³ ✓** ——"完整闭合体"这条调整生效，刀口问题消失 |
| Anatomy operand | **全过 ✓**（体积 +499.69mm³）—— 但第一轮它 `closed=false`：`_closed` 拿**排序后的顶点集**去扇封盖，自交/扭转（J.2 只当栅格化障壁用，所以没暴露）；改成**有序环扇 + 按体积定向**后才合法 |
| Boolean UNION | manifold3d **0.3s**，closed/manifold/朝向/体积全过；唯一"自交"是 **1 对微面**（0.0007 / 0.00015mm²，法向夹角 dot=0.94，间距 24µm）→ 按"面积阈值 = (体素/2)²"判为 **micro_defect**，**照常计数上抛**不隐藏 |
| Collar | 仍是 146 原始 margin 顶点 / p95 **0.023mm**（Boolean 完全不碰它） |
| 组件裁剪 | 6 件 → **1 件**（丢弃 5 个与 collar 不相连的碎块 / 4968 面） |

### 两个仍然卡住的（都有量化值）

1. **boundary_loops = 2**（938 + 542 顶点）—— 丢弃碎块后 body 仍多一个环。
   深锥化牺牲封盖把它从 855 压到 542（封盖原本会顶到 seam 之上），
   剩下的一个是 envelope 在**冠足迹之外**（颈缘以下/邻牙侧）的壳面被裁在
   `d ≥ d1` 后残留。**下一刀要按"是否在冠足迹内"给 envelope operand 收边**，
   而不是继续加裁剪。
2. **seam 台阶**：guard band p95 **0.281mm** / max 0.452mm（深锥化后从 0.508 降下来）
   —— envelope 在 seam 附近**本来就该接管**（该处 anatomy 也不达标），代价就是台阶；
   台阶能不能缝由 ring gate 决定，而不是在这里硬失败。**根治办法是把 seam 放到
   "anatomy 已达标"的位置**（本例 anatomy 大面积不达标 → seam 得往上抬，
   或者接受台阶并明确记录）。

### 结论

Boolean 路线的价值已验证：**volume topology 交给成熟 solid kernel 是对的**
（operand 全过、union 0.3s 精确、微缺陷可证据化）。现在挡路的不是"求 min() 的精度"，
而是 **envelope 该在冠足迹之外收边** 和 **seam 位置**这两件几何决策 ——
下一轮只动这两处，不改融合语义、不改厚度场。

## 8. J.2.1-B 收口轮 —— 第一个拓扑合法的 CrownOuter（提交见 git log）

用户 2026-10-08 规格里的"下一刀"是两件几何决策：**envelope 在冠足迹内收边** +
**seam 位置规则**。实测下来，挡路的其实是四个**真 bug/真机制**，逐个解决后才谈得上收边：

### 8.1 修掉的（都是可复现的真问题）

| # | 问题 | 证据 | 修法 |
|---|---|---|---|
| 1 | `_clip_by_scalar` 的 `n_in==2` 分支多补了一个三角形 `(m01,m20,p0)` | 干净 icosphere + 平面裁剪本该只有一圈纯插值点，实测 222 条边界边里 **128 条 mixed**（原顶点—插值点）→ 环数虚高 | 删掉该面：截面弦 `(m01,m20)` 不再被消费两次，边界边走插值点。修后同测试 **190 条全 nn、0 mixed** |
| 2 | **envelope operand 是"埋在 A 里的薄壳"** | `union(A,B)` 变成 **7~8 件**，6 件负体积（空腔）；winding 探针显示那些"空腔"内部其实在 B 里 → 核输出是错的 | **内腔填实**：B = `{s_int ≤ T_used}`（去掉 `s_int ≥ 0`）。A（已闭合的实心冠）本来就覆盖 intaglio 内侧，集合语义不变，但 B 从薄壳变成实心块 |
| 3 | seam 处 packing：collar 取 anatomy、body 取 union，**两层在同一张曲面上各自三角化** | collar 单独 0 / body 单独 0 / 合起来 **137**（浅叠） | collar 与 body 的切面**错开** `seam_gap = 0.10mm`，缝由 zig-zag 带承担 |
| 4 | 缝合把 strip 建在**重采样副本**上 | 最终网格 **3 件 / 5 个边界环 / 自交 1347**，主环不再是 margin | 缝合直接用两条环的**原顶点**（消费两侧全部顶点，无 T 接头） |

### 8.2 收敛后的实测（真实例 `cf_b1b73bae221211b1`，用户指定的验收表）

| QC | 要求 | 实测 |
|---|---|---|
| connected components | 1 | **1** |
| boundary loops | 1 | **1** |
| boundary = prescribed Margin | 接近 collar 精度 | **p95 0.0234 / max 0.0292 mm；146 个原 margin 顶点** |
| non-manifold edges | 0 | **0** |
| orientation inconsistent | 0 | **0** |
| self-intersections | 0 | **5（severe 1 / micro 4）** ✗ 仍差一口 |
| degenerate faces | — | **0** |
| negative coverage | 0 | **0** |
| envelope violation（ramp 外） | ≈数值容差 | **0** |
| envelope violation（ramp band 内） | 记录 | **62**（如实计入，交 Thickness Repair） |
| 来源面积占比 | 不是"半颗牙都被重建" | ANATOMY_RETAINED **55.7%** / ENVELOPE_DOMINATED **32.9%** / BOOLEAN_SEAM 11.4% |

### 8.3 关键机制：颈缘 guard band 的 **ramp**

`d ≤ env_start(0.50)` 不放 envelope 材料，`env_start → d1+guard(0.85)` 线性升到满 `T(x)`。
理由是实测的：anatomy 在 `d∈[0.6,0.8)` 的厚度 p05 0.208 / 最大亏 0.352mm，
若让满 T 在 d1 处直接生效，沿颈缘会顶出一圈 **0.1~0.45mm 的台阶**（不可缝；
collar ring 与 body ring 的弧长质心差 1.18mm 就是这么来的）。
改 ramp 后：**seam 区（d ≤ d1）p50 0.000 / p95 0.0022 / max 0.077mm**
（union = anatomy，两环同源，缝连续）；代价是 `d∈[0.6,0.85]` 的包络修正按比例推迟
（62 个采样点，band 之外 0 点）。覆盖 `t≥T` 比例 **99.04%**，`t_min 0.253mm`。

### 8.4 仍差的一口（如实记录，未磨平）

* **5 对自交（1 对 severe）**，全部落在 zig-zag 缝带自身：两侧环顶点数 276 ↔ 781，
  缝带高度只有 0.10mm，长条三角在局部折起来。试过两条路：
  - `seam_gap 0.15` → ring gate 的 Hausdorff 0.351 > 0.30 直接不合格；
  - 把稀疏环**共形加密**（边界边上插点：`_conforming_ring` / `_split_edge`，
    已实现并单元验证：拆一条边界边 → 边界面数 +1、边界边 +1、无 T 接头）
    → 拓扑没坏，但自交反而 5 → 18（加密后的重采样与原顶点分布不匹配）。
  两条都如实回退。**下一步**：把加密做成"以重采样环为唯一边界"（现在环是"原顶点
  ∪ 插值点"，所以加密不减少折叠），或者对缝带做局部翻转修正。
* status 因此是 **FAILED（只由那 1 对 severe 触发）**，其余全过 —— 没有为了让
  表变绿去调阈值；`micro/severe` 是**逐对**判的（两个面都 ≥ (体素/2)² 且法向不
  近平行才算真交叉），证据（面积分位数、法向角中位数）一并上抛。

### 8.5 这一轮的方法论教训（写给下一次）

1. **"算法没问题、只是参数没调好"之前，先把网格工具的局部算子验证一遍**
   （icosphere + 平面/距离场裁剪），`_clip_by_scalar` 那个假边界边就是这一步抓出来的。
2. **外挂内核的 output 必须用自己的口径复算**：manifold3d 返回 `status=NoError`、
   genus 正常、体积正常，结果却是错的（7 件、含负体积空腔）；用质点 winding 一测
   就露馅。**不要用 kernel 的自我报告替代几何证据。**
3. **同一条曲面的两种三角化不能各自占用同一段空间**（seam 的 137 对浅叠）——
   要么错开切面，要么共享网格。

## 9. J.2.1-C —— Seam Topology Repair：self-intersection 5 → **0**

用户 2026-10-08 规格：**不要再碰 Boolean / Envelope / collar / ramp / seam_gap**，
只修 seam 自身；先输出那一对 severe，再局部 edge flip，不行才上 DP。已按此执行。

### 9.1 那一对 severe 的身份（先取证）

```
face_a 46313  area 0.03507 m²  ← 长条（collar 侧疏采样的长边）
face_b 46316  area 0.00432 mm²
interior: abs_normal_dot 0.918（23° 夹角，真折叠）
centroid_gap 0.0349 mm，交线段长 0.0771 mm
lattice 步：~916 与 ~919（两者相隔 3 步 → 单四边形翻转够不着）
```

**关键取证**：缺陷附近两条环**没有互穿** —— 最近顶点距 0.0857 mm、缝带高度
p50 0.1082 / min 0.0797 mm。⇒ 这不是"环的空间关系"问题，**纯粹是三角化选择**
问题 ⇒ DP 可以根治（若是环互穿，任何路径都救不了）。

### 9.2 Phase 1：局部 edge flip（保留）

把缝带看成 **lattice 路径**（`path[k] ∈ {'a','b'}` = 这一小步推进哪一侧）：
**交换相邻两步 = 翻转该四边形的对角线** —— 纯局部、可逆、**不动任何顶点**。
实测：翻转 step 572 → 自交 5→4（severe 仍是 1）；缺陷主场（915/917/922）
三个候选翻转全部无效（severe 1 不变、total 反而 6/5/7）—— 与 9.1 的
"相隔 3 步"一致。Phase 1 保留为低成本首选。

### 9.3 Phase 2：DP annulus triangulation（这轮的决定性一刀）

状态 `(i, j, s)`（s = 上一小步推的是 A 还是 B），转移只有 `a`/`b` 两种
→ **天然无 T-junction**；转移代价 = 形状(1−归一化质量) + 长边惩罚
+ 法向不连续惩罚，**法向与上一片反号直接取 ∞（就是折叠）**。
回溯 (0,0) → (nA,nB) 得全局最优路径。**不新增、不移动任何顶点。**

实测（真实例，同一条 seam）：

| | greedy zig-zag | Phase1 flip | **DP (Phase 2)** |
|---|---:|---:|---:|
| hits（severe + micro） | 5（1 + 4） | 4（1 + 3） | **0（0 + 0）** |
| degenerate faces | 0 | 0 | **0** |

合成环自检（40 ↔ 121 点、不同半径的同心环）：路径长度严格 nA+nB、
**边界边恰好 nA+nB = 两条环本身**（即缝与两侧网格共形、无多余边界）、
面积 min/max 0.0064/0.0177 mm²（无退化）。

### 9.4 验收（口径与上一轮完全一致，没有换）

| QC | 要求 | 实测 |
|---|---|---|
| connected components | 1 | **1** |
| boundary loops | 1 | **1** |
| boundary = prescribed Margin | 接近 collar 精度 | **p95 0.0234 / max 0.0292 mm；146 个原 margin 顶点** |
| non-manifold edges | 0 | **0** |
| orientation inconsistent | 0 | **0** |
| self-intersections | **0（含 micro）** | **0** ✅ |
| degenerate faces | 0 | **0** |
| negative coverage | 0 | **0** |
| envelope violation（ramp 外） | ≈数值容差 | **0** |
| envelope violation（ramp band 内） | 记录 | 62（推迟给 Thickness Repair） |

来源分布与偏差分账**未被 seam repair 破坏**：
ANATOMY_RETAINED 55.7% / ENVELOPE_DOMINATED 32.9% / BOOLEAN_SEAM 11.4%；
ANATOMY_ALREADY_VALID `abs_p50 = 0.0mm`；ENVELOPE_REPAIRED n=2079
（signed_median 0.214 / p95 0.596mm = 实际加厚的量）；seam 区 union−anatomy
p50 0.000 / p95 0.0022 / max 0.077mm；coverage `t≥T` 99.04%，t_min 0.265mm。

> 说明：`ENVELOPE_REPAIRED` 的 signed_median 由 0.030 变 0.214，是**缝带三角化
> 换了**导致少数从 intaglio 出发的射线先打到缝带（缝带是真网格面），不是几何回归；
> 覆盖/负覆盖/t_min 三项均一致或更好（t_min 0.253 → 0.265）。

### 9.5 结论：CrownOuter J.2 冻结

满足 `Anatomy ∪ MinThickness` + Intaglio 覆盖 + prescribed Margin + 合法拓扑的
第一版 Outer。**停止继续优化 Outer**，主线切到 Contact / Occlusion 硬化：

```
penetration ↓  AND  new self-intersection = 0
AND orientation valid  AND non-manifold = 0
```
（不允许再出现"为了把 penetration 清零而换 2% 翻面"。）

### J.2.1-C 补丁：seam repair 的确定性收敛（用户 2026-10-09）

**问题**：同一 frozen J.2 对另一份 anatomy 输入，seam repair 从 5→0 变成 **6→1**
（severe 0 / micro 1），QC 按冻结标准应判 `J2_GEOMETRY_QC_FAIL`（不允许"1 个 micro
可以放过"）。根因：Phase 1 只试**前 6 个**候选翻转（`pos[:6]`），Phase 2 的 DP
目标是形状质量（形状/长边/法向代价），两者都不保证 `total → 0`。

**修法（确定性鲁棒性修复，非新算法方向）**：新增 **Phase 3 残差爬山环** ——
从当前最优点出发，围绕**残余命中**位置穷举相邻步交换（每轮上限 40 次试算），
按 `(severe, total)` 严格改进接受并重算残余，直到 0 或无改进。仍是"只翻对角线、
一个顶点不动"的同一机制。

**验证**：同输入重跑 `j2fix_1` → `self_intersections 6→0`（severe 1→0），
其余 QC 全绿（1 component / 1 loop / 边界 p95 0.0234mm / 非流形 0 / 无负覆盖）。
`best_key[1] > 0` 才进入 Phase 3 ⇒ 原本就干净的输入路径**逐位不变**（无回归面）。

