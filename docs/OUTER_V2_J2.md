# J.2 —— Level-Set Envelope（隐式包络 + 体融合）

**状态**：**融合语义 ✓ / 提取质量 ✗**。覆盖失败被修掉了（内表面侧违反 28.1% → 7.1%、
负壁厚 0），但 marching cubes 的输出还不干净（非流形 40041、自交 62、颈缘边界没落到
margin 上）→ **几何安全门如实判 FAILED**。下一提交解决提取，不改融合语义。

用户 2026-10-02 锁定的两条工程决策（本模块完全照做）：

```
1. Minimum Envelope 用隐式场整体生成，不再逐顶点修 H.1。
2. Anatomy 与 Envelope 做体/隐式融合：已满足约束的 anatomy 保留，
   违反区域由 envelope 接管。
```

## 0. 先接受用户对 J.1 措辞的更正

> **"等距面按定义不可能自交"不能作为生产保证。**

正确。隐式 level set 只是避免了 J.1 那种"顶点各自乱跑把三角网折起来"的机制；
当 offset 超过局部 feature size、碰到 medial axis、`T(x)` 空间变化、或网格分辨率不足时，
仍会出现奇点、拓扑变化、非流形提取。**所以 `geometry_safety_gate` 照常必跑**，
本模块的 `res.safety` 就是它 —— 而且正是它把本版判成了失败，没有放过去。

## 1. 场（这一步是本次的核心语义）

```
s_int(x)  = 到"闭合 intaglio 实体"的带符号距离（负=预备体内）
φ_A(x)    = 到"闭合 anatomy 实体"的带符号距离（负=冠材料内）
T_ext(x)  = T_required(最近内表面点)
F(x)      = min( φ_A(x), s_int(x) − T_ext(x) )      # ← 融合
F = +1e6  where(x 在颈缘平面之下)
F(x)=0 → 外表面
```

**为什么是 `min`（并集）**：材料 = anatomy 已经够厚的部分 ∪ 包络要求的部分 ——
谁在外面听谁的；违反处自动由包络接管，**不再有 1.6mm 的顶点级大位移**。

**为什么不再叠 `max(−s_int, ·)`**：那会把"预备体之外"也做成一个 0-交线，于是
marching cubes 吐出"外表面 ∪ 贴着预备体的那一层 ∪ 颈缘刀口 slit"（实测 202k 面、
边界顶点 20k、自交 1786、边界环落在离 margin 4mm 的颊侧）。去掉之后，预备体内部
两项都为负 → **那里没有 0-交线** → 0-水平集**恰好就是外表面**：anatomy 保留区与
包络接管区沿 `s_int = T ∩ φ_A = 0` 这条干净的曲线相接，颈缘处 `T→0` 自然收到
共享边界上。

**`T(x)` 不是常数（用户要求不能假装知道过渡形态）**：

```
T_required(d) = near · min(d/d0, 1)                                   d < d0
              = near + (full−near)·clip((d−d0)/(full_dist−d0), 0, 1)  d ≥ d0
```

`d` = 到颈缘曲线的距离，`d0 = margin_blend_mm`（参数）。**颈缘上 T=0** —— 这是
"共享颈缘边界"的几何事实（两面同一条曲线），也是包络不会在颈缘外扩的原因。
构造侧另加 `safety_mm`（默认 0.05）保证 ≥ Thickness Audit 的判定规则；
`linear`/`step` 两种读法沿用 `ThicknessParameters.near_margin_mode`，
**provenance 显式**，不偷偷冻结一种解释。

## 2. 实现链（与用户给的流程图一一对应）

| 步骤 | 实现 |
|---|---|
| Narrow-band SDF | 闭合实体 → 表面体素化 → `distance_transform_edt` → 符号用**连通域标记**（膨胀表面做障壁，取含种子实体内点的分量）——不用射线奇偶，对体素化漏缝稳健 |
| `F = s_int − T_ext` | `T_ext` = 网格点上按最近内表面**顶点**查规则值（KD-tree，向量化） |
| Envelope 与 anatomy 融合 | `min(φ_A, s_int − T_ext)` |
| 提取 | `skimage.measure.marching_cubes(F, 0.0)` |
| Cervical boundary 恢复 | 把提取面的边界顶点吸到颈缘曲线上（密采样 polyline，容差可配；超容差的**如实计数不硬吸**） |
| Geometry Safety Gate | `geometry_safety_gate` + 闭合体积符号 |

## 3. 真实例结果（cf_b1b73bae221211b1，网格 117×122×109 @0.12mm = 1.56M 格）

**内表面侧覆盖（验收口径；`d_margin ≥ 0.6` 的 6636 个采样点，沿外法向射线测）**：

| | H.1 anatomy | **J.2** |
|---|---:|---:|
| `t ≥ T` 的比例 | 71.9% | **92.9%** |
| 违反点数 | 1874 | **473（7.1%）** |
| 负壁厚（`t ≤ 0`） | 0 | **0** |
| `t` 分布 p05 / p50 | — | 0.589 / 0.851 |

**融合账**：包络接管 64742 格、anatomy 保留 34971 格；`frac_F_negative = 18.2%`。

**解剖偏差（用户要求的分区报告，不是全冠 RMS）**：

| 区 | n | median | p95 | max |
|---|---:|---:|---:|---:|
| `ANATOMY_ALREADY_VALID` | 5338 | −0.149mm | 0.121 | 1.819 |
| `ENVELOPE_REPAIRED` | 2063 | **+0.295mm** | **0.578** | 3.075 |

（`ENVELOPE_REPAIRED` 的数字是合理的 —— 要补的就是这个量级；`ALREADY_VALID`
区的 −0.149mm 偏离目前**不可信**：提取出来的网格本身不干净，射线会打到垃圾面上，
等提取干净后必须复测。这也是"分区报告"的价值：不混在一起才看得出哪一区可疑。）

## 4. 验收表（用户定的八项）

| 项 | 目标 | 实测 | 判定 |
|---|---|---|---|
| Outer 落进 Intaglio/Prep | 0 | 射线口径**负壁厚 0** | ✅ |
| Intaglio-side envelope violation | ≈0 / 明确容差 | 28.1% → **7.1%**（残留来自 0.12mm 体素） | ⚠️ 大幅改善未达标 |
| Cervical loop displacement | 0.0 | 边界环没落在 margin 上（到 margin p50 **4.05mm**） | ❌ |
| 新增 self-intersection | 0 | **62** | ❌ |
| 非流形 | 0 | **40041** | ❌ |
| 大面积 flipped orientation | 0 | 绕向不一致边 **3469** | ❌ |
| 原本已满足厚度的 anatomy | 尽量不动 | median −0.149mm（**受网格垃圾影响，待复测**） | ⚠️ |
| 解剖偏差分区报告 | 分 valid / repaired | ✅ 已实现 | ✅ |

**结论**：**融合语义是对的**（覆盖失败确实被修掉、包络按需接管、颈缘处 T→0 自然收边），
**卡在提取质量**：marching cubes 在 `T→0` 的颈缘薄楔 + 体素化 EDT 的噪声上产生了
非流形与自交，边界环也因此没有落在 margin 上。安全门如实拒绝 —— 这次没有"用折叠
换进度"的空间。

## 5. 下一提交（提取，不动融合）

1. **窄带精确 SDF 替代 EDT 采样**：对网格点在窄带内做精确点-三角距离（复用
   `SurfaceQuery`），只在窄带外才用 EDT —— 现在整场都是采样栅格化的 EDT，
   voxel 级噪声直接进了 0-水平集；
2. **颈缘处理**：把 `T→0` 的薄楔换成"颈缘带内直接取 intaglio 的边界曲线"，即
   在提取前把颈缘带（`d < d0`）赋成解剖侧的场，避免薄楔；
3. **提取后处理**：保形重网格（isotropic remesh）+ 边界顶点吸附到 margin；
4. 之后**必须**重跑：`geometry_safety_gate`（自交/非流形/绕向/边界）+ Thickness
   Audit 的完整八项，尤其复测 `ANATOMY_ALREADY_VALID` 区的位移。
