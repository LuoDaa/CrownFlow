# NeighborGeometryAvailability —— 邻牙几何可用性审计（提交 I 前置）

**日期**：2026-09-29。结论先行：**18 个 FDI46 病例中，只有 1 例
（`cf_b1b73bae221211b1`）双侧邻牙局部表面可用**；13 例语料里根本没有牙弓
扫描，4 例有候选模型但不同坐标系/缺目标牙位。审计工具
`backend/tools/audit_neighbor_geometry.py`（纯 stdlib，服务器跑），
逐例明细表 `neighbor_geometry_audit.csv`（随仓库提交，无 PHI）。

## 审计设计（用户规格，不分割整颗 45/47）

1. **同坐标系证明**：margin 点 → 牙弓顶点距离分布；对 sectioned 模型
   （46 预备体被抽走、margin 悬在插座上方）改用 **GT 冠 → 牙弓**的
   "贴附比例"（冠采样点距牙弓 < 0.8mm 的比例 ≥ 2%）——冠与 margin 同帧、
   且物理上坐在牙弓里，这是更可靠的证据。
2. **患者系裁 ROI**：Origin=margin 质心，X=AxisMesial、Y=Z×X、Z=就位轴；
   每侧 u ∈ [margin边+0.8, +13mm]，v/w 由该例 margin 尺寸导出。
   **ROI 尺寸没有写死成临床常量**。
3. **每侧状态**：AVAILABLE / MISSING_TOOTH / SECTIONED_OR_HOLE / AMBIGUOUS /
   OUT_OF_FRAME。判别"插座 vs 邻牙"靠**角向覆盖**：插座壁贴着 margin 圈、
   覆盖 ~360°；真邻牙只占一段弧。
4. **GT 反求**：Final Crown ↔ 邻接 ROI 的最小距离 / 接触点数 / 接触面积
   估计 / 接触质心；XML 设计参数（`DistanceToNeighbor`）一并列出。

## 结果

| 类别 | 数量 |
|---|---:|
| 双侧邻牙可用（eligible = BOTH） | **1** |
| 单侧可用 | 0 |
| 完全不可用 | 17 |
| ↳ 语料无任何牙弓扫描（所有 STL 都是单牙/冠/代型） | 13 |
| ↳ 有候选模型但 OUT_OF_FRAME / MISSING | 4 |

可用例 `cf_b1b73bae221211b1`（arch = index 绑定资产，crown 贴附比例 17.3%）：

| 侧 | 状态 | ROI 点数 | 到 margin 最近 | GT 反求 |
|---|---|---:|---:|---|
| mesial | AVAILABLE | 18,950 | 0.84mm | min **0.079mm**，接触 ~2.6mm² |
| distal | AVAILABLE | 20,535 | 2.95mm | min **0.036mm**，接触 ~11.1mm² |

**"设计参数 vs 实际几何"对照已成立**：5 个有设计文件的病例里 FDI46 的
`DistanceToNeighbor = 0.0`（要求紧接触），而技师实际做出的最小间隙是
0.036~0.079mm（顶点级；真实表面接触估计 0~0.05mm）——**设计参数写 0
不等于最终几何 0**，Contact Engine 的目标值应该来自 GT 反求，不是照抄参数。
`CrownGenDistanceToNeighbor` 在这批病例里未写出。

## 为什么 13 例没有牙弓

这批 exocad 语料是**设计产物导出**（冠/代型/桥架 CAD STL），牙弓扫描只在
少数病例目录里出现。审计还发现**同一病例目录名会同时出现在 all_data 与
EXO 两个根**（一个放设计产物、一个放扫描）——工具已按"全部同名目录"检索。

## 对提交 I 的影响

* Contact Engine v1 的**真实数据回归只有 1 例**（双侧）——不足以单独证明，
  但足以打通链路；算法正确性主要靠**合成测试**（两个已知接触距离的邻牙
  patch + 已知应答）保证。
* 扩大可用集的两条路（按成本排序）：
  1. **登记补数据**：向数据方索要这 18 例对应订单的颌位/牙弓扫描
     （STL/PLY 均可）——审计工具可直接复检；
  2. **配准救回 4 例**：OUT_OF_FRAME 的候选模型与患者帧做刚性配准
     （以 GT 冠 ↔ 模型上 46 插座缘为对应），配准残差达标后同一审计
     会转 AVAILABLE。这是有明确验收标准的独立小任务。
* Contact v1 的接口按既定设计（`optimize_contact(crown_outer, frame,
  mesial_patch, distal_patch, parameters) -> ContactResult`），
  只动近远中接触区、锁颈缘与咬合解剖。
