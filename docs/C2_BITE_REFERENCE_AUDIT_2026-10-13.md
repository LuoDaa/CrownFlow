# C2 第一步：Bite Reference Audit（2026-10-13）

用户冻结的顺序：`Prep Conditioning ✅ → Bite Reference Audit ✅ → Bite Registration → …`。
本批**不写任何配准算法**，只回答"15 例里三份扫描各自处于什么坐标语义、几何能不能读、
vendor reference 在哪"。术语按用户修正执行：`biteregister.DCM` = **bite/alignment scan**，
**不是**"exocad 算出的注册真值"；只有**已注册的上下颌相对位姿**才是 reference，
且 schema 里只写 `EXOCAD_REGISTERED_POSE`，**绝不写 GROUND_TRUTH**。

产物：`数据/bite_reference_audit.json`（模块 `app/services/bite_reference_audit.py`，只读）。

## 1. 角色是元数据真值（不用文件名猜）

`constructionInfo` 的 `ScanFiles/ScanFile` 给出每份扫描的角色，且**文件名与角色不绑定**：

| `PartType` | 角色 | 实测例 |
|---|---|---|
| `UnsegmentedJaw` | 预备体颌扫描 | 张哲森 → `-upperjaw.DCM`；邱治钧 → `-lowerjaw.DCM` |
| `Antagonist` | 对颌扫描 | 张哲森 → `-lowerjaw.DCM`；邱治钧 → `-upperjaw.DCM` |
| `BiteRegister` | **咬合对齐扫描** | `-biteregister.DCM` |
| `WaxupScan` / `SituScan` | 蜡型 / 就位扫描 | `0702.stl` / `1.stl` |

⇒ "upperjaw/lowerjaw" 这两个词**不代表**预备体/对颌（同一份 `-lowerjaw.DCM` 在两例里角色相反）。
每份扫描还带 `TransformationMatrix`（38/38 全部 `CUSTOM`，非单位阵）与 `FileHash`。

## 2. 硬结论：15 例的扫描几何**全部不可读**

15/15 病例的 4 份 DCM（prep jaw / antagonist / bite register / bite register 2）都是
`<Schema>CE</Schema>` 的 **3Shape CE 加密载荷** —— 没有合法密钥，顶点载荷读不出来。
`jaw_pair_readable = 0`、`bite_alignment_scan_readable = 0`、`c2_feasibility = NO_READABLE_JAW_OR_BITE_GEOMETRY` × 15。

所以第二份 `biteregister2.DCM`（14 例有）**是不是第二个咬合区扫描，本轮无法用几何回答** ——
只能记"存在"，不能编语义。同理 `TransformationMatrix` 我们也**无法验证它到底做了什么**
（矩阵本身可读，但没有可读几何去检验它把谁对齐到谁）。

## 3. 目录里可读的资产（这才是新信息）

| 类别 | 数量 | 说明 |
|---|---:|---|
| `MODEL_DIE`（逐牙位代型） | **3** | 82404110733：FDI 24/25/27（`吴宝刚-24-modeldie.stl` 等） |
| `MODEL_BASE`（模型底座） | 2 | 24–27 / 37–34 两段 |
| `CAD_RESTORATION` | 2（元数据映射）| `construction_files` 里 `tooth_numbers=[24,25,26,27]`、`material=ZI/氧化锆` |
| `UNKNOWN_ASSET`（未归类 STL） | **63** | 命名不含 modeldie/modelbase/cad 且不在 `construction_files` 里 |
| `ENCRYPTED_SCAN` | 60 | 4×15 份 DCM |

两个必须记下的口径问题：
1. **只有 2 个病例的 `constructionInfo` 带 `ConstructionFileList`** ⇒ 其余病例的 CAD 导出文件
   无法用元数据映射牙位，只能落到 `UNKNOWN_ASSET`（63 个）。
2. **一个目录里可能有多个 `constructionInfo`**（邱治钧有 2 份：`… 2r2.5 荣耀 B组-.constructionInfo`
   与 `邱治钧.constructionInfo`）⇒ 目录 ≠ 单个订单；审计只取字典序第一个并在 `notes` 里写明。

## 4. 代型能不能当"逐牙位支撑资产"：**不能，跨坐标系**

把 3 个 `MODEL_DIE` 喂给 P1 契约（用该 FDI 的 metadata margin + 插入轴）：

```
FDI24 margin→面 45.60 mm · FDI25 40.29 mm · FDI27 31.00 mm → REVIEW_REQUIRED / MARGIN_NOT_ON_SUPPORT_SURFACE
```

**30–45 mm 不可能是"边缘线画歪了"，只能是两个坐标系**（代型在模型/打印系，margin 在设计系）。
这个发现同时暴露了 P1 契约的一个缺陷并已修复（见 `docs/P1_PREP_CONDITIONING_2026-10-13.md` 的更正节）：
只做拓扑判据时，margin 离网格很远会让"margin 不落在开放边界上"**平凡成立**，
从而产出**假 READY**。修复后先过"同坐标系健全性门"（`margin→面 ≤ 1.0 mm`）再谈拓扑。

⇒ 结论：**代型是"有几何、但坐标系不通"**，要用它必须先确定 **design→model 变换**
（`constructionInfo` 里可读的 `MatrixToScanDataFiles`（纯旋转，奇异值全 1）与
`ConstructionFile/ZRotationMatrix` 是两个候选，本轮不验证）。这是"Prep Asset Production"
那一批的具体入口，不是 C2 的活。

## 5. C2 的数据计划（本批最重要的产出）

把可读几何全扫一遍后，只有 **1629-王煜-STL** 同时具备"可读 + 语义齐全"：

| 文件 | 面数 | 与谁共坐标（实测） |
|---|---:|---|
| `02_口扫数据/PreparationScan.stl` | 92,613 | ↔ `Unsectioned_LowerJaw`：p50 **0.115 mm**、94.0% ≤0.5mm ⇒ 同一坐标系 |
| `02_口扫数据/AntagonistScan.stl` | 155,179 | ↔ `Unsectioned_UpperJaw`：p50 **0.111 mm**、99.7% ≤0.5mm ⇒ 同一坐标系 |
| `01_主要病例模型/Unsectioned_LowerJaw.stl` ↔ `…UpperJaw.stl` | 152,960 / 208,298 | min **0.003 mm**（接触带相切）、p50 11 mm ⇒ **已注册的咬合对** |
| `02_口扫数据/AbutmentAlignmentScan.stl` | 274,182 | ↔ LowerJaw 34.0% ≤0.5mm；↔ UpperJaw 4.5% ⇒ 偏向下颌侧，**不是**两面咬合扫描 |

三条推论：

1. **1629 可以给出 `EXOCAD_REGISTERED_POSE`**：`T_L→U` 可以从两副已注册的
   `Unsectioned_*Jaw.stl` 直接测得（几何可读、位置合理）。它是目前唯一能做
   **vendor-reference surface benchmark**（C2 指标第二层）的病例。
2. **但它不能用来"解"配准**：所有文件已经在同一坐标系里，"注册"在那份数据上是恒等变换。
   拿它做 ICP 测试等于自己给自己出没有难度的题。
3. **现有数据里没有任何一个"上下颌各自独立坐标系 + 咬合扫描桥接"的可读病例**
   （15 例加密；3Shape 包加密；`1.0测试数据` 只有上下颌模型 + 定位件、无咬合扫描）。
   ⇒ C2 的开发必须用**合成自注册**：取 1629 的已注册对，施加已知随机刚体变换制造"独立坐标系"，
   再从接触带裁出咬合扫描当桥，然后用已知真值检验 pipeline 能否恢复。
   这与项目既有做法一致（合成几何 + 可构造真值，精度断言放在合成上）。

因此 C2 的验收分两层，需要在实现前定口径：

```text
第一层（合成自注册，有构造真值）：恢复误差 Δt / Δθ + 双侧配准残差
第二层（1629 真实病例）：与 EXOCAD_REGISTERED_POSE 的表面偏差 median/p95（一级指标）
                        Δt/Δθ 只作 debug（两套坐标约定不同，逐元素比较可能无意义）
```

## 6. 本批运行

本机全量 `pytest tests`：**616 passed / 10 skipped / 2 xfailed / 0 failed**
（审计前 608；新增 8 条：C2 审计 7 + P1 同坐标系门 1）。本批零几何算法、零配准算法；
新增的只有一个只读审计模块 + 产物 JSON，以及 P1 契约的一道健全性门（修的是假阳性）。

## 7. 本批明确没做

1. 不做配准、不做 ICP、不做粗对齐（下一批）；
2. 不把咬合"好看"当优化目标（用户明令）：O1 Occlusion 只能作**事后 plausibility gate**；
3. 不碰 63 个 `UNKNOWN_ASSET` 的归属（需要更多元数据或人工标注）；
4. 不验证 `MatrixToScanDataFiles` / `ZRotationMatrix` 的语义（属于 Prep Asset Production）。
