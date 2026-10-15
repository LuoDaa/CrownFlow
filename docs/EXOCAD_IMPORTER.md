# exocad 证据数据导入

把 `数据/15个完整病例` 里的 exocad 工程文件转成结构化数据。
这一层是参数系统的真值来源。

---

## 1. 支持的来源文件

| 文件 | 格式 | 提供什么 | 可读性 |
|---|---|---|---|
| `.constructionInfo` | XML | **per-tooth 真实设计参数**（75–134 个/牙位） | ✅ 明文 |
| `.dentalProject` | XML | 病例级标注：牙位 / 修复体类型 / 材料 / `OrderID` | ✅ `<Notes>` 明文 |
| `.3ox` / `.3OX` | UTF-16 XML | `OrderID`、`ScanSource`、PHI 字段 | ✅ 明文 |
| `.matrix4` | XML | 4×4 变换矩阵 | ✅ 明文（角色见 §4） |
| `.dentalCAD` | 二进制容器（~100 MB） | 可能含 Margin 曲线与就位方向真值 | ❌ 需专门做格式反向工程 |
| `.DCM` | 3Shape CE 加密 | 口扫网格 | ❌ **实测全部无法解码**，见 §6 |

---

## 2. 必须用 XML 解析器，禁止正则

`.constructionInfo` 里存在自闭合标签：

```xml
<Parameter>
  <Value xsi:nil="true" />
  <Type>LoadToothModelForceUsingPonticMode</Type>
</Parameter>
```

正则 `<Value>(.*?)</Value>.*?<Type>(.*?)</Type>` **不匹配** `<Value xsi:nil="true" />`，
于是这些 Parameter 被跳过，其后的 Value/Type **全部错位配对**，参数值被静默张冠李戴。

实测后果：`CementGapThickness` 的总记录数被算成 **7** 而非 **25**，
取值分布也完全错（漏掉 `0.025×10` 这个众数）。

**本模块一律使用 `xml.etree.ElementTree`。** 这是个"看起来能跑、数字全错"的典型陷阱 ——
正则能成功执行、能返回结果、不报任何错。

---

## 3. 病例身份

**只认 `source_order_id`**（来自 `.3ox` 的 `<OrderID>`，回退到 `.dentalProject` 的
`订单:` 字段）。目录名与文件名均不可信，实测证据：

- **同一订单出现在两个目录**：`82404020415薛文爱` 与 `薛文爱` 同属订单
  `03c3329e-87f8-43f3-93d5-c5c9537ee4d1`。
- **同名 `.matrix4` 横跨 4 个病例目录**：`吴宝刚.matrix4` 同时出现在
  `82404110682` / `82404110733` / `82404110783` / `82404140726`；
  另有 `张.matrix4` / `李.matrix4` / `王.matrix4`。

`discover_cases(raw_root)` 返回 `{source_order_id: [case_dir, ...]}`。
实测 15 个目录归为 **14 个订单组**，其中 1 组是重复。

**训练/验证拆分必须按订单分组，不能按目录随机拆分** —— 否则同一口扫描会同时
出现在训练集与验证集两侧。

---

## 4. `.matrix4` 的角色判定

**结论：它不是逐病例咬合配准矩阵。**

实测证据：15 个 `.matrix4` 文件内容**逐字节完全相同**（15 个文件只有 1 个内容 hash），
平移非零但固定、旋转 176.5°、`det = +1`。逐病例的咬合配准不可能在 15 位不同患者上完全一致。

因此 `matrix4.py` 做的是**判定**而不是假定：

| 角色 | 判定条件 | 可否驱动 Occlusion |
|---|---|---|
| `FIXED_COORDINATE_TRANSFORM` | 全部文件内容一致 | ❌ 禁止 |
| `CANDIDATE_BITE_TRANSFORM_UNVERIFIED` | 逐病例不同，但未通过空间验证 | ❌ 验证前禁止 |
| `BITE_TRANSFORM` | 逐病例不同 **且** 已通过空间验证 | ✅ |
| `UNKNOWN` | 无文件 | ❌ |

空间验证指：应用后检查上下颌是否形成合理咬合（对颌距离、牙弓对齐、包围盒、
咬合面朝向）。**本模块不自行升级角色** —— 内容一致就只能是坐标系约定变换。

后果要写清楚：**咬合空间变换在现有 15 例里没有真值。**
只有 `1629-王煜` 一例上下颌天然在同一坐标系。所以：

| 真值类型 | 现有数据 |
|---|---|
| 对颌**身份**（哪颗牙是对颌） | ✅ 部分已有，见 §5 |
| 咬合**空间变换** | ❌ 大部分病例没有明确真值 |

这两件事必须严格分开。

---

## 5. 意外收获：对颌身份真值

`.constructionInfo` 的 `ReconstructionType` 里出现了 `Antagonist`
（`82404090703邱治钧` 1 条、`MY10075901` 4 条），另有 `HealthyTooth`。

**这是"exocad 认为哪颗牙是对颌"的现成标签**，可直接用于验证对颌识别算法
（`ExocadCase.antagonist_teeth`）。

`FunctionalRole` 由 `ReconstructionType` 推导：

| ReconstructionType | FunctionalRole |
|---|---|
| `AnatomicCrown` / `Coping` / `AnatomicPontic` / `ReducedPontic` | `RESTORATION` |
| `Antagonist` | `ANTAGONIST` |
| `HealthyTooth` | `REFERENCE` |

---

## 6. 不可用的部分（要如实说）

**59 个 `.DCM` 全部无法解码**，实测全部报：

```
HPSEncryptionError: Encryption key not found.
Set the HPS_ENCRYPTION_KEY environment variable or provide the key.
```

它们是 3Shape CE 加密的。`hpsdecode` 能正确识别并报错，但需要合法的
`HPS_ENCRYPTION_KEY` 才能读出网格。

`.dentalCAD`（51 个，最大 ~100 MB）是二进制容器，本模块不解析。
可能含 Margin 曲线与就位方向真值，但需要专门的格式反向工程。

**因此当前能取到的几何只有 70 个 STL，而其中最终冠的 STL 含内表面，
不能作为 `crown_outer_gt` 训练标签。** 详见 `DATASET.md`。

---

## 7. PHI 脱敏

PHI 分布在**六层**，所以不能"先读进系统再清理"：

1. 目录名 —— 刘金鸿 / 薛文爱 / 张哲森 / 武淑琴 / 邱治钧 / 张寿山 / 成兰敏 / 王煜 / 吴宝刚
2. STL 文件名 —— `吴宝刚-24-25-26-27-bridge_slm_cad.stl`、`张竹林82404110783-…-B组 (1).stl`
3. `.matrix4` 文件名 —— `吴宝刚.matrix4`
4. `.3ox` XML 字段 —— `Patient_LastName` / `PatientId` / `LabID` / `ClinicID` / `OperatorID`
5. `.dentalProject` Notes —— 订单号与姓名
6. `import.log` —— UNC 共享路径 `\\192.168.169.171\cad共享\…`

`sanitizer.py` 的原则：

- **源病例目录永不被修改。** 脱敏是"复制 + 重命名 + 重写文本"到新目录。
- 输出文件名只由 `source_order_id` 派生的稳定哈希与序号构成，不含任何原始名称。
- 无法安全清理内容的二进制容器（`.dentalCAD` 等）**默认排除**，
  必须显式 `include_binary=True` 才复制，并标记为待人工处理。
- 映射表含 PHI，写入时单独存放并明确标注**禁止提交**。

```bash
# 只读扫描，看有哪些 PHI
python -c "from app.importers.exocad.sanitizer import scan_phi; ..."

plan = build_deid_plan(case_dir, out_root, source_order_id=oid)   # 不写文件
apply_deid_plan(plan)                                            # 输出到新目录
write_mapping(plan, mapping_path)                                # 含 PHI，勿提交
```

---

## 8. 代码结构

```
app/importers/exocad/
├─ models.py             四态模型、ReconstructionType、FunctionalRole、ExocadCase
├─ construction_info.py  XML 解析（禁止正则）+ ZERO_IS_UNRESOLVED 注册表
├─ dental_project.py     Notes 解析 → 牙位 / 修复体 / 材料 / OrderID
├─ three_ox.py           UTF-16 XML → OrderID / ScanSource / PHI 字段
├─ matrix4.py            角色判定（内容一致性 → 固定变换）
├─ sanitizer.py          PHI 扫描 + 导入阶段脱敏
└─ case_loader.py        聚合为 ExocadCase；discover_cases 按订单分组
```

验收测试：`tests/acceptance/parameter/test_parameter_regression.py`
（含病例身份分组、`.matrix4` 角色、对颌标签、PHI 校验）
