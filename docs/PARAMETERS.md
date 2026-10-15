# 参数系统

本系统的核心约束只有一条：**任何进入几何计算的数值都必须可溯源**。
代码里不允许出现 `minimum_thickness = 0.8` 这类字面量。

---

## 1. 为什么需要这个系统

旧实现把临床/加工参数写死在代码里：

```python
cement_gap_mm = 0.06        # 实测 exocad 真实众数是 0.025
minimum_thickness_mm = 0.8  # 实测 exocad 真实值是 0.6，且边缘区允许降到 0.2
```

后果不是"不够精确"，而是**方向性错误**：`minimum_thickness = 0.8` 一刀切会把真实可加工
牙冠的 **33.4%** 判为过薄（用 Möller–Trumbore 光线求交实测真值冠壁厚：中位 0.909 mm，
颈部三分之一中位 0.654–0.780 mm），导致最小厚度修正把整个颈部三分之一外推最多 1.63 mm。

参数系统的职责就是把这类数值赶出代码，并且**保留它们的来源与不确定性**。

---

## 2. 四态参数模型

exocad 的 `.constructionInfo` **只写出被显式设定或偏离默认的参数**（实测每牙位
47–134 个不等）。因此"某个参数没写出来"和"某个参数写了 0"是两种不同的东西，
必须分开表达。

| 状态 | 含义 | 能否参与自动设计 |
|---|---|---|
| `EXPLICIT` | exocad 写出了值，语义无歧义 | ✅ |
| `EXPLICIT_BUT_UNRESOLVED` | 写了值，但该值在临床上不可信 | ❌ 必须先人工定性 |
| `MISSING_INTERNAL_DEFAULT_UNKNOWN` | 没写出。含义是"exocad 内部默认值"，该默认值不在文件里，**因此未知** | ❌ |
| `NO_CONSTRAINT` | 写出 NaN，表示此项无约束 | ❌ 不得被替换成任何数值 |

实测四态命中（15 例 / 38 条牙位记录）：

```
EXPLICIT                      2661
NO_CONSTRAINT                  453
EXPLICIT_BUT_UNRESOLVED          4
```

那 4 条全是 `CementGapThickness = 0`，分布在两个病例：

| 病例 | FDI |
|---|---|
| `82404060702张哲森` | 12, 13 |
| `82404140726` | 35, 36 |

0 粘接间隙意味着牙冠内表面与预备体直接贴合、无法就位，不可能是设计意图。
所以它们被标为 `EXPLICIT_BUT_UNRESOLVED`，`require()` 会直接抛异常，
**在语义查清之前不允许进入自动设计**。判定哪些参数"取 0 不可信"的注册表在
`app/importers/exocad/construction_info.py` 的 `ZERO_IS_UNRESOLVED`，每条都带理由。

反例（0 是合理意图，**不**在此表内）：`DistanceToNeighbor = 0`（有意接触）、
`UntouchableZoneNearPreparationMargin = 0`（不要保护带）、`CrownAdditionalSpacing* = 0`。

---

## 3. 参数分类（由数据推导，不硬编码）

`app/parameters/classify.py` 从证据数据推导每个参数的稳定性等级。规则：

| 类别 | 判定条件 | 能否提供默认值 |
|---|---|---|
| `LAB_GLOBAL_PRESET` | 所有已写出记录取值一致 | ✅ 作为**该实验室**默认值 |
| `RESTORATION_PRESET` | 每个修复体类型内部一致、类型之间不同 | ✅ 按修复体类型 |
| `LAB_PRESET_VARIANT` | 取值 ≤ 3 种但不按修复体类型分组 | ❌ 保留为可选项 |
| `CASE_VARIABLE` | 取值 ≥ 4 种 | ❌ 真变量，绝不可作默认值 |
| `UNKNOWN` | 记录 < 3 条 | ❌ |

**`LAB_GLOBAL_PRESET` 的名字是刻意的。** 它只说明"在导入的这批记录里取值一致"，
因此可以成为该 Lab Profile 的默认值，**但不是全局临床真理，不得硬编码进算法核心**。
换一个实验室的数据重跑，分类会变。

实测分类结果（`lab_profiles/inhouse_76007.json`，依据 14 个订单组 / 38 条牙位记录）：

```
LAB_GLOBAL_PRESET         93
RESTORATION_PRESET         5
LAB_PRESET_VARIANT        17
CASE_VARIABLE              5
UNKNOWN                    6
```

重点参数：

| 参数 | 类别 | 取值分布 |
|---|---|---|
| `MinThickness` | `RESTORATION_PRESET` | `{0.4: 2, 0.6: 25}` — 冠 0.6 / 减量桥体 0.4 |
| `CementGapThickness` | `CASE_VARIABLE` | `{0.02: 7, 0.025: 10, 0.03: 2, 0.08: 2}` |
| `MillingDiameter` | `CASE_VARIABLE` | `{0.5: 10, 0.6: 4, 0.95: 3, 1.0: 6, 1.2: 8}` — 2.4 倍跨度 |
| `MillingMachineAxes` | `LAB_PRESET_VARIANT` | `{3: 12, 5: 19}` — 结构性的，不是数值 |
| `CrownBottomInsertionAngleBetweenTeethLimit` | `LAB_GLOBAL_PRESET` | `20`（25/25） |
| `CrownAdaptationShapePreservation` | `LAB_GLOBAL_PRESET` | `0.001`（25/25） |
| `CrownAdaptationMarginForShapePreservation` | `LAB_GLOBAL_PRESET` | `0.4`（25/25） |

`CementGapThickness` 是唯一一个真正逐病例变化的间隙参数 —— 所以
**不能为它提供默认值**。实测代码里原本写的 0.06 比众数 0.025 大 2.4 倍。

### 参数作用域

`MillingDiameter` 与 `MillingMachineAxes` 的 `scope` 是 `ORDER`（加工/订单维度），
**不属于材料 profile**。把它们写进 `materials/zirconia.json` 是错误的键位 ——
刀具直径在实测中跨 2.4 倍，机轴数有 3 与 5 两种，都不是材料属性。

材料 profile 只承载材料学参数（间隙、壁厚、接触/咬合目标）。
刀具与机台走 `app/manufacturing/ManufacturingProfile`。

---

## 4. 解析优先级

`app/parameters/resolver.py`，顺序固定不可改：

```
Tooth Explicit
    ↓
Case / Order Explicit
    ↓
Restoration-type Preset
    ↓
Lab Preset
    ↓
System Default
```

两条硬性规则：

**规则一：`MISSING_INTERNAL_DEFAULT_UNKNOWN` 只能"继续找下一层"，绝不能"跌落成 0"。**
某一层没有这个参数表示"该层未设定"，继续向下是正确的；最终仍找不到时返回
`MISSING` 且 `value is None`。

**规则二：某一层已经写了值就不继续向下找**，即使那个值是 `EXPLICIT_BUT_UNRESOLVED`
或 `NO_CONSTRAINT`。用 lab preset 的 0.025 去顶替 exocad 写的 `CementGapThickness = 0`，
等于把"待确认"静默改成"已确认"。

`SYSTEM_DEFAULTS` **故意留空**。本项目没有可引用的临床/法规来源来定义系统级默认值，
而"猜一个数"比留空更危险：留空会让解析结果可见、可断言；猜数会让它看起来像真值。
`allow_system_defaults` 默认为 `False`。

### 用法

```python
from app.parameters.profiles import load_lab_profile
from app.parameters.resolver import ParameterResolver

profile = load_lab_profile("app/parameters/lab_profiles/inhouse_76007.json")
resolved = ParameterResolver(profile).resolve(
    tooth=tooth_record, order_id=order_id, restoration_type="AnatomicCrown",
)

resolved["MinThickness"].value          # 0.6
resolved["MinThickness"].source         # ResolvedSource.TOOTH_EXPLICIT
resolved.require("CementGapThickness")  # 未解析时抛 ParameterResolutionError
resolved.blockers                       # 会阻止自动设计的参数
```

**不要写 `resolved["X"].value or 0`。** 未知必须是 `None`，并且 `require()` 会拦住它。

---

## 5. 派生参数（功能依赖 ≠ 经验值）

有些参数不是独立的实验室 preset，而是**另一个参数的确定性函数**。

实测（25 条同时写出这两个参数的记录，相关度 **100%**，无例外）：

```
MillingMachineAxes = 3  →  AdditionalBlockOutAngleAfterCementGap =  0   (10/10)
                           UntouchableZoneNearPreparationMargin =  0   (10/10)
MillingMachineAxes = 5  →  AdditionalBlockOutAngleAfterCementGap = -1   (15/15)
                           UntouchableZoneNearPreparationMargin = 0.1  (15/15)
```

与 exocad 官方对 3 轴/5 轴材料区分 `AdditionalBlockOutAngleAfterCementGap` 的说法一致。

**这两个参数的双峰分布曾被我误判为"两套工艺风格并存"。** 它不是风格，是功能依赖 ——
把它们当独立 preset 查表，等于把一个确定性关系压平成经验值。

`app/parameters/derived.py` 的 `DERIVED_RULES` 注册这些规则，每条带证据说明。
解析链在逐名解析之后跑一遍**派生遍**，只补未解析项，绝不覆盖病例显式值。

**规则不适用时保持未解析。** 机轴数为 4（已知取值集合之外）时，这两个参数保持
`MISSING_INTERNAL_DEFAULT_UNKNOWN` 并标注 `derived_rule_not_applicable`，
而不是用最近似的表项敷衍 —— 否则"规则不适用于这个机台"会被伪装成"规则已应用"，
后果是给一台 4 轴机套上 3 轴或 5 轴的参数。

验收测试会**逐 (参数, 机轴数) 核验实测取值与规则表一致**，数据一旦出现例外立刻失败。

---

## 6. Lab Profile 的生成与 PHI

```bash
python -m app.parameters.build_lab_profile \
    --raw "D:/code/牙冠自动生成/数据/15个完整病例" \
    --lab-id inhouse_76007
```

生成物 `app/parameters/lab_profiles/<lab-id>.json` **可以提交进 Git** ——
它只含参数名、数值、计数与依据规模。

写出前有强制 PHI 校验（`assert_no_phi`）：检查**数据派生字段**（参数名、
修复体类型名、lab_id、notes）必须为 ASCII 无 PHI；不检查本模块自己撰写的中文
`rationale`（那是代码级文案，不是来源数据，把它算进去会让校验每次都被自己的说明文字触发）。

**分组方式**：按 `source_order_id`，不按目录。实测同一订单出现在两个目录，
按目录统计会重复计数并造成训练/验证泄露。

---

## 7. 相关模块

```
app/importers/exocad/        证据数据导入（四态模型的来源）
app/parameters/
  ├─ models.py               ParameterCell / ResolvedParameter / ParameterSet
  ├─ classify.py             由数据推导分类
  ├─ derived.py              派生规则注册表
  ├─ profiles.py             Lab Profile 读写 + PHI 校验
  ├─ resolver.py             固定优先级解析链
  └─ build_lab_profile.py    CLI
app/manufacturing/models.py  ManufacturingProfile（刀具与机台，ORDER 维度）
```

验收测试：`tests/acceptance/parameter/test_parameter_regression.py`（21 项）
