# Localization Classifier QA（提交 2.2 收尾的人工核验）

**日期**：2026-09-28。**方法**：从 Arch Asset Audit v2 判为
`FULL_ARCH_WITH_TARGET` 的 480 行 / 304 例里，按
（来源 × target_surface_distance 区间）分层抽样 39 个候选，
渲染目标 `<Center>` 邻域的**俯视 + 侧视深度图**（纯 stdlib z-buffer，
`app/groundtruth/qa_render.py`），逐例目检判定。

渲染脚本：`python -m app.groundtruth.qa_render --audit <csv> --out <dir>`
（深度图亮=离相机近，十字 = Center 投影；俯视沿就位方向，侧视沿其面内主方向。）

## 判定结果（31 例可渲染，8 例因哈希不匹配/资产缺失跳过）

| 判定 | 数量 | 占比 | 特征 |
|---|---:|---:|---|
| **A：目标 Prep 确实在** | **5** | 16% | 俯视见环状颈缘 + 亮中心；侧视见**带肩台的凸起 stub** |
| **B：Prep 被抽走，只剩洞/插座** | **7** | 23% | 俯视**环 + 暗中心**（洞底）；侧视见洞窟/窄槽 |
| **C：其它**（完整牙列 / 牙龈 / 模型底座 / 扫描伪影） | **17** | 55% | 中心落在平坦组织、非目标牙、底座或破碎网格上 |
| 不确定 | 2 | 6% | 侧视无明确肩台，prep 与完整牙难分 |

（不确定例按最坏情况计入非 A。）

## 混淆矩阵（分类器 vs 人工，正类 = "目标 Prep 在"）

```
                 人工 A    人工 非 A
判 WITH_TARGET      5         26        → precision ≈ 0.16–0.19
```

**结论：当前 `surface_present` 判据不可用。** v2 报出的 480 行 / 304 例
约 80% 是假阳性，`LocalizationGTSet` 不能按它冻结。

## 主要假阳模式（比"洞缘牙龈"更杂）

1. **B 类（插座）**：sectioned model 的代型被抽走，留下的插座从Center 附近看
   仍有"环 + 3.5mm 内表面"——距离判据完全被洞缘满足。
2. **完整解剖牙列**：不少弓模型本身就是全解剖形态（未做 sectioning，
   也没有该牙的预备体——预备发生在口内，模型是复制牙龈上的完整牙）。
3. **Center 落在底座/别牙/牙龈**：`<Center>` 与该资产上实际被选中的表面
   相差几 mm，距离判据照样满足。

## 可区分特征（下一版分类器的依据，未实现、未调参）

A 与 B/C 在深度图上有明确的物理区别：

```
A（Prep 在）：中心是凸起 —— depth(中心盘 r<1.5mm) − depth(环形带 r=4~6mm) 明显为正
B（插座）  ：中心是洞   —— 同一差值明显为负（洞底比洞缘深）
C          ：中心与周围近乎等高（平的组织/底座），或根本没有环状结构
```

即判据应从"**Center 附近有没有表面**"改为
"**Center 处是否高出周围环形带一个预备体量级（~1mm 量级，待定）**"。
`qa_render.py` 的 z-buffer 已经能直接算这个量，无需新依赖。
**具体阈值必须再用全量分布 + 一轮 QA 验证后才能定，本文档不预设数字。**

## 对冻结算量的影响

```
480/304（v2 判 WITH_TARGET）
→ 人工估计真 Prep 在：约 16–19% ≈ 50–90 行 / 几十例
→ 且其中还要叠加既有闸门（margin/axis 有效、颌别一致等）
```

`LocalizationGTSet` 的最终规模要等新判据实现并重跑 audit 后由脚本产生。
按 30 例的样本量，precision 的 95% 置信区间约为 [0.07, 0.32]，
所以"真 Prep 在"的总量级仍可能在几十例——**Target Locator 走
"公开数据预训练 + click-seed fallback"的路线判断不变。**

## 数据边界声明

QA 用到的只有深度图（纯几何）与 case_uuid 哈希；患者姓名不出现在
任何输出里。渲染 PNG 留在服务器 `/tmp/qa/`（本轮证据），不入 Git。


---

## v3 阶段性发现（2026-09-28 第二轮，特征已全量提取）

Stage A 全量完成：1,677 特征行（OK 281 / NO_LOCAL_POINTS 856 /
MISSING_LOCAL_FRAME 535 / NO_CENTER 5）。31 个 QA1 标签 join 后：

### 发现 1：40% 的 QA1 行没有局部牙架

12/31 例的 ci 里**有 `<Axis>` 但没有 AxisOcclusalOfToothModel/AxisMesial**，
v3 正式特征按规格拒绝 fallback，只能标 MISSING_LOCAL_FRAME。
候选修复（未实现）：**margin-PCA 帧** —— 颈缘是平面曲线，
其 PCA 法向即颈缘平面法向，面内主轴即近远中方向；
全部来自 canonical constructionInfo，不依赖牙架元素是否写出。

### 发现 2：`<Axis>` 的符号跨病例不一致（v3 的 z 轴不能直接用它）

同一种 z 定义（margin-PCA 法向，符号对齐 `<Axis>`）下：

    A（Prep 在）  cf_04e657e/24   prom +3.87   cf_1020cdd/17  +4.40   ✅ 正
    B（插座）     cf_055267/27    prom +3.16   cf_179a472/34  +3.36   ❌ 也正
    B（插座）     cf_0d357ee/36   prom −1.39                          ✅ 负

插座与预备体给出**同号**的 prominence —— 唯一解释是
`<Axis>` 在不同病例里一个朝合方一个朝龈方（符号语义不一致）。
因此任何"z 沿 `<Axis>`"的特征都不可能在全量上直接比符号。

### 对策（v3.1，待实现）：符号无关特征

放弃"先定上方向再比高低"，改用**局部 z 值域内的相对位置**（与符号无关）：

    z_range      局部（2~6mm 环带+中心）的 z 极差
    rel_pos      (center_med − z_min) / z_range
                 Prep（stub 顶）→ rel_pos ≈ 1
                 插座（洞底）  → rel_pos ≈ 0
                 平坦          → z_range ≈ 0，退化单列
    center_spread  中心盘内 z 的极差（插座壁大、stub 顶小）
    |prominence|   幅度量级

这些配合 QA2 的 50–60 个人工标签再定 PREP_PRESENT / PREP_MISSING /
AMBIGUOUS 判据。QA1 数据不足以下结论，本轮**不**改判据、不重跑 audit。

### QA1 跳样重计数（路径 bug 修复后）

QA1 的 8 例"无法渲染"实为我的目录定位 bug（病例目录是嵌套的：
all_data/exo2/<id>、all_data/complete_cases/<id>、EXO/CAD-Data/<id>…），
不是资产类型问题 —— 用户担心的选择偏差**不存在**。
计数：QA_RENDERED 31 / QA_PATH_BUG 8（其中哈希名 1、嵌套目录 7）。


---

## v3.1 投影结论 + 根因（2026-09-28 第三轮）

### v3.1 全量 Stage A

    OK 281 → 503（margin-PCA 帧救回 474 行，FEATURE_UNAVAILABLE 仅 61）
    1,677 特征行，839s

### Stage B 投影：A/B/C 在 v3.1 特征上仍然重叠

    A（3 例可算）prominence_abs med ≈ 0.115 —— 反而低于 C 组 med 0.84
    rel_pos / coverage / shoulder / spread 全部交叠

### 根因被实验钉死：`<Center>` 不在预备体轴上

对 QA1 确认的 A 例 cf_0335f16/36（深度图见清晰 stub）做 13×13、±6mm、
1mm 步的**子中心格搜索**：

    Center 处      prom_abs = 0.115
    (du=+6, dv=+3) prom_abs = 3.664, rel_pos = 0.92, coverage = 0.80
    峰值 / Center = 31.9×

**exocad 的 `<Center>` 与预备体轴心偏移可达 6mm 量级**。
绕 Center 的固定小圆盘（r<1.5mm）特征会完全错过预备体 ——
这是 Stage B 里"A 组 prom≈0、与 C 重叠"的主因，不是凸起判据本身错了。

### 这意味着什么

* 单纯 Center 局部统计（无论什么阈值）不可用 —— 用户预判成立。
* 峰值处特征（prom 3.66 / rel_pos 0.92 / cov 0.80）**是可分的形态**，
  所以有据可依的下一步是 **v3.2 = 子中心格搜索**：
  在 Center ±6mm 找 prep-like 峰值，用峰值处特征判别；
  QA2 的假阳验证也应在峰值处做（邻牙尖同样会成峰，假阳率未知）。
* 工程量：lattice 脚本已验证（单例 13×13 秒级），包进 v3_features 即可。

### 决策点（留给下一轮）

  选项 A：v3.2 格搜索 + 按峰值采样重设 QA2（~60 例）→ 有望得到
          真正的 TP/FP，进而决定 LocalizationGTSet 规模。
  选项 B：止损 —— Target Locator 降级为
          （3Shape 自带牙位坐标 / 公开整颌数据 / 用户点击兜底），
          主线转 32 模板 → Crown Outer。

两组证据都已备齐：A 的依据是 31.9× 的强信号；B 的依据是
即便格搜索可用，真 Prep 病例量级也只有几十，且邻牙尖假阳率未知。

---

## 决策点已结案（2026-09-28）→ 见 [`LOCALIZATION_QA2.md`](LOCALIZATION_QA2.md)

走了**选项 A**（v3.2 格搜索 + 重设 QA2），结果**落到选项 B 的结论**：

* v3.2（子中心格搜索 + 符号无关特征 + 肩台/覆盖条件）的 raw 判据
  precision **0.127**；再加上 margin 轮廓约束也只有 **0.222** ——
  与本文档测得的 v2 判据（≈0.10～0.17）**统计上不可区分**。
* 全语料实测：v3.2 输出的最高峰有 **79% 落在目标牙自己的 margin 轮廓之外**，
  即落在邻牙牙尖 / 完整牙列 / 模型底座上。
* 本文档结尾"邻牙尖假阳率未知"的预判得到确认，而且更糟：用户提出的
  附加条件"肩台环形结构 / 连续覆盖"**没有区分度**
  （落错牙的肩台对比度中位 2.167 > 落在本牙 1.854，覆盖率两者都是满值 1.0），
  另外还出现了"目标牙 6mm 内没有几何却报出轮廓内凸起"的逻辑矛盾假阳性。

于是：**终止这条自有数据 Target Locator 路线**
（`LOCALIZATION_ROUTE_STATUS = "TERMINATED_GEOMETRY_SELF_DATA"`），
不再做 v3.3/v4；`LocalizationGTSet` 改为**只准入人工判定为 A 的牙位**
（两轮合计 10 个）。本文档"真 Prep 病例量级也只有几十"的估计同样成立：
55 例分层样本里只有 7 例是 A。
