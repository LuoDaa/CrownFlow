# Crown 1.0 / FDI26 私有回归

日期：2026-10-16  
分支：`feature/crown-v1-fdi26`

## 目的

这条回归链专门用于真实 FDI26 扫描对，验证 Crown 1.0 的几个上游几何约束是否同时成立：

1. 模板横向放置消费已确认患者坐标帧，不再用 world-X 猜近远中方向；
2. 牙库冠的颈缘边界硬拟合患者 Margin，颈缘不能漂；
3. 冠高在局部咬合适配之前先进入对颌可行区间；短冠不能停在半空，过高冠不能靠后续咬合求解器削掉几毫米；
4. 近远中接触从连续 wall patch 形成，不允许单顶点 / 针尖式接触；
5. 真实测试 STL 不进入 Git 历史，所有病例产物只在本地回归目录生成。

这不是生产放行说明。所有结果仍标记 `NOT_PRODUCTION_APPROVED`，真实邻牙 patch、咬合位姿、Margin 与插入轴都必须按正常工作流确认。

## 私有 runner

```bash
python backend/tools/run_fdi26_private_regression.py \
  --upper /absolute/path/to/upper.stl \
  --lower /absolute/path/to/lower.stl \
  --out /tmp/crown-v1-fdi26
```

可选参数：

```text
--library <牙库名>
--margin-mode SUPRA|SUB
--margin-radius-mm 1.0
--min-thickness-mm 0.5
--occlusal-target-mm 0.05
```

runner 不上传文件、不写数据库，也不会把病例 STL 复制进仓库。输入文件只记录本地文件名与 SHA256 到输出目录的 `report.json`，用于同一台机器上的可重复性核对。

## 执行阶段

### 1. Raw scan triage

`app.geometry.scan_pair_triage` 只做连通分量工程筛选：

- 最大 major component 视为整颌；
- 后续 major component 只有满足宽松 tooth-scale 尺寸、紧致度与面数比例条件时，才可成为 detached prep / die candidate；
- 小标记点、扫描碎片不能被当成基牙；
- 这些阈值仅用于私有回归分流，不是解剖学 / 临床判定阈值，也不进入 live tooth localization。

若找不到独立 support，状态为 `BLOCKED / DETACHED_SUPPORT_NOT_FOUND`，不继续猜。

### 2. Axis + Margin proposal

runner 对独立 support 先做 provisional sidewall axis，再调用现有 M1 Margin 引擎。

M1 没有给出 `AUTO_ACCEPTED` 闭环时，状态为 `REVIEW_REQUIRED / AUTO_MARGIN_NOT_ACCEPTED`。此时不继续构冠；真实工作流应人工复核 Margin / Axis。

### 3. Margin-constrained CrownOuter

如果当前安装的牙库存在兼容 FDI26 模板，则用 `CrownOuterConfig(body_scale_decoupling=True)` 生成初始外冠。

没有可用 FDI26 模板时返回 `REVIEW_REQUIRED / FDI26_TEMPLATE_UNAVAILABLE`，不拿其他牙位静默替代。

### 4. Axial placement

原始上下颌扫描若已处于同一静态咬合坐标中，私有 runner 直接把下颌主 component 作为对颌参考，仅用于回归：

- 先测 `prep_top + minimum_thickness` 与对颌天花板；
- 冠过高时做有限轴向压缩；
- 冠虽可行但明显低于对颌安全上界时，做有限轴向增高；
- 颈缘固定带位移保持 0；
- 超过允许缩放范围或空间不足时 fail-loud，不输出“看起来像牙”的假成功。

### 5. Proximal

私有 raw scan runner **不从整颌网格猜 FDI25 / FDI27 邻牙 patch**。这一阶段在报告里保持：

```json
{
  "status": "NOT_ASSESSED",
  "reason": "RAW_ARCH_HAS_NO_CONFIRMED_MESIAL_DISTAL_NEIGHBOR_PATCHES"
}
```

完整近远中接触仍通过正常应用工作流消费已确认邻牙 patch；求解器内部已经把最近点 seed 扩成连续 wall patch，并由单测锁定。

## 本地输出

典型输出：

```text
fdi26_support_candidate.stl
fdi26_crown_margin_fitted.stl
fdi26_crown_axially_placed.stl
report.json
```

`report.json` 必须保留每一阶段的 `status / reason / provenance`。任何 `NOT_ASSESSED / REVIEW_REQUIRED / BLOCKED` 都不能在 UI 或报告层包装成通过。

## 自动测试

新增单测覆盖 raw scan triage：

```bash
pytest backend/tests/unit/geometry/test_scan_pair_triage.py -q
pytest backend/tests/unit/geometry/test_axial_placement.py -q
pytest backend/tests/unit/geometry/test_proximal_patch_broadening.py -q
```

FDI26 私有 STL 不进入 CI；CI 只跑合成网格与公开仓库内资产。真实扫描回归使用上面的本地 runner。

## 1.0 验收口径

FDI26 私有回归只能在以下条件全部满足时记为 `PASS_PRIVATE_REGRESSION`：

- 独立 support 识别成功；
- M1 给出可用 Margin；
- 有真实 FDI26 模板；
- CrownOuter 生成成功且颈缘约束通过；
- axial envelope 可评估；
- 轴向放置方案可行，形变 gate 未拒绝；
- 产物与报告均成功落盘。

即使上述条件全部满足，也不代表近远中接触、动态咬合、制造可达性或临床适合度已经认证。生产放行仍走现有最终 QC / manufacturing audit。
