# CrownFlow 1.0：冠边缘截面与受约束补厚

日期：2026-10-13。状态：工程审核版本，未提交，未重启现有服务；不具备制造放行资格。

## 本批完成

- 参数化冠边缘：水平边缘宽度、斜向过渡长度、角度、垂直高度。保留确认 Margin 的位置；below-margin 固定为 0。旧设计默认保持刀口模式，需明确启用新截面。
- 最小厚度修正：测量冠体内面顶点与面心到外面的最近距离和法向首交厚度，针对薄区小步向外移动；内面不动，终止线不动。保护已确认邻接／静态咬合参考附近的顶点；拒绝新增翻面、自交或参考面冲突。不满足阈值时保留最后安全修正并返回 REVIEW_REQUIRED。
- 最终 STL 重读后再次检查厚度，不能把修正器的中间结果当最终合格。颈部按截面独立报告，全冠原始薄区不隐藏。
- 间隙增加互不重叠的 cervical margin band、axial wall、occlusal/top 三个采样区域；虚拟盖片继续排除。
- 参数、算法版本和修正结果进入最终 provenance。修改 profile 会改变依赖摘要，使旧内冠过期。
- 最近点和射线查询改为小批次，限制长颈部三角面的候选内存；无射线命中保持未解决值，不填造厚度。

未修改 SupportExtractor、blockout、gap field 或 Boolean 算法。公开 CAD 文档只用于参数语义对齐，不声称实现其内部算法。[exocad 冠边缘和加工参数](https://wiki.exocad.com/wiki/index.php/Designing_the_inside_of_the_crown)、[最小厚度选项](https://wiki.exocad.com/wiki/index.php/Set_Minimum_Thickness)。

## 操作

在“设计参数”展开“冠边缘与补厚”，启用参数化冠边缘，填写截面尺寸；截面轴向高度必须小于颈部过渡高度。需要补厚时勾选“构建时补厚冠体外表面”，重新确认参数，再构建内冠。

在“检查导出”分别查看壁厚、边缘截面、接触与咬合及完整工程报告。未确认的邻牙／咬合不构成受保护真值；报告会保留未评估状态。已声明确认但来源过期时阻止补厚，要求重新确认。

![参数界面](../output/crown-bottom-review/border-parameters.png)

## 真实病例结果

病例“基台测试1-46”保持原基台、Margin、轴、0.06 mm gap 和先前径向 1.15 倍外冠调整。新截面：水平 0.25 mm、斜长 0.6 mm、相对轴 45°、垂直 0.1 mm；颈部过渡 2.4 mm。这些是工程参数，尚无材料厂商验证。

| 项目 | 实测结果 |
|---|---|
| 全链及 STL 重读 | 成功，闭合 watertight |
| 接缝关联 | 每条共享 Margin 边关联两个面 |
| 重读 Margin→seam 最大误差 | 约 0.00000055 mm |
| 自交／流形检查 | 当前候选筛查通过；不是穷尽式认证 |
| 补厚安全步骤 | 接受 1 步，最大位移 0.15 mm；后续步骤被几何门阻止 |
| 全冠原始薄区面积 | 旧版 34.60 → 新版 **27.08 mm²** |
| 冠体薄区面积 | **2.98 mm²**（颈部过渡以上的独立口径，不能与旧版全冠面积直接比较） |
| 冠体采样最小厚度 | **0.199 mm**，未达到配置 0.8 mm |
| 冠体厚度 p05 | 修正前 0.433 → 修正后约 0.550 mm |
| 全冠 gap 中位数 | 0.060574 mm，内面未变 |
| 颈部 gap 中位数 / p95 | 0.03228 / 0.10004 mm |
| 轴壁 gap 中位数 / p95 | **0.03419 / 0.03998 mm** |
| 顶部 gap 中位数 / p95 | 0.06079 / 0.06120 mm |
| 修正状态 | **REVIEW_REQUIRED** |
| 生产状态 | **NOT_PRODUCTION_APPROVED** |

该病例未确认邻接和静态咬合约束，接触覆盖为 NOT_ASSESSED。组件身份仍是工程测试声明，未用厂商元数据证明它属于最终修复基台。

**结论：本批新增了可执行的边缘与补厚链，但未把这例修成制造合格冠。** 残余薄壁、区域间隙差异和未认证工艺都必须保留，不能因为闭合或全冠 gap 中位数接近配置就放行。

## 交付与修改文件

新版结果保存在 `output/crown-bottom-review/real-test1-46/`：`final.stl`、`prepared_support.stl`、`manufacturing_qc.json`、`result.json`。旧 `output/abutment-review/real-test1-46/final.stl` 未被覆盖。

新版 STL SHA256：`4b94920537fcaf1827bc612e930cd653db348602e0419e0adadb1f9e2908e541`。

| 文件 | 修改 |
|---|---|
| backend/app/geometry/crown_border.py（新增） | 轴向局部坐标下偏置截面、周向对应连接；偏置自交拒绝 |
| backend/app/geometry/minimum_thickness.py（新增） | 小步外移、保护参考、复用 deformation gate、修正与冲突报告 |
| backend/app/geometry/nonplanar_crown.py | 在已有 Boolean 分离内外面后接入补厚和截面，再复用共享 seam |
| backend/app/geometry/manufacturing_qc.py | 分批查询、冠体／颈部厚度口径、分区 gap |
| backend/app/services/pipeline.py | 新几何编排与最终重读复核 |
| backend/app/services/contact_measurement.py | 复用已确认参考及来源校验建立保护面 |
| backend/app/services/auto_fit.py | 候选预检接入相同补厚链 |
| backend/app/api/routes.py | 最终构造接入保护参考与 provenance |
| backend/app/api/designer.py | profile 截面一致性校验 |
| backend/app/models.py | 四个边缘参数；沿用 enforce_minimum_thickness |
| frontend/src/App.jsx | 允许确认合法的新补厚配置 |
| frontend/src/components/ParametersLibraryStep.jsx、parameters.css | 独立折叠截面设置及补厚开关 |
| frontend/src/components/QcExportStep.jsx | 原始／冠体薄区分别显示，增加边缘截面审核页 |
| backend/tests/integration/test_manufacturing_border.py（新增） | 修正、保护、截面、参数、射线批次、真实 E2E |
| backend/tests/integration/test_existing_abutment.py | 复用真实病例测试步骤，分目录保存新版结果 |

## 验证记录

- 新功能包含真实 E2E 的运行：4 passed。
- 后续射线空命中边界及合成支撑／螺丝孔回归：6 passed（与前项有重叠，不合计为 10 个独立用例）。
- 旧 nonplanar/designer 流程回归：12 passed。
- 较早的支撑与新合成功能组合：12 passed。
- 前端测试：9 passed；独立审核目录构建通过。浏览器检查了参数启用与数值显示。
- 真实重跑期间曾发现截面连接自交和查询内存问题，均修复后验证。质量门没有被放宽。

## 未完成的制造条件

本批不是全局最小厚度解析证明。受约束修正有 12 步和 100000 面的工程预算；无法安全修正时不承诺达到阈值。颈部截面没有厂商验收规则，不能用冠体分区免除其制造要求。

材料／工艺厂商规则、铣刀补偿、连续就位认证、批量真实病例验收均未完成。本例仍需处理薄壁及分区 gap；现有高度场 gap 近似没有被重写。未经确认的邻接／咬合以及局部符号有效性不足的保护面不能作为生产安全依据。
