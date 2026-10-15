# WelooAI牙冠自动设计 · 工作区操作

界面参考本地 `3shape演示` 的阶段工具、大画布、显示器和透明度布局。
以下为 WelooAI 自己的快捷键映射，不宣称与 3Shape / exocad 原厂一致。

| 操作 | 快捷键 |
|---|---|
| 适应模型 | F |
| 顶 / 前 / 右 / 左 / 底视图（扫描坐标系） | 1 / 2 / 3 / 4 / 5 |
| 线框 / 网格 | W / G |
| 快捷键帮助 / 退出拾取与帮助 | ? / Esc |
| 工作台上颌 / 下颌 / 支撑体 / 边缘线显示 | U / L / S / M |
| Margin 绘制撤销最后一点 | Ctrl+Z / Backspace |
| Margin 绘制恢复点 | Ctrl+Shift+Z / Ctrl+Y |
| 保存当前 Margin | Ctrl+S |
| 流水线外冠 / 内表面 | O / I |
| 流水线壁厚 / 邻牙接触 / 咬合热力图 | T / C / A |

鼠标左键拖动旋转、右键拖动平移、滚轮缩放视图。
输入框、下拉框和编辑文本时不触发快捷键；长按不重复执行。
Mac 可使用 Command 替代 Ctrl。

显示切换、透明度和相机缩放不修改 STL。Margin 撤销/恢复仅作用于当前绘点，
不撤销数据库修订或几何生成。热力图仅在实际产物可用时开启。
被阶段门阻断的运行显示“需复核”，保留最后阶段结果，不冒充最终冠。

验证：`node frontend/tests/viewShortcuts.test.js`；在 frontend 目录执行 `npm run build`。

## 界面参考与本轮调整（2026-10-09）

- 3Shape Automate：https://www.3shape.com/services/automate — 提交订单、设计偏好、复核结果的短流程。
- exocad DentalCAD：https://wiki.exocad.com/wiki/index.php/CAD_Module — 分组显示、透明度、标准视角、Wizard/Expert 工具分层。
- Medit Design：https://support.medit.com/hc/en-us/articles/360052059852--Design-Overview-and-General-Information — 按上下文给出操作引导。

首页默认提供订单包导入和继续设计入口；保留现有后端能力边界。新导入仍进入设计准备，不宣称 ZIP 导入即能无人值守生成。生成页显示数据缺项并禁止已知无法执行的主链；主链进度与实验入口分离，常用参数直接显示、颈缘参数折叠。设计准备和生成与复核沿用同一套视图工具与快捷键。

验证包含生产构建、快捷键自检、真实病例显示、参数恢复、帮助与线框操作，以及首页/准备/复核页面在 390px 下的横向溢出检查。未通过浏览器启动真实几何任务或修改已有病例。
