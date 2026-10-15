# CrownFlow 1.0 优化源码包清单

本包基于用户提供的 `CrownFlow_Complete(1).zip` 当前代码快照继续修改，**保留原工程源码、测试、文档和配置**。

本轮重点改动：

- Proximal coarse → fine：避免“距离达标但冠壁被拉成长尖/长桥”；
- Proximal / Step-5 增加相对 shape-quality gate；
- Fine proximal 扩宽过渡环并限制主要位移方向；
- Step 5 编辑状态机、结构化失败回滚、失败面高亮；
- Add / Remove / Smooth / Flatten / Move / Rotate / **Local Drag** / Measure；
- 接触状态、厚度参考、异步局部预览；
- 保留 Margin/cervical lock、revision、Undo/Redo、Finalize invalidation。

详细说明：`docs/STEP5_PROXIMAL_MICROADJUST_1_0_2026-10-15.md`。

## 为控制体积未打包的运行时/生成物

以下不是源码，因此未包含：

- `.git/`
- `frontend/node_modules/`、`frontend/dist/`
- Python/pytest 缓存
- Playwright 临时目录
- `output/`
- `backend/run_out/`、`backend/debug_out/`
- 运行期 SQLite 数据库
- `data/cases/`、`data/crown_runs/`、`data/debug/`
- 诊断 JSON/GLB/PNG 等运行产物

保留了依赖清单、前后端源码、测试、工具脚本、文档，以及 `data/templates/` 中用于开发的模板数据。

## 建议安装

后端：

```bash
cd backend
python -m pip install -r requirements.txt
```

前端：

```bash
cd frontend
npm ci
npm run build
```

如果从旧 Windows 工程目录复制到 Linux/macOS，请先删除旧 `node_modules` 再 `npm ci`，不要复用跨平台二进制依赖。
