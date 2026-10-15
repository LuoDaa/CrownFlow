# 验收套件

**完成标准不是"STL 成功生成"，而是"与真实病例真值的定量误差下降"。**

---

## 1. 目录结构

```
backend/tests/
├─ test_axis.py / test_dcm.py / test_margin.py / test_registration.py   几何单元测试
├─ integration/
│  ├─ conftest.py               isolated_db：数据库与数据目录指向 tmp_path
│  └─ test_revision_chain.py    修订链不变量（5 项）
└─ acceptance/
   ├─ conftest.py                fixture 解析 + baseline 加载 + xfail 施加
   ├─ _metrics.py                FDI36 度量
   ├─ _synthetic.py              合成几何 fixture（圆台 / 随机刚体变换）
   ├─ baselines/
   │  ├─ baseline.json           **提交进 Git**
   │  └─ regenerate.py           生成 / 刷新 baseline
   ├─ parameter/
   │  └─ test_parameter_regression.py   21 项
   └─ geometry/
      ├─ test_insertion_axis.py         24 项（合成几何，精度断言）
      ├─ test_gt_margin_extraction.py   13 项（真值颈缘提取器自身）
      └─ test_fdi36_golden.py           真实病例量化守门
```

当前状态：**80 passed / 3 xfailed**。

---

## 2. `baseline.json` 的四个部分

```json
{
  "schema_version": 2,
  "generated_at": "...",
  "metric_versions": { ... },        // 每个度量的定义版本
  "retracted_expectations": { ... }, // 被推翻过的判据及其原因（防止重犯）
  "legacy_measurement": { ... },     // 旧定义下的数值，不参与断言
  "measurements": { ... },           // 当前实测值 —— 用于在 diff 里看 before/after
  "targets":      { ... },           // 验收阈值 —— 测试断言的是它
  "known_defects": { ... }           // 已知缺陷 → xfail 的文字理由与工单描述
}
```

**`metric_versions` 是必需的，不是装饰。** 度量定义变更时（例如 Margin 真值提取
方式从"最低 0.4 mm 带"改为"真实交界缝"），任何用到它的指标数值都会变。
若不做版本标记，Git diff 会让人误以为

```
margin_band_clearance_median_mm: 0.515 -> 0.254
```

是几何算法改善，实际里面混了度量定义变更 —— **无法归因**。
跨版本数值禁止直接比较；要比较必须用同一版本重跑。

**`retracted_expectations` 记录被推翻的判据。** 已经有过一个：
`insertion_axis_reports_review_when_undercut_remains` 曾断言 FDI36 必须报
REVIEW_REQUIRED，后被证明是错误 Margin 的产物。写清原因，以免后人重犯。

**`measurements` 与 `targets` 分离**是关键设计：测试断言 `targets`，
所以当前不达标的项会**失败**（并以 xfail 标记为已知缺陷）；
而 `measurements` 让每次几何改动的效果在 diff 里可见：

```diff
 insertion_axis.undercut_area_ratio:
- 0.3739
+ 0.2141
```

有另一条断言 `test_measurements_are_current` 保证 `measurements` 与当前代码一致 ——
避免"改了代码但 baseline 没更新"。

---

## 3. xfail 烧除机制

已知缺陷用 `xfail(strict=True)` 标记，施加方式集中在 `conftest.py`：

```python
KNOWN_DEFECT_TESTS = {
    "test_margin_band_clearance_within_range": "margin_band_clearance_median_mm",
    ...
}

def pytest_collection_modifyitems(config, items):
    known = json.loads(BASELINE_PATH.read_text()).get("known_defects", {})
    for item in items:
        key = KNOWN_DEFECT_TESTS.get(item.name)
        if key and key in known:
            item.add_marker(pytest.mark.xfail(strict=True, reason=f"[已知缺陷] {key}: {known[key]}"))
```

**用集中映射而不是逐个测试写装饰器**，是因为标记必须与 `baseline.known_defects`
的文字理由保持同源。

`strict=True` 是刻意的：**任何一项意外通过都会让测试报错**，从而强制把该项从
`known_defects` 里删掉并转为正常断言。目标是 **xfail 数量归零**。

**已验证断言不是空的**（这是 xfail 机制最容易变成自欺的地方）：

| 实验 | 结果 |
|---|---|
| 清空 `known_defects` | 4 项几何断言**真失败**（另加 `test_known_defects_are_documented` 也失败） |
| 伪造"已修好"（放宽阈值） | 报 `XPASS(strict)` → 测试失败 |

---

## 4. 当前 3 项已知缺陷

| 指标 | 实测 | 目标 | 原因 |
|---|---|---|---|
| `margin_band_clearance_median_mm` | **0.515** | 0.015–0.05 | `build_blockout_cutter` 用投影柱充当内腔基面 |
| `supra_margin_clearance_p95_mm` | **0.965** | < 0.10 | 同上 |
| `isolated_chain_outer_deviation_p95_mm` | **0.976** | < 0.05 | `enforce_clearance` 用顶点到 cutter 顶点的距离当壁厚，误判并外推 37% 顶点最多 1.63 mm |

三项都在 `baseline.known_defects` 里带修复方向说明。

### 已通过的项

| 断言 | 值 |
|---|---|
| `boolean_backend` = `manifold3d` | ✅ |
| `margin_trim_watertight`（含 `boundary_edges == 0`） | ✅ |
| `boolean_roundtrip_deviation_p95_mm` | **0.0**（精确布尔） |
| `final_watertight` / `final_components == 1` | ✅ |
| 就位方向四项（见 `INSERTION_AXIS.md` §6） | ✅ 全部通过（状态 OK，倒凹 0.117） |
| 真值颈缘提取器 13 项（见 `GT_MARGIN.md` §6） | ✅ |
| 参数回归 21 项 | ✅ |
| 修订链不变量 5 项 | ✅ |

---

## 5. Fixture 解析

验收套件需要**仓库外**的真实病例数据（患者数据不进 Git）。

| 环境变量 | 默认查找位置 |
|---|---|
| `CROWNFLOW_FDI36_DIR` | `D:\code\牙冠自动生成\数据\1629-王煜-STL` |
| `CROWNFLOW_EXOCAD_CASES_DIR` | `D:\code\牙冠自动生成\数据\15个完整病例` |

找不到时**带明确理由 skip**，而不是静默通过：

```
FDI36 基准病例不可用。请设置 CROWNFLOW_FDI36_DIR 指向病例目录。
患者数据不进入 Git，因此这里 skip 而不是失败 —— 但几何验收断言在
该数据可用前无法生效，这一点是已知的。 / 默认查找位置 ... 不存在。
```

**注意**：这意味着在没有数据的机器上跑 CI，几何断言会全部 skip。
这是已知的取舍 —— 要么把 fixture 放到 CI 可达的受控位置，要么接受 skip。

---

## 6. 刷新 baseline

```bash
cd backend
python -m tests.acceptance.baselines.regenerate            # 只打印当前测量值
python -m tests.acceptance.baselines.regenerate --write     # 写入 baseline.json
```

**修好一项后必须做三件事**（否则 `strict` 会报错拦你）：

1. 从 `KNOWN_DEFECTS`（`regenerate.py`）删掉该项
2. 从 `KNOWN_DEFECT_TESTS`（`conftest.py`）删掉对应映射
3. 刷新 `baseline.json`

---

## 7. 新增一条验收断言的流程

1. 在 `_metrics.py` 里加一个度量函数（返回可序列化的 dict）。
2. 在 `regenerate.py` 的 `TARGETS` 里加阈值，**并写清 `why`** ——
   每一项都要能回答"这条断言是为了拦住哪个具体失败模式"。
3. 在 `measure()` 里调用它，`--write` 刷新 baseline。
4. 在对应测试文件里写断言。若当前不达标，加进 `KNOWN_DEFECTS` + `KNOWN_DEFECT_TESTS`。
5. 跑测试：不达标项应 xfail，达标项应通过。

## 8. 删除一条判据的流程

若某条判据被证明**不成立**（不是"修不好"），要删的是判据本身，而不是放宽阈值。
做法与新增一致，但必须：

- 在原来的位置留一段说明，写清**为什么不成立**、**证据是什么**
- 不要保留一个"看起来还能测点什么"的弱化版本

已有两个先例，都写进了 `baseline.retracted_expectations` 或本文件：

1. `insertion_axis_error_deg < 10°（vs +Z）` 被删除，改为四项
   （见 `INSERTION_AXIS.md` §6）—— +Z 不是该病例的真值。
2. `insertion_axis_reports_review_when_undercut_remains` 期望值从
   REVIEW_REQUIRED 改为 OK（见 `GT_MARGIN.md` §7）—— 原期望建立在错误的 Margin 上。

**特别注意第二种**：不要为了让某个 xfail 烧掉，而把物理上正确的结论改判为缺陷。
反向迎合"看起来在改进"是比漏测更危险的事。

---

## 9. 合成几何 fixture

`_synthetic.py` 提供可构造真值的几何：

```python
frustum(r_bottom, r_top, height, sections=120, tilt_deg=0.0, bulge=1.0)
    -> (mesh, base_ring)     # base_ring 就是精确的 Margin 环

true_axis_for_tilt(tilt_deg) -> 真值轴（倾斜后的 +Z）

random_rigid(rng)            -> 随机刚体变换矩阵
apply_rigid(mesh, T) / transform_points(points, T)
```

**自建圆台而不是用 `trimesh.creation.cone`**：后者的底面是带中心点的圆盘
（不是环），把它当 Margin 会让 `validate_margin` 失败，而这与求解器无关 ——
测试 fixture 本身要干净。

`base_ring` 由构造保证精确，所以 Margin 相关断言不会被"环提取不准"这类
无关噪声污染。这一点很重要：真实病例上的 Margin 提取确实还不够可靠（见 `DATASET.md`）。
