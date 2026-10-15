# M1 Geometry Auto Margin Baseline —— 第一版结果与诊断

**输入**：代型网格 + 就位轴（严格限制）。**GT 只在 evaluate 出现**，生成阶段不看。
**数据**：AutoMarginGTSet v1，只用 `train`(67 行) + `val`(19 行)；
**`test`(9 病例 / 24 行) 保持封存** —— 评测器不传 `--allow-test` 直接拒绝运行，
本次没有看过任何 test 数字。

代码：`app/geometry/auto_margin/`（models / prep_frame / features / geometry_score /
candidate_band / graph_search / loop / spline / projection / quality / engine）
+ `tools/run_auto_margin_m1.py`。

## 一、结论先行：脊线信号是真的，缺的是"整圈一致"

以 `sharp`（按边相邻的二面角锐度）为例，**逐点中位误差 0.070mm**（train）/ 0.071mm（val）
—— 说明几何特征确实**指到了 Margin 上**。但 P95 是 1.2~1.8mm、Hausdorff 2~4mm：

| score | 分片 | median | p95 | hausdorff | ≤0.10 占比 | 闭环率 |
|---|---|---|---|---|---|---|
| **sharp** | train | **0.070** | 1.381 | 1.870 | 0.64 | **1.00** |
| | val | **0.071** | 1.802 | 2.161 | 0.60 | **1.00** |
| **normal_turn** | train | 0.070 | **1.166** | **1.576** | 0.61 | 0.99 |
| | val | 0.091 | **0.861** | **1.412** | 0.57 | 1.00 |
| sharp_axis | train | 0.075 | 1.872 | 2.038 | 0.61 | 0.99 |
| | val | 0.083 | 3.485 | 4.012 | 0.55 | 1.00 |
| sharp_down | train | 4.805 | 7.212 | 7.795 | 0.00 | 0.43 |
| | val | 5.069 | 8.511 | 9.245 | 0.00 | 0.26 |
| radius_break | train | 3.258 | 7.742 | 8.280 | 0.04 | 0.88 |
| | val | 3.704 | 8.655 | 9.179 | 0.05 | 0.95 |

**按行看（一行 = 一条预测环）**：

| score | 分片 | median ≤ 0.20（环基本在 Margin 上） | p95 ≤ 0.50（整圈都紧） |
|---|---|---|---|
| sharp | train / val | 72% / 68% | **39% / 37%** |
| normal_turn | train / val | 76% / 58% | 25% / 21% |
| sharp_axis | train / val | 64% / 63% | 34% / 16% |

预测环周长 / GT 周长 中位 **1.589** —— 环被拉长，正是局部"跑偏到别的脊"造成的。

**两个候选特征**：`sharp` 的中位与闭环率最好；`normal_turn`（法向与轴夹角的变化率）
的 p95/Hausdorff 最好。19 行 val 上两者差异在噪声量级，**M1 v1 不据此冻结**。
`sharp_down` / `radius_break` 明确不可用（中位 3~5mm，锁错了结构）。

## 二、失败模式（这是 M1 最有价值的产出）

1. **局部跑偏**（多数）：环的大部分贴在 Margin 上，但少数扇区跳到**另一条脊**
   —— 底盖棱、预备体上的解剖脊、或者划痕。单条环内自相矛盾（median 好、p95 差），
   正对应"每扇区独立 argmax，没有任何整圈一致性约束"。
2. **整例锁错**（少数）：如 `cf_be0da476093` 的三个牙位（median 4.5~6.3mm），
   score 把某条非 Margin 的结构整体排到了前面。
3. **自交**：430 条预测里 116 条的自交计数 > 0 —— 闭环本身成立（闭环率 0.99~1.00），
   但**投到垂直于轴的平面上会自交**，说明环在轴向上有来回跳。
4. 扇区覆盖：`sharp` 下 filled_sector_frac ≈ 1.00（按面的角向跨度登记之后；
   只按质心登记时只有 0.93）。

## 三、下一版（M1 v2，明确的做法，不是猜）

**把"每扇区独立 argmax"换成"格点上的整圈最短路径"**：

```
格点 = 扇区 i × 该扇区的候选脊 k（不是只取 argmax）
代价 = Σ_i [ −w·score(i,k) ] + λ·Σ_i smooth(|Δh|, |Δr|)
求    = 环上最小代价闭合路径（DP over sectors × candidates，闭环用固定起点扫一遍）
```

这条直接针对上面的失败模式 1/2/3：**整圈一致性变成优化目标的一部分**，而不是
靠事后平滑。实现落在 `graph_search.py`（文件名本来就是为它准备的）+ 可在
`candidate_band.py` 里给每个扇区保留 Top-K 候选。

在此之前不冻结 M1 特征与参数（`sharp` vs `normal_turn` 由 v2 的 val 结果定），
**test 继续保持封存**。

## 四、复现

```bash
cd /tmp/gt
# 评测（4 分片；test 默认封存）
for i in 1 2 3 4; do python3 run_auto_margin_m1.py --out /tmp/automargin \
    --split train,val --shard $i/4 & done; wait
python3 tally_m1.py; python3 tally_m1b.py
# 合成测试（本地）
python -m pytest tests/unit/geometry/test_auto_margin_m1.py
```
