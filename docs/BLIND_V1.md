# Blind v1（Final Blind 数据集）—— 三层隔离与冻结流程

用户 2026-10-01 决定：AutoMarginGTSet v1 的 44 病例已全部曝光（train/val/hard-val），
只能用作**开发语料**；final blind 必须用**从未进入过 M0/M1 任何环节**的新病例。

## 数据集（`modeldie` 路线）

| 项 | 值 |
|---|---|
| 扫描的未使用病例 | **2068**（全库 2116 减去已曝光的 44，另有 4 例无 margin） |
| 其中有 `*modeldie` 候选的 | 91 |
| **几何选择成功（MARGIN_ON_DIE）** | **91 病例 / 198 行（100%）** |
| 选中代型的 margin 投影 p95 | p50 0.0000 / p90 0.0038 / **max 0.0077 mm** |
| 命中来源 | 全部 `MODEL_INFO`（模型坐标系；modeldie 正是模型导出） |
| 每病例牙位数 | 1 牙位 51 例、2 牙位 17、3 牙位 6、4 牙位 5、5 牙位 4、6 牙位 3、7 牙位 3、8 牙位 2 |

`modelbase` 探针（104 例 / 188 行成功）**不进 v1**：它还没有证明与 modeldie 语义
等价，留到 AutoMarginGTSet v2 extension 单独验证（用户 §9）。

## 三层隔离（`app/geometry/auto_margin/blind_guard.py`）

| 层 | 文件 | 内容 | 谁能读 |
|---|---|---|---|
| ① | `blind_selection_record.json` | 选择证据：margin_source、proj_p50/p95/max、形态与拓扑结论、why_selected | 只有建集工具 |
| ② | `blind_generation_manifest.json` | **只有部署时真实可得的输入**：case_uuid / fdi / source / die_asset_id / die_sha256 / axis / mesh_n_faces / input_schema_version | 生成进程 |
| ③ | `blind_evaluation_manifest.json` | GT Margin 点列 + GT QC provenance + selection evidence | 只有 evaluator |

**生成侧禁止一切 GT 派生字段**（不只是 Margin 点）：`margin / gt / p95 / proj /
max_mm / side_method / above_frac / asset_kind / die_like / qc / topology / …`
全部在禁用名单里；另有折线点列检测（≥3 个三元组即拒绝）。

独立核验（不依赖工具自身的 assert）：生成侧全文搜不到任何 GT 派生关键词；
三元组出现 **198 次 = 恰好每条一个 `axis`**，无点列。

## 流程（严格按用户 §新 Blind 十步）

```bash
# 1-4 冻结：commit / config_id / QC 阈值 / AUTO_ACCEPT 规则  ← 已完成
# 5   生成（不读 GT）
python3 run_blind_generation.py --out /tmp/automargin \
    --gen-manifest /tmp/automargin/blind_generation_manifest.json --source modeldie
# 6-7 预测落盘 + seal（sha256）
# 8   独立进程评估（先校验 seal，哈希不符直接拒绝）
python3 blind_evaluate.py --out /tmp/automargin --source modeldie \
    --pred /tmp/automargin/blind_predictions_modeldie_<stamp>.jsonl
# 9-10 一次性四层报告：Coverage / Accuracy-given-accept（行权+病例权）/ Failure audit
```

**评估阶段才允许看 GT**；生成阶段禁止 import/open ③（有 AST 级测试盯着）。

## 四层报告要回答的问题（用户 §8）

1. Coverage：AUTO_ACCEPT / REVIEW / FAILED 各多少；
2. Accepted-only 精度：median / p95 / Hausdorff / ≤0.20 / ≤0.50；
3. 病例权：病例内先取 median 再聚合（避免多牙病例支配统计）；
4. Failure audit：REVIEW 原因、分布，以及最关键的一项 ——
   **AUTO_ACCEPT 里有没有 p95 > 1mm 的 catastrophic miss**（产品化阶段这比平均
   p95 更重要：算法"非常自信地输出了严重错误"必须接近 0）。
