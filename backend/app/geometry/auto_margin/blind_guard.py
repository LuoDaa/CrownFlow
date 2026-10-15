"""Blind 三层隔离（用户 2026-10-01 收紧版）。

**生成进程只允许看到部署时真实可获得的输入。** 由 GT Margin 派生出来的任何
统计（margin→die 投影 p95/max、side method、above_frac、依赖 GT 的拓扑结论、
甚至 margin 来自哪份快照）都属于**选择证据**，一律不进生成侧 —— 这些量在真实
部署时根本不存在，让生成算法看到它们会污染 blind 的全部意义。

三层：

    ① selection/eligibility（可以含 GT 派生信息，只在建集时用，生成永不读取）
         case_uuid / fdi / 候选资产哈希 / margin_source / proj_p95 / proj_max /
         DIE_LIKE 与拓扑 QC / side QC / why_selected
    ② blind_generation_manifest（喂给 Auto Margin 的**唯一**输入）
         case_uuid / fdi / source / die_asset_id / die_sha256 /
         axis / mesh_n_vertices / mesh_n_faces / input_schema_version
    ③ blind_evaluation_manifest（只有 evaluator 打开）
         case_uuid / fdi / margin_source / GT Margin 点列 / GT QC provenance /
         selection evidence

放在包里是为了被单测直接 import；`tools/` 只负责调用。
"""

from __future__ import annotations

#: ② 生成侧**只允许**这些键（= 真实部署时能拿到的输入）
#: 注：**不放 mesh_n_vertices** —— 扩容表里没有该字段，写了只会得到 0（假值）。
#: 宁可不写：生成侧每个字段都必须是真实可得的输入，不能有"看起来有其实是 0"。
GEN_KEYS = ("case_uuid", "fdi", "source", "die_asset_id", "die_sha256", "axis",
            "mesh_n_faces", "input_schema_version")

#: ③ 评估侧允许的键
EVAL_KEYS = ("case_uuid", "fdi", "margin_source", "margin_gt", "n_margin",
             "gt_qc", "selection_evidence")

#: 生成侧禁止出现的**种子名片段**。注意：
#: * "p95"/"proj"/"max_mm" 全禁 —— 投影统计是 GT 派生的
#: * "margin_source"/"side_method"/"above_frac"/"asset_kind"/"die_like" 全禁
#: * 比上一版严得多：上一版只禁了 margin_gt/label/pred 那类"明显"的
FORBIDDEN_SUBSTR = (
    "margin", "gt", "label", "pred", "p95", "proj", "max_mm", "error",
    "chamfer", "hausdorff", "side_method", "above_frac", "asset_kind",
    "die_like", "qc", "topology", "curvature",
)


def assert_generation_manifest_clean(manifest) -> None:
    """② 的结构自检：允许清单 + 折线点列 + 禁用种子名。"""
    def walk(o, path="", top=False):
        if isinstance(o, dict):
            for k, v in o.items():
                low = str(k).lower()
                for bad in FORBIDDEN_SUBSTR:
                    if bad in low:
                        raise AssertionError(
                            "generation manifest 出现 GT 派生的键 %r（路径 %s）"
                            % (k, path or "/"))
                if top and k not in GEN_KEYS:
                    raise AssertionError(
                        "generation manifest 顶层出现非允许键 %r（允许：%s）"
                        % (k, GEN_KEYS))
                walk(v, path + "/" + str(k))
        elif isinstance(o, (list, tuple)):
            if len(o) >= 3 and all(
                    isinstance(e, (list, tuple)) and len(e) == 3
                    and all(isinstance(x, (int, float)) for x in e)
                    for e in o):
                raise AssertionError(
                    "generation manifest 疑似包含点列/折线（路径 %s，%d 点）"
                    % (path or "/", len(o)))
            for i, e in enumerate(o):
                walk(e, "%s[%d]" % (path, i))
    walk(manifest, top=True)


def assert_evaluation_manifest_ok(manifest) -> None:
    """③ 评估侧不该混入输入键（axis 出现在评估侧 = 放错层）。"""
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "axis":
                    raise AssertionError(
                        "评估侧不该带输入键 axis（路径 %s）" % (path or "/"))
                walk(v, path + "/" + str(k))
        elif isinstance(o, (list, tuple)):
            for i, e in enumerate(o):
                walk(e, "%s[%d]" % (path, i))
    walk(manifest)


def split_selection_evidence(row):
    """把一行选择记录拆成 (②可公开的输入, ①选择证据)。

    **只有**输入类字段进生成侧；margin→die 投影统计、side/形态/拓扑结论
    全部留在选择侧。
    """
    gen = {
        "case_uuid": row["case_uuid"],
        "fdi": str(row["fdi"]),
        "source": row.get("source", ""),
        "die_asset_id": row.get("best_asset_id", ""),
        "die_sha256": row.get("best_asset_id", ""),   # 同一把尺子：内容哈希
        "axis": [round(float(x), 6) for x in row["axis"]],
        "mesh_n_faces": int(row.get("best_n_faces") or 0),
        "input_schema_version": "blind_gen_v1",
    }
    ev = {
        "case_uuid": row["case_uuid"], "fdi": str(row["fdi"]),
        "margin_source": row.get("best_method", ""),
        "margin_gt": None,          # 由 manifest 工具现场解析后填
        "n_margin": int(row.get("n_margin") or 0),
        "gt_qc": {
            "proj_p50_mm": row.get("best_p50_mm"),
            "proj_p95_mm": row.get("best_p95_mm"),
            "proj_max_mm": row.get("best_max_mm"),
            "decision": row.get("decision"),
        },
        "selection_evidence": {
            "n_snapshots": row.get("n_snapshots"),
            "n_candidates": row.get("n_candidates"),
            "why_selected": ("margin projects onto the chosen asset within "
                             "p95<=0.20mm; asset positively selected by "
                             "geometry, not filename"),
        },
    }
    return gen, ev
