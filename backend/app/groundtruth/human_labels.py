"""人工牙位判定登记表 —— Localization 两条 QA 线的**唯一**来源。

## 为什么需要这个模块

``LocalizationGTSet`` 原本由 ``arch_asset_class == FULL_ARCH_WITH_TARGET``
自动准入（v2 判据"Center 附近有表面"）。两轮人工 QA 把这个判据证伪了：

* **QA1**（31 例，只从 WITH_TARGET 抽，v3.0 特征投影）：只有 3 例是真的
  预备体在（A），precision ~0.10。
* **QA2**（55 例，分层含高分峰/in_poly 工作点/低分/UNKNOWN/METADATA）：
  v3.2 格搜索 + margin 多边形判据的 precision 见 ``docs/LOCALIZATION_QA2.md``。
  最刺眼的单例：**全样本最高分候选（prom 8.78）是底座 + 完整磨牙**。

结论写死在这里，不再靠自动判据准入：**只有人工判为 A 的牙位才允许进
``LocalizationGTSet``**，而这条"靠自有 exocad 扫描自动定位目标牙"的路线
已经被终止（见 ``LOCALIZATION_ROUTE_STATUS``）。

## 标注口径（用户规格：只标 A/B/C/U）

    A = 预备体（prep/代型）确实在可观察几何里，且峰位落在它上面
    B = 代型被抽走，只剩洞 / 沟槽（sectioned base）
    C = 峰落在其它牙（天然完整牙、邻牙牙尖）、底座、或扫描伪影上
    U = 看不清，无法判定

**B 不是 A**：洞的存在恰恰说明预备体已被移除，不构成定位真值。
"""
from __future__ import annotations

from dataclasses import dataclass

#: 允许进入 ``LocalizationGTSet`` 的唯一人工标签。
HUMAN_PREP_POSITIVE = "A"

#: 合法的标签集合（``human_label`` 输出与 QA manifest 都用它校验）。
ALLOWED_LABELS = frozenset({"A", "B", "C", "U"})


@dataclass(frozen=True)
class PrepVerdict:
    """一个人工判定。``case_key`` 可能是 case_uuid 的**唯一前缀**（QA1 如此）。"""

    label: str
    qa_round: str
    case_key: str
    fdi: int

    def __post_init__(self) -> None:
        if self.label not in ALLOWED_LABELS:
            raise ValueError(f"非法人工标签 {self.label!r}，只允许 A/B/C/U")


# ---------------------------------------------------------------- QA1（31 例）
#: v3.0 特征投影那一轮，按 (case_uuid 前缀, fdi) 存。只从
#: ``FULL_ARCH_WITH_TARGET`` 抽，所以**不能**用它做 precision 之外的推断
#: （没有负类样本，recall 无法估计）。
QA1_LABELS: dict[tuple[str, int], str] = {
    ("cf_005dc7b22dd", 11): "C", ("cf_0258d57bac2", 11): "U",
    ("cf_0335f160d84", 35): "C", ("cf_0335f160d84", 36): "A",
    ("cf_04e657e6225", 24): "A", ("cf_05dee5a5045", 36): "C",
    ("cf_05e78fabf85", 15): "C", ("cf_06143c10636", 12): "C",
    ("cf_0118203cf35", 25): "C", ("cf_0a552112b96", 22): "C",
    ("cf_0fff08ee00c", 22): "C", ("cf_1463311eee3", 26): "C",
    ("cf_1648b1bc089", 22): "U", ("cf_179a4722db1", 34): "B",
    ("cf_17d82bed7de", 37): "B", ("cf_0258d57bac2", 22): "U",
    ("cf_04e657e6225", 33): "U", ("cf_055267292a2", 27): "B",
    ("cf_0e231125713", 16): "C", ("cf_0e5c6a0fd55", 25): "U",
    ("cf_0f8b042a97a", 46): "C", ("cf_100609129f3", 27): "C",
    ("cf_083aad0878a", 24): "C", ("cf_0d357ee5924", 36): "B",
    ("cf_1020cdd7640", 17): "A", ("cf_117dacffd32", 46): "C",
    ("cf_13f1ceb5f05", 16): "C", ("cf_13f1ceb5f05", 15): "C",
    ("cf_1771f382d62", 24): "C", ("cf_6fc88119791", 15): "C",
    ("cf_6fc88119791", 11): "C",
}


# ---------------------------------------------------------------- QA2（55 例）
#: v3.2 格搜索那一轮，键是**完整** case_uuid（QA2 的 manifest 是完整 uuid）。
#: 由 ``docs/LOCALIZATION_QA2.md`` 冻结；改这里必须同时改那份文档。
#:
#: 实测分布：A=7 / B=2 / C=42 / U=4。
#: **刻意过采样了 in_poly=True 这批**（v3.2 唯一会判 Prep 的工作点），
#: 所以这里的比例不能当总体比例读 —— 总体只会更差。
QA2_LABELS: dict[tuple[str, int], str] = {
    ("cf_9593a3da3b1e3409", 36): "C",
    ("cf_3e365fd609d5a85a", 21): "C",
    ("cf_ee381ad3e0e04b45", 47): "A",
    ("cf_ffb6f0cb356ca848", 32): "U",
    ("cf_9e45cb360d8409b4", 15): "C",
    ("cf_a243254eadea0b85", 32): "C",
    ("cf_3e9478dd4181472c", 27): "C",
    ("cf_35d32e1b7a731ad0", 21): "C",
    ("cf_9e45cb360d8409b4", 44): "C",
    ("cf_4eb14580a61b071e", 14): "C",
    ("cf_fe645cc532c36cda", 16): "A",
    ("cf_bf0e61da23a8a34b", 45): "A",
    ("cf_918e4e9948fc232e", 22): "C",
    ("cf_2fe820adf9b440fd", 46): "C",
    ("cf_15afe00929c040f8", 14): "C",
    ("cf_4d1162d079648847", 16): "C",
    ("cf_73813c4a97693bfe", 46): "B",
    ("cf_b71e6f7def991fa9", 13): "A",
    ("cf_eb4a25c302e5ae2a", 11): "C",
    ("cf_a11a60c891cbd2de", 37): "A",
    ("cf_913e0d4ebd3b4561", 36): "C",
    ("cf_bb24c6a1779cc561", 46): "C",
    ("cf_a94d455ccde32e79", 15): "C",
    ("cf_c533f356eb4cef0c", 22): "C",
    ("cf_5a6f18cc1349a978", 35): "C",
    ("cf_9f3a7d2d4e5760e7", 45): "C",
    ("cf_a9fa2afa422ec952", 36): "A",
    ("cf_f547b232c173b216", 45): "U",
    ("cf_c6ac4c93adb9905c", 45): "C",
    ("cf_84fec1df0576d32b", 33): "C",
    ("cf_dac42eb1fc1a6192", 37): "A",
    ("cf_3e3bf04cc11b75a3", 14): "C",
    ("cf_637255ed37cb5989", 36): "C",
    ("cf_14611ff0f46877f3", 44): "C",
    ("cf_485c9dea34ec8517", 43): "U",
    ("cf_5c84b6532ae6264c", 43): "C",
    ("cf_8be39f199d7d7af5", 24): "C",
    ("cf_d8ac811ee84cf465", 21): "C",
    ("cf_005dc7b22ddbc1ca", 11): "C",
    ("cf_a94d455ccde32e79", 12): "C",
    ("cf_b1e30086277b0b09", 24): "C",
    ("cf_d4b1cc2ba960f982", 23): "C",
    ("cf_106a04fc5bb24ee5", 45): "U",
    ("cf_6ce79eff7ba0a36c", 13): "C",
    ("cf_5c884747e8d680ff", 15): "C",
    ("cf_758022c0d9ed2fb3", 16): "C",
    ("cf_23af5ce6b0e60a57", 12): "C",
    ("cf_910d96981137a2fc", 11): "C",
    ("cf_5ba6ba8039266c0c", 46): "C",
    ("cf_85d34e3ad1df2166", 23): "C",
    ("cf_4fbba56bce3c2d9f", 33): "B",
    ("cf_d5cdd15dfb61cbfa", 27): "C",
    ("cf_6f5d0c42b3f2e3c7", 46): "C",
    ("cf_106a04fc5bb24ee5", 47): "C",
    ("cf_224f72506d86c43b", 16): "C",
}


#: QA2 的选样分层 -> (样本数, 其中 A 数)。**过采样工作点的证据**：
#: ``in_poly`` 是 v3.2 唯一会判 Prep 的那批，被刻意抽了 12 例。
QA2_STRATA: dict[str, tuple[int, int]] = {
    "A_high": (10, 1),
    "in_poly": (12, 4),
    "border": (12, 2),
    "mid": (11, 0),
    "miss_strong": (1, 0),
    "unk_strong": (9, 0),
}

#: QA2 的判定规则与混淆矩阵（peak1_in_polygon 为预测正类）。
#:
#: 注意两个不同的"工作点"，别混为一谈：
#:   * **raw**：取最高峰即判 Prep —— 在 55 例上全部触发，precision 7/55=0.13；
#:   * **in_poly**：再要求峰落在目标牙 margin 轮廓内 —— TP=4/FP=14，precision 0.22。
#: 后者是 v3.2 相对 v2 的全部增益，而 QA1 测得的 v2 是 ~0.17
#: （3/31，且 QA1 也是过采样正类）—— **两者在统计上不可区分**。
QA2_CONFUSION = {
    "rule": "peak1_in_polygon",
    "TP": 4, "FP": 14, "FN": 3, "TN": 33,
    # 预测正类的总数 = 整个 55 例样本里 in_poly=True 的行数（18）。
    # 其中 12 例来自专门的 in_poly 分层，另 6 例是 A_high/border 里本就
    # in_poly 的（idx 0、6、45、47、51、54）—— 所以 18 != 分层容量 12。
    "in_poly_rows_in_sample": 18,
    "precision": 0.222, "recall_on_sample": 0.571,
    "raw_top_peak_fires": 55, "raw_top_peak_TP": 7, "raw_precision": 0.127,
    "in_poly_precision": 0.222,
}


#: 轮次 -> 原始标签表。新增 QA 轮次只需在这里加一行 + 把标签填进去。
QA_ROUNDS: dict[str, dict[tuple[str, int], str]] = {
    "QA1": QA1_LABELS,
    "QA2": QA2_LABELS,
}


def all_verdicts() -> list[PrepVerdict]:
    """展平所有轮次的判定，供统计与文档生成使用。"""
    out = []
    for rnd, table in QA_ROUNDS.items():
        for (key, fdi), label in table.items():
            out.append(PrepVerdict(label, rnd, key, int(fdi)))
    return out


def verdicts_for(case_uuid: str, fdi: int) -> list[PrepVerdict]:
    """取某个 (case_uuid, fdi) 的全部人工判定。

    QA1 的键是 12 字符前缀，QA2 是完整 uuid，所以用 ``startswith`` 解析。
    同牙位跨轮次都给过标签时全部返回 —— 调用方自己决定要不要当冲突处理。
    """
    if not case_uuid:
        return []
    out = []
    for rnd, table in QA_ROUNDS.items():
        for (key, kfdi), label in table.items():
            if int(kfdi) != int(fdi):
                continue
            if case_uuid == key or case_uuid.startswith(key):
                out.append(PrepVerdict(label, rnd, key, int(kfdi)))
    return out


def human_prep_verdict(case_uuid: str, fdi: int) -> PrepVerdict | None:
    """人工判定；无判定或跨轮次冲突（不同标签）时返回 ``None``。

    冲突返回 ``None`` 是刻意的：冲突说明标注口径有问题，宁可把该牙位
    排除出真值集，也不要静默挑一个标签。
    """
    vs = verdicts_for(case_uuid, fdi)
    if not vs:
        return None
    labels = {v.label for v in vs}
    if len(labels) > 1:
        return None
    return vs[-1]


def labels_agree_across_rounds() -> list[tuple[str, int, dict]]:
    """跨轮次重叠且标签不一致的 (case_key, fdi) —— 标注自洽性检查用。

    QA1/QA2 的样本有交叠（都在同一个 WITH_TARGET 池里），重叠部分的标签
    一致率本身就是一轮免费的人工标注可靠性验证。
    """
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    for v in all_verdicts():
        by_key.setdefault((v.case_key, v.fdi), {})[v.qa_round] = v.label
    conflicts = []
    for (key, fdi), per_round in sorted(by_key.items()):
        if len(per_round) > 1 and len(set(per_round.values())) > 1:
            conflicts.append((key, fdi, per_round))
    return conflicts


def qa_round_summary() -> dict[str, dict[str, int]]:
    """每轮各标签的计数 —— 文档与冻结 manifest 都直接用它。"""
    out: dict[str, dict[str, int]] = {}
    for rnd, table in QA_ROUNDS.items():
        counts: dict[str, int] = {}
        for label in table.values():
            counts[label] = counts.get(label, 0) + 1
        out[rnd] = dict(sorted(counts.items()))
    return out


# ---------------------------------------------------------------- 路线状态
#: **自有数据 Target Locator 路线：终止。**
#:
#: 触发这个状态的是 QA2 的三条实测（详见 ``docs/LOCALIZATION_QA2.md``）：
#:
#: 1. v3.2 格搜索的最高峰有 ~79% 落在**目标牙自身 margin 轮廓之外**，
#:    即落在邻牙/它牙上 —— 直接目视 55 例确认。
#: 2. 用户要求的附加条件（肩台环形结构 / 连续覆盖）**没有区分度**：
#:    落错牙的峰肩台对比度中位 2.167 > 落在本牙 1.854，覆盖率两者都是 1.0。
#: 3. 目标牙位 6mm 内**没有任何几何**的行，照样出强峰且落在 margin 轮廓内
#:    —— 铁定假阳性。
#:
#: 因此不再做 v3.3/v4。生产路线改走：3Shape 病例直接读牙位 + 3D Origin；
#: 普通 STL 走 Target Locator，不确定时由用户点选目标牙。
LOCALIZATION_ROUTE_STATUS = "TERMINATED_GEOMETRY_SELF_DATA"

#: 该状态对应的证据文件名（相对 docs/），供报告与测试交叉引用。
LOCALIZATION_ROUTE_EVIDENCE = "docs/LOCALIZATION_QA2.md"
