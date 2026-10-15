"""人工 Prep 判定登记表（``human_labels``）的规则测试。

这些测试把"定位路线为什么被终止"的**数字**钉死，防止后人改标签时
只改一半（比如改了 QA2_LABELS 却没改 QA2_STRATA / QA2_CONFUSION）。
"""
import pytest

from app.groundtruth import human_labels as hl


# ---------------------------------------------------------------- 基础约定

def test_positive_label_is_A_and_route_is_terminated():
    assert hl.HUMAN_PREP_POSITIVE == "A"
    assert hl.LOCALIZATION_ROUTE_STATUS == "TERMINATED_GEOMETRY_SELF_DATA"
    assert hl.LOCALIZATION_ROUTE_EVIDENCE == "docs/LOCALIZATION_QA2.md"


def test_every_shipped_label_is_legal():
    for v in hl.all_verdicts():
        assert v.label in hl.ALLOWED_LABELS


def test_illegal_label_is_rejected():
    with pytest.raises(ValueError):
        hl.PrepVerdict("PREP_OK", "QA9", "cf_x", 11)


# ---------------------------------------------------------------- 前缀解析

def test_qa1_short_prefix_resolves_against_full_uuid():
    """QA1 的键是 12 字符前缀，QA2 是完整 uuid；同一个牙位都要能查到。"""
    # QA1 里 ('cf_005dc7b22dd', 11) = "C"，QA2 里完整 uuid 也是 "C"
    v = hl.human_prep_verdict("cf_005dc7b22ddbc1ca", 11)
    assert v is not None and v.label == "C"
    # 前缀本身也能查
    assert hl.human_prep_verdict("cf_005dc7b22dd", 11) is not None


def test_unknown_case_has_no_verdict():
    assert hl.human_prep_verdict("cf_ffffffffffffffff", 11) is None
    assert hl.human_prep_verdict("", 11) is None


def test_wrong_fdi_has_no_verdict():
    """同一病例的其它牙位不能被误当成已判定。"""
    assert hl.human_prep_verdict("cf_005dc7b22ddbc1ca", 12) is None


def test_conflicting_labels_across_rounds_resolve_to_none(monkeypatch):
    """跨轮次标签冲突时必须返回 None —— 宁可排除，不可静默挑一个。"""
    monkeypatch.setitem(hl.QA1_LABELS, ("cf_conflict", 11), "A")
    monkeypatch.setitem(hl.QA2_LABELS, ("cf_conflict0000000", 11), "C")
    assert hl.human_prep_verdict("cf_conflict0000000", 11) is None
    vs = hl.verdicts_for("cf_conflict0000000", 11)
    assert {v.label for v in vs} == {"A", "C"}


# ---------------------------------------------------------------- 数字自洽

def test_round_summary_matches_shipped_tables():
    assert hl.qa_round_summary() == {
        "QA1": {"A": 3, "B": 4, "C": 19, "U": 5},
        "QA2": {"A": 7, "B": 2, "C": 42, "U": 4},
    }


def test_qa2_strata_sum_and_A_counts_agree_with_labels():
    assert sum(n for n, _a in hl.QA2_STRATA.values()) == len(hl.QA2_LABELS) == 55
    assert sum(a for _n, a in hl.QA2_STRATA.values()) == 7
    assert sum(1 for v in hl.QA2_LABELS.values() if v == "A") == 7


def test_qa2_in_poly_working_point_is_oversampled():
    """in_poly 是 v3.2 唯一会判 Prep 的工作点，选样刻意抽了 12 例。"""
    n, a = hl.QA2_STRATA["in_poly"]
    assert n == 12
    assert a == 4


def test_qa2_confusion_is_internally_consistent():
    c = hl.QA2_CONFUSION
    # 预测正类 = 整个样本里 in_poly=True 的行数，不只是那个专门分层的容量。
    assert c["TP"] + c["FP"] == c["in_poly_rows_in_sample"] == 18
    assert c["TP"] + c["FP"] >= hl.QA2_STRATA["in_poly"][0]
    assert c["precision"] == pytest.approx(c["TP"] / (c["TP"] + c["FP"]), abs=1e-3)
    # raw 规则（取最高峰即判 Prep）在样本上全部触发
    assert c["raw_top_peak_fires"] == len(hl.QA2_LABELS)
    assert c["raw_precision"] == pytest.approx(
        c["raw_top_peak_TP"] / c["raw_top_peak_fires"], abs=1e-3)
    # 混淆矩阵必须覆盖全部已定标（非 U、且有 in_poly 值）的行
    assert c["TP"] + c["FP"] + c["FN"] + c["TN"] == 54


def test_qa2_precision_is_not_better_than_the_criterion_it_replaced():
    """**终止路线的核心论据**：v3.2 相对 v2 没有实质增益。

    QA1 测 v2（"Center 附近有表面"）precision ≈ 3/31 ≈ 0.10；
    QA2 测 v3.2 的 in_poly 工作点 precision = 4/18 ≈ 0.22。
    两者都远不足以自动准入，且样本量下不可区分。
    """
    qa1_precision = 3 / 31
    assert hl.QA2_CONFUSION["in_poly_precision"] < 0.25
    assert hl.QA2_CONFUSION["raw_precision"] < 0.15
    assert abs(hl.QA2_CONFUSION["in_poly_precision"] - qa1_precision) < 0.15


def test_shipped_tables_have_no_cross_round_conflicts():
    assert hl.labels_agree_across_rounds() == []


def test_human_confirmed_prep_rows_are_ten():
    """两轮加起来只有 10 个牙位被人工确认为 A —— 这就是 LocalizationGTSet 的上限。"""
    a_rows = {(v.case_key, v.fdi) for v in hl.all_verdicts() if v.label == "A"}
    assert len(a_rows) == 10
