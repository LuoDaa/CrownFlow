"""QA2 渲染器的选样与标注逻辑测试（纯计算，不读真实数据）。

选样是**结论的一部分**：分层容量与实际样本数的差别必须可复现，
否则"55 例而不是 68 例"会被误读成抽样失误。
"""
import pytest

from app.groundtruth import qa2_render as q2


def _lat(i, cls="FULL_ARCH_WITH_TARGET", score=5.0, prom=5.0, poly="False",
         basis="GEOMETRY_FULL_ARCH"):
    return {"case_uuid": "cf_case%03d" % i, "fdi": str(10 + (i % 40)),
            "asset_id": "asset%03d" % i, "candidate_class": cls,
            "classification_basis": basis, "score": score, "prom": prom,
            "cov": 1.0, "peak1_in_poly": poly,
            "target_surface_distance": "1.0"}


def _join(rows):
    """造出与 render() 里同形的 features 索引。"""
    feat = {}
    for r in rows:
        feat[(r["case_uuid"], r["fdi"], r["asset_id"])] = {
            "profile_r": "1.0;2.0", "profile_z": "0.0;1.0",
            "asset_path_local_only": "x.stl"}
    return feat


def test_select_caps_at_55_and_dedups():
    rows = [_lat(i, poly="True" if i % 3 == 0 else "False") for i in range(400)]
    sel = q2.select(rows, _join(rows))
    assert len(sel) <= q2.CAP == 55
    keys = [(lr["case_uuid"], lr["fdi"]) for _t, lr, _f in sel]
    assert len(keys) == len(set(keys)), "同一 (case_uuid, fdi) 不得重复"


def test_select_oversamples_the_in_poly_working_point():
    """in_poly 是 v3.2 唯一会判 Prep 的工作点，必须被抽满。

    注意构造要贴近真实语料：真实数据里高分档（A_high）大多是 poly=False
    （最高分那几例恰恰是落错牙的），所以 in_poly 分层才有独立样本可用。
    """
    rows = []
    for i in range(400):
        poly = "True" if i >= 40 else "False"     # 高分档 poly=False
        rows.append(_lat(i, poly=poly, score=9.0 - i * 0.01))
    sel = q2.select(rows, _join(rows))
    tags = [t for t, _lr, _f in sel]
    assert tags.count("in_poly") == 14            # 分层容量抽满
    for t, lr, _f in sel:
        if t == "in_poly":
            assert lr["peak1_in_poly"] == "True"


def test_select_weak_and_missing_strata_are_empty_when_corpus_has_none():
    """**这正是 QA2 只出 55 例而不是 68 例的原因**：
    没有 prom<1.0 的 WITH_TARGET 行，TARGET_MISSING 也只有极少数。"""
    rows = [_lat(i, prom=5.0) for i in range(50)]
    rows += [_lat(900, cls="FULL_ARCH_TARGET_MISSING", prom=2.3)]
    sel = q2.select(rows, _join(rows))
    tags = [t for t, _lr, _f in sel]
    assert "v2pos_weak" not in tags
    assert tags.count("miss_strong") == 1


def test_select_puts_metadata_basis_under_meta_tag():
    # METADATA 行放低分档，且避开 mid 分层（wt[30:42]）已消费的名次，
    # 否则去重会把它们吃掉 —— 真实语料里 meta 也是这种"只有 meta 分层要"的行。
    rows = [_lat(i, score=9.0 - i * 0.01,
                 basis="METADATA_UNSEGMENTED_JAW" if 45 <= i < 50
                 else "GEOMETRY_FULL_ARCH")
            for i in range(60)]
    sel = q2.select(rows, _join(rows))
    assert [t for t, _lr, _f in sel].count("meta") == 5


def test_select_excludes_rows_without_a_rendered_feature_row():
    rows = [_lat(i) for i in range(10)]
    assert q2.select(rows, {}) == []


# ---------------------------------------------------------------- 标注

class _Top:
    """可寻址的假图像行（grid 为二维 list）。"""
    def __init__(self, grid=200):
        self.g = [[0] * grid for _ in range(grid)]


def test_mark_polygon_draws_a_dotted_outline_and_reports_point_count():
    grid = 60
    top = _Top(grid)
    ring = [[3.0 * __import__("math").cos(t), 3.0 * __import__("math").sin(t), 0.0]
            for t in [i * 0.2 for i in range(32)]]
    n = q2._mark_polygon(top.g, ring, (0.0, 0.0, 0.0), [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0], grid, 12.0)
    assert n == len(ring)
    lit = sum(1 for row in top.g for v in row if v == 255)
    assert lit > 0, "轮廓应当画出了像素"
    # 点线：不是每一个边界像素都点亮
    assert lit < 4 * grid


def test_mark_polygon_is_a_noop_below_three_points():
    grid = 60
    top = _Top(grid)
    assert q2._mark_polygon(top.g, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                            (0.0, 0.0, 0.0), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                            grid, 12.0) == 0
    assert all(v == 0 for row in top.g for v in row)


def test_mark_peaks_skips_empty_and_marks_present():
    grid = 60
    top = _Top(grid)
    q2._mark_peaks(top.g, {"peak1_du": "", "peak1_dv": "",
                           "peak2_du": None, "peak2_dv": "1.0"}, grid, 12.0)
    assert all(v == 0 for row in top.g for v in row), "空值不应画任何像素"
    q2._mark_peaks(top.g, {"peak1_du": "0.0", "peak1_dv": "0.0"}, grid, 12.0)
    assert sum(1 for row in top.g for v in row if v == 255) > 0


def test_profile_tile_handles_empty_and_draws_a_curve():
    blank = q2.profile_tile([], [])
    assert len(blank) == 200 and all(v == 255 for v in blank[0])
    tile = q2.profile_tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                           [0.0, 1.0, 2.0, 1.0, 0.0, -1.0])
    assert sum(1 for row in tile for v in row if v == 0) > 0


def test_strata_capacity_matches_the_documented_selection():
    caps = dict(q2.STRATA)
    assert caps["A_high"] == 10 and caps["in_poly"] == 14
    assert sum(caps.values()) == 85, "容量合计（去重前）"
    assert q2.CAP == 55 and q2.PER_SHEET == 5
