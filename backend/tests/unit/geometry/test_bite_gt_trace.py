"""item 8：GT-near candidate / cluster / winner 诊断字段（DIAGNOSTIC_ONLY）。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bite_fixture import canonical_world, scanner_inputs, moved, _rigid
from app.geometry.bite_registration import (BiteRegistrationProfile, gt_near_candidate_diagnostics,
                                            preprocess, register_pair)

FIELDS = ("diagnostic_only", "num_raw_candidates", "num_clusters", "winner_support", "runner_up_support",
          "winner_margin", "gt_candidate_generated", "gt_nearest_candidate_index", "gt_nearest_candidate_rank",
          "gt_nearest_candidate_pose_error", "gt_nearest_candidate_cluster_rank", "selected_candidate_rank",
          "selected_candidate_cluster_rank", "selected_cluster_rank", "selected_pose_error_to_gt",
          "gt_cluster_rank", "gt_inside_winner_cluster")


def _clean_pair(profile):
    """干净的单侧配对：拼接 bite 对单侧下颌会被另一侧 patch 污染，追踪必须用单侧 patch。

    ⚠️ GT 记账：patch 上额外加了 `R`，所以 patch→jaw 的真值是 ``A_lower @ inv(R)``，不是 ``A_lower``。
    实测踩到：漏掉这一项时诊断报 9.0°/1.414mm"位姿误差"，而真实情况是 solver 精确复原
    （同一对的质量为 rmse 1.5e-15 / inlier 1.000）——把测试的记账错误算成了算法错误。
    """
    A_lower = _rigid([0.3, 0.7, -0.4], 7.5, [0.8, -1.2, 2.0])
    world = canonical_world(well_posed=True)
    inputs, _ = scanner_inputs(world, A_lower=A_lower)
    R = _rigid([0, 1, 0], 9, [1, 0, -1])
    lower_patch = moved(world["lower_patch_world"], R)
    pair = register_pair(preprocess(lower_patch, profile), preprocess(inputs["lower"], profile), profile)
    gt_patch_to_jaw = np.asarray(A_lower, float) @ np.linalg.inv(np.asarray(R, float))
    return pair, gt_patch_to_jaw


def test_gt_near_diagnostics_fields_and_semantics():
    profile = BiteRegistrationProfile()
    pair, gt_patch_to_jaw = _clean_pair(profile)
    # candidates 与 gt 同为 patch→jaw 方向 ⇒ 不得再取逆
    report = gt_near_candidate_diagnostics(pair, gt_patch_to_jaw, candidates_are_inverse=False)
    print("[gt-trace]", {k: report[k] for k in ("gt_candidate_generated", "gt_nearest_candidate_index",
          "gt_nearest_candidate_rank", "selected_candidate_rank", "num_raw_candidates", "num_clusters",
          "gt_cluster_rank", "selected_cluster_rank", "winner_support", "gt_inside_winner_cluster")})
    print("[gt-trace] gt_nearest_pose_error=%s selected_pose_error=%s"
          % (report["gt_nearest_candidate_pose_error"], report["selected_pose_error_to_gt"]))
    for key in FIELDS:
        assert key in report, "缺字段 %s" % key
    assert report["diagnostic_only"] is True
    assert report["num_raw_candidates"] >= 1 and report["num_clusters"] >= 1
    assert isinstance(report["gt_candidate_generated"], bool)
    assert isinstance(report["gt_inside_winner_cluster"], bool)
    assert report["gt_nearest_candidate_pose_error"]["rotation_deg"] >= 0.0
    assert report["selected_pose_error_to_gt"]["translation_mm"] >= 0.0
    # rank 语义：一律是**求解器排序**（refined 列表序）
    assert report["gt_nearest_candidate_rank"] == report["gt_nearest_candidate_index"] + 1
    # 单侧干净配对：GT 候选必须生成、必须就是求解器首选、winner 与它一致、且落在获胜簇内
    assert report["gt_candidate_generated"] is True, report
    assert report["gt_nearest_candidate_rank"] == 1, report
    assert report["selected_candidate_rank"] == 1, report
    assert report["selected_cluster_rank"] == 1, report
    assert report["gt_cluster_rank"] == 1, report
    assert report["gt_inside_winner_cluster"] is True, report
    assert report["gt_nearest_candidate_pose_error"]["rotation_deg"] < 1.0, report
    assert report["gt_nearest_candidate_pose_error"]["translation_mm"] < 0.5, report


def test_gt_trace_discriminates_generation_and_winner_failures():
    """纯诊断注入（不跑 solver、不改 winner 选择）：验证诊断**有能力**报出类型 1 与类型 3。

    没有这条测试，"三类失败可区分"就只是注释里的一句话。
    """
    _, gt = _clean_pair(BiteRegistrationProfile())
    far = np.eye(4)
    far[:3, 3] = [12.0, -7.5, 4.0]                                   # 远超 near 阈值
    far_matrix = far @ gt                                            # 与 GT 同向但错位

    def synthetic(matrices, cluster_ranks, chosen, clusters):
        return {"refined_matrices": [np.asarray(m, float).tolist() for m in matrices],
                "refined_cluster_rank": list(cluster_ranks), "clusters": clusters,
                "coarse_pool_size": len(matrices), "winner_cluster_support": 1,
                "matrix": np.asarray(chosen, float).tolist()}

    # 类型 3：GT 候选在池里且就在获胜簇内，但求解器首选挑错了 ⇒ rank 不一致
    type3 = gt_near_candidate_diagnostics(
        synthetic([far_matrix, gt], [2, 1], far_matrix, [{"member_count": 1}, {"member_count": 1}]),
        gt, candidates_are_inverse=False)
    assert type3["gt_candidate_generated"] is True, type3
    assert type3["gt_nearest_candidate_rank"] == 2, type3              # GT 候选排在求解器第 2
    assert type3["selected_candidate_rank"] == 1, type3                # winner 是求解器第 1
    assert type3["gt_inside_winner_cluster"] is True, type3            # 且在获胜簇内 ⇒ 类型 3
    assert type3["gt_nearest_candidate_cluster_rank"] == 1, type3

    # 类型 1：池里根本没有 GT-近候选
    type1 = gt_near_candidate_diagnostics(
        synthetic([far_matrix], [1], far_matrix, [{"member_count": 1}]), gt, candidates_are_inverse=False)
    assert type1["gt_candidate_generated"] is False, type1
    assert type1["gt_nearest_candidate_pose_error"]["translation_mm"] > 0.5, type1
    assert type1["selected_candidate_rank"] == 1, type1

    # 类型 2：GT 候选在池里但落在**非**获胜簇（fragmentation / 错误聚类）
    type2 = gt_near_candidate_diagnostics(
        synthetic([gt, far_matrix], [2, 1], far_matrix, [{"member_count": 3}, {"member_count": 1}]),
        gt, candidates_are_inverse=False)
    assert type2["gt_candidate_generated"] is True, type2
    assert type2["gt_nearest_candidate_rank"] == 1, type2
    assert type2["gt_inside_winner_cluster"] is False, type2
    assert type2["gt_cluster_rank"] == 2, type2
