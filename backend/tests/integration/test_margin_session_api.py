"""Margin 提案会话的 API 契约（Margin 步骤重构，2026-10-13）。

用**合成台阶基台**（下粗上细两段圆柱的并集，带明确外缘棱与内角）走真实接口，
不碰真实病例库。六条契约被钉死：

1. 自动候选**不覆盖**正式 Margin（会话里只有候选，design.margin_json 不变、无新 revision）；
2. 多个强候选接近但形状不同 ⇒ `REVIEW_REQUIRED`，且此时**不许直接 confirm**；
3. 加 anchor 后曲线必须**穿过 anchor 的投影**（硬约束，算法不得忽略）；
4. 磁性修线**不改远端**（未被替换的下标逐位相等）；
5. `confirm` 之后才出现正式 Margin revision，且走既有 `record_manual_margin` 语义；
6. **指征感知**：天然牙 ↔ `NATURAL_PREP_MARGIN`、已有实体基台 ↔ `ABUTMENT_CROWN_FINISH_LINE`
   共用同一套能力；只有 `CUSTOM_IMPLANT_ABUTMENT` 被拒（`UNSUPPORTED_INDICATION`），
   无语义的指征（螺丝固位冠）按名字被拒（`BOUNDARY_SEMANTICS_NOT_DEFINED`）；
   `confirm` 之后 revision 必须**自证** boundary_semantics。

外加一条"整体上移/下移不是 world-Z 平移"的语义断言。
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient


def _step_die(max_edge=0.15):
    lo = trimesh.creation.cylinder(radius=3.0, height=1.0, segments=64).apply_translation([0, 0, -0.5])
    up = trimesh.creation.cylinder(radius=2.2, height=1.0, segments=64).apply_translation([0, 0, 0.5])
    body = trimesh.boolean.union([lo, up], engine="manifold")
    if max_edge:
        v, f = trimesh.remesh.subdivide_to_size(body.vertices, body.faces, max_edge=max_edge)
        body = trimesh.Trimesh(v, f, process=True)
    return body


@pytest.fixture()
def client(isolated_db):
    from app.main import app
    return TestClient(app)


def _fixture(client, mesh=None, support_type="prep",indication=None):
    case = client.post("/api/cases", json={"name": "margin-session"}).json()["id"]
    die = mesh if mesh is not None else _step_die(max_edge=0.25)
    r = client.post(f"/api/cases/{case}/assets/support",
                    files={"file": ("die.stl", die.export(file_type="stl"), "model/stl")})
    assert r.status_code == 200, r.text
    design = client.post("/api/designs", json={"case_id": case, "target_fdi": 46,
                                              "support_type": support_type,
                                              "draft": True,'indication':indication}).json()["id"]
    # 会话必须绑定在**已确认轴**上
    assert client.post(f"/api/designs/{design}/axis", json={"axis": [0, 0, 1]}).status_code == 200
    return design


def _displacement_profile(after, base):
    """形状无关的位移剖面：每个 after 点配到最近的 base 点，返回 (距离, 位移向量)。"""
    from scipy.spatial import cKDTree

    d, idx = cKDTree(np.asarray(base, float)).query(np.asarray(after, float), k=1)
    return d, np.asarray(after, float) - np.asarray(base, float)[idx]


def _revisions(client, design) -> list:
    return client.get(f"/api/designs/{design}/revisions").json()["revisions"]


# ---------------------------------------------------------------- 1. 候选不覆盖
def test_detect_creates_proposals_without_touching_official_margin(client):
    design = _fixture(client)
    before = client.get(f"/api/designs/{design}").json()
    r = client.post(f"/api/designs/{design}/margin/detect", json={"feature_mode": "AUTO", "top_k": 1})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["session_id"] and body["source_mesh_sha256"]
    assert body["proposals"], "自动检测应给出至少一个候选"
    assert all(p["rank"] >= 1 for p in body["proposals"])
    assert body["active_proposal_id"] == body["proposals"][0]["proposal_id"]
    assert body["algorithm_version"]
    assert body["release_status"] == "NOT_PRODUCTION_APPROVED"

    after = client.get(f"/api/designs/{design}").json()
    assert after["margin_json"] == before["margin_json"], "候选阶段不得改正式 Margin"
    assert not any(rev["kind"].startswith("margin") for rev in _revisions(client, design)), \
        "候选阶段不得产生 margin revision"


def test_custom_implant_abutment_is_still_rejected(client):
    """只有自定义种植基台被拒：没有确认过的实体基台就不能当边界来源。"""
    design = _fixture(client, support_type="abutment",indication='CUSTOM_IMPLANT_ABUTMENT')
    r = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["status"] == "UNSUPPORTED_INDICATION"
    assert "种植基台" in detail["message"]
    assert detail["indication"] == "CUSTOM_IMPLANT_ABUTMENT"


def test_legacy_implant_abutment_aliases_normalize_to_the_same_rejection():
    """历史别名（IMPLANT_ABUTMENT / …_WITH_SCREW_CHANNEL）归一化到同一族，不新增第二类拒因。

    设计创建接口不接受这两个名字（它们不进 indication 白名单，只可能出现在存量行里），
    所以这里直接钉归一化函数本身。
    """
    from app.services import margin_session as ms

    for legacy in ("IMPLANT_ABUTMENT", "IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL"):
        assert ms.normalize_indication(legacy) == "CUSTOM_IMPLANT_ABUTMENT"
        with pytest.raises(ms.UnsupportedIndication):
            ms.boundary_semantics_for(legacy)
    assert ms.normalize_indication(None) == "NATURAL_PREP"
    assert ms.boundary_semantics_for("CROWN_ON_EXISTING_ABUTMENT") ==         ms.SEMANTICS_ABUTMENT_FINISH_LINE
    assert ms.boundary_semantics_for("NATURAL_PREP") == ms.SEMANTICS_NATURAL


def test_indication_without_boundary_semantics_is_rejected_by_name(client):
    """螺丝固位冠的边界是种植平台，不是基台终止线 —— 不静默按天然牙处理。"""
    design = _fixture(client, support_type="abutment", indication="SCREW_RETAINED_CROWN")
    r = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["status"] == "BOUNDARY_SEMANTICS_NOT_DEFINED"
    assert "边界语义" in detail["message"]


def test_existing_abutment_runs_the_full_boundary_session(client):
    """已有实体基台：detect / Top-K / anchor / magnetic / move / smooth / confirm 全部可用。

    基台检测使用肩部外缘算法；后续共用原有编辑与确认会话。
    """
    support = trimesh.creation.revolve(
        np.array([[0,-4],[1.3,-4],[3,0],[2.2,0],[2.2,4],[0,4]]), sections=128)
    design = _fixture(client, mesh=support, support_type="abutment", indication="CROWN_ON_EXISTING_ABUTMENT")

    detected = client.post(f"/api/designs/{design}/margin/detect",
                           json={"feature_mode": "AUTO", "top_k": 2})
    assert detected.status_code == 200, detected.text
    body = detected.json()
    assert body["boundary_semantics"] == "ABUTMENT_CROWN_FINISH_LINE"
    assert body["boundary_label"] == "基台冠终止线"
    assert body["indication"] == "CROWN_ON_EXISTING_ABUTMENT"
    assert body["proposals"], "已有实体基台同样要给出 Top-K 候选"

    base = np.asarray(body["active_points"], float)
    anchor = [2.6, 0.0, 0.1]                      # 肩台上表面附近

    anchored = client.post(f"/api/designs/{design}/margin/add-anchor",
                           json={"anchors": [anchor], "feature_mode": "AUTO"})
    assert anchored.status_code == 200, anchored.text
    assert anchored.json()["boundary_semantics"] == "ABUTMENT_CROWN_FINISH_LINE"

    curve = np.asarray(anchored.json()["active_points"], float)
    sel = curve[curve[:, 1] > 0.8 * curve[:, 1].max()][:6]
    stroke = (sel * np.array([1.25, 1.25, 1.0])).tolist()   # 贴着曲线外偏，保证被接受
    replaced = client.post(f"/api/designs/{design}/margin/replace-segment",
                           json={"stroke": stroke, "radius_mm": 1.0})
    assert replaced.status_code == 200, replaced.text

    assert client.post(f"/api/designs/{design}/margin/move",
                       json={"delta_mm": 0.1}).status_code == 200
    assert client.post(f"/api/designs/{design}/margin/smooth",
                       json={"strength": 0.4}).status_code == 200

    before = client.get(f"/api/designs/{design}").json()
    confirmed = client.post(f"/api/designs/{design}/margin/confirm",
                            json={"proposal_id": replaced.json()["active_proposal_id"]})
    assert confirmed.status_code == 200, confirmed.text
    result = confirmed.json()
    assert result["boundary_semantics"] == "ABUTMENT_CROWN_FINISH_LINE"
    assert result["boundary_label"] == "基台冠终止线"
    assert result["revision_no"] >= 1

    after = client.get(f"/api/designs/{design}").json()
    assert after["margin_json"] != before["margin_json"], "confirm 之后候选才成为正式边界"

    revision = next(rev for rev in _revisions(client, design)
                    if rev["kind"] == "margin_manual" and rev["is_current"])
    assert revision["payload"]["boundary_semantics"] == "ABUTMENT_CROWN_FINISH_LINE", \
        "revision 必须自证这条边界是什么（不能靠 support_type 事后猜）"
    assert revision["operator_actions"][0]["boundary_semantics"] == "ABUTMENT_CROWN_FINISH_LINE"


def test_natural_tooth_confirm_records_natural_prep_margin_semantics(client):
    """天然牙保持现有 NATURAL_PREP_MARGIN，一字不改。"""
    design = _fixture(client)
    detected = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1}).json()
    assert detected["boundary_semantics"] == "NATURAL_PREP_MARGIN"
    assert detected["boundary_label"] == "天然牙边缘线"

    confirmed = client.post(f"/api/designs/{design}/margin/confirm", json={})
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["boundary_semantics"] == "NATURAL_PREP_MARGIN"

    revision = next(rev for rev in _revisions(client, design)
                    if rev["kind"] == "margin_manual" and rev["is_current"])
    assert revision["payload"]["boundary_semantics"] == "NATURAL_PREP_MARGIN"


def test_unconfirmed_scanned_implant_part_cannot_be_the_boundary_source(client):
    """已有实体基台也要**先确认**：扫描定位件（implant_candidate 且未确认）仍然 409。"""
    from app.api import routes as api_routes

    design = _fixture(client, support_type="abutment", indication="CROWN_ON_EXISTING_ABUTMENT")
    detail = client.get(f"/api/designs/{design}").json()
    support = api_routes.asset(detail["case_id"], "support")
    with api_routes.db() as connection:
        connection.execute("UPDATE assets SET mesh_info_json=? WHERE id=?",
                           (json.dumps({"implant_candidate": {"confidence": "high"},
                                        "restorative_abutment_confirmed": False}),
                            support["id"]))
    r = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    assert r.status_code == 409, r.text
    assert "修复基台" in r.json()["detail"]


# ---------------------------------------------------------------- 2. 歧义
def test_ambiguity_flags_force_review_required_and_block_direct_confirm(client, monkeypatch):
    """两个强候选分数接近但形状不同 ⇒ REVIEW_REQUIRED，且**不许**直接 confirm。"""
    from app.services import margin_session as ms

    design = _fixture(client)
    detected = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1}).json()
    session = ms.STORE.get(design)
    assert session is not None

    base = session.proposals[0].points
    shifted = [[p[0], p[1], p[2] + 1.0] for p in base]        # 同分但形状差 1mm
    session.add(ms.MarginProposal(proposal_id="shifted", points=shifted, score=base and 1.0,
                                  rank=0, source="MODE_VARIANT", status="MEASURED"))
    session.proposals[0].score = 1.0
    session.proposals[1].score = 0.99
    session.ambiguity_flags = [
        "STRONG_CANDIDATES_DISAGREE(score %.3f vs %.3f, p95 %.3f mm)" % (1.0, 0.99, 1.0)]
    session.status = "REVIEW_REQUIRED"

    blocked = client.post(f"/api/designs/{design}/margin/confirm", json={})
    assert blocked.status_code == 409, blocked.text
    assert "并列强候选" in blocked.json()["detail"]

    # 加了纠偏点之后允许确认（用户已经给出判断）
    anchor = base[0]
    added = client.post(f"/api/designs/{design}/margin/add-anchor", json={"anchors": [anchor]})
    assert added.status_code == 200, added.text
    confirmed = client.post(f"/api/designs/{design}/margin/confirm", json={})
    assert confirmed.status_code == 200, confirmed.text
    assert detected["proposals"][0]["n_points"] > 0


# ---------------------------------------------------------------- 3. anchor 硬约束
def test_anchor_is_a_hard_constraint(client):
    design = _fixture(client)
    session = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1}).json()
    # 取一个明显偏离当前曲线的点做 anchor（在台阶上表面附近）
    anchor = [2.6, 0.0, 0.9]
    r = client.post(f"/api/designs/{design}/margin/add-anchor", json={"anchors": [anchor]})
    assert r.status_code == 200, r.text
    body = r.json()
    points = np.asarray(body["active_points"], float)
    assert 0 < len(points) <= session["proposals"][0]["n_points"] + 1, "点数应基本保持"

    from app.geometry.margin_edit import project_to_surface
    projected = project_to_surface(_step_die(), np.asarray([anchor], float))[0]
    distance = np.min(np.linalg.norm(points - projected, axis=1))
    assert distance <= 1e-6 + 1e-3, "曲线必须逐点穿过 anchor 的投影（硬约束），实测 %.4f mm" % distance
    assert body["anchors"] == [anchor]


def test_anchor_count_is_capped(client):
    design = _fixture(client)
    client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    many = [[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [-2.0, 0.0, 0.5], [0.0, -2.0, 0.5], [1.4, 1.4, 0.5]]
    r = client.post(f"/api/designs/{design}/margin/add-anchor", json={"anchors": many})
    assert r.status_code == 422, r.text          # 上限 4 个（模型层直接拦）


# ---------------------------------------------------------------- 4. 磁性修线不改远端
def test_magnetic_segment_keeps_far_side_bit_identical(client):
    design = _fixture(client, mesh=_smooth_cone_die())
    before = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1}).json()
    base = np.asarray(before["active_points"], float)
    # stroke 必须**贴着当前曲线**画（用曲线自己 +Y 侧的点向外偏 0.3mm 当 stroke）
    sel = base[base[:, 1] > 0.8 * base[:, 1].max()][:6]
    stroke = (sel * np.array([1.25, 1.25, 1.0])).tolist()      # 明显外偏，避免被吸附抹平
    r = client.post(f"/api/designs/{design}/margin/replace-segment",
                    json={"stroke": stroke, "radius_mm": 1.0})
    assert r.status_code == 200, r.text
    after = np.asarray(r.json()["active_points"], float)
    assert 0 < len(after) <= len(base)          # 允许入库去重带来的点数百位级变化
    assert r.json()["operations"][-1]["action"] == "segment_replace"

    # 远端（与 stroke 相对的 -Y 半侧）必须**逐位不变**：用集合成员判定，
    # 不依赖索引（去重会让索引整体位移）
    far = base[base[:, 1] < -1.0]
    assert len(far), "夹具应保证存在远端"
    for point in far:
        distance = np.min(np.linalg.norm(after - point, axis=1))
        assert distance < 1e-9, "磁性修线不得改动远端 Margin（实测偏移 %.6f mm）" % distance
    moved_distance, _ = _displacement_profile(after, base)
    assert moved_distance.max() > 1e-3, "被 stroke 覆盖的区间应当被改动（实测无点位移）"


def test_stroke_far_from_margin_is_refused(client):
    design = _fixture(client)
    client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    r = client.post(f"/api/designs/{design}/margin/replace-segment",
                    json={"stroke": [[9.0, 9.0, 9.0], [9.5, 9.5, 9.0]], "radius_mm": 0.5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "REVIEW_REQUIRED", "stroke 没碰到 Margin 时必须如实报 REVIEW"
    assert any("SEGMENT_REPLACE" in flag for flag in body["ambiguity_flags"])


# ---------------------------------------------------------------- 5. confirm 才写 revision
def test_confirm_is_the_only_way_to_create_a_margin_revision(client):
    design = _fixture(client)
    client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1})
    client.post(f"/api/designs/{design}/margin/move", json={"delta_mm": 0.1})
    client.post(f"/api/designs/{design}/margin/smooth", json={"strength": 0.4})
    assert not any(rev["kind"].startswith("margin") for rev in _revisions(client, design))

    # 合成台阶基台会让 AUTO 与特征模式给出并列强候选 ⇒ 必须**指名**候选（人工判断）
    detected = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 3}).json()
    chosen = detected["proposals"][0]["proposal_id"]
    confirmed = client.post(f"/api/designs/{design}/margin/confirm",
                            json={"proposal_id": chosen})
    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["margin_source"] == "manual"
    assert body["operator_selected_proposal"] is True
    assert body["revision_no"] >= 1

    revisions = _revisions(client, design)
    margin_revisions = [rev for rev in revisions if rev["kind"].startswith("margin")]
    assert len(margin_revisions) == 1, margin_revisions
    assert json.loads(client.get(f"/api/designs/{design}").json()["margin_json"]) == body["points"]

    # 会话已关闭：再 confirm 必须拒绝（不能拿旧会话重复写）
    assert client.post(f"/api/designs/{design}/margin/confirm", json={}).status_code == 409


# ---------------------------------------------------------------- 6. 移动语义
def _smooth_cone_die(max_edge=0.25):
    """光滑锥台（没有棱）：Margin 落在斜面上，"沿表面移动"才有唯一定义。"""
    cone = trimesh.creation.cone(radius=3.0, height=4.0, sections=64)
    cone.apply_translation([0, 0, 2.0])
    base = trimesh.creation.cylinder(radius=3.0, height=1.0, segments=64).apply_translation([0, 0, -0.5])
    body = trimesh.boolean.union([base, cone], engine="manifold")
    if max_edge:
        v, f = trimesh.remesh.subdivide_to_size(body.vertices, body.faces, max_edge=max_edge)
        body = trimesh.Trimesh(v, f, process=True)
    return body


def test_move_either_applies_a_healthy_ring_or_refuses(client):
    """移动的**通用不变量**：要么产出一条健康的曲线，要么如实拒绝且不改动候选。

    （"棱上沿表面移动无唯一定义"只是拒绝的触发条件之一；把它写成状态断言会与夹具耦合，
    这里钉的是真正要保证的事：**绝不交一条被压扁/带重复点的 Margin**。）
    """
    design = _fixture(client)
    before = np.asarray(client.post(f"/api/designs/{design}/margin/detect",
                                    json={"top_k": 1}).json()["active_points"], float)
    r = client.post(f"/api/designs/{design}/margin/move", json={"delta_mm": 0.3})
    assert r.status_code == 200, r.text
    body = r.json()
    after = np.asarray(body["active_points"], float)
    if body["status"] == "REVIEW_REQUIRED":
        assert any("MOVE" in flag or "COLLAPSED" in flag for flag in body["ambiguity_flags"])
        assert after.shape == before.shape, "拒绝时不得改动候选"
    else:
        spacing = np.linalg.norm(np.roll(after, -1, axis=0) - after, axis=1)
        assert spacing.min() >= 0.02, "结果必须满足 Margin 校验器的相邻间距"
        assert len({tuple(np.round(p, 5)) for p in after}) == len(after), "结果不得有重复点"


def test_move_is_along_surface_not_world_z(client):
    design = _fixture(client, mesh=_smooth_cone_die())
    before = client.post(f"/api/designs/{design}/margin/detect", json={"top_k": 1}).json()
    base = np.asarray(before["active_points"], float)
    r = client.post(f"/api/designs/{design}/margin/move", json={"delta_mm": 0.3})
    assert r.status_code == 200, r.text
    after = np.asarray(r.json()["active_points"], float)
    body = r.json()
    operations = body["operations"]
    direction = " ".join(str(op.get("direction") or op.get("reason") or "") for op in operations)
    # 方向语义必须写清楚是"沿局部表面/插入轴"，不是 world-Z（world-Z 会让 X/Y 位移恒为 0）
    assert "local_surface_projected" in direction or "AMBIGUOUS" in direction, direction

    if body["status"] == "REVIEW_REQUIRED":
        # 颈缘常落在颈缘棱上：该处"沿表面"无唯一定义时就该如实拒绝，且不改动候选
        assert after.shape == base.shape
        return
    distance, delta = _displacement_profile(after, base)
    assert distance.max() > 1e-4, "应用成功却一个点都没动"
    assert distance.max() < 1.0, "移动量应在合理范围（不是整段飞走）"
    tangential = np.linalg.norm(delta[:, :2], axis=1)
    assert tangential.max() > 1e-9 or distance.max() < 0.05,         "沿局部表面的移动不应退化成 world-Z（除非位移极小）"
