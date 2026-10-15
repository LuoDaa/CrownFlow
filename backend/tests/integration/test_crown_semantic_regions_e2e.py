"""Crown 语义区域自动建立 + 邻牙/对颌适配在新病例上直接可用（§13 的 10 条）。

全部走真实 API：`ready` 夹具建病例 → margin/axis/profile → 确认坐标帧 → /outer/auto 生成牙冠，
之后**不做任何人工语义标注**。

两个夹具各有用处：
- `ready`：原始夹具（一个 2 环的裸圆柱冠）。它故意很稀疏 ⇒ 轴向壁上没有顶点，
  用来验证"区域为空时如实报 NO_APPLICABLE_SIDE，而不是假装成功"。
- `dense`：把模板细分后重新放置，再加一个**真的与冠穿插**的对颌 —— 这才是真实牙冠的处境，
  用来验证自动区域、邻牙/对颌适配、接触图重算。
"""
import json
import numpy as np
import pytest
import trimesh

from .test_auto_fit import ready, prepare, confirm  # noqa: F401  夹具复用


def margin_points(count=32, radius=2.0):
    theta = np.arange(count) * 2 * np.pi / count
    return np.c_[radius * np.cos(theta), radius * np.sin(theta), np.zeros(count)]


@pytest.fixture
def dense(ready):  # noqa: F811
    """细分后的牙冠模板（真实牙冠的顶点密度）。"""
    from app.config import TEMPLATE_DIR
    client, path, _, neighbors = ready
    template = trimesh.creation.cylinder(radius=3, height=6, sections=32)
    template.apply_translation([0, 0, 2])
    template = template.subdivide().subdivide()
    template.export(TEMPLATE_DIR / "FDI_46.stl")
    margin = margin_points()
    placed = client.post(path + "/outer/auto", json={"margin_points": margin.tolist(), "insertion_axis": [0, 0, 1]})
    assert placed.status_code == 200, placed.text
    return client, path, neighbors


def session(client, path, regions=False):
    response = client.get(path + "/sculpt-session" + ("?regions=1" if regions else ""))
    assert response.status_code == 200, response.text
    return response.json()


def outer_path(client, path):
    return client.get(path).json()["outer_path"]


def edit(client, path, body):
    return client.post(path + "/edit-outer", json={**body, "base_outer_path": outer_path(client, path)})


def confirm_bite(client, path, z_offset=-0.5):
    """确认静态咬合位姿。

    ``z_offset=-0.5`` 把对颌盒下移 0.5 mm ⇒ 冠顶（咬合面）陷入 0.5 mm：夹具**真的有穿透**，
    且深度落在咬合域内（``-1.0`` 时弹性带会被拉过对颌底面，触发防劣化门并回滚）。
    """
    matrix = np.eye(4)
    matrix[2, 3] = z_offset
    return confirm(client, path, "bite", {"transform": matrix.tolist(),
                                          "source_method": "REGISTERED_EXPORT", "units": "mm"})


def antagonist_penetration(report):
    """对颌穿透 = max(0, -最小有符号距离)；口径与后端测量服务一致。"""
    entry = report["contacts"]["reports"].get("antagonist")
    if not entry or entry.get("signed_min_mm") is None:
        return None
    return max(0.0, -float(entry["signed_min_mm"]))


def last_payload(client, path, operation):
    payloads = client.get(path + "/revisions").json()["revisions"]
    for revision in reversed(payloads):
        if revision["payload"].get("operation") == operation:
            return revision["payload"]
    raise AssertionError("没有找到 operation=%s 的修订" % operation)


# ────────────────────────────────── Test 1 / 2
def test_generated_crown_has_occlusal_mesial_distal_without_manual_annotation(dense):  # noqa: F811
    client, path, neighbors = dense
    report = session(client, path)
    region_map = report["region_map"]
    assert report["manual_region_names"] == [], "新病例不应该有任何人工标注"
    assert report["manual_annotation_required"] is False
    counts = region_map["counts"]
    domains = region_map["domain_counts"]
    assert counts["occlusal"] > 0 and domains["occlusal"] > 0, counts
    assert counts["mesial"] > 0 and domains["mesial"] > 0, counts
    assert counts["distal"] > 0 and domains["distal"] > 0, counts
    assert counts["margin"] > 0 and report["region_map"]["protected_vertex_count"] > 0, counts
    # ⚠️ inner / cervical 的**标签计数**可以为 0：本夹具的冠环只落在 z=0/1.5/3/4.5/6，
    # 颈缘带内没有顶点（不是失败）。保护口径的"无遗漏"由单元测试的覆盖不变量来钉。
    assert np.asarray(region_map["axes"]["mesial"]).shape == (3,)
    assert "MESIAL_REGION_EMPTY" not in region_map["flags"]
    assert "DISTAL_REGION_EMPTY" not in region_map["flags"]


def test_occlusal_excludes_margin_inner_and_protected_vertices(dense):  # noqa: F811
    client, path, _ = dense
    report = session(client, path, regions=True)
    weights = report["region_weights"]
    protection = np.asarray(report["region_protection"])
    occlusal = np.asarray(weights["occlusal"])
    assert (occlusal >= 0.5).any()
    assert not (occlusal >= 0.5)[protection <= 0].any(), "咬合域碰到了保护顶点"
    assert not (occlusal >= 0.5)[np.asarray(weights["margin"]) >= 0.5].any(), "咬合域包含边缘带"
    assert not (occlusal >= 0.5)[np.asarray(weights["inner"]) >= 0.5].any(), "咬合域包含内表面"
    assert (protection <= 0).any() and report["region_map"]["protected_vertex_count"] > 0


def test_protected_regions_are_locked_against_every_edit(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    report = session(client, path, regions=True)
    before = np.asarray(report["vertices"])
    locked = np.asarray(report["region_protection"]) <= 0
    assert locked.any()
    center = [before[locked].mean(axis=0).tolist()]
    response = edit(client, path, {"operation": "remove", "centers": center, "radius_mm": 3.0, "strength_mm": 0.2})
    assert response.status_code == 200, response.text
    after = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    np.testing.assert_allclose(after[locked], before[locked], atol=0)
    assert np.linalg.norm(after - before, axis=1).max() > 0, "外表面确实被改动了"


# ────────────────────────────────── Test 3 / 4
def test_adapt_neighbors_works_without_manual_region(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    assert session(client, path)["manual_region_names"] == []
    before = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    response = edit(client, path, {"operation": "adapt_mesial", "target_distance_mm": 0.05})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "APPLIED"
    payload = last_payload(client, path, "adapt_mesial")
    assert payload["status"] == "APPLIED"
    assert payload["region_report"]["method"] == "crown_regions/local_frame/1"
    assert payload["region_report"]["counts"]["mesial"] > 0
    assert payload["margin_displacement_mm"] == 0.0, "保护顶点位移必须为 0"
    assert payload["adapt_steps"][0]["status"] == "PASS_ENGINEERING"
    chain = payload["proximal_coarse_fine"][0]
    assert chain["patch_validity"]["status"] == "VALID"
    assert chain["after_fine"]["domain_signed_min_mm"] <= .10
    assert chain["after_fine"]["margin_drift_mm"] < .01
    assert abs(chain["after_fine"]["crown_height_mm"] - chain["before"]["crown_height_mm"]) < .05
    after = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    assert abs(after[:, 2].max() - before[:, 2].max()) < 0.05
    assert np.linalg.norm(after - before, axis=1).max() > 0


def test_missing_neighbor_side_is_reported_not_an_error(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)          # distal 被标成 ABSENT
    assert json.loads(client.get(path).json()["neighbors_json"])["distal"]["confirmation_status"] == "ABSENT"
    before = outer_path(client, path)
    revisions = client.get(path + "/revisions").json()["n_revisions"]
    response = edit(client, path, {"operation": "adapt_distal", "target_distance_mm": 0.05})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "NO_APPLICABLE_SIDE"
    assert response.json()["edit"]["skipped_sides"][0] == {"side": "distal", "status": "NO_NEIGHBOR",
                                                           "reason": "SKIPPED_ABSENT"}
    assert outer_path(client, path) == before, "没有邻牙不得改动几何"
    assert client.get(path + "/revisions").json()["n_revisions"] == revisions, "不得为空改动写修订"
    # 另一侧照常可用
    assert edit(client, path, {"operation": "adapt_mesial", "target_distance_mm": 0.05}).status_code == 200


def test_sparse_crown_reports_empty_region_instead_of_faking_success(ready):  # noqa: F811
    """裸 2 环圆柱冠没有轴向壁顶点：自动区域如实报空，适配返回 NO_APPLICABLE_SIDE。"""
    client, path, _, neighbors = ready
    prepare(client, path, neighbors)
    region_map = session(client, path)["region_map"]
    assert "MESIAL_REGION_EMPTY" in region_map["flags"] or region_map["counts"]["mesial"] == 0
    assert "DISTAL_REGION_EMPTY" in region_map["flags"]
    before = outer_path(client, path)
    response = edit(client, path, {"operation": "adapt_mesial", "target_distance_mm": 0.05})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "NO_APPLICABLE_SIDE"
    assert outer_path(client, path) == before


# ────────────────────────────────── Test 5 / 6 / 7
def test_adapt_antagonist_works_without_manual_occlusal_and_reduces_penetration(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    assert confirm_bite(client, path).status_code == 200
    assert session(client, path)["manual_region_names"] == []
    penetration_before = antagonist_penetration(session(client, path))
    assert penetration_before is not None and penetration_before > 0, \
        "夹具必须先造成真实穿插，否则这条测试没有意义：%s" % penetration_before
    before = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    response = edit(client, path, {"operation": "adapt_antagonist", "target_distance_mm": 0.05})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "APPLIED"
    payload = last_payload(client, path, "adapt_antagonist")
    assert payload["region_report"]["counts"]["occlusal"] > 0
    penetration_after = antagonist_penetration(session(client, path))
    assert penetration_after < penetration_before, (penetration_before, penetration_after)
    # Test 7：保护顶点位移 ≈ 0
    protection = np.asarray(session(client, path, regions=True)["region_protection"])
    locked = protection <= 0
    after = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    np.testing.assert_allclose(after[locked], before[locked], atol=1e-9)
    assert payload["margin_displacement_mm"] == 0.0
    assert len(after) == len(before)


# ────────────────────────────────── Test 8 / 9 / 10
def test_contact_map_is_recalculated_after_crown_update(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    assert confirm_bite(client, path).status_code == 200
    before_contacts = session(client, path)["contacts"]
    assert before_contacts["status"] == "MEASURED", before_contacts
    assert edit(client, path, {"operation": "adapt_antagonist", "target_distance_mm": 0.05}).status_code == 200
    payload = last_payload(client, path, "adapt_antagonist")
    assert payload["contact_before"]["reports"]["antagonist"] != payload["contact_after"]["reports"]["antagonist"], \
        "接触账必须在新的冠形上重算"
    after_contacts = session(client, path)["contacts"]
    assert after_contacts["reports"]["antagonist"]["signed_min_mm"] != \
        before_contacts["reports"]["antagonist"]["signed_min_mm"]
    assert after_contacts["reports"]["antagonist"]["signed_min_mm"] > \
        before_contacts["reports"]["antagonist"]["signed_min_mm"]


def test_free_forming_still_works_after_automatic_regions(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    report = session(client, path, regions=True)
    before = np.asarray(report["vertices"])
    protection = np.asarray(report["region_protection"])
    occlusal = np.asarray(report["region_weights"]["occlusal"]) >= 0.5
    center = [before[occlusal].mean(axis=0).tolist()]
    response = edit(client, path, {"operation": "add", "centers": center, "radius_mm": 1.5, "strength_mm": 0.05})
    assert response.status_code == 200, response.text
    after = np.asarray(trimesh.load(outer_path(client, path), force="mesh").vertices)
    assert np.linalg.norm(after - before, axis=1).max() > 0
    np.testing.assert_allclose(after[protection <= 0], before[protection <= 0], atol=0)
    # 笔刷 + 语义区域变换都要能用**自动**区域（没有任何人工标注）
    response = edit(client, path, {"operation": "region_transform", "semantic_region": "mesial",
                                   "translation": [0.01, 0, 0], "frame_angle_deg": 0})
    assert response.status_code == 200, response.text
    assert last_payload(client, path, "region_transform")["semantic_region_source"] == "AUTO_GEOMETRY"
    assert session(client, path)["manual_region_names"] == []


def test_finalize_and_stl_export_still_work(dense):  # noqa: F811
    client, path, neighbors = dense
    prepare(client, path, neighbors)
    assert confirm_bite(client, path).status_code == 200
    status = client.get(path + "/status").json()
    request = {"expected_input_digest": status["input_digest"], **status["confirmed_revisions"]}
    response = client.post(path + "/finalize", json=request)
    assert response.status_code == 200, response.text
    assert response.json()["qc"]
    stl = client.get(path + "/final.stl")
    assert stl.status_code == 200 and len(stl.content) > 1000
