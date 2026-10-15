"""Crown 语义区域自动生成（CrownRegionMap）单元测试：局部系、权重形状、保护口径。"""
import json
import numpy as np
import pytest
import trimesh

from app.geometry.crown_regions import (REGION_NAMES, RegionMapError, build_region_map,
                                        region_weights, resolve_anatomical_axes)
from app.geometry.crown_outer.auto_fit_adapter import FitSurface
from app.geometry.crown_outer.patient_frame import PatientFrame, build_patient_frame
from app.geometry.sculpt import cervical_weights


def ring(radius=2.0, count=32, z=0.0):
    theta = np.arange(count) * 2 * np.pi / count
    return np.c_[radius * np.cos(theta), radius * np.sin(theta), np.full(count, z)]


def corrugated_crown(height=6.0, radius=3.0, rings=24, sections=48):
    """带环向起伏的冠：既有轴向环（能分层），又有颊舌/近远中差异（能分侧）。"""
    theta = np.linspace(-np.pi, np.pi, sections, endpoint=False)
    z = np.linspace(0.0, height, rings)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    r = radius + 0.35 * np.cos(2 * tt) + 0.20 * np.sin(3 * tt) * (zz / height)
    vertices = np.c_[r.ravel() * np.cos(tt).ravel(), r.ravel() * np.sin(tt).ravel(), zz.ravel()]
    faces = []
    for i in range(sections):
        for j in range(rings - 1):
            a = (i % sections) * rings + j
            b = a + 1
            c = ((i + 1) % sections) * rings + j + 1
            d = ((i + 1) % sections) * rings + j
            faces += [[a, b, c], [a, c, d]]
    bottom = len(vertices)
    vertices = np.vstack([vertices, [0, 0, 0]])
    top = len(vertices)
    vertices = np.vstack([vertices, [0, 0, height]])
    for i in range(sections):
        faces.append([bottom, ((i + 1) % sections) * rings, i * rings])
        faces.append([top, i * rings + rings - 1, ((i + 1) % sections) * rings + rings - 1])
    mesh = trimesh.Trimesh(vertices, np.asarray(faces, int), process=True)
    mesh.fix_normals()
    return mesh


def design_for(margin, fdi=46):
    return {"margin_json": json.dumps(np.asarray(margin).tolist()), "target_fdi": fdi}


FRAME = PatientFrame(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     revision_id="frame-r1")
REF = {"buccal": [0, 1, 0], "revision_id": "frame-r1", "status": "HUMAN_CONFIRMED_ANATOMICAL_DIRECTIONS"}


def test_region_names_and_continuous_weights():
    mesh = corrugated_crown()
    margin = ring()
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF)
    assert tuple(region_map.weights) == REGION_NAMES, "8 个语义区域必须全部存在"
    for name, weights in region_map.weights.items():
        assert weights.shape == (len(mesh.vertices),), name
        assert np.isfinite(weights).all() and weights.min() >= 0.0 and weights.max() <= 1.0, name
        # 连续权重：只要区域非空，就不能退化成二值 mask
        if weights.max() > 0:
            assert np.unique(np.round(weights, 6)).size > 2, name
    counts = region_map.counts()
    for name in ("occlusal", "mesial", "distal"):
        assert counts[name] > 0, "%s 自动区域为空：%s" % (name, counts)
    # 这个夹具的冠底面正好落在边缘线上，没有龈下/内表面几何 ⇒ inner 合法为空（不是失败）
    assert counts["inner"] == 0
    assert region_map.report()["flags"] is not None


def test_occlusal_excludes_margin_and_protected_region():
    mesh = corrugated_crown()
    margin = ring()
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF)
    protection = np.asarray(region_map.protection)
    occlusal = region_map.domain("occlusal")
    assert occlusal.any()
    assert not occlusal[protection <= 0].any(), "咬合域不得包含被保护的颈缘/边缘/内表面顶点"
    assert not occlusal[np.asarray(region_map.weights["margin"]) >= 0.5].any()
    assert not occlusal[np.asarray(region_map.weights["inner"]) >= 0.5].any()
    # 咬合域集中在冠方：其最低高度明显高于边缘线
    from app.geometry.nonplanar_crown import MarginCurtain
    height = MarginCurtain(margin, FRAME.z).axial_above(mesh.vertices)
    assert height[occlusal].min() > 0.5


def test_protection_is_the_single_existing_implementation():
    mesh = corrugated_crown()
    margin = ring()
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF)
    # 保护权重就是既有唯一实现（不复制、不改写）
    np.testing.assert_allclose(region_map.protection, cervical_weights(mesh, margin, FRAME.z), atol=0)
    # 保护类区域的**形变权重恒为 0**（§7：INNER/MARGIN 权重 0，CERVICAL 接近 0）
    for name in ("inner", "margin", "cervical"):
        np.testing.assert_array_equal(region_map.deformation_weight(name), np.zeros(len(mesh.vertices)))
        assert not region_map.domain(name).any()
    # 被保护的顶点必须至少落进一个保护标签（保护口径有覆盖、无遗漏）
    covered = region_map.protection_labels()
    assert covered[np.asarray(region_map.protection) <= 0].all()
    # 可形变域一律不碰保护顶点
    for name in ("occlusal", "mesial", "distal", "buccal", "lingual"):
        assert not region_map.domain(name)[np.asarray(region_map.protection) <= 0].any(), name


def test_mesial_and_distal_are_on_opposite_sides():
    mesh = corrugated_crown()
    margin = ring()
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF)
    mesial = region_map.domain("mesial")
    distal = region_map.domain("distal")
    assert mesial.any() and distal.any()
    assert not (mesial & distal).any()
    assert mesh.vertices[mesial][:, 0].mean() > 0
    assert mesh.vertices[distal][:, 0].mean() < 0
    # 不是"整半颗牙"：轴向壁上只有一段参与邻接
    assert mesial.sum() < len(mesh.vertices) * 0.35


def test_mesial_distal_follow_confirmed_frame_not_world_xyz():
    """把整个场景绕 Z 转 90°：世界 X 变了，但（确认帧下的）近中/远中必须跟着转。"""
    mesh = corrugated_crown()
    margin = ring()
    rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    spun = mesh.copy().apply_transform(rotation)
    spun_margin = trimesh.transform_points(margin, rotation)
    spun_frame = PatientFrame(rotation[:3, 3], rotation[:3, :3] @ FRAME.z,
                              rotation[:3, :3] @ FRAME.x, rotation[:3, :3] @ FRAME.y, revision_id="frame-r1")
    spun_ref = {**REF, "buccal": (rotation[:3, :3] @ np.array([0, 1.0, 0])).tolist()}
    base = build_region_map(design_for(margin), mesh, FRAME, REF)
    spun_map = build_region_map(design_for(spun_margin), spun, spun_frame, spun_ref)
    np.testing.assert_allclose(base.weights["mesial"], spun_map.weights["mesial"], atol=1e-9)
    np.testing.assert_allclose(base.weights["occlusal"], spun_map.weights["occlusal"], atol=1e-9)
    # 区域跟着**材质**一起转：旋转后的近中域 = 旋转前的近中域绕同一轴转 90°
    expected = set(np.flatnonzero(base.domain("mesial")).tolist())
    got = set(np.flatnonzero(spun_map.domain("mesial")).tolist())
    assert got == expected, "近中域没有跟随局部系（而是被世界轴决定了）"


def test_neighbor_geometry_defines_side_when_frame_unconfirmed():
    mesh = corrugated_crown()
    margin = ring()
    neighbor = trimesh.creation.icosphere(subdivisions=2, radius=1.5).apply_translation([0, 4.5, 3])
    # 没有确认帧（revision_id=None）⇒ 用真实邻牙几何定向
    geometric = build_patient_frame(margin, FRAME.z)
    ref = {"revision_id": None}
    region_map = build_region_map(design_for(margin), mesh, geometric, ref,
                                  neighbors={"mesial": FitSurface(neighbor)})
    assert "MESIAL_AXIS_FROM_NEIGHBOR_GEOMETRY_UNVERIFIED_BUCCAL" in region_map.flags
    mesial = region_map.domain("mesial")
    assert mesial.any()
    # 邻牙在 +Y 侧 ⇒ 近中方向应为 +Y（而不是世界 +X）
    assert mesh.vertices[mesial][:, 1].mean() > 0


def test_confirmed_frame_conflicting_with_neighbor_geometry_is_flagged_and_overridden():
    mesh = corrugated_crown()
    margin = ring()
    neighbor = trimesh.creation.icosphere(subdivisions=2, radius=1.5).apply_translation([0, 4.5, 3])
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF,
                                  neighbors={"mesial": FitSurface(neighbor)})
    assert "MESIAL_AXIS_TAKEN_FROM_NEIGHBOR_GEOMETRY_FRAME_CONFLICT" in region_map.flags
    assert region_map.domain("mesial").any()
    assert mesh.vertices[region_map.domain("mesial")][:, 1].mean() > 0


def test_no_frame_and_no_neighbor_is_an_explicit_failure():
    mesh = corrugated_crown()
    with pytest.raises(RegionMapError) as error:
        build_region_map(design_for(ring()), mesh, build_patient_frame(ring(), FRAME.z), {"revision_id": None})
    assert error.value.code == "LOCAL_FRAME_UNCONFIRMED_AND_NO_CONFIRMED_NEIGHBOR_GEOMETRY"


def test_neighbor_distance_limits_proximal_region():
    """邻牙很远时近中域必须收缩（不能把整侧都算成接触区）。"""
    mesh = corrugated_crown()
    margin = ring()
    near = trimesh.creation.icosphere(subdivisions=2, radius=1.5).apply_translation([4.4, 0, 3])
    far = trimesh.creation.icosphere(subdivisions=2, radius=1.5).apply_translation([40.0, 0, 3])
    with_neighbor = build_region_map(design_for(margin), mesh, FRAME, REF, neighbors={"mesial": FitSurface(near)})
    without = build_region_map(design_for(margin), mesh, FRAME, REF, neighbors={"mesial": FitSurface(far)})
    assert with_neighbor.counts()["mesial"] > without.counts()["mesial"]
    assert without.neighbor_distance_mm["mesial"] > 30.0
    assert with_neighbor.neighbor_distance_mm["mesial"] < 3.0


def test_manual_annotation_wins_over_automatic_region():
    mesh = corrugated_crown()
    margin = ring()
    region_map = build_region_map(design_for(margin), mesh, FRAME, REF)
    manual = np.zeros(len(mesh.vertices))
    manual[:10] = 1.0
    weights, source = region_weights({"regions": {"mesial": manual.tolist()}}, "mesial", region_map)
    assert source == "MANUAL_CONFIRMED"
    np.testing.assert_allclose(weights, manual)
    automatic, source = region_weights({}, "mesial", region_map)
    assert source == "AUTO_GEOMETRY"
    np.testing.assert_allclose(automatic, region_map.weights["mesial"])
    with pytest.raises(RegionMapError):
        region_weights({}, "cusp_1", region_map)


def test_region_map_does_not_mutate_mesh_and_is_deterministic():
    mesh = corrugated_crown()
    margin = ring()
    before = mesh.vertices.copy()
    first = build_region_map(design_for(margin), mesh, FRAME, REF)
    second = build_region_map(design_for(margin), mesh, FRAME, REF)
    np.testing.assert_array_equal(mesh.vertices, before)
    for name in REGION_NAMES:
        np.testing.assert_array_equal(first.weights[name], second.weights[name])
    assert first.report()["counts"] == second.report()["counts"]
