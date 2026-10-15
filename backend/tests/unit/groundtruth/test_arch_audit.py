"""提交 2.1 弓段/整颌资产审计的单测。

这里**不碰真实数据**（真实语料在 ``/data1/welo/dentistry``，含患者姓名），
全部用合成的 STL + 合成的 constructionInfo 验证判据本身：
主方向展布能不能分开"细长条"与"马蹄形"、连通分量算得对不对、
元数据优先有没有生效、分类是否带依据、索引是否仍然纯 ASCII。
"""

from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

import pytest

from app.groundtruth import arch_audit as A


# ---------------------------------------------------------------- 合成 STL

def write_stl(path: Path, triangles) -> Path:
    """写一个合法的二进制 STL：**每个面 50 字节**（12 float + 2 字节属性）。

    少写这 2 字节会让读取端整体错位，量出来的是垃圾 —— 这也是本文件早期
    自查时踩到的坑，所以写在这里并加断言。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            f.write(struct.pack("<12fH", 0.0, 0.0, 1.0, *a, *b, *c, 0))
    assert path.stat().st_size == 84 + 50 * len(triangles)
    return path


def box(x, y, z, ox=0.0, oy=0.0, oz=0.0):
    """封闭长方体：8 顶点 / 12 面（体积=面积/边都可解析核对）。"""
    v = [(ox, oy, oz), (ox + x, oy, oz), (ox + x, oy + y, oz), (ox, oy + y, oz),
         (ox, oy, oz + z), (ox + x, oy, oz + z), (ox + x, oy + y, oz + z), (ox, oy + y, oz + z)]
    q = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
         (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    return [(v[i], v[j], v[k]) for i, j, k in q]


def dense_box(x, y, z, steps=6, ox=0.0, oy=0.0, oz=0.0):
    """细分的长方体：顶点间距 ~x/steps，接近真实网格的密度。

    网格查询给的是"到最近**顶点**"的距离，稀疏网格（只有 8 个角点）会把距离
    夸大 —— 测试里必须用稠密网格，否则量到的是角点到中心的距离。
    """
    tris = []
    for axis in range(3):
        for side in (0.0, 1.0):
            for i in range(steps):
                for j in range(steps):
                    u0, u1 = i / steps, (i + 1) / steps
                    v0, v1 = j / steps, (j + 1) / steps
                    quad = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
                    pts = []
                    for u, v in quad:
                        c = [0.0, 0.0, 0.0]
                        c[axis] = side
                        others = [k for k in range(3) if k != axis]
                        c[others[0]] = u
                        c[others[1]] = v
                        scale = (x, y, z)
                        pts.append((ox + c[0] * scale[0], oy + c[1] * scale[1],
                                    oz + c[2] * scale[2]))
                    tris.append((pts[0], pts[1], pts[2]))
                    tris.append((pts[0], pts[2], pts[3]))
    return tris


def horseshoe(radius=30.0, blocks=24, size=6.0):
    """马蹄形：沿半圆排布的小方块 —— 两个平面方向都有展布。"""
    tris = []
    for i in range(blocks):
        theta = math.pi * i / (blocks - 1)
        tris += box(size, size, size,
                    ox=radius * math.cos(theta) - size / 2,
                    oy=radius * math.sin(theta) - size / 2)
    return tris


# ---------------------------------------------------------------- 几何特征

def test_principal_extents_separate_bar_from_horseshoe(tmp_path):
    bar = A.mesh_features(str(write_stl(tmp_path / "bar.stl", box(60, 6, 6))))
    arch = A.mesh_features(str(write_stl(tmp_path / "arch.stl", horseshoe())))
    # 细长条：第二主方向远小于第一主方向
    assert bar.second_plane_ratio < 0.20, bar.second_plane_ratio
    # 马蹄形：两个平面方向都有明显展布（这正是长桥与整颌的分界）
    assert arch.second_plane_ratio > 0.45, arch.second_plane_ratio
    assert arch.third_plane_ratio < 0.30, arch.third_plane_ratio
    assert bar.max_extent == pytest.approx(60.0, abs=0.01)


def test_bbox_and_area_are_exact_on_a_box(tmp_path):
    feats = A.mesh_features(str(write_stl(tmp_path / "b.stl", box(10, 20, 30))))
    assert feats.extents == pytest.approx((10.0, 20.0, 30.0), abs=0.01)
    assert feats.bbox_volume == pytest.approx(6000.0, rel=1e-6)
    # 长方体表面积 = 2(ab+bc+ca) = 2(200+600+300) = 2200
    assert feats.surface_area == pytest.approx(2200.0, rel=1e-6)


def test_components_split_disjoint_shells(tmp_path):
    two = A.mesh_features(str(write_stl(tmp_path / "two.stl",
                                       box(10, 10, 10) + box(10, 10, 10, ox=40))))
    assert two.components == 2
    assert two.largest_component_ratio == pytest.approx(0.5, abs=0.02)
    one = A.mesh_features(str(write_stl(tmp_path / "one.stl", box(10, 10, 10))))
    assert one.components == 1
    assert one.largest_component_ratio == pytest.approx(1.0, abs=0.02)


def test_topology_is_skipped_not_faked_when_too_big(tmp_path):
    """面数超预算时 components 必须是"没算"，不能填 0 或 1 冒充。"""
    path = write_stl(tmp_path / "big.stl", box(10, 10, 10))
    feats, _grid = A.analyze_asset(str(path), topology_budget=-1)
    assert feats.topology_skipped is True
    assert feats.components is None


def test_analyze_asset_single_pass_returns_grid(tmp_path):
    feats, grid = A.analyze_asset(str(write_stl(tmp_path / "b.stl",
                                                dense_box(10, 10, 10))))
    assert feats.ok and grid is not None
    assert grid.n_points > 0
    # 稠密网格上，盒心 (5,5,5) 到最近顶点的距离就是到表面的距离 = 5 mm
    d, found = grid.nearest_distance((5.0, 5.0, 5.0))
    assert found and d == pytest.approx(5.0, abs=0.2)


# ---------------------------------------------------------------- 中心邻域

def test_center_occupancy_tells_present_from_void(tmp_path):
    """预备体还在（中心紧贴预备体表面）与代型被抽走（中心周围是空的）可分。"""
    prep = write_stl(tmp_path / "prep.stl", dense_box(2, 2, 2, steps=4, ox=-1, oy=-1, oz=-1))
    _feats, grid = A.analyze_asset(str(prep))
    present = A.center_features(grid, (0.0, 0.0, 0.0))
    assert present.dist_min is not None and present.dist_min <= 3.5
    assert present.count_r4 > 0
    assert present.occupancy_class == "surface_present"

    far = write_stl(tmp_path / "far.stl", dense_box(10, 10, 10, ox=40, oy=40, oz=40))
    _feats2, grid2 = A.analyze_asset(str(far))
    void = A.center_features(grid2, (0.0, 0.0, 0.0))
    assert void.saturated is True and void.count_r4 == 0
    assert void.occupancy_class == "void"

    # 3.5~6 mm 之间必须说"不知道"，不能硬判
    mid = write_stl(tmp_path / "mid.stl", dense_box(10, 10, 10))
    _feats3, grid3 = A.analyze_asset(str(mid))
    ambiguous = A.center_features(grid3, (5.0, 5.0, 5.0))
    assert ambiguous.occupancy_class == "ambiguous", ambiguous.dist_min


def test_center_features_without_grid_is_unknown():
    feats = A.center_features(None, (0.0, 0.0, 0.0))
    assert feats.occupancy_class == "unknown"
    assert feats.dist_min is None


# ---------------------------------------------------------------- 元数据

class _FakeScan:
    def __init__(self, file_name, part_type):
        self.file_name, self.part_type = file_name, part_type


class _FakeTooth:
    def __init__(self, fdi, center=None):
        self.tooth_fdi, self.center = fdi, center


class _FakeCase:
    def __init__(self, scans=(), teeth=()):
        self.scan_files, self.teeth = list(scans), list(teeth)


def test_metadata_labels_and_role_matching(tmp_path):
    case = _FakeCase(scans=[
        _FakeScan("邱治钧-upperjaw.DCM", "Antagonist"),
        _FakeScan("邱治钧-lowerjaw.DCM", "UnsegmentedJaw"),
        _FakeScan("1.stl", "SituScan"),
    ], teeth=[_FakeTooth(36, (0.0, 0.0, 0.0))])
    meta = A.case_metadata(case)
    assert meta.has_unsegmented_jaw_label and meta.has_antagonist_label
    assert meta.n_targets == 1
    # 元数据里是 .DCM，同一病例的转换版 STL 也能按 stem 认出来
    assert A.role_for_asset(meta, str(tmp_path / "邱治钧-upperjaw.stl")) == "Antagonist"
    assert A.role_for_asset(meta, str(tmp_path / "1.stl")) == "SituScan"
    assert A.role_for_asset(meta, str(tmp_path / "unknown.stl")) is None
    # 上/下颌提示是从原名抽的 ASCII token，不写中文名
    assert meta.labels("Antagonist")[0]["side"] == "upper"
    assert meta.labels("UnsegmentedJaw")[0]["side"] == "lower"


def test_metadata_antagonist_is_not_target_present(tmp_path):
    feats = A.mesh_features(str(write_stl(tmp_path / "jaw.stl", horseshoe())))
    cls = A.classify_asset(feats, role="Antagonist", center=None)
    assert (cls.asset_class, cls.basis, cls.confidence) == \
        ("FULL_ARCH_ANTAGONIST", "METADATA_ANTAGONIST", "high")


def test_metadata_unsegmented_splits_on_target_presence(tmp_path):
    feats = A.mesh_features(str(write_stl(tmp_path / "jaw.stl", horseshoe())))
    present = A.CenterFeatures(dist_min=1.2, count_r4=40, count_r6=120)
    missing = A.CenterFeatures(dist_min=9.0, count_r4=0, saturated=True)
    with_target = A.classify_asset(feats, role="UnsegmentedJaw", center=present)
    without = A.classify_asset(feats, role="UnsegmentedJaw", center=missing)
    assert (with_target.asset_class, with_target.basis) == \
        ("FULL_ARCH_WITH_TARGET", "METADATA_UNSEGMENTED_JAW")
    assert (without.asset_class, without.basis) == \
        ("FULL_ARCH_TARGET_MISSING", "METADATA_UNSEGMENTED_JAW")


# ---------------------------------------------------------------- 分类

def test_geometry_classification_basis_and_confidence(tmp_path):
    tooth = A.mesh_features(str(write_stl(tmp_path / "t.stl", box(9, 8, 11))))
    bar = A.mesh_features(str(write_stl(tmp_path / "bar.stl", box(60, 6, 6))))
    arch = A.mesh_features(str(write_stl(tmp_path / "arch.stl", horseshoe())))
    small = A.classify_asset(tooth, role=None, center=None)
    assert (small.asset_class, small.basis) == ("SINGLE_TOOTH", "GEOMETRY_SINGLE_TOOTH")
    assert small.confidence == "high"
    elongated = A.classify_asset(bar, role=None, center=None)
    assert (elongated.asset_class, elongated.basis) == \
        ("PARTIAL_ARCH", "GEOMETRY_PARTIAL_ARCH")
    # 弓形尺度但缺 Center：绝不能算成"含目标的整颌"（那是承重假设，必须保守）
    no_center = A.classify_asset(arch, role=None, center=None)
    assert no_center.asset_class == "UNKNOWN"
    assert no_center.basis == "GEOMETRY_FULL_ARCH"
    # 有了 Center 且中心有面 → 才算正样本
    positive = A.classify_asset(arch, role=None, center=A.CenterFeatures(
        dist_min=1.0, count_r4=10))
    assert positive.asset_class == "FULL_ARCH_WITH_TARGET"


def test_classification_evidence_is_ascii_and_numeric(tmp_path):
    feats = A.mesh_features(str(write_stl(tmp_path / "t.stl", box(9, 8, 11))))
    cls = A.classify_asset(feats, role=None, center=None)
    cls.evidence.encode("ascii")                      # 不含中文
    assert "max_extent=" in cls.evidence and "second_ratio=" in cls.evidence


# ---------------------------------------------------------------- 端到端与输出

MINIMAL_CI = """<?xml version="1.0" encoding="utf-8"?>
<ConstructionInfo>
  <ScanFiles>
    <ScanFile><FileName>{case}-lowerjaw.DCM</FileName><PartType>UnsegmentedJaw</PartType></ScanFile>
    <ScanFile><FileName>{case}-upperjaw.DCM</FileName><PartType>Antagonist</PartType></ScanFile>
  </ScanFiles>
  <Tooth>
    <Number>36</Number>
    <Center><x>{cx}</x><y>{cy}</y><z>{cz}</z></Center>
    <ReconstructionType>AnatomicCrown</ReconstructionType>
  </Tooth>
</ConstructionInfo>
"""


def _case_dir(root: Path, case: str, *, center):
    case_dir = root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / f"{case}.constructionInfo").write_text(
        MINIMAL_CI.format(case=case, cx=center[0], cy=center[1], cz=center[2]),
        encoding="utf-8")
    write_stl(case_dir / f"{case}-jaw.stl", box(70, 50, 8, ox=-35, oy=0, oz=-4))
    return case_dir


def test_audit_case_end_to_end_with_metadata_and_center(tmp_path):
    """整条链路：真 CI 解析 → 元数据角色 → 几何 → 中心邻域 → 一行审计。"""
    from app.groundtruth.indexer import discover_cases

    _case_dir(tmp_path, "82403180486", center=(0.0, 5.0, 0.0))
    cases = {k: v for k, v in discover_cases([str(tmp_path)]).items()
             if v.has_construction_info}
    rows = A.audit_cases(cases, do_hash=False, topology=True)
    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_class"] in A.ASSET_CLASSES
    assert row["classification_basis"] in A.BASES
    assert row["case_has_metadata_unsegmented_jaw"] == 1
    assert row["case_has_metadata_antagonist"] == 1
    # 元数据里有对颌（是 .DCM，读不了几何）→ 至少给出引用，而不是拿几何瞎猜
    assert row["antagonist_basis"] == "METADATA_ANTAGONIST_REFERENCE"
    assert row["antagonist_metadata_side"] == "upper"
    assert row["case_unsegmented_jaw_side"] == "lower"
    assert row["target_fdi"] == 36
    assert row["has_center"] == 1


def test_write_outputs_is_phi_free_and_complete(tmp_path):
    from app.groundtruth.indexer import discover_cases

    _case_dir(tmp_path, "82403180486", center=(0.0, 5.0, 0.0))
    cases = {k: v for k, v in discover_cases([str(tmp_path)]).items()
             if v.has_construction_info}
    rows = A.audit_cases(cases, do_hash=False)
    out = tmp_path / "out"
    summary = A.write_outputs(rows, out)

    text = (out / "arch_asset_audit.csv").read_text(encoding="ascii")   # 非 ASCII 会抛错
    reader = list(csv.DictReader(text.splitlines()))
    assert list(reader[0].keys()) == list(A.AUDIT_COLUMNS)
    assert summary["n_assets"] == len(rows)
    assert summary["class_counts"]
    assert summary["threshold_sweep"] and summary["provisional_thresholds"]
    loaded = json.loads((out / "arch_asset_summary.json").read_text(encoding="ascii"))
    assert loaded["note"].startswith("provisional thresholds only")


def test_distribution_fields_are_present_for_2_2(tmp_path):
    """2.2 要靠这些分布定阈值，所以它们必须在汇总里。"""
    from app.groundtruth.indexer import discover_cases

    _case_dir(tmp_path, "82403180486", center=(0.0, 5.0, 0.0))
    cases = {k: v for k, v in discover_cases([str(tmp_path)]).items()
             if v.has_construction_info}
    summary = A.write_outputs(A.audit_cases(cases, do_hash=False), tmp_path / "out")
    for key in ("second_plane_ratio", "third_plane_ratio", "max_extent", "surface_area"):
        assert key in summary["all_assets_feature_quantiles"]
        assert summary["all_assets_feature_quantiles"][key]["n"] >= 1
