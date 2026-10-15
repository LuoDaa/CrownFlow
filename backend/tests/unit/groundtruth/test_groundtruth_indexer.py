"""GroundTruthIndexer 的单元测试 —— **合成夹具，可提交 Git**。

真实数据在 ``/data1/welo/dentistry``（含患者姓名文件名、内网路径），
只做本地 acceptance（``tests/acceptance/exocad_real/``），不进 Git。

这里覆盖既定完成标准的每一条，并锁住那个 ``bridge_slm_cad`` 映射 bug：

    旧实现只取文件名里**最后一个**两位数字，于是
    ``...-12-11-21-22-bridge_slm_cad.stl`` 被当成"只有 FDI 22" ——
    一座 4 连桥会被误判成单冠，直接污染 ``final_crown_fdi_count`` 与
    ``single_unit``，进而污染 SingleAnatomicCrownGTSet。
"""

from __future__ import annotations

import math
import os
import re
import struct
from pathlib import Path

import pytest

from tests.unit.importers.test_construction_design import build_construction_info

from app.groundtruth.assets import (
    StlProbe,
    assert_phi_free,
    fdis_in_name,
    probe_stl,
    safe_asset_path,
    safe_case_id,
    watertight_and_euler,
)
from app.groundtruth.indexer import (
    CaseFiles,
    build_index,
    classify_arch_sectioned,
    discover_cases,
    write_outputs,
)
from app.groundtruth.models import (
    GT_SETS,
    ToothGroundTruth,
    build_all_sets,
    case_group_key,
    is_margin_valid,
)


#: 仓库后端根目录（给子进程测试做 cwd）。
ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- STL 夹具

def _tris_from_verts(verts, quads):
    out = []
    for a, b, c, d in quads:
        out.append((verts[a], verts[b], verts[c]))
        out.append((verts[a], verts[c], verts[d]))
    return out


def _write_tris(path: Path, tris) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(bytes(80))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", 0.0, 0.0, 1.0))
            for pt in t:
                f.write(struct.pack("<3f", *pt))
            f.write(struct.pack("<H", 0))
    return path


def _read_tris(path: Path):
    raw = path.read_bytes()
    n = struct.unpack("<I", raw[80:84])[0]
    return [((v[3], v[4], v[5]), (v[6], v[7], v[8]), (v[9], v[10], v[11]))
            for v in (struct.unpack_from("<12f", raw, 84 + i * 50)
                      for i in range(n))]


def write_box_stl(path: Path, size: float, center=(0.0, 0.0, 0.0),
                  *, subdiv: int = 1) -> Path:
    """写轴对齐立方体的二进制 STL（纯 struct，不需要 numpy/trimesh）。

    ``subdiv`` 把每个面细分成 ``subdiv x subdiv`` 个四边形。**必须能细分**才能让
    "到表面距离"有意义：只有 8 个顶点的粗盒子只能量到"到角点的距离"，
    而真实扫描网格是密的（三角形约 0.3 mm）。实测粗盒子把"中心到表面 2.5 mm"
    量成 4.33 mm，足以把 unsectioned 误判成 unknown。
    """
    h = size / 2.0
    cx, cy, cz = center
    verts = []
    quads = []
    for (i, j), axis, sg in ((( 0, 1), 2, -1.0), ((0, 1), 2, 1.0),
                             ((0, 2), 1, -1.0), ((0, 2), 1, 1.0),
                             ((1, 2), 0, -1.0), ((1, 2), 0, 1.0)):
        base = len(verts)
        for a in range(subdiv + 1):
            for b in range(subdiv + 1):
                pt = [0.0, 0.0, 0.0]
                pt[axis] = sg * h
                pt[i] = (a / subdiv - 0.5) * 2 * h
                pt[j] = (b / subdiv - 0.5) * 2 * h
                verts.append((cx + pt[0], cy + pt[1], cz + pt[2]))
        for a in range(subdiv):
            for b in range(subdiv):
                i0 = base + a * (subdiv + 1) + b
                quads.append((i0, i0 + 1, i0 + subdiv + 2, i0 + subdiv + 1))
    return _write_tris(path, _tris_from_verts(verts, quads))


#: 夹具 constructionInfo 里的 <Center>。小方块（"预备体"）必须落在这里，
#: 否则测的就不是"目标中心处有没有表面"。
TARGET_CENTER = (13.9, 10.8, -12.35)


def write_hollow_arch(path: Path, center=TARGET_CENTER) -> Path:
    """模拟 **unsectioned** 整颌：大箱子 + **目标中心处**的密网格小实体。

    小实体必须是**密网格**：只有 8 个顶点的粗盒子会把"中心到表面 2.5 mm"
    量成"到角点 4.33 mm"，足以把 unsectioned 误判成 unknown。
    真实扫描网格是密的（三角形约 0.3 mm），这里用 subdiv=8 模拟。
    "预备体"尺寸取 5 mm（半径 2.5 mm），与真实预备体量级一致 ——
    做成 10 mm 会让中心到表面 5 mm，超出判据 3.5 mm 而被误判成 unknown。
    """
    part = path.with_name(path.name + ".part")
    write_box_stl(part, 5.0, center, subdiv=8)   # 半径 2.5 mm，模拟真实预备体尺度
    tris = _read_tris(part)
    part.unlink()

    h1 = 30.0
    v_big = [(sx * h1, sy * h1, sz * h1)
             for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris.extend(_tris_from_verts(v_big, quads))
    return _write_tris(path, tris)


def write_arch_with_void(path: Path) -> Path:
    """模拟 **sectioned** modelbase：只有大箱子，目标中心处是空的。"""
    return write_box_stl(path, 60.0)


# ---------------------------------------------------------------- 夹具树

def _dense_margin_xml(n: int = 120, radius: float = 5.0) -> str:
    """生成一条真实量级的闭合颈缘环（供 groundtruth 夹具使用）。"""
    parts = []
    for i in range(n):
        a = 2.0 * math.pi * i / n
        parts.append(f"<Vec3><x>{13.9 + radius * math.cos(a):.4f}</x>"
                     f"<y>{10.8 + radius * math.sin(a):.4f}</y>"
                     f"<z>{-12.35:.4f}</z></Vec3>")
    return "".join(parts)


def build_case_tree(root: Path, *, case_dir_name: str,
                    crown_name: str = "CASE-17-crown_cad.stl",
                    arch: str = "sectioned",
                    margin: bool = True,
                    axis: bool = True) -> Path:
    """构造一个最小病例目录（两个 root 也由调用方自己造）。"""
    d = root / case_dir_name
    d.mkdir(parents=True, exist_ok=True)
    xml = build_construction_info(with_geometry=margin and axis)
    # 共享夹具的颈缘线只有 5 个点，过不了 is_margin_valid（要求 >=20 点）。
    # 这里换成真实量级的闭合环：120 点、半径 5 mm -> 周长约 31.4 mm、段长约 0.26 mm。
    xml = re.sub(r"<Margin>.*?</Margin>", "<Margin>" + _dense_margin_xml() + "</Margin>",
                 xml, flags=re.S)
    (d / "CASE.constructionInfo").write_text(xml, encoding="utf-8")
    if crown_name:
        write_box_stl(d / crown_name, 9.0, center=(13.9, 10.8, -12.35))
    write_box_stl(d / "CASE-17-modeldie.stl", 8.0, center=(13.9, 10.8, -12.35))
    if arch == "sectioned":
        write_arch_with_void(d / "CASE-17-modelbase.stl")
    elif arch == "unsectioned":
        write_hollow_arch(d / "CASE-17-modelbase.stl")
    return d


# ============================================================ ⑥ bug 修复

def test_bridge_filename_yields_all_fdis_not_just_the_last():
    """**本提交要修的那个 bug。** 桥件必须报出全部 FDI。"""
    name = "82403170708-150318-2M1-A组-12-11-21-22-bridge_slm_cad.stl"
    assert fdis_in_name(name) == [12, 11, 21, 22]
    # 旧实现的效果：只取最后一个 -> [22]，于是 4 连桥被当成单冠
    assert fdis_in_name(name)[-1] == 22
    assert len(fdis_in_name(name)) == 4


def test_single_crown_filename_yields_one_fdi():
    assert fdis_in_name("82404080845-17-crown_cad.stl") == [17]
    assert len(fdis_in_name("82404080845-17-crown_cad.stl")) == 1


def test_fdi_extraction_does_not_eat_long_numbers():
    """不能把订单号/日期里的两位数字误当 FDI。"""
    got = fdis_in_name("82404080845-150318-2M1-A组-12-11-bridge_slm_cad.stl")
    assert got == [12, 11]
    assert 82 not in got and 40 not in got


# ============================================================ ⑦⑧ PHI

def test_safe_case_id_keeps_ascii_ids_and_hashes_names():
    assert safe_case_id("82403180486") == "82403180486"
    assert safe_case_id("047-150491") == "047-150491"
    # 含中文（患者姓名）-> 换成哈希，绝不原样写出
    got = safe_case_id("82403180486-160323李挨林B组")
    assert got.startswith("case_")
    assert "李" not in got
    got.encode("ascii")           # 必须可 ASCII 编码


def test_safe_asset_path_never_contains_original_filename():
    p = safe_asset_path("82403180486", "final_crown", 0, ".stl")
    assert p == "82403180486/final_crown__00.stl"
    assert "李挨林" not in p and "B组" not in p


def test_assert_phi_free_rejects_non_ascii():
    assert_phi_free([{"a": "82403180486", "b": "ok"}])
    with pytest.raises(ValueError, match="非 ASCII"):
        assert_phi_free([{"a": "82403180486", "b": "李挨林"}])


def test_index_output_is_pure_ascii_even_with_named_inputs(tmp_path):
    """输入端文件名含患者姓名时，输出索引里不得出现它。"""
    root = tmp_path / "EXO"
    name = "82403180486-160323李挨林 B组"
    build_case_tree(root, case_dir_name=name)
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    summary = write_outputs(rows, tmp_path / "out")
    text = (tmp_path / "out" / "tooth_index.csv").read_text(encoding="ascii")
    assert "李" not in text and "B组" not in text
    assert summary["n_rows"] >= 1


# ============================================================ ② 一行一病例×FDI

def test_one_row_per_case_fdi(tmp_path):
    root = tmp_path / "EXO"
    build_case_tree(root, case_dir_name="82404080845")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    # 夹具里有两个牙位：17（修复）与 47（对颌）
    assert sorted(r.fdi for r in rows) == [17, 47]
    r17 = next(r for r in rows if r.fdi == 17)
    assert r17.reconstruction_type == "AnatomicCrown"
    assert r17.has_margin and r17.margin_point_count == 120
    assert r17.margin_valid is True
    assert r17.has_axis and r17.axis_valid
    assert r17.has_center and r17.has_tooth_frame and r17.has_tooth_library
    assert r17.tooth_model_matrix_valid
    assert r17.parameter_count == 2


# ============================================================ ③ 跨 root 归并

def test_same_case_split_across_two_roots_is_one_case(tmp_path):
    """实测 EXO/ 与 all_data/ 是同一病例的两半，必须合并而不是算成两个病例。"""
    exo = tmp_path / "EXO" / "82404080845"
    exo.mkdir(parents=True)
    write_arch_with_void(exo / "82404080845-160545-A组-47-modelbase (1).stl")

    all_data = tmp_path / "all_data" / "82404080845"
    all_data.mkdir(parents=True)
    (all_data / "82404080845.constructionInfo").write_text(
        build_construction_info(), encoding="utf-8")
    write_box_stl(all_data / "82404080845-17-crown_cad.stl", 9.0,
                  center=(13.9, 10.8, -12.35))

    cases = discover_cases([str(tmp_path / "EXO"), str(tmp_path / "all_data")])
    assert len(cases) == 1, f"应合并成 1 个病例，实际 {list(cases)}"
    files = cases["82404080845"]
    assert files.has_construction_info            # 来自 all_data
    assert any("modelbase" in p for p in files.stls)   # 来自 EXO
    rows = build_index(cases, crown_topology=False, do_hash=False)
    assert len({r.case_uuid for r in rows}) == 1


def test_case_group_key_groups_all_teeth_of_one_case(tmp_path):
    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    assert len({case_group_key(r) for r in rows}) == 1


# ============================================================ ④ 桥不误判单冠

def test_bridge_is_not_single_unit(tmp_path):
    root = tmp_path / "all_data"
    d = root / "82403170708"
    d.mkdir(parents=True)
    # 该病例的 constructionInfo 里导出的是 4 连桥
    (d / "82403170708.constructionInfo").write_text(
        build_construction_info().replace("CASE-17-crown_cad.stl",
                                          "CASE-12-11-21-22-bridge_slm_cad.stl")
        .replace("<ToothNumbers><int>17</int></ToothNumbers>",
                 "<ToothNumbers><int>12</int><int>11</int>"
                 "<int>21</int><int>22</int></ToothNumbers>"),
        encoding="utf-8")
    write_box_stl(d / "CASE-12-11-21-22-bridge_slm_cad.stl", 24.0)
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    # 夹具 constructionInfo 里的牙位是 17；桥件覆盖 12/11/21/22，
    # 所以 FDI 17 拿不到冠 —— 关键是这里**绝不能**把桥当成 17 的单冠
    r17 = next(r for r in rows if r.fdi == 17)
    assert r17.final_crown_single_unit is False
    if r17.has_final_crown:
        assert r17.final_crown_fdi_count > 1


def test_bridge_matched_by_filename_reports_multiple_fdi(tmp_path):
    """退到文件名匹配时，也必须报出桥的全部 FDI（而不是最后一个）。"""
    root = tmp_path / "all_data"
    d = root / "82403170708"
    d.mkdir(parents=True)
    (d / "82403170708.constructionInfo").write_text(
        build_construction_info(with_exports=False), encoding="utf-8")
    write_box_stl(d / "CASE-17-16-15-bridge_slm_cad.stl", 24.0)
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    r17 = next(r for r in rows if r.fdi == 17)
    assert r17.has_final_crown
    assert r17.final_crown_fdi_count == 3
    assert r17.final_crown_single_unit is False


# ============================================================ ⑤ sectioned/unsectioned

def test_classify_arch_sectioned_truth_table():
    assert classify_arch_sectioned(1.0, 1.0) == "unsectioned"
    assert classify_arch_sectioned(30.0, 0.0) == "sectioned"
    assert classify_arch_sectioned(2.0, 0.1) == "sectioned"     # 近但四周空 -> 洞边
    assert classify_arch_sectioned(30.0, 0.9) == "sectioned"    # 中心很远 -> 空洞
    assert classify_arch_sectioned(4.5, 0.4) == "unknown"       # 两信号不一致
    assert classify_arch_sectioned(None, None) == "unknown"


def test_unsectioned_arch_is_a_localization_positive(tmp_path, monkeypatch):
    from app.groundtruth import human_labels

    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845", arch="unsectioned")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    r17 = next(r for r in rows if r.fdi == 17)
    assert r17.has_full_arch_candidate
    assert r17.arch_is_sectioned_or_unsectioned == "unsectioned"
    assert r17.target_center_has_surface is True
    # arch_asset_class 由 freeze.py 从审计 CSV 回填（弓判定只有审计那一处）。
    # 这里手工置上，验证的是**集合规则本身**。
    r17.arch_asset_class = "FULL_ARCH_WITH_TARGET"
    # **但自动预筛已经不够了**：v2 的"附近有表面"与 v3.2 的"格搜索有峰"
    # 都被两轮人工 QA 证伪（precision 只有 ~0.13~0.22），所以准入还必须有
    # 一个人工 A 判定。这条断言就是那个政策变更本身。
    assert r17 not in build_all_sets(rows)["LocalizationGTSet"]
    monkeypatch.setitem(human_labels.QA2_LABELS, (r17.case_uuid, r17.fdi), "A")
    assert r17 in build_all_sets(rows)["LocalizationGTSet"]


def test_sectioned_arch_is_excluded_from_localization_set(tmp_path):
    """**承重假设的落点**：代型被抽走的 modelbase 不能当正样本。"""
    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845", arch="sectioned")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    r17 = next(r for r in rows if r.fdi == 17)
    assert r17.has_full_arch_candidate
    assert r17.arch_is_sectioned_or_unsectioned == "sectioned"
    assert r17.target_center_has_surface is False
    assert r17 not in build_all_sets(rows)["LocalizationGTSet"]


# ============================================================ ⑥ 数量由脚本产生

def test_set_counts_come_from_rules_not_constants(tmp_path):
    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845", arch="unsectioned")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    summary = write_outputs(rows, tmp_path / "out")
    sets = build_all_sets(rows)
    for name, members in sets.items():
        assert summary["set_counts"][name] == len(members), name
    # 所有集合名都必须出现在报告里（没有"漏报"的集合）
    assert set(summary["set_counts"]) == set(GT_SETS)


def test_single_anatomic_crown_requires_every_clause(tmp_path):
    """逐条拆掉一个条件，集合成员就必须掉出去 —— 证明是规则而非数量。"""
    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845", arch="unsectioned")
    rows = build_index(discover_cases([str(root)]), crown_topology=False, do_hash=False)
    r17 = next(r for r in rows if r.fdi == 17)
    keep = dict(reconstruction_type="AnatomicCrown",
                final_crown_fdi_count=1, final_crown_single_unit=True,
                final_crown_watertight=True,
                has_axis=True, axis_valid=True, has_final_crown=True,
                margin_crown_consistent=True, has_modeldie=True,
                modeldie_same_frame=True,
                has_margin=True, margin_point_count=434,
                margin_perimeter_mm=32.35, margin_max_segment_mm=0.14)
    for k, v in keep.items():
        setattr(r17, k, v)
    assert r17 in build_all_sets(rows)["SingleAnatomicCrownGTSet"]

    for clause, bad in (("reconstruction_type", "Coping"),
                        ("final_crown_fdi_count", 2),
                        ("final_crown_single_unit", False),
                        ("axis_valid", False),
                        ("has_final_crown", False),
                        ("margin_crown_consistent", False),
                        ("final_crown_watertight", False),
                        ("modeldie_same_frame", False)):
        original = getattr(r17, clause)
        setattr(r17, clause, bad)
        assert r17 not in build_all_sets(rows)["SingleAnatomicCrownGTSet"], \
            f"去掉 {clause}={bad!r} 后仍被判为单冠真值"
        setattr(r17, clause, original)


def test_restorative_excludes_antagonist_and_pontic():
    from app.groundtruth.models import RESTORATIVE_TYPES
    assert "Antagonist" not in RESTORATIVE_TYPES
    assert "HealthyTooth" not in RESTORATIVE_TYPES
    assert "AnatomicCrown" in RESTORATIVE_TYPES


# ============================================================ ⑨ 无重依赖

def test_indexer_imports_in_an_environment_without_numpy():
    """**这条是服务器冒烟测试换来的。** 索引器必须能在只有 stdlib 的机器上导入。

    只扫源码是不够的：第一版只检查了 ``app/groundtruth/*`` 的 import 语句，
    测试通过，但一上服务器就挂 —— 因为 ``app.importers.exocad.__init__`` 会
    eager import ``case_loader`` -> ``matrix4`` -> numpy。
    所以这里**真的**在阻断 numpy/trimesh/scipy 的子进程里导入一次。
    """
    import subprocess
    import sys
    code = chr(10).join([
        'import sys',
        "for m in ('numpy', 'trimesh', 'scipy'):",
        '    sys.modules[m] = None',
        # 冻结链上的每一个模块都要能在纯 stdlib 机器上导入：
        # indexer -> models -> human_labels（人工判定表）；freeze 也是纯 stdlib。
        'import app.groundtruth.indexer',
        'import app.groundtruth.human_labels',
        'import app.groundtruth.freeze',
        'import app.groundtruth.qa2_render',
        'from app.importers.exocad import parse_construction_case',
        "print('ok')",
    ])
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "阻断 numpy/trimesh/scipy 后导入失败：" + r.stderr[-1500:])


def test_indexer_sources_are_stdlib_only_by_scan():
    """廉价的二次检查：``app/groundtruth`` 自身的 import 只允许 stdlib。

    传递依赖由上面那条子进程测试覆盖。
    """
    import re
    here = Path(__file__).resolve().parents[3] / "app" / "groundtruth"
    pattern = r"^[ \t]*(?:import|from)[ \t]+([A-Za-z_][\w.]*)"
    import_re = re.compile(pattern, re.M)
    allowed = {"__future__", "argparse", "csv", "dataclasses", "hashlib",
               "json", "math", "os", "pathlib", "re", "statistics", "struct",
               "sys", "typing", "enum", "importlib", "urllib", "zlib", "glob",
               # 本包内部互相引用（human_labels 是纯 stdlib 政策表），
               # 传递依赖由上面那条子进程测试覆盖。
               "app"}
    for name in ("assets.py", "indexer.py", "models.py", "arch_audit.py",
                 "human_labels.py", "freeze.py", "qa_render.py",
                 "qa2_render.py", "__init__.py"):
        src = (here / name).read_text(encoding="utf-8")
        mods = {m.group(1).split(".")[0] for m in import_re.finditer(src)}
        assert mods <= allowed, f"{name} 引入了非 stdlib 模块: {mods - allowed}"


# ============================================================ STL 层

def test_probe_stl_reads_bbox_and_scale_class(tmp_path):
    p = write_box_stl(tmp_path / "crown.stl", 9.0, center=(10.0, 0.0, 0.0))
    pr = probe_stl(str(p))
    assert pr.ok and pr.faces == 12
    assert pr.scale_class == "single_tooth"
    assert pr.center == pytest.approx((10.0, 0.0, 0.0))
    assert pr.max_extent_mm == pytest.approx(9.0, abs=1e-4)

    big = write_box_stl(tmp_path / "arch.stl", 60.0)
    assert probe_stl(str(big)).scale_class == "arch"


def test_probe_stl_flags_truncated_file(tmp_path):
    p = write_box_stl(tmp_path / "x.stl", 5.0)
    data = p.read_bytes()
    p.write_bytes(data[:84 + 50 * 5])         # 砍掉一半面
    pr = probe_stl(str(p))
    assert pr.ok and pr.size_mismatch is True
    assert pr.faces == 5


def test_watertight_and_euler_on_closed_box(tmp_path):
    p = write_box_stl(tmp_path / "box.stl", 4.0)
    wt, euler = watertight_and_euler(str(p))
    assert wt is True
    assert euler == 2


def test_margin_validity_needs_closed_uniform_ring():
    from app.groundtruth.models import ToothGroundTruth
    good = ToothGroundTruth(case_uuid="cf_x", source_case_id="1",
                            source_order_id="1", fdi=17,
                            reconstruction_type="AnatomicCrown",
                            has_margin=True, margin_point_count=434,
                            margin_perimeter_mm=32.35, margin_max_segment_mm=0.14)
    assert is_margin_valid(good)
    thin = ToothGroundTruth(case_uuid="cf_x", source_case_id="1",
                            source_order_id="1", fdi=17,
                            reconstruction_type="AnatomicCrown",
                            has_margin=True, margin_point_count=5,
                            margin_perimeter_mm=30.0, margin_max_segment_mm=0.1)
    assert not is_margin_valid(thin)          # 点数太少
    broken = ToothGroundTruth(case_uuid="cf_x", source_case_id="1",
                              source_order_id="1", fdi=17,
                              reconstruction_type="AnatomicCrown",
                              has_margin=True, margin_point_count=200,
                              margin_perimeter_mm=30.0, margin_max_segment_mm=12.0)
    assert not is_margin_valid(broken)        # 环是断的


# ============================================================ LocalizationGT 准入
# 提交 2.2：**只允许 FULL_ARCH_WITH_TARGET**。

def test_antagonist_arch_never_enters_localization_set():
    """**回归测试（用户明确要求的那条）。**

    目标 FDI 17 + 完整对颌模型 + Center 有值 —— 绝不能进 LocalizationGTSet。
    对颌弓是完整弓，但目标牙在对侧颌；拿它训练 locator 等于让模型
    在一颗不存在的牙上学习预测。
    """
    from app.groundtruth.models import (
        LOCALIZATION_ADMITTED_CLASS, localization_exclusion_reasons,
    )
    row = ToothGroundTruth(case_uuid="cf_x", source_case_id="82404080845",
                           source_order_id="82404080845", fdi=17,
                           reconstruction_type="AnatomicCrown",
                           has_margin=True, margin_point_count=434,
                           margin_perimeter_mm=32.35, margin_max_segment_mm=0.14,
                           has_axis=True, axis_valid=True,
                           has_center=True, center=(13.9, 10.8, -12.35),
                           has_full_arch_candidate=True,
                           target_center_has_surface=True,
                           arch_is_sectioned_or_unsectioned="unsectioned",
                           arch_asset_class="FULL_ARCH_ANTAGONIST",
                           design_source_kind="CONSTRUCTION_INFO",
                           coordinate_frame_kind="EXPORT_CANONICAL")
    sets = build_all_sets([row])
    assert row not in sets["LocalizationGTSet"]
    assert row not in sets["EndToEndGTSet"]
    # 但它在 Axis/Margin 集合里仍然有效 —— 对颌弓不影响 margin/axis 真值
    assert row in sets["MarginGTSet"] and row in sets["AxisGTSet"]
    reasons = localization_exclusion_reasons([row])
    assert reasons.get("excluded_antagonist") == 1
    assert "admitted" not in reasons
    assert LOCALIZATION_ADMITTED_CLASS == "FULL_ARCH_WITH_TARGET"


def test_localization_exclusion_reasons_name_every_bucket(monkeypatch):
    """排除原因必须能一条命令回答「672 例为什么只剩 N 例」。"""
    from app.groundtruth import human_labels
    from app.groundtruth.models import localization_exclusion_reasons

    def mk(**kw):
        base = dict(case_uuid="cf_x", source_case_id="1", source_order_id="1",
                    fdi=17, reconstruction_type="AnatomicCrown",
                    has_margin=True, margin_point_count=434,
                    margin_perimeter_mm=32.35, margin_max_segment_mm=0.14,
                    has_axis=True, axis_valid=True, has_center=True,
                    center=(0.0, 0.0, 0.0), has_full_arch_candidate=True,
                    design_source_kind="CONSTRUCTION_INFO")
        base.update(kw)
        return ToothGroundTruth(**base)

    # 过了自动预筛之后，去向全由人工判定决定 —— 三条路都要有样本。
    monkeypatch.setitem(human_labels.QA2_LABELS, ("cf_admitted", 17), "A")
    monkeypatch.setitem(human_labels.QA2_LABELS, ("cf_labelC", 17), "C")

    rows = [
        mk(case_uuid="cf_admitted", arch_asset_class="FULL_ARCH_WITH_TARGET"),
        mk(case_uuid="cf_labelC", arch_asset_class="FULL_ARCH_WITH_TARGET"),
        mk(case_uuid="cf_no_verdict", arch_asset_class="FULL_ARCH_WITH_TARGET"),
        mk(case_uuid="cf_ant", arch_asset_class="FULL_ARCH_ANTAGONIST"),
        mk(case_uuid="cf_tm", arch_asset_class="FULL_ARCH_TARGET_MISSING"),
        mk(case_uuid="cf_pa", arch_asset_class="PARTIAL_ARCH"),
        mk(case_uuid="cf_unk", arch_asset_class="UNKNOWN"),
        mk(case_uuid="cf_nc", arch_asset_class="FULL_ARCH_WITH_TARGET",
           has_center=False),
        mk(case_uuid="cf_nac", arch_asset_class="FULL_ARCH_WITH_TARGET",
           has_full_arch_candidate=False),
        mk(case_uuid="cf_ax", arch_asset_class="FULL_ARCH_WITH_TARGET",
           axis_valid=False),
        mk(case_uuid="cf_mg", arch_asset_class="FULL_ARCH_WITH_TARGET",
           margin_point_count=3),
    ]
    reasons = localization_exclusion_reasons(rows)
    assert reasons["admitted"] == 1
    for k in ("excluded_antagonist", "excluded_target_missing",
              "excluded_partial_arch", "excluded_unknown",
              "excluded_missing_center", "excluded_no_arch_candidate",
              "excluded_no_valid_axis", "excluded_not_margin_gt",
              "excluded_no_human_verdict", "excluded_human_label_C"):
        assert reasons.get(k) == 1, f"{k} 计数不对: {reasons}"
    assert sum(reasons.values()) == len(rows)


def test_exclusion_reasons_reconcile_with_the_actual_set(monkeypatch):
    """**回归**：报告的 ``admitted`` 必须等于 ``LocalizationGTSet`` 的实际大小。

    这里踩过一个坑：reasons 链漏了 ``is_geometric_gt_source``
    （``margin_gt_set`` 的第三道门），于是 modelInfo 行被算成 admitted，
    报告说 9 而集合实际只有 4 —— **报告在撒谎，而且只有把两边对账才发现**。
    所以这个测试不比常量，而是把两个函数放在一起对账。
    """
    from app.groundtruth import human_labels
    from app.groundtruth.models import (
        build_all_sets,
        localization_exclusion_reasons,
    )

    def mk(case_uuid, src, **kw):
        base = dict(case_uuid=case_uuid, source_case_id="1",
                    source_order_id="1", fdi=17,
                    reconstruction_type="AnatomicCrown",
                    has_margin=True, margin_point_count=434,
                    margin_perimeter_mm=32.35, margin_max_segment_mm=0.14,
                    has_axis=True, axis_valid=True, has_center=True,
                    center=(0.0, 0.0, 0.0), has_full_arch_candidate=True,
                    arch_asset_class="FULL_ARCH_WITH_TARGET",
                    design_source_kind=src)
        base.update(kw)
        return ToothGroundTruth(**base)

    rows = [
        mk("cf_ci_A", "CONSTRUCTION_INFO"),
        mk("cf_mi_A", "MODEL_INFO"),        # 人工也是 A，但快照来源不合格
        mk("cf_ci_C", "CONSTRUCTION_INFO"),
        mk("cf_ci_none", "CONSTRUCTION_INFO"),   # 无人判定
    ]
    monkeypatch.setitem(human_labels.QA2_LABELS, ("cf_ci_A", 17), "A")
    monkeypatch.setitem(human_labels.QA2_LABELS, ("cf_mi_A", 17), "A")
    monkeypatch.setitem(human_labels.QA2_LABELS, ("cf_ci_C", 17), "C")

    reasons = localization_exclusion_reasons(rows)
    members = build_all_sets(rows)["LocalizationGTSet"]
    assert [m.case_uuid for m in members] == ["cf_ci_A"]
    assert reasons["admitted"] == len(members) == 1
    assert reasons.get("excluded_not_geometric_gt_source") == 1
    assert reasons.get("excluded_human_label_C") == 1
    assert reasons.get("excluded_no_human_verdict") == 1
    assert sum(reasons.values()) == len(rows)


# ============================================================ 快照来源政策
# design-source audit 的结论落进代码：几何 GT 只认 constructionInfo。

def _gt_row(**kw):
    base = dict(case_uuid="cf_x", source_case_id="1", source_order_id="1",
                fdi=17, reconstruction_type="AnatomicCrown",
                has_margin=True, margin_point_count=434,
                margin_perimeter_mm=32.35, margin_max_segment_mm=0.14,
                has_axis=True, axis_valid=True, has_center=True,
                center=(0.0, 0.0, 0.0), has_final_crown=True,
                final_crown_fdi_count=1, final_crown_single_unit=True,
                final_crown_watertight=True, margin_crown_consistent=True,
                has_modeldie=True, modeldie_same_frame=True)
    base.update(kw)
    return ToothGroundTruth(**base)


def test_modelinfo_rows_are_excluded_from_every_geometric_gt_set():
    """**本轮的核心政策**：modelInfo 快照不得进入任何几何 GT 集合。

    实测依据（全量 3566 对拟合）：modelInfo 与 constructionInfo 的颈缘是
    同一条折线（段长差 med 0.078mm）但**不是**刚体/相似/镜像关系
    （rigid 残差 med 1.69mm，0 对 <=0.01mm），变换与 XML 所有矩阵
    精确匹配 0/3566，mi-only 的 418 个 Margin 牙位全部无冠可验证。
    """
    ci_row = _gt_row(design_source_kind="CONSTRUCTION_INFO",
                     coordinate_frame_kind="EXPORT_CANONICAL",
                     geometry_validation_status="CANONICAL")
    mi_row = _gt_row(design_source_kind="MODEL_INFO",
                     coordinate_frame_kind="MODEL_INFO_NATIVE",
                     geometry_validation_status="UNVALIDATABLE_NO_CROWN")
    sets = build_all_sets([ci_row, mi_row])
    for name in ("MarginGTSet", "AxisGTSet", "SupportGTSet",
                 "OuterCrownGTSet", "SingleAnatomicCrownGTSet"):
        members = sets[name]
        assert ci_row in members, f"ci 行应在 {name}"
        assert mi_row not in members, f"mi 行不得进 {name}"


def test_indexer_stamps_provenance_from_the_parsed_snapshot(tmp_path):
    """indexer 必须为每行盖 provenance 章：来源种类、来源文件哈希、坐标帧。"""
    from app.groundtruth.indexer import discover_cases
    from tests.unit.importers.test_construction_design import (
        build_construction_info,
    )
    root = tmp_path / "all_data"
    d = root / "82404080845"
    d.mkdir(parents=True)
    (d / "CASE.constructionInfo").write_text(build_construction_info(),
                                             encoding="utf-8")
    rows = build_index(discover_cases([str(root)]), crown_topology=False,
                       do_hash=False)
    for r in rows:
        assert r.design_source_kind == "CONSTRUCTION_INFO"
        assert r.coordinate_frame_kind == "EXPORT_CANONICAL"
        assert r.design_source_asset_id  # 有哈希
        assert r.geometry_validation_status in ("CANONICAL",)


def test_indexer_stamps_modelinfo_rows_and_gates_them(tmp_path):
    """只有 modelInfo 的病例：行照常生成（metadata/研究候选），但 provenance
    如实标注，且进不了几何 GT。"""
    import re
    from app.groundtruth.indexer import discover_cases
    from tests.unit.importers.test_construction_design import (
        build_construction_info,
    )
    root = tmp_path / "all_data"
    d = root / "82404080999"
    d.mkdir(parents=True)
    xml = build_construction_info()
    xml = re.sub(r"<Margin>.*?</Margin>", "<Margin>" + _dense_margin_xml() + "</Margin>",
                 xml, flags=re.S)
    (d / "CASE.modelInfo").write_text(xml, encoding="utf-8")
    rows = build_index(discover_cases([str(root)]), crown_topology=False,
                       do_hash=False)
    assert rows
    for r in rows:
        assert r.design_source_kind == "MODEL_INFO"
        assert r.coordinate_frame_kind == "MODEL_INFO_NATIVE"
        assert r.geometry_validation_status == "UNVALIDATABLE_NO_CROWN"
    sets = build_all_sets(rows)
    assert all(not sets[name] for name in
               ("MarginGTSet", "AxisGTSet", "SingleAnatomicCrownGTSet"))


def test_geometric_gt_source_is_explicit_policy():
    from app.groundtruth.models import (
        CANONICAL_DESIGN_SOURCE, FRAME_EXPORT_CANONICAL, is_geometric_gt_source,
    )
    assert CANONICAL_DESIGN_SOURCE == "CONSTRUCTION_INFO"
    assert is_geometric_gt_source(
        _gt_row(design_source_kind="CONSTRUCTION_INFO"))
    assert not is_geometric_gt_source(
        _gt_row(design_source_kind="MODEL_INFO"))
    assert not is_geometric_gt_source(_gt_row(design_source_kind="UNKNOWN"))
    assert FRAME_EXPORT_CANONICAL == "EXPORT_CANONICAL"


def test_freeze_round_trips_provenance_columns(tmp_path):
    """freeze 读取索引时不得丢 provenance（否则集合闸会在冻结后失效）。"""
    from app.groundtruth import freeze as fz
    row = _gt_row(design_source_kind="CONSTRUCTION_INFO",
                  coordinate_frame_kind="EXPORT_CANONICAL",
                  geometry_validation_status="CANONICAL")
    from app.groundtruth.models import COLUMNS
    flat = {k: "" for k in COLUMNS}
    flat.update(row.as_row())
    import csv as _csv
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    with (index_dir / "tooth_index.csv").open("w", newline="",
                                              encoding="ascii") as f:
        w = _csv.DictWriter(f, fieldnames=list(COLUMNS))
        w.writeheader()
        w.writerow(flat)
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "arch_asset_audit.csv").write_text(
        "case_uuid,target_fdi,candidate_class\n"
        "cf_x,17,FULL_ARCH_WITH_TARGET\n",
        encoding="ascii")
    man = fz.freeze(index_dir, audit_dir, tmp_path / "out", "test_v1")
    frozen = list(_csv.DictReader(
        (tmp_path / "out" / "frozen_tooth_index.csv").open(encoding="ascii")))
    assert frozen[0]["design_source_kind"] == "CONSTRUCTION_INFO"
    assert frozen[0]["coordinate_frame_kind"] == "EXPORT_CANONICAL"
    assert man["set_counts"]["SingleAnatomicCrownGTSet"] == 1


# ============================================================ ⑦ 代型资产身份政策
# 用户决策 2026-09-30 ③：**没有文件名/几何证据时，绝不能再把 nearest STL
# 命名成 modeldie**。旧回退（bbox 中心离设计中心最近的 STL）实测把修复体 CAD
# 壳当成了代型（94.9% 的 eligible 行），详见 docs/AUTO_MARGIN_M0_ASSET_AUDIT.md。

def test_fallback_asset_is_never_called_modeldie(tmp_path):
    """只有编号 STL（没有 *modeldie*）时：只写 nearest_asset_*，不写 modeldie。"""
    import app.groundtruth.indexer as ix
    root = tmp_path / "all_data"
    d = build_case_tree(root, case_dir_name="82404080845")
    (d / "CASE-17-modeldie.stl").unlink()      # 去掉唯一的文件名证据
    cases = discover_cases([str(root)])
    ix._DIE_SELECTION.clear()
    rows = build_index(cases, crown_topology=False, do_hash=False)
    row = next(r for r in rows if r.fdi == 17)     # 夹具含多个牙位
    assert row.has_modeldie is False, "回退资产被当成了代型"
    assert row.modeldie_asset_id is None
    assert row.nearest_asset_id is not None, "回退候选仍应记录（供诊断）"
    assert row.asset_selection_method == "NEAREST_TO_CENTER_FALLBACK"


def test_filename_evidence_sets_modeldie_with_method(tmp_path):
    """有 *modeldie* 且文件名含 FDI：has_modeldie=True + FILENAME_AND_FDI。"""
    import app.groundtruth.indexer as ix
    root = tmp_path / "all_data"
    build_case_tree(root, case_dir_name="82404080845")
    ix._DIE_SELECTION.clear()
    rows = build_index(discover_cases([str(root)]), crown_topology=False,
                       do_hash=False)
    row = next(r for r in rows if r.fdi == 17)
    assert row.has_modeldie is True
    assert row.modeldie_asset_id
    assert row.asset_selection_method == "FILENAME_AND_FDI"


def test_geometric_evidence_beats_filename(tmp_path):
    """几何证据（Margin 投影验证）优先于文件名证据，且能救回编号 STL。"""
    from app.groundtruth.assets import content_hash
    import app.groundtruth.indexer as ix
    root = tmp_path / "all_data"
    d = build_case_tree(root, case_dir_name="82404080845")
    (d / "CASE-17-modeldie.stl").unlink()      # 没有文件名证据
    cases = discover_cases([str(root)])
    # 先用一次空证据表跑一遍拿到该行的 case_uuid（真实流程里它来自索引行）
    ix._DIE_SELECTION.clear()
    uuid = next(r.case_uuid for r in
                build_index(cases, crown_topology=False, do_hash=False)
                if r.fdi == 17)
    target = d / "CASE-17-crown_cad.stl"       # 编号命名的候选
    h = content_hash(str(target))
    sel = tmp_path / "die_selection.csv"
    sel.write_text("case_uuid,fdi,die_asset_id,err_p95_mm,selection\n"
                   "%s,17,%s,0.0001,MARGIN_ON_DIE\n" % (uuid, h),
                   encoding="ascii")
    ix._DIE_SELECTION.clear()
    assert ix.load_die_selection(str(sel)) == 1
    rows = build_index(cases, crown_topology=False, do_hash=True)
    row = next(r for r in rows if r.fdi == 17)
    assert row.has_modeldie is True
    assert row.asset_selection_method == "MARGIN_PROJECTION_VERIFIED"
    assert row.modeldie_asset_id == h
    ix._DIE_SELECTION.clear()


def test_die_selection_ignores_unverified_rows(tmp_path):
    """selection != MARGIN_ON_DIE 的行不进证据表（best 仍不达标就不算证据）。"""
    import app.groundtruth.indexer as ix
    sel = tmp_path / "die_selection.csv"
    sel.write_text("case_uuid,fdi,die_asset_id,err_p95_mm,selection\n"
                   "cf_aaa,17,sha:deadbeef,3.5,BEST_STILL_OFF\n",
                   encoding="ascii")
    ix._DIE_SELECTION.clear()
    assert ix.load_die_selection(str(sel)) == 0
    assert ix._DIE_SELECTION == {}
