"""``.eoff`` 解析与专业牙库导入的测试。

**用合成的 .eoff 字节**，不依赖仓库外的真实牙库（真实数据只在本机，
根目录由 ``CROWNFLOW_TOOTH_LIBRARY_ROOT`` 指定）。真实库的检查放在
``test_real_library_*``，没有环境变量就跳过。
"""
import math
import os
import struct

import pytest

from app.importers.tooth_library import (
    ENV_LIBRARY_ROOT,
    EoffError,
    ToothLibraryError,
    closed_loop_bits,
    discover_libraries,
    jaw_for_fdi,
    label_bits_present,
    parse_bow_xml,
    parse_eoff_bytes,
    sections_cover_file,
    tooth_index_for_fdi,
)
from app.importers.tooth_library.eoff import (
    FACE_RECORD,
    HEADER_LEN,
    NAME_FIELD,
    parse_eoff,
)

REAL_ROOT = os.environ.get(ENV_LIBRARY_ROOT)


# ============================================================ 合成 eoff 构造

def build_eoff(verts, faces, labels=None, sections=(), magic=b"OFF BINARY"):
    """按已对账的布局写出 .eoff 字节。

    sections: [(name, payload_or_None)]；``END_EXTENSION`` 会自动补在最后。
    """
    body = bytearray()
    body += magic + b"\n"
    n_edges = 3 * len(faces) // 2
    body += struct.pack("<3I", len(verts), len(faces), n_edges)
    for v in verts:
        body += struct.pack("<3f", *v)
    for f in faces:
        body += struct.pack("<H", 3) + struct.pack("<3I", *f)
    for name, payload in list(sections):
        nm = name.encode("ascii")
        assert len(nm) <= NAME_FIELD
        body += nm + b"\x00" * (NAME_FIELD - len(nm))
        if payload is not None:
            body += struct.pack("<I", len(payload)) + payload
    nm = b"END_EXTENSION"
    body += nm + b"\x00" * (NAME_FIELD - len(nm))
    return bytes(body)


#: 正四面体：4 顶点 4 面（闭合）。
TETRA_V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
TETRA_F = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def octahedron(radius=5.0, n_lat=1):
    """八面体（闭合，6 顶点 8 面）—— 用来测闭环之外的东西。"""
    v = [(radius, 0, 0), (-radius, 0, 0), (0, radius, 0),
         (0, -radius, 0), (0, 0, radius), (0, 0, -radius)]
    f = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
         (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    return v, f


def ring_band(n=12, radius=4.0, z=0.0):
    """造一个"环"：两圈顶点，中间一圈带标签 —— 用于测闭环判据。

    返回 (verts, faces, loop_indices)。
    """
    verts, faces = [], []
    lo, hi = z - 2.0, z + 2.0
    for k, zz in enumerate((lo, z, hi)):
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append((radius * math.cos(a), radius * math.sin(a), zz))
    for k in range(2):
        for i in range(n):
            j = (i + 1) % n
            a, b = k * n + i, k * n + j
            c, d = (k + 1) * n + j, (k + 1) * n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    loop = [n + i for i in range(n)]          # 中间那圈
    return verts, faces, loop


# ============================================================ 头部与布局

def test_parses_header_vertices_and_faces():
    raw = build_eoff(TETRA_V, TETRA_F)
    m = parse_eoff_bytes(raw)
    assert m.n_vertices == 4 and m.n_faces == 4
    assert m.declared_edge_count == 6
    assert m.vertices[1] == (1.0, 0.0, 0.0)
    assert m.faces == TETRA_F
    assert m.content_hash and len(m.content_hash) == 16


def test_rejects_non_off_binary_magic():
    with pytest.raises(EoffError, match="OFF BINARY"):
        parse_eoff_bytes(b"solid foo\n" + b"\x00" * 64)


def test_rejects_truncated_face_block():
    raw = build_eoff(TETRA_V, TETRA_F)
    with pytest.raises(EoffError, match="面块越界"):
        parse_eoff_bytes(raw[:HEADER_LEN + 4 * 12 + 8])


def test_rejects_non_triangle_face_record():
    raw = bytearray(build_eoff(TETRA_V, TETRA_F))
    off = HEADER_LEN + 4 * 12
    struct.pack_into("<H", raw, off, 4)       # 伪造成四边形
    with pytest.raises(EoffError, match="顶点数是 4"):
        parse_eoff_bytes(bytes(raw))


def test_rejects_out_of_range_face_index():
    raw = bytearray(build_eoff(TETRA_V, TETRA_F))
    off = HEADER_LEN + 4 * 12 + 2
    struct.pack_into("<I", raw, off, 99)
    with pytest.raises(EoffError, match="索引越界"):
        parse_eoff_bytes(bytes(raw))


def test_face_record_size_matches_the_documented_layout():
    """面记录 14 字节是本格式最容易搞错的地方（不是 12 也不是 16）。"""
    assert FACE_RECORD == 14 and HEADER_LEN == 23 and NAME_FIELD == 25


# ============================================================ 扩展段

def test_sections_with_and_without_payload_round_trip():
    payload = struct.pack("<4I", 1, 2, 3, 4)
    raw = build_eoff(TETRA_V, TETRA_F,
                     sections=[("MARKER", None),
                               ("VERTEX_FEATURE_LABELS", payload)])
    m = parse_eoff_bytes(raw)
    names = [s.name for s in m.sections]
    assert names == ["MARKER", "VERTEX_FEATURE_LABELS", "END_EXTENSION"]
    assert m.sections[0].payload is None
    assert m.sections[1].n_uint32 == 4
    assert m.sections[1].payload == payload


def test_sections_cover_file_is_reported(tmp_path):
    p = tmp_path / "t.eoff"
    p.write_bytes(build_eoff(TETRA_V, TETRA_F,
                             sections=[("MARKER", None)]))
    assert sections_cover_file(str(p)) is True


def test_label_count_must_equal_vertex_count():
    """标签项数与 V 不符时要明确报错 —— 这是格式对账的关键不变量。"""
    bad = struct.pack("<3I", 1, 2, 3)      # 只有 3 项，V=4
    raw = build_eoff(TETRA_V, TETRA_F,
                     sections=[("VERTEX_FEATURE_LABELS", bad)])
    with pytest.raises(EoffError, match="与 V=4 不符"):
        parse_eoff_bytes(raw)


# ============================================================ 逐顶点标签

def test_labels_are_read_per_vertex():
    labels = [0, 0x800, 0x800, 0x400 | 0x800]
    raw = build_eoff(TETRA_V, TETRA_F,
                     sections=[("VERTEX_FEATURE_LABELS",
                                struct.pack("<4I", *labels))])
    m = parse_eoff_bytes(raw)
    assert m.labels == labels
    assert label_bits_present(m) == [0x400, 0x800]


def test_without_labels_section_labels_are_none():
    m = parse_eoff_bytes(build_eoff(TETRA_V, TETRA_F))
    assert m.labels is None
    assert label_bits_present(m) == []
    assert closed_loop_bits(m) == []


def test_closed_loop_detection_finds_the_middle_ring():
    """**核心判据**：牙颈线必须是一维闭环（单一分量 + 度数恰好 2）。"""
    verts, faces, loop = ring_band(n=12)
    labels = [0] * len(verts)
    for i in loop:
        labels[i] = 0x800
    raw = build_eoff(verts, faces,
                     sections=[("VERTEX_FEATURE_LABELS",
                                struct.pack("<%dI" % len(labels), *labels))])
    m = parse_eoff_bytes(raw)
    cands = closed_loop_bits(m)
    top = [c for c in cands if c.is_clean_loop]
    assert len(top) == 1, cands
    c = top[0]
    assert c.bit == 0x800
    assert c.n_vertices == 12
    assert c.n_components == 1 and c.max_degree == 2
    # 周长 = 12 边形，外接圆半径 4mm
    expect = 12 * 2 * 4.0 * math.sin(math.pi / 12)
    assert c.perimeter_mm == pytest.approx(expect, rel=0.02)


def test_a_2d_patch_is_not_reported_as_a_loop():
    """整片冠部（度数 6）不能被当成牙颈线 —— 否则边界条件就错了。"""
    verts, faces, _loop = ring_band(n=12)
    labels = [0] * len(verts)
    for i in range(12, 24):                    # 中间整圈 + 上圈 = 一片
        labels[i] = 0x800
    for i in range(24, 36):
        labels[i] = 0x800
    raw = build_eoff(verts, faces,
                     sections=[("VERTEX_FEATURE_LABELS",
                                struct.pack("<%dI" % len(labels), *labels))])
    m = parse_eoff_bytes(raw)
    clean = [c for c in closed_loop_bits(m) if c.is_clean_loop]
    assert clean == [], clean


def test_a_broken_ring_is_not_a_loop():
    """断开的环（有两个度数 1 的端点）不是牙颈线。

    **这条测试抓到过一个真 bug**：判据原先只看 max_degree == 2，
    而开弧的端点度数是 1、内部是 2，于是 max 也等于 2 —— 断环被误判成闭环。
    修法是同时要求 min_degree == 2。边界条件用错会把形变拉断。
    """
    verts, faces, loop = ring_band(n=12)
    labels = [0] * len(verts)
    for i in loop[:-2]:                        # 故意少两个顶点
        labels[i] = 0x800
    raw = build_eoff(verts, faces,
                     sections=[("VERTEX_FEATURE_LABELS",
                                struct.pack("<%dI" % len(labels), *labels))])
    m = parse_eoff_bytes(raw)
    clean = [c for c in closed_loop_bits(m) if c.is_clean_loop]
    assert clean == []


# ============================================================ FDI 映射

def test_fdi_maps_to_jaw_and_file_index():
    assert jaw_for_fdi(46) == "lowerjaw"
    assert jaw_for_fdi(36) == "lowerjaw"
    assert jaw_for_fdi(16) == "upperjaw"
    assert jaw_for_fdi(26) == "upperjaw"
    for fdi in (11, 18, 21, 28, 31, 38, 41, 48):
        assert 1 <= tooth_index_for_fdi(fdi) <= 8
    assert tooth_index_for_fdi(46) == 6
    assert tooth_index_for_fdi(11) == 1


def test_illegal_fdi_is_rejected():
    with pytest.raises(ToothLibraryError):
        jaw_for_fdi(56)
    with pytest.raises(ToothLibraryError):
        tooth_index_for_fdi(49)


# ============================================================ 颌 XML

BOW_XML = """<?xml version="1.0"?>
<ToothBow>
  <ModelTeeth>
    <ModelTooth>
      <Number>46</Number>
      <Position><x>1</x><y>2</y><z>3</z></Position>
      <AxisOcclusal><x>0</x><y>0</y><z>2</z></AxisOcclusal>
      <AxisMesial><x>1</x><y>0</y><z>0</z></AxisMesial>
      <BuccocervicalPoint><x>0</x><y>0</y><z>0</z></BuccocervicalPoint>
    </ModelTooth>
  </ModelTeeth>
</ToothBow>
"""


def test_bow_xml_parses_position_and_axes(tmp_path):
    p = tmp_path / "lowerjaw.xml"
    p.write_text(BOW_XML)
    bow = parse_bow_xml(str(p))
    assert set(bow) == {46}
    e = bow[46]
    assert e.position == (1.0, 2.0, 3.0)
    _org, z, x, y = e.frame()
    assert z == pytest.approx((0.0, 0.0, 1.0))      # 归一化
    assert x == pytest.approx((1.0, 0.0, 0.0))
    # y = z × x，右手系
    assert y == pytest.approx((0.0, 1.0, 0.0))


def test_bow_xml_rejects_wrong_root(tmp_path):
    p = tmp_path / "x.xml"
    p.write_text("<NotABow/>")
    with pytest.raises(ToothLibraryError, match="ToothBow"):
        parse_bow_xml(str(p))


def test_zero_landmarks_are_reported_as_absent(tmp_path):
    """XML 的解剖标志点全零 —— 所以牙颈线不能从 XML 拿（只能从 .eoff 标签）。"""
    from app.importers.tooth_library import landmark_coverage
    p = tmp_path / "lowerjaw.xml"
    p.write_text(BOW_XML)
    cov = landmark_coverage(str(p))
    assert cov["BuccocervicalPoint"] == 0
    assert all(v == 0 for v in cov.values())


def test_discover_libraries_requires_a_root():
    if REAL_ROOT:
        pytest.skip("本机设了真实牙库环境变量，跳过缺省值检查")
    with pytest.raises(ToothLibraryError, match=ENV_LIBRARY_ROOT):
        discover_libraries(None)


def test_discover_libraries_rejects_missing_teeth_dir(tmp_path):
    with pytest.raises(ToothLibraryError, match="teeth/"):
        discover_libraries(str(tmp_path))


def test_discover_libraries_lists_every_library_dir(tmp_path):
    for name in ("A", "B"):
        (tmp_path / "teeth" / name).mkdir(parents=True)
    libs = discover_libraries(str(tmp_path))
    assert [l.name for l in libs] == ["A", "B"]


# ============================================================ 开放底边界

def _cone_open_bottom(n=16, r=4.0, z=6.0):
    """底部开放的锥：底环 n 个顶点 + 顶尖 1 个。边界 = 底环。"""
    verts = [(r * math.cos(2 * math.pi * i / n),
              r * math.sin(2 * math.pi * i / n), 0.0) for i in range(n)]
    verts.append((0.0, 0.0, z))                     # 顶尖
    faces = [(i, (i + 1) % n, n) for i in range(n)]
    return verts, faces


def test_boundary_loop_returns_ordered_open_ring():
    from app.importers.tooth_library import boundary_loop
    verts, faces = _cone_open_bottom()
    m = parse_eoff_bytes(build_eoff(verts, faces))
    loop = boundary_loop(m)
    assert loop is not None
    assert len(loop) == 16
    assert set(loop) == set(range(16)), "底环顶点应恰好是全部边界顶点"
    # 有序：相邻两项必须是网格邻居
    adj = m.vertex_adjacency()
    for a, b in zip(loop, loop[1:] + loop[:1]):
        assert b in adj[a]


def test_boundary_loop_rejects_two_disjoint_rings():
    """ring_band 上下都开口（两个独立边界环）→ 不能唯一确定，返回 None。"""
    from app.importers.tooth_library import boundary_loop
    verts, faces, _loop = ring_band(n=12)
    m = parse_eoff_bytes(build_eoff(verts, faces))
    assert boundary_loop(m) is None


def test_boundary_loop_none_for_watertight_mesh():
    from app.importers.tooth_library import boundary_loop
    assert boundary_loop(parse_eoff_bytes(build_eoff(TETRA_V, TETRA_F))) is None


# ============================================================ 真实牙库（可选）

needs_real = pytest.mark.skipif(
    not REAL_ROOT or not os.path.isdir(REAL_ROOT or ""),
    reason="未设置 %s（真实牙库在仓库之外）" % ENV_LIBRARY_ROOT)


@needs_real
def test_real_library_fdi46_template_parses_and_has_a_clean_margin_loop():
    """真实牙库 FDI46：模板可解析，且**至少有一位构成一维闭环**（牙颈线）。

    这是 CrownOuter v0 的边界条件来源，所以必须由测试守住。
    """
    from app.importers.tooth_library import find_margin_loop
    lib = _chosen_library()
    m = lib.template_mesh(46)
    assert m.n_vertices > 1000 and m.n_faces > 1000
    assert m.labels is not None and len(m.labels) == m.n_vertices
    got = find_margin_loop(m)
    assert got is not None, "FDI46 模板没有可恢复的牙颈线闭环"
    cand, bits = got
    assert 15.0 < cand.perimeter_mm < 60.0, cand
    assert cand.is_clean_loop


@needs_real
def test_real_library_chosen_library_covers_all_32_teeth():
    """**选库的依据**：炼金王 32/32 有牙颈环；超神2 只有 14/32。

    这条测试把"为什么第一颗牙固定用炼金王"钉住 —— 不是为了好看，
    而是另外两个库里有一个的标注不完整。
    """
    from app.importers.tooth_library import find_margin_loop
    fdis = [q * 10 + n for q in (1, 2, 3, 4) for n in range(1, 9)]
    coverage = {}
    for lib in discover_libraries():
        n_ok = sum(1 for f in fdis
                   if find_margin_loop(lib.template_mesh(f)) is not None)
        coverage[lib.name] = n_ok
    assert coverage["炼金王"] == 32, coverage
    assert coverage.get("洋溢-长夜同在") == 32, coverage
    # 超神2 的标注不完整：这不是我们解析的问题，所以记录成事实而不是失败
    assert coverage.get("超神2", 0) < 32, coverage


@needs_real
def test_real_library_missing_margin_loop_is_reported_not_guessed():
    """超神2 FDI46 取不到牙颈环 —— 必须返回 None，而不是随便挑一位。"""
    from app.importers.tooth_library import find_margin_loop
    libs = {l.name: l for l in discover_libraries()}
    if "超神2" not in libs:
        pytest.skip("本机牙库没有 超神2")
    assert find_margin_loop(libs["超神2"].template_mesh(46)) is None


@needs_real
def test_real_library_margin_loop_bit_identity_is_the_same_for_chosen_library():
    """选定的库里，牙颈线位跨牙位一致（0x800），便于审计与对账。"""
    from app.importers.tooth_library import find_margin_loop
    lib = _chosen_library()
    for fdi in (46, 36, 16, 26, 11, 47):
        cand, bits = find_margin_loop(lib.template_mesh(fdi))
        assert bits == frozenset({0x800}), (fdi, bits)


def _chosen_library():
    from app.importers.tooth_library import find_margin_loop  # noqa: F401
    libs = {l.name: l for l in discover_libraries()}
    if "炼金王" not in libs:
        pytest.skip("本机牙库没有 炼金王")
    return libs["炼金王"]


# ============================================================ 依赖约束

def test_tooth_library_imports_without_numpy_or_trimesh():
    """本包必须能在只有 stdlib 的数据机上导入（解析与对账都在那边跑）。

    只扫源码不够 —— 传递依赖看不出来（``app/groundtruth`` 就踩过这个坑）。
    所以真的在阻断 numpy/trimesh/scipy 的子进程里导入一次。
    """
    import pathlib
    import subprocess
    import sys
    root = pathlib.Path(__file__).resolve().parents[3]
    code = chr(10).join([
        "import sys",
        "for m in ('numpy', 'trimesh', 'scipy'):",
        "    sys.modules[m] = None",
        "import app.importers.tooth_library",
        "from app.importers.tooth_library import parse_eoff, find_margin_loop",
        "print('ok')",
    ])
    r = subprocess.run([sys.executable, "-c", code], cwd=str(root),
                       capture_output=True, text=True)
    assert r.returncode == 0, (
        "阻断 numpy/trimesh/scipy 后导入失败：" + r.stderr[-1200:])
