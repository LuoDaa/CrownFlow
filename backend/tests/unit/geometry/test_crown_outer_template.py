"""CrownOuter 模板标准化的测试。

合成网格测切分规则（不依赖仓库外的真实牙库）；真实牙库的不变量用
``CROWNFLOW_TOOTH_LIBRARY_ROOT`` 门控。
"""
import math
import os

import numpy as np
import pytest

from app.geometry.crown_outer import (
    TemplateError,
    component_labels,
    load_crown_template,
    split_crown_and_root,
)

REAL_ROOT = os.environ.get("CROWNFLOW_TOOTH_LIBRARY_ROOT")


# ============================================================ 合成网格

def tube(n=12, r=4.0, zs=(-6.0, -3.0, 0.0, 3.0, 6.0), radii=None):
    """沿 z 的一串圆环 + 三角带。``radii`` 可逐环给半径（造腰/锥）。"""
    radii = radii or [r] * len(zs)
    verts = []
    for z, rr in zip(zs, radii):
        for i in range(n):
            a = 2 * math.pi * i / n
            verts.append((rr * math.cos(a), rr * math.sin(a), z))
    faces = []
    for k in range(len(zs) - 1):
        for i in range(n):
            j = (i + 1) % n
            a, b = k * n + i, k * n + j
            c, d = (k + 1) * n + j, (k + 1) * n + i
            faces.append((a, b, c))
            faces.append((a, c, d))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def ring_at(k, n=12):
    return np.arange(k * n, (k + 1) * n, dtype=np.int64)


# ============================================================ 连通分量

def test_component_labels_splits_at_the_excluded_ring():
    v, f = tube()
    seam = ring_at(2)                      # z=0 那圈
    ex = np.zeros(len(v), dtype=bool)
    ex[seam] = True
    lab = component_labels(len(v), f, ex)
    assert (lab[seam] == -1).all()
    below = set(lab[ring_at(0)].tolist()) | set(lab[ring_at(1)].tolist())
    above = set(lab[ring_at(3)].tolist()) | set(lab[ring_at(4)].tolist())
    assert -1 not in below and -1 not in above
    assert len(below) == 1 and len(above) == 1
    assert below != above, "环两侧必须是不同分量"


def test_component_labels_keeps_ring_intact_when_nothing_excluded():
    v, f = tube()
    lab = component_labels(len(v), f, np.zeros(len(v), dtype=bool))
    assert len(set(lab.tolist())) == 1


# ============================================================ 拓扑切分

def test_split_puts_crown_above_and_root_below_a_waist_ring():
    """腰部（半径最小的一圈）做缝 → 冠在 +Z 侧、根在 -Z 侧。"""
    v, f = tube(radii=[3.0, 4.5, 2.0, 4.5, 3.2])
    seam = ring_at(2)                      # 半径 2.0，明显是腰
    crown, root, info = split_crown_and_root(v, f, seam)
    assert not crown[seam].any() and not root[seam].any()
    assert crown.sum() + root.sum() + len(seam) == len(v)
    # 冠 = 最高那两圈
    assert crown[ring_at(3)].all() and crown[ring_at(4)].all()
    assert root[ring_at(0)].all() and root[ring_at(1)].all()
    assert info["n_components_after_seam"] == 2
    assert info["n_other_components"] == 0


def test_split_rejects_a_seam_that_is_not_a_topological_waist():
    """缝不绕完一圈（不是拓扑腰）时必须报错，不能"切开了"却切成错的。

    这是最危险的失败模式：环没绕完，冠与根仍连通，若静默按连通分量取，
    会把整个网格当冠部，形变就会作用到牙根上。
    """
    v, f = tube(radii=[3.0, 4.5, 2.0, 4.5, 3.2])
    seam = ring_at(2)[:-3]                 # 故意只取环的一部分
    with pytest.raises(TemplateError, match="不是拓扑腰"):
        split_crown_and_root(v, f, seam)


def test_split_counts_stray_fragments_instead_of_hiding_them():
    """被缝隔开的孤立碎片要如实计入并报数，不能静默吞掉。"""
    v, f = tube()
    # 加一个孤立三角（与主管不相连）在中间高度
    extra = np.array([[0.0, 0.0, 0.5], [0.2, 0.0, 0.5], [0.0, 0.2, 0.5]])
    v2 = np.vstack([v, extra])
    f2 = np.vstack([f, np.array([[len(v), len(v) + 1, len(v) + 2]])])
    crown, root, info = split_crown_and_root(v2, f2, ring_at(2))
    assert info["n_other_components"] == 1
    assert info["n_other_vertices"] == 3
    # 碎片归到根部（保守：不属于冠部就不该参与冠部形变）
    assert root[len(v):].all()
    assert not crown[len(v):].any()


def test_split_rejects_empty_remainder():
    v, f = tube(n=3, zs=(0.0, 1.0), radii=[1.0, 1.0])
    with pytest.raises(TemplateError, match="没有剩余顶点"):
        split_crown_and_root(v, f, np.arange(6, dtype=np.int64))


# ============================================================ 真实牙库（可选）

needs_real = pytest.mark.skipif(
    not REAL_ROOT or not os.path.isdir(REAL_ROOT or ""),
    reason="未设置 CROWNFLOW_TOOTH_LIBRARY_ROOT（真实牙库在仓库之外）")


def _chosen():
    from app.importers.tooth_library import discover_libraries
    libs = {l.name: l for l in discover_libraries()}
    if "炼金王" not in libs:
        pytest.skip("本机牙库没有 炼金王")
    return libs["炼金王"]


@needs_real
def test_real_template_dispatch_prefers_label_loop_over_holes():
    """**分发优先级守卫**：标签环优先于开放边界。

    炼金王 FDI46 的网格有一个 82 顶点、周长 9.7mm 的小破洞（且在轴向下半部
    —— 位置检查拦不住它）。若按"有开放边界就用边界"，会把这个破洞当成
    牙颈边界（周长 9.7mm vs 真颈环 33.9mm），形变边界就完全错了。
    显式标签是设计者意图，优先级必须更高。
    """
    from app.importers.tooth_library import discover_libraries
    from app.geometry.crown_outer.template import (
        MARGIN_SOURCE_LABEL_LOOP,
        MARGIN_SOURCE_OPEN_BOUNDARY,
    )
    libs = {l.name: l for l in discover_libraries()}
    if "炼金王" not in libs:
        pytest.skip("本机牙库没有 炼金王")
    t = load_crown_template(libs["炼金王"], 46)
    assert t.margin_source == MARGIN_SOURCE_LABEL_LOOP
    assert t.margin_loop_bit == (0x800,)
    assert 30.0 < t.margin_perimeter_mm < 38.0
    # 有根牙形：必须完成了冠/根切分
    assert t.root_vertex_mask.sum() > 1000


@needs_real
def test_real_template_open_boundary_fallback_for_crown_only_mesh():
    """冠形网格（超神2，底面开放、无根）走开放边界回退路径。

    该库的标签不成环（边界顶点邻接少），标签路径自然失败，
    底边界环（FDI46：111 顶点全部度数 2）就是牙颈边界。
    """
    from app.importers.tooth_library import discover_libraries
    from app.geometry.crown_outer.template import MARGIN_SOURCE_OPEN_BOUNDARY
    libs = {l.name: l for l in discover_libraries()}
    if "超神2" not in libs:
        pytest.skip("本机牙库没有 超神2")
    t = load_crown_template(libs["超神2"], 46)
    assert t.margin_source == MARGIN_SOURCE_OPEN_BOUNDARY
    assert t.margin_loop_bit == ()
    assert len(t.margin_loop_indices) == 111
    assert 24.0 < t.margin_perimeter_mm < 29.0
    # 纯冠形：没有根
    assert t.root_vertex_mask.sum() == 0
    assert t.crown_vertex_mask.sum() + len(t.margin_loop_indices) \
        == len(t.vertices)
    assert 6.0 < t.crown_height_mm < 10.0


@needs_real
def test_real_template_library_file_dedup_counts():
    """224 个 .eoff 只有 144 份唯一内容：teeth 与 pontics 之间 80 对完全相同。

    模板清点必须按内容哈希去重，不能把 pontics 的重复文件当独立模板。
    """
    import hashlib
    root = REAL_ROOT
    seen = set()
    total = 0
    for dp, _dn, fns in os.walk(root):
        for f in fns:
            if not f.endswith(".eoff"):
                continue
            total += 1
            with open(os.path.join(dp, f), "rb") as fh:
                seen.add(hashlib.sha256(fh.read()).hexdigest())
    assert total == 224 and len(seen) == 144


@needs_real
def test_real_template_fdi46_normalizes_cleanly():
    t = load_crown_template(_chosen(), 46)
    s = t.summary()
    # 颈缘：146 点、周长与"牙颈线位 0x800"一致
    assert s["n_margin_loop"] == 146
    assert t.margin_loop_bit == (0x800,)
    assert 30.0 < s["margin_perimeter_mm"] < 38.0
    # 切分必须干净：2 个分量、无碎片
    assert s["n_components_after_seam"] == 2
    assert s["n_other_components"] == 0
    assert s["n_crown_vertices"] + s["n_root_vertices"] + s["n_margin_loop"] \
        == s["n_vertices"]
    # 磨牙冠的合理量级
    assert 5.0 < s["crown_height_mm"] < 10.0
    assert 9.0 < s["crown_span_xy_mm"][0] < 13.0
    assert 9.0 < s["crown_span_xy_mm"][1] < 13.0
    # 咬合朝向应接近 +Z
    assert t.occlusal_direction[2] > 0.9


@needs_real
def test_real_template_origin_is_at_margin_centroid():
    t = load_crown_template(_chosen(), 46)
    assert np.linalg.norm(t.margin_centroid) < 1e-9
    # 冠部整体在原点上方（+Z 朝咬合面）
    assert t.vertices[t.crown_vertex_mask, 2].min() > -2.0
    assert t.crown_height_mm > 5.0


@needs_real
def test_real_template_split_is_orientation_sensitive():
    """把模板绕 X 轴翻转 180° 后，齿尖/齿冠的上下关系必须被检测为异常。

    这条守住"走向判据真的在起作用"，而不是永远返回 True。
    """
    from app.geometry.crown_outer.template import split_crown_and_root as sp
    t = load_crown_template(_chosen(), 46)
    v = t.vertices.copy()
    v[:, 2] *= -1.0                        # 翻转：根尖朝上
    v[:, 1] *= -1.0                        # 保持右手系（Z 翻转 + Y 翻转）
    crown, root, _ = sp(v, t.faces, t.margin_loop_indices)
    # 翻转后"含最高点的分量"变成原来的根部
    assert root.sum() == (t.crown_vertex_mask).sum() or \
        crown.sum() == t.root_vertex_mask.sum()
    assert crown.sum() != t.crown_vertex_mask.sum()
