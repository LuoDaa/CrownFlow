"""``.eoff`` 解析器 —— 专业牙库的网格格式（纯 stdlib）。

## 格式（由字节对账得出，不是猜的）

```
偏移 0        b"OFF BINARY\\n"                        (11 字节)
偏移 11       uint32 V, uint32 F, uint32 E            (12 字节)
偏移 23       V × 12  float32 x,y,z                   顶点
23 + 12V      F × 14  uint16 n(=3) + uint32 i0,i1,i2  三角面
之后          扩展段序列，每段：
                  25 字节名字段（ASCII + \\0 填充）
                  [uint32 字节长度 + 该长度的负载]     ← 只有部分段有负载
              最后一段名为 END_EXTENSION
```

对账依据（三个库、5 个文件一致）：

* 头部读出的 ``E`` 满足 ``E == 3F/2``；
* 面记录里 ``n`` 字段**全部等于 3**（17392/17392），索引**全部**落在 ``[0, V)``；
* 段序列**恰好覆盖到文件末尾**；
* ``VERTEX_FEATURE_LABELS`` 的项数**恰好等于 V**。

## 顶点特征标签（关键资产）

``VERTEX_FEATURE_LABELS`` 是逐顶点位标志。实测其中若干位构成**一维闭环**
（单一连通分量 + 每个顶点度数恰好 2 + 周长 25~43 mm），即**牙颈线/轮廓线**。
这消掉了"Crown Template 的牙颈边界从哪来"这个最大的未知：
**不需要猜 CEJ**（对照：``ml/template_studio.py`` 对另一套无标注牙库只能给
"候选切割面，不是已识别的真实牙颈线"）。

位的确切语义**没有官方文档，而且跨库不一致**（实测，32 牙位全扫）：

| 库 | 取到闭环的牙位 | 牙颈线位 | 另一个环 |
|---|---|---|---|
| 炼金王 | **32 / 32** | ``0x800``（较低较短） | ``0x400``（较高较长） |
| 洋溢-长夜同在 | **32 / 32** | ``0x800`` | ``0x400``；上颌另有 ``0x20000000`` 与 0x800 几乎重合 |
| 超神2 | **14 / 32** | ``0x800``（仅部分牙位） | 无 |

超神2 缺 18 个牙位（含 FDI46/36/11 与整个下颌），**用单一位与 3 位以内的
所有并集都找不到闭环** —— 它的标注本身不完整。

所以本模块**不写死位号**：``closed_loop_bits`` / ``find_margin_loop`` 用几何
判据把"环"与"面片/开弧"分开，找不到就返回 ``None``，绝不硬凑。
需要牙颈线的下游（CrownOuter）必须先 ``find_margin_loop`` 并接受可能为 None。
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

#: 段名字段长度（实测 25：三个段名 + 补零都恰好 25 字节）。
NAME_FIELD = 25

#: 头部长度：magic(11) + V/F/E(12)。
HEADER_LEN = 23

#: 单条面记录长度：uint16 顶点数 + 3 × uint32 索引。
FACE_RECORD = 14

#: 三角面的顶点数字段必须取这个值（实测全部如此）。
TRIANGLE_N = 3

MAGIC = b"OFF BINARY"

#: 段名。
SECTION_VERTEX_FEATURE_LABELS = "VERTEX_FEATURE_LABELS"
SECTION_END = "END_EXTENSION"


class EoffError(ValueError):
    """格式不满足本模块已对账过的布局 —— 明确报错，不做猜测性回退。"""


@dataclass
class EoffSection:
    name: str
    payload: bytes | None
    offset: int

    @property
    def n_uint32(self) -> int:
        return 0 if self.payload is None else len(self.payload) // 4


@dataclass
class EoffMesh:
    """一个 ``.eoff`` 牙模网格。索引是 ``0..V-1`` 的顶点下标。"""

    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]
    labels: list[int] | None
    sections: list[EoffSection] = field(default_factory=list)
    declared_vertex_count: int = 0
    declared_face_count: int = 0
    declared_edge_count: int = 0
    content_hash: str = ""

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    # ------------------------------------------------------------ 量测
    def bbox(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))

    def centroid(self) -> tuple[float, float, float]:
        n = self.n_vertices or 1
        return (sum(v[0] for v in self.vertices) / n,
                sum(v[1] for v in self.vertices) / n,
                sum(v[2] for v in self.vertices) / n)

    def vertex_adjacency(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {}
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
        return adj

    def boundary_edges(self) -> list[tuple[int, int]]:
        """只出现一次的边。牙模在牙颈线处可以是不闭合的。"""
        from collections import Counter
        cnt: Counter = Counter()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                cnt[(min(u, v), max(u, v))] += 1
        return [e for e, n in cnt.items() if n == 1]

    def connected_vertex_components(self, subset) -> list[list[int]]:
        """``subset``（顶点下标集合）在网格邻接下分成几个连通分量。"""
        s = set(subset)
        adj = self.vertex_adjacency()
        seen: set[int] = set()
        out: list[list[int]] = []
        for v in sorted(s):
            if v in seen:
                continue
            comp, stack = [], [v]
            seen.add(v)
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj.get(u, ()):
                    if w in s and w not in seen:
                        seen.add(w)
                        stack.append(w)
            out.append(sorted(comp))
        return out

    def subset_loop_perimeter(self, subset) -> float:
        """``subset`` 内部边的总长（成环时即周长）。"""
        s = set(subset)
        adj = self.vertex_adjacency()
        per = 0.0
        for v in s:
            for w in adj.get(v, ()):
                if w in s and v < w:
                    per += math.dist(self.vertices[v], self.vertices[w])
        return per


# ================================================================ 解析

def _parse_sections(raw: bytes, start: int) -> tuple[list[EoffSection], int]:
    """顺序解析扩展段。返回 (段列表, 结束偏移)。"""
    out: list[EoffSection] = []
    p = start
    while p < len(raw):
        if p + NAME_FIELD > len(raw):
            raise EoffError("段名字段越界（文件被截断？）偏移 %d" % p)
        blob = raw[p:p + NAME_FIELD]
        nul = blob.find(b"\x00")
        name = blob[:nul if nul >= 0 else NAME_FIELD].decode("ascii", "replace")
        name_at = p
        p += NAME_FIELD
        payload = None
        # 有负载的段：紧跟 uint32 长度。用可自洽性区分（标记段没有长度字段，
        # 其后的字节是下一段的名字，必然大得离谱）。
        if p + 4 <= len(raw):
            n = struct.unpack_from("<I", raw, p)[0]
            if 0 < n <= len(raw) - p - 4 and n % 4 == 0:
                payload = raw[p + 4:p + 4 + n]
                p += 4 + n
        out.append(EoffSection(name=name, payload=payload, offset=name_at))
        if name == SECTION_END:
            break
    return out, p


def parse_eoff_bytes(raw: bytes, *, want_labels: bool = True) -> EoffMesh:
    if not raw.startswith(MAGIC):
        raise EoffError("不是 OFF BINARY（前 16 字节 %r）" % raw[:16])
    if len(raw) < HEADER_LEN:
        raise EoffError("文件过短 %d 字节" % len(raw))
    V, F, E = struct.unpack_from("<3I", raw, 11)
    off_f = HEADER_LEN + V * 12
    end_f = off_f + F * FACE_RECORD
    if end_f > len(raw):
        raise EoffError(
            "面块越界：需要 %d 字节，实际只有 %d（V=%d F=%d）"
            % (end_f, len(raw), V, F))

    vertices = [struct.unpack_from("<3f", raw, HEADER_LEN + i * 12)
                for i in range(V)]
    faces: list[tuple[int, int, int]] = []
    for i in range(F):
        r = off_f + i * FACE_RECORD
        n = struct.unpack_from("<H", raw, r)[0]
        if n != TRIANGLE_N:
            raise EoffError("第 %d 条面记录的顶点数是 %d，期望 3" % (i, n))
        a, b, c = struct.unpack_from("<3I", raw, r + 2)
        if not (0 <= a < V and 0 <= b < V and 0 <= c < V):
            raise EoffError("第 %d 条面索引越界：(%d,%d,%d) V=%d"
                            % (i, a, b, c, V))
        faces.append((a, b, c))

    sections, covered = _parse_sections(raw, end_f)
    labels = None
    if want_labels:
        for s in sections:
            if s.name == SECTION_VERTEX_FEATURE_LABELS and s.payload is not None:
                labels = list(struct.unpack("<%dI" % s.n_uint32, s.payload))
                if len(labels) != V:
                    raise EoffError(
                        "VERTEX_FEATURE_LABELS 有 %d 项，与 V=%d 不符"
                        % (len(labels), V))
    return EoffMesh(
        vertices=vertices, faces=faces, labels=labels, sections=sections,
        declared_vertex_count=V, declared_face_count=F, declared_edge_count=E,
        content_hash=hashlib.sha256(raw).hexdigest()[:16])


def parse_eoff(path: str, *, want_labels: bool = True) -> EoffMesh:
    with open(path, "rb") as fh:
        return parse_eoff_bytes(fh.read(), want_labels=want_labels)


def sections_cover_file(path: str) -> bool:
    """扩展段是否恰好覆盖到文件末尾 —— 格式对账的独立检查。"""
    raw = open(path, "rb").read()
    V, F, _E = struct.unpack_from("<3I", raw, 11)
    _secs, covered = _parse_sections(raw, HEADER_LEN + V * 12 + F * FACE_RECORD)
    return covered == len(raw)


# ================================================================ 标签位

def label_bits_present(mesh: EoffMesh) -> list[int]:
    """出现过的位（每个元素是单 bit 掩码），按低位到高位。"""
    if not mesh.labels:
        return []
    acc = 0
    for lab in mesh.labels:
        acc |= lab
    return [1 << b for b in range(32) if acc >> b & 1]


def bit_members(mesh: EoffMesh, bit: int) -> list[int]:
    """**该位被置位**的顶点（含与其它位同时置位者）。"""
    return [i for i, lab in enumerate(mesh.labels or []) if lab & bit]


def exclusive_bit_members(mesh: EoffMesh, bit: int) -> list[int]:
    """**只**该位置位的顶点（排除组合标签）。"""
    return [i for i, lab in enumerate(mesh.labels or []) if lab == bit]


@dataclass
class LoopCandidate:
    """一个"位 → 一维闭环"的候选。"""

    bit: int
    n_vertices: int
    n_components: int
    min_degree: int
    max_degree: int
    perimeter_mm: float
    z_median: float

    @property
    def is_clean_loop(self) -> bool:
        """一维闭环：单一连通分量 + **每个顶点度数恰好 2**。

        ``min_degree == 2`` 不能省：一条**开弧**（环缺一段）的端点度数为 1，
        而内部顶点度数为 2，只看 ``max_degree == 2`` 会把断环误判成闭环。
        牙颈线是闭环，边界条件用错会导致形变拉断。
        """
        return (self.n_components == 1 and self.min_degree == 2
                and self.max_degree == 2)


def closed_loop_bits(mesh: EoffMesh, *, exclusive: bool = False,
                     axis: int = 2) -> list[LoopCandidate]:
    """找出**构成一维闭环**的标签位。

    牙颈线在网格上必须是"一圈"：单一连通分量 + 每个顶点在子集内的邻接度数
    恰好 2。这条判据是纯几何的，不依赖对位含义的任何假设 —— 所以即使位的
    语义无文档、**且跨库不一致**，也能可靠地把"环"与"面片/开弧"分开。

    ``axis`` 只用于报告 ``z_median``（默认 2 = z）。
    """
    if not mesh.labels:
        return []
    adj = mesh.vertex_adjacency()
    out: list[LoopCandidate] = []
    for bit in label_bits_present(mesh):
        sel = exclusive_bit_members(mesh, bit) if exclusive \
            else bit_members(mesh, bit)
        if len(sel) < 3:
            continue
        comps = mesh.connected_vertex_components(sel)
        s = set(sel)
        degs = [len(adj.get(v, set()) & s) for v in s]
        zs = sorted(mesh.vertices[v][axis] for v in sel)
        out.append(LoopCandidate(
            bit=bit, n_vertices=len(sel), n_components=len(comps),
            min_degree=min(degs), max_degree=max(degs),
            perimeter_mm=round(mesh.subset_loop_perimeter(sel), 3),
            z_median=round(zs[len(zs) // 2], 3)))
    out.sort(key=lambda c: (not c.is_clean_loop, -c.n_vertices))
    return out


#: 只有"稀疏"的位才可能是牙颈线（一维环 vs 二维面片）。
#: 实测牙颈线 90~190 个顶点，冠部面片 1000+ —— 用 500 作上界很宽松。
LOOP_BIT_MAX_VERTICES = 500


def boundary_loop(mesh: EoffMesh) -> list[int] | None:
    """开放网格的底边界，按**有序环**返回；不是"单一闭环"时返回 ``None``。

    实测（超神2 库）：teeth 下的网格是**未封底的冠形**，底边界就是一条
    干净的闭环（FDI46：111 个边界顶点全部度数 2），它本身就是牙颈边界 ——
    不需要查标签位。而 pontics 是封底的（无边界）。

    判据是纯拓扑的：边界边构成的子图里每个顶点度数恰好 2，且从任一点
    能走回起点并覆盖全部边界顶点（排除"多个独立环"的情形）。
    """
    edges = mesh.boundary_edges()
    if not edges:
        return None
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(ns) != 2 for ns in adj.values()):
        return None                      # 开弧或分叉，不是干净闭环
    start = min(adj)
    loop = [start]
    prev, cur = None, start
    while True:
        nxt = [w for w in adj[cur] if w != prev]
        if not nxt:
            return None
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    if len(loop) != len(adj):
        return None                      # 多个独立环
    return loop


def find_margin_loop(mesh: EoffMesh, *,
                     prefer_lowest: bool = True) -> tuple[LoopCandidate,
                                                           frozenset[int]] | None:
    """求牙颈线闭环，返回 ``(候选, 参与该环的位集)``；找不到返回 ``None``。

    **不猜位号。** 位语义无官方文档，且实测**跨库不一致**：

    * 炼金王：牙颈线 = ``0x800``（较低较短），轮廓线 = ``0x400``（较高较长）；
    * 洋溢-长夜同在：同样 ``0x800``/``0x400``，上颌另有近乎重复的
      ``0x20000000``（25.3mm，与 0x800 几乎重合）；
    * 超神2：只有部分牙位有闭环（FDI16 的 ``0x800``），
      **FDI46 / FDI11 用单一位与 3 位以内的所有并集都找不到闭环** ——
      它的标注不完整。所以本函数允许返回 ``None``，绝不硬凑。

    搜索顺序（都用几何判据验收，不满足就是没有）：
    单一位 → 单一位的"独占"含义 → 稀疏位两两并集。
    """
    if not mesh.labels:
        return None
    seen: dict[frozenset[int], LoopCandidate] = {}

    def consider(bits: frozenset[int]) -> LoopCandidate | None:
        sel: set[int] = set()
        for b in bits:
            sel |= set(bit_members(mesh, b))
        if len(sel) < 3:
            return None
        s = set(sel)
        if len(mesh.connected_vertex_components(s)) != 1:
            return None
        adj = mesh.vertex_adjacency()
        degs = [len(adj.get(v, set()) & s) for v in s]
        if min(degs) != 2 or max(degs) != 2:
            return None
        zs = sorted(mesh.vertices[v][2] for v in s)
        return LoopCandidate(
            bit=(min(bits) if len(bits) == 1 else 0),
            n_vertices=len(sel), n_components=1, min_degree=2, max_degree=2,
            perimeter_mm=round(mesh.subset_loop_perimeter(s), 3),
            z_median=round(zs[len(zs) // 2], 3))

    small = [b for b in label_bits_present(mesh)
             if 3 <= len(bit_members(mesh, b)) <= LOOP_BIT_MAX_VERTICES]
    for b in small:
        c = consider(frozenset({b}))
        if c:
            seen[frozenset({b})] = c
    for b in small:
        sel = exclusive_bit_members(mesh, b)
        if len(sel) < 3:
            continue
        s = set(sel)
        if len(mesh.connected_vertex_components(s)) != 1:
            continue
        adj = mesh.vertex_adjacency()
        degs = [len(adj.get(v, set()) & s) for v in s]
        if min(degs) == 2 and max(degs) == 2:
            zs = sorted(mesh.vertices[v][2] for v in s)
            seen[frozenset({b})] = LoopCandidate(
                bit=b, n_vertices=len(s), n_components=1, min_degree=2,
                max_degree=2,
                perimeter_mm=round(mesh.subset_loop_perimeter(s), 3),
                z_median=round(zs[len(zs) // 2], 3))
    for i, a in enumerate(small):
        for b in small[i + 1:]:
            key = frozenset({a, b})
            if key in seen:
                continue
            c = consider(key)
            if c:
                seen[key] = c

    if not seen:
        return None
    items = sorted(seen.items(),
                   key=lambda kv: (kv[1].z_median if prefer_lowest
                                   else -kv[1].z_median))
    bits, cand = items[0]
    return cand, bits
