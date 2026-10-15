"""资产层：纯 stdlib 的 STL 探测、内容哈希、脱敏命名、sectioned/unsectioned 判定。

**为什么是纯 stdlib**：566 GB 数据在服务器上（`/data1/welo/dentistry`），而那台机器
没有 numpy/trimesh。索引器必须能就地跑，否则"把文件堆变成索引"就要先把数据搬走。
所以这里不 import numpy / trimesh —— 有测试断言这一点。

**为什么不用文件名判断几何**：实测同一目录里叫 `-modelbase` 的文件
有的是 7 mm 单牙、有的是 58 mm 弓段，命名不可靠。所有分类都由几何量（包围盒、
顶点密度、表面距离）决定，文件名只作为**交叉验证的旁证**。
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from dataclasses import dataclass, field

# ---------------------------------------------------------------- 尺度阈值

#: 单牙（代型/冠）的最大边长上限。实测单牙 7~17 mm。
SINGLE_TOOTH_MAX_MM = 25.0
#: 弓段 / 整颌尺度下限。实测弓段 50~65 mm，整颌 61~87 mm。
ARCH_MIN_MM = 40.0
#: 目标牙中心到弓表面的距离阈值：小于它视为"该处有实体表面"。
CENTER_SURFACE_NEAR_MM = 3.5
#: 判"sectioned"（代型被抽走、中心处是空洞）的距离下限。
CENTER_SURFACE_VOID_MM = 6.0

#: 提取文件名里**全部**两位 FDI（11~48）。
#:
#: 这是本模块要修的那个 bug 的核心：旧实现只取**最后一个**两位数字，
#: 于是 `...-12-11-21-22-bridge_slm_cad.stl` 被当成"只有 FDI 22"，
#: 于是一座 4 连桥会被误判成**单冠**。桥件必须能报出多个牙位。
_FDI_RE = re.compile(r"(?<![0-9])([1-4][1-8])(?![0-9])")


def fdis_in_name(name: str) -> list[int]:
    """从文件名里抽出**所有** FDI，去重并保持出现顺序。

    ``CASE-12-11-21-22-bridge_slm_cad.stl`` -> ``[12, 11, 21, 22]``
    """
    out: list[int] = []
    for m in _FDI_RE.finditer(os.path.basename(name)):
        v = int(m.group(1))
        if v not in out:
            out.append(v)
    return out


# ---------------------------------------------------------------- STL 读取

@dataclass
class StlProbe:
    """一个 STL 的几何摘要。"""

    path: str
    bytes: int
    faces: int
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    #: 头部声明的面数与文件大小不一致（截断/异常写入）。
    size_mismatch: bool = False
    ascii_stl: bool = False
    ok: bool = True
    error: str | None = None

    @property
    def extents(self) -> tuple[float, float, float]:
        return tuple(self.hi[i] - self.lo[i] for i in range(3))

    @property
    def max_extent_mm(self) -> float:
        return max(self.extents) if self.ok else 0.0

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((self.hi[i] + self.lo[i]) / 2.0 for i in range(3))

    @property
    def bbox_diag_mm(self) -> float:
        return math.dist(self.lo, self.hi) if self.ok else 0.0

    @property
    def scale_class(self) -> str:
        m = self.max_extent_mm
        if m <= 0:
            return "unknown"
        if m <= SINGLE_TOOTH_MAX_MM:
            return "single_tooth"
        if m < ARCH_MIN_MM:
            return "multi_tooth"
        return "arch"


def probe_stl(path: str, *, read_vertices: bool = False) -> StlProbe:
    """读 STL 的头部与（可选）顶点，返回几何摘要。

    ``read_vertices=False`` 时仍会扫描全部面记录来求包围盒 —— 二进制 STL 没有
    独立的顶点表，包围盒只能从面记录里扫出来。代价是读一遍文件。
    """
    def _fail(msg: str, size: int = 0) -> StlProbe:
        return StlProbe(path=path, bytes=size, faces=0, lo=(0.0,) * 3,
                        hi=(0.0,) * 3, ok=False, error=msg)

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return _fail(f"stat 失败: {exc}")
    if size < 84:
        return _fail("文件小于 84 字节，不是二进制 STL", size)

    try:
        with open(path, "rb") as f:
            head = f.read(84)
            if head[:5] == b"solid":
                chunk = head + f.read(2048 - 84)
                printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in chunk)
                if printable / max(len(chunk), 1) > 0.95:
                    return StlProbe(path=path, bytes=size, faces=0, lo=(0.0,) * 3,
                                    hi=(0.0,) * 3, ascii_stl=True,
                                    ok=False, error="ASCII STL，未支持")
            n = struct.unpack("<I", head[80:84])[0]
            mismatch = 84 + n * 50 != size
            if mismatch:
                n = max(0, (size - 84) // 50)
            f.seek(84)
            buf = f.read(n * 50)
    except OSError as exc:
        return _fail(f"读取失败: {exc}", size)

    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for i in range(n):
        try:
            v = struct.unpack_from("<12f", buf, i * 50)
        except struct.error:
            break
        for j in (3, 6, 9):
            for k in range(3):
                x = v[j + k]
                if x < lo[k]:
                    lo[k] = x
                if x > hi[k]:
                    hi[k] = x
    if not (math.isfinite(lo[0]) and math.isfinite(hi[0])):
        return _fail("没有可用的顶点坐标", size)
    return StlProbe(path=path, bytes=size, faces=n,
                    lo=(lo[0], lo[1], lo[2]), hi=(hi[0], hi[1], hi[2]),
                    size_mismatch=mismatch)


@dataclass
class VertexGrid:
    """顶点空间网格 —— 支撑"点到表面距离"与"中心处有没有表面"两个查询。

    纯 Python 里逐点暴力查 25 万个顶点太慢，格子把每次查询压到常数级。
    """

    cell: float
    lo: tuple[float, float, float]
    buckets: dict[tuple[int, int, int], list[tuple[float, float, float]]]
    n_points: int

    def _key(self, p) -> tuple[int, int, int]:
        return (int(math.floor((p[0] - self.lo[0]) / self.cell)),
                int(math.floor((p[1] - self.lo[1]) / self.cell)),
                int(math.floor((p[2] - self.lo[2]) / self.cell)))

    def nearest_distance(self, p, *, search_radius_mm: float = 15.0,
                         ) -> tuple:
        """到最近顶点的距离 -> ``(距离, 是否真的找到)``。

        **在 ``search_radius_mm`` 内找不到时返回该半径本身、并把 found 置 False**
        —— 这是一个**下界**（真实距离 >= 该值）。对"中心附近有没有表面"这类判据
        足够用，而且避免为了量一个 30 mm 外的距离去扫半径 30 的球
        （那是 O(r^3)，会把索引从几分钟跑成几小时）。

        调用方**必须**检查 found：把饱和值当精确距离用是错的。
        """
        cx, cy, cz = self._key(p)
        max_rings = max(1, int(math.ceil(search_radius_mm / self.cell)))
        best = math.inf
        for r in range(max_rings + 1):
            found = False
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != r:
                            continue
                        for q in self.buckets.get((cx + dx, cy + dy, cz + dz), ()):
                            found = True
                            d = math.dist(p, q)
                            if d < best:
                                best = d
            if found and best <= (r + 1) * self.cell:
                break
        if not math.isfinite(best):
            return float(search_radius_mm), False
        return best, True

    def coverage(self, p, radius: float, *, n_dirs: int = 26) -> float:
        """``p`` 周围有多少比例的方向上能在 ``radius`` 内碰到表面。

        一个**空洞**（代型被抽走的 sectioned base）会在朝向洞口的若干个方向上
        找不到任何顶点，于是覆盖率明显偏低；预备体还在时各方向都有面。
        这种"四周有没有面"的统计比单看最近距离更稳 —— 单看最近距离时，
        中心落在洞边亦可能很近。
        """
        dirs: list[tuple[float, float, float]] = []
        for sx in (-1.0, 0.0, 1.0):
            for sy in (-1.0, 0.0, 1.0):
                for sz in (-1.0, 0.0, 1.0):
                    if sx == sy == sz == 0.0:
                        continue
                    norm = math.sqrt(sx * sx + sy * sy + sz * sz)
                    dirs.append((sx / norm, sy / norm, sz / norm))
        if n_dirs < len(dirs):
            step = len(dirs) / n_dirs
            dirs = [dirs[int(i * step)] for i in range(n_dirs)]
        hit = 0
        for d in dirs:
            for t in range(1, 5):
                q = (p[0] + d[0] * radius * t / 4.0,
                     p[1] + d[1] * radius * t / 4.0,
                     p[2] + d[2] * radius * t / 4.0)
                if self.nearest_distance(
                        q, search_radius_mm=self.cell * 2.0)[0] <= self.cell * 2.0:
                    hit += 1
                    break
        return hit / len(dirs)


def read_vertex_grid(path: str, *, cell: float = 1.0,
                     max_points: int = 120_000) -> VertexGrid | None:
    """读 STL 顶点并建格。点数超限时按步长抽样（只用于距离统计，不影响量级）。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    if size < 84:
        return None
    with open(path, "rb") as f:
        head = f.read(84)
        if head[:5] == b"solid":
            return None
        n = struct.unpack("<I", head[80:84])[0]
        if 84 + n * 50 != size:
            n = max(0, (size - 84) // 50)
        f.seek(84)
        buf = f.read(n * 50)

    stride = max(1, (n * 3) // max_points) if n * 3 > max_points else 1
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    pts: list[tuple[float, float, float]] = []
    counter = 0
    for i in range(n):
        try:
            v = struct.unpack_from("<12f", buf, i * 50)
        except struct.error:
            break
        for j in (3, 6, 9):
            counter += 1
            if counter % stride:
                continue
            p = (v[j], v[j + 1], v[j + 2])
            if not all(math.isfinite(x) for x in p):
                continue
            pts.append(p)
            for k in range(3):
                if p[k] < lo[k]:
                    lo[k] = p[k]
                if p[k] > hi[k]:
                    hi[k] = p[k]
    if not pts:
        return None
    buckets: dict[tuple[int, int, int], list] = {}
    lo_t = (lo[0], lo[1], lo[2])
    for p in pts:
        key = (int(math.floor((p[0] - lo_t[0]) / cell)),
               int(math.floor((p[1] - lo_t[1]) / cell)),
               int(math.floor((p[2] - lo_t[2]) / cell)))
        buckets.setdefault(key, []).append(p)
    return VertexGrid(cell=cell, lo=lo_t, buckets=buckets, n_points=len(pts))


# ---------------------------------------------------------------- 拓扑检查

def watertight_and_euler(path: str) -> tuple[bool | None, int | None]:
    """``(水密?, 欧拉数)``。

    水密判据：每条无向边**恰好**被 2 个面使用。欧拉数 ``V - E + F``
    对单颗冠应为 2（亏格 0）；远大于 2 或为负说明网格是碎的或带多个把手。
    纯 Python 建边表，5 万面的冠约 0.5 s。
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(84)
            if size < 84 or head[:5] == b"solid":
                return None, None
            n = struct.unpack("<I", head[80:84])[0]
            if 84 + n * 50 != size:
                n = max(0, (size - 84) // 50)
            buf = f.read(n * 50)
    except OSError:
        return None, None

    edges: dict[tuple[int, int], int] = {}
    verts: set[tuple[float, float, float]] = set()
    for i in range(n):
        try:
            v = struct.unpack_from("<12f", buf, i * 50)
        except struct.error:
            break
        tri = []
        for j in (3, 6, 9):
            p = (round(v[j], 3), round(v[j + 1], 3), round(v[j + 2], 3))
            verts.add(p)
            tri.append(p)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (tri[a], tri[b]) if tri[a] <= tri[b] else (tri[b], tri[a])
            edges[key] = edges.get(key, 0) + 1
    if not edges:
        return None, None
    watertight = all(c == 2 for c in edges.values())
    euler = len(verts) - len(edges) + n
    return watertight, euler


def content_hash(path: str, *, chunk: int = 1 << 20) -> str | None:
    """文件内容 SHA-256（前 16 位十六进制）。

    索引里只存它，用来追溯回原始资产 —— 这样既保留可追溯性，
    又不需要把含患者姓名的原文件名写进 CSV。
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- 脱敏

#: 允许出现在索引里的 ASCII 形态：数字/字母/点/下划线/横线。
_SAFE_TOKEN_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]*$")


def safe_case_id(raw: str) -> str:
    """把目录名压成可安全写入索引的 id。

    **生产数据的目录名本身可能是干净的**（实测都是数字订单号，如 ``82403180486``），
    但文件名里确实出现过患者姓名。这里采取的策略是：只要是纯 ASCII 的
    数字/字母/``._-`` 组合就原样保留（可追溯），**一旦含非 ASCII 或其它字符，
    就换成内容哈希**，绝不把中文姓名写进索引。
    """
    if _SAFE_TOKEN_RE.match(raw or ""):
        return raw
    return "case_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def safe_asset_path(case_id: str, role: str, index: int, ext: str) -> str:
    """资产在索引里的**合成**相对路径。

    刻意**不保留原文件名** —— 原文件名可能含患者姓名（实测
    ``82403180486-160323李挨林 B组 (1).stl``）。角色 + 序号 + 内容哈希
    足以在后续训练集里定位文件，又不会把姓名重新泄露进 CSV。
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    return f"{case_id}/{role}__{index:02d}{ext.lower()}"


def safe_text_token(raw: str | None, *, prefix: str) -> str | None:
    """把自由文本压成可 ASCII 写出的 token。

    纯 ASCII（数字/字母/``._-`` 组合）原样保留，可读且可追溯；
    含非 ASCII 就换成 ``<prefix>_<sha1[:10]>``。这样"同一个值"仍然映射到
    "同一个 token"，分组/统计不受影响，而中文不会被写进索引。

    为什么需要它：实测 ``<ToothLibraryName>`` 的值是 ``超神4``
    （厂商/自定义库名）。它既不是 PHI，也不该让整个索引中止 ——
    该做的是脱敏。
    """
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if _SAFE_TOKEN_RE.match(text):
        return text
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"


def assert_phi_free(rows, *, allow_keys=("source_case_id",)) -> None:
    """写文件前的最后一道闸：索引里不得出现非 ASCII 字符。

    非 ASCII 在这个语料里就意味着中文姓名 / 门诊名 / 备注。
    允许 ``allow_keys`` 之外的字段只含可打印 ASCII。
    """
    bad: list[str] = []
    for i, row in enumerate(rows):
        for k, v in row.items():
            if not isinstance(v, str) or not v:
                continue
            try:
                v.encode("ascii")
            except UnicodeEncodeError:
                bad.append(f"第 {i} 行字段 {k} 含非 ASCII: {v[:40]!r}")
        if len(bad) > 5:
            break
    if bad:
        raise ValueError("索引里出现非 ASCII（很可能是 PHI），拒绝写出：\n  "
                         + "\n  ".join(bad))
