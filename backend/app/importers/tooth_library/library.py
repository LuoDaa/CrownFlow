"""专业牙库（``牙齿数据库``）的发现、FDI→模板映射与牙位局部系（纯 stdlib）。

## 目录约定

```
<root>/
  teeth/<库名>/{lowerjaw,upperjaw}/<n>.eoff      n = 1..8（牙位序号）
  teeth/<库名>/{lowerjaw,upperjaw}/<n>-reduced-*.eoff   内冠/上瓷等变体
  metadata/<库名>/{lowerjaw,upperjaw}.xml        ToothBow：FDI 级的 Position/Axis
  metadata/<库名>/preview*.png                   仅供人工预览
  pontics/<库名>/{lowerjaw,upperjaw}/            桥体
```

## XML 给什么

``<ModelTooth><Number>`` 就是 **FDI 编号**（不是序号），每颌 16 个；每个牙位给出
``Position`` 与 ``AxisOcclusal`` / ``AxisMesial``（颌坐标系下的牙位局部系）。
``BuccoocclusalCusp*`` / ``ContactPoint*`` / ``BuccocervicalPoint`` /
``LinguocervicalPoint`` 等解剖标志点**在这套牙库里全部是 0**（实测 16/16 全零），
所以**不能**从 XML 拿牙颈线 —— 牙颈线在 ``.eoff`` 的
``VERTEX_FEATURE_LABELS`` 位标志里（见 ``eoff.py``）。

## FDI → 文件

``file index n`` = FDI 的个位数（1..8）；``jaw`` 由象限决定：
象限 3/4 是下颌、1/2 是上颌。左/右的镜像**不靠文件名区分**，而是由 XML 里
每个 FDI 自己的 ``AxisMesial`` 决定 —— 所以映射必须先经颌装配（
``arch_points``）再回到局部系，不能直接拿 ``.eoff`` 原坐标当"左右通用"。
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .eoff import EoffMesh, parse_eoff

#: 牙库根目录的环境变量。**不写死路径**：真实数据在仓库之外，路径随机器变化。
ENV_LIBRARY_ROOT = "CROWNFLOW_TOOTH_LIBRARY_ROOT"

JAW_LOWER = "lowerjaw"
JAW_UPPER = "upperjaw"

#: 修复体变体后缀 —— 这些不是"完整解剖冠"，Crown Outer 要用纯 <n>.eoff。
VARIANT_MARKERS = ("-reduced", "-thimble", "-内冠", "-颊侧上瓷")


class ToothLibraryError(ValueError):
    pass


# ---------------------------------------------------------------- FDI 工具

def jaw_for_fdi(fdi: int) -> str:
    q = fdi // 10
    if q in (1, 2):
        return JAW_UPPER
    if q in (3, 4):
        return JAW_LOWER
    raise ToothLibraryError("非法 FDI %r（象限必须是 1..4）" % (fdi,))


def tooth_index_for_fdi(fdi: int) -> int:
    """FDI → 牙位序号 n（1..8），即 ``<n>.eoff`` 的 n。"""
    n = fdi % 10
    if not 1 <= n <= 8:
        raise ToothLibraryError("非法 FDI %r（个位必须是 1..8）" % (fdi,))
    return n


def is_right_side(fdi: int) -> bool:
    """患者右侧 = 象限 4（下）、1（上）。"""
    return (fdi // 10) in (1, 4)


# ---------------------------------------------------------------- 颌 XML

@dataclass(frozen=True)
class BowEntry:
    """XML 里的一个 ``ModelTooth``：颌坐标系下的牙位位姿。"""

    fdi: int
    position: tuple[float, float, float]
    axis_occlusal: tuple[float, float, float]
    axis_mesial: tuple[float, float, float]

    def frame(self) -> tuple[tuple[float, float, float],
                             tuple[float, float, float],
                             tuple[float, float, float]]:
        """(原点, z=AxisOcclusal, x=AxisMesial, y=z×x) 正交化后的牙位局部系。

        与 ``app/groundtruth/v3_features.tooth_local_frame`` 同一约定
        （Z = 就位/咬合向，X = 近中向，Y = Z×X），便于把患者侧与牙库侧对齐。
        """
        z = _unit(self.axis_occlusal)
        x = _unit(self.axis_mesial)
        x = _orthogonalize(x, z)
        y = _cross(z, x)
        return self.position, z, x, y


def _unit(v):
    n = sum(c * c for c in v) ** 0.5
    if n < 1e-12:
        raise ToothLibraryError("零向量无法归一化")
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _orthogonalize(x, z):
    d = _dot(x, z)
    return _unit(tuple(x[i] - d * z[i] for i in range(3)))


def _text_vec(el) -> tuple[float, float, float] | None:
    if el is None:
        return None
    vals = []
    for k in ("x", "y", "z"):
        c = el.find(k)
        if c is None or c.text is None:
            return None
        vals.append(float(c.text))
    if all(abs(v) < 1e-12 for v in vals):
        return None          # 全零 = 未填充的标志点
    return (vals[0], vals[1], vals[2])


def parse_bow_xml(path: str) -> dict[int, BowEntry]:
    """解析 ``lowerjaw.xml`` / ``upperjaw.xml`` → {FDI: BowEntry}。"""
    root = ET.parse(path).getroot()
    if root.tag != "ToothBow":
        raise ToothLibraryError("根元素是 <%s>，期望 <ToothBow>" % root.tag)
    out: dict[int, BowEntry] = {}
    for mt in root.findall("./ModelTeeth/ModelTooth"):
        num = mt.findtext("Number")
        pos = _text_vec(mt.find("Position"))
        z = _text_vec(mt.find("AxisOcclusal"))
        x = _text_vec(mt.find("AxisMesial"))
        if num is None or pos is None or z is None or x is None:
            raise ToothLibraryError("ModelTooth 缺少 Number/Position/Axis*")
        fdi = int(num)
        out[fdi] = BowEntry(fdi=fdi, position=pos, axis_occlusal=z,
                            axis_mesial=x)
    if not out:
        raise ToothLibraryError("XML 里没有 ModelTooth")
    return out


def landmark_coverage(path: str) -> dict[str, int]:
    """统计解剖标志点里**非零**的个数 —— 用来说明"XML 不提供牙颈线"。"""
    root = ET.parse(path).getroot()
    names = ("BuccoocclusalCusp1", "BuccoocclusalCusp2", "LinguoocclusalCusp1",
             "LinguoocclusalCusp2", "AntagonistCuspContact1",
             "AntagonistCuspContact2", "ContactPoint1", "ContactPoint2",
             "BuccalPoint", "BuccocervicalPoint", "LingualPoint",
             "LinguocervicalPoint")
    out = {n: 0 for n in names}
    for mt in root.findall("./ModelTeeth/ModelTooth"):
        for n in names:
            if _text_vec(mt.find(n)) is not None:
                out[n] += 1
    return out


# ---------------------------------------------------------------- 牙库

@dataclass
class ToothLibrary:
    """一个库（如 ``炼金王``）。"""

    name: str
    root: str

    @property
    def teeth_dir(self) -> str:
        return os.path.join(self.root, "teeth", self.name)

    @property
    def metadata_dir(self) -> str:
        return os.path.join(self.root, "metadata", self.name)

    def bow(self, jaw: str) -> dict[int, BowEntry]:
        p = os.path.join(self.metadata_dir, "%s.xml" % jaw)
        if not os.path.isfile(p):
            raise ToothLibraryError("缺少 %s" % p)
        return parse_bow_xml(p)

    def template_path(self, fdi: int) -> str:
        """完整解剖冠模板文件（``<n>.eoff``，排除 reduced/内冠等变体）。"""
        jaw = jaw_for_fdi(fdi)
        n = tooth_index_for_fdi(fdi)
        p = os.path.join(self.teeth_dir, jaw, "%d.eoff" % n)
        if not os.path.isfile(p):
            raise ToothLibraryError("模板不存在：%s" % p)
        return p

    def variants(self, fdi: int) -> list[str]:
        """该牙位的非完整冠变体（内冠/上瓷/马龙桥），供将来使用。"""
        jaw = jaw_for_fdi(fdi)
        n = tooth_index_for_fdi(fdi)
        d = os.path.join(self.teeth_dir, jaw)
        out = []
        for f in sorted(os.listdir(d)):
            if not f.endswith(".eoff"):
                continue
            if not f.startswith("%d-" % n):
                continue
            if any(m in f for m in VARIANT_MARKERS):
                out.append(os.path.join(d, f))
        return out

    def template_mesh(self, fdi: int, **kw) -> EoffMesh:
        return parse_eoff(self.template_path(fdi), **kw)

    def bow_entry(self, fdi: int) -> BowEntry:
        return self.bow(jaw_for_fdi(fdi))[fdi]

    # ------------------------------------------------------------ 装配
    def arch_points(self, jaw: str, *, labels: bool = False) -> dict:
        """把该颌 16 颗牙按 XML 位姿装配到颌坐标系。

        这一步同时是**映射正确性的验证**：如果 FDI→文件或镜像判错了，
        装配出来的牙列会出现明显重叠或错位（见 ``assembly_report``）。
        """
        bow = self.bow(jaw)
        out = {}
        for fdi, e in sorted(bow.items()):
            mesh = self.template_mesh(fdi, want_labels=labels)
            org, z, x, y = e.frame()
            pts = []
            for v in mesh.vertices:
                pts.append(tuple(org[i] + v[0] * x[i] + v[1] * y[i] + v[2] * z[i]
                                 for i in range(3)))
            out[fdi] = {"entry": e, "mesh": mesh, "points": pts,
                        "basis": (x, y, z), "origin": org}
        return out

    def local_points(self, fdi: int, *, labels: bool = False):
        """模板顶点表达在该 FDI 的**牙位局部系**里（原点 = XML Position）。

        牙位局部系：Z = AxisOcclusal，X = AxisMesial，Y = Z×X。
        左右侧的区别由 XML 的 AxisMesial 承载，所以这里出来的点**已经是
        该 FDI 该侧的形状**，可以直接拿去和患者牙位对齐。
        """
        e = self.bow_entry(fdi)
        mesh = self.template_mesh(fdi, want_labels=labels)
        org, z, x, y = e.frame()
        pts = []
        for v in mesh.vertices:
            d = (v[0] - org[0], v[1] - org[1], v[2] - org[2])
            pts.append((_dot(d, x), _dot(d, y), _dot(d, z)))
        return e, mesh, pts


def discover_libraries(root: str | None = None) -> list[ToothLibrary]:
    """列出根目录下所有库。``root`` 缺省取 ``CROWNFLOW_TOOTH_LIBRARY_ROOT``。"""
    root = root or os.environ.get(ENV_LIBRARY_ROOT)
    if not root:
        raise ToothLibraryError(
            "未指定牙库根目录：传 root 或设 %s" % ENV_LIBRARY_ROOT)
    tdir = os.path.join(root, "teeth")
    if not os.path.isdir(tdir):
        raise ToothLibraryError("没有 teeth/ 子目录：%s" % tdir)
    out = []
    for name in sorted(os.listdir(tdir)):
        d = os.path.join(tdir, name)
        if not os.path.isdir(d):
            continue
        out.append(ToothLibrary(name=name, root=root))
    if not out:
        raise ToothLibraryError("teeth/ 下没有库目录：%s" % tdir)
    return out


def assembly_report(lib: ToothLibrary, jaw: str) -> dict:
    """装配该颌并给出几何自检：相邻牙是否重叠、整体跨度是否合理。"""
    parts = lib.arch_points(jaw)
    fdis = sorted(parts)
    boxes = {}
    for fdi in fdis:
        pts = parts[fdi]["points"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        boxes[fdi] = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
    ext = (min(b[0] for b in boxes.values()), max(b[1] for b in boxes.values()),
           min(b[2] for b in boxes.values()), max(b[3] for b in boxes.values()),
           min(b[4] for b in boxes.values()), max(b[5] for b in boxes.values()))
    # 同象限相邻牙位的 bbox 交集（只看 x 跨度重叠 = 可能穿插）
    overlaps = []
    for i, a in enumerate(fdis):
        for b in fdis[i + 1:]:
            if a // 10 != b // 10:
                continue
            if abs(a - b) != 1:
                continue
            ba, bb = boxes[a], boxes[b]
            if ba[1] > bb[0] and bb[1] > ba[0]:
                overlaps.append((a, b))
    return {"jaw": jaw, "n_teeth": len(fdis), "bbox_all": tuple(round(v, 2) for v in ext),
            "adjacent_x_overlaps": overlaps,
            "per_tooth_bbox": {k: tuple(round(v, 2) for v in v2)
                               for k, v2 in boxes.items()}}
