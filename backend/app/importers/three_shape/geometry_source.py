"""扫描几何的来源与 companion mesh 绑定。

## 为什么需要这个

真实 3Shape 包里的扫描是 HPS/CE、顶点加密，拿不到 mesh
（见 ``hps_scan.py`` 的实测记录）。所以必须有第二条路：
**操作员从 3Shape 导出 STL，放进同一个包里**。

但导入器**不能"看到 STL 就用"** —— 必须明确 STL 与原扫描角色的绑定关系。
禁止任何位置式规则（"包里第一个 STL = Prep"），那种规则以后一定出事故。

## 绑定优先级（三档，全部只认显式依据）

    1. ORDER_REFERENCE        订单 XML 的扫描条目直接引用了该 mesh 文件
    2. SIBLING_SAME_STEM      与扫描同目录、同主文件名，仅扩展名不同
                              （如 Scans/Lower/PreparationScan.dcm 与 .stl）
    3. 无绑定                    宁可没有几何，也不猜

第 2 档正是用户认可的命名约定 ``Scans/<Jaw>/<Role>Scan.stl``。
两档都不满足时**返回未绑定**并把候选列进诊断，绝不回退到顺序。

## 双 provenance

同时存在 ``PreparationScan.dcm``（CE 加密）与 ``PreparationScan.stl``（可解码）时，
结果必须两者都留：

    original_geometry_source  = HPS_CE_ENCRYPTED
    preferred_geometry_source = COMPANION_STL
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: 可作为 companion 的网格扩展名，按优先顺序（.stl 优先于 .ply）。
COMPANION_SUFFIXES: tuple[str, ...] = (".stl", ".ply", ".obj")


class GeometrySource(str, Enum):
    """扫描几何的实际来源。"""

    HPS_DECODED = "HPS_DECODED"
    HPS_CE_ENCRYPTED = "HPS_CE_ENCRYPTED"
    HPS_UNSUPPORTED = "HPS_UNSUPPORTED"
    COMPANION_STL = "COMPANION_STL"
    COMPANION_PLY = "COMPANION_PLY"
    COMPANION_OBJ = "COMPANION_OBJ"
    NONE = "NONE"

    @property
    def is_usable(self) -> bool:
        return self in (
            GeometrySource.HPS_DECODED,
            GeometrySource.COMPANION_STL,
            GeometrySource.COMPANION_PLY,
            GeometrySource.COMPANION_OBJ,
        )

    @classmethod
    def for_companion(cls, suffix: str) -> "GeometrySource":
        return {
            ".stl": cls.COMPANION_STL,
            ".ply": cls.COMPANION_PLY,
            ".obj": cls.COMPANION_OBJ,
        }.get(suffix.lower(), cls.NONE)


class BindingEvidence(str, Enum):
    """companion 与扫描的绑定依据。"""

    ORDER_REFERENCE = "ORDER_REFERENCE"
    SIBLING_SAME_STEM = "SIBLING_SAME_STEM"
    NONE = "NONE"


@dataclass
class CompanionBinding:
    binding: BindingEvidence
    scan_path: str | None = None
    mesh_path: str | None = None
    mesh_suffix: str | None = None
    #: 同主名候选（可能 .stl 与 .ply 并存）；记录被选中的与未选中的。
    alternatives: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def bound(self) -> bool:
        return self.binding is not BindingEvidence.NONE and self.mesh_path is not None


def _mesh_candidates(resolved_paths: list[str], scan_path: str) -> list[tuple[str, str]]:
    """与 scan_path 同目录、同主文件名的 mesh 候选，按扩展名优先级排序。

    返回 [(路径, 扩展名)]，.stl 优先于 .ply。
    """
    scan = Path(scan_path)
    out: list[tuple[int, str]] = []
    for cand in resolved_paths:
        cp = Path(cand)
        if cp.suffix.lower() not in COMPANION_SUFFIXES:
            continue
        if cp.parent != scan.parent or cp.stem != scan.stem:
            continue
        out.append((COMPANION_SUFFIXES.index(cp.suffix.lower()), cand))
    out.sort()
    return [(cand, Path(cand).suffix.lower()) for _, cand in out]


def bind_companion(
    scan_path: str,
    resolved_paths: list[str],
    *,
    order_referenced_mesh: str | None = None,
) -> CompanionBinding:
    """为一个扫描找它的 companion mesh。

    ``order_referenced_mesh``：订单 XML 的扫描条目里若有文件名且能解析成
    已解析路径，则由调用方传入 —— 它优先级最高。

    找不到就返回未绑定。**不回退到顺序**。
    """
    # 档 1：订单显式引用
    if order_referenced_mesh and order_referenced_mesh in resolved_paths:
        p = Path(order_referenced_mesh)
        if p.suffix.lower() in COMPANION_SUFFIXES:
            return CompanionBinding(
                binding=BindingEvidence.ORDER_REFERENCE,
                scan_path=scan_path, mesh_path=order_referenced_mesh,
                mesh_suffix=p.suffix.lower(),
                note="订单 XML 的扫描条目直接引用了该 mesh")

    # 档 2：同目录同主名
    candidates = _mesh_candidates(resolved_paths, scan_path)
    if not candidates:
        return CompanionBinding(
            binding=BindingEvidence.NONE, scan_path=scan_path,
            note=("未找到 companion mesh（需与扫描同目录、同主文件名，"
                  "如 Scans/Lower/PreparationScan.stl）。"
                  "不做位置式绑定，宁可没有几何也不猜。"))

    chosen_path, chosen_suffix = candidates[0]
    return CompanionBinding(
        binding=BindingEvidence.SIBLING_SAME_STEM,
        scan_path=scan_path, mesh_path=chosen_path, mesh_suffix=chosen_suffix,
        alternatives=[c for c, _ in candidates[1:]],
        note=("与扫描同目录同主名。" if len(candidates) == 1
              else f"同主名候选 {[c for c, _ in candidates]}，按扩展名优先级取 {chosen_suffix}"))
