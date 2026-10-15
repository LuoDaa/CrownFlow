"""导入一个 3Shape 订单包（目录或 ZIP）→ ``ThreeShapePackage``。

入口只有一个：:func:`import_package`。支持目录与 ZIP 两种形态
（实际操作员上传的是 ZIP；`测试_胡昭` fixture 是解开的目录）。

**不靠文件名猜。** 每个文件的身份都从 ``AliasFilenames.xml`` 的映射还原：

    胡昭/胡昭.xml                        -> 订单主 XML（basename == 包根名）
    胡昭/Materials.xml                   -> 材料库 + CAD 参数
    胡昭/Manufacturers.3ml               -> 厂商库（ZIP 套一层）
    胡昭/Scans/Lower/PreparationScan.dcm -> 预备体（颌位从路径判定）
    胡昭/Scans/Upper/AntagonistScan.dcm  -> 对颌
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .aliases import find_package_root, infer_jaw_from_path, parse_alias_map
from .geometry_source import GeometrySource, bind_companion
from .hps_scan import classify_companion_mesh, classify_scan
from .materials import (collect_cad_parameters, material_record, parse_materials,
                        subtype_hint_from_tooth_element_type)
from .models import (
    CasePrescription,
    ScanAsset,
    ScanDecodeStatus,
    ScanRole,
    ThreeShapePackage,
)
from .order import parse_order

ALIAS_FILENAME = "AliasFilenames.xml"
#: 扫描**容器**后缀。companion mesh（STL 等）不算扫描 ——
#: 它由 geometry_source.bind_companion 绑到对应扫描上。
#: 若包里只有 mesh 而没有 HPS 容器，才把孤立的 mesh 单独当扫描（见下）。
_SCAN_SUFFIXES = frozenset({".dcm"})
_MESH_SUFFIXES = frozenset({".stl", ".ply", ".obj"})
_NON_ORDER_STEMS = frozenset({"materials", "manufacturers", "aliasfilenames"})


@contextmanager
def _package_root(path: Path) -> Iterator[Path]:
    """把目录或 ZIP 统一成"一个可读目录"。ZIP 解到临时目录，退出时清理。"""
    path = Path(path)
    if path.is_dir():
        yield path
        return
    if path.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="three_shape_"))
        try:
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            # 有些 ZIP 外面还套一层目录
            entries = [p for p in tmp.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                yield entries[0]
            else:
                yield tmp
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return
    raise ValueError(f"不支持的包形态：{path.name}（需要目录或 .zip）")


def open_package_root(path: Path | str) -> Iterator[Path]:
    """公开 ``_package_root``：在**一个持久的可读目录**上工作。

    ``import_package`` 自己管上下文的，返回后 ZIP 的解包目录已经被删掉 ——
    于是 ``ScanAsset.absolute_path`` 全部失效。任何需要在导入之后继续读网格的
    调用方（例如把 companion STL 拷进病例目录）必须用这个：

        with open_package_root(zip_path) as root:
            pkg = import_package(root)        # 传目录，不再另开临时目录
            ...                               # 这里 absolute_path 仍然有效
    """
    return _package_root(Path(path))


def mesh_physical_path(pkg: ThreeShapePackage, root: Path,
                       logical_path: str) -> Path | None:
    """逻辑路径 → 磁盘路径。物理文件名就是 GUID，只能用 alias 映射还原。

    ``root`` 必须是**仍然存在**的包根目录（见 :func:`open_package_root`）。
    """
    for guid, original in pkg.alias_map.items():
        if original == logical_path:
            candidate = Path(root) / guid
            return candidate if candidate.is_file() else None
    return None


def _find_alias_file(root: Path) -> Path | None:
    for p in root.rglob(ALIAS_FILENAME):
        return p
    return None


def _locate_files(root: Path, alias_map: dict[str, str]) -> tuple[Path | None, dict[str, tuple[Path, str]]]:
    """按映射把 GUID 文件还原成真实路径。

    返回 ``(order_xml_path, {resolved_path: (实际文件, guid)})``。
    映射里提到但磁盘上不存在的条目会被跳过（并可由调用方诊断）。
    """
    # **扩展名必须取自"解析后的原始名"，不能取自物理文件。**
    # 实测包内文件名就是 ``{GUID}`` 本身、**没有扩展名** ——
    # 在物理路径上取 suffix 永远是空串，会让下面所有按后缀分派的分支全部失效
    # （起初订单 XML 判不出、扫描一个都筛不到，就是这个原因）。
    resolved: dict[str, tuple[Path, str]] = {}
    for guid, original in alias_map.items():
        candidate = root / guid
        if candidate.is_file():
            resolved[original] = (candidate, guid)

    pkg_root_name = find_package_root(alias_map)
    order_xml: Path | None = None

    # 规则一：basename == 包根目录名 的那个 xml
    if pkg_root_name:
        for original, (path, _) in resolved.items():
            resolved_path = Path(original)
            if (resolved_path.suffix.lower() == ".xml"
                    and resolved_path.stem == pkg_root_name
                    and resolved_path.name.lower() not in _NON_ORDER_STEMS):
                order_xml = path
                break

    # 规则二（回退）：不在 Scans/ 下、且不是已知辅助文件的 xml
    if order_xml is None:
        for original, (path, _) in sorted(resolved.items()):
            if Path(original).suffix.lower() != ".xml":
                continue
            if "/scans/" in original.lower() or Path(original).stem.lower() in _NON_ORDER_STEMS:
                continue
            order_xml = path
            break

    return order_xml, resolved


def import_package(path: Path | str, *, mesh_loader=None) -> ThreeShapePackage:
    """导入订单包。目录或 ZIP 均可。

    ``mesh_loader`` 仅用于测试注入；默认走项目的 ``load_mesh``。
    """
    source = Path(path)
    with _package_root(source) as root:
        alias_file = _find_alias_file(root)
        if alias_file is None:
            raise ValueError(
                f"{source.name} 里没有 {ALIAS_FILENAME}，不是 3Shape 订单包。"
                "该文件是还原 GUID 文件名的唯一依据，不能靠文件名猜。")

        alias_map = parse_alias_map(alias_file)
        order_xml, resolved = _locate_files(root, alias_map)

        pkg = ThreeShapePackage(source_name=source.name, alias_map=alias_map)
        pkg.diagnostics["package_root_name"] = find_package_root(alias_map)
        pkg.diagnostics["n_alias_entries"] = len(alias_map)
        pkg.diagnostics["n_files_resolved"] = len(resolved)
        pkg.diagnostics["missing_from_disk"] = sorted(set(alias_map.values()) - set(resolved))

        # ---- 订单 XML -> 处方
        if order_xml is not None:
            parsed = parse_order(order_xml)
            pkg.prescription = parsed["prescription"]
            pkg.teeth = parsed["teeth"]
            pkg.order_meta = parsed["order_meta"]
            pkg.phi_fields.update(parsed["phi_fields"])
            pkg.diagnostics["order_xml"] = next(
                (k for k, (v, _) in resolved.items() if v == order_xml), order_xml.name)
            scans_index = {s.get("scan_id"): s for s in parsed["scans"] if s.get("scan_id")}
        else:
            scans_index = {}
            pkg.diagnostics["order_xml"] = None

        # ---- 材料库 + CAD 参数（按订单里的 MaterialID 取）
        material_path = next(
            (path for original, (path, _) in resolved.items()
             if Path(original).name.lower() == "materials.xml"), None)
        if material_path is not None:
            node = parse_materials(material_path)
            hint = subtype_hint_from_tooth_element_type(
                pkg.prescription.tooth_element_type_id)
            pkg.diagnostics["subtype_hint"] = hint
            pkg.material_record = material_record(
                node, pkg.prescription.material_id, subtype_hint=hint)
            pkg.cad_parameters = [p.as_dict() for p in collect_cad_parameters(
                node, pkg.prescription.material_id, subtype_hint=hint)]
            pkg.diagnostics["n_cad_parameters"] = len(pkg.cad_parameters)
        else:
            pkg.diagnostics["n_cad_parameters"] = 0

        # ---- 扫描：三态判定 + 颌位（从映射路径判定，不猜 ScanType）
        for original, (abs_path, guid) in sorted(resolved.items()):
            # 同上：按解析后的名字判后缀
            if Path(original).suffix.lower() not in _SCAN_SUFFIXES:
                continue
            status, meta, mesh, error = classify_scan(abs_path, mesh_loader=mesh_loader)
            asset = ScanAsset(
                alias_guid=guid, resolved_path=original,
                scan_id=None, scan_type=_scan_type_from_path(original),
                absolute_path=str(abs_path), decode_status=status,
                plaintext_meta=meta, error=error,
                role=_role_from_path(original),
                original_geometry_source=_source_from_status(status),
            )
            if mesh is not None:
                asset.mesh_info = _safe_mesh_info(mesh)
                asset.preferred_geometry_source = _source_from_status(status)
            pkg.scans.append(asset)

        _attach_scan_types_from_order(pkg, scans_index)

        # 兜底：包里只有 mesh、没有 HPS 容器时，把孤立 mesh 当扫描
        # （有同名 .dcm 的 mesh 是 companion，不在此列）
        if not pkg.scans:
            scan_stems = {Path(p).stem for p in resolved if Path(p).suffix.lower() in _MESH_SUFFIXES}
            for original, (abs_path, guid) in sorted(resolved.items()):
                if Path(original).suffix.lower() not in _MESH_SUFFIXES:
                    continue
                status, meta, mesh, error = classify_companion_mesh(
                    abs_path, logical_suffix=Path(original).suffix, mesh_loader=mesh_loader)
                if mesh is None:
                    continue
                asset = ScanAsset(
                    alias_guid=guid, resolved_path=original,
                    scan_id=None,
                    scan_type=_scan_type_from_path(original),
                    absolute_path=str(abs_path), decode_status=status,
                    plaintext_meta=meta, error=error,
                    role=_role_from_path(original),
                    original_geometry_source=GeometrySource.for_companion(
                        Path(original).suffix),
                    preferred_geometry_source=GeometrySource.for_companion(
                        Path(original).suffix),
                    mesh_info=_safe_mesh_info(mesh),
                )
                pkg.scans.append(asset)

        # ---- 颌位
        for asset in pkg.scans:
            jaw = infer_jaw_from_path(asset.resolved_path)
            if asset.is_preparation and jaw:
                pkg.prescription.preparation_jaw = jaw
            elif asset.is_antagonist and jaw:
                pkg.prescription.antagonist_jaw = jaw


        # ---- companion mesh 绑定：严格按显式依据，**绝不位置绑定**
        resolved_paths = sorted(resolved)
        for asset in pkg.scans:
            binding = bind_companion(
                asset.resolved_path, resolved_paths,
                order_referenced_mesh=_order_referenced_mesh(asset, scans_index))
            asset.companion = binding
            if asset.preferred_geometry_source is None:
                asset.preferred_geometry_source = _source_from_status(asset.decode_status)
            if not binding.bound:
                continue
            mesh_abs = resolved[binding.mesh_path][0]
            _st, _meta, comp_mesh, comp_error = classify_companion_mesh(
                mesh_abs, logical_suffix=binding.mesh_suffix, mesh_loader=mesh_loader)
            if comp_mesh is not None:
                # 两份 provenance 都留：原件 CE 加密，实际用 companion
                asset.preferred_geometry_source = GeometrySource.for_companion(binding.mesh_suffix)
                asset.companion_mesh_info = _safe_mesh_info(comp_mesh)
                if asset.mesh_info is None:
                    asset.mesh_info = asset.companion_mesh_info
                asset.error = None
            else:
                asset.companion_mesh_info = {"error": comp_error}

        pkg.diagnostics["companion_bindings"] = {
            s.resolved_path: (s.companion.binding.value if s.companion else None)
            for s in pkg.scans}
        pkg.diagnostics["geometry_for_design_ready"] = pkg.geometry_available_for_design
        pkg.diagnostics["scan_decode_status"] = {
            s.resolved_path: s.decode_status.value for s in pkg.scans}
        return pkg


def _role_from_path(original_path: str) -> ScanRole:
    """从映射路径判定角色。"""
    low = (_scan_type_from_path(original_path) or "").lower()
    if low in ("stpreperation", "stpreparation"):
        return ScanRole.PREPARATION
    if low in ("stantagonistmodel", "stantagonist", "stopposing"):
        return ScanRole.ANTAGONIST
    return ScanRole.OTHER


def _source_from_status(status: ScanDecodeStatus) -> GeometrySource:
    if status is ScanDecodeStatus.DECODED:
        return GeometrySource.HPS_DECODED
    if status is ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED:
        return GeometrySource.HPS_CE_ENCRYPTED
    return GeometrySource.HPS_UNSUPPORTED


def _order_referenced_mesh(asset: ScanAsset, scans_index: dict) -> str | None:
    """订单扫描条目里若引用了文件名，转成已解析路径。

    实测该包的 TDM_Item_Scan 没有可用的 FileName，通常返回 None ——
    此时靠同主名规则。**不因为拿不到就退化成顺序绑定。**
    """
    for entry in scans_index.values():
        fn = (entry.get("file_name") or "").replace("\\", "/")
        if fn and Path(fn).stem == Path(asset.resolved_path).stem:
            return fn
    return None


def _safe_mesh_info(mesh) -> dict:
    from ...geometry.io import mesh_info
    try:
        return mesh_info(mesh)
    except Exception as exc:      # pragma: no cover - 防御
        return {"error": f"{type(exc).__name__}: {exc}"}


#: 从映射路径判定扫描类型。比 ScanType 的字符串语义可靠 ——
#: 实测路径就是 ``Scans/Lower/PreparationScan.dcm`` / ``Scans/Upper/AntagonistScan.dcm``。
_PATH_SCAN_TYPES: tuple[tuple[str, str], ...] = (
    ("preparation", "stPreperation"),
    ("preperation", "stPreperation"),   # 3Shape 自己拼错过，兼容两种
    ("antagonist", "stAntagonistModel"),
    ("opposing", "stAntagonistModel"),
    ("scanabutment", "stAbutment"),
    ("abutment", "stAbutment"),
)


def _scan_type_from_path(original_path: str) -> str | None:
    low = original_path.lower()
    for token, scan_type in _PATH_SCAN_TYPES:
        if token in low:
            return scan_type
    return None


def _attach_scan_types_from_order(pkg: ThreeShapePackage, scans_index: dict) -> None:
    """用订单 XML 的 ``TDM_List_Scan`` 条目补 ``scan_id``。

    **只按数量匹配时补 scan_id，不补 scan_type** ——
    顺序只是启发式，猜错类型比留空更糟。数量不一致就不补，并把情况记进诊断。
    """
    unresolved = [s for s in pkg.scans if s.scan_id is None]
    if not unresolved:
        return
    entries = sorted(scans_index.values(), key=lambda e: str(e.get("scan_id") or ""))
    if len(entries) != len(unresolved):
        pkg.diagnostics["scan_id_matching"] = (
            f"订单列出 {len(entries)} 个扫描、包内 {len(unresolved)} 个未匹配，数量不一致，"
            "不按顺序猜测 scan_id")
        return
    for asset, entry in zip(sorted(unresolved, key=lambda s: s.resolved_path), entries):
        asset.scan_id = entry.get("scan_id")
        if not asset.scan_type:
            asset.scan_type = None      # 仍不猜类型
    pkg.diagnostics["scan_id_matching"] = "按路径字母序与订单扫描列表顺序一一对应"
