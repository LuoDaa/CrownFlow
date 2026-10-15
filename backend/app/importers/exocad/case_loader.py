"""把一个 exocad 病例目录聚合成 ``ExocadCase``。

病例身份规则（实测强制）：**只认 ``source_order_id``**。
目录名与文件名均不可信 —— 实测同一订单 ``03c3329e`` 出现在 ``82404020415薛文爱``
与 ``薛文爱`` 两个目录；``吴宝刚.matrix4`` 横跨 4 个不同病例目录。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree.ElementTree import ParseError as ETParseError

from .construction_info import parse_construction_info
from .dental_project import parse_dental_project
from .matrix4 import classify_matrix4_role, parse_matrix4
from .models import ExocadCase
from .sanitizer import scan_phi
from .three_ox import parse_three_ox


def _files(case_dir: Path, suffix: str) -> list[Path]:
    """按扩展名收集文件，大小写不敏感。"""
    suffix = suffix.lower()
    return sorted(p for p in case_dir.rglob("*") if p.is_file() and p.suffix.lower() == suffix)


def load_case(case_dir: Path | str) -> ExocadCase:
    """从单个病例目录加载。order_id 优先取 ``.3ox``，回退到 ``.dentalProject``。"""
    case_dir = Path(case_dir)

    order_id: str | None = None
    scan_source: str | None = None
    three_shape_order_no: str | None = None
    phi: dict[str, str] = {}

    for p in _files(case_dir, ".3ox"):
        info = parse_three_ox(p)
        order_id = order_id or info.order_id
        scan_source = scan_source or info.scan_source
        three_shape_order_no = three_shape_order_no or info.three_shape_order_no
        phi.update(info.phi_fields)

    restoration_note: str | None = None
    material_note: str | None = None
    teeth_from_notes: list[int] = []
    for p in _files(case_dir, ".dentalproject"):
        dp = parse_dental_project(p)
        order_id = order_id or dp.order_id
        restoration_note = restoration_note or dp.restoration_note
        material_note = material_note or dp.material_note
        teeth_from_notes = teeth_from_notes or dp.teeth_from_notes

    matrices = []
    for p in _files(case_dir, ".matrix4"):
        try:
            matrices.append(parse_matrix4(p))
        except (ValueError, ETParseError) as exc:
            phi.setdefault("matrix4_parse_error", f"{p.name}: {exc}")
    verdict = classify_matrix4_role(matrices)

    teeth = []
    for p in _files(case_dir, ".constructioninfo"):
        teeth.extend(parse_construction_info(p, source_order_id=order_id))

    findings = scan_phi(case_dir)
    phi_paths = sorted({f.where for f in findings
                        if f.kind.value in ("PATH_NAME", "BINARY_CANNOT_SCRUB")})
    for f in findings:
        if f.kind.value in ("XML_FIELD", "LOG_NETWORK_PATH"):
            phi.setdefault(f"{f.kind.value}:{f.where}", f.detail)

    case_uuid = "cf_" + hashlib.sha256((order_id or case_dir.name).encode("utf-8")).hexdigest()[:16]

    return ExocadCase(
        case_uuid=case_uuid,
        source_order_id=order_id,
        case_dir_name=case_dir.name,
        teeth=teeth,
        restoration_note=restoration_note,
        material_note=material_note,
        teeth_from_notes=teeth_from_notes,
        scan_source=scan_source,
        three_shape_order_no=three_shape_order_no,
        phi_fields=phi,
        phi_paths=phi_paths,
        matrix4_role=verdict.role.value,
    )


def discover_cases(raw_root: Path | str) -> dict[str, list[Path]]:
    """扫描根目录，返回 ``{source_order_id: [case_dir, ...]}``。

    **这是防训练/验证泄露的唯一正确分组方式**：按订单分组，不按目录。
    同一 ``source_order_id`` 下的所有目录必须整体进入同一侧。
    """
    raw_root = Path(raw_root)
    groups: dict[str, list[Path]] = {}

    for d in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        order_id = None
        for suffix in (".3ox", ".dentalproject"):
            for p in _files(d, suffix):
                order_id = (parse_three_ox(p).order_id if suffix == ".3ox"
                            else parse_dental_project(p).order_id)
                if order_id:
                    break
            if order_id:
                break
        key = order_id or f"NO_ORDER_ID::{d.name}"
        groups.setdefault(key, []).append(d)

    return groups
