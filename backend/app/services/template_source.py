"""唯一 Template Source Truth（A0）。

**为什么需要它**：live 主链此前有两个互相矛盾的模板来源 ——
`api/routes.py::initialize_draft_outer` 走的是
`TEMPLATE_DIR/FDI_xx.stl → template_studio 候选 → 第一个可用牙库`，
而 `services/auto_fit.py::propose` 的 report 又优先写 `landmarks['_selected_template'].sha256`。
⇒ 用户确认了牙库 A，真正的 baseline 几何可能来自 B，而报告写着 A：
`Hash_reported ≠ Source(Geometry)`，审计链在这条路径上是断的。

本模块提供**唯一入口** :func:`resolve_template_source`：

    design → ResolvedTemplate(path, sha256, fdi, source_kind, ...)

之后 draft / AutoFit / report / dependency digest 一律消费同一个对象，
任何模块都不允许再自己 discover 模板。

纪律：
- 用户已确认（``landmarks['_selected_template']``）时，**先校验磁盘上的真实字节**
  （文件存在 + sha256 与记录一致），不一致直接 **fail loud**（``TEMPLATE_HASH_MISMATCH``），
  **绝不静默回退**到别的模板 —— 静默回退正是本模块要消灭的失败模式。
- 没有确认记录时才走回退链，并在 ``source_kind`` 里如实标注是哪一档
  （``FALLBACK_DESIGN_DIR`` / ``FALLBACK_TEMPLATE_STUDIO`` / ``FALLBACK_LIBRARY``）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .design_dependencies import file_digest

#: ``source_kind`` 取值（写进报告/依赖摘要，禁止改语义）
USER_SELECTED = "USER_SELECTED"
FALLBACK_DESIGN_DIR = "FALLBACK_DESIGN_DIR"
FALLBACK_TEMPLATE_STUDIO = "FALLBACK_TEMPLATE_STUDIO"
FALLBACK_LIBRARY = "FALLBACK_LIBRARY"


class TemplateSourceError(ValueError):
    """模板来源无法确定（缺文件 / hash 不符 / 无可用模板）。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__("%s: %s" % (code, message))


@dataclass(frozen=True)
class ResolvedTemplate:
    """本次生成**真正消费**的那个模板（路径 + 真实字节 hash）。"""

    path: Path
    sha256: str
    fdi: int
    source_kind: str
    library: str | None = None
    proposal_id: str | None = None
    selection_revision_id: str | None = None
    metadata_path: Path | None = None
    metadata_sha256: str | None = None
    #: 仅 ``FALLBACK_LIBRARY`` 时有值：放置需要库对象（不进 report）
    library_obj: Any = field(default=None, repr=False, compare=False)

    @property
    def is_user_selected(self) -> bool:
        return self.source_kind == USER_SELECTED

    def as_report(self) -> dict:
        """写进报告/修订的 provenance（不含库对象本体）。"""
        return {
            "source_kind": self.source_kind,
            "path_name": self.path.name,
            "sha256": self.sha256,
            "fdi": self.fdi,
            "library": self.library,
            "proposal_id": self.proposal_id,
            "selection_revision_id": self.selection_revision_id,
            "metadata_sha256": self.metadata_sha256,
        }


def _selected_record(design) -> dict | None:
    try:
        landmarks = json.loads(design.get("landmarks_json") or "{}")
    except (TypeError, ValueError):
        return None
    record = landmarks.get("_selected_template")
    return record if isinstance(record, dict) else None


def _from_selected(record: dict, fdi: int) -> ResolvedTemplate:
    raw = record.get("source_path")
    if not raw:
        raise TemplateSourceError(
            "TEMPLATE_SELECTION_INCOMPLETE", "确认记录缺少 source_path，无法确定几何来源")
    path = Path(raw)
    if not path.is_file():
        raise TemplateSourceError(
            "TEMPLATE_SOURCE_MISSING",
            "已确认的模板文件不存在：%s。请重新选择并确认牙库，不能改回默认模板" % path)
    actual = file_digest(path)
    recorded = record.get("sha256")
    if recorded and actual != recorded:
        raise TemplateSourceError(
            "TEMPLATE_HASH_MISMATCH",
            "已确认模板的字节已变化（记录 %s…，实际 %s…）：%s。"
            "拒绝用别的模板冒充，请重新确认牙库" % (str(recorded)[:12], actual[:12], path.name))
    metadata_path = record.get("metadata_path")
    metadata_sha = record.get("metadata_sha256")
    if metadata_path and metadata_sha and Path(metadata_path).is_file():
        if file_digest(Path(metadata_path)) != metadata_sha:
            raise TemplateSourceError(
                "TEMPLATE_METADATA_HASH_MISMATCH",
                "模板元数据字节已变化：%s" % Path(metadata_path).name)
    library_obj = None
    library_name = record.get("library")
    if library_name:
        # 选中的是**牙库件**（source_path 是库内文件）：放置必须走库路径，需要库对象。
        from ..importers.tooth_library import discover_libraries
        from ..config import TOOTH_LIBRARY_ROOT
        try:
            library_obj = next((lib for lib in discover_libraries(str(TOOTH_LIBRARY_ROOT))
                                if lib.name == library_name), None)
        except Exception as exc:
            raise TemplateSourceError(
                "TEMPLATE_LIBRARY_UNAVAILABLE", "牙库不可用（%s）：%s" % (library_name, exc)) from exc
        if library_obj is None:
            raise TemplateSourceError(
                "TEMPLATE_LIBRARY_UNAVAILABLE",
                "已确认的牙库 %s 当前不可发现；拒绝改用其它库" % library_name)
    return ResolvedTemplate(
        path=path, sha256=actual, fdi=int(fdi), source_kind=USER_SELECTED,
        library=library_name, proposal_id=record.get("proposal_id"),
        selection_revision_id=record.get("geometry_frame_revision_id"),
        metadata_path=Path(metadata_path) if metadata_path else None,
        metadata_sha256=metadata_sha, library_obj=library_obj)


def _from_fallback(fdi: int) -> ResolvedTemplate:
    """没有用户确认时的回退链（顺序与历史行为一致，逐档如实标注）。"""
    from ..config import CASE_DIR, DATA_DIR, TEMPLATE_DIR, TOOTH_LIBRARY_ROOT  # noqa: F401

    candidate = TEMPLATE_DIR / ("FDI_%d.stl" % fdi)
    if candidate.is_file():
        return ResolvedTemplate(path=candidate, sha256=file_digest(candidate),
                                fdi=fdi, source_kind=FALLBACK_DESIGN_DIR)

    studio_dir = DATA_DIR / "template_studio" / "candidates" / ("FDI_%d" % fdi)
    studio = studio_dir / "crown_template.stl"
    if studio.is_file() and (studio_dir / "local_frame.json").is_file():
        return ResolvedTemplate(path=studio, sha256=file_digest(studio),
                                fdi=fdi, source_kind=FALLBACK_TEMPLATE_STUDIO)

    from ..importers.tooth_library import discover_libraries
    errors: list[str] = []
    try:
        libraries = list(discover_libraries(str(TOOTH_LIBRARY_ROOT)))
    except Exception as exc:                            # 牙库根没配/不可读
        libraries = []
        errors.append("牙库根不可用: %s" % exc)
    for library in libraries:
        try:
            path = Path(library.template_path(fdi))
        except Exception as exc:                       # 该库没有此牙位
            errors.append("%s: %s" % (library.name, exc))
            continue
        if path.is_file():
            return ResolvedTemplate(path=path, sha256=file_digest(path), fdi=fdi,
                                    source_kind=FALLBACK_LIBRARY, library=library.name,
                                    library_obj=library)
        errors.append("%s: 模板文件不存在" % library.name)
    raise TemplateSourceError(
        "NO_TEMPLATE_AVAILABLE",
        "FDI %d 无可用冠形模板（设计目录/候选/牙库均未命中）：%s" % (fdi, "；".join(errors)))


def resolve_template_source(design) -> ResolvedTemplate:
    """**唯一**模板来源解析（draft / AutoFit / report / digest 共用）。

    ``design`` 是 designs 行（dict-like，需有 ``target_fdi`` 与 ``landmarks_json``）。
    """
    fdi = int(design["target_fdi"])
    record = _selected_record(design)
    if record:
        return _from_selected(record, fdi)
    return _from_fallback(fdi)
