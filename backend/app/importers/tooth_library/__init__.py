"""专业牙库（``teeth/*.eoff``）的导入层。

* ``eoff``    —— ``OFF BINARY`` 二进制网格 + 逐顶点特征标签（含牙颈线闭环）
* ``library`` —— FDI → 模板文件映射、XML 牙位局部系、颌装配自检

真实牙库在仓库之外，根目录由 ``CROWNFLOW_TOOTH_LIBRARY_ROOT`` 指定。
本包纯 stdlib（与 ``app/groundtruth`` 同样的约束），所以能在无 numpy 的
数据机上直接跑解析与对账。
"""
from __future__ import annotations

from .eoff import (
    EoffError,
    EoffMesh,
    EoffSection,
    LoopCandidate,
    bit_members,
    boundary_loop,
    closed_loop_bits,
    find_margin_loop,
    exclusive_bit_members,
    label_bits_present,
    parse_eoff,
    parse_eoff_bytes,
    sections_cover_file,
)
from .library import (
    ENV_LIBRARY_ROOT,
    BowEntry,
    ToothLibrary,
    ToothLibraryError,
    assembly_report,
    discover_libraries,
    jaw_for_fdi,
    landmark_coverage,
    parse_bow_xml,
    tooth_index_for_fdi,
)

__all__ = [
    "EoffError", "EoffMesh", "EoffSection", "LoopCandidate",
    "bit_members", "boundary_loop", "closed_loop_bits",
    "exclusive_bit_members",
    "find_margin_loop",
    "label_bits_present", "parse_eoff", "parse_eoff_bytes",
    "sections_cover_file",
    "ENV_LIBRARY_ROOT", "BowEntry", "ToothLibrary", "ToothLibraryError",
    "assembly_report", "discover_libraries", "jaw_for_fdi",
    "landmark_coverage", "parse_bow_xml", "tooth_index_for_fdi",
]
