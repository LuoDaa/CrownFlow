"""3Shape 订单包导入。

入口：``import_package(path)``（目录或 ZIP）→ ``ThreeShapePackage``；
``validate_package(pkg)`` → ``ValidationReport``。

**实测结论**：`测试_胡昭` 包的元数据层完全可读（牙位/修复体/材料/CAD 参数），
但两个扫描的顶点数据是 3Shape CE 加密的（Blowfish），
三态判定为 ``SUPPORTED_BUT_ENCRYPTED`` —— mesh 拿不到。详见 ``hps_scan.py``。
"""

from .models import (
    CasePrescription,
    ImportStatus,
    ScanAsset,
    ScanDecodeStatus,
    ThreeShapePackage,
    ToothElement,
    UNIVERSAL_TO_FDI,
)
from .aliases import find_package_root, infer_jaw_from_path, parse_alias_map
from .materials import MappingStatus, ParameterValue, collect_cad_parameters
from .order import parse_order, restoration_type_from_id
from .hps_scan import classify_scan, plaintext_metadata
from .package import import_package, mesh_physical_path, open_package_root
from .validator import CheckResult, ValidationReport, validate_package

__all__ = [
    "import_package",
    "open_package_root",
    "mesh_physical_path",
    "validate_package",
    "ValidationReport",
    "CheckResult",
    "ThreeShapePackage",
    "CasePrescription",
    "ScanAsset",
    "ScanDecodeStatus",
    "ToothElement",
    "ImportStatus",
    "UNIVERSAL_TO_FDI",
    "parse_alias_map",
    "find_package_root",
    "infer_jaw_from_path",
    "parse_order",
    "restoration_type_from_id",
    "collect_cad_parameters",
    "ParameterValue",
    "MappingStatus",
    "classify_scan",
    "plaintext_metadata",
]
