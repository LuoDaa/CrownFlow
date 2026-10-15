"""exocad 病例文件导入器。

支持的来源文件：
- ``.constructionInfo``  per-tooth 真实设计参数（XML，必须用 XML parser）
- ``.dentalProject``     病例级标注（牙位/修复体/材料）与 OrderID
- ``.3ox`` / ``.3OX``    订单交换消息（UTF-16 XML），含 PHI 与 ScanSource
- ``.matrix4``           固定坐标系约定变换（**不是**逐病例咬合配准，见 matrix4.py）

设计约束（来自真实数据实测，勿违反）：
1. ``.constructionInfo`` 里存在 ``<Value xsi:nil="true" />`` 这类自闭合标签，
   正则匹配 ``<Value>(.*?)</Value>`` 会跳过它们并使其后的 Value/Type 全部错位配对。
   一律使用 ``xml.etree``。
2. exocad 只写出被显式设定或偏离默认的参数。某牙位缺失某参数**不等于该参数为 0**，
   而是 "exocad 内部默认值，未知"。见 models.ParameterState。
3. 病例身份只认 ``source_order_id``（来自 .3ox / .dentalProject），
   目录名与文件名均不可信（实测：同一订单出现在两个目录；同名 .matrix4 横跨 4 个病例目录）。
"""

#: 名字 -> (子模块, 该模块里的属性名)。惰性导入，避免为一个 XML 解析
#: 就把 numpy 拖进来 —— 实测服务器上没有 numpy，而索引器要就地跑。
_LAZY: dict[str, tuple[str, str]] = {
    # models
    "ParameterCell": ("models", "ParameterCell"),
    "ParameterState": ("models", "ParameterState"),
    "ParameterScope": ("models", "ParameterScope"),
    "ReconstructionType": ("models", "ReconstructionType"),
    "FunctionalRole": ("models", "FunctionalRole"),
    "ExocadCase": ("models", "ExocadCase"),
    "ExocadToothRecord": ("models", "ExocadToothRecord"),
    "ExocadConstructionCase": ("models", "ExocadConstructionCase"),
    "ExocadConstructionFile": ("models", "ExocadConstructionFile"),
    "ExocadScanFile": ("models", "ExocadScanFile"),
    # construction_info
    "parse_construction_info": ("construction_info", "parse_construction_info"),
    "parse_construction_case": ("construction_info", "parse_construction_case"),
    # dental_project
    "parse_dental_project": ("dental_project", "parse_dental_project"),
    "DentalProjectInfo": ("dental_project", "DentalProjectInfo"),
    # three_ox
    "parse_three_ox": ("three_ox", "parse_three_ox"),
    # matrix4（会 import numpy —— 所以必须惰性）
    "parse_matrix4": ("matrix4", "parse_matrix4"),
    "Matrix4Role": ("matrix4", "Matrix4Role"),
    "classify_matrix4_role": ("matrix4", "classify_matrix4_role"),
    # case_loader（也会拉 numpy）
    "load_case": ("case_loader", "load_case"),
    "discover_cases": ("case_loader", "discover_cases"),
    # sanitizer
    "scan_phi": ("sanitizer", "scan_phi"),
    "build_deid_plan": ("sanitizer", "build_deid_plan"),
    "apply_deid_plan": ("sanitizer", "apply_deid_plan"),
    "write_mapping": ("sanitizer", "write_mapping"),
    "PhiKind": ("sanitizer", "PhiKind"),
}


def __getattr__(name: str):
    """PEP 562：按需导入。`from app.importers.exocad import parse_matrix4` 照旧可用。"""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(f".{target[0]}", __name__)
    value = getattr(mod, target[1])
    globals()[name] = value          # 缓存，后续访问不再走 __getattr__
    return value


def __dir__():
    return sorted(__all__)


__all__ = [
    "ParameterCell",
    "ParameterState",
    "ParameterScope",
    "ReconstructionType",
    "FunctionalRole",
    "ExocadCase",
    "ExocadToothRecord",
    "parse_construction_info",
    "parse_dental_project",
    "parse_three_ox",
    "parse_matrix4",
    "Matrix4Role",
    "classify_matrix4_role",
    "load_case",
    "discover_cases",
    "scan_phi",
    "build_deid_plan",
    "apply_deid_plan",
    "write_mapping",
    "PhiKind",
]
