"""3Shape 订单包导入的数据模型。

关键约定：**扫描的三态**。`测试_胡昭` 实测结果是 `SUPPORTED_BUT_ENCRYPTED` ——
格式支持（`CE` ∈ ``hpsdecode.SUPPORTED_SCHEMAS``），元数据全可读，
但顶点坐标是 Blowfish 加密的，拿不到 mesh。详见 hps_scan.py 的实测记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Universal（美国编号）1–32 → FDI。
#: 与 ``app/api/routes.py`` 的 ``UNIVERSAL_TO_FDI`` 必须一致 ——
#: 有测试断言两者相同，防止各自漂移。
UNIVERSAL_TO_FDI: dict[int, int] = {
    1: 18, 2: 17, 3: 16, 4: 15, 5: 14, 6: 13, 7: 12, 8: 11,
    9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
    17: 38, 18: 37, 19: 36, 20: 35, 21: 34, 22: 33, 23: 32, 24: 31,
    25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48,
}


class ScanDecodeStatus(str, Enum):
    """扫描几何的可解码状态（三态，不允许第四种）。"""

    DECODED = "DECODED"
    SUPPORTED_BUT_ENCRYPTED = "SUPPORTED_BUT_ENCRYPTED"
    UNSUPPORTED = "UNSUPPORTED"


class ScanRole(str, Enum):
    """扫描在病例里的角色。**从映射路径判定，不从 ScanType 的字符串语义猜。**"""

    PREPARATION = "PREPARATION"
    ANTAGONIST = "ANTAGONIST"
    OTHER = "OTHER"


class ImportStatus(str, Enum):
    CASE_READY = "CASE_READY"
    CASE_NOT_READY = "CASE_NOT_READY"
    FAILED = "FAILED"


@dataclass
class ScanAsset:
    """订单里的一个扫描。``resolved_path`` 由 AliasFilenames 还原，不靠文件名猜。"""

    alias_guid: str
    resolved_path: str
    scan_id: str | None
    scan_type: str | None           # stPreperation / stAntagonistModel / …
    absolute_path: str | None = None
    decode_status: ScanDecodeStatus = ScanDecodeStatus.UNSUPPORTED
    #: 明文可读的扫描元数据（即使几何加密也能拿到）
    plaintext_meta: dict[str, Any] = field(default_factory=dict)
    mesh_info: dict[str, Any] | None = None
    error: str | None = None
    role: ScanRole = ScanRole.OTHER
    jaw: str | None = None
    #: HPS 原件自身的来源。与 preferred_geometry_source 分开 ——
    #: 前者说原件（实测是 HPS_CE_ENCRYPTED），后者说实际可用的几何。
    original_geometry_source: Any = None
    preferred_geometry_source: Any = None
    companion: Any = None
    companion_mesh_info: dict[str, Any] | None = None

    @property
    def geometry_state(self) -> str:
        """实际可用的几何状态。

        有可用几何（companion 解码成功或 HPS 本身可解）-> ``GEOMETRY_READY``；
        否则回落到原件三态（实测 ``SUPPORTED_BUT_ENCRYPTED``）。

        这样"元数据就绪但几何不可用"不会被误报成"可以开始设计"。
        """
        pref = self.preferred_geometry_source
        if pref is not None and getattr(pref, "is_usable", False):
            return "GEOMETRY_READY"
        return self.decode_status.value

    @property
    def geometry_ready(self) -> bool:
        pref = self.preferred_geometry_source
        return bool(pref is not None and getattr(pref, "is_usable", False))

    @property
    def geometry_source_path(self) -> str:
        """**实际提供几何的**逻辑路径。companion 优先，否则回落原件。

        注意这是 AliasFilenames 里的逻辑路径，不是磁盘路径 ——
        物理文件名是 GUID。要用它取磁盘文件见 ``package.mesh_physical_path``。
        """
        if self.geometry_ready and self.companion is not None and self.companion.bound:
            return self.companion.mesh_path
        return self.resolved_path

    @property
    def is_preparation(self) -> bool:
        return (self.scan_type or "").lower() in ("stpreperation", "stpreparation")

    @property
    def is_antagonist(self) -> bool:
        return (self.scan_type or "").lower() in ("stantagonistmodel", "stantagonist")

    def as_dict(self) -> dict:
        return {
            "alias_guid": self.alias_guid,
            "resolved_path": self.resolved_path,
            "scan_id": self.scan_id,
            "scan_type": self.scan_type,
            "decode_status": self.decode_status.value,
            "role": self.role.value,
            "jaw": self.jaw,
            "geometry_state": self.geometry_state,
            "geometry_ready": self.geometry_ready,
            "geometry_source_path": self.geometry_source_path,
            "original_geometry_source": getattr(self.original_geometry_source, "value", None),
            "preferred_geometry_source": getattr(self.preferred_geometry_source, "value", None),
            "companion_binding": (self.companion.binding.value
                                  if self.companion is not None else None),
            "companion_mesh_path": (self.companion.mesh_path
                                    if self.companion is not None else None),
            "plaintext_meta": dict(self.plaintext_meta),
            "mesh_info": self.mesh_info,
            "companion_mesh_info": self.companion_mesh_info,
            "error": self.error,
        }


@dataclass(frozen=True)
class ToothElement:
    tooth_element_id: str | None
    tooth_number_universal: int | None
    tooth_element_type_id: str | None
    cache_tooth_type_class: str | None      # teCrown / teInlay / …
    anatomical: bool | None
    post_and_core: bool | None

    @property
    def target_fdi(self) -> int | None:
        if self.tooth_number_universal is None:
            return None
        return UNIVERSAL_TO_FDI.get(int(self.tooth_number_universal))

    def as_dict(self) -> dict:
        return {
            "tooth_element_id": self.tooth_element_id,
            "tooth_number_universal": self.tooth_number_universal,
            "target_fdi": self.target_fdi,
            "tooth_element_type_id": self.tooth_element_type_id,
            "cache_tooth_type_class": self.cache_tooth_type_class,
            "anatomical": self.anatomical,
            "post_and_core": self.post_and_core,
        }


@dataclass
class CasePrescription:
    """从订单 XML 抽出的统一处方。**用户不需要再手填牙位与材料。**"""

    target_fdi: int | None = None
    restoration_type: str | None = None      # CROWN / …
    material_name: str | None = None         # Zirkon
    material_id: str | None = None           # Configuration_11_Material1
    tooth_element_type_id: str | None = None
    preparation_jaw: str | None = None       # lower（由 stPreperation 所属目录推断）
    antagonist_jaw: str | None = None        # upper
    process_status: str | None = None        # psScanned
    source_system: str = "3shape"

    def as_dict(self) -> dict:
        return {
            "target_fdi": self.target_fdi,
            "restoration_type": self.restoration_type,
            "material_name": self.material_name,
            "material_id": self.material_id,
            "tooth_element_type_id": self.tooth_element_type_id,
            "preparation_jaw": self.preparation_jaw,
            "antagonist_jaw": self.antagonist_jaw,
            "process_status": self.process_status,
            "source_system": self.source_system,
        }


@dataclass
class ThreeShapePackage:
    """一个订单包的完整导入结果。"""

    source_name: str
    alias_map: dict[str, str] = field(default_factory=dict)
    scans: list[ScanAsset] = field(default_factory=list)
    teeth: list[ToothElement] = field(default_factory=list)
    prescription: CasePrescription = field(default_factory=CasePrescription)
    material_record: dict[str, Any] | None = None
    cad_parameters: list[dict] = field(default_factory=list)
    order_meta: dict[str, Any] = field(default_factory=dict)
    phi_fields: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def scan_of_type(self, kind: str) -> ScanAsset | None:
        for s in self.scans:
            if kind == "preparation" and s.is_preparation:
                return s
            if kind == "antagonist" and s.is_antagonist:
                return s
        return None

    @property
    def geometry_available(self) -> bool:
        return any(s.geometry_ready for s in self.scans)

    @property
    def geometry_available_for_design(self) -> bool:
        """进入自动设计所需的最小几何集：**预备体与对颌都要有可用几何**。

        只看"有没有任一扫描可用"不够 —— 只有对颌能解、预备体不能解时仍不能设计。
        """
        prep = self.scan_of_type("preparation")
        ant = self.scan_of_type("antagonist")
        return bool(prep and prep.geometry_ready and ant and ant.geometry_ready)

    def as_dict(self) -> dict:
        return {
            "source_name": self.source_name,
            "alias_map": dict(self.alias_map),
            "scans": [s.as_dict() for s in self.scans],
            "teeth": [t.as_dict() for t in self.teeth],
            "prescription": self.prescription.as_dict(),
            "material_record": self.material_record,
            "cad_parameters": list(self.cad_parameters),
            "order_meta": dict(self.order_meta),
            "phi_fields": dict(self.phi_fields),
            "geometry_available": self.geometry_available,
            "geometry_available_for_design": self.geometry_available_for_design,
            "diagnostics": dict(self.diagnostics),
        }
