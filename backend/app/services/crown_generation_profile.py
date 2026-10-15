"""参数驱动的冠生成：统一 `CrownGenerationProfile`（本批 P0 的脊柱）。

设计纪律（与项目既有原则一致）：

1. **一个 profile，所有参数都带 provenance**。报告里每个字段都能回答
   "这个值是谁给的"（USER / DERIVED / MISSING），不再是散落在三个命名空间里的裸 dict。
2. **禁止把 MISSING 静默变 0**。取值一律 `raw.get(name)`，绝不写 `raw.get(name) or 0`；
   缺少的值保持 ``None`` 并在 ``missing()`` 里登记。哪个阶段真正需要它，由
   ``require()``/``require_group()`` 显式报错（fail-loud），而不是拿 0 去算几何。
3. **不复制几何能力**。本模块只做解析、来源标注、别名归一与缺失登记，
   不导入任何几何求解器（保持可被 API/报告层安全引用）。
4. **接触面积是有版本、未认证的工程口径**（见 ``CONTACT_AREA_DEFINITION``）：
   面积极小/极大的绝对可信度尚未验证，因此 mode→mm² 的目标来自显式配置表，
   且报告必带 ``validated=False``。UI 不许硬编码面积数值。

扁平存量名 → profile 名（别名表）保证既有 `parameters_json` 与前端旧字段继续可用：
    proximal_target_mm        → proximal.target_distance_mm
    occlusal_target_mm        → occlusion.target_clearance_mm
    cement_gap_mm             → intaglio.cement_gap_mm
    ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

#: 值的来源标注
SOURCE_USER = "USER"
SOURCE_DERIVED = "DERIVED"
SOURCE_DEFAULT_POLICY = "DEFAULT_POLICY"
SOURCE_MISSING = "MISSING"

#: 接触面积的工程口径（必须与报告一起出现；不是一个可随便替换的抽象概念）
CONTACT_AREA_DEFINITION = "sum_of_vertex_areas_where_signed_distance_le_threshold"
CONTACT_AREA_VERSION = "vertex_area_signed_threshold/1"
CONTACT_AREA_VALIDATED = False
#: "进入接触"的判定阈值：与目标距离的偏差在它以内才算接触带（工程口径）
CONTACT_AREA_BAND_MM = 0.05

#: mode → 面积目标（mm²）。**这是配置，不是 UI 常量**；将来 Material/LabProfile 可覆盖。
CONTACT_AREA_TARGETS_MM2 = {"SMALL": 2.0, "MEDIUM": 6.0, "LARGE": 12.0}
CONTACT_AREA_MODES = tuple(CONTACT_AREA_TARGETS_MM2)

UNDERCUT_REMOVE = "REMOVE"
UNDERCUT_KEEP = "KEEP"
UNDERCUT_POLICIES = (UNDERCUT_REMOVE, UNDERCUT_KEEP)


class ProfileParameterMissing(KeyError):
    """某个阶段真正需要的参数是 MISSING —— 拒绝用 0/猜值代替。"""

    def __init__(self, paths, stage: str):
        self.paths = list(paths)
        self.stage = stage
        super().__init__("%s 缺少必需参数（不接受默认值顶替）：%s" % (stage, ", ".join(self.paths)))


#: profile 分组 → (字段名, 扁平存量名)。扁平名来自 DesignCreate/parameters_json。
_GROUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "proximal": (
        ("target_distance_mm", "proximal_target_mm"),
        ("contact_area_mode", "proximal_contact_area_mode"),
        ("contact_area_target_mm2", "proximal_contact_area_target_mm2"),
        ("mesial_enabled", "proximal_mesial_enabled"),
        ("distal_enabled", "proximal_distal_enabled"),
    ),
    "occlusion": (
        ("target_clearance_mm", "occlusal_target_mm"),
        ("enabled", "occlusal_enabled"),
        ("require_confirmed_bite_pose", "occlusal_require_confirmed_bite_pose"),
    ),
    "intaglio": (
        ("cement_gap_mm", "cement_gap_mm"),
        ("margin_gap_mm", "margin_gap_mm"),
        ("cement_gap_start_mm", "cement_gap_start_mm"),
        ("cement_gap_top_mm", "cement_gap_top_mm"),
        ("margin_band_mm", "margin_band_mm"),
    ),
    "thickness": (
        ("minimum_thickness_mm", "minimum_thickness_mm"),
        ("enforce", "enforce_minimum_thickness"),
    ),
    "undercut": (
        ("policy", "undercut_policy"),
        ("blockout_angle_deg", "blockout_angle_deg"),
    ),
    "border": (
        ("smooth_transition", "border_smooth_transition"),
        ("blend_radius_mm", "border_blend_radius_mm"),
        ("blend_width_mm", "border_blend_width_mm"),
    ),
    "manufacturing": (
        ("tool_radius_mm", "tool_radius_mm"),
        ("tool_compensation_enabled", "tool_compensation_enabled"),
    ),
    "outer": (
        ("uniform_scale", "outer_uniform_scale"),
        ("md_scale", "outer_md_scale"),
        ("bl_scale", "outer_bl_scale"),
        ("height_scale", "outer_height_scale"),
    ),
}

#: 这些字段有**明确的策略默认**（规格里写死的策略，不是"缺失当 0"）。
_POLICY_DEFAULTS: dict[str, Any] = {
    "occlusion.require_confirmed_bite_pose": True,
    "occlusion.enabled": True,
    "undercut.policy": UNDERCUT_REMOVE,
    "border.smooth_transition": False,
    "manufacturing.tool_compensation_enabled": False,
    "proximal.mesial_enabled": True,
    "proximal.distal_enabled": True,
}


def _coerce(value):
    """只做类型归一，**不做**缺省填充。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text or None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


@dataclass
class CrownGenerationProfile:
    """一次冠生成的全部参数 + 每项的来源标注。"""

    #: 扁平名 → 原始值（原样保留，便于 provenance 与审计）
    raw: dict = field(default_factory=dict)
    #: "group.field" → {"value":..., "source":..., "flat_name":...}
    provenance: dict = field(default_factory=dict)
    #: 解析时发现的问题（例如未知键）
    notes: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- 取值
    def get(self, path: str, default=None):
        entry = self.provenance.get(path)
        if entry is None or entry["source"] == SOURCE_MISSING:
            return default
        return entry["value"]

    def source(self, path: str) -> str:
        return (self.provenance.get(path) or {}).get("source", SOURCE_MISSING)

    def value(self, path: str):
        """返回 ``(value, source)`` —— 调用方必须显式处理 MISSING。"""
        entry = self.provenance.get(path) or {"value": None, "source": SOURCE_MISSING}
        return entry["value"], entry["source"]

    def missing(self) -> list:
        return sorted(path for path, entry in self.provenance.items()
                      if entry["source"] == SOURCE_MISSING)

    def require(self, *paths: str, stage: str) -> dict:
        """阶段级 fail-loud：需要的参数里只要有一个 MISSING 就拒绝。"""
        absent = [path for path in paths if self.source(path) == SOURCE_MISSING]
        if absent:
            raise ProfileParameterMissing(absent, stage)
        return {path: self.get(path) for path in paths}

    # ---------------------------------------------------------------- 面向几何
    def effective_contact_area_target_mm2(self):
        """mode → mm²（DERIVED）；显式给了 mm² 就以显式值为准（USER）。"""
        explicit, explicit_source = self.value("proximal.contact_area_target_mm2")
        if explicit_source != SOURCE_MISSING and explicit is not None:
            return float(explicit), SOURCE_USER
        mode, mode_source = self.value("proximal.contact_area_mode")
        if mode_source == SOURCE_MISSING or mode is None:
            return None, SOURCE_MISSING
        mode = str(mode).upper()
        if mode not in CONTACT_AREA_TARGETS_MM2:
            raise ValueError("未知 contact_area_mode=%r（可选 %s）" % (mode, ", ".join(CONTACT_AREA_MODES)))
        return float(CONTACT_AREA_TARGETS_MM2[mode]), SOURCE_DERIVED

    def undercut_is_keep(self) -> bool:
        return str(self.get("undercut.policy") or UNDERCUT_REMOVE).upper() == UNDERCUT_KEEP

    # ---------------------------------------------------------------- 报告
    def as_report(self) -> dict:
        payload = {group: {name: self.get("%s.%s" % (group, name)) for name, _ in fields}
                   for group, fields in _GROUPS.items()}
        payload["_provenance"] = {path: dict(entry) for path, entry in sorted(self.provenance.items())}
        payload["_missing"] = self.missing()
        payload["_notes"] = list(self.notes)
        payload["_defaults_policy"] = dict(_POLICY_DEFAULTS)
        payload["contact_area"] = {"definition": CONTACT_AREA_DEFINITION,
                                   "version": CONTACT_AREA_VERSION,
                                   "validated": CONTACT_AREA_VALIDATED,
                                   "band_mm": CONTACT_AREA_BAND_MM,
                                   "targets_mm2": dict(CONTACT_AREA_TARGETS_MM2)}
        return payload


def from_parameters(parameters: dict | None) -> CrownGenerationProfile:
    """从 `parameters_json`（或等价 dict）解析 profile。

    **不静默补 0**：没有键 / 值为 None ⇒ MISSING（除非该字段有显式策略默认）。
    """
    raw = dict(parameters or {})
    profile = CrownGenerationProfile(raw=raw)
    known_flat = set()
    for group, fields in _GROUPS.items():
        for name, flat in fields:
            known_flat.add(flat)
            path = "%s.%s" % (group, name)
            if flat in raw and raw[flat] is not None:
                value, source = _coerce(raw[flat]), SOURCE_USER
            elif name == "contact_area_target_mm2":
                value, source = None, SOURCE_MISSING          # 由 mode 派生，见 effective_*
            elif path in _POLICY_DEFAULTS:
                value, source = _POLICY_DEFAULTS[path], SOURCE_DEFAULT_POLICY
            else:
                value, source = None, SOURCE_MISSING
            profile.provenance[path] = {"value": value, "source": source,
                                        "flat_name": flat, "group": group, "field": name}
            profile.groups.setdefault(group, {})[name] = value
    # 未知键只登记，不猜测（可能是别的子系统在用）
    for key in sorted(raw):
        if key not in known_flat:
            profile.notes.append("UNKNOWN_PARAMETER:%s" % key)
    return profile


def from_design(design) -> CrownGenerationProfile:
    """design 行 → profile。``parameters_json`` 是唯一参数来源。"""
    if isinstance(design, dict):
        parameters = design.get("parameters_json")
    else:
        parameters = getattr(design, "parameters_json", None)
    if isinstance(parameters, str) or parameters is None:
        parameters = json.loads(parameters or "{}")
    return from_parameters(parameters)
