"""Crown 语义区域（CrownRegionMap）：从**冠几何 + 就位轴 + 牙位朝向**自动建立。

设计纪律
--------
1. **不依赖世界 XYZ**。所有分类都在**冠局部系**里做：
   ``z`` = 就位轴（occlusal 方向）、``x`` = 近中、``y`` = 颊侧。
   局部系优先取**已人工确认**的患者帧（`confirmed_patient_frame` / `autofit_frame_json`，
   其 ``x`` 就是人工确认的近中方向）；没有确认帧时退回**真实邻牙几何**定近远中方向。
2. 区域权重是**连续**的 ``[0,1]``（soft weight），不是硬二值 mask。硬域只是 ``weights>=0.5``
   且 ``protection>0`` 的派生量，供适配求解器使用。
3. 保护只有一份实现：``geometry.sculpt.cervical_weights``（INNER / MARGIN / CERVICAL 全在里面，
   因为这三类都位于颈缘带及其下方）。本模块只是把它按「内表面 / 边缘带 / 颈缘」重新分解成标签，
   **不改写、不复制**那套权重。适配、自由塑形、接触分析都必须用同一份保护。

所有阈值都是工程默认值，未做临床验证（``ENGINEERING_DEFAULT_UNVALIDATED``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .nonplanar_crown import MarginCurtain
from .sculpt import cervical_weights

REGION_NAMES = ("inner", "margin", "cervical", "mesial", "distal", "buccal", "lingual", "occlusal")
OUTER_REGIONS = ("mesial", "distal", "buccal", "lingual", "occlusal")
PROTECTED_REGIONS = ("inner", "margin", "cervical")

#: 工程默认（未临床验证）。
OCCLUSAL_NORMAL_LOW = 0.15      # 法向与就位轴的点积下限（低于此值不算咬合面）
OCCLUSAL_NORMAL_FULL = 0.75     # 达到此值即认为完全朝冠方
OCCLUSAL_DEPTH_MM = 2.0         # 咬合面从冠顶往下取多深（含牙尖/窝/沟）
LATERAL_NORMAL_MIN = 0.25       # 侧面法向：与就位轴夹角大到这个程度才可能参与邻接
AXIAL_WALL_MM = 0.55            # |n·z| 超过它就不再算轴向壁
CONTACT_RADIUS_MM = 3.5         # 邻牙距离窗口（超出即认为不可能接触）
CONTACT_FULL_MM = 0.5           # 窗口内这个距离以内权重为满
CONTACT_HEIGHT_LOW_MM = 0.6
#: 邻牙距离项的下限：邻牙很远时**不允许把整片近接区清零**
#: （实测：近中裁片 5.3mm 远 ⇒ proximity=0 ⇒ 区域为空 ⇒ 适配返回 NO_APPLICABLE_SIDE）。
#: 有下限后区域始终存在；能否真正贴合由求解器如实报 CONSTRAINT_CONFLICT。
CONTACT_PROXIMITY_FLOOR = 0.60     # 邻接带下界（离边缘线多高开始可能接触）
MARGIN_BAND_MM = 0.4            # 「边缘带」标签宽度
MIN_OCCLUSAL_WINDOW_MM = 0.3    # 咬合窗口比这还浅就不给咬合权重（宁可报不可用）

THRESHOLD = 0.5


class RegionMapError(ValueError):
    """拿不到可用语义时的**显式**失败（绝不退回世界 XYZ 猜）。"""

    def __init__(self, code, message=None):
        super().__init__(message or code)
        self.code = code


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9 or not np.isfinite(n):
        raise RegionMapError("DEGENERATE_DIRECTION")
    return v / n


def _as_surface(value):
    """`confirmed_neighbor_surfaces` 给的是 ``(FitSurface, entry)``；这里两种形状都接受。"""
    if isinstance(value, tuple):
        return value[0] if value else None
    return value


@dataclass
class CrownRegionMap:
    """一个冠形上的语义区域权重与保护权重（同一份数据供所有消费者使用）。"""

    weights: dict            # name -> (n,) 连续权重
    protection: np.ndarray   # (n,) 0 = 完全保护（= cervical_weights，唯一实现）
    axes: dict               # {'mesial','buccal','occlusal'}
    provenance: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    fdi: int | None = None
    neighbor_distance_mm: dict = field(default_factory=dict)

    def domain(self, name, *, threshold=THRESHOLD):
        """可形变域：权重足够 **且** 不被保护。保护类区域（inner/margin/cervical）恒为空。"""
        return (self.deformation_weight(name) >= threshold) & (self.protection > 0)

    def deformation_weight(self, name, *, threshold=THRESHOLD):
        """**形变权重**（口径统一在这里定义）：

        - INNER / MARGIN / CERVICAL：恒为 0 —— 这三类只作为保护标签存在，任何操作都不许改；
        - MESIAL / DISTAL / BUCCAL / LINGUAL / OCCLUSAL：区域权重本身（已乘过 protection，
          所以颈缘带及其下方天然为 0）。
        """
        if name in PROTECTED_REGIONS:
            return np.zeros(len(self.protection))
        return np.asarray(self.weights[name], float)

    def counts(self, *, threshold=THRESHOLD):
        return {name: int((np.asarray(w) >= threshold).sum()) for name, w in self.weights.items()}

    def domain_counts(self, *, threshold=THRESHOLD):
        return {name: int(self.domain(name, threshold=threshold).sum()) for name in self.weights}

    def protection_labels(self, *, threshold=THRESHOLD):
        """保护类标签覆盖情况：每个被保护顶点都至少落进 inner/margin/cervical 之一。"""
        covered = np.zeros(len(self.protection), bool)
        for name in PROTECTED_REGIONS:
            covered |= np.asarray(self.weights[name]) >= threshold
        return covered

    def weights_report(self):
        return {name: np.asarray(w).tolist() for name, w in self.weights.items()}

    def report(self):
        return {
            "method": "crown_regions/local_frame/1",
            "caliber": "ENGINEERING_DEFAULT_UNVALIDATED",
            "fdi": self.fdi,
            "axes": {key: np.asarray(value).tolist() for key, value in self.axes.items()},
            "provenance": list(self.provenance),
            "flags": list(self.flags),
            "counts": self.counts(),
            "domain_counts": self.domain_counts(),
            "protected_vertex_count": int((self.protection <= 0).sum()),
            "neighbor_distance_mm": {k: (None if v is None else float(v)) for k, v in self.neighbor_distance_mm.items()},
            "thresholds": {"weights": THRESHOLD, "occlusal_depth_mm": OCCLUSAL_DEPTH_MM,
                           "contact_radius_mm": CONTACT_RADIUS_MM, "margin_band_mm": MARGIN_BAND_MM},
        }


# --------------------------------------------------------------------- 局部系
def resolve_anatomical_axes(frame, ref, *, crown_centroid, neighbors=None, fdi=None):
    """确定 (近中方向, 颊侧方向, 就位轴) 与它们的来源。

    优先级（§4/§6）：
      1. 已人工确认的患者帧（``frame.x`` = 确认的近中方向，``ref['buccal']`` = 确认的颊侧）；
      2. 病例里**真实邻牙几何**（已确认的 mesial/distal 裁片质心方向）；
      3. 两者都没有 ⇒ 抛 ``RegionMapError``，不用世界轴猜。

    ⚠️ 当 1 与 2 冲突时，**按本批要求取邻牙几何**（真实解剖优先），并把冲突写进 flags，
    不静默翻转人工确认的方向。
    """
    z = _unit(frame.z)
    provenance = []
    flags = []
    conflict = False

    neighbor_dir = {}
    for side, raw in (neighbors or {}).items():
        surface = _as_surface(raw)
        if side not in ("mesial", "distal") or surface is None:
            continue
        points = np.asarray(surface.mesh.vertices, float)
        if not len(points):
            continue
        delta = points.mean(axis=0) - np.asarray(crown_centroid, float)
        planar = delta - float(np.dot(delta, z)) * z
        if float(np.linalg.norm(planar)) > 1e-6:
            neighbor_dir[side] = planar / np.linalg.norm(planar)

    confirmed = frame.revision_id is not None
    mesial = None
    if confirmed:
        mesial = _unit(np.asarray(frame.x, float) - float(np.dot(frame.x, z)) * z)
        provenance.append({"axis": "mesial", "source": "CONFIRMED_PATIENT_FRAME",
                          "detail": "人工确认的近中方向（autofit_frame_json.x）"})
    if neighbor_dir.get("mesial") is not None:
        alignment = float(np.dot(mesial, neighbor_dir["mesial"])) if mesial is not None else 1.0
        if mesial is None:
            mesial = neighbor_dir["mesial"]
            provenance.append({"axis": "mesial", "source": "NEIGHBOR_GEOMETRY",
                              "detail": "由已确认的近中邻牙裁片质心方向确定"})
        elif alignment < 0.0:
            conflict = True
            mesial = neighbor_dir["mesial"]
            provenance.append({"axis": "mesial", "source": "NEIGHBOR_GEOMETRY_OVERRIDES_FRAME",
                              "detail": "已确认帧的近中方向与真实近中邻牙几何相反，按邻牙几何取"})
            flags.append("MESIAL_AXIS_TAKEN_FROM_NEIGHBOR_GEOMETRY_FRAME_CONFLICT")
    elif mesial is None and neighbor_dir.get("distal") is not None:
        mesial = -neighbor_dir["distal"]
        provenance.append({"axis": "mesial", "source": "NEIGHBOR_GEOMETRY_INFERRED_OPPOSITE_DISTAL"})
        flags.append("MESIAL_AXIS_INFERRED_AS_OPPOSITE_OF_DISTAL_NEIGHBOR")
    if mesial is None:
        raise RegionMapError("LOCAL_FRAME_UNCONFIRMED_AND_NO_CONFIRMED_NEIGHBOR_GEOMETRY",
                             "没有确认的患者帧，也没有可用的已确认邻牙几何，无法确定近远中方向")
    if not conflict and neighbor_dir.get("mesial") is not None and not confirmed:
        flags.append("MESIAL_AXIS_FROM_NEIGHBOR_GEOMETRY_UNVERIFIED_BUCCAL")

    buccal = None
    stated = ref.get("buccal") if isinstance(ref, dict) else None
    if stated is not None:
        candidate = np.asarray(stated, float)
        candidate = candidate - float(np.dot(candidate, z)) * z
        if float(np.linalg.norm(candidate)) > 1e-6:
            buccal = _unit(candidate)
            provenance.append({"axis": "buccal", "source": "CONFIRMED_PATIENT_FRAME"})
    if buccal is None:
        buccal = _unit(np.cross(z, mesial))
        provenance.append({"axis": "buccal", "source": "RIGHT_HAND_RULE_FROM_MESIAL_AND_OCCLUSAL",
                          "detail": "颊侧绝对朝向未确认，只保证与近中/就位轴正交"})
        flags.append("BUCCAL_AXIS_RIGHT_HAND_RULE_UNVERIFIED")

    provenance.append({"axis": "occlusal", "source": "INSERTION_AXIS"})
    if fdi is not None:
        provenance.append({"axis": "fdi", "value": int(fdi)})
    return {"mesial": mesial, "buccal": buccal, "occlusal": z}, provenance, flags, neighbor_dir


# --------------------------------------------------------------------- 主体
def build_region_map(design, mesh, frame, ref, *, neighbors=None, neighbor_reports=None, fdi=None):
    """计算一个冠形的 CrownRegionMap（幂等、确定性、不修改输入网格）。"""
    import json

    vertices = np.asarray(mesh.vertices, float)
    normals = np.asarray(mesh.vertex_normals, float)
    margin = np.asarray(json.loads(design["margin_json"]), float)
    axis = np.asarray(frame.z, float)
    protection = cervical_weights(mesh, margin, axis)
    height = MarginCurtain(margin, axis).axial_above(vertices)
    fdi = int(design.get("target_fdi", fdi)) if (design.get("target_fdi") is not None or fdi is not None) else fdi

    axes, provenance, flags, neighbor_dir = resolve_anatomical_axes(
        frame, ref, crown_centroid=vertices.mean(axis=0), neighbors=neighbors, fdi=fdi)

    origin = np.asarray(frame.origin, float)
    local = vertices - origin
    across = local @ axes["mesial"]
    occlusal_normal = normals @ axes["occlusal"]
    lateral = 1.0 - _smoothstep((np.abs(occlusal_normal) - LATERAL_NORMAL_MIN) / (1.0 - LATERAL_NORMAL_MIN))

    # ── 保护类标签：由唯一一份 protection 派生出 inner / margin / cervical ──
    inner = _smoothstep((-height) / MARGIN_BAND_MM)
    margin_band = np.clip(1.0 - np.abs(height) / MARGIN_BAND_MM, 0.0, 1.0) * (1.0 - inner)
    cervical = np.clip(1.0 - protection - inner - margin_band, 0.0, 1.0)

    # ── 咬合面：连续区域（含牙尖/窝/沟），到轴向壁衰减，到颈缘为 0 ──
    top = float(np.max(height)) if len(height) else 0.0
    window_lo = max(top - OCCLUSAL_DEPTH_MM, MARGIN_BAND_MM + 0.2)
    if top - window_lo < MIN_OCCLUSAL_WINDOW_MM:
        occlusal = np.zeros(len(vertices))
        flags.append("OCCLUSAL_WINDOW_TOO_SHALLOW")
        provenance.append({"region": "occlusal", "source": "UNAVAILABLE",
                          "detail": "冠高不足以区分咬合面与颈缘带"})
    else:
        uplift = _smoothstep((occlusal_normal - OCCLUSAL_NORMAL_LOW) / (OCCLUSAL_NORMAL_FULL - OCCLUSAL_NORMAL_LOW))
        upward = np.clip(height, 0.0, None)
        depth = _smoothstep((upward - window_lo) / (top - window_lo))
        occlusal = uplift * depth * protection
        provenance.append({"region": "occlusal", "source": "AUTO_GEOMETRY",
                          "detail": "法向朝就位轴 + 冠顶窗口 %.1fmm，连续权重含牙尖/窝/沟" % OCCLUSAL_DEPTH_MM})

    weights = {"inner": inner, "margin": margin_band, "cervical": cervical, "occlusal": occlusal}

    # ── 近中 / 远中 / 颊侧 / 舌侧：轴向壁上真正可能参与接触的那一段 ──
    neighbor_distance = {}
    for name, sign in (("mesial", 1.0), ("distal", -1.0)):
        facing = np.clip(normals @ (sign * axes["mesial"]), 0.0, 1.0)
        band_lo = CONTACT_HEIGHT_LOW_MM
        band_hi = max(window_lo, band_lo + MIN_OCCLUSAL_WINDOW_MM)
        contact_band = _smoothstep((height - band_lo) / 0.5) * (1.0 - _smoothstep((height - band_hi) / 0.5))
        proximity = np.ones(len(vertices))
        surface = _as_surface((neighbors or {}).get(name))
        if surface is not None:
            distance = _distance_to_surface(surface, vertices)
            neighbor_distance[name] = float(distance.min()) if len(distance) else None
            proximity = 1.0 - _smoothstep((distance - CONTACT_FULL_MM) / CONTACT_RADIUS_MM)
            nearest = float(distance.min()) if len(distance) else None
            far = nearest is not None and nearest > CONTACT_RADIUS_MM
            if far:
                # 只在"邻牙确实很远"时给软下限：区域仍然存在（不再返回 NO_APPLICABLE_SIDE），
                # 但**不**去撑大近侧邻牙正常的那些侧（实测撑大后求解器会整体回滚，属回归）。
                flags.append("NEIGHBOR_FAR:" + name)
                proximity = np.maximum(proximity, CONTACT_PROXIMITY_FLOOR)
            provenance.append({"region": name, "source": "AUTO_GEOMETRY_WITH_NEIGHBOR_DISTANCE",
                              "detail": "朝向 + 高度带 + 到真实邻牙距离（软项，下限 %.2f；最近 %.3fmm）"
                                        % (CONTACT_PROXIMITY_FLOOR, nearest if nearest is not None else float("nan"))})
        else:
            neighbor_distance[name] = None
            flags.append("NO_NEIGHBOR:" + name)
            provenance.append({"region": name, "source": "AUTO_GEOMETRY_NO_NEIGHBOR",
                              "detail": "该侧没有可确认的邻牙几何，只有朝向/高度带判据"})
        weights[name] = facing * lateral * contact_band * protection * proximity
        weights[name] = np.clip(weights[name], 0.0, 1.0)
        if not np.any(weights[name] >= THRESHOLD):
            flags.append(("%s_REGION_EMPTY" % name.upper()))

    for name, axis_vector in (("buccal", axes["buccal"]), ("lingual", -axes["buccal"])):
        facing = np.clip(normals @ axis_vector, 0.0, 1.0)
        weights[name] = np.clip(facing * lateral * protection, 0.0, 1.0)
    provenance.append({"region": "buccal/lingual", "source": "AUTO_GEOMETRY",
                      "detail": "颊舌向朝向；只用于显示与排除，不作为适配目标"})

    for name in REGION_NAMES:
        weights.setdefault(name, np.zeros(len(vertices)))

    if neighbor_reports:
        for side, report in neighbor_reports.items():
            if report.get("status") in ("SKIPPED_ABSENT", "NOT_ASSESSED"):
                flags.append("NEIGHBOR_%s:%s" % (side.upper(), report.get("status")))

    return CrownRegionMap(weights={name: weights[name] for name in REGION_NAMES}, protection=np.asarray(protection, float),
                          axes=axes, provenance=provenance, flags=sorted(set(flags)), fdi=fdi,
                          neighbor_distance_mm=neighbor_distance)


def _distance_to_surface(surface, points):
    from scipy.spatial import cKDTree
    cloud = np.asarray(surface.mesh.vertices, float)
    if not len(cloud):
        return np.full(len(points), np.inf)
    return cKDTree(cloud).query(points)[0]


def region_weights(state, name, region_map):
    """语义权重**唯一取用口**：人工标注优先，其次自动几何。"""
    manual = (state.get("regions") or {}).get(name) if state else None
    if manual is not None:
        return np.asarray(manual, float), "MANUAL_CONFIRMED"
    if region_map is not None and name in region_map.weights:
        return np.asarray(region_map.weights[name], float), "AUTO_GEOMETRY"
    raise RegionMapError("REGION_UNAVAILABLE:" + str(name), "该语义区域既没有人工标注也没有自动几何")
