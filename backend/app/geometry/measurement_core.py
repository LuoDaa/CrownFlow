"""几何测量核心：三个域（Contact / Occlusion / Thickness）**共用 primitive，不共用 DTO**。

B1（2026-10-12）的结论：这三个域的"符号"和"有效性"**本来就不是同一个量**，强行塞进一个
DTO 只会制造新的"看起来统一、实际口径不同"的假象。所以这里只提供：

1. **唯一的口径词汇表**（``SIGN_*`` / ``VALIDITY_*``）—— 报告里必须写清楚用的是哪一种；
2. **共用的数值 primitive**（``stats`` / ``validity_counts``）—— 统计形状不再各写一遍；
3. **领域契约表** ``DOMAIN_CONTRACTS`` —— 一张表回答"谁负责相交、谁是目标相对量、
   输入契约是什么"，避免跨域误读（例如把 occlusion 的"亏欠"当 contact 的"穿透深度"）。

三个域各自的 DTO 仍然独立：
``services.contact_measurement``（Contact）、``crown_outer.occlusion``（Occlusion）、
``crown_outer.thickness``（Thickness）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 口径词汇表
#: **与 Contact 共享的实体内外判定**（O1 之后 Contact 与 Occlusion 都用这一个实现）
SIGN_SOLID_CONTAINMENT = "SOLID_CONTAINMENT"
#: **与 Contact 共享的已复核外向法向**（开放裁片；调用方声明 → 局部可信）
SIGN_REVIEWED_NORMAL_LOCAL = "REVIEWED_NORMAL_LOCAL"
#: Thickness 自己的：**平滑法向**（最近面三顶点的面积加权法向均值）
SIGN_SMOOTHED_VERTEX_NORMAL = "SMOOTHED_VERTEX_NORMAL"
#: 符号不可得（只给幅值，拒绝穿透结论）
SIGN_UNSIGNED_ONLY = "UNSIGNED_ONLY"
#: 兼容别名（Contact 服务层 2026-10-12 之前用的名字；同为一种东西，不要再新增名字）
SIGN_GLOBAL_INSIDE_OUTSIDE = SIGN_SOLID_CONTAINMENT
SIGN_REVIEWED_LOCAL_NORMAL = SIGN_REVIEWED_NORMAL_LOCAL

#: 接触点附近的**符号歧义带**（mm）：d < ε 时 inside/outside 在浮点意义上不可判，
#: 强行取符号会得到 ±1e-12 级别的随机翻号，所以判为歧义、不进决策集。
DEFAULT_CONTAINMENT_EPSILON_MM = 1e-06

#: 有效性约定（决定"哪些采样允许参与判定"）
VALIDITY_BOUNDARY_AND_CREASE = "BOUNDARY_AND_CREASE"   # Contact：边界 + 折痕
VALIDITY_BOUNDARY_ONLY = "BOUNDARY_ONLY"               # Occlusion / Thickness：仅边界安全
VALIDITY_NONE = "NONE"                                 # 不设限（闭合实体判定时）

DOMAIN_CONTACT = "CONTACT"
DOMAIN_OCCLUSION = "OCCLUSION"
DOMAIN_THICKNESS = "THICKNESS"


def stats(values) -> dict | None:
    """共用统计形状：``p50/p95/max``（mm）。空输入 → ``None``（不假装有数）。"""
    values = np.asarray(values, float)
    if not len(values):
        return None
    return {"p50_mm": float(np.quantile(values, .5)),
            "p95_mm": float(np.quantile(values, .95)),
            "max_mm": float(values.max())}


def validity_counts(valid) -> dict:
    """共用有效性账：有效 / 歧义（无效）条数。"""
    valid = np.asarray(valid, bool)
    return {"valid_count": int(valid.sum()), "ambiguous_count": int((~valid).sum())}


# ---------------------------------------------------------------- 共享符号 primitive
def is_closed_oriented_solid(mesh) -> bool:
    """闭合、绕向一致、体积为正 ⇒ 实体内外判定在几何上有意义。"""
    try:
        return bool(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0)
    except Exception:                                              # noqa: BLE001
        return False


@dataclass
class SignSource:
    """**全项目唯一的符号 primitive**（O1）。

    Contact（`FitSurface`）与 Occlusion 都必须经由它取符号，规则只有三条：

    ==========================  ===========================================
    ``method``                  符号
    ==========================  ===========================================
    ``SOLID_CONTAINMENT``       闭合实体：``-d``（在内部）/ ``+d``（在外部）
    ``REVIEWED_NORMAL_LOCAL``   开放裁片 + **调用方声明**的外向法向：局部可信
    ``UNSIGNED_ONLY``           两者都没有：**不提供 signed**，只有幅值
    ==========================  ===========================================

    ``query()`` 返回 ``(closest, face_ids, signed, distance, valid)`` —— 与
    `NeighborPatch.clearance_detail` / `FitSurface.query` 同一个形状；
    ``UNSIGNED_ONLY`` 时 ``signed`` 全为 ``nan``（**不是 0**：0 会被误读成"贴面"）。
    ``d < epsilon_mm`` 的点判为歧义（``valid=False``），避免浮点级随机翻号。
    """

    method: str
    mesh: object = None
    patch: object = None
    outward: object = None
    epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM

    # ---------------------------------------------------------- 构造
    @classmethod
    def for_mesh(cls, mesh, outward=None, *, epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM):
        from .crown_outer.contact import NeighborPatch
        patch = NeighborPatch(np.asarray(mesh.vertices), np.asarray(mesh.faces))
        if is_closed_oriented_solid(mesh):
            return cls(SIGN_SOLID_CONTAINMENT, mesh=mesh, patch=patch, epsilon_mm=epsilon_mm)
        normal = _unit(outward)
        if normal is None:
            return cls(SIGN_UNSIGNED_ONLY, mesh=mesh, patch=patch, epsilon_mm=epsilon_mm)
        patch.normal_sign = 1 if float(np.median(patch.normals @ normal)) > 0 else -1
        return cls(SIGN_REVIEWED_NORMAL_LOCAL, mesh=mesh, patch=patch, outward=normal,
                   epsilon_mm=epsilon_mm)

    @classmethod
    def for_patch(cls, patch, outward=None, *, epsilon_mm: float = DEFAULT_CONTAINMENT_EPSILON_MM):
        """从一个 `NeighborPatch` 构造；**只有调用方声明了外向法向才算已复核**。"""
        import trimesh
        mesh = trimesh.Trimesh(np.asarray(patch.vertices), np.asarray(patch.faces), process=False)
        return cls.for_mesh(mesh, outward, epsilon_mm=epsilon_mm)

    @property
    def signed_available(self) -> bool:
        return self.method != SIGN_UNSIGNED_ONLY

    # ---------------------------------------------------------- 查询
    def query(self, points):
        points = np.asarray(points, float)
        if points.ndim == 1:
            points = points[None, :]
        boundary = 0.0 if self.method == SIGN_SOLID_CONTAINMENT else 0.5
        cp, fids, signed, distance, valid = self.patch.closest_with_validity(
            points, boundary_safety_mm=boundary)
        distance = np.asarray(distance, float)
        valid = np.asarray(valid, bool)
        if self.method == SIGN_SOLID_CONTAINMENT:
            signed = np.where(self.mesh.contains(points), -distance, distance)
            valid = np.ones(len(points), bool)
        elif self.method == SIGN_REVIEWED_NORMAL_LOCAL:
            normals = self.patch.normals[fids] * self.patch.normal_sign
            valid = valid & ((normals @ np.asarray(self.outward, float)) > 0.5)
        else:
            signed = np.full(len(points), np.nan)
        # 接触点歧义带：d < ε 不判内外（两种方法都适用 —— 贴面点在浮点上没有方向）
        ambiguous = distance < float(self.epsilon_mm)
        valid = valid & (~ambiguous)
        return cp, fids, np.asarray(signed, float), distance, valid

    def ambiguous_mask(self, points) -> np.ndarray:
        _cp, _fids, _signed, distance, _valid = self.query(points)
        return distance < float(self.epsilon_mm)

    def as_report(self) -> dict:
        return {"sign_method": self.method, "signed_available": self.signed_available,
                "epsilon_mm": float(self.epsilon_mm)}


def _unit(vector):
    if vector is None:
        return None
    array = np.asarray(vector, float)
    norm = float(np.linalg.norm(array))
    if array.shape != (3,) or not np.isfinite(array).all() or norm < 1e-8:
        return None
    return array / norm


@dataclass(frozen=True)
class DomainContract:
    """一个域的测量契约（写进报告，也给测试当"单一真相"）。"""

    domain: str
    sign_method: str
    sign_semantics: str
    validity: str
    penetration_meaning: str
    owns_crossing_audit: bool
    input_contract: str
    implementation: str

    def as_report(self) -> dict:
        return {"domain": self.domain, "sign_method": self.sign_method,
                "sign_semantics": self.sign_semantics, "validity_convention": self.validity,
                "penetration_meaning": self.penetration_meaning,
                "owns_crossing_audit": self.owns_crossing_audit,
                "input_contract": self.input_contract, "implementation": self.implementation}


CONTACT_CONTRACT = DomainContract(
    domain=DOMAIN_CONTACT,
    sign_method=f"{SIGN_GLOBAL_INSIDE_OUTSIDE} | {SIGN_REVIEWED_LOCAL_NORMAL} | {SIGN_UNSIGNED_ONLY}",
    sign_semantics="positive=gap_to_reference, negative=penetration_into_reference, |signed|=euclidean_distance",
    validity=VALIDITY_BOUNDARY_AND_CREASE,
    penetration_meaning="max(0, -signed) = 穿进参考面的深度（正值 = 穿透深度）",
    owns_crossing_audit=False,
    input_contract="参考面为闭合有向实体（实体内外）或已确认外向法向的开放裁片",
    implementation="services.contact_measurement.measure_surface",
)

OCCLUSION_CONTRACT = DomainContract(
    domain=DOMAIN_OCCLUSION,
    #: O1（2026-10-13）：**一次性质心投票已退役** —— 闭合对颌改用与 Contact 相同的
    #: 实体内外判定；开放裁片必须由调用方声明外向法向；两者都没有 ⇒ UNSIGNED_ONLY。
    sign_method=f"{SIGN_SOLID_CONTAINMENT} | {SIGN_REVIEWED_NORMAL_LOCAL} | {SIGN_UNSIGNED_ONLY}",
    sign_semantics="signed<0 = 穿进对颌；报告量 = max(target_gap - signed, 0)（**目标相对亏欠**）",
    validity=VALIDITY_BOUNDARY_ONLY,
    penetration_meaning="target_gap - signed（含目标间隙；不是纯穿透深度）",
    owns_crossing_audit=False,
    input_contract="冠网格必须有可识别颈环（锁定区依赖它）；对颌须为闭合实体或已声明外向法向的裁片",
    implementation="crown_outer.occlusion.solve_occlusion_collision",
)

THICKNESS_CONTRACT = DomainContract(
    domain=DOMAIN_THICKNESS,
    sign_method=SIGN_SMOOTHED_VERTEX_NORMAL,
    sign_semantics="signed = 到最近面的有符号距离（平滑法向）；冲突量 = signed - target（负 = 与约束冲突）",
    validity=VALIDITY_BOUNDARY_ONLY,
    penetration_meaning="signed - target（相对约束目标的余量；负 = 冲突）",
    owns_crossing_audit=True,
    input_contract="外表面必须带开放颈缘（否则无法定位 margin）；内表面为配合面",
    implementation="crown_outer.thickness.audit_thickness",
)

#: 域 → 契约。跨域读数字前先看这张表。
DOMAIN_CONTRACTS: dict[str, DomainContract] = {
    DOMAIN_CONTACT: CONTACT_CONTRACT,
    DOMAIN_OCCLUSION: OCCLUSION_CONTRACT,
    DOMAIN_THICKNESS: THICKNESS_CONTRACT,
}


def domain_report(domain: str, **extra) -> dict:
    """给某个域的报告打上契约标签（报告里必须能看出这是哪个域的测量）。"""
    contract = DOMAIN_CONTRACTS[domain]
    return {**contract.as_report(), **extra}


@dataclass
class DomainMeasurement:
    """三个域共用的**标注容器**（不是共用的 DTO：各自的值域/键由域自己决定）。

    ``values`` 放该域自己的量；``as_report()`` 把契约标签与它并排放进报告。
    """

    contract: DomainContract
    values: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def as_report(self) -> dict:
        return {**self.contract.as_report(), "values": dict(self.values),
                "notes": list(self.notes)}
