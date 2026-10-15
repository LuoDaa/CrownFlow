"""颈环相似摆位与冠体缩放衰减。

周长比只完全作用于颈环；冠体沿局部咬合轴逐渐回到原始尺寸。
``Placement.matrix`` 只描述颈环相似变换，不能表示整个非线性冠体变换。

    R  = Rp · Rt^T        （列空间：模板局部轴 → 患者局部轴）
    s0 = perimeter(patient_margin) / perimeter(template_cervical_loop)
    V' = s0 · R · (V − t_origin) + p_origin

行/列向量约定与项目统一：``V' = s0 · ((V − t_origin) @ R^T) + p_origin``。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patient_frame import PatientFrame
from .template import CrownTemplate

#: 旋转矩阵行列式偏离 +1 的容差。
DET_TOL = 1e-6


@dataclass
class Placement:
    rotation: np.ndarray      # (3, 3)
    translation: np.ndarray   # (3,)
    scale: float
    #: **H.2 各向异性缩放**：模板局部系的 (s_x, s_y, s_z)，x=近远中、
    #: y=颊舌、z=咬合轴。None = 各向同性（`scale`），H.1 行为不变。
    scale_vec: np.ndarray | None = None

    def svec(self) -> np.ndarray:
        if self.scale_vec is None:
            return np.full(3, float(self.scale))
        return np.asarray(self.scale_vec, dtype=np.float64)

    def matrix(self) -> np.ndarray:
        """4×4 齐次矩阵（先缩放后旋转平移：p' = S R p + t，S 可各向异性）。"""
        m = np.eye(4)
        m[:3, :3] = self.rotation @ np.diag(self.svec())
        m[:3, 3] = self.translation
        return m

    def apply(self, vertices: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=np.float64)
        return (v * self.svec()[None, :]) @ self.rotation.T + self.translation

    def apply_body_decoupled(self, vertices: np.ndarray,
                             margin_indices: np.ndarray,
                             crown_height_mm: float,
                             md_body_keep: float | None = None) -> np.ndarray:
        """颈环按缩放比缩放；冠部沿局部咬合轴线性回到原始尺寸。

        各向异性时逐轴衰减：``s(weight) = 1 + weight·(s_axis − 1)``
        （`scale_vec` 为 None 时退化成 H.1 的标量行为，逐位相同）。

        ``md_body_keep``（**H.2**）：近远中轴（局部 X）在**冠部顶**保留多少
        缩放量。H.1 的默认衰减在冠顶回到 0 → 冠体完全是模板原尺寸，
        近远中收窄根本传不到中段（实测 s_md=0.85 只让半宽动 0.18mm、
        穿透不动）。H.2 要求"颈缘对齐 margin、中段按邻牙收窄、咬合区别压扁
        cusp"，所以 MD 轴用更平缓的剖面：``w = 1 − (1−keep)·t``。
        """
        v = np.asarray(vertices, dtype=np.float64)
        if crown_height_mm <= 0:
            raise ValueError("模板冠高必须大于零")
        z0 = float(v[margin_indices, 2].mean())
        t = np.clip((v[:, 2] - z0) / crown_height_mm, 0.0, 1.0)
        weight = 1.0 - t
        weight[margin_indices] = 1.0
        sv = self.svec()
        local_scale = 1.0 + weight[:, None] * (sv[None, :] - 1.0)
        if md_body_keep is not None and self.scale_vec is not None:
            keep = float(np.clip(md_body_keep, 0.0, 1.0))
            w_md = 1.0 - (1.0 - keep) * t
            w_md[margin_indices] = 1.0
            local_scale[:, 0] = 1.0 + w_md * (sv[0] - 1.0)
        return (v * local_scale) @ self.rotation.T + self.translation


def rotation_between_frames(t_x, t_y, t_z, p_x, p_y, p_z) -> np.ndarray:
    """把模板局部轴 (tx,ty,tz) 转到患者局部轴 (px,py,pz) 的旋转。

    两套基都必须是严格正交右手系（det≈+1），否则 fail-loud ——
    左手基混进相似变换会静默引入镜像，冠会左右反戴。
    """
    def _check(m):
        det = float(np.linalg.det(m))
        if abs(det - 1.0) > 1e-6:
            raise ValueError("局部基不是右手系（det=%.6f）" % det)
        err = float(np.abs(m.T @ m - np.eye(3)).max())
        if err > 1e-6:
            raise ValueError("局部基不正交（误差 %.3g）" % err)

    _check(np.stack([t_x, t_y, t_z], axis=1))
    _check(np.stack([p_x, p_y, p_z], axis=1))
    t = np.stack([t_x, t_y, t_z], axis=1)
    p = np.stack([p_x, p_y, p_z], axis=1)
    return p @ t.T


def template_axes(template: CrownTemplate) -> tuple[np.ndarray, ...]:
    """模板局部轴：来自牙库 XML 的牙位局部系（Z=AxisOcclusal、X=AxisMesial）。

    模板顶点已表达在该系里且原点移到颈环质心，所以这里只要返回
    单位轴本身。X/Y 符号二义由边界对应阶段吸收。
    """
    z = template.occlusal_direction
    # occlusal_direction 是冠部质心方向，接近 +Z 但不严格；摆位用严格的
    # 牙位局部 Z —— 从顶点云重新正交化会引入任意性，直接用 z 轴本身。
    z = np.array([0.0, 0.0, 1.0])
    x = np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return x, y, z


def place_template(template: CrownTemplate, patient_frame: PatientFrame,
                   patient_margin_perimeter_mm: float) -> Placement:
    """相似摆位：模板颈环 → 患者颈缘，周长比例定均匀缩放。"""
    t_x, t_y, t_z = template_axes(template)
    r = rotation_between_frames(t_x, t_y, t_z,
                                patient_frame.x, patient_frame.y,
                                patient_frame.z)
    scale = float(patient_margin_perimeter_mm / template.margin_perimeter_mm)
    if not (0.2 < scale < 5.0):
        raise ValueError(
            "均匀缩放比 %.3f 超出合理范围 —— 模板与患者 Margin 尺寸差了一个量级，"
            "先怀疑 FDI/镜像映射错误" % scale)
    return Placement(rotation=r, translation=patient_frame.origin.copy(),
                     scale=scale)


def patient_margin_perimeter(margin_points: np.ndarray) -> float:
    """有序闭环的周长。"""
    p = np.asarray(margin_points, dtype=np.float64)
    seg = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return float(seg.sum())
