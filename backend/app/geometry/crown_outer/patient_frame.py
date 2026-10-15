"""患者牙位局部坐标系（Patient Frame，方案 §6）。

**不依赖世界 XYZ**。以 Margin + Insertion Axis 为核心：

    Origin = Patient Margin 的 3D 质心
    Z      = normalize(Insertion Axis)
    X      = Margin 投影到 ⊥Z 平面后的 PCA 长主轴（去 Z 分量、归一化）
    Y      = normalize(Z × X)，再用 X = Y × Z 重算，得严格正交右手基

X/Y 的**正负方向允许二义**（第一阶段没有邻牙信息，不推断解剖近中/远中
绝对语义）；边界对应阶段同时测试 forward/reverse + cyclic shift 来吸收
这个二义，而不是在这里猜。

Margin 投影 PCA 的目的不是推断解剖语义，而是建立稳定横向主轴，
使初始摆放在 rigid transform 下保持等变。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PatientFrameError(ValueError):
    pass


@dataclass
class PatientFrame:
    origin: np.ndarray        # (3,)
    z: np.ndarray             # (3,) 单位向量 = insertion axis
    x: np.ndarray             # (3,) 单位向量
    y: np.ndarray             # (3,) 单位向量
    #: 该帧被人工确认时对应的修订 id（A0：报告里要能追溯到"哪一次确认"）。
    revision_id: str | None = None

    def matrix(self) -> np.ndarray:
        """列 = (X, Y, Z) 的旋转矩阵（右手系，det≈+1）。"""
        return np.stack([self.x, self.y, self.z], axis=1)


def validate_patient_frame(frame: "PatientFrame", *, tol: float = 1e-6) -> None:
    """A0 纪律：确认帧必须是**右手正交**基，否则拒绝（不许静默正交化）。

    检查 |x|=|y|=|z|=1、两两点积 ≈0、det>0（x×y≈z）。
    """
    for name, v in (("x", frame.x), ("y", frame.y), ("z", frame.z)):
        arr = np.asarray(v, dtype=float)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise PatientFrameError("帧的 %s 不是有效三维向量" % name)
        if abs(float(np.linalg.norm(arr)) - 1.0) > tol:
            raise PatientFrameError("帧的 %s 不是单位向量" % name)
    x, y, z = (np.asarray(v, dtype=float) for v in (frame.x, frame.y, frame.z))
    if abs(float(x @ y)) > 1e-6 or abs(float(x @ z)) > 1e-6 or abs(float(y @ z)) > 1e-6:
        raise PatientFrameError("帧的 X/Y/Z 不两两正交，拒绝使用")
    if float(np.linalg.det(np.stack([x, y, z], axis=1))) <= 0:
        raise PatientFrameError("帧不是右手系（x×y≠z），拒绝使用")
    origin = np.asarray(frame.origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise PatientFrameError("帧原点不是有效三维点")


def confirmed_patient_frame(origin, mesial, buccal, insertion,
                            *, revision_id: str | None = None) -> "PatientFrame":
    """由**人工确认**的近中/颊侧/就位方向构造帧（A0 唯一入口）。

    只做归一化 + 右手性校验；**不做** world-X 推断、不静默正交化。
    """
    z = np.asarray(insertion, dtype=float)
    x = np.asarray(mesial, dtype=float)
    y = np.asarray(buccal, dtype=float)
    nz, nx, ny = (float(np.linalg.norm(v)) for v in (z, x, y))
    for name, n in (("mesial", nx), ("buccal", ny), ("insertion", nz)):
        if n < 1e-9 or not np.isfinite(n):
            raise PatientFrameError("确认帧的 %s 方向退化" % name)
    frame = PatientFrame(origin=np.asarray(origin, dtype=float), z=z / nz, x=x / nx, y=y / ny,
                         revision_id=revision_id)
    validate_patient_frame(frame)
    return frame


def build_patient_frame(margin_points: np.ndarray, insertion_axis: np.ndarray
                        ) -> PatientFrame:
    margin = np.asarray(margin_points, dtype=np.float64)
    z = np.asarray(insertion_axis, dtype=np.float64)
    nz = float(np.linalg.norm(z))
    if nz < 1e-9:
        raise PatientFrameError("insertion_axis 是零向量")
    z = z / nz

    origin = margin.mean(axis=0)

    # 投影到 ⊥Z 平面（过 origin）
    rel = margin - origin
    proj = rel - np.outer(rel @ z, z)

    # 退化检查：投影面积过小 = Margin 在轴向上塌成一线，PCA 无意义
    area_scale = float(np.sqrt(np.mean(np.sum(proj * proj, axis=1))))
    if area_scale < 1e-6:
        raise PatientFrameError(
            "Margin 投影到 ⊥axis 平面后退化（rms %.3g）—— 不构成可用闭环"
            % area_scale)

    # 平面内 PCA：长主轴为 X 候选。
    # 注意 cov 是 3D 的，但 proj ⊥ z，所以最小特征值恒 ≈ 0（z 方向）——
    # 共线判据必须比较**面内的两个**特征值（次小 vs 最大）。
    cov = proj.T @ proj / max(len(proj), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)          # 升序
    if eigvals[2] <= 1e-12:
        raise PatientFrameError("Margin 投影 PCA 奇异（重合点？）")
    if eigvals[1] < 1e-6 * eigvals[2]:
        raise PatientFrameError(
            "Margin 投影近似共线（面内长短轴特征值比 %.3g）—— "
            "无法建立稳定横向轴" % (eigvals[1] / eigvals[2]))
    x = eigvecs[:, -1]
    x = x - (x @ z) * z                              # 去 Z 分量
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        raise PatientFrameError("PCA 主轴与 insertion axis 平行")
    x = x / nx
    y = np.cross(z, x)
    ny = float(np.linalg.norm(y))
    if ny < 1e-9:
        raise PatientFrameError("Z×X 退化")
    y = y / ny
    x = np.cross(y, z)                               # 严格正交回算
    return PatientFrame(origin=origin, z=z, x=x, y=y)
