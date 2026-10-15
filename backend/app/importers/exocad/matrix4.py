"""``.matrix4`` 解析与角色判定。

**结论先行（由实测数据支撑，不是推断）：``.matrix4`` 不是逐病例咬合配准矩阵。**

证据：`数据/15个完整病例` 下的 15 个 ``.matrix4`` 文件内容**逐字节完全相同**
（15 个文件只有 1 个内容 hash），平移非零但固定、旋转 176.5°、``det = +1``。
逐病例的咬合配准不可能在 15 位不同患者上完全一致。

因此本模块：
- 通过 content hash 分组来**判定**角色，而不是假设；
- 只有当矩阵**逐病例不同**时才认为它是咬合候选，且仍然需要外部空间验证
  （应用后检查上下颌是否形成合理咬合）才能升级为 ``BITE_TRANSFORM``；
- ``FIXED_COORDINATE_TRANSFORM`` 永远不得驱动 Occlusion 引擎。

矩阵为行向量约定：``p' = p @ R + t``，其中 ``t = M[3, :3]``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import xml.etree.ElementTree as ET


class Matrix4Role(str, Enum):
    """``.matrix4`` 在流程中的角色。只有 ``BITE_TRANSFORM`` 可以驱动 Occlusion。"""

    #: 全病例内容一致 —— 坐标系约定变换。**禁止**用于 Occlusion。
    FIXED_COORDINATE_TRANSFORM = "FIXED_COORDINATE_TRANSFORM"

    #: 逐病例不同，但尚未通过空间验证。**禁止**在验证前用于 Occlusion。
    CANDIDATE_BITE_TRANSFORM_UNVERIFIED = "CANDIDATE_BITE_TRANSFORM_UNVERIFIED"

    #: 逐病例不同且已通过空间验证（上下颌形成合理咬合）。可用于 Occlusion。
    BITE_TRANSFORM = "BITE_TRANSFORM"

    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Matrix4:
    source_file: str
    matrix: np.ndarray

    @property
    def rotation(self) -> np.ndarray:
        return self.matrix[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """行向量约定下平移在最后一行。"""
        return self.matrix[3, :3]

    @property
    def rotation_angle_deg(self) -> float:
        trace = float(np.trace(self.rotation))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))))

    def apply(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        return pts @ self.rotation + self.translation

    def content_hash(self) -> str:
        return hashlib.sha256(np.round(self.matrix, 12).tobytes()).hexdigest()[:16]


def parse_matrix4(path: Path | str) -> Matrix4:
    """解析 exocad 的 ``<Matrix4>`` XML（``<_00>`` … ``<_33>``）。"""
    path = Path(path)
    root = ET.parse(path).getroot()
    values = []
    for i in range(4):
        for j in range(4):
            tag = f"_{i}{j}"
            text = root.findtext(tag)
            if text is None:
                raise ValueError(f"{path.name} 缺少 <{tag}>，不是完整的 4x4 矩阵")
            values.append(float(text))
    return Matrix4(source_file=str(path), matrix=np.array(values).reshape(4, 4))


@dataclass
class Matrix4RoleVerdict:
    role: Matrix4Role
    reason: str
    distinct_content_hashes: int
    n_files: int
    rotation_angle_deg: float | None = None
    groups: dict[str, list[str]] = field(default_factory=dict)

    @property
    def usable_for_occlusion(self) -> bool:
        return self.role is Matrix4Role.BITE_TRANSFORM


def classify_matrix4_role(matrices: list[Matrix4]) -> Matrix4RoleVerdict:
    """根据一批 ``.matrix4`` 的内容一致性判定其角色。

    这是**判定**，不是假设。调用方必须使用返回的 role，不得自行把它当作咬合矩阵。
    """
    if not matrices:
        return Matrix4RoleVerdict(
            role=Matrix4Role.UNKNOWN,
            reason="没有提供 .matrix4 文件",
            distinct_content_hashes=0,
            n_files=0,
        )

    groups: dict[str, list[str]] = {}
    for m in matrices:
        groups.setdefault(m.content_hash(), []).append(Path(m.source_file).parent.name)

    if len(groups) == 1:
        angle = matrices[0].rotation_angle_deg
        return Matrix4RoleVerdict(
            role=Matrix4Role.FIXED_COORDINATE_TRANSFORM,
            reason=(
                f"{len(matrices)} 个文件内容完全相同（1 个内容 hash），旋转 {angle:.1f}°。"
                "逐病例的咬合配准不可能在不同患者间完全一致，因此这是坐标系约定变换。"
                "禁止用于 Occlusion。"
            ),
            distinct_content_hashes=1,
            n_files=len(matrices),
            rotation_angle_deg=angle,
            groups=groups,
        )

    return Matrix4RoleVerdict(
        role=Matrix4Role.CANDIDATE_BITE_TRANSFORM_UNVERIFIED,
        reason=(
            f"{len(matrices)} 个文件分属 {len(groups)} 个不同内容，逐病例不同 → 可能是咬合相关。"
            "但在通过空间验证（应用后上下颌应形成合理咬合：对颌距离、牙弓对齐、包围盒、"
            "咬合面朝向）之前，不得标记为 BITE_TRANSFORM，也不得用于 Occlusion。"
        ),
        distinct_content_hashes=len(groups),
        n_files=len(matrices),
        groups=groups,
    )
