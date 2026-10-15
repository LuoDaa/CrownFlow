"""GroundTruthIndexer：把 exocad 生产数据的文件堆变成可复现的「病例 × FDI」真值索引。

    python -m app.groundtruth.indexer --root <病例根> --out <输出目录>

**纯 stdlib**（数据在服务器上，那台机器没有 numpy/trimesh，索引器必须能就地跑）。
只做索引，不改任何几何算法、不训练模型。
"""

from .models import (
    COLUMNS,
    GT_SETS,
    RESTORATIVE_TYPES,
    ToothGroundTruth,
    build_all_sets,
    case_group_key,
    is_margin_valid,
)

__all__ = [
    "COLUMNS",
    "GT_SETS",
    "RESTORATIVE_TYPES",
    "ToothGroundTruth",
    "build_all_sets",
    "case_group_key",
    "is_margin_valid",
]
