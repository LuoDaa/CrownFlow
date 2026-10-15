"""HPS 扫描的可解码性判定 —— **三态，不允许第四种**。

## 实测结论（`测试_胡昭`，两份扫描）

格式是 HPS 1.1，`<Schema>CE</Schema>`。`CE` 在
``hpsdecode.SUPPORTED_SCHEMAS == ('CA','CB','CC','CE')`` 里，即**格式支持**，
但 hpsdecode 自己的文档写着：

    The CE schema is CC with Blowfish encryption on vertex data

实测两处独立证据印证了这一点：

| 载荷 | 大小 | 熵 (bit/byte) | 判读 |
|---|---|---|---|
| `<Facets>` 连通性 | 295,901 B / 260,445 面 | **2.29** | 明文（CC 压缩） |
| `<Vertices>` 顶点 | 1,575,648 B / 131,304 点 | **7.9999** | **Blowfish 加密** |

把顶点载荷按小端 float32×3 解释，得到 `-inf … +inf`（值域 1e25–1e37）并含 NaN ——
**不是坐标**。而 `1,575,648 = 131,304 × 12` 恰好是未加密时 float32×3 的大小。

文件里 `EKID="1"` 记录了使用的密钥 ID，**密钥不在包内**。
所以顶点坐标拿不到，mesh 也就拿不到。

**因此三态结果是 `SUPPORTED_BUT_ENCRYPTED`，不是 `UNSUPPORTED`** ——
这个区别有实际意义：不是解析器不支持，而是缺合法密钥。换一份未加密导出的
HPS（或导出 STL）就能走通，解析器不用改。

## 不做的事

不尝试提取或绕过 3Shape 的 CE 密钥。理由：
1. 这是对厂商技术保护措施的规避，有法律与授权风险；
2. 与项目一贯原则冲突 —— 我们在 `.matrix4`、`.dentalCAD`、59 个加密 DCM 上
   都是"读不出就读不出，明确报错"，从未反向工程厂商格式；
3. 没必要：操作员在 3Shape 里导出 STL 只需一步。
"""

from __future__ import annotations

import base64
import collections
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .models import ScanDecodeStatus

#: 加密与压缩的熵阈值（bit/byte）。实测明文连通性 2.29、加密顶点 7.9999。
ENTROPY_ENCRYPTED_MIN = 7.5

#: 这些 schema 一律需要密钥（hpsdecode 的 ENCRYPTED_SCHEMAS）。
ENCRYPTED_SCHEMAS: frozenset[str] = frozenset({"CE"})
SUPPORTED_SCHEMAS: frozenset[str] = frozenset({"CA", "CB", "CC", "CE"})


def _tag_attr(text: str, tag: str) -> dict[str, str]:
    m = re.search(rf"<{tag}\s+([^>]*?)/?>", text)
    if not m:
        return {}
    return dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))


def _b64_payload(text: str, tag: str) -> bytes | None:
    m = re.search(rf"<{tag}\s+[^>]*?>(.*?)</{tag}>", text, re.S)
    if not m:
        return None
    s = re.sub(r"\s", "", m.group(1))
    if not s:
        return None
    return base64.b64decode(s + "=" * (-len(s) % 4))


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = collections.Counter(data)
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def plaintext_metadata(path: Path | str) -> dict[str, Any]:
    """抽出**明文可读**的部分：容器版本、schema、面/点计数、密钥 ID、来源。

    这些字段即使几何被加密也能拿到，是"包里确实有东西"的证据，
    也让上层能给出"缺什么"的明确提示而不用去解开几何。
    """
    p = Path(path)
    text = p.read_bytes().decode("utf-8", "ignore")

    schema_m = re.search(r"<Schema>([^<]*)</Schema>", text)
    facets_attr = _tag_attr(text, "Facets")
    verts_attr = _tag_attr(text, "Vertices")
    props = dict(re.findall(r'<Property\s+name="([^"]+)"\s+value="([^"]*)"\s*/?>', text))

    meta: dict[str, Any] = {
        "container_version": (re.search(r'<HPS version="([^"]*)"', text) or [None, None])[1],
        "schema": schema_m.group(1) if schema_m else None,
        "facet_count": int(facets_attr["facet_count"]) if "facet_count" in facets_attr else None,
        "vertex_count": int(verts_attr["vertex_count"]) if "vertex_count" in verts_attr else None,
        "signature_hash": (re.search(r"<SignatureHash>([^<]*)</SignatureHash>", text) or [None, None])[1],
        "encryption_key_id": props.get("EKID"),
        "scan_source": props.get("ScanSource"),
        "source_app": props.get("SourceApp"),
        "file_size_bytes": p.stat().st_size,
    }

    # 加密证据：对顶点载荷算熵，并试按 float32 解释
    payload = _b64_payload(text, "Vertices")
    if payload:
        meta["vertex_payload_bytes"] = len(payload)
        meta["vertex_payload_entropy"] = round(_entropy(payload), 4)
        if len(payload) % 12 == 0 and len(payload) > 0:
            f = np.frombuffer(payload, dtype="<f4").reshape(-1, 3)
            finite = np.isfinite(f).all(axis=1)
            meta["vertex_as_float32_finite_ratio"] = round(float(finite.mean()), 4)
            meta["vertex_looks_like_coordinates"] = bool(
                finite.mean() > 0.999 and len(f) > 0
                and float(np.abs(f[finite]).max()) < 1e3)
    facets_payload = _b64_payload(text, "Facets")
    if facets_payload:
        meta["facet_payload_bytes"] = len(facets_payload)
        meta["facet_payload_entropy"] = round(_entropy(facets_payload), 4)
    return meta


def classify_scan(path: Path | str, *, mesh_loader=None) -> tuple[ScanDecodeStatus, dict, Any, str | None]:
    """判定单个扫描的可解码状态。

    返回 ``(status, plaintext_meta, mesh_or_None, error)``。

    ``mesh_loader`` 供测试注入；默认走项目的 ``app.geometry.io.load_mesh``，
    与上传路径共用同一套解析与报错口径。
    """
    meta = plaintext_metadata(path)
    schema = (meta.get("schema") or "").upper()

    if schema not in SUPPORTED_SCHEMAS:
        return (ScanDecodeStatus.UNSUPPORTED, meta, None,
                f"schema {schema!r} 不在受支持集合 {sorted(SUPPORTED_SCHEMAS)} 内")

    if mesh_loader is None:
        from ...geometry.io import load_mesh
        mesh_loader = load_mesh

    try:
        mesh = mesh_loader(Path(path))
        return ScanDecodeStatus.DECODED, meta, mesh, None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if schema in ENCRYPTED_SCHEMAS:
            # CE 且无合法密钥 -> 支持该格式但缺密钥。
            # 注意不是 UNSUPPORTED：解析器不用改，换一份未加密导出即可走通。
            return ScanDecodeStatus.SUPPORTED_BUT_ENCRYPTED, meta, None, message
        return ScanDecodeStatus.UNSUPPORTED, meta, None, message


def classify_companion_mesh(
    mesh_path: Path | str,
    *,
    logical_suffix: str | None = None,
    mesh_loader=None,
) -> tuple[ScanDecodeStatus, dict, Any, str | None]:
    """判定 companion mesh（STL/PLY/OBJ）。

    **不能复用 :func:`classify_scan`** —— 它先解析 HPS 外壳找 ``<Schema>``，
    对二进制 STL 会得到空 schema 并直接判 UNSUPPORTED，于是永远不会去加载网格。

    注意 ``logical_suffix`` 必须由调用方传入：包内物理文件名是 GUID、没有扩展名，
    无法从路径推断格式。
    """
    meta: dict[str, Any] = {
        "kind": "companion_mesh",
        "logical_suffix": logical_suffix,
        "file_size_bytes": Path(mesh_path).stat().st_size,
    }
    try:
        if mesh_loader is not None:
            mesh = mesh_loader(Path(mesh_path))
        else:
            mesh = _load_mesh_by_logical_suffix(mesh_path, logical_suffix)
        if mesh is None or len(mesh.faces) == 0:
            raise ValueError("不是有效的三角网格")
        meta["vertices"] = int(len(mesh.vertices))
        meta["faces"] = int(len(mesh.faces))
        return ScanDecodeStatus.DECODED, meta, mesh, None
    except Exception as exc:
        return ScanDecodeStatus.UNSUPPORTED, meta, None, f"{type(exc).__name__}: {exc}"


def _load_mesh_by_logical_suffix(mesh_path: Path | str, logical_suffix: str | None):
    """按**逻辑后缀**加载网格。

    不能走 ``load_mesh``：它让 trimesh 按扩展名选加载器，而包内物理文件名是
    GUID、**没有扩展名**，会报 ``NotImplementedError: file_type '...{A6}' not supported``。
    这是同一个「GUID 名不带扩展名」问题在本项目里第三次出现
    （前两次：类型判定、STL 写出）。所以这里显式读字节 + 显式 file_type。
    """
    import io

    import trimesh

    suffix = (logical_suffix or "").lower().lstrip(".")
    if not suffix:
        raise ValueError("companion mesh 缺少逻辑后缀，无法确定格式")
    data = Path(mesh_path).read_bytes()
    obj = trimesh.load(io.BytesIO(data), file_type=suffix, force="mesh", process=True)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError(f"不是单一三角网格：{type(obj).__name__}")
    return obj
