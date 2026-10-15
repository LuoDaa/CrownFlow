"""AliasFilenames.xml -> GUID 到真实路径的映射。

订单包把文件平铺成 GUID 名，靠这个映射还原原始目录结构。
禁止靠文件名猜 —— 实测 GUID 文件名不含任何语义。
"""

from __future__ import annotations

from pathlib import Path

from ._xml import load_object_graph


def parse_alias_map(path: Path | str) -> dict[str, str]:
    """返回 {alias_guid: original_relative_path}。

    原始路径用反斜杠分隔（3Shape 在 Windows 下生成），统一转成正斜杠。
    """
    node = load_object_graph(path)
    out: dict[str, str] = {}
    for item in node.descendants("TAliasFilenameItem"):
        original = item.get("OriginalName")
        alias = item.get("AliasName")
        if original and alias:
            out[alias] = original.replace("\\", "/")
    return out


def find_package_root(alias_map: dict[str, str]) -> str | None:
    """从映射推断包内根目录名（实测为患者目录名，属 PHI）。

    取所有原始路径的公共首段；不一致时返回 None（说明是拼接包）。
    """
    heads = {p.split("/")[0] for p in alias_map.values() if "/" in p}
    return heads.pop() if len(heads) == 1 else None


def infer_jaw_from_path(original_path: str) -> str | None:
    """从路径里的 Lower/Upper 目录名推断颌位。

    比猜 ScanType 的语义可靠 —— 实测扫描目录就是 Scans/Lower 与 Scans/Upper。
    """
    low = original_path.lower()
    for token, jaw in (("/lower/", "lower"), ("/upper/", "upper")):
        if token in low:
            return jaw
    return None
