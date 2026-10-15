"""3Shape 的 XML 方言：`Object` / `Property` / `List` 的对象图。

`AliasFilenames.xml`、订单 XML、`Materials.xml` 都用同一套结构：

    <Object name="主对象" type="TSomething">
      <Property name="A" value="…"/>
      <Object name="子对象" type="TChild">
        <Property name="B" value="…"/>
      </Object>
      <List name="Items">
        <Object type="TItem">…</Object>
      </List>
    </Object>

**禁止用正则解析。** 这与 exocad `.constructionInfo` 的教训同源：
属性值里可能出现转义实体（实测 `Wax-&gt;Gold Crown`），
而 `<Property ... />` 是自闭合标签 —— 正则很容易错位或漏项。
一律走 `xml.etree`。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    """对象图里的一个 `<Object>`。"""

    name: str
    type: str
    properties: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.properties.get(key, default)

    def child(self, type_name: str) -> "Node | None":
        for c in self.children:
            if c.type == type_name:
                return c
        return None

    def descendants(self, type_name: str) -> list["Node"]:
        out = []
        for c in self.children:
            if c.type == type_name:
                out.append(c)
            out.extend(c.descendants(type_name))
        return out


def _parse_object(el: ET.Element) -> Node:
    node = Node(name=el.get("name", ""), type=el.get("type", ""))
    for prop in el.findall("Property"):
        n = prop.get("name")
        if n is not None:
            # ET 已解开 XML 实体（&gt; → >），值保持原样
            node.properties[n] = prop.get("value", "") or ""
    for child in el:
        if child.tag == "Object":
            node.children.append(_parse_object(child))
        elif child.tag == "List":
            for item in child.findall("Object"):
                node.children.append(_parse_object(item))
    return node


def load_object_graph(path: Path | str) -> Node:
    """读一个 3Shape XML，返回根 ``Node``（剥掉最外层的容器标签）。"""
    path = Path(path)
    root = ET.parse(path).getroot()
    inner = root.find("Object")
    if inner is None:
        raise ValueError(f"{path.name} 里没有 <Object> 根节点，不是 3Shape 对象图")
    return _parse_object(inner)


def load_properties(path: Path | str) -> Node:
    """只取 ``<Properties>`` 下的平铺属性（HPS 扫描文件的结构）。"""
    path = Path(path)
    root = ET.parse(path).getroot()
    node = Node(name=path.name, type=root.tag)
    props = root.find(".//Properties")
    if props is not None:
        for prop in props.findall("Property"):
            n = prop.get("name")
            if n is not None:
                node.properties[n] = prop.get("value", "") or ""
    # HPS 的 Packed_geometry 也带属性（facet_count / vertex_count）
    for tag in ("Facets", "Vertices", "Schema"):
        el = root.find(f".//{tag}")
        if el is not None:
            node.properties[f"_{tag}"] = el.text or ""
            for k, v in el.attrib.items():
                node.properties[f"{tag}.{k}"] = v
    return node
