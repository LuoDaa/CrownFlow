"""PHI 扫描与导入阶段脱敏。

实测（`数据/15个完整病例`）PHI 分布在**六层**，所以不能"先读进系统再清理"：

1. 目录名                     刘金鸿 / 薛文爱 / 张哲森 / 武淑琴 / 邱治钧 / 张寿山 / 成兰敏 / 王煜 / 吴宝刚
2. STL 文件名                 吴宝刚-24-25-26-27-bridge_slm_cad.stl、张竹林82404110783-…-B组 (1).stl
3. ``.matrix4`` 文件名        吴宝刚.matrix4（同一文件名横跨 4 个不同病例目录）
4. ``.3ox`` XML 字段          Patient_LastName / PatientId / LabID / ClinicID / OperatorID
5. ``.dentalProject`` Notes   订单号与姓名
6. ``import.log``             UNC 共享路径 ``\\\\192.168.169.171\\cad共享\\…``

设计原则：
- **源病例目录永不被修改。** 脱敏是"复制 + 重命名 + 重写文本"到新目录。
- 输出文件名只由 ``source_order_id`` 派生的稳定哈希与序号构成，不含任何原始名称。
- 无法安全清理内容的二进制容器（``.dentalCAD``，实测 100 MB 量级）默认**排除**，
  必须显式 ``include_binary=True`` 才复制，并标记为待人工处理。
- 映射表含 PHI，写入时单独放并明确标注禁止提交。
"""

from __future__ import annotations

import hashlib
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .three_ox import PHI_TAGS

#: 路径中出现连续 2–4 个 CJK 字符即视为姓名类 PHI。
_CJK_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")
#: 日志/文本中的 UNC 共享路径。
_UNC_RE = re.compile(r"\\\\\\\\[^\\\s]+\\\\[^\s]*")
#: 纯 IPv4 主机（共享路径常以 IP 形式出现）。
_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")

#: 这些扩展名的内容是二进制容器，无法在本阶段擦除内部 PHI，默认排除。
BINARY_UNSCRUBBABLE_SUFFIXES = {".dentalcad", ".3oxz", ".db"}
#: 这些扩展名是文本，可以重写内容。
TEXT_SCRUBBABLE_SUFFIXES = {".dentalproject", ".matrix4", ".log", ".constructioninfo", ".modelinfo"}


class PhiKind(str, Enum):
    PATH_NAME = "PATH_NAME"
    XML_FIELD = "XML_FIELD"
    LOG_NETWORK_PATH = "LOG_NETWORK_PATH"
    BINARY_CANNOT_SCRUB = "BINARY_CANNOT_SCRUB"


@dataclass(frozen=True)
class PhiFinding:
    kind: PhiKind
    where: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "where": self.where, "detail": self.detail}


@dataclass
class DeidEntry:
    original_rel: str
    deid_rel: str
    action: str            # copy | rewrite | exclude
    reasons: list[str] = field(default_factory=list)


@dataclass
class DeidPlan:
    source_root: str
    out_root: str
    entries: list[DeidEntry] = field(default_factory=list)

    @property
    def excluded(self) -> list[DeidEntry]:
        return [e for e in self.entries if e.action == "exclude"]

    @property
    def phi_bearing_paths(self) -> list[str]:
        return [e.original_rel for e in self.entries
                if any("PATH_NAME" in r for r in e.reasons)]


def scan_phi(case_dir: Path) -> list[PhiFinding]:
    """扫描一个病例目录，返回全部 PHI 发现。只读，不修改任何文件。"""
    case_dir = Path(case_dir)
    findings: list[PhiFinding] = []

    # 1) 目录名与文件名
    for p in [case_dir, *case_dir.rglob("*")]:
        if p == case_dir:
            rel = "."
        else:
            rel = str(p.relative_to(case_dir))
        for part in Path(rel).parts:
            for m in _CJK_NAME_RE.findall(part):
                findings.append(PhiFinding(PhiKind.PATH_NAME, rel, f"路径分量含姓名片段: {m}"))

    # 2) XML 字段
    for p in case_dir.rglob("*"):
        if p.suffix.lower() in (".3ox",) or p.name.lower().endswith(".3ox"):
            try:
                root = ET.parse(p).getroot()
            except ET.ParseError:
                continue
            for tag in PHI_TAGS:
                value = root.findtext(f".//{tag}")
                if value:
                    findings.append(PhiFinding(
                        PhiKind.XML_FIELD, str(p.relative_to(case_dir)), f"{tag}={value}"))
        elif p.suffix.lower() == ".dentalproject":
            try:
                root = ET.parse(p).getroot()
            except ET.ParseError:
                continue
            notes = root.findtext("Notes") or ""
            for m in _CJK_NAME_RE.findall(notes):
                findings.append(PhiFinding(
                    PhiKind.XML_FIELD, str(p.relative_to(case_dir)), f"Notes 含姓名片段: {m}"))

    # 3) 日志中的网络路径
    for p in case_dir.rglob("*.log"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in _UNC_RE.findall(text):
            findings.append(PhiFinding(
                PhiKind.LOG_NETWORK_PATH, str(p.relative_to(case_dir)), f"UNC 共享路径: {m}"))
        if not _UNC_RE.search(text):
            for m in _IP_RE.findall(text):
                findings.append(PhiFinding(
                    PhiKind.LOG_NETWORK_PATH, str(p.relative_to(case_dir)), f"内网 IP: {m}"))

    # 4) 无法擦除的二进制容器
    for p in case_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in BINARY_UNSCRUBBABLE_SUFFIXES:
            findings.append(PhiFinding(
                PhiKind.BINARY_CANNOT_SCRUB, str(p.relative_to(case_dir)),
                f"{p.suffix} 是二进制容器，本阶段无法擦除内部 PHI"))

    return findings


def _case_slug(source_order_id: str | None, case_dir_name: str) -> str:
    """从订单 ID 派生稳定、无 PHI 的病例标识。"""
    basis = source_order_id or hashlib.sha256(case_dir_name.encode("utf-8")).hexdigest()
    return "case_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]


def build_deid_plan(
    case_dir: Path,
    out_root: Path,
    *,
    source_order_id: str | None = None,
    include_binary: bool = False,
) -> DeidPlan:
    """为一个病例生成脱敏计划。**不写任何文件。**"""
    case_dir = Path(case_dir)
    slug = _case_slug(source_order_id, case_dir.name)
    plan = DeidPlan(source_root=str(case_dir), out_root=str(Path(out_root) / slug))

    counter = 0
    for p in sorted(case_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(case_dir)
        reasons = [f"{f.kind.value}: {f.detail}" for f in scan_phi(case_dir)
                   if f.where == str(rel)]

        suffix = p.suffix.lower()
        if suffix in BINARY_UNSCRUBBABLE_SUFFIXES and not include_binary:
            plan.entries.append(DeidEntry(
                original_rel=str(rel), deid_rel="", action="exclude",
                reasons=reasons + ["二进制容器默认排除；需人工处理后再入库"]))
            continue

        counter += 1
        new_name = f"asset_{counter:03d}{suffix}"
        action = "rewrite" if suffix in TEXT_SCRUBBABLE_SUFFIXES else "copy"
        if suffix in BINARY_UNSCRUBBABLE_SUFFIXES:
            action = "copy"
            reasons = reasons + ["二进制内容未擦除，仅重命名"]
        plan.entries.append(DeidEntry(
            original_rel=str(rel), deid_rel=new_name, action=action, reasons=reasons))

    return plan


def apply_deid_plan(plan: DeidPlan) -> None:
    """执行脱敏计划：复制 / 重写文本到输出目录。源目录只读。"""
    out_root = Path(plan.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    src_root = Path(plan.source_root)

    for e in plan.entries:
        if e.action == "exclude":
            continue
        src = src_root / e.original_rel
        dst = out_root / e.deid_rel
        if e.action == "copy":
            shutil.copy2(src, dst)
            continue
        _rewrite_text(src, dst)


def _rewrite_text(src: Path, dst: Path) -> None:
    """重写文本文件：清 XML PHI 字段、清日志网络路径、把姓名片段替换为占位符。"""
    suffix = src.suffix.lower()

    if suffix in (".dentalproject", ".matrix4", ".modelinfo", ".constructioninfo"):
        try:
            tree = ET.parse(src)
        except ET.ParseError:
            shutil.copy2(src, dst)
            return
        root = tree.getroot()
        for tag in PHI_TAGS:
            for el in root.iter(tag):
                el.text = "REDACTED"
        for el in root.iter("Notes"):
            if el.text:
                el.text = _UNC_RE.sub("REDACTED_SHARE", _IP_RE.sub("REDACTED_IP", el.text))
                el.text = _CJK_NAME_RE.sub("REDACTED", el.text)
        tree.write(dst, encoding="utf-8", xml_declaration=True)
        return

    try:
        text = src.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        shutil.copy2(src, dst)
        return
    text = _UNC_RE.sub("REDACTED_SHARE", text)
    text = _IP_RE.sub("REDACTED_IP", text)
    if suffix == ".log":
        text = _CJK_NAME_RE.sub("REDACTED", text)
    dst.write_text(text, encoding="utf-8")


def write_mapping(plan: DeidPlan, path: Path) -> None:
    """写出源→脱敏映射表。

    **该文件含 PHI。** 必须存放在受控目录，绝不可提交到 Git 或放入训练目录。
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_WARNING": "此文件包含 PHI，禁止提交到 Git / 训练目录 / 普通研发目录。",
        "source_root": plan.source_root,
        "out_root": plan.out_root,
        "entries": [
            {"original_rel": e.original_rel, "deid_rel": e.deid_rel,
             "action": e.action, "reasons": e.reasons}
            for e in plan.entries
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
