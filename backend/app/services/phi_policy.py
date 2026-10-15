"""PHI 数据生命周期策略（G0 / 2026-10-12）。

**这不是一个 bool 开关，而是三档数据生命周期**：

    IDENTIFIED            显式 opt-in：原始数据（含源包）在受控区保留，便于复核来源；
        │                 **不是默认**，必须由调用方明确要求。
        ↓
    PSEUDONYMIZED   ←── 默认：工作区/库/凭据/日志不含原始 PHI 值，
        │                只保留"字段存在性 + 分类"与**技术 provenance**（来源 sha256、
        │                解析器版本、文件角色）——来源可追，身份不可追。
        ↓
    DEIDENTIFIED_EXPORT   对外导出/分享：连伪名化的工作区凭据都不进包，导出前必须过闸门。

## 为什么默认不是 IDENTIFIED

把"保留原始包"当默认，等于让**每一个**导入都携带 PHI，然后指望以后有人来清理。
默认必须是"工作区里根本没有原始 PHI"，需要原始包的人显式开口（并且在
PSEUDONYMIZED 下显式请求保留会被**拒绝**并提示改档 —— 不是静默照做）。

## 为什么不存"姓名哈希"

``SHA256(患者姓名)`` **不是匿名化**：姓名空间极小，字典攻击可还原；而且同一姓名在不同病例里
形成**可链接标识**（跨病例关联本身就是 PHI 泄露）。所以默认**连哈希都不存**，只存
``{"present": true, "classification": ...}``。

只有当业务确实需要"是否同一来源"时，才用 ``PHI_LINKAGE_HMAC_SECRET`` 开启
``HMAC-SHA256(secret, normalized_value)`` —— 密钥不在数据里，跨病例链接需要密钥。

## 纪律

- 任何面向操作员/日志/异常的消息里**不得回显 PHI 值**（本模块的异常只报字段名与位置）；
- 技术 provenance（sha256/角色/解析器版本）必须保留 —— 否则"来源可追"这条也没了；
- 本模块不做几何、不碰 revision 语义。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

#: 环境变量兜底（未显式传 policy 时用）。非法值直接报错，不静默取默认。
POLICY_ENV = "CROWNFLOW_PHI_POLICY"
#: 只有显式配置了密钥才做 HMAC 联动标识（默认 None = 不存任何摘要）。
LINKAGE_SECRET_ENV = "CROWNFLOW_PHI_LINKAGE_HMAC_SECRET"

#: 路径中出现连续 2–4 个 CJK 字符即视为姓名类 PHI（与 exocad sanitizer 同口径）。
_CJK_NAME_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")

#: PHI 字段分类（报告里给分类，不给值）。
CLASS_PATIENT_IDENTIFIER = "PATIENT_IDENTIFIER"
CLASS_ORDER_IDENTIFIER = "ORDER_IDENTIFIER"
CLASS_CONTACT = "CONTACT"
CLASS_ADDRESS = "ADDRESS"
CLASS_OPERATOR = "OPERATOR"

_CLASSIFICATION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("patient", "name", "birth", "gender", "sex"), CLASS_PATIENT_IDENTIFIER),
    (("orderid", "order_id", "refno", "guid", "caseid", "patientguid"), CLASS_ORDER_IDENTIFIER),
    (("contact", "phone", "email", "mobile", "fax"), CLASS_CONTACT),
    (("address", "city", "zip", "street", "delyvery", "delivery", "ship"), CLASS_ADDRESS),
    (("operator", "doctor", "client", "lab", "clinic", "customer", "erp"), CLASS_OPERATOR),
)


def classify_field(name: str) -> str:
    """按字段名分类（分类用于报告与统计，**不含值**）。"""
    low = (name or "").lower()
    for needles, label in _CLASSIFICATION_RULES:
        if any(needle in low for needle in needles):
            return label
    return CLASS_PATIENT_IDENTIFIER


class PhiPolicy(str, Enum):
    IDENTIFIED = "IDENTIFIED"
    PSEUDONYMIZED = "PSEUDONYMIZED"
    DEIDENTIFIED_EXPORT = "DEIDENTIFIED_EXPORT"


DEFAULT_POLICY = PhiPolicy.PSEUDONYMIZED


class PhiPolicyError(ValueError):
    """策略本身不合法，或"请求的动作与策略冲突"。**消息里绝不回显 PHI 值。**"""


def resolve_policy(value: "PhiPolicy | str | None" = None) -> PhiPolicy:
    """解析策略：显式参数 > 环境变量 > 默认 PSEUDONYMIZED。非法值 fail loud。"""
    candidate = value if value is not None else os.environ.get(POLICY_ENV)
    if candidate is None:
        return DEFAULT_POLICY
    if isinstance(candidate, PhiPolicy):
        return candidate
    try:
        return PhiPolicy(str(candidate).strip().upper())
    except ValueError as exc:
        raise PhiPolicyError(
            "未知 PHI 策略 %r；只能是 %s" % (candidate, " / ".join(p.value for p in PhiPolicy))) from exc


@dataclass(frozen=True)
class WorkspaceRules:
    """某一档策略下，工作区允许出现什么。"""

    policy: PhiPolicy
    copy_source_package: bool
    persist_raw_values: bool
    sanitize_names: bool
    export_raw_phi: bool

    def as_report(self) -> dict:
        return {"policy": self.policy.value,
                "copy_source_package": self.copy_source_package,
                "persist_raw_phi_values": self.persist_raw_values,
                "sanitize_names": self.sanitize_names,
                "export_raw_phi": self.export_raw_phi}


def workspace_rules(policy: "PhiPolicy | str | None" = None) -> WorkspaceRules:
    resolved = resolve_policy(policy)
    if resolved is PhiPolicy.IDENTIFIED:
        return WorkspaceRules(resolved, copy_source_package=True, persist_raw_values=True,
                              sanitize_names=False, export_raw_phi=False)
    # PSEUDONYMIZED 与 DEIDENTIFIED_EXPORT 的工作区规则相同；
    # 区别在于导出：EXPORT 档连伪名化凭据都不进包（见 export_gate）。
    return WorkspaceRules(resolved, copy_source_package=False, persist_raw_values=False,
                          sanitize_names=True, export_raw_phi=False)


# ---------------------------------------------------------------- 字段脱敏

@dataclass(frozen=True)
class PhiFieldReport:
    """单个 PHI 字段的**报告**（有值吗、属于哪类、可选联动摘要）。"""

    name: str
    present: bool
    classification: str
    value_digest: str | None = None

    def as_dict(self) -> dict:
        out = {"present": self.present, "classification": self.classification}
        if self.value_digest is not None:
            out["value_digest"] = self.value_digest
        return out


def _digest(value: str, secret: str) -> str:
    normalized = re.sub(r"\s+", "", str(value)).strip().lower()
    return hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def redact_phi_fields(fields: dict, *, policy: "PhiPolicy | str | None" = None,
                      secret: str | None = None) -> dict:
    """把 PHI 字段字典变成**可持久化**的报告。

    - ``IDENTIFIED``：原值保留（显式档；调用方要为此负责）；
    - 其他档：只留 ``{present, classification, [value_digest]}``；
      ``value_digest`` 仅在**显式配置密钥**时出现（HMAC，防字典攻击与跨病例链接）。
    """
    rules = workspace_rules(policy)
    resolved_secret = secret if secret is not None else os.environ.get(LINKAGE_SECRET_ENV)
    report: dict[str, dict] = {}
    for name, value in (fields or {}).items():
        present = value not in (None, "")
        if rules.persist_raw_values:
            entry = {"present": present, "classification": classify_field(name)}
            if present:
                entry["value"] = value
            report[name] = entry
            continue
        digest = _digest(value, resolved_secret) if (present and resolved_secret) else None
        report[name] = PhiFieldReport(str(name), present, classify_field(name), digest).as_dict()
    return report


def drop_phi_keys(mapping: dict, *, phi_names) -> dict:
    """从"顺带带出来的整份属性表"里删掉 PHI 键（值绝不进报告）。

    ``order_meta`` 这类字段是**整份 XML 属性**，PHI 就混在里面 ——
    只做白名单式过滤，不做"看起来像姓名就删"的启发式。
    """
    phi = {str(n).lower() for n in phi_names}
    return {k: v for k, v in (mapping or {}).items() if str(k).lower() not in phi}


# ---------------------------------------------------------------- 名称脱敏

def sanitize_name(value: str, *, policy: "PhiPolicy | str | None" = None,
                  redactions=(), fallback: str = "subject", keep_ascii: bool = True) -> str:
    """把可能含 PHI 的**名字/路径**变成稳定、可读、无身份的名字。

    - ``IDENTIFIED``：原样返回（显式档）；
    - 其他档：已知 PHI 值整体替换为 ``fallback``；CJK 姓名串替换为 ``fallback``；
      ASCII 数字/字母/``_-./`` 保留（GUID、FDI、角色名都在这一类）。
    """
    if workspace_rules(policy).sanitize_names is False:
        return value
    text = str(value)
    for needle in redactions:
        needle = str(needle or "").strip()
        if needle:
            text = re.sub(re.escape(needle), fallback, text, flags=re.IGNORECASE)
    if keep_ascii:
        text = _CJK_NAME_RE.sub(fallback, text)
    else:
        text = re.sub(r"[^A-Za-z0-9._/-]", fallback, text)
    text = re.sub(r"(%s)([/\\]*\1)+" % re.escape(fallback), fallback, text)
    return text.strip("/\\ ") or fallback


def sanitize_logical_path(logical: str, *, policy: "PhiPolicy | str | None" = None,
                          redactions=()) -> str:
    """逻辑路径逐段脱敏（保留目录结构语义，去掉身份信息）。"""
    if workspace_rules(policy).sanitize_names is False:
        return str(logical)
    parts = [p for p in re.split(r"([/\\])", str(logical))]
    return "".join(sanitize_name(p, policy=policy, redactions=redactions)
                   if p not in ("/", "\\") else p for p in parts)


# ---------------------------------------------------------------- 校验闸门

def find_phi_values(payload, needles) -> list[str]:
    """在任意 JSON 结构里找 needle 命中。

    **返回值是不透明的标记名（``needle_1``…按入参顺序编号），不是 needle 本身** ——
    否则异常消息与日志会把 PHI 值原样带出来（"报告 PHI" 本身就违反纪律）。
    调用方手里有 ``needles`` 列表，需要时自己按序号回查，序号不会离开本进程。
    """
    hits: list[str] = []
    wanted = [(i + 1, str(n)) for i, n in enumerate(needles or []) if str(n or "").strip()]
    if not wanted:
        return hits

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value)
                walk(str(key))
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            low = node.lower()
            for index, needle in wanted:
                label = "needle_%d" % index
                if needle.lower() in low and label not in hits:
                    hits.append(label)

    walk(payload)
    return hits


def assert_no_phi(payload, *, needles, where: str, policy: "PhiPolicy | str | None" = None) -> None:
    """负向闸门：出现 PHI 就失败。**报错只说命中哪个 needle 名与位置，不回显值。**"""
    resolved = resolve_policy(policy)
    if resolved is PhiPolicy.IDENTIFIED:
        return                      # 显式档允许原值；导出仍要过 export_gate
    hits = find_phi_values(payload, needles)
    if hits:
        raise PhiPolicyError(
            "PHI 泄漏闸门拦下：%s 里出现 PHI 值（命中 %d 个标记：%s）"
            % (where, len(hits), ", ".join(hits)))


@dataclass
class ExportGateReport:
    policy: str
    entries: list = field(default_factory=list)
    violations: list = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.violations

    def as_report(self) -> dict:
        return {"policy": self.policy, "clean": self.clean,
                "entries": list(self.entries), "violations": list(self.violations)}


#: 导出包里永远不允许出现的条目名（原始包、映射表）。
FORBIDDEN_EXPORT_ENTRIES = ("source_package.zip", "deid_mapping", "_deid_mapping")

#: 导出包里条目名允许出现的字符（我们自己的命名：design.json / history/1/outer.stl）。
_EXPORT_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


def export_gate(zip_bytes: bytes, *, policy: "PhiPolicy | str | None" = None,
                needles=(), phi_keys=()) -> ExportGateReport:
    """导出闸门（DEIDENTIFIED_EXPORT）：结构规则 + 值扫描，双保险。

    结构规则（与策略无关，任何档都不该违反）：
    - 条目名只允许 ASCII 安全字符集（我们的命名里没有中文；有就是有人把名字写进去了）；
    - 条目名、条目内容不得出现 ``source_package.zip`` / 脱敏映射表；
    - 文本条目（``.json``/``.txt``）不得出现 ``phi_keys`` 里的键名（PHI 字段名本身就是线索）。

    值扫描：``needles`` 传已知 PHI 值（例如该病例以 IDENTIFIED 导入时留下的原值）。
    """
    import io
    import zipfile

    resolved = resolve_policy(policy)
    report = ExportGateReport(policy=resolved.value)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            name = info.filename
            report.entries.append(name)
            if not _EXPORT_SAFE_NAME_RE.match(name):
                report.violations.append("ENTRY_NAME_NOT_ASCII_SAFE:%s" % name)
            lowered = name.lower()
            for forbidden in FORBIDDEN_EXPORT_ENTRIES:
                if forbidden in lowered:
                    report.violations.append("FORBIDDEN_ENTRY:%s" % name)
            if not name.lower().endswith((".json", ".txt", ".csv")):
                continue
            try:
                text = z.read(info).decode("utf-8", errors="replace")
            except Exception:                                    # noqa: BLE001
                report.violations.append("ENTRY_UNREADABLE:%s" % name)
                continue
            hits = find_phi_values(text, needles)
            if hits:
                report.violations.append("PHI_VALUE:%s:%s" % (name, ",".join(hits)))
            for key in phi_keys:
                if '"%s"' % key in text:
                    report.violations.append("PHI_KEY:%s:%s" % (name, key))
    return report


def assert_export_clean(zip_bytes: bytes, *, needles=(), phi_keys=(), where: str = "export.zip",
                        policy: "PhiPolicy | str | None" = None) -> ExportGateReport:
    report = export_gate(zip_bytes, policy=policy, needles=needles, phi_keys=phi_keys)
    if not report.clean:
        raise PhiPolicyError(
            "%s 未通过 PHI 导出闸门（%d 处）：%s。若确需导出原始身份信息，请显式走 IDENTIFIED 受控流程。"
            % (where, len(report.violations), "；".join(sorted(set(report.violations))[:6])))
    return report


def file_sha256(path: Path) -> str:
    """技术 provenance：来源文件字节哈希（**与 PHI 无关**，必须保留）。"""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def technical_provenance(path: Path, *, importer: str, importer_version: str,
                         schema: str | None = None, role: str | None = None) -> dict:
    """可持久化的来源凭据：没有身份信息，只有"这份几何从哪来"。"""
    return {"importer": importer, "importer_version": importer_version,
            "schema": schema, "role": role,
            "source_sha256": file_sha256(path) if Path(path).is_file() else None}


def dump_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def phi_needles_for_case(case_id: str) -> list[str]:
    """收集**该病例**已知的 PHI 原值，用于导出值扫描。

    来源只有两处，都不会凭空生成身份信息：

    1. 导入凭据里的 ``phi_report``（仅当该病例是以 ``IDENTIFIED`` 导入时才有原值）；
    2. 病例显示名 ``cases.name``（默认档下已脱敏，IDENTIFIED 下可能是真名）。

    默认档下这两处都没有原始 PHI ⇒ 返回值只可能为空或极少 —— 值扫描是**第二道**防线，
    第一道是导入时的"不落原始值"。
    """
    import json as _json

    from ..config import CASE_DIR
    from ..db import db

    needles: list[str] = []
    with db() as conn:
        row = conn.execute("SELECT name FROM cases WHERE id=?", (case_id,)).fetchone()
    if row is not None and row["name"]:
        needles.append(str(row["name"]))
    record_path = Path(CASE_DIR) / case_id / "three_shape_import.json"
    if record_path.is_file():
        try:
            record = _json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
        for entry in (record.get("phi_report") or {}).values():
            value = (entry or {}).get("value")
            if isinstance(value, str) and value.strip():
                needles.append(value)
        for asset_entry in record.get("assets") or []:
            value = (asset_entry or {}).get("logical_path")
            if isinstance(value, str) and value.strip():
                needles.append(Path(value).name)
    return sorted({n for n in needles if len(n) >= 2})
