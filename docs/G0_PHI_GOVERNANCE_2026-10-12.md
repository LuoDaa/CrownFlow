# G0：PHI 数据生命周期治理（2026-10-12）

范围：**只做治理** —— 策略枚举 + 导入持久化规则 + 导出规则 + 负向测试。
不碰几何、不碰 revision 语义、不顺手修 occlusion（那是下一批 O1）。

## 1. 策略：三档数据生命周期（不是 bool）

```
IDENTIFIED            显式 opt-in：原始数据（含源订单包）在受控区保留，便于复核来源
    │                 **不是默认**，必须调用方明确要求
    ↓
PSEUDONYMIZED   ←── 默认：工作区 / 库 / 凭据 / 日志不含原始 PHI 值；
    │                 只留"字段存在性 + 分类"与**技术 provenance**（来源 sha256、
    │                 解析器版本、文件角色）—— 来源可追、身份不可追
    ↓
DEIDENTIFIED_EXPORT   对外导出：连伪名化凭据都不进包，导出前必须过闸门
```

`app/services/phi_policy.py`（新模块）：`PhiPolicy` / `DEFAULT_POLICY=PSEUDONYMIZED` /
`resolve_policy()`（显式参数 > `CROWNFLOW_PHI_POLICY` > 默认；非法值 fail loud）/
`workspace_rules()`（每档允许什么：保留原始包、持久化原值、名字脱敏、导出原值）/
`redact_phi_fields()` / `sanitize_name()` / `sanitize_logical_path()` / `drop_phi_keys()` /
`assert_no_phi()` / `export_gate()` / `assert_export_clean()` / `phi_needles_for_case()` /
`technical_provenance()`。

### 默认档强制项（逐条对应你列的要求）

| 要求 | 落点 |
|---|---|
| 不复制 `source_package.zip` 到病例工作区 | `commit()` 默认 `copy_source=False`；`source_package_retained=False` |
| 不持久化 `phi_fields` 原值，只留存在性/分类/可选 hash | `redact_phi_fields()` → 凭据里是 `phi_report`，`phi_fields` 键消失 |
| `original_name` 不含 PHI 的包根/路径 | `sanitize_logical_path()` 后再落 `assets.original_name` 与凭据 |
| 患者名不进 case ID / 文件名 / artifact path / 日志 | case ID 是 `uuid4().hex`；名字经 `sanitize_name()`；异常消息只报序号不报值 |
| 技术 provenance 继续保留 | 凭据与 `cases.order_json` 里有 `source_sha256` / `importer_version` / `phi_policy` / 角色 / 校验面板 |
| 需要原始包必须显式 `IDENTIFIED` | `commit(keep_source_package=True)` 在默认档下**抛错**并提示改档（不是静默照做） |

### 关于"姓名哈希"（按你的意见）

默认**连哈希都不存**（`redact_phi_fields` 只在显式配置 `CROWNFLOW_PHI_LINKAGE_HMAC_SECRET`
时才写 `value_digest`，且用 `HMAC-SHA256(secret, normalized)` 而非裸 SHA256）——
理由写进了模块 docstring：姓名空间小可字典攻击，且同姓名跨病例形成可链接标识。
负向测试里明确断言默认档 `"value_digest" not in report[...]`。

## 2. 导出闸门

`export_gate(zip_bytes, needles=..., phi_keys=...)`，结构与值双保险：

- **结构**（任何档都该满足）：条目名只允许 ASCII 安全字符集（我们的命名里没有中文；
  出现中文就是有人把名字写进去了）、不得含 `source_package.zip` / 脱敏映射表、
  文本条目正文不得出现 PHI 字段名；
- **值**：命中 `phi_needles_for_case(case_id)` 就拒绝。needle 来源只有两处：
  导入凭据里 IDENTIFIED 档留下的原值、病例显示名。

已接到 `/api/designs/{id}/package.zip`（工程包 = 对外产物）。

**顺手修掉一个我自己写的泄漏**：闸门的初始实现把命中的 needle **原值**拼进异常消息 ——
测试当场抓住（`assert PHI_MARKER not in str(err.value)`）。现在返回值是不透明的
`needle_1/needle_2`（按入参顺序编号），调用方按序号回查，序号不出进程。

## 3. 负向测试（G0 的主要交付物）

`tests/integration/test_g0_phi_governance.py`（9 条）。每个用例注入假身份信息
（`SyntheticPatientZhaoQi` / `SyntheticStreetNo42`，非真实数据），然后全文搜索：

| 测试 | 证明的事 |
|---|---|
| `..._keeps_phi_out_of_the_whole_case_workspace` | 工作区**文件名 + 文件内容**里搜不到 PHI；原始包不在 |
| `..._keeps_phi_out_of_db_columns` | `assets.original_name` / `cases.name` / `cases.order_json` 都干净，且 provenance 仍在 |
| `test_record_keeps_phi_report_but_never_the_values` | 凭据里只有"存在性 + 分类"，无 `value`、无 `value_digest`；`order_meta` 里 PHI 键被剔 |
| `test_default_policy_rejects_legacy_keep_source_package_request` | 默认档下"顺手保留原始包"被**拒绝** |
| `test_preview_reports_policy_without_side_effects` | preview 仍零副作用（解析层仍拿得到 PHI 供脱敏用） |
| `test_identified_policy_retains_raw_package_and_values`（**反向**） | 显式 IDENTIFIED ⇒ 原始包与原值**可以**保留 |
| `test_export_gate_blocks_phi_even_for_identified_cases`（**反向**） | 源病例是 IDENTIFIED 也不许导出 PHI；且错误消息不回显值 |
| `test_export_gate_structural_rules` | 非 ASCII 条目名 / 原始包 / 映射表 / PHI 键名一律拒绝 |
| `test_policy_resolution_defaults_to_pseudonymized_and_rejects_unknown` | 默认值 = PSEUDONYMIZED；未知档位 fail loud |

同时更新两条既有测试（政策变更，非放宽）：
`test_import_writes_traceable_record`（不再断言保留原始包，改为断言**不存在** +
provenance 仍在）、`test_case_name_can_be_supplied` → `..._is_sanitized_under_default_policy`。

## 4. 本批运行

本机全量 `pytest tests`：**585 passed / 10 skipped / 2 xfailed / 0 failed**
（G0 前 576 passed / 2 xfailed；新增 9 条 G0 负向测试，去掉 0 条）。G0 是纯治理批 ——
几何与 revision 语义零改动，几何相关的既有测试全部原样通过。

## 5. 未做（明确边界）

1. **exocad / ground-truth 离线工具链没有并入本策略**。它有自己的闸门
   （`groundtruth/assert_phi_free`：写 CSV/JSON 前出现非 ASCII 就拒绝），本批**未统一**——
   统一它属于后续治理项，且要先确认哪些离线产物是"研究内部"、哪些会对外。
2. **`revisions.json` 里的操作员姓名**未处理：那是**员工**标识，不属患者 PHI；
   要不要在导出里脱敏是产品决定，本批只保证患者身份信息不出去。
3. **数据库历史行**未回填：本批只保证**新写入**合规（存量病例的
   `order_json` 若有旧格式，需要一次显式迁移才谈得上清理）。
4. `case_name` 只做"姓名类 CJK 串替换"，不做人工审核队列 —— 若操作员把姓名写成
   拼音/英文，默认档不会识别（这也是为什么导出闸门要用**值**扫描而不是只靠名字规则）。
