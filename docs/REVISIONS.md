# 设计修订链

`app/revisions/` — 不可变修订记录 + 人工结果不被静默覆盖

---

## 1. 存在理由

旧实现有两个实质缺陷：

**缺陷一：人工 Margin 会丢失。** `POST /api/designs/{id}/margin` 在前端**从不被调用**
（`api.saveMargin` 是死代码），手绘的 Margin 只有一路走到 `finalize` 时才落库。
中间任何刷新或离开都会丢失。

**缺陷二：自动提案会静默覆盖人工结果。** `suggest-margin` 直接写
`designs.margin_json`，把技师的修改无声地替换掉。

---

## 2. 数据模型

```sql
CREATE TABLE design_revisions(
  id TEXT PRIMARY KEY,
  design_id TEXT NOT NULL,
  revision_no INTEGER NOT NULL,      -- 每设计内自增，UNIQUE(design_id, revision_no)
  kind TEXT NOT NULL,
  algorithm_version TEXT,
  parameter_set_json TEXT,           -- 本次使用的参数集（可溯源）
  margin_source TEXT,                -- auto | manual | auto+manual
  operator TEXT,
  approved_by TEXT,
  operator_actions_json TEXT,
  artifacts_json TEXT,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
```

**修订不可变。** 每次变更插入新行，不更新旧行。

`designs` 表新增两列：`current_margin_revision_id`（当前生效版本）与 `approved_by`。

SQLite 的 `CREATE TABLE IF NOT EXISTS` **不会给已存在的表补列**，
因此 `db.py` 有轻量迁移 `_MIGRATIONS` + `ALTER TABLE ADD COLUMN`。
工作区里已有一个含病例数据的 `crownflow.sqlite3`，不迁移会直接报"没有列"。

---

## 3. 修订类型

| kind | 何时产生 | 是否当前生效 |
|---|---|---|
| `margin_auto_accepted` | 自动提案，且此前无人工 Margin | ✅ |
| `margin_candidate` | 自动提案，且当前含人工成分 | ❌ 待决定 |
| `margin_manual` | 人工绘制/编辑 | ✅ |
| `axis_candidate` | 自动求解就位方向 | ❌ 永远不是 |
| `axis_accepted` | 人工确认/修改就位方向 | — |
| `outer` | 生成外冠 | — |
| `final` | 构建最终牙冠 | — |

### Margin 来源归因（`MarginSource`）

```
manual           纯人工绘制，无自动前序
auto             纯自动
auto+manual      先采纳了自动提案，再人工编辑
```

`is_human_authored` 为真的两类（`manual` / `auto+manual`）**不得被自动提案覆盖**。

实测归因序列（人工绘制 → 再人工编辑 → 自动提案 → 采纳自动 → 再人工编辑）：

```
#1 manual        #2 manual        #3 margin_candidate (applied=False)
#4 auto (采纳 #3)                 #5 auto+manual
```

---

## 4. 核心不变量

### 不变量一：自动提案永不静默覆盖人工结果

`record_auto_margin_candidate()` 返回 `(revision, applied)`：

- 当前生效的 Margin 含人工成分 → **只登记 `margin_candidate`**，`applied=False`，
  当前生效版本不变，响应带 `requires_decision=True`。
- 之前无 Margin，或生效的是纯自动结果 → 直接生效（重跑同一算法不算覆盖人工作品）。

前端据此提供：**替换 / 保留人工 / 取消**。

### 不变量二：人工 Margin 立即落库

`POST /api/designs/{id}/margin` 是手绘 Margin 的正式保存路径，保存即成为当前生效版本。

### 不变量三：候选被消费后不再要求决策

`revision_chain()` 把被后续接受修订通过
`payload.basis.candidate_revision_no` 引用过的候选算作**已消费**：

```python
consumed = {(r.payload.get("basis") or {}).get("candidate_revision_no")
            for r in revisions if not r.is_candidate}
candidates = [r for r in revisions
              if r.is_candidate and not r.is_current and r.revision_no not in consumed]
```

实测踩到的坑：没有这条规则时，采纳后候选未被标记消费，
`requires_decision` **永久为 True**，前端每次都弹决策框。

### 不变量四：审计要能回答"为什么接受这个方向"

`POST /api/designs/{id}/axis` 写入 `axis_accepted` 时，会把候选当时的求解器结论
一并存入 `payload.basis`：

```json
{
  "type": "axis", "axis": [...], "source": "manual",
  "note": "技师手动改为垂直",
  "basis": {
    "candidate_revision_no": 2,
    "solver": {
      "status": "REVIEW_REQUIRED", "confidence": 0.8521,
      "undercut_area_ratio": 0.2306, "margin_visibility_ratio": 0.7778,
      "objective_is_flat": null
    },
    "solver_reasons": [
      "最优轴仍有较大倒凹（面积比 0.231 > 0.15），需要 blockout 或人工调整",
      "Margin 沿最优轴可见率偏低（0.778 < 0.9）"
    ]
  }
}
```

只记录"方向被改成什么"是不够的 —— 复核时要看的是"当时系统怎么报的、谁决定接受它"。

### 不变量五：不产生多余候选

`outer/auto` **不登记**就位方向候选。传入的方向可能刚被人工确认过；
再登记一条 `axis_candidate` 会让修订链冒出一个并不需要决策的待决定项，
使 `requires_decision` 误报为 True（实测踩到过）。

---

## 5. 实测修订链示例

完整流程（保存 Margin → 求解轴 → 确认轴 → 生成外冠 → 最终化）：

```
#1 margin_manual      source=manual
#2 axis_candidate     source=auto    solver.status=REVIEW_REQUIRED undercut=0.2306
#3 axis_accepted      source=manual  basis→#2  approved_by=tech_A
#4 outer
#5 final
```

`requires_decision` 在采纳前为 `True`（pending `[2]`），采纳后为 `False`
（`consumed_candidates = [2]`）。

---

## 6. API

| 方法 | 路径 | 用途 |
|---|---|---|
| `POST` | `/api/designs/{id}/margin` | 保存人工 Margin（立即生效） |
| `POST` | `/api/designs/{id}/suggest-margin` | 自动提案（含人工时只登记候选） |
| `POST` | `/api/designs/{id}/suggest-axis` | 求解就位方向 → `axis_candidate` |
| `POST` | `/api/designs/{id}/axis` | 人工确认/修改 → `axis_accepted` |
| `GET`  | `/api/designs/{id}/axis` | 取当前方向 |
| `GET`  | `/api/designs/{id}/revisions` | 完整修订链 + `requires_decision` |
| `GET`  | `/api/designs/{id}/revisions/{no}` | 单条修订 |
| `POST` | `/api/designs/{id}/revisions/{no}/accept` | 把候选提升为当前生效 |

完整修订链随设计包 ZIP 导出（`revisions.json`），与 `design.json`、
最终 STL 放在一起。

---

## 7. 验收

`tests/integration/test_revision_chain.py`（5 项）：

- 手绘 Margin 保存即落库
- 自动提案不得覆盖人工结果，只登记候选
- 采纳时记录 `basis`（当时求解器状态与理由），并消费该候选
- `outer/auto` 不产生多余轴候选
- 无 Margin 的降级路径可 JSON 序列化

测试用 `isolated_db` fixture 把数据库与数据目录指到 `tmp_path`。
**绝不允许碰工作区里的真实病例库。** 为此 `db.py` 的 `_db_path()` 在运行时
读取 `config.DB_PATH` 而非 import 时绑定，使其可被 monkeypatch。
