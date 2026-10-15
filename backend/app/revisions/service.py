"""设计修订链服务。

核心不变量：

- **修订不可变。** 每次变更插入新行，不更新旧行。
- **自动提案永不静默覆盖人工结果。** 若当前 Margin 修订含人工成分，
  ``suggest`` 只写入一个 ``*_candidate`` 修订，当前生效版本不变，
  由调用方（前端）决定 Replace / Compare / Cancel。
- ``designs.margin_json`` 保留为"当前生效修订"的缓存，向后兼容旧代码路径。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..db import db
from .models import DesignRevision, MarginSource, RevisionKind, margin_payload


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _next_revision_no(conn, design_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(revision_no), 0) AS n FROM design_revisions WHERE design_id=?",
        (design_id,),
    ).fetchone()
    return int(row["n"]) + 1


def _current_revision_id(conn, design_id: str) -> str | None:
    row = conn.execute(
        "SELECT current_margin_revision_id FROM designs WHERE id=?", (design_id,)
    ).fetchone()
    return row["current_margin_revision_id"] if row else None


def append_revision(
    revision: DesignRevision,
    *,
    make_current: bool = False,
    operator: str | None = None,
) -> DesignRevision:
    """插入一条修订。``make_current=True`` 时同时把它设为当前生效版本。"""
    revision.operator = operator or revision.operator
    revision.created_at = revision.created_at or _now()
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if revision.kind is RevisionKind.AUTOFIT_REJECTED:
            if conn.execute("SELECT 1 FROM design_revisions WHERE design_id=? AND kind='outer' AND json_extract(payload_json,'$.proposal_id')=?",(revision.design_id,revision.payload.get('proposal_id'))).fetchone():
                from fastapi import HTTPException
                raise HTTPException(409,'此候选已采纳，请使用外冠撤销/恢复，不可再退回候选')
        revision.revision_no = _next_revision_no(conn, revision.design_id)
        row = revision.to_row()
        conn.execute(
            """INSERT INTO design_revisions(
                   id, design_id, revision_no, kind, algorithm_version,
                   parameter_set_json, margin_source, operator, approved_by,
                   operator_actions_json, artifacts_json, payload_json, created_at)
               VALUES(:id,:design_id,:revision_no,:kind,:algorithm_version,
                      :parameter_set_json,:margin_source,:operator,:approved_by,
                      :operator_actions_json,:artifacts_json,:payload_json,:created_at)""",
            row,
        )
        if make_current:
            if revision.kind is RevisionKind.PROFILE:
                import json
                conn.execute("""UPDATE designs SET parameters_json=?,profile_json=?,current_profile_revision_id=?,
                    final_path=NULL,cutter_path=NULL,qc_json=NULL,final_dependency_digest=NULL,status='profile_ready',updated_at=? WHERE id=?""",
                    (json.dumps(revision.parameter_set,sort_keys=True),json.dumps(revision.payload,sort_keys=True,ensure_ascii=False),revision.id,_now(),revision.design_id))
            else:
                _set_current_margin(conn, revision,confirmed=revision.kind is RevisionKind.MARGIN_MANUAL)
        if revision.kind is RevisionKind.AXIS_ACCEPTED:
            import json
            conn.execute("""UPDATE designs SET axis_json=?,current_axis_revision_id=?,final_path=NULL,cutter_path=NULL,
                qc_json=NULL,final_dependency_digest=NULL,status='axis_ready',updated_at=? WHERE id=?""",
                (json.dumps(revision.payload['axis']),revision.id,_now(),revision.design_id))
    revision.is_current = make_current
    return revision


def _set_current_margin(conn, revision: DesignRevision, *, confirmed=False) -> None:
    if revision.kind not in (RevisionKind.MARGIN_MANUAL,RevisionKind.MARGIN_CANDIDATE,RevisionKind.MARGIN_AUTO_ACCEPTED):
        raise ValueError('只有 Margin 修订可以成为当前边缘线')
    conn.execute(
        """UPDATE designs
              SET current_margin_revision_id=?, confirmed_margin_revision_id=?, margin_json=?, final_path=NULL,
                  cutter_path=NULL, qc_json=NULL, final_dependency_digest=NULL, status='margin_ready', updated_at=?
            WHERE id=?""",
        (
            revision.id,
            revision.id if confirmed else None,
            __import__("json").dumps(revision.payload.get("points", [])),
            _now(),
            revision.design_id,
        ),
    )
    if revision.kind is RevisionKind.MARGIN_MANUAL or (
        revision.margin_source is not None and revision.margin_source.is_human_authored
    ):
        conn.execute(
            "UPDATE designs SET approved_by=NULL WHERE id=?", (revision.design_id,)
        )


def list_revisions(design_id: str) -> list[DesignRevision]:
    with db() as conn:
        current = _current_revision_id(conn, design_id)
        rows = conn.execute(
            "SELECT * FROM design_revisions WHERE design_id=? ORDER BY revision_no",
            (design_id,),
        ).fetchall()
    return [DesignRevision.from_row(r, current_id=current) for r in rows]


def get_revision(design_id: str, revision_no: int) -> DesignRevision | None:
    with db() as conn:
        current = _current_revision_id(conn, design_id)
        row = conn.execute(
            "SELECT * FROM design_revisions WHERE design_id=? AND revision_no=?",
            (design_id, revision_no),
        ).fetchone()
    return DesignRevision.from_row(row, current_id=current) if row else None


def current_margin(design_id: str) -> DesignRevision | None:
    with db() as conn:
        current = _current_revision_id(conn, design_id)
        if not current:
            return None
        row = conn.execute(
            "SELECT * FROM design_revisions WHERE id=?", (current,)
        ).fetchone()
    return DesignRevision.from_row(row, current_id=current) if row else None


def accept_revision(design_id: str, revision_no: int, *, approved_by: str | None = None) -> DesignRevision:
    """把某个修订提升为当前生效版本。"""
    revision = get_revision(design_id, revision_no)
    if revision is None:
        raise KeyError(f"design {design_id} 没有修订 #{revision_no}")
    if revision.kind is not RevisionKind.MARGIN_CANDIDATE:
        raise ValueError('采纳接口只允许 MARGIN_CANDIDATE；其他修订请使用各自确认接口')
    with db() as conn:
        _set_current_margin(conn, revision,confirmed=True)
        if approved_by:
            conn.execute(
                "UPDATE design_revisions SET approved_by=? WHERE id=?",
                (approved_by, revision.id),
            )
            conn.execute(
                "UPDATE designs SET approved_by=? WHERE id=?", (approved_by, design_id)
            )
    revision.is_current = True
    revision.approved_by = approved_by or revision.approved_by
    return revision


def record_manual_margin(
    design_id: str,
    points: list[list[float]],
    *,
    operator: str | None = None,
    note: str | None = None,
    prior_source: MarginSource | None = None,
    boundary_semantics: str | None = None,
) -> DesignRevision:
    """记录一次人工 Margin 绘制/编辑，并立即设为当前生效版本。

    人工结果一旦产生就立刻落库 —— 这是修掉"手绘 Margin 只有走到 finalize 才保存"
    那个断链的地方。

    ``boundary_semantics``（例如 ``NATURAL_PREP_MARGIN`` / ``ABUTMENT_CROWN_FINISH_LINE``）
    一并写进 payload：确认之后这条修订必须自证它是哪种边界。
    """
    source = (MarginSource.AUTO_THEN_MANUAL
              if prior_source is not None and prior_source is MarginSource.AUTO
              else MarginSource.MANUAL)
    revision = DesignRevision(
        design_id=design_id,
        revision_no=0,
        kind=RevisionKind.MARGIN_MANUAL,
        payload=margin_payload(points, note=note, boundary_semantics=boundary_semantics),
        margin_source=source,
        operator_actions=[{"action": "set_margin", "point_count": len(points), "at": _now(),
                           "boundary_semantics": boundary_semantics}],
    )
    return append_revision(revision, make_current=True, operator=operator)


def record_auto_margin_candidate(
    design_id: str,
    points: list[list[float]],
    *,
    algorithm_version: str | None = None,
    operator: str | None = None,
    note: str | None = None,
) -> tuple[DesignRevision, bool]:
    """记录一次自动 Margin 提案。

    返回 ``(revision, applied)``：

    - ``applied=True``  —— 之前没有生效的 Margin，或生效的是自动结果，
      提案直接成为当前生效版本（重跑同一算法不算覆盖人工作品）。
    - ``applied=False`` —— 当前生效版本含人工成分，因此**只记录为候选**，
      供调用方决定 Replace / Compare / Cancel。
    """
    prior = current_margin(design_id)
    prior_is_human = prior is not None and (
        prior.margin_source is not None and prior.margin_source.is_human_authored
    )

    if prior_is_human:
        revision = DesignRevision(
            design_id=design_id,
            revision_no=0,
            kind=RevisionKind.MARGIN_CANDIDATE,
            payload=margin_payload(
                points,
                note=note or "自动提案；当前生效版本为人工结果，需由用户决定是否替换",
            ),
            algorithm_version=algorithm_version,
            margin_source=MarginSource.AUTO,
            operator_actions=[{
                "action": "suggest_margin",
                "point_count": len(points),
                "at": _now(),
                "supersedes_candidate_of": prior.id,
            }],
        )
        return append_revision(revision, make_current=False, operator=operator), False

    revision = DesignRevision(
        design_id=design_id,
        revision_no=0,
        kind=RevisionKind.MARGIN_AUTO_ACCEPTED,
        payload=margin_payload(points, note=note),
        algorithm_version=algorithm_version,
        margin_source=MarginSource.AUTO,
        operator_actions=[{"action": "suggest_margin", "point_count": len(points), "at": _now()}],
    )
    return append_revision(revision, make_current=True, operator=operator), True


def record_axis(
    design_id: str,
    axis: list[float],
    *,
    source: str,
    algorithm_version: str | None = None,
    operator: str | None = None,
    note: str | None = None,
    approved_by: str | None = None,
    result_payload: dict | None = None,
    based_on_candidate: int | None = None,
    based_on_solver: dict | None = None,
    based_on_reasons: list | None = None,
) -> DesignRevision:
    """记录一次就位方向变更。

    ``source="auto"`` 登记为候选（``AXIS_CANDIDATE``）；``source="manual"`` 表示
    人工确认或调整，登记为 ``AXIS_ACCEPTED`` 并记录确认人。
    """
    is_manual = source != "auto"
    payload: dict = {"type": "axis", "axis": axis, "source": source}
    if note:
        payload["note"] = note
    if result_payload:
        # 保留求解器的状态与倒凹指标，使审计链能回答"当时为什么接受这个方向"
        payload["solver"] = {
            k: result_payload.get(k) for k in
            ("status", "confidence", "score", "undercut_area_ratio",
             "max_undercut_depth_mm", "margin_visibility_ratio", "deviation_deg",
             "objective_is_flat")
        }
        payload["solver_reasons"] = (result_payload.get("diagnostics") or {}).get("reasons", [])

    if based_on_candidate is not None:
        # 人工采纳时把候选当时的求解结论一并存档 —— 审计要能回答"为什么接受它"
        payload["basis"] = {
            "candidate_revision_no": based_on_candidate,
            "solver": based_on_solver,
            "solver_reasons": based_on_reasons or [],
        }

    revision = DesignRevision(
        design_id=design_id,
        revision_no=0,
        kind=(RevisionKind.AXIS_ACCEPTED if is_manual else RevisionKind.AXIS_CANDIDATE),
        payload=payload,
        algorithm_version=algorithm_version,
        approved_by=approved_by,
        operator_actions=[{"action": "set_axis", "source": source, "at": _now()}],
    )
    return append_revision(revision, make_current=False, operator=operator)


def latest_axis_candidate(design_id: str) -> DesignRevision | None:
    """最近一条自动求解的就位方向候选。

    用途：人工确认方向时把"当时求解器怎么报的"一同写入**接受修订**，
    否则审计链只能回答"方向被改成什么"，回答不了"为什么接受它"
    —— 而后者才是复核时真正要看的。
    """
    with db() as conn:
        row = conn.execute(
            """SELECT * FROM design_revisions
                WHERE design_id=? AND kind=?
                ORDER BY revision_no DESC LIMIT 1""",
            (design_id, RevisionKind.AXIS_CANDIDATE.value),
        ).fetchone()
    return DesignRevision.from_row(row) if row else None


def revision_chain(design_id: str) -> dict[str, Any]:
    """返回完整修订链摘要，供审计与前端"比较"功能使用。

    ``requires_decision`` 的判定：只有**尚未被消费**的候选才算待决定。
    一个候选如果已被后续某个接受修订通过 ``payload.basis.candidate_revision_no``
    引用，说明它已经被处理过（采纳或明确放弃），不应再让前端弹决策框 ——
    否则每次采纳后都会永久显示"有待决定候选"（实测踩到过）。
    """
    revisions = list_revisions(design_id)
    current = next((r for r in revisions if r.is_current), None)

    consumed = {
        (r.payload.get("basis") or {}).get("candidate_revision_no")
        for r in revisions if not r.is_candidate
    }
    candidates = [
        r for r in revisions
        if r.is_candidate and not r.is_current and r.revision_no not in consumed
    ]

    # ── 外形修订游标（P1-9）：撤销/重做必须线性、可预测，不能靠 ``is_current`` 猜。
    # 当前外形 = 设计行 ``outer_path`` 对得上的那个外形修订（路径不符时按内容 digest 兜底）。
    outers = sorted((r for r in revisions if r.kind == RevisionKind.OUTER), key=lambda r: r.revision_no)
    cursor = _outer_cursor(design_id, outers)
    position = outers.index(cursor) if cursor is not None and cursor in outers else None

    return {
        "design_id": design_id,
        "n_revisions": len(revisions),
        "current_margin_revision_no": current.revision_no if current else None,
        "current_margin_source": (current.margin_source.value
                                  if current and current.margin_source else None),
        "pending_candidates": [r.revision_no for r in candidates],
        "consumed_candidates": sorted(x for x in consumed if x is not None),
        "requires_decision": bool(candidates),
        "outer_history": [r.revision_no for r in outers],
        "current_outer_revision_no": cursor.revision_no if cursor is not None else None,
        "previous_outer_revision_no": outers[position - 1].revision_no if position else None,
        "next_outer_revision_no": (outers[position + 1].revision_no
                                   if position is not None and position + 1 < len(outers) else None),
        "can_undo_outer": bool(position), "can_redo_outer": position is not None and position + 1 < len(outers),
        "revisions": [r.as_dict() for r in revisions],
    }


def _outer_cursor(design_id, outers):
    """当前外形修订 = ``designs.outer_path`` 对应的那条外形修订。"""
    if not outers:
        return None
    from ..db import db
    with db() as conn:
        row = conn.execute("SELECT outer_path FROM designs WHERE id=?", (design_id,)).fetchone()
    path = (dict(row).get("outer_path") if row else None) or None
    if path:
        for revision in reversed(outers):
            if revision.artifacts.get("outer") == path:
                return revision
    digest = _digest(path)
    if digest:
        for revision in reversed(outers):
            payload = revision.payload or {}
            stored = payload.get("output_sha256") or payload.get("after_sha256")
            artifact = revision.artifacts.get("outer")
            if stored == digest or (artifact and _digest(artifact) == digest):
                return revision
    return outers[-1]


def _digest(path):
    from pathlib import Path
    from ..services.design_dependencies import file_digest
    try:
        return file_digest(path) if path and Path(path).is_file() else None
    except OSError:
        return None
