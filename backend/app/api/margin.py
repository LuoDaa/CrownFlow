"""Margin 提案会话 API（Margin 步骤重构，2026-10-13）。

六个端点，全部**只操作会话**（进程内、不落库）；只有 ``/margin/confirm`` 会写正式
Margin revision，且复用既有 ``record_manual_margin`` 那条 truth-source 路径：

    POST /designs/{id}/margin/detect           自动候选（Top-K，不覆盖正式 Margin）
    POST /designs/{id}/margin/add-anchor       加 1–4 个纠偏点（硬约束局部重描）
    POST /designs/{id}/margin/replace-segment  磁性修线（只替换 stroke 覆盖区间）
    POST /designs/{id}/margin/move             整体上移/下移（沿局部表面+插入轴，非 world-Z）
    POST /designs/{id}/margin/smooth           平滑度（保锁）
    POST /designs/{id}/margin/confirm          确认 → 正式 Margin revision

与既有 ``/auto-margin-m1`` / ``/margin-snap`` / ``/section-preview`` 的关系：
前者是**算法入口**（本模块直接调用它们背后的引擎/吸附），后者继续独立可用；
本模块不复制任何几何算法。
"""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .routes import design_row, asset
from ..services import margin_session as session_service
from ..services.design_dependencies import file_digest
from ..services.prep_conditioning import Indication

router = APIRouter(prefix="/api/designs")


class DetectRequest(BaseModel):
    feature_mode: Literal["AUTO", "SUPRAGINGIVAL", "SUBGINGIVAL"] = "AUTO"
    seed_point: list[float] | None = None
    radius_mm: float = Field(default=1.0, ge=0.1, le=5.0)
    apply_snap: bool = True
    top_k: int = Field(default=3, ge=1, le=5)


class AnchorRequest(BaseModel):
    anchors: list[list[float]] = Field(min_length=1, max_length=4)
    feature_mode: Literal["AUTO", "SUPRAGINGIVAL", "SUBGINGIVAL"] = "AUTO"
    radius_mm: float = Field(default=1.0, ge=0.1, le=5.0)


class SegmentRequest(BaseModel):
    stroke: list[list[float]] = Field(min_length=2)
    radius_mm: float = Field(default=1.5, ge=0.1, le=5.0)
    feature_mode: Literal["AUTO", "SUPRAGINGIVAL", "SUBGINGIVAL"] = "AUTO"


class MoveRequest(BaseModel):
    delta_mm: float = Field(ge=-3.0, le=3.0)


class SmoothRequest(BaseModel):
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    locked_index: list[int] = Field(default_factory=list)


class ConfirmRequest(BaseModel):
    expected_input_digest: str | None = None
    operator: str | None = None
    #: 人类显式选定的候选（存在并列强候选时必须给）
    proposal_id: str | None = None


#: 会话必须绑定在**同一份基牙网格 + 同一份已确认轴**上，否则候选就不可信了。
def _axis_for(d) -> list[float]:
    axis = json.loads(d.get("axis_json") or "null")
    if not axis or len(axis) != 3:
        raise HTTPException(409, "请先确认插入轴（会话必须绑定在已确认轴上）")
    return [float(v) for v in axis]


def _support_for(d):
    support = asset(d["case_id"], "support")
    if not support:
        raise HTTPException(409, "请先定位目标基牙支撑（Margin 检测需要基牙网格）")
    return support


def _indication_for(d) -> str:
    """指征分流：天然牙与已有实体基台共用同一套边界流程（语义不同、能力相同）。"""
    from ..services.abutment_support import indication_of
    return session_service.normalize_indication(indication_of(d))


def _guard_indication(d) -> str:
    """放行天然牙与已有实体基台；只有自定义种植基台与"无语义指征"被拒。

    返回归一化后的 indication；边界语义由 ``boundary_semantics_for`` 判定（不猜）。
    """
    indication = _indication_for(d)
    if indication=='CROWN_ON_EXISTING_ABUTMENT':
        support=asset(d['case_id'],'support')
        details=json.loads(support['mesh_info_json'] or '{}') if support else {}
        if details.get('implant_candidate') and not details.get('restorative_abutment_confirmed'):
            raise HTTPException(409,'请先确认实际修复基台；扫描杆不能作为 Crown Finish Line 的来源')
    try:
        session_service.boundary_semantics_for(indication)
    except session_service.UnsupportedIndication as exc:
        raise HTTPException(409, {
            "message": "自定义种植基台不进边界候选流程（1.0 支持范围外的指征）",
            "status": exc.status, "indication": indication,
            "alternative": "请提供设计者确认的最终修复基台几何；已有实体基台可直接使用本接口"}) from exc
    except session_service.BoundarySemanticsNotDefined as exc:
        raise HTTPException(409, {
            "message": ("该指征的支撑/边界/螺丝通道语义与已有实体基台不同，当前没有定义相应的 "
                        "BoundarySemantics（边界语义），因此 fail-loud；不拿天然牙或基台语义顶替"),
            "status": exc.status, "indication": indication,
            "alternative": "需要先为该指征定义 boundary_semantics 才能开放"}) from exc
    return indication


def _report(session) -> dict:
    payload = session.as_report()
    payload["release_status"] = "NOT_PRODUCTION_APPROVED"
    payload["note"] = ("候选只存在于本会话，**不是**正式 Margin；confirm 才写 revision。"
                       "boundary_semantics=%s（%s）"
                       % (payload.get("boundary_semantics"), payload.get("boundary_label")))
    return payload


@router.post("/{design_id}/margin/detect")
def margin_detect(design_id: str, p: DetectRequest):
    d = design_row(design_id)
    indication = _guard_indication(d)
    support = _support_for(d)
    axis = _axis_for(d)
    try:
        session = session_service.detect(
            design_id, support_path=support["path"], axis=axis,
            source_mesh_sha256=file_digest(support["path"]),
            feature_mode=p.feature_mode, seed_point=p.seed_point,
            radius_mm=p.radius_mm, apply_snap=p.apply_snap, top_k=p.top_k,
            indication=indication)
    except session_service.UnsupportedIndication as exc:
        raise HTTPException(409, {"message": str(exc), "status": exc.status,
                                  "indication": exc.indication}) from exc
    except session_service.BoundarySemanticsNotDefined as exc:
        raise HTTPException(409, {"message": str(exc), "status": exc.status,
                                  "indication": exc.indication}) from exc
    return _report(session)


@router.post("/{design_id}/margin/add-anchor")
def margin_add_anchor(design_id: str, p: AnchorRequest):
    d = design_row(design_id)
    indication = _guard_indication(d)
    support = _support_for(d)
    axis = _axis_for(d)
    try:
        session = session_service.add_anchors(
            design_id, p.anchors, support_path=support["path"], axis=axis,
            source_mesh_sha256=file_digest(support["path"]),
            feature_mode=p.feature_mode, radius_mm=p.radius_mm, indication=indication)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(409, "尚无 Margin 会话，请先自动检测") from exc
    return _report(session)


@router.post("/{design_id}/margin/replace-segment")
def margin_replace_segment(design_id: str, p: SegmentRequest):
    d = design_row(design_id)
    indication = _guard_indication(d)
    support = _support_for(d)
    axis = _axis_for(d)
    try:
        session = session_service.replace_segment(
            design_id, p.stroke, support_path=support["path"], axis=axis,
            source_mesh_sha256=file_digest(support["path"]),
            radius_mm=p.radius_mm, feature_mode=p.feature_mode, indication=indication)
    except KeyError as exc:
        raise HTTPException(409, "尚无 Margin 会话，请先自动检测") from exc
    return _report(session)


@router.post("/{design_id}/margin/move")
def margin_move(design_id: str, p: MoveRequest):
    d = design_row(design_id)
    indication = _guard_indication(d)
    support = _support_for(d)
    axis = _axis_for(d)
    try:
        session = session_service.move(
            design_id, p.delta_mm, support_path=support["path"], axis=axis,
            source_mesh_sha256=file_digest(support["path"]), indication=indication)
    except KeyError as exc:
        raise HTTPException(409, "尚无 Margin 会话，请先自动检测") from exc
    return _report(session)


@router.post("/{design_id}/margin/smooth")
def margin_smooth(design_id: str, p: SmoothRequest):
    d = design_row(design_id)
    indication = _guard_indication(d)
    support = _support_for(d)
    axis = _axis_for(d)
    try:
        session = session_service.smooth(
            design_id, strength=p.strength, locked_index=p.locked_index,
            support_path=support["path"], axis=axis,
            source_mesh_sha256=file_digest(support["path"]), indication=indication)
    except KeyError as exc:
        raise HTTPException(409, "尚无 Margin 会话，请先自动检测") from exc
    return _report(session)


@router.post("/{design_id}/margin/confirm")
def margin_confirm(design_id: str, p: ConfirmRequest):
    """确认候选 → 正式 Margin revision（**唯一**写状态的动作）。"""
    d = design_row(design_id)
    _guard_indication(d)
    support = _support_for(d)
    sha = file_digest(support["path"])
    if p.expected_input_digest:
        from ..services.design_dependencies import dependency_digest
        if dependency_digest(d) != p.expected_input_digest:
            raise HTTPException(409, "确认请求引用的输入版本已过期，请重新载入")
    try:
        result = session_service.confirm(design_id, operator=p.operator,
                                         source_mesh_sha256=sha, proposal_id=p.proposal_id)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(409, "尚无 Margin 会话，请先自动检测") from exc
    return {**result, "release_status": "NOT_PRODUCTION_APPROVED",
            "note": "已生成正式 Margin revision；候选会话已关闭"}
