"""Margin Proposal Session（Margin 步骤重构，2026-10-13）。

目标：把 Margin 做成 exocad/3Shape 那种"**自动候选 + 少量纠偏点 + 局部快速修线**"的工作流。

纪律（不得违反）：

1. **不重写 M1 求解器**：自动检测一律调 `geometry.auto_margin.engine.predict_margin`
   （冻结配置），本模块只做编排、候选排名、锚点局部重描与局部编辑；
2. **proposal 不覆盖正式 Margin**：候选只活在本 session 里（进程内，带 TTL），
   只有 `confirm` 才走既有 `POST /margin` 那条路生成**正式 revision**；
3. **不自信地选错**：多个强候选"分数接近但形状不同"时返回 `REVIEW_REQUIRED`；
4. **indication 感知**：同一条曲线在不同指征下含义不同，会话必须显式携带
   ``boundary_semantics``（天然牙预备体边缘线 vs 已有实体基台上的冠终止线）。
   天然牙与**已有实体基台**共用同一套 detect/Top-K/anchor/magnetic/edit/move/smooth/confirm；
   只有 ``CUSTOM_IMPLANT_ABUTMENT`` 被拒（`UNSUPPORTED_INDICATION`），
   没有定义边界语义的指征（如螺丝固位冠的平台边界）被拒（`BOUNDARY_SEMANTICS_NOT_DEFINED`）；
5. 锚点是**硬约束**：`retrace_with_anchors` 保证曲线逐点穿过锚点投影。

session 不落库、不写文件；进程重启丢失是**刻意的**（候选不是真相）。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SEMANTICS_NATURAL = "NATURAL_PREP_MARGIN"
SEMANTICS_ABUTMENT_FINISH_LINE = "ABUTMENT_CROWN_FINISH_LINE"

#: indication → 这条边界的语义（**不是**从 support_type 猜出来的）
BOUNDARY_SEMANTICS = {
    "NATURAL_PREP": SEMANTICS_NATURAL,
    "NATURAL_TOOTH_SINGLE_ANATOMICAL_CROWN": SEMANTICS_NATURAL,
    "CROWN_ON_EXISTING_ABUTMENT": SEMANTICS_ABUTMENT_FINISH_LINE,
}

#: 给前端/报告用的中文标签（措辞的唯一来源）
BOUNDARY_LABELS = {
    SEMANTICS_NATURAL: "天然牙边缘线",
    SEMANTICS_ABUTMENT_FINISH_LINE: "基台冠终止线",
}

DEFAULT_INDICATION = "NATURAL_PREP"

#: 只有它被拒（1.0 范围外）
UNSUPPORTED_INDICATIONS = {"CUSTOM_IMPLANT_ABUTMENT"}
#: 同一族的历史别名：归一化后仍然是 CUSTOM_IMPLANT_ABUTMENT（不新增第二类拒因）
INDICATION_ALIASES = {"IMPLANT_ABUTMENT": "CUSTOM_IMPLANT_ABUTMENT",
                      "IMPLANT_ABUTMENT_WITH_SCREW_CHANNEL": "CUSTOM_IMPLANT_ABUTMENT"}
#: 语义与本流程不同、且尚未定义 BoundarySemantics 的指征：fail-loud，不静默当天然牙处理。
#: 注意：这只说明"未定义"，不代表已知它对应某条特定的几何线。
NO_BOUNDARY_SEMANTICS = {"SCREW_RETAINED_CROWN"}


class UnsupportedIndication(Exception):
    """该 indication 不进边界候选流程（例如自定义种植基台）。"""

    status = "UNSUPPORTED_INDICATION"

    def __init__(self, indication: str):
        super().__init__("UNSUPPORTED_INDICATION:%s" % indication)
        self.indication = indication


class BoundarySemanticsNotDefined(Exception):
    """该 indication 的支撑/边界/螺丝通道语义与本流程不同，且**尚未定义**相应 BoundarySemantics。

    例如螺丝固位冠：它是直接安装在植体上、另有螺丝通道工作流的另一种 construction，
    支撑与边界语义都不能沿用 `CROWN_ON_EXISTING_ABUTMENT`。这里只声明"没有定义"，
    不预先规定它对应哪一条几何线。"""

    status = "BOUNDARY_SEMANTICS_NOT_DEFINED"

    def __init__(self, indication: str):
        super().__init__("BOUNDARY_SEMANTICS_NOT_DEFINED:%s" % indication)
        self.indication = indication


def normalize_indication(indication: str | None) -> str:
    """别名归一化；缺省按天然牙预备体处理（live 的既有默认）。"""
    name = indication or DEFAULT_INDICATION
    return INDICATION_ALIASES.get(name, name)


def boundary_semantics_for(indication: str | None) -> str:
    """给出该指征下的边界语义；没有定义就**拒绝**，绝不静默找个默认语义。"""
    name = normalize_indication(indication)
    if name in UNSUPPORTED_INDICATIONS:
        raise UnsupportedIndication(name)
    if name in NO_BOUNDARY_SEMANTICS:
        raise BoundarySemanticsNotDefined(name)
    semantics = BOUNDARY_SEMANTICS.get(name)
    if semantics is None:
        raise BoundarySemanticsNotDefined(name)
    return semantics

#: 候选"分数接近"的相对阈值：最优与次优的分数差小于它 ⇒ 视为强候选并列
AMBIGUITY_SCORE_REL = 0.05
#: 并列且形状差异超过它（p95 点距，mm）⇒ 必须人工确认
AMBIGUITY_SHAPE_P95_MM = 0.35
#: session 存活时间（秒）
SESSION_TTL_S = 3600

KIND_AUTO = "AUTO"
KIND_SUPRAGINGIVAL = "SUPRAGINGIVAL"
KIND_SUBGINGIVAL = "SUBGINGIVAL"
FEATURE_MODES = (KIND_AUTO, KIND_SUPRAGINGIVAL, KIND_SUBGINGIVAL)

#: 前端 feature_mode → `margin_feature_snap` 的模式
_SNAP_MODE = {KIND_SUPRAGINGIVAL: "SUPRA", KIND_SUBGINGIVAL: "SUB"}


@dataclass
class MarginProposal:
    """一个候选闭环（**不是**正式 Margin）。"""

    proposal_id: str
    points: list
    score: float
    rank: int
    source: str                       # AUTO_M1 / MODE_VARIANT / ANCHOR_RETRACE / LOCAL_EDIT
    status: str                       # MEASURED / REVIEW_REQUIRED / NOT_ASSESSED
    feature_mode: str = KIND_AUTO
    rank_stats: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.points or [])


@dataclass
class BoundarySession:
    """一次 Margin 提案会话（进程内；不落库、不写文件）。"""

    session_id: str
    design_id: str
    source_mesh_sha256: str
    algorithm_version: str
    axis: list
    #: 指征与它决定的边界语义：confirm 时必须原样写进 revision。
    indication: str = DEFAULT_INDICATION
    boundary_semantics: str = SEMANTICS_NATURAL
    seed_point: list | None = None
    anchors: list = field(default_factory=list)
    feature_mode: str = KIND_AUTO
    proposals: list = field(default_factory=list)
    active_proposal_id: str | None = None
    ambiguity_flags: list = field(default_factory=list)
    status: str = "PROPOSED"
    operations: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---------------------------------------------------------------- 内部
    def _touch(self) -> None:
        self.updated_at = time.time()

    def _log(self, action: str, **detail) -> None:
        self.operations.append({"action": action, "at": self.updated_at, **detail})
        self._touch()

    def _anchor_index_hint(self) -> tuple:
        """锚点最近的环索引（去重时保护它们；没有锚点就返回空）。"""
        return ()

    def active(self) -> MarginProposal | None:
        if not self.proposals:
            return None
        if self.active_proposal_id is None:
            return self.proposals[0]
        return next((p for p in self.proposals if p.proposal_id == self.active_proposal_id),
                    self.proposals[0])

    def set_active(self, proposal_id: str) -> MarginProposal:
        found = next((p for p in self.proposals if p.proposal_id == proposal_id), None)
        if found is None:
            raise KeyError(proposal_id)
        self.active_proposal_id = proposal_id
        self._log("select_proposal", proposal_id=proposal_id)
        return found

    def add(self, proposal: MarginProposal) -> MarginProposal:
        from ..geometry.margin_edit import dedupe_ring

        cleaned = dedupe_ring(proposal.points)
        if cleaned["dropped_index"]:
            # 入库前去重：候选必须一眼就是"校验器能接受的形状"
            proposal.notes.append("入库前合并 %d 个过近点" % len(cleaned["dropped_index"]))
            proposal.points = [list(map(float, point)) for point in cleaned["points"]]
        self.proposals.append(proposal)
        self.proposals.sort(key=lambda p: (-p.score, p.proposal_id))
        for i, item in enumerate(self.proposals, start=1):
            item.rank = i
        if self.active_proposal_id is None:
            self.active_proposal_id = self.proposals[0].proposal_id
        self._touch()
        return proposal

    def replace_active_points(self, points, *, source: str, note: str = "") -> MarginProposal:
        current = self.active()
        if current is None:
            raise KeyError("no active proposal")
        updated = MarginProposal(
            proposal_id=current.proposal_id, points=[list(map(float, p)) for p in points],
            score=current.score, rank=current.rank, source=source, status=current.status,
            feature_mode=current.feature_mode, rank_stats=dict(current.rank_stats),
            confidence=dict(current.confidence), notes=list(current.notes) + ([note] if note else []))
        self.proposals = [updated if p.proposal_id == current.proposal_id else p
                          for p in self.proposals]
        self._log(source.lower(), n_points=updated.n_points, note=note)
        return updated

    def as_report(self) -> dict:
        return {"session_id": self.session_id, "design_id": self.design_id,
                "source_mesh_sha256": self.source_mesh_sha256,
                "algorithm_version": self.algorithm_version,
                # 边界语义随会话走：前端措辞与 revision 记录都以它为准
                "indication": self.indication,
                "boundary_semantics": self.boundary_semantics,
                "boundary_label": BOUNDARY_LABELS.get(self.boundary_semantics),
                "seed_point": self.seed_point, "anchors": self.anchors,
                "feature_mode": self.feature_mode,
                "active_proposal_id": self.active_proposal_id,
                "ambiguity_flags": list(self.ambiguity_flags),
                "status": self.status, "operations": list(self.operations),
                # 报表按 rank 排序列出全部候选；同时给出**当前活跃候选**的点，
                # 免得调用方在"同分并列"时自己猜哪一个才是编辑后的结果。
                "active_points": (self.active().points if self.active() else []),
                "proposals": [{**asdict(p), "n_points": p.n_points} for p in self.proposals]}


#: 兼容别名：老的导入名仍然可用（本模块的类已改名为 BoundarySession）。
MarginSession = BoundarySession


class _SessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._by_design: dict[str, BoundarySession] = {}

    def put(self, session: BoundarySession) -> BoundarySession:
        with self._lock:
            self._by_design[session.design_id] = session
            return session

    def get(self, design_id: str) -> BoundarySession | None:
        with self._lock:
            session = self._by_design.get(design_id)
            if session and time.time() - session.updated_at > SESSION_TTL_S:
                self._by_design.pop(design_id, None)
                return None
            return session

    def clear(self, design_id: str) -> None:
        with self._lock:
            self._by_design.pop(design_id, None)


STORE = _SessionStore()


# ---------------------------------------------------------------- 几何入口
def _m1_proposal(tris, axis, feature_mode: str, *, radius_mm: float, apply_snap: bool,
                 source: str) -> MarginProposal | None:
    """跑一次 M1（冻结配置）+（可选）特征吸附，产出一个候选。"""
    result = auto_margin_m1_paths(tris, axis, feature_mode, radius_mm, apply_snap)
    if result is None:
        return None
    points, status, score, rank_stats, confidence, notes = result
    return MarginProposal(proposal_id=uuid.uuid4().hex, points=points, score=float(score),
                          rank=0, source=source, status=status, feature_mode=feature_mode,
                          rank_stats=rank_stats, confidence=confidence, notes=notes)


def auto_margin_m1_paths(tris, axis, feature_mode: str, radius_mm: float, apply_snap: bool):
    """调 M1 引擎（冻结配置）并返回 ``(points, status, score, rank_stats, confidence, notes)``。

    这里**不复制**求解逻辑：只把引擎的输出搬过来，并按 ``feature_mode`` 做一次
    已有 ``snap_to_feature``（SUPRA/SUB）；AUTO 档不做吸附（保持引擎原样）。
    """
    from ..geometry.auto_margin.engine import predict_margin
    from ..geometry.auto_margin.models import STATUS_AUTO_ACCEPTED
    from ..geometry.margin_feature_snap import snap_to_feature

    result = predict_margin(np.asarray(tris, float), np.asarray(axis, float), return_candidates=True)
    notes = [result.reason] if result.reason else []
    if result.status != STATUS_AUTO_ACCEPTED or result.margin is None or not result.margin.points:
        return [], result.status, 0.0, dict(result.candidate_rank_stats), dict(result.confidence), notes
    points = [[float(v) for v in p] for p in result.margin.points]
    snap_mode = _SNAP_MODE.get(feature_mode)
    if apply_snap and snap_mode:
        snapped = snap_to_feature(np.asarray(tris, float), points, axis, snap_mode, radius_mm)
        if snapped.get("status") == "APPLIED":
            points = snapped["points"]
        else:
            notes.append("snap:%s" % snapped.get("reason"))
    # **归一化正分数**（0..1，越大越好）：直接用引擎的能量值会是负的、读起来像"负分"。
    # 只用几何量（能量/能量差/胜出裕度），不看 GT。
    energy = float(result.loop_energy or 0.0)
    margin = float(result.energy_gap or 0.0)
    score = 1.0 / (1.0 + max(energy, 0.0)) + 0.5 * (1.0 - 1.0 / (1.0 + max(margin, 0.0)))
    score = float(min(score, 1.0))
    return points, result.status, score, dict(result.candidate_rank_stats), dict(result.confidence), notes


def _load_support(support_path):
    """``asset().path`` 是字符串；``load_mesh`` 要 ``Path``。"""
    from ..geometry.io import load_mesh

    return load_mesh(Path(support_path))


def _shape_p95(a, b) -> float:
    """两条闭环的最短点距 p95（mm）——用来判断"分数接近但形状不同"。"""
    from scipy.spatial import cKDTree

    pa, pb = np.asarray(a, float), np.asarray(b, float)
    if not len(pa) or not len(pb):
        return float("inf")
    d1, _ = cKDTree(pb).query(pa, k=1, workers=-1)
    d2, _ = cKDTree(pa).query(pb, k=1, workers=-1)
    return float(np.quantile(np.r_[d1, d2], 0.95))


def detect(design_id: str, *, support_path, axis, source_mesh_sha256: str,
           feature_mode: str = KIND_AUTO, seed_point=None, radius_mm: float = 1.0,
           apply_snap: bool = True, algorithm_version: str = "welooai/auto-margin-m1",
           top_k: int = 3, indication: str | None = None) -> BoundarySession:
    """自动检测：产出 Top-K 候选（**不覆盖正式 Margin**），并给出歧义旗标。"""
    from ..geometry.io import load_mesh

    name = normalize_indication(indication)
    semantics = boundary_semantics_for(name)

    mesh = _load_support(support_path)
    if name == 'CROWN_ON_EXISTING_ABUTMENT':
        from ..geometry.abutment_margin import extract_abutment_margin, ALGORITHM_VERSION
        points, quality = extract_abutment_margin(mesh, axis)
        session = BoundarySession(session_id=uuid.uuid4().hex, design_id=design_id,
                                  source_mesh_sha256=source_mesh_sha256,
                                  algorithm_version=ALGORITHM_VERSION, axis=quality['axis'],
                                  indication=name, boundary_semantics=semantics,
                                  seed_point=list(map(float, seed_point)) if seed_point else None,
                                  feature_mode=feature_mode)
        proposal = MarginProposal(proposal_id=uuid.uuid4().hex, points=points, score=1.0,
                                  rank=0, source='ABUTMENT_SHOULDER', status='REVIEW_REQUIRED',
                                  feature_mode=feature_mode, rank_stats=quality,
                                  confidence={'clinical_accuracy_assessed': False},
                                  notes=['非平面肩部外缘候选，需要人工确认'])
        session.add(proposal)
        session.active_proposal_id = proposal.proposal_id
        session.status = 'REVIEW_REQUIRED'
        session._log('detect', n_proposals=1, method=ALGORITHM_VERSION)
        return STORE.put(session)
    tris = np.asarray(mesh.triangles, float)
    modes = [feature_mode]
    for fallback in (KIND_SUPRAGINGIVAL, KIND_SUBGINGIVAL):
        if fallback != feature_mode and len(modes) < max(2, top_k):
            modes.append(fallback)

    session = BoundarySession(session_id=uuid.uuid4().hex, design_id=design_id,
                            source_mesh_sha256=source_mesh_sha256,
                            algorithm_version=algorithm_version,
                            axis=[float(v) for v in np.asarray(axis, float)],
                            indication=name, boundary_semantics=semantics,
                            seed_point=list(map(float, seed_point)) if seed_point else None,
                            feature_mode=feature_mode)
    for mode in modes:
        proposal = _m1_proposal(tris, axis, mode, radius_mm=radius_mm, apply_snap=apply_snap,
                                source="AUTO_M1" if mode == feature_mode else "MODE_VARIANT")
        if proposal is not None:
            session.add(proposal)

    if not session.proposals:
        session.status = "REVIEW_REQUIRED"
        session.ambiguity_flags.append("NO_PROPOSAL（引擎未给出可接受闭环）")
        return STORE.put(session)

    best = session.proposals[0]
    session.active_proposal_id = best.proposal_id
    # 歧义判定：分数接近 **且** 形状差异显著 ⇒ 必须人工确认
    for other in session.proposals[1:]:
        close_score = abs(best.score - other.score) <= AMBIGUITY_SCORE_REL * max(abs(best.score), 1e-9)
        shape_gap = _shape_p95(best.points, other.points)
        if close_score and shape_gap > AMBIGUITY_SHAPE_P95_MM:
            session.ambiguity_flags.append(
                "STRONG_CANDIDATES_DISAGREE(score %.3f vs %.3f, p95 %.3f mm)"
                % (best.score, other.score, shape_gap))
    session.status = "REVIEW_REQUIRED" if (session.ambiguity_flags or
                                          best.status != "AUTO_ACCEPTED") else "PROPOSED"
    session._log("detect", n_proposals=len(session.proposals), feature_mode=feature_mode)
    return STORE.put(session)

# ---------------------------------------------------------------- 会话操作
def _current(design_id: str, *, support_path, axis, source_mesh_sha256: str,
             feature_mode: str = KIND_AUTO, radius_mm: float = 1.0,
             apply_snap: bool = True, algorithm_version: str = "welooai/auto-margin-m1",
             indication: str | None = None) -> BoundarySession:
    """取当前会话；没有（或基牙网格/指征已变）就先 detect（**不落库**）。"""
    session = STORE.get(design_id)
    if (session is not None and session.source_mesh_sha256 == source_mesh_sha256
            and session.indication == normalize_indication(indication)):
        return session
    return detect(design_id, support_path=support_path, axis=axis,
                  source_mesh_sha256=source_mesh_sha256, feature_mode=feature_mode,
                  radius_mm=radius_mm, apply_snap=apply_snap,
                  algorithm_version=algorithm_version, indication=indication)


def _active_points(session: BoundarySession) -> np.ndarray:
    active = session.active()
    if active is None or not active.points:
        raise KeyError("no active proposal")
    return np.asarray(active.points, float)


def _promote(session: BoundarySession, points, *, source: str, note: str,
             status: str = "MEASURED") -> bool:
    """把编辑结果作为**新候选**并**激活它**（旧候选仍可切回）。

    ⚠️ 必须显式把 ``active_proposal_id`` 指向**新候选**：编辑结果与自动候选分数相同时，
    按分数排序会把旧候选留在第一位，于是"编辑没生效"看起来像算法问题（实测踩到）。
    """
    from ..geometry.margin_edit import dedupe_ring

    # 统一去重：编辑操作可能把相邻点并到一起，而 Margin 校验器要求"点唯一 + 间距 ≥0.02mm"。
    # 在**唯一入口**处收拾，避免每个操作各自实现一遍（也保证确认时不再出现 400）。
    cleaned = dedupe_ring(points, keep_index=session._anchor_index_hint())
    original = len(np.asarray(points, float).reshape(-1, 3))
    if original and len(cleaned["dropped_index"]) > 0.2 * original:
        # 编辑把环压塌了（典型原因：在棱上沿表面移动）⇒ **拒绝这次编辑**，保留旧候选
        session.ambiguity_flags.append(
            "EDIT_COLLAPSED_RING(合并 %d/%d 点)" % (len(cleaned["dropped_index"]), original))
        session.status = "REVIEW_REQUIRED"
        session._log(source.lower(), rejected="collapsed_ring")
        return False
    # Reject an invalid edit before publishing it, preserving the last valid shoulder candidate.
    from ..api.routes import _validated_margin_points
    from fastapi import HTTPException
    try:
        validated = _validated_margin_points(cleaned['points'])
    except HTTPException as exc:
        session.ambiguity_flags.append('EDIT_INVALID_RING(%s)' % exc.detail)
        session.status = 'REVIEW_REQUIRED'
        session._log(source.lower(), rejected='invalid_ring', reason=exc.detail)
        return False
    note = note + ("；合并 %d 个过近点" % len(cleaned["dropped_index"])
                   if cleaned["dropped_index"] else "")
    proposal = MarginProposal(
        proposal_id=uuid.uuid4().hex,
        points=validated,
        score=1.0, rank=0, source=source, status=status,
        feature_mode=session.feature_mode, notes=[note])
    session.proposals.insert(0, proposal)
    session.proposals.sort(key=lambda item: (-item.score, item.proposal_id))
    for index, item in enumerate(session.proposals, start=1):
        item.rank = index
    session.active_proposal_id = proposal.proposal_id
    session._log(source.lower(), note=note)
    return True


def add_anchors(design_id: str, anchors, *, support_path, axis, source_mesh_sha256: str,
                feature_mode: str = KIND_AUTO, radius_mm: float = 1.0,
                apply_snap: bool = True, indication: str | None = None,
                max_anchors: int = 4) -> BoundarySession:
    """加 1–4 个纠偏点：锚点作为**硬约束**局部重描（只改锚点之间的弧）。"""
    from ..geometry.margin_edit import retrace_with_anchors

    session = _current(design_id, support_path=support_path, axis=axis,
                       source_mesh_sha256=source_mesh_sha256, feature_mode=feature_mode,
                       radius_mm=radius_mm, apply_snap=apply_snap, indication=indication)
    array = np.asarray(anchors, float).reshape(-1, 3)
    if not len(array):
        raise ValueError("至少需要一个 anchor 点")
    if len(array) > max_anchors:
        raise ValueError("anchor 最多 %d 个（当前 %d）" % (max_anchors, len(array)))
    session.anchors = [[float(v) for v in point] for point in array]
    mesh = _load_support(support_path)
    result = retrace_with_anchors(mesh, _active_points(session), array, axis=session.axis,
                                  mode=_SNAP_MODE.get(feature_mode, "SUPRA"), radius_mm=radius_mm)
    if result["status"] != "APPLIED":
        session.ambiguity_flags.append("ANCHOR_RETRACE_%s" % result.get("reason"))
        session.status = "REVIEW_REQUIRED"
        return STORE.put(session)
    if not _promote(session, result["points"], source="ANCHOR_RETRACE",
             note="锚点硬约束重描；改动 %d 点，锚点 %d 个"
                  % (len(result["moved_index"]), len(array))):
        return STORE.put(session)
    session.status = "PROPOSED"
    session.ambiguity_flags = []
    return STORE.put(session)


def replace_segment(design_id: str, stroke, *, support_path, axis, source_mesh_sha256: str,
                    radius_mm: float = 1.5, feature_mode: str = KIND_AUTO,
                    indication: str | None = None) -> BoundarySession:
    """磁性修线：只替换 stroke 覆盖的区间，**远端逐位锁死**。"""
    from ..geometry.margin_edit import replace_arc

    session = _current(design_id, support_path=support_path, axis=axis,
                       source_mesh_sha256=source_mesh_sha256, feature_mode=feature_mode,
                       indication=indication)
    mesh = _load_support(support_path)
    result = replace_arc(mesh, _active_points(session), stroke, axis=session.axis,
                         radius_mm=radius_mm, mode=_SNAP_MODE.get(feature_mode, "SUPRA"))
    if result["status"] != "APPLIED":
        session.ambiguity_flags.append("SEGMENT_REPLACE_%s" % result.get("reason"))
        session.status = "REVIEW_REQUIRED"
        return STORE.put(session)
    if not _promote(session, result["points"], source="SEGMENT_REPLACE",
             note="替换 %d 点，未触碰 %d 点，被替换区间长度 %d"
                  % (len(result["moved_index"]), len(result["untouched_index"]),
                     result["replaced_arc_length"])):
        return STORE.put(session)
    session.status = "PROPOSED"
    return STORE.put(session)


def move(design_id: str, delta_mm: float, *, support_path, axis, source_mesh_sha256: str,
         indication: str | None = None) -> BoundarySession:
    """整体上移/下移：沿局部表面切平面内的插入轴投影移动，再投影回网格。"""
    from ..geometry.margin_edit import move_along_surface

    session = _current(design_id, support_path=support_path, axis=axis,
                       source_mesh_sha256=source_mesh_sha256, indication=indication)
    mesh = _load_support(support_path)
    result = move_along_surface(mesh, _active_points(session), axis=session.axis,
                                delta_mm=float(delta_mm))
    if result["status"] != "APPLIED":
        # 棱上"沿表面移动"没有唯一定义 ⇒ 如实拒绝，且**不改动候选**
        session.ambiguity_flags.append("MOVE_%s" % result.get("reason"))
        session.status = "REVIEW_REQUIRED"
        session._log("move_rejected", reason=result.get("reason"))
        return STORE.put(session)
    if not _promote(session, result["points"], source="MOVE",
             note="delta=%.3f mm；%s" % (delta_mm, result["direction"])):
        return STORE.put(session)
    session.status = "PROPOSED"
    return STORE.put(session)


def smooth(design_id: str, *, strength: float = 0.5, locked_index=(), support_path, axis,
           source_mesh_sha256: str, indication: str | None = None) -> BoundarySession:
    """平滑度调节（保锁）。"""
    from ..geometry.margin_edit import smooth_ring

    session = _current(design_id, support_path=support_path, axis=axis,
                       source_mesh_sha256=source_mesh_sha256, indication=indication)
    locked = list(locked_index or [])
    result = smooth_ring(_active_points(session), strength=strength, locked_index=locked)
    if result["status"] != "APPLIED":
        session.ambiguity_flags.append("SMOOTH_%s" % result.get("reason"))
        session.status = "REVIEW_REQUIRED"
        return STORE.put(session)
    if not _promote(session, result["points"], source="SMOOTH",
             note="strength=%.2f，锁定 %d 点" % (float(strength), len(locked))):
        return STORE.put(session)
    session.status = "PROPOSED"
    return STORE.put(session)


def confirm(design_id: str, *, operator: str | None = None,
            source_mesh_sha256: str | None = None,
            proposal_id: str | None = None) -> dict:
    """把候选**确认为正式 Margin**：复用既有 `/margin` 那条 truth-source 路径。

    ``proposal_id`` = **人类显式选定**的候选（歧义场景下必须给）：
    歧义旗标还在时，不给 id 直接确认会被拒；给了 id 视为已做人工判断。
    """
    from ..api.routes import _validated_margin_points
    from ..revisions import record_manual_margin

    session = STORE.get(design_id)
    if session is None:
        raise KeyError("no session")
    if source_mesh_sha256 and session.source_mesh_sha256 != source_mesh_sha256:
        raise ValueError("基牙网格已变化，请重新检测（当前会话基于旧网格）")
    if proposal_id:
        try:
            active = session.set_active(proposal_id)
        except KeyError as exc:
            raise ValueError("指定的候选不存在：%s" % proposal_id) from exc
    else:
        active = session.active()
    if active is None:
        raise KeyError("no active proposal")
    pending = [flag for flag in session.ambiguity_flags if flag.startswith("STRONG_CANDIDATES")]
    if pending and proposal_id is None and not session.anchors:
        raise ValueError("存在并列强候选（%s）：请指名候选（proposal_id）或加纠偏点再确认"
                         % "；".join(pending))
    points = _validated_margin_points(active.points)
    # `record_manual_margin` 返回**单个** DesignRevision（不是元组），且立即生效
    revision = record_manual_margin(design_id, points, operator=operator,
                                    boundary_semantics=session.boundary_semantics)
    session.status = "CONFIRMED"
    session._log("confirm", proposal_id=active.proposal_id, revision_no=revision.revision_no)
    result = {"revision_no": revision.revision_no, "applied": True,
              "margin_source": revision.margin_source, "points": points,
              "indication": session.indication,
              "boundary_semantics": session.boundary_semantics,
              "boundary_label": BOUNDARY_LABELS.get(session.boundary_semantics),
              "proposal_id": active.proposal_id, "anchors": list(session.anchors),
              "ambiguity_flags": list(session.ambiguity_flags),
              "operator_selected_proposal": bool(proposal_id)}
    STORE.clear(design_id)
    return result
