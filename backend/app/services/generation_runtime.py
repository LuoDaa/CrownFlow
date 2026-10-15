"""Runtime audit + reuse cache for CAD-1 automatic crown generation.

This module deliberately keeps two concerns together because they must be audited
as one performance story:

* every expensive generation stage emits a durable JSONL timing event and a
  human-readable backend log line;
* patient geometry that is invariant across automatic correction rounds is
  cached by content-derived keys instead of rebuilt on every request.

The cache is an optimisation only. Keys include the geometry/provenance inputs
that own the result; a key miss always rebuilds from source geometry.  Cached
meshes are persisted under the case directory so sequential requests, process
restarts and later review can all explain what was reused.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
import time
from uuid import uuid4

from ..config import CASE_DIR
from ..geometry.io import load_mesh, save_mesh

# uvicorn.error is visible in the normal backend terminal without requiring an
# extra logging configuration.  The same events are also persisted as JSONL.
_LOG = logging.getLogger("uvicorn.error")
_OBJECT_CACHE = OrderedDict()
_CACHE_LOCK = threading.Lock()
_OBJECT_CACHE_LIMIT = 16


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def intaglio_cache_payload(*, support_sha256, margin, axis, parameters, prepared_support=None):
    """Only include parameters that can change the blockout/gap envelope.

    Contact or occlusal targets intentionally do not invalidate this cache.
    """
    keys = (
        "cement_gap_mm", "cement_gap_start_mm", "cement_gap_top_mm",
        "margin_gap_mm", "margin_extension_mm", "untouchable_zone_mm",
        "machine_axes", "undercut_policy", "blockout_angle_deg",
        "border_horizontal_mm", "border_angled_mm", "border_angle_deg",
        "border_vertical_mm", "tool_radius_mm", "tool_compensation_enabled",
    )
    return {
        "version": "intaglio-preview-cache/v1",
        "support_sha256": support_sha256,
        "margin": margin,
        "axis": axis,
        "parameters": {key: parameters.get(key) for key in keys},
        "prepared_support": prepared_support,
    }


def antagonist_cache_payload(*, scan_sha256, transform, frame_revision_id, frame_z):
    return {
        "version": "antagonist-surface-cache/v1",
        "scan_sha256": scan_sha256,
        "transform": transform,
        "frame_revision_id": frame_revision_id,
        "frame_z": frame_z,
    }


def neighbor_cache_payload(*, region_sha256, outward_normal):
    return {
        "version": "neighbor-surface-cache/v1",
        "region_sha256": region_sha256,
        "outward_normal": outward_normal,
    }


def _cache_dir(case_id: str) -> Path:
    path = CASE_DIR / case_id / ".generation_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_mesh(case_id: str, namespace: str, payload, builder, *, trace=None):
    """Disk-backed immutable mesh cache.

    ``builder`` returns ``(mesh, metadata)``.  A copy is returned so callers can
    mutate their working mesh without corrupting the cache.
    """
    key = stable_key(payload)
    root = _cache_dir(case_id)
    mesh_path = root / f"{namespace}_{key}.stl"
    meta_path = root / f"{namespace}_{key}.json"
    if mesh_path.is_file() and meta_path.is_file():
        try:
            mesh = load_mesh(mesh_path)
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if trace is not None:
                trace.cache(namespace, True, key, path=str(mesh_path))
            return mesh.copy(), metadata, True, mesh_path
        except Exception:
            # Corrupt/partial cache is never trusted; rebuilding is safer.
            pass
    started = time.perf_counter()
    mesh, metadata = builder()
    save_mesh(mesh, mesh_path)
    meta_path.write_text(json.dumps(metadata or {}, ensure_ascii=False, indent=1), encoding="utf-8")
    if trace is not None:
        trace.cache(namespace, False, key, path=str(mesh_path), build_elapsed_s=time.perf_counter() - started)
    return mesh.copy(), metadata or {}, False, mesh_path


def cached_object(namespace: str, payload, builder, *, trace=None):
    """Small in-process cache for immutable acceleration/query objects.

    FitSurface construction creates reusable search/sign structures.  These
    objects are not serialised; the underlying transformed meshes are separately
    disk-cached when useful.
    """
    key = f"{namespace}:{stable_key(payload)}"
    with _CACHE_LOCK:
        value = _OBJECT_CACHE.get(key)
        if value is not None:
            _OBJECT_CACHE.move_to_end(key)
    if value is not None:
        if trace is not None:
            trace.cache(namespace, True, key.split(":", 1)[1], memory=True)
        return value, True
    started = time.perf_counter()
    value = builder()
    with _CACHE_LOCK:
        _OBJECT_CACHE[key] = value
        _OBJECT_CACHE.move_to_end(key)
        while len(_OBJECT_CACHE) > _OBJECT_CACHE_LIMIT:
            _OBJECT_CACHE.popitem(last=False)
    if trace is not None:
        trace.cache(namespace, False, key.split(":", 1)[1], memory=True,
                    build_elapsed_s=time.perf_counter() - started)
    return value, False


class GenerationTrace:
    """Append-only per-design audit plus per-candidate timing snapshot."""

    def __init__(self, case_id: str, design_id: str, *, baseline_sha256: str | None = None):
        self.case_id = case_id
        self.design_id = design_id
        self.run_id = uuid4().hex
        self.baseline_sha256 = baseline_sha256
        self.started_at = _utc_now()
        self._started = time.perf_counter()
        self.events = []
        self.audit_path = CASE_DIR / case_id / f"{design_id}_generation_audit.jsonl"
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.event("RUN_START", status="STARTED", baseline_sha256=baseline_sha256)

    def _append(self, event: dict):
        payload = {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "design_id": self.design_id,
            "timestamp": _utc_now(),
            **event,
        }
        self.events.append(payload)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        elapsed = payload.get("elapsed_s")
        suffix = f" elapsed={elapsed:.3f}s" if isinstance(elapsed, (int, float)) else ""
        _LOG.info("CAD1 run=%s stage=%s status=%s%s", self.run_id,
                  payload.get("stage"), payload.get("status"), suffix)
        return payload

    def event(self, stage: str, *, status="INFO", **metadata):
        return self._append({"stage": stage, "status": status, **metadata})

    def cache(self, namespace: str, hit: bool, key: str, **metadata):
        return self.event(
            f"CACHE_{namespace.upper()}",
            status="HIT" if hit else "MISS",
            cache_key=key,
            **metadata,
        )

    @contextmanager
    def span(self, stage: str, **metadata):
        started = time.perf_counter()
        self.event(stage, status="STARTED", **metadata)
        try:
            yield
        except Exception as exc:
            self._append({
                "stage": stage,
                "status": "ERROR",
                "elapsed_s": time.perf_counter() - started,
                "error": str(exc),
                **metadata,
            })
            raise
        else:
            self._append({
                "stage": stage,
                "status": "DONE",
                "elapsed_s": time.perf_counter() - started,
                **metadata,
            })

    def summary(self):
        completed = [event for event in self.events if event.get("status") in ("DONE", "ERROR")]
        stage_seconds = {}
        for event in completed:
            if isinstance(event.get("elapsed_s"), (int, float)):
                stage_seconds[event["stage"]] = stage_seconds.get(event["stage"], 0.0) + float(event["elapsed_s"])
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "total_elapsed_s": float(time.perf_counter() - self._started),
            "stage_elapsed_s": stage_seconds,
            "cache_hits": sum(1 for event in self.events if event.get("status") == "HIT"),
            "cache_misses": sum(1 for event in self.events if event.get("status") == "MISS"),
            "audit_jsonl": str(self.audit_path),
        }

    def snapshot(self, path: Path, *, extra=None) -> Path:
        summary = self.summary()
        self.event("RUN_END", status="DONE", total_elapsed_s=summary["total_elapsed_s"])
        payload = {"summary": self.summary(), "events": self.events}
        if extra:
            payload["extra"] = extra
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        return path
