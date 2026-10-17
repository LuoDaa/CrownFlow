"""Invalidate older intaglio previews after patient-owned hollow-crown update."""
from __future__ import annotations

from .services import generation_runtime as _runtime

_ORIGINAL = _runtime.intaglio_cache_payload


def _intaglio_cache_payload_v5(*args, **kwargs):
    payload = dict(_ORIGINAL(*args, **kwargs))
    payload["version"] = "intaglio-preview-cache/v5-patient-owned-open-surface-exact-margin"
    return payload


_runtime.intaglio_cache_payload = _intaglio_cache_payload_v5
