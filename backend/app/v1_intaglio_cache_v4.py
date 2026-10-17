"""Invalidate pre-1.0 intaglio previews after exact Margin-boundary ownership."""
from __future__ import annotations

from .services import generation_runtime as _runtime

_ORIGINAL_INTAGLIO_CACHE_PAYLOAD = _runtime.intaglio_cache_payload


def _intaglio_cache_payload_v4(*args, **kwargs):
    payload = dict(_ORIGINAL_INTAGLIO_CACHE_PAYLOAD(*args, **kwargs))
    payload["version"] = "intaglio-preview-cache/v4-exact-confirmed-margin-boundary"
    return payload


_runtime.intaglio_cache_payload = _intaglio_cache_payload_v4
