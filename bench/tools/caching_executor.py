"""Caching executor that wraps real API method calls with a file cache.

Behavior requested:
- read mode: MUST use cache-only pseudo API. On miss -> raise error.
- other modes (write=off/unspecified): ALWAYS call real API AND record to cache.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Callable

from ..config import is_read_mode, get_cache_dir, get_pseudo_api_mode
from ..cache.cache_store import FileCacheStore
from .arg_normalizer import normalize_args


def _tool_signature(parameters_schema: Dict[str, Any], description: str) -> str:
    payload = {"description": description, "parameters": parameters_schema or {}}
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _build_cache_key(tool_name: str, normalized_args: Dict[str, Any], signature: str) -> str:
    payload = {
        "tool": tool_name,
        "args": normalized_args,
        "sig": signature,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class CachingExecutor:
    def __init__(self) -> None:
        self._store = FileCacheStore(get_cache_dir())
        self._last_meta: Dict[str, Any] = {}

    def execute(
        self,
        *,
        tool_name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        api_callable: Callable[..., Any],
        raw_args: Dict[str, Any],
    ) -> Any:
        signature = _tool_signature(parameters_schema, description)
        normalized = normalize_args(tool_name, raw_args, parameters_schema)
        key = _build_cache_key(tool_name, normalized, signature)

        mode = get_pseudo_api_mode() or "write"
        if is_read_mode():
            cached = self._store.get(key, tool_name=tool_name)
            if cached is None:
                # record meta before raising
                self._last_meta = {"mode": mode, "hit": False, "key": key}
                raise RuntimeError(
                    f"Pseudo-API(read): cache miss for {tool_name} with key={key}. Seed the cache first."
                )
            self._last_meta = {"mode": mode, "hit": True, "key": key}
            return cached.get("data")

        # write mode (default): always call real API and record
    # Call real API with normalized arguments to satisfy method signatures
    # (e.g., alias mapping like symbol->shcode for LSStock).
        result = api_callable(**normalized)
        try:
            # Include input parameters in cache record
            self._store.put(
                key, 
                result, 
                tool_name=tool_name,
                input_params=normalized,
                raw_args=raw_args
            )
        except Exception:
            # Cache write failures should not break tool execution
            pass
        # meta for write path
        self._last_meta = {"mode": mode, "hit": False, "key": key}
        return result

    def get_last_meta(self) -> Dict[str, Any]:
        return dict(self._last_meta)
