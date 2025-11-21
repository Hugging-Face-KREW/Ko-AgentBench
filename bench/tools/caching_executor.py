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

    def _sanitize_result(self, tool_name: str, result: Any) -> Any:
        """Sanitize result data to reduce token usage and cache size."""
        if not isinstance(result, (dict, list)):
            return result

        # 1. Map Tools (Naver/Tmap/Kakao) - Remove heavy coordinates
        if "Directions_naver" in tool_name:
            if isinstance(result, dict) and "route" in result:
                # Remove detailed path coordinates
                for key in result["route"]:
                    if isinstance(result["route"][key], list):
                        for route_opt in result["route"][key]:
                            if "path" in route_opt:
                                route_opt.pop("path", None)
                            if "section" in route_opt:
                                route_opt.pop("section", None)
                            if "guide" in route_opt:
                                # Keep guide but maybe simplify if needed, currently keeping it
                                pass

        elif "Route_tmap" in tool_name:  # CarRoute_tmap, WalkRoute_tmap
            if isinstance(result, dict) and "features" in result:
                for feature in result["features"]:
                    if "geometry" in feature:
                        feature.pop("geometry", None)
                    if "properties" in feature and "index" in feature["properties"]:
                        # Keep only essential properties
                        pass

        elif "POISearch_tmap" in tool_name:
            if isinstance(result, dict) and "searchPoiInfo" in result:
                pois = result["searchPoiInfo"].get("pois", {}).get("poi", [])
                for poi in pois:
                    poi.pop("newAddressList", None)
                    poi.pop("evChargers", None)

        # 2. Market/Finance Tools - Truncate long lists
        elif "MarketList_" in tool_name:  # Upbit, Bithumb
            # Upbit returns list directly or dict with market list
            if isinstance(result, list):
                if len(result) > 20:
                    return result[:20]
            elif isinstance(result, dict):
                # Bithumb usually returns {"status":..., "data": [...]}
                if "data" in result and isinstance(result["data"], list):
                    if len(result["data"]) > 20:
                        result["data"] = result["data"][:20]
                # Upbit sometimes returns dict wrapper
                if "markets" in result and isinstance(result["markets"], list):
                    if len(result["markets"]) > 20:
                        result["markets"] = result["markets"][:20]

        return result

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
            
            # Sanitize cached data before returning (for backward compatibility with old large caches)
            data = cached.get("data")
            return self._sanitize_result(tool_name, data)

        # write mode (default): always call real API and record
        # Call real API with normalized arguments to satisfy method signatures
        # (e.g., alias mapping like symbol->shcode for LSStock).
        result = api_callable(**normalized)
        
        # Sanitize result BEFORE saving to cache
        result = self._sanitize_result(tool_name, result)
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
