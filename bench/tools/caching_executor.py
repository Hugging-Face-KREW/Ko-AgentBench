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

# Maximum number of items to keep in market/finance list results
# This limit reduces token usage and cache size while preserving enough data
# for typical trading operations and market analysis
MAX_MARKET_LIST_ITEMS = 20


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
        """Sanitize result data to reduce token usage and cache size.
        
        Uses shallow copying and selective field copying to avoid the overhead
        of deepcopy while preventing mutation of the original result object.
        """
        if not isinstance(result, (dict, list)):
            return result

        # 1. Map Tools (Naver/Tmap/Kakao) - Remove heavy coordinates
        if "Directions_naver" in tool_name:
            if isinstance(result, dict) and "route" in result:
                # Create a shallow copy of the result dict
                result = result.copy()
                result["route"] = result["route"].copy()
                
                # Remove detailed path coordinates
                for key in result["route"]:
                    if isinstance(result["route"][key], list):
                        # Create new list with sanitized route options
                        sanitized_routes = []
                        for route_opt in result["route"][key]:
                            if isinstance(route_opt, dict):
                                # Copy only fields we want to keep (exclude path, section)
                                sanitized_opt = {
                                    k: v for k, v in route_opt.items() 
                                    if k not in {"path", "section"}
                                }
                                sanitized_routes.append(sanitized_opt)
                            else:
                                sanitized_routes.append(route_opt)
                        result["route"][key] = sanitized_routes

        elif "Route_tmap" in tool_name:  # CarRoute_tmap, WalkRoute_tmap
            if isinstance(result, dict) and "features" in result:
                # Create a shallow copy of the result dict
                result = result.copy()
                # Create new features list with geometry removed
                sanitized_features = []
                for feature in result["features"]:
                    if isinstance(feature, dict):
                        # Copy only fields we want to keep (exclude geometry)
                        sanitized_feature = {
                            k: v for k, v in feature.items() 
                            if k not in {"geometry"}
                        }
                        sanitized_features.append(sanitized_feature)
                    else:
                        sanitized_features.append(feature)
                result["features"] = sanitized_features

        elif "POISearch_tmap" in tool_name:
            if isinstance(result, dict) and "searchPoiInfo" in result:
                pois_data = result["searchPoiInfo"].get("pois")
                if isinstance(pois_data, dict):
                    poi_list = pois_data.get("poi", [])
                    if isinstance(poi_list, list):
                        for poi in poi_list:
                            if isinstance(poi, dict):
                                poi.pop("newAddressList", None)
                                poi.pop("evChargers", None)
        # 2. Market/Finance Tools - Truncate long lists
        elif "MarketList_" in tool_name:  # Upbit, Bithumb
            # For truncation, slicing already creates a new list, no need for extra copy
            if isinstance(result, list):
                if len(result) > MAX_MARKET_LIST_ITEMS:
                    return result[:MAX_MARKET_LIST_ITEMS]
            elif isinstance(result, dict):
                # Create shallow copy to avoid mutation
                result = result.copy()
                # Bithumb usually returns {"status":..., "data": [...]}
                if "data" in result and isinstance(result["data"], list):
                    if len(result["data"]) > MAX_MARKET_LIST_ITEMS:
                        result["data"] = result["data"][:MAX_MARKET_LIST_ITEMS]
                # Upbit sometimes returns dict wrapper
                elif "markets" in result and isinstance(result["markets"], list):
                    if len(result["markets"]) > MAX_MARKET_LIST_ITEMS:
                        result["markets"] = result["markets"][:MAX_MARKET_LIST_ITEMS]

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
