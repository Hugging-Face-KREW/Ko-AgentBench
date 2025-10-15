"""Argument normalization helpers for stable cache keys.

Responsibilities:
- Fill defaults from parameters_schema
- Enforce basic types (string/integer/number/boolean) where possible
- Remove unknown keys
- Sort keys deterministically
- Provide per-tool alias mapping (e.g., symbol -> shcode for LSStock)
"""

from __future__ import annotations

from typing import Any, Dict


def _apply_defaults(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = schema.get("properties", {}) or {}
    for key, prop in props.items():
        if key not in args and "default" in prop:
            args[key] = prop["default"]


def _coerce_basic_types(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = schema.get("properties", {}) or {}
    for key, prop in props.items():
        if key not in args:
            continue
        typ = prop.get("type")
        if typ == "integer":
            try:
                if not isinstance(args[key], int):
                    args[key] = int(args[key])
            except Exception:
                pass
        elif typ == "number":
            try:
                if not isinstance(args[key], (int, float)):
                    args[key] = float(args[key])
            except Exception:
                pass
        elif typ == "boolean":
            v = args[key]
            if isinstance(v, str):
                if v.lower() in ("true", "1", "yes"): args[key] = True
                if v.lower() in ("false", "0", "no"): args[key] = False
        elif typ == "string":
            if not isinstance(args[key], str):
                try:
                    args[key] = str(args[key])
                except Exception:
                    pass


def _strip_unknown(args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    props = set((schema.get("properties", {}) or {}).keys())
    unknown = [k for k in args.keys() if k not in props]
    for k in unknown:
        # Keep unknowns for backward-compat? For cache stability we drop.
        args.pop(k, None)


def _apply_aliases(tool_name: str, args: Dict[str, Any]) -> None:
    # Per-tool alias rules.
    if tool_name == "StockPrice_ls":
        # Accept symbol as alias of shcode
        if "shcode" not in args and "symbol" in args:
            args["shcode"] = args.pop("symbol")
        # Default exchgubun if missing
        if "exchgubun" not in args:
            args["exchgubun"] = "K"


def normalize_args(tool_name: str, args: Dict[str, Any], parameters_schema: Dict[str, Any]) -> Dict[str, Any]:
    # Work on a copy
    norm = dict(args or {})
    # tool-specific aliasing first (so defaults/types apply after)
    _apply_aliases(tool_name, norm)
    # schema-driven normalization
    if parameters_schema:
        _apply_defaults(norm, parameters_schema)
        _coerce_basic_types(norm, parameters_schema)
        _strip_unknown(norm, parameters_schema)
    # Sort keys deterministically
    return {k: norm[k] for k in sorted(norm.keys())}
