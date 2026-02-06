"""Global configuration helpers for Ko-AgentBench.

Cache mode configuration:
- "read": Use cache-only pseudo API. If cache miss, raise error.
- "write": Always call real API and record the result to cache.

Cache directory: base directory for tool call cache (default: bench/cache)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# Global state for cache mode (set by run_benchmark_with_logging.py)
_CACHE_MODE: Optional[str] = None


def _normalize_cache_mode(mode: Optional[str]) -> str:
    if not isinstance(mode, str):
        return ""
    normalized = mode.strip().lower()
    return normalized if normalized in {"read", "write"} else ""


def set_cache_mode(mode: str) -> None:
    """Set the cache mode programmatically.
    
    Args:
        mode: "read" or "write"
    """
    global _CACHE_MODE
    normalized = _normalize_cache_mode(mode)
    if not normalized:
        raise ValueError(f"Invalid cache mode: {mode!r}. Expected 'read' or 'write'.")
    _CACHE_MODE = normalized


def get_pseudo_api_mode() -> str:
    """Return pseudo API mode.

    - "read": strict cache replay
    - "write": write mode (always call real API and record cache)
    
    Checks programmatic setting first, then falls back to environment variable (deprecated).
    """
    if _CACHE_MODE in {"read", "write"}:
        return _CACHE_MODE

    env_mode = _normalize_cache_mode(os.getenv("KOAB_PSEUDO_API_MODE"))
    if env_mode:
        return env_mode
    return "read"


def is_read_mode() -> bool:
    return get_pseudo_api_mode() == "read"


def get_cache_dir() -> Path:
    base = os.getenv("KOAB_CACHE_DIR", "bench/cache")
    p = Path(base)
    # Ensure base exists lazily; creators will mkdir as needed.
    return p
