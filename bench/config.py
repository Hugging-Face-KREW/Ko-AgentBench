"""Global configuration helpers for Ko-AgentBench.

- KOAB_PSEUDO_API_MODE: "read" | other
  * "read": Use cache-only pseudo API. If cache miss, raise error.
  * other (default): Always call real API and record the result to cache.

- KOAB_CACHE_DIR: base directory for tool call cache (default: bench/cache)
"""

from __future__ import annotations

import os
from pathlib import Path


def get_pseudo_api_mode() -> str:
    """Return pseudo API mode.

    - "read": strict cache replay
    - others: write mode (always call real API and record cache)
    """
    return os.getenv("KOAB_PSEUDO_API_MODE", "write").strip().lower()


def is_read_mode() -> bool:
    return get_pseudo_api_mode() == "read"


def get_cache_dir() -> Path:
    base = os.getenv("KOAB_CACHE_DIR", "bench/cache")
    p = Path(base)
    # Ensure base exists lazily; creators will mkdir as needed.
    return p
