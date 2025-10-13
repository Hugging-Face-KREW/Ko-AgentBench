"""Filesystem cache store for tool call results.

Key design:
- External code will compute a stable cache_key (hex string) and pass it here.
- Directory layout (new): <base>/<tool_name>/<shard>/<key>.json
- Backward-compat read: falls back to legacy layout <base>/<shard>/<key>.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class CacheRecord:
    key: str
    payload: dict
    created_at: str


class FileCacheStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def _path_for_key(self, key: str, tool_name: Optional[str] = None) -> Path:
        shard = key[:2]
        base = self.base_dir / tool_name if tool_name else self.base_dir
        d = base / shard
        return d / f"{key}.json"

    def get(self, key: str, tool_name: Optional[str] = None) -> Optional[dict]:
        # Try new layout first: <base>/<tool>/<shard>/<key>.json
        path = self._path_for_key(key, tool_name=tool_name)
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        # Backward-compat: try legacy path without tool_name
        legacy_path = self._path_for_key(key, tool_name=None)
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

        return None

    def put(self, key: str, data: dict, tool_name: Optional[str] = None) -> None:
        path = self._path_for_key(key, tool_name=tool_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        record = {
            "key": key,
            "tool": tool_name,
            "created_at": datetime.utcnow().isoformat(),
            "data": data,
        }
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
