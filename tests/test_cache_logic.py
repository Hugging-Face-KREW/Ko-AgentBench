import importlib
import os
import shutil
from pathlib import Path

import bench.config as config_module
from bench.cache.cache_store import FileCacheStore
from bench.tools.arg_normalizer import normalize_args
from bench.tools.caching_executor import (
    CachingExecutor,
    _build_cache_key,
    _legacy_tool_signature,
)


def test_config_mode_env_fallback_and_validation():
    saved_env = os.environ.get("KOAB_PSEUDO_API_MODE")
    try:
        os.environ["KOAB_PSEUDO_API_MODE"] = "write"
        cfg = importlib.reload(config_module)
        assert cfg.get_pseudo_api_mode() == "write"

        cfg.set_cache_mode("read")
        assert cfg.get_pseudo_api_mode() == "read"

        raised = False
        try:
            cfg.set_cache_mode("invalid")
        except ValueError:
            raised = True
        assert raised
    finally:
        if saved_env is None:
            os.environ.pop("KOAB_PSEUDO_API_MODE", None)
        else:
            os.environ["KOAB_PSEUDO_API_MODE"] = saved_env
        importlib.reload(config_module)


def test_caching_executor_read_mode_supports_legacy_key():
    cache_dir = Path("/tmp/koab_cache_legacy_support_test")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KOAB_CACHE_DIR"] = str(cache_dir)

    # Reload config so module-level _CACHE_MODE resets to unset.
    cfg = importlib.reload(config_module)
    cfg.set_cache_mode("read")

    tool_name = "LegacyTool"
    description = "legacy description"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    raw_args = {"query": "hello", "count": "3"}
    normalized = normalize_args(tool_name, raw_args, schema)

    # Seed only legacy key (description+schema signature).
    legacy_key = _build_cache_key(
        tool_name,
        normalized,
        _legacy_tool_signature(schema, description),
    )
    store = FileCacheStore(cache_dir)
    store.put(legacy_key, {"ok": True, "source": "legacy"}, tool_name=tool_name)

    calls = {"n": 0}

    def api_callable(**kwargs):
        calls["n"] += 1
        return {"ok": False, "kwargs": kwargs}

    executor = CachingExecutor()
    result = executor.execute(
        tool_name=tool_name,
        description=description,
        parameters_schema=schema,
        api_callable=api_callable,
        raw_args=raw_args,
    )
    meta = executor.get_last_meta()

    assert result["ok"] is True
    assert calls["n"] == 0
    assert meta.get("hit") is True
    assert meta.get("key_version") == "legacy"

