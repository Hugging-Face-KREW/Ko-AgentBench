"""Seed the pseudo-API cache by running tasks online (write-through).

Usage:
  KOAB_PSEUDO_API_MODE=write python -m scripts.seed_cache --levels L1 L2

This will execute datasets and populate bench/cache with responses so that
KOAB_PSEUDO_API_MODE=read can replay them without network/API keys.
"""

from __future__ import annotations

import argparse
import os

from pathlib import Path
from typing import List

from bench.tools.tool_catalog import resolve_tool_classes, TOOL_CATALOG
from bench.tools.tool_registry import ToolRegistry
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.runner import BenchmarkRunner, Judge


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Dataset directory (contains L*.json)")
    parser.add_argument("--levels", nargs="*", default=["L1"], help="Levels to run e.g. L1 L2 L7")
    parser.add_argument("--model", default="gpt-4.1", help="Model name for adapter")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    # Ensure write-through mode
    mode = os.getenv("KOAB_PSEUDO_API_MODE", "write").lower()
    if mode == "read":
        print("[seed] Warning: KOAB_PSEUDO_API_MODE=read; overriding to write for seeding.")
        os.environ["KOAB_PSEUDO_API_MODE"] = "write"

    # Lazy import to reuse dataset loader from run_benchmark_with_logging
    from run_benchmark_with_logging import load_benchmark_datasets, convert_dataset_to_tasks

    datasets = load_benchmark_datasets(args.data_dir)
    if not datasets:
        print("[seed] No datasets found.")
        return

    adapter = LiteLLMAdapter(args.model)
    judge = Judge(llm_adapter=adapter)

    for level in args.levels:
        tasks_raw = datasets.get(level)
        if not tasks_raw:
            print(f"[seed] Skip {level}: not found")
            continue
        tasks = convert_dataset_to_tasks(tasks_raw)

        # Collect and register tools
        all_tools: List[str] = []
        for t in tasks:
            for nm in t.get("available_tools", []):
                if nm not in all_tools:
                    all_tools.append(nm)
        tool_classes = resolve_tool_classes(all_tools)
        registry = ToolRegistry()
        for cls in tool_classes:
            registry.register_tool(cls)

        runner = BenchmarkRunner(adapter, registry, judge, max_steps=args.max_steps, timeout=args.timeout)
        print(f"[seed] Running {level} with {len(tasks)} tasks; tools={registry.get_available_tools()}")
        for task in tasks:
            try:
                _ = runner.run_task(task)
            except Exception as e:
                print(f"[seed] Task {task.get('id') or task.get('task_id')}: error: {e}")


if __name__ == "__main__":
    main()
