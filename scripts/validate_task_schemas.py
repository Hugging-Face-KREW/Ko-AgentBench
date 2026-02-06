"""Validate Ko-AgentBench task files (v1/v2-compatible).

Usage:
  uv run python scripts/validate_task_schemas.py
  uv run python scripts/validate_task_schemas.py --strict
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.tasks import TaskLoader


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate task schemas for Ko-AgentBench")
    parser.add_argument("--tasks-dir", default="bench/tasks", help="Task directory (default: bench/tasks)")
    parser.add_argument("--pattern", default="L*.json", help="Glob pattern for task files (default: L*.json)")
    parser.add_argument("--recursive", action="store_true", help="Use recursive glob search")
    parser.add_argument("--strict", action="store_true", help="Fail on v2 schema validation errors")
    args = parser.parse_args()

    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.exists():
        print(f"[ERROR] tasks dir not found: {tasks_dir}")
        return 1

    loader = TaskLoader(tasks_dir=tasks_dir)
    files = sorted(tasks_dir.rglob(args.pattern) if args.recursive else tasks_dir.glob(args.pattern))
    if not files:
        print(f"[ERROR] no files found in: {tasks_dir} (pattern={args.pattern})")
        return 1

    total_tasks = 0
    total_v2 = 0
    total_errors = 0
    total_validator_missing = 0

    for file_path in files:
        rel = file_path.relative_to(tasks_dir)
        tasks = loader.load_tasks(str(rel))
        summary = loader.validate_tasks(tasks, file_label=str(rel), strict=False)

        total_tasks += summary.get("total_tasks", 0)
        total_v2 += summary.get("v2_tasks", 0)
        errors = len(summary.get("errors", []))
        total_errors += errors
        if summary.get("v2_tasks", 0) > 0 and not summary.get("validator_available", False):
            total_validator_missing += 1

        print(
            f"[OK] {file_path.name}: total={summary.get('total_tasks', 0)} "
            f"v2={summary.get('v2_tasks', 0)} errors={errors} "
            f"validator={summary.get('validator_available', False)}"
        )

    print(
        f"\n[SUMMARY] files={len(files)} total_tasks={total_tasks} "
        f"v2_tasks={total_v2} errors={total_errors} "
        f"validator_missing={total_validator_missing}"
    )

    if args.strict and total_errors > 0:
        print("[FAIL] strict mode enabled and validation errors found.")
        return 2
    if args.strict and total_validator_missing > 0:
        print("[FAIL] strict mode enabled and schema validator unavailable for v2 tasks.")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
