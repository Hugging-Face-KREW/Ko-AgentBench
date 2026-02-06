"""Build v2 sample task set (A+B) from existing v1 tasks.

Usage:
  uv run python scripts/build_v2_sample_tasks.py
  uv run python scripts/build_v2_sample_tasks.py --mode ab200 --target-count 200 --out-dir bench/tasks_v2_ab200
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "bench" / "tasks"
DEFAULT_OUT_DIR = REPO_ROOT / "bench" / "tasks_v2"

# Total 20 tasks across Track A/B
SELECTION_PER_LEVEL = {
    "L1": 4,
    "L2": 4,
    "L3": 3,
    "L4": 3,
    "L5": 2,
    "L6": 2,
    "L7": 2,
}


def infer_track(level: int) -> str:
    if 1 <= level <= 3:
        return "A"
    if 4 <= level <= 7:
        return "B"
    return "C"


def infer_difficulty(level: int) -> str:
    if level <= 2:
        return "easy"
    if level <= 4:
        return "medium"
    if level <= 6:
        return "hard"
    return "expert"


def infer_domain(tools: List[str]) -> str:
    t = " ".join(tools).lower()
    if any(k in t for k in ["kakao", "tmap", "direction", "route", "geocoding", "poi"]):
        return "mobility"
    if any(k in t for k in ["upbit", "bithumb", "crypto"]):
        return "crypto"
    if any(k in t for k in ["stock", "marketindex", "kis", "ls_"]):
        return "finance"
    if any(k in t for k in ["aladin", "item"]):
        return "commerce"
    if any(k in t for k in ["search", "blog", "news", "video", "web"]):
        return "search"
    return "general"


def unique_preserve(seq: List[Any]) -> List[Any]:
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def get_tool_names(task: Dict[str, Any]) -> List[str]:
    names = []
    for a in task.get("golden_action", []):
        if isinstance(a, dict):
            tool = a.get("tool")
            if tool and tool != "reuse":
                names.append(tool)
    for t in task.get("expected_tools", []):
        if isinstance(t, str):
            names.append(t)
    for t in task.get("essential_tools", []):
        if isinstance(t, str):
            names.append(t)
    conv = task.get("conversation_tracking")
    if isinstance(conv, dict):
        for turn in conv.get("turns", []):
            if not isinstance(turn, dict):
                continue
            if isinstance(turn.get("actions"), list):
                for a in turn.get("actions", []):
                    if isinstance(a, dict):
                        tool = a.get("tool")
                        if isinstance(tool, str):
                            names.append(tool)
            if isinstance(turn.get("action"), dict):
                tool = turn["action"].get("tool")
                if isinstance(tool, str):
                    names.append(tool)
    return unique_preserve(names)


def build_action_graph(task: Dict[str, Any]) -> Dict[str, Any]:
    golden_actions = task.get("golden_action", [])
    if isinstance(golden_actions, dict):
        golden_actions = [golden_actions]

    # Normalize and sort by step if present.
    normalized = []
    for idx, action in enumerate(golden_actions, 1):
        if not isinstance(action, dict):
            continue
        tool = action.get("tool")
        if not tool or tool == "reuse":
            continue
        step = action.get("step", idx)
        normalized.append((int(step), tool, action.get("args", {}) or {}))
    normalized.sort(key=lambda x: x[0])

    nodes = []
    step_to_node = {}
    for step, tool, args in normalized:
        node_id = f"s{step}"
        nodes.append({"id": node_id, "tool": tool, "args": args})
        step_to_node[step] = node_id

    edges = []
    for flow in task.get("data_flow", []):
        if not isinstance(flow, dict):
            continue
        from_step = flow.get("from_step")
        to_step = flow.get("to_step")
        from_node = step_to_node.get(from_step)
        to_node = step_to_node.get(to_step)
        if not from_node or not to_node:
            continue
        edges.append(
            {
                "from": from_node,
                "to": to_node,
                "mapping": [
                    {
                        "from_field": flow.get("from_output", ""),
                        "to_arg": flow.get("to_parameter", ""),
                    }
                ],
            }
        )

    min_steps = task.get("minimum_steps")
    if min_steps is None:
        min_steps = max(1, len(nodes))

    return {
        "nodes": nodes,
        "edges": edges,
        "min_steps": int(min_steps),
    }


def build_required_fields(task: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    for gf in task.get("golden_fields", []):
        if not isinstance(gf, dict):
            continue
        for f in gf.get("fields", []):
            if isinstance(f, str):
                fields.append(f)
    return unique_preserve(fields)


def build_conversation(task: Dict[str, Any]) -> Dict[str, Any] | None:
    conv = task.get("conversation_tracking")
    if not isinstance(conv, dict):
        return None
    turns = conv.get("turns")
    if not isinstance(turns, list):
        return None

    normalized = []
    for idx, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        if role not in {"user", "assistant", "tool"}:
            continue
        content = turn.get("content", "")
        normalized.append(
            {
                "turn": int(turn.get("turn_number", idx)),
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            }
        )
    if not normalized:
        return None

    conversation: Dict[str, Any] = {"turns": normalized}

    # Preserve deterministic memory tests from v1 (if present).
    eval_ctx = conv.get("evaluation_context")
    if isinstance(eval_ctx, dict):
        context_tests = eval_ctx.get("context_tests")
        if isinstance(context_tests, list):
            filtered_tests = []
            for test in context_tests:
                if not isinstance(test, dict):
                    continue
                expected_action = test.get("expected_action")
                if not isinstance(expected_action, dict):
                    continue
                tool = expected_action.get("tool")
                if not isinstance(tool, str) or not tool:
                    continue
                filtered_tests.append(test)
            if filtered_tests:
                conversation["context_tests"] = filtered_tests

    return conversation


def deterministic_checks_for_level(level: int) -> List[str]:
    if level <= 2:
        return ["TRF1", "ArgF1", "ExecSR", "GroundAns"]
    if level <= 4:
        return ["TRF1", "ArgF1", "ExecSR", "StepEff", "GroundAns"]
    return ["TRF1", "ArgF1", "ExecSR", "RecoverSR", "StepEff", "CtxRecall", "GroundAns"]


def metric_weights(metrics: List[str], judge_checks: List[str]) -> Dict[str, float]:
    names = metrics + judge_checks
    if not names:
        return {}
    w = 1.0 / len(names)
    return {name: round(w, 6) for name in names}


def convert_task(v1_task: Dict[str, Any]) -> Dict[str, Any]:
    level_raw = v1_task.get("task_level")
    level = int(level_raw) if isinstance(level_raw, int) else 0
    if level <= 0:
        task_id = str(v1_task.get("task_id", ""))
        m = re.match(r"^L(\d+)-", task_id)
        if m:
            level = int(m.group(1))

    track = infer_track(level)
    tools = get_tool_names(v1_task)
    required_tools = unique_preserve(v1_task.get("essential_tools", []) or tools)
    optional_tools = [t for t in tools if t not in required_tools]

    action_graph = build_action_graph(v1_task)
    required_fields = build_required_fields(v1_task)
    judge_checks = ["SR"]
    det_checks = deterministic_checks_for_level(level)
    domain = infer_domain(tools)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    available_tools = [
        {
            "name": t,
            "description": f"Tool schema for {t}",
            "input_schema": {
                "type": "object",
                "additionalProperties": True,
            },
        }
        for t in tools
    ]

    task_v2 = {
        "schema_version": "2.0",
        "task_id": f"{v1_task.get('task_id')}-V2",
        "track": track,
        "level": level,
        "locale": {"language": "ko-KR", "timezone": "Asia/Seoul"},
        "instruction": {"primary": v1_task.get("instruction", ""), "variants": []},
        "available_tools": available_tools,
        "golden": {
            "required_tools": required_tools,
            "optional_tools": optional_tools,
            "action_graph": action_graph,
            "expected_answer": {
                "required_fields": required_fields,
                "forbidden_claims": [],
            },
        },
        "evaluation": {
            "deterministic_checks": det_checks,
            "judge_checks": judge_checks,
            "metric_weights": metric_weights(det_checks, judge_checks),
        },
        "runtime_constraints": {
            "max_steps": max(10, int(action_graph["min_steps"]) + 2),
            "timeout_sec": 60,
            "max_tool_calls": 20,
        },
        "metadata": {
            "domain": domain,
            "difficulty": infer_difficulty(level),
            "author": "ko-agentbench-v2-bootstrap",
            "created_at": created_at,
            "tags": ["v2-sample", "migrated-from-v1", f"source:{v1_task.get('task_id')}"],
        },
    }

    if isinstance(v1_task.get("error_injection"), dict):
        ei = v1_task.get("error_injection", {})
        targets = []
        tool = ei.get("tool")
        if isinstance(tool, str) and tool:
            targets.append(tool)
        task_v2["error_injection"] = {
            "enabled": True,
            "policy": "single_failure",
            "targets": targets,
        }

    conv = build_conversation(v1_task)
    if conv is not None:
        task_v2["conversation"] = conv

    return task_v2


def _clone_with_variant(base_task: Dict[str, Any], variant_idx: int) -> Dict[str, Any]:
    """Create a synthetic instruction variant while preserving evaluation structure."""
    t = json.loads(json.dumps(base_task))
    instruction = (t.get("instruction") or {}).get("primary", "")

    if variant_idx == 1:
        new_instruction = instruction + " 핵심 수치만 간단히 알려줘."
    elif variant_idx == 2:
        new_instruction = "요약해서 답해줘. " + instruction
    else:
        # Mild noise variant (keeps meaning while adding realistic user typo style)
        new_instruction = instruction.replace("알려줘", "알려쥬").replace("주세요", "주세용")

    t["task_id"] = f"{t.get('task_id')}-VAR{variant_idx}"
    t.setdefault("instruction", {})
    t["instruction"]["primary"] = new_instruction

    tags = ((t.get("metadata") or {}).get("tags") or [])
    if isinstance(tags, list):
        tags = tags + [f"synthetic-variant:{variant_idx}"]
    else:
        tags = [f"synthetic-variant:{variant_idx}"]
    t["metadata"]["tags"] = tags

    return t


def _write_by_level(tasks: List[Dict[str, Any]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale level files from previous runs.
    for old in out_dir.glob("L*.json"):
        try:
            old.unlink()
        except Exception:
            pass

    by_level: Dict[str, List[Dict[str, Any]]] = {}
    for t in tasks:
        level = int(t.get("level", 0))
        key = f"L{level}"
        by_level.setdefault(key, []).append(t)

    total_written = 0
    for level_name, rows in sorted(by_level.items(), key=lambda x: x[0]):
        out_path = out_dir / f"{level_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        total_written += len(rows)
        print(f"[OK] wrote {len(rows)} tasks -> {out_path}")

    return total_written


def build_sample20(out_dir: Path) -> int:
    """Build 20-task A+B sample set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for level_name, take_n in SELECTION_PER_LEVEL.items():
        src_path = SRC_DIR / f"{level_name}.json"
        if not src_path.exists():
            print(f"[WARN] missing source file: {src_path}")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        if not isinstance(tasks, list):
            print(f"[WARN] invalid source format: {src_path}")
            continue

        selected = tasks[:take_n]
        converted = [convert_task(t) for t in selected]

        out_path = out_dir / f"{level_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        total_written += len(converted)
        print(f"[OK] wrote {len(converted)} tasks -> {out_path}")

    return total_written


def build_ab_target(out_dir: Path, target_count: int) -> int:
    """Build A+B dataset up to target count using v1->v2 conversion + synthetic variants."""
    if target_count < 1:
        raise ValueError("target_count must be >= 1")

    base_tasks: List[Dict[str, Any]] = []
    for level in range(1, 8):
        src = SRC_DIR / f"L{level}.json"
        if not src.exists():
            continue
        with open(src, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        base_tasks.extend(convert_task(r) for r in rows)

    if not base_tasks:
        raise RuntimeError("No source tasks found to build A+B dataset")

    generated: List[Dict[str, Any]] = []
    seen_ids = set()

    # First include base converted tasks.
    for t in base_tasks:
        tid = t.get("task_id")
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        generated.append(t)
        if len(generated) >= target_count:
            break

    # Then add synthetic variants until target reached.
    variant_idx = 1
    while len(generated) < target_count:
        added_in_round = 0
        for t in base_tasks:
            v = _clone_with_variant(t, variant_idx)
            tid = v.get("task_id")
            if tid in seen_ids:
                continue
            seen_ids.add(tid)
            generated.append(v)
            added_in_round += 1
            if len(generated) >= target_count:
                break
        if added_in_round == 0:
            # Guard against unexpected collisions.
            variant_idx += 1
            if variant_idx > 50:
                raise RuntimeError("Unable to generate enough unique variant tasks")
        else:
            variant_idx += 1

    return _write_by_level(generated, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v2 A+B task sets")
    parser.add_argument(
        "--mode",
        choices=["sample20", "ab200", "ab-target"],
        default="sample20",
        help="Build mode: sample20 (default), ab200, or ab-target",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="Target task count when mode=ab-target or ab200 (default: 200)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if args.mode in {"ab200", "ab-target"} and out_dir == DEFAULT_OUT_DIR:
        out_dir = REPO_ROOT / "bench" / "tasks_v2_ab200"

    if args.mode == "sample20":
        total_written = build_sample20(out_dir)
        print(f"[DONE] total v2 sample tasks written: {total_written}")
        return 0

    if args.mode == "ab200":
        total_written = build_ab_target(out_dir, 200)
        print(f"[DONE] total v2 A+B tasks written: {total_written}")
        return 0

    total_written = build_ab_target(out_dir, args.target_count)
    print(f"[DONE] total v2 A+B tasks written: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
