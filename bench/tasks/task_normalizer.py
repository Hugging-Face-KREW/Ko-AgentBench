"""Task normalization helpers for benchmark runner input.

This module isolates v1/v2 task compatibility logic from the runner entrypoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _is_v2_task(task: Dict[str, Any]) -> bool:
    schema_version = task.get("schema_version")
    return isinstance(schema_version, str) and schema_version.startswith("2.")


def _instruction_text(task: Dict[str, Any]) -> str:
    ins = task.get("instruction", "")
    if isinstance(ins, dict):
        return ins.get("primary", "") or ""
    return ins or ""


def _normalized_tool_names(task: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    raw_tools = task.get("available_tools") or task.get("tools") or []
    for item in raw_tools:
        if isinstance(item, str):
            if item not in names:
                names.append(item)
            continue
        if isinstance(item, dict):
            tool_name = item.get("name") or item.get("tool")
            if isinstance(tool_name, str) and tool_name and tool_name not in names:
                names.append(tool_name)
    return names


def _derive_v2_golden_action(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    golden = task.get("golden", {})
    action_graph = golden.get("action_graph", {}) if isinstance(golden, dict) else {}
    nodes = action_graph.get("nodes", []) if isinstance(action_graph, dict) else []

    actions: List[Dict[str, Any]] = []
    for idx, node in enumerate(nodes, 1):
        if not isinstance(node, dict):
            continue
        tool = node.get("tool")
        if not tool:
            continue
        actions.append(
            {
                "step": idx,
                "tool": tool,
                "args": node.get("args", {}) or {},
            }
        )
    return actions


def _derive_v2_data_flow(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    golden = task.get("golden", {})
    action_graph = golden.get("action_graph", {}) if isinstance(golden, dict) else {}
    nodes = action_graph.get("nodes", []) if isinstance(action_graph, dict) else []
    edges = action_graph.get("edges", []) if isinstance(action_graph, dict) else []

    node_to_step: Dict[str, int] = {}
    for idx, node in enumerate(nodes, 1):
        if isinstance(node, dict) and node.get("id") is not None:
            node_to_step[str(node.get("id"))] = idx

    data_flow: List[Dict[str, Any]] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue

        from_node = str(edge.get("from"))
        to_node = str(edge.get("to"))
        from_step = node_to_step.get(from_node)
        to_step = node_to_step.get(to_node)
        if not from_step or not to_step:
            continue

        mapping = edge.get("mapping", [])
        if not isinstance(mapping, list):
            continue

        for item in mapping:
            if not isinstance(item, dict):
                continue
            from_field = item.get("from_field")
            to_arg = item.get("to_arg")
            if not from_field or not to_arg:
                continue
            data_flow.append(
                {
                    "from_step": from_step,
                    "from_output": from_field,
                    "to_step": to_step,
                    "to_parameter": to_arg,
                }
            )

    return data_flow


def _derive_v2_conversation_tracking(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    conversation = task.get("conversation")
    if not isinstance(conversation, dict):
        return None

    turns = conversation.get("turns")
    if not isinstance(turns, list):
        return None

    normalized_turns = []
    for idx, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            continue

        role = turn.get("role")
        content = turn.get("content", "")
        if role not in {"user", "assistant", "tool"}:
            continue

        normalized_turns.append(
            {
                "turn_number": turn.get("turn", idx),
                "role": role,
                "content": content,
            }
        )

    if not normalized_turns:
        return None

    tracking: Dict[str, Any] = {
        "conversation_id": task.get("task_id"),
        "total_turns": len(normalized_turns),
        "turns": normalized_turns,
    }

    context_tests = conversation.get("context_tests")
    if isinstance(context_tests, list):
        normalized_tests = [test for test in context_tests if isinstance(test, dict)]
        if normalized_tests:
            tracking["evaluation_context"] = {"context_tests": normalized_tests}

    return tracking


def _extract_context_tests(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    # v2 preferred location
    conversation = task.get("conversation")
    if isinstance(conversation, dict):
        tests = conversation.get("context_tests")
        if isinstance(tests, list):
            return [test for test in tests if isinstance(test, dict)]

    # v1 fallback location
    conversation_tracking = task.get("conversation_tracking")
    if isinstance(conversation_tracking, dict):
        eval_ctx = conversation_tracking.get("evaluation_context")
        if isinstance(eval_ctx, dict):
            tests = eval_ctx.get("context_tests")
            if isinstance(tests, list):
                return [test for test in tests if isinstance(test, dict)]

    return []


def _collect_tools_from_conversation_tracking(
    conversation_tracking: Optional[Dict[str, Any]],
    tools_needed: List[str],
) -> None:
    if not isinstance(conversation_tracking, dict):
        return

    turns = conversation_tracking.get("turns", [])
    if not isinstance(turns, list):
        return

    for turn in turns:
        if not isinstance(turn, dict):
            continue
        if "actions" in turn and isinstance(turn.get("actions"), list):
            for action in turn["actions"]:
                if not isinstance(action, dict):
                    continue
                tool_name = action.get("tool")
                if isinstance(tool_name, str) and tool_name and tool_name not in tools_needed:
                    tools_needed.append(tool_name)
        if "action" in turn and isinstance(turn.get("action"), dict):
            tool_name = turn["action"].get("tool")
            if isinstance(tool_name, str) and tool_name and tool_name not in tools_needed:
                tools_needed.append(tool_name)


def normalize_task_for_runner(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one raw task into runner input format."""
    if not isinstance(task, dict):
        return None

    is_v2 = _is_v2_task(task)

    tools_needed = _normalized_tool_names(task)

    golden_action = task.get("golden_action", [])
    if is_v2 and not golden_action:
        golden_action = _derive_v2_golden_action(task)
    if isinstance(golden_action, dict):
        golden_action = [golden_action]

    conversation_tracking = task.get("conversation_tracking")
    if is_v2 and conversation_tracking is None:
        conversation_tracking = _derive_v2_conversation_tracking(task)

    for action in golden_action:
        if not isinstance(action, dict):
            continue
        tool_name = action.get("tool")
        if isinstance(tool_name, str) and tool_name and tool_name not in tools_needed:
            tools_needed.append(tool_name)

    _collect_tools_from_conversation_tracking(conversation_tracking, tools_needed)

    if is_v2:
        required_tools = (
            ((task.get("golden") or {}).get("required_tools") or [])
            if isinstance(task.get("golden"), dict)
            else []
        )
        for tool_name in required_tools:
            if isinstance(tool_name, str) and tool_name and tool_name not in tools_needed:
                tools_needed.append(tool_name)

    data_flow = task.get("data_flow", [])
    minimum_steps = task.get("minimum_steps")
    if is_v2:
        if not data_flow:
            data_flow = _derive_v2_data_flow(task)
        if minimum_steps is None:
            minimum_steps = (
                ((task.get("golden") or {}).get("action_graph") or {}).get("min_steps")
                if isinstance(task.get("golden"), dict)
                else None
            )

    resp_schema = task.get("resp_schema", {})
    if is_v2 and not resp_schema:
        required_fields = (
            (((task.get("golden") or {}).get("expected_answer") or {}).get("required_fields") or [])
            if isinstance(task.get("golden"), dict)
            else []
        )
        resp_schema = {"required_fields": required_fields}

    instruction_text = _instruction_text(task)
    level_value = task.get("task_level")
    if level_value is None:
        level_value = task.get("level", 0)

    category_value = task.get("task_category")
    if category_value is None:
        track = task.get("track")
        domain = (
            (task.get("metadata") or {}).get("domain")
            if isinstance(task.get("metadata"), dict)
            else None
        )
        if track and domain:
            category_value = f"track_{track}:{domain}"
        elif track:
            category_value = f"track_{track}"
        else:
            category_value = "unknown"

    return {
        "id": task.get("task_id", "unknown"),
        "description": instruction_text,
        "expected_output": resp_schema,
        "available_tools": tools_needed,
        # Backward compatibility with runner internals
        "tools": tools_needed,
        "level": level_value,
        "category": category_value,
        "golden_action": golden_action,
        "conversation_tracking": conversation_tracking,
        "arg_schema": task.get("arg_schema", {}),
        "minimum_steps": minimum_steps,
        "data_flow": data_flow,
        "error_injection": task.get("error_injection"),
        "fallback_options": task.get("fallback_options", []),
        "schema_version": task.get("schema_version"),
        "track": task.get("track"),
        "metadata": task.get("metadata", {}),
        "golden": task.get("golden", {}),
        "evaluation": task.get("evaluation", {}),
        "context_tests": _extract_context_tests(task),
    }


def convert_tasks_for_runner(
    dataset_tasks: List[Dict[str, Any]],
    *,
    schema_mode: str = "auto",
) -> List[Dict[str, Any]]:
    """Normalize a dataset task list into runner input format.

    Args:
        dataset_tasks: Raw tasks loaded from task files.
        schema_mode: `auto` (all), `v2` (only schema_version 2.x), `v1` (non-v2).
    """
    tasks_obj: Any = dataset_tasks
    if isinstance(tasks_obj, list) and tasks_obj and isinstance(tasks_obj[0], list):
        tasks_obj = tasks_obj[0]

    converted: List[Dict[str, Any]] = []
    if not isinstance(tasks_obj, list):
        return converted

    if schema_mode not in {"auto", "v1", "v2"}:
        raise ValueError(f"Unsupported schema_mode={schema_mode!r}. Expected auto|v1|v2.")

    for task in tasks_obj:
        if not isinstance(task, dict):
            continue

        is_v2 = _is_v2_task(task)
        if schema_mode == "v2" and not is_v2:
            continue
        if schema_mode == "v1" and is_v2:
            continue

        normalized = normalize_task_for_runner(task)
        if normalized is not None:
            converted.append(normalized)

    return converted
