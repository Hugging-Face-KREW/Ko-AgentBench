"""Helpers for selecting metrics per task schema."""

from __future__ import annotations

from typing import Any, Dict, List

from bench.runner.metrics import METRICS, get_metrics_for_level


def select_metrics_for_task(level_num: int, task_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve per-task metric set with v2 evaluation config and v1 fallback."""
    default_metrics = get_metrics_for_level(level_num)

    evaluation = task_schema.get("evaluation")
    if not isinstance(evaluation, dict):
        return default_metrics

    requested_names: List[str] = []
    det_checks = evaluation.get("deterministic_checks", [])
    judge_checks = evaluation.get("judge_checks", [])

    if isinstance(det_checks, list):
        requested_names.extend([name for name in det_checks if isinstance(name, str)])
    if isinstance(judge_checks, list):
        requested_names.extend([name for name in judge_checks if isinstance(name, str)])

    # Optional fallback for older task payloads that directly put metric names in weights.
    if not requested_names:
        metric_weights = evaluation.get("metric_weights", {})
        if isinstance(metric_weights, dict):
            requested_names.extend([name for name in metric_weights.keys() if isinstance(name, str)])

    if not requested_names:
        return default_metrics

    selected: Dict[str, Any] = {}
    for metric_name in requested_names:
        if metric_name in METRICS and metric_name not in selected:
            selected[metric_name] = METRICS[metric_name]

    repetitions = task_schema.get("repetitions", 1)
    try:
        repetitions = int(repetitions)
    except Exception:
        repetitions = 1

    if repetitions > 1:
        for metric_name in ("pass@k", "stability@k"):
            if metric_name in METRICS and metric_name not in selected:
                selected[metric_name] = METRICS[metric_name]

    return selected or default_metrics
