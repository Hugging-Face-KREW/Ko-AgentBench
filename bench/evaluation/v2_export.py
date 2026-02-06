"""v2 evaluation.schema JSONL export helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from bench.runner.metrics import LLMJudgeMetric, METRICS


def _is_numeric_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _split_model_id(model_id: str) -> Dict[str, str]:
    if isinstance(model_id, str) and "/" in model_id:
        provider, name = model_id.split("/", 1)
        return {"provider": provider, "name": name}
    return {"provider": "unknown", "name": str(model_id)}


def _track_from_level(level_num: int) -> str:
    if 1 <= level_num <= 3:
        return "A"
    if 4 <= level_num <= 7:
        return "B"
    if 8 <= level_num <= 10:
        return "C"
    return "C"


def _level_sort_key(level_name: str) -> int:
    if not isinstance(level_name, str) or not level_name.startswith("L"):
        return 10**9
    try:
        return int(level_name[1:])
    except Exception:
        return 10**9


def _metric_source(metric_name: str, judge_metrics: Set[str]) -> str:
    metric_obj = METRICS.get(metric_name)
    if metric_name in judge_metrics or isinstance(metric_obj, LLMJudgeMetric):
        return "judge"
    return "deterministic"


def _weighted_mean_dict(items: List[Dict[str, Any]]) -> Optional[float]:
    if not items:
        return None
    total_weight = 0.0
    weighted_sum = 0.0
    for item in items:
        score = item.get("score")
        weight = item.get("weight")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            continue
        if weight <= 0:
            continue
        total_weight += float(weight)
        weighted_sum += float(score) * float(weight)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _normalize_metric_weights(metric_names: List[str], metric_weights: Dict[str, Any]) -> Dict[str, float]:
    if not metric_names:
        return {}

    raw_weights: Dict[str, float] = {}
    for name in metric_names:
        raw = metric_weights.get(name, 1.0)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw <= 0:
            raw = 1.0
        raw_weights[name] = float(raw)

    total = sum(raw_weights.values())
    if total <= 0:
        uniform = 1.0 / len(metric_names)
        return {name: uniform for name in metric_names}

    return {name: raw_weights[name] / total for name in metric_names}


def _build_v2_evaluation_record(
    *,
    level_name: str,
    task_eval: Dict[str, Any],
    run_id: str,
    evaluated_at: str,
    model_id: str,
    track_config: Dict[str, Any],
    judge_metrics: Set[str],
) -> Optional[Dict[str, Any]]:
    if not isinstance(task_eval, dict):
        return None

    task_id = task_eval.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None

    metrics = task_eval.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    evaluation_cfg = task_eval.get("evaluation", {})
    metric_weights = {}
    if isinstance(evaluation_cfg, dict):
        maybe_weights = evaluation_cfg.get("metric_weights")
        if isinstance(maybe_weights, dict):
            metric_weights = maybe_weights

    eligible_metrics = []
    for metric_name, metric_data in metrics.items():
        if metric_name in {"RRR", "pass@k", "stability@k"}:
            continue
        if not isinstance(metric_data, dict):
            continue
        score = metric_data.get("score")
        if _is_numeric_score(score):
            eligible_metrics.append(metric_name)

    if not eligible_metrics:
        return None

    normalized_weights = _normalize_metric_weights(eligible_metrics, metric_weights)

    metric_scores: List[Dict[str, Any]] = []
    for metric_name in eligible_metrics:
        metric_data = metrics.get(metric_name, {})
        score = float(metric_data.get("score"))
        score = max(0.0, min(1.0, score))
        details = metric_data.get("details", {})
        if not isinstance(details, dict):
            details = {}

        metric_scores.append(
            {
                "metric_id": metric_name,
                "source": _metric_source(metric_name, judge_metrics),
                "score": score,
                "weight": normalized_weights.get(metric_name, 0.0),
                "details": details,
            }
        )

    det_items = [m for m in metric_scores if m["source"] == "deterministic"]
    judge_items = [m for m in metric_scores if m["source"] == "judge"]

    deterministic_score = _weighted_mean_dict(det_items)
    judge_score = _weighted_mean_dict(judge_items)

    if deterministic_score is None and judge_score is None:
        return None
    if deterministic_score is None:
        deterministic_score = judge_score
    if judge_score is None:
        judge_score = deterministic_score

    try:
        level_num = int(str(level_name)[1:])
    except Exception:
        level_num = int(task_eval.get("level", 0) or 0)

    track = task_eval.get("track")
    if track not in {"A", "B", "C", "SAFETY"}:
        track = _track_from_level(level_num)

    default_cfg = {"deterministic_ratio": 0.7, "judge_ratio": 0.3}
    cfg = track_config.get(track, default_cfg)
    deterministic_ratio = float(cfg.get("deterministic_ratio", default_cfg["deterministic_ratio"]))
    judge_ratio = float(cfg.get("judge_ratio", default_cfg["judge_ratio"]))

    track_score = deterministic_ratio * float(deterministic_score) + judge_ratio * float(judge_score)
    track_score = max(0.0, min(1.0, track_score))

    aggregate: Dict[str, Any] = {
        "deterministic_score": max(0.0, min(1.0, float(deterministic_score))),
        "judge_score": max(0.0, min(1.0, float(judge_score))),
        "track_score": track_score,
        "total_score": track_score,
    }

    repetitions = task_eval.get("repetitions", 1)
    try:
        repetitions = max(1, int(repetitions))
    except Exception:
        repetitions = 1

    pass_score = ((metrics.get("pass@k") or {}).get("score") if isinstance(metrics.get("pass@k"), dict) else None)
    if _is_numeric_score(pass_score):
        aggregate["pass_at_k"] = {
            "k": repetitions,
            "value": max(0.0, min(1.0, float(pass_score))),
        }

    stability_score = (
        (metrics.get("stability@k") or {}).get("score")
        if isinstance(metrics.get("stability@k"), dict)
        else None
    )
    if _is_numeric_score(stability_score):
        aggregate["stability_at_k"] = {
            "k": repetitions,
            "value": max(0.0, min(1.0, float(stability_score))),
        }

    model_info = _split_model_id(model_id)
    evaluation_id = f"{run_id}:{level_name}:{task_id}"

    return {
        "schema_version": "2.0",
        "evaluation_id": evaluation_id,
        "run_id": run_id,
        "task_id": task_id,
        "track": track,
        "model": model_info,
        "metric_scores": metric_scores,
        "aggregate": aggregate,
        "meta": {
            "evaluated_at": evaluated_at,
            "evaluator_version": "ko-agentbench-v2",
        },
    }


def build_v2_evaluation_records(
    report: Dict[str, Any],
    *,
    model_id: str,
    date: str,
    track_config: Dict[str, Any],
    judge_metrics: Set[str],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    run_id = f"{date}_{model_id.replace('/', '_')}"
    evaluated_at = datetime.now().isoformat()

    by_level = report.get("by_level", {})
    if not isinstance(by_level, dict):
        return records

    for level_name in sorted(by_level.keys(), key=_level_sort_key):
        level_data = by_level.get(level_name, {})
        if not isinstance(level_data, dict):
            continue

        task_evaluations = level_data.get("task_evaluations", [])
        if not isinstance(task_evaluations, list):
            continue

        for task_eval in task_evaluations:
            record = _build_v2_evaluation_record(
                level_name=level_name,
                task_eval=task_eval,
                run_id=run_id,
                evaluated_at=evaluated_at,
                model_id=model_id,
                track_config=track_config,
                judge_metrics=judge_metrics,
            )
            if record is not None:
                records.append(record)

    return records


def validate_v2_evaluation_records(
    records: List[Dict[str, Any]],
    *,
    schema_path: Path = Path("bench/schemas/v2/evaluation.schema.json"),
) -> Dict[str, Any]:
    summary = {
        "validator_available": False,
        "total_records": len(records),
        "errors": [],
    }

    try:
        from jsonschema import Draft202012Validator
    except Exception:
        return summary

    if not schema_path.exists():
        return summary

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        validator = Draft202012Validator(schema)
    except Exception:
        return summary

    summary["validator_available"] = True

    for idx, record in enumerate(records):
        errors = sorted(validator.iter_errors(record), key=lambda e: list(e.path))
        if not errors:
            continue
        error_messages = []
        for err in errors[:5]:
            path = ".".join(str(p) for p in err.path) or "<root>"
            error_messages.append(f"{path}: {err.message}")
        summary["errors"].append(
            {
                "index": idx,
                "evaluation_id": record.get("evaluation_id"),
                "messages": error_messages,
            }
        )

    return summary


def export_v2_jsonl_from_report(
    report: Dict[str, Any],
    *,
    output_file: Path,
    model_id: str,
    date: str,
    track_config: Dict[str, Any],
    judge_metrics: Set[str],
) -> Dict[str, Any]:
    records = build_v2_evaluation_records(
        report,
        model_id=model_id,
        date=date,
        track_config=track_config,
        judge_metrics=judge_metrics,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")

    validation = validate_v2_evaluation_records(records)
    return {
        "records": records,
        "validation": validation,
    }

