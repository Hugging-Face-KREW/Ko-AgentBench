"""Scoring helpers for v2 track aggregation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Set, Tuple


DEFAULT_TRACK_CONFIG = {
    "A": {
        "levels": {1, 2, 3},
        "weight": 0.40,
        "deterministic_ratio": 0.85,
        "judge_ratio": 0.15,
    },
    "B": {
        "levels": {4, 5, 6, 7},
        "weight": 0.35,
        "deterministic_ratio": 0.70,
        "judge_ratio": 0.30,
    },
    "C": {
        "levels": {8, 9, 10},
        "weight": 0.25,
        "deterministic_ratio": 0.60,
        "judge_ratio": 0.40,
    },
}


DEFAULT_JUDGE_METRICS = {
    "SR",
    "ArgAcc",
    "EffScore",
    "ContextRetention",
    "RefRecall",
}


def level_sort_key(level_name: str) -> int:
    """Sort level names like L1..L10 numerically."""
    if not isinstance(level_name, str):
        return 10**9
    if not level_name.startswith("L"):
        return 10**9
    try:
        return int(level_name[1:])
    except Exception:
        return 10**9


def weighted_mean(pairs: Iterable[Tuple[float, float]]) -> Optional[float]:
    """Return weighted mean for (value, weight) pairs."""
    total_weight = 0.0
    weighted_sum = 0.0
    for value, weight in pairs:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            continue
        if weight <= 0:
            continue
        total_weight += float(weight)
        weighted_sum += float(value) * float(weight)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def calculate_v2_scores(
    by_level: Dict[str, Any],
    *,
    track_config: Dict[str, Dict[str, Any]],
    judge_metrics: Set[str],
) -> Dict[str, Any]:
    """Aggregate per-level metrics into track scores and overall v2 score."""
    track_buckets = {
        track: {"deterministic": [], "judge": []}
        for track in track_config.keys()
    }

    for level_name, level_data in by_level.items():
        try:
            level_num = int(level_name[1:])
        except Exception:
            continue

        track_name = None
        for candidate, cfg in track_config.items():
            if level_num in cfg.get("levels", set()):
                track_name = candidate
                break

        if track_name is None:
            continue
        if not isinstance(level_data, dict):
            continue

        raw_level_weight = level_data.get("evaluated_tasks", 0)
        try:
            level_weight = max(1, int(raw_level_weight))
        except Exception:
            level_weight = 1

        metrics = level_data.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        for metric_name, metric_score in metrics.items():
            if metric_name in {"RRR", "pass@k", "stability@k"}:
                continue
            if not isinstance(metric_score, (int, float)) or isinstance(metric_score, bool):
                continue

            source = "judge" if metric_name in judge_metrics else "deterministic"
            track_buckets[track_name][source].append((float(metric_score), float(level_weight)))

    tracks_output: Dict[str, Any] = {}
    overall_pairs = []

    for track_name, cfg in track_config.items():
        det_score = weighted_mean(track_buckets[track_name]["deterministic"])
        judge_score = weighted_mean(track_buckets[track_name]["judge"])

        if det_score is not None and judge_score is not None:
            track_score = (
                float(cfg.get("deterministic_ratio", 0.7)) * det_score
                + float(cfg.get("judge_ratio", 0.3)) * judge_score
            )
        elif det_score is not None:
            track_score = det_score
        elif judge_score is not None:
            track_score = judge_score
        else:
            track_score = None

        weight = float(cfg.get("weight", 0.0))
        tracks_output[track_name] = {
            "weight": weight,
            "deterministic_ratio": float(cfg.get("deterministic_ratio", 0.7)),
            "judge_ratio": float(cfg.get("judge_ratio", 0.3)),
            "deterministic_score": det_score,
            "judge_score": judge_score,
            "track_score": track_score,
        }

        if track_score is not None:
            overall_pairs.append((track_score, weight))

    overall_score = weighted_mean(overall_pairs)

    return {
        "tracks": tracks_output,
        "overall_score": overall_score,
        "note": "Track C 데이터가 없으면 A/B 가중치로 자동 재정규화됩니다.",
    }
