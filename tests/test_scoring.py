import math

from bench.evaluation.scoring import (
    DEFAULT_JUDGE_METRICS,
    DEFAULT_TRACK_CONFIG,
    calculate_v2_scores,
    weighted_mean,
)


def test_weighted_mean_skips_invalid_items():
    score = weighted_mean([(1.0, 2.0), (0.0, 0.0), (True, 1.0), (0.5, 1.0)])
    assert math.isclose(score, (1.0 * 2.0 + 0.5 * 1.0) / 3.0)


def test_calculate_v2_scores_reweights_when_track_c_missing():
    by_level = {
        "L1": {
            "evaluated_tasks": 2,
            "metrics": {
                "TRF1": 0.8,
                "SR": 0.6,
                "RRR": 1.0,
            },
        },
        "L2": {
            "evaluated_tasks": 1,
            "metrics": {
                "TRF1": 0.5,
                "SR": 0.4,
            },
        },
        "L4": {
            "evaluated_tasks": 1,
            "metrics": {
                "Coverage": 0.9,
                "SR": 0.2,
                "pass@k": 1.0,
            },
        },
    }

    result = calculate_v2_scores(
        by_level,
        track_config=DEFAULT_TRACK_CONFIG,
        judge_metrics=DEFAULT_JUDGE_METRICS,
    )

    tracks = result["tracks"]
    assert math.isclose(tracks["A"]["deterministic_score"], 0.7)
    assert math.isclose(tracks["A"]["judge_score"], 1.6 / 3.0)
    assert math.isclose(tracks["A"]["track_score"], 0.675)

    assert math.isclose(tracks["B"]["deterministic_score"], 0.9)
    assert math.isclose(tracks["B"]["judge_score"], 0.2)
    assert math.isclose(tracks["B"]["track_score"], 0.69)

    assert tracks["C"]["track_score"] is None
    assert math.isclose(result["overall_score"], 0.682)
