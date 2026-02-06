from bench.evaluation.metric_selector import select_metrics_for_task
from bench.runner.metrics import get_metrics_for_level


def test_select_metrics_from_evaluation_checks_and_repetitions():
    task_schema = {
        "evaluation": {
            "deterministic_checks": ["TRF1", "ArgF1", "UNKNOWN_METRIC"],
            "judge_checks": ["SR"],
        },
        "repetitions": 3,
    }

    selected = select_metrics_for_task(1, task_schema)
    assert set(selected.keys()) == {"TRF1", "ArgF1", "SR", "pass@k", "stability@k"}


def test_select_metrics_from_metric_weights_fallback():
    task_schema = {
        "evaluation": {
            "metric_weights": {
                "ToolAcc": 0.7,
                "ArgAcc": 0.3,
                "NOT_EXISTS": 1.0,
            }
        },
        "repetitions": 1,
    }

    selected = select_metrics_for_task(2, task_schema)
    assert set(selected.keys()) == {"ToolAcc", "ArgAcc"}


def test_select_metrics_returns_level_default_when_no_evaluation():
    task_schema = {"repetitions": 1}
    selected = select_metrics_for_task(2, task_schema)
    defaults = get_metrics_for_level(2)
    assert set(selected.keys()) == set(defaults.keys())
