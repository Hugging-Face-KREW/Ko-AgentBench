from pathlib import Path

from bench.evaluation.v2_export import (
    build_v2_evaluation_records,
    export_v2_jsonl_from_report,
    validate_v2_evaluation_records,
)


TRACK_CONFIG = {
    "A": {"deterministic_ratio": 0.85, "judge_ratio": 0.15},
    "B": {"deterministic_ratio": 0.70, "judge_ratio": 0.30},
    "C": {"deterministic_ratio": 0.60, "judge_ratio": 0.40},
}


def _sample_report():
    return {
        "by_level": {
            "L1": {
                "task_evaluations": [
                    {
                        "task_id": "L1-SAMPLE",
                        "level": 1,
                        "track": "A",
                        "evaluation": {
                            "metric_weights": {"TRF1": 0.6, "SR": 0.4}
                        },
                        "repetitions": 3,
                        "metrics": {
                            "TRF1": {"score": 1.0, "details": {"f1": 1.0}},
                            "SR": {"score": 0.5, "details": {"reason": "partial"}},
                            "pass@k": {"score": 1.0, "details": {}},
                            "stability@k": {"score": 0.6667, "details": {}},
                            "RRR": {"score": 1.0, "details": {}},
                        },
                    }
                ]
            }
        }
    }


def test_build_v2_evaluation_records_excludes_passk_from_metric_scores():
    report = _sample_report()
    records = build_v2_evaluation_records(
        report,
        model_id="azure/gpt-5-mini",
        date="20260206",
        track_config=TRACK_CONFIG,
        judge_metrics={"SR"},
    )
    assert len(records) == 1
    record = records[0]
    metric_ids = [m["metric_id"] for m in record["metric_scores"]]
    assert "TRF1" in metric_ids
    assert "SR" in metric_ids
    assert "pass@k" not in metric_ids
    assert "stability@k" not in metric_ids
    assert record["aggregate"]["pass_at_k"]["k"] == 3
    assert record["aggregate"]["stability_at_k"]["value"] > 0.6


def test_export_v2_jsonl_and_validate(tmp_path: Path | None = None):
    if tmp_path is None:
        from tempfile import mkdtemp

        tmp_path = Path(mkdtemp(prefix="v2_export_test_"))

    report = _sample_report()
    output_file = tmp_path / "evaluation_v2.jsonl"
    result = export_v2_jsonl_from_report(
        report,
        output_file=output_file,
        model_id="azure/gpt-5-mini",
        date="20260206",
        track_config=TRACK_CONFIG,
        judge_metrics={"SR"},
    )
    assert output_file.exists()
    assert len(result["records"]) == 1

    validation = validate_v2_evaluation_records(result["records"])
    assert validation["validator_available"] is True
    assert validation["errors"] == []
