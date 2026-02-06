import csv
import json
from pathlib import Path
from tempfile import mkdtemp

from bench.evaluation.report_export import (
    export_csv_summary,
    export_json_report,
    export_markdown_report,
)


def _sample_report():
    return {
        "summary": {
            "model": "azure/gpt-5-mini",
            "judge_model": "azure/gpt-5-mini",
            "execution_date": "20260206",
            "evaluation_date": "2026-02-06T17:30:00",
            "total_tasks": 1,
            "evaluated_tasks": 1,
            "sample_size": 1,
            "release_scope": "AB",
            "bootstrap_ci_enabled": True,
            "bootstrap_samples": 1000,
            "v2_scoring": {
                "overall_score": 0.8123,
                "tracks": {
                    "A": {"weight": 0.4, "deterministic_score": 0.8, "judge_score": 0.9, "track_score": 0.815},
                    "B": {"weight": 0.35, "deterministic_score": 0.7, "judge_score": 0.8, "track_score": 0.73},
                    "C": {"weight": 0.25, "deterministic_score": None, "judge_score": None, "track_score": None},
                },
                "note": "Track C 데이터가 없으면 A/B 가중치로 자동 재정규화됩니다.",
            },
        },
        "by_level": {
            "L1": {
                "total_tasks": 1,
                "evaluated_tasks": 1,
                "metrics": {
                    "RRR": 1.0,
                    "SR": 0.75,
                    "TRF1": 1.0,
                    "CtxRecall": None,
                },
                "metric_confidence_intervals": {
                    "RRR": {"lower": 1.0, "upper": 1.0},
                    "SR": {"lower": 0.5, "upper": 1.0},
                },
                "metadata": {
                    "average_execution_time": 1.23,
                    "average_tokens_per_task": 123.0,
                    "average_prompt_tokens": 80.0,
                    "average_completion_tokens": 43.0,
                    "average_tps": 56.0,
                    "total_tokens": 123,
                    "total_execution_time": 1.23,
                    "ttft": {"average": 0.4567, "min": 0.4, "max": 0.6},
                },
                "task_evaluations": [],
            }
        },
    }


def test_export_json_csv_markdown_reports():
    report = _sample_report()
    out_dir = Path(mkdtemp(prefix="report_export_test_"))
    json_file = out_dir / "evaluation_report.json"
    csv_file = out_dir / "evaluation_summary.csv"
    md_file = out_dir / "evaluation_report.md"

    export_json_report(report, json_file)
    export_csv_summary(report, csv_file, level_sort_key=lambda lvl: int(lvl[1:]))
    export_markdown_report(
        report,
        md_file,
        level_sort_key=lambda lvl: int(lvl[1:]),
        level_descriptions={"1": "단일 도구 호출"},
        metric_descriptions={"RRR": "응답 반환율", "SR": "성공률", "TRF1": "도구 F1"},
    )

    assert json_file.exists()
    assert csv_file.exists()
    assert md_file.exists()

    parsed_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert parsed_json["summary"]["model"] == "azure/gpt-5-mini"

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "Level"
    assert rows[1][0] == "L1"
    # CtxRecall is None, should be blank in CSV.
    header = rows[0]
    cidx = header.index("CtxRecall")
    assert rows[1][cidx] == ""

    md_text = md_file.read_text(encoding="utf-8")
    assert "## v2 Track 점수" in md_text
    assert "| **L1** |" in md_text
    assert "**RRR**" in md_text

