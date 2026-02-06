"""Report export helpers for evaluation outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict


def export_json_report(report: Dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def export_csv_summary(
    report: Dict[str, Any],
    output_file: Path,
    *,
    level_sort_key: Callable[[str], int],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_metrics = set()
    for level_data in report.get("by_level", {}).values():
        all_metrics.update((level_data.get("metrics") or {}).keys())

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "Level",
            "Total_Tasks",
            "Evaluated_Tasks",
            "Avg_Exec_Time",
            "Avg_Tokens",
            "Avg_TPS",
            "Avg_TTFT",
        ] + sorted(all_metrics)
        writer.writerow(header)

        for level in sorted(report.get("by_level", {}).keys(), key=level_sort_key):
            level_data = report["by_level"][level]
            metadata = level_data.get("metadata", {})
            row = [
                level,
                level_data.get("total_tasks", 0),
                level_data.get("evaluated_tasks", 0),
                f"{metadata.get('average_execution_time', 0):.2f}",
                f"{metadata.get('average_tokens_per_task', 0):.2f}",
                f"{metadata.get('average_tps', 0):.2f}",
                f"{metadata.get('ttft', {}).get('average', 0):.4f}",
            ]
            for metric in sorted(all_metrics):
                score = (level_data.get("metrics") or {}).get(metric)
                row.append(f"{score:.4f}" if isinstance(score, (int, float)) else "")
            writer.writerow(row)


def export_markdown_report(
    report: Dict[str, Any],
    output_file: Path,
    *,
    level_sort_key: Callable[[str], int],
    level_descriptions: Dict[str, str],
    metric_descriptions: Dict[str, str],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    key_metrics = {
        "L1": ["TRF1", "ArgF1", "ToolAcc", "ArgAcc"],
        "L2": ["TRF1", "ArgF1", "SelectAcc"],
        "L3": ["StepEff", "GroundAns", "FSM", "PSM"],
        "L4": ["StepEff", "GroundAns", "Coverage"],
        "L5": ["RecoverSR", "StepEff", "AdaptiveRoutingScore", "FallbackSR"],
        "L6": ["CtxRecall", "StepEff", "EffScore", "RedundantCallRate"],
        "L7": ["CtxRecall", "GroundAns", "ContextRetention"],
        "L8": ["TRF1", "ArgF1", "ExecSR", "PolicySafe"],
        "L9": ["TRF1", "ArgF1", "ExecSR", "PolicySafe"],
        "L10": ["TRF1", "ArgF1", "ExecSR", "PolicySafe"],
    }

    level_names = {
        "1": "Level 1: 단일 도구 호출",
        "2": "Level 2: 도구 선택",
        "3": "Level 3: 멀티스텝 추론",
        "4": "Level 4: 멀티소스 통합",
        "5": "Level 5: 오류 처리",
        "6": "Level 6: 컨텍스트 재사용",
        "7": "Level 7: 멀티턴 대화",
        "8": "Level 8: MCP-Live 기초",
        "9": "Level 9: MCP-Live 복합",
        "10": "Level 10: MCP-Live 고난도",
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Ko-AgentBench 평가 보고서\n\n")

        summary = report.get("summary", {})
        f.write("## 실행 정보\n\n")
        f.write(f"- **평가 대상 모델**: {summary.get('model', '')}\n")
        f.write(f"- **Judge 모델**: {summary.get('judge_model', '')}\n")
        f.write(f"- **실행 날짜**: {summary.get('execution_date', '')}\n")
        evaluation_date = str(summary.get("evaluation_date", ""))
        f.write(f"- **평가 날짜**: {evaluation_date[:10]}\n")
        f.write(f"- **릴리스 범위**: {summary.get('release_scope', 'N/A')}\n")
        f.write(f"- **Bootstrap CI**: {summary.get('bootstrap_ci_enabled', False)}")
        if summary.get("bootstrap_ci_enabled", False):
            f.write(f" (samples={summary.get('bootstrap_samples', 0)})")
        f.write("\n")
        f.write(f"- **총 태스크**: {summary.get('evaluated_tasks', 0)}/{summary.get('total_tasks', 0)}\n")
        if summary.get("sample_size"):
            f.write(f"- **샘플링**: 레벨당 {summary.get('sample_size')}개\n")

        v2_scoring = summary.get("v2_scoring", {})
        if isinstance(v2_scoring, dict) and v2_scoring.get("overall_score") is not None:
            f.write(f"- **v2 종합 점수 (Track 가중 평균)**: {v2_scoring['overall_score']:.4f}\n")
        f.write("\n")

        if isinstance(v2_scoring, dict) and isinstance(v2_scoring.get("tracks"), dict):
            f.write("## v2 Track 점수\n\n")
            f.write("| Track | Weight | Deterministic | Judge | Track Score |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            for track_name in ["A", "B", "C"]:
                track_data = v2_scoring["tracks"].get(track_name, {})

                def _fmt(v: Any) -> str:
                    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

                f.write(
                    f"| **{track_name}** | {_fmt(track_data.get('weight'))} | "
                    f"{_fmt(track_data.get('deterministic_score'))} | "
                    f"{_fmt(track_data.get('judge_score'))} | {_fmt(track_data.get('track_score'))} |\n"
                )
            note = v2_scoring.get("note")
            if note:
                f.write(f"\n> {note}\n\n")

        f.write("## 📊 성능 요약\n\n")
        f.write("| Level | 태스크 수 | 평균 실행시간 | 평균 TPS | 평균 TTFT | 주요 지표 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")

        by_level = report.get("by_level", {})
        for level in sorted(by_level.keys(), key=level_sort_key):
            level_data = by_level[level]
            metadata = level_data.get("metadata", {})
            metrics = level_data.get("metrics", {})
            task_count = f"{level_data.get('evaluated_tasks', 0)}/{level_data.get('total_tasks', 0)}"
            exec_time = (
                f"{metadata.get('average_execution_time', 0):.1f}초"
                if "average_execution_time" in metadata
                else "N/A"
            )
            tps = f"{metadata.get('average_tps', 0):.0f}" if "average_tps" in metadata else "N/A"
            ttft = (
                f"{metadata.get('ttft', {}).get('average', 0):.3f}초"
                if "ttft" in metadata
                else "N/A"
            )

            key_metric_strs = []
            for km in key_metrics.get(level, []):
                if km in metrics and isinstance(metrics[km], (int, float)):
                    key_metric_strs.append(f"{km}: {metrics[km]:.3f}")
            key_metric_str = ", ".join(key_metric_strs) if key_metric_strs else "N/A"
            f.write(f"| **{level}** | {task_count} | {exec_time} | {tps} | {ttft} | {key_metric_str} |\n")
        f.write("\n")

        f.write("## 레벨별 성능\n\n")

        for level in sorted(by_level.keys(), key=level_sort_key):
            level_data = by_level[level]
            level_num = level[1:] if level.startswith("L") else level
            metric_cis = level_data.get("metric_confidence_intervals", {})
            f.write(f"### {level_names.get(level_num, level)}\n\n")

            if level_num in level_descriptions:
                f.write(f"> 태스크 의도: {level_descriptions[level_num]}\n>\n")

            f.write(f"- 태스크 수: {level_data.get('evaluated_tasks', 0)}/{level_data.get('total_tasks', 0)}\n")
            metadata = level_data.get("metadata", {})
            if "average_execution_time" in metadata:
                f.write(f"- 평균 실행시간: {metadata['average_execution_time']:.2f}초\n")
            f.write("\n**메트릭 점수:**\n\n")

            metrics = level_data.get("metrics", {})
            if metrics:
                priority_metrics = ["RRR", "SR"]
                for metric_name in priority_metrics:
                    if metric_name in metrics:
                        score = metrics[metric_name]
                        if not isinstance(score, (int, float)):
                            continue
                        description = metric_descriptions.get(metric_name, "설명 없음")
                        ci = metric_cis.get(metric_name)
                        ci_text = ""
                        if isinstance(ci, dict) and "lower" in ci and "upper" in ci:
                            ci_text = f" (95% CI: {ci['lower']:.3f}~{ci['upper']:.3f})"
                        f.write(f"- **{metric_name}**: {score:.3f}{ci_text} - {description}\n")

                for metric_name in sorted(metrics.keys()):
                    if metric_name in priority_metrics:
                        continue
                    score = metrics[metric_name]
                    if not isinstance(score, (int, float)):
                        continue
                    description = metric_descriptions.get(metric_name, "설명 없음")
                    ci = metric_cis.get(metric_name)
                    ci_text = ""
                    if isinstance(ci, dict) and "lower" in ci and "upper" in ci:
                        ci_text = f" (95% CI: {ci['lower']:.3f}~{ci['upper']:.3f})"
                    f.write(f"- **{metric_name}**: {score:.3f}{ci_text} - {description}\n")
            else:
                f.write("- (메트릭 없음)\n")
            f.write("\n")

        f.write("## 토큰 사용량 및 성능 지표\n\n")

        total_tokens = 0
        total_time = 0
        total_tasks = 0
        all_ttft_values = []

        for level_data in by_level.values():
            metadata = level_data.get("metadata", {})
            tasks = level_data.get("evaluated_tasks", 0)
            total_tasks += tasks
            total_tokens += metadata.get("total_tokens", 0)
            total_time += metadata.get("total_execution_time", 0)
            ttft_avg = metadata.get("ttft", {}).get("average", 0)
            if ttft_avg > 0:
                all_ttft_values.extend([ttft_avg] * tasks)

        overall_tps = total_tokens / total_time if total_time > 0 else 0
        overall_avg_ttft = sum(all_ttft_values) / len(all_ttft_values) if all_ttft_values else 0

        f.write("### 전체 요약\n\n")
        f.write(f"- **총 처리 토큰**: {total_tokens:,}개\n")
        f.write(f"- **총 실행 시간**: {total_time:.2f}초\n")
        f.write(f"- **전체 평균 TPS**: {overall_tps:.2f} tokens/sec\n")
        f.write(f"- **전체 평균 TTFT**: {overall_avg_ttft:.4f}초\n\n")

        f.write("### 레벨별 상세\n\n")
        f.write("| Level | 평균 토큰 수 | 입력 토큰 | 출력 토큰 | TPS | TTFT (평균) | TTFT (최소/최대) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")

        for level in sorted(by_level.keys(), key=level_sort_key):
            level_data = by_level[level]
            metadata = level_data.get("metadata", {})
            avg_tokens = metadata.get("average_tokens_per_task", 0)
            avg_prompt = metadata.get("average_prompt_tokens", 0)
            avg_completion = metadata.get("average_completion_tokens", 0)
            tps = metadata.get("average_tps", 0)
            ttft_data = metadata.get("ttft", {})
            ttft_avg = ttft_data.get("average", 0)
            ttft_min = ttft_data.get("min", 0)
            ttft_max = ttft_data.get("max", 0)
            ttft_range = f"{ttft_min:.4f} / {ttft_max:.4f}" if ttft_min > 0 else "N/A"
            f.write(
                f"| **{level}** | {avg_tokens:.0f} | {avg_prompt:.0f} | {avg_completion:.0f} | "
                f"{tps:.0f} | {ttft_avg:.4f}초 | {ttft_range} |\n"
            )
        f.write("\n")

