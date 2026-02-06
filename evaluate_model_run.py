"""Ko-AgentBench 모델 실행 결과 평가 스크립트

특정 날짜/모델의 벤치마크 결과를 평가하고 종합 리포트를 생성합니다.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from bench.runner.metrics import (
    EvalContext,
    METRICS,
    LLMJudgeMetric,
)
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.adapters.openrouter_adapter import OpenRouterAdapter
from bench.evaluation.metric_selector import select_metrics_for_task
from bench.evaluation.v2_export import export_v2_jsonl_from_report
from bench.evaluation.report_export import (
    export_csv_summary,
    export_json_report,
    export_markdown_report,
)
from bench.evaluation.scoring import (
    DEFAULT_JUDGE_METRICS,
    DEFAULT_TRACK_CONFIG,
    calculate_v2_scores,
    level_sort_key as sort_level_name,
)

# Import API keys from secrets (환경변수 설정은 main()에서)
try:
    from configs.secrets import (
        AZURE_API_KEY,
        AZURE_API_BASE,
        AZURE_API_VERSION,
        ANTHROPIC_API_KEY,
        GEMINI_API_KEY,
        OPENROUTER_API_KEY,
        OPENROUTER_APP_URL,
        OPENROUTER_APP_TITLE,
    )
except ImportError:
    print("[경고] configs.secrets를 import할 수 없습니다.")
    AZURE_API_KEY = None
    AZURE_API_BASE = None
    AZURE_API_VERSION = None
    ANTHROPIC_API_KEY = None
    GEMINI_API_KEY = None
    OPENROUTER_API_KEY = None
    OPENROUTER_APP_URL = None
    OPENROUTER_APP_TITLE = None


class ModelRunEvaluator:
    """모델 실행 결과 평가 클래스"""

    # Level별 태스크 의도 설명
    LEVEL_DESCRIPTIONS = {
        '1': '가장 기본적인 API 호출 능력 검증. 주어진 단일 도구를 정확한 파라미터로 호출할 수 있는지 확인',
        '2': '여러 후보 도구 중 최적의 API를 선택하는 능력 검증. 주어진 도구 목록 중 가장 적합한 도구를 선택할 수 있는지 확인',
        '3': '여러 도구를 순차적으로 호출하고, 한 도구의 결과를 다음 도구의 입력으로 사용하여 복잡한 문제를 해결하는 능력 검증',
        '4': '여러 도구를 병렬적으로 호출하여 얻은 정보를 종합하고, 비교·분석하여 최종 결론을 도출하는 능력 검증',
        '5': '정보가 부족하거나 API 호출이 실패하는 등 예외적인 상황에 대처하는 능력 검증',
        '6': '이전 대화에서 얻은 Tool 호출 결과를 기억하고, 불필요한 API 재호출 없이 효율적으로 답변하는 능력 검증',
        '7': '여러 턴에 걸친 대화의 핵심 맥락을 기억하고, 이를 새로운 질문과 연결하여 정확한 Tool calling을 수행하는 능력 검증'
    }

    # Metric별 상세 설명
    METRIC_DESCRIPTIONS = {
        'RRR': '응답 반환율. 모델이 기술적 오류(Exception, Timeout 등) 없이 최종 응답을 반환했는지 여부',
        'ArgAcc': '인자 정확도. 도구 호출 시 전달한 인자들이 정답과 일치하는지 평가',
        'CallEM': '호출 완전 일치. 도구명과 모든 인자가 정답과 완벽하게 일치하는지 평가',
        'EPR_CVR': '유효 호출 비율. 생성한 도구 호출이 스키마상 유효하고 실행 가능한지 평가',
        'RespOK': '응답 파싱 성공. 도구 실행 결과를 성공적으로 파싱했는지 평가',
        'SR': '성공률. 주어진 태스크를 최종적으로 성공했는지 여부',
        'ToolAcc': '도구 선택 정확도. 정답 도구를 정확하게 선택했는지 평가',
        'pass@k': '태스크별 pass@k. k회 시도 중 최소 1회 성공했는지(0/1) 평균',
        'stability@k': '반복 안정성. 태스크를 k번 반복 수행했을 때 성공 비율 평균',
        'SelectAcc': '최종 선택 도구 정확도. 여러 후보군 중에서 최종적으로 올바른 도구를 선택했는지 평가',
        'FSM': '정답 경로 완전 일치. 정해진 도구 호출 순서와 완벽하게 일치하는지 평가',
        'PSM': '정답 경로 부분 일치. 정답 경로의 일부를 얼마나 포함하고 있는지 평가',
        'ΔSteps_norm': '최소 경로 대비 효율. 이론적인 최소 호출 횟수 대비 얼마나 효율적인 경로를 생성했는지 평가',
        'Coverage': '소스 커버리지. 정보를 수집해야 하는 여러 소스를 누락 없이 호출했는지 평가',
        'SourceEPR': '소스별 유효 호출 비율. 병렬적으로 호출한 각 도구가 유효했는지 개별적으로 평가',
        'AdaptiveRoutingScore': '적응형 라우팅 점수. 주입된 도구 실패 이후 얼마나 신속하게 대체 경로로 전환하는지 평가',
        'FallbackSR': '대체 경로 성공률. 특정 도구 실패 시 다른 도구를 활용해 성공하는 비율',
        'EffScore': '효율 점수. 이론적 최소 호출 수 대비 실제 호출 수로 계산한 효율성',
        'RedundantCallRate': '불필요 호출 비율. 정보를 이미 알고 있음에도 불필요하게 도구를 다시 호출하는 비율',
        'ContextRetention': '맥락 유지율. 여러 턴에 걸친 대화의 핵심 맥락을 답변에 올바르게 유지하는지 평가',
        'RefRecall': '장기 회상 비율. 대화 초반의 정보를 마지막 턴에서 다시 질문했을 때 정확히 기억해내는지 평가',
        'TRF1': '필수 도구 집합 탐지 F1 (Tool Retrieval F1)',
        'ArgF1': '필수 도구 인자 키/값 정합성 F1 (Argument Grounding F1)',
        'ExecSR': '유효 실행 성공 비율 (Execution Success Rate)',
        'RecoverSR': '실패 주입 후 대체 경로로 회복 성공했는지 비율',
        'StepEff': '최소 스텝 대비 실제 스텝 효율 (min(1, min_steps/actual_steps))',
        'GroundAns': '도구 결과 필드 커버리지 기반 근거 응답 점수',
        'CtxRecall': '대화 맥락 테스트(기억해야 할 제약/선호) 충족 비율',
        'PolicySafe': '안전/정책 위반 없는 응답/도구 사용 준수 점수'
    }

    TRACK_CONFIG = DEFAULT_TRACK_CONFIG
    JUDGE_METRICS = DEFAULT_JUDGE_METRICS

    def __init__(
            self,
            date: str,
            model: str,
            judge_models: List[str],
            sample_size: Optional[int] = None,
            levels: Optional[List[str]] = None,
            verbose: bool = False,
            release_scope: str = "AB",
            bootstrap_samples: int = 1000,
            enable_bootstrap_ci: bool = True,
            judge_max_retries: int = 5,
            judge_backoff_base: float = 2.0,
            judge_backoff_cap: float = 120.0,
            strict_v2_eval_schema: bool = False,
    ):
        self.date = date
        self.model = model
        self.judge_models = judge_models
        self.sample_size = sample_size
        self.levels = levels  # 평가할 레벨 목록
        self.verbose = verbose
        self.release_scope = (release_scope or "AB").upper()
        self.bootstrap_samples = max(1, int(bootstrap_samples))
        self.enable_bootstrap_ci = enable_bootstrap_ci
        self.judge_max_retries = max(1, int(judge_max_retries))
        self.judge_backoff_base = max(0.0, float(judge_backoff_base))
        self.judge_backoff_cap = max(self.judge_backoff_base, float(judge_backoff_cap))
        self.strict_v2_eval_schema = strict_v2_eval_schema
        self.judge_adapters = []

        # LLM Judge adapter 초기화
        print(f"[초기화] Judge 모델 로딩: {', '.join(self.judge_models)}")

        for judge_model in self.judge_models:
            try:
                if judge_model.lower().startswith("openrouter/"):
                    adapter = OpenRouterAdapter(judge_model)
                else:
                    adapter = LiteLLMAdapter(judge_model)
                self.judge_adapters.append(adapter)
                print(f"{judge_model} 초기화 완료")
            except Exception as e:
                print(f"{judge_model} 초기화 실패: {e}")

        if not self.judge_adapters:
            print(f"[경고] Judge 모델 초기화 실패")
        else:
            self._inject_judge_to_metrics()
            self._configure_judge_retry_policies()
            print(f"[OK] {len(self.judge_adapters)}개 Judge 모델 초기화 완료")

    def _inject_judge_to_metrics(self):
        """Judge 모델을 필요한 메트릭에 주입"""
        if not self.judge_adapters:
            return

        judge_metrics = [
            'SR',               # 공통
            'ArgAcc',           # L1
            'EffScore',         # L6
            'ContextRetention', # L7
            'RefRecall'         # L7
        ]

        for metric_name in judge_metrics:
            if metric_name in METRICS:
                if hasattr(METRICS[metric_name], '__dict__'):
                    METRICS[metric_name].llm_adapters = self.judge_adapters
                    METRICS[metric_name].llm_adapter = self.judge_adapters[0]

    def _configure_judge_retry_policies(self):
        """Apply judge retry/backoff policy to all LLM-judge metrics."""
        for metric in METRICS.values():
            if isinstance(metric, LLMJudgeMetric):
                metric.judge_max_retries = self.judge_max_retries
                metric.judge_backoff_base = self.judge_backoff_base
                metric.judge_backoff_cap = self.judge_backoff_cap

    @staticmethod
    def _level_sort_key(level_name: str) -> int:
        return sort_level_name(level_name)



    def find_result_files(self) -> Dict[str, Path]:
        """날짜와 모델로 레벨 결과 파일(L1, L2, ..., L10)을 찾기"""
        results_dir = Path('logs/benchmark_results')

        # 모델명을 파일명 패턴으로 변환 (azure/gpt-4.1 -> azure_gpt-4.1)
        model_pattern = self.model.replace('/', '_')

        level_files = {}

        # 기존 패턴: logs/benchmark_results/L{level}_{model}_{date}*.json)
        pattern = f"L*_{model_pattern}_{self.date}*.json"
        files = list(results_dir.glob(pattern))

        for f in files:
            level = f.name.split('_')[0]
            if level.startswith('L') and level[1:].isdigit():
                level_files[level] = f

        # 변경 패턴: logs/benchmark_results/by_model/{model}/{date_timestamp}/L*.json)
        if not level_files:
            by_model_dir = results_dir / 'by_model' / model_pattern

            if by_model_dir.exists():
                date_dirs = sorted(
                    [d for d in by_model_dir.iterdir() if d.is_dir() and d.name.startswith(self.date)],
                    key=lambda p: p.name,
                    reverse=True,
                )

                for date_dir in date_dirs:
                    level_candidates = sorted(
                        [p for p in date_dir.glob("L*.json") if p.stem.startswith("L") and p.stem[1:].isdigit()],
                        key=lambda p: self._level_sort_key(p.stem),
                    )
                    for level_file in level_candidates:
                        level_name = level_file.stem
                        if level_name not in level_files:
                            level_files[level_name] = level_file

        return level_files

    def evaluate_level(self, file_path: Path, level: str) -> Dict[str, Any]:
        """특정 레벨의 결과 파일 평가"""
        level_num = int(level[1:])  # "L10" -> 10

        # 결과 파일 로드
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        tasks = data.get("results", [])

        if self.verbose:
            print(f"  파일: {file_path.name}")
            print(f"  전체 태스크: {len(tasks)}")

        if not tasks:
            print(f"  [경고] 평가할 태스크가 없습니다.")
            return {
                "level": level_num,
                "file": file_path.name,
                "metadata": metadata,
                "total_tasks": 0,
                "evaluated_tasks": 0,
                "metric_averages": {},
                "task_evaluations": []
            }

        # 샘플링
        if self.sample_size:
            tasks_to_evaluate = tasks[:self.sample_size]
            if self.verbose:
                print(f"  샘플링: {len(tasks_to_evaluate)}개 태스크")
        else:
            tasks_to_evaluate = tasks

        # 각 태스크 평가
        all_evaluations = []
        metric_name_union = set()

        for idx, task_result in enumerate(tasks_to_evaluate, 1):
            task_id = task_result.get("task_id", f"unknown_{idx}")

            if self.verbose:
                print(f"  [{idx}/{len(tasks_to_evaluate)}] {task_id}")

            # task_schema 구성 
            task_schema = {
                "task_id": task_result.get("task_id"),
                "instruction": task_result.get("instruction"),
                "task_level": task_result.get("level"),
                "task_category": task_result.get("category"),
                "golden_action": task_result.get("golden_action", []),
                "minimum_steps": task_result.get("minimum_steps"),
                "data_flow": task_result.get("data_flow", []),
                "available_tools": task_result.get("expected_tools", []),
                "error_injection": task_result.get("error_injection"),
                "fallback_options": task_result.get("fallback_options", []),
                "resp_schema": task_result.get("resp_schema"),
                "arg_schema": task_result.get("arg_schema"),
                "repetitions": task_result.get("repetitions", 1),  # pass@k용
                "schema_version": task_result.get("schema_version"),
                "track": task_result.get("track"),
                "metadata": task_result.get("metadata", {}),
                "golden": task_result.get("golden", {}),
                "evaluation": task_result.get("evaluation", {}),
                "context_tests": task_result.get("context_tests", []),
                "conversation_tracking": task_result.get("conversation_tracking"),
            }

            # logs 구성
            logs = {
                "success": task_result.get("success", False),
                "tool_invocations": task_result.get("tool_calls", []),
                "tool_calls": task_result.get("tool_calls", []),  # AdaptiveRoutingScore/FallbackSR 평가용
                "actual_output": task_result.get("final_response", ""),
                "final_response": task_result.get("final_response", ""),
                "conversation_log": task_result.get("conversation_log", {}),
                "repetition_results": task_result.get("repetition_results", []),  # pass@k용
            }

            # EvalContext 생성 및 메트릭 평가
            try:
                ctx = EvalContext(task_schema=task_schema, logs=logs)
                metrics = select_metrics_for_task(level_num, task_schema)

                task_evaluation = {
                    "task_id": task_id,
                    "success": task_result.get("success", False),
                    "level": level_num,
                    "track": task_schema.get("track"),
                    "evaluation": task_schema.get("evaluation", {}),
                    "repetitions": task_schema.get("repetitions", 1),
                    "metrics": {}
                }

                for metric_name, metric in metrics.items():
                    try:
                        result = metric.evaluate(ctx)
                        task_evaluation["metrics"][metric_name] = {
                            "score": result.score,
                            "details": result.details
                        }
                    except Exception as e:
                        if self.verbose:
                            print(f"    [오류] {metric_name}: {e}")
                        task_evaluation["metrics"][metric_name] = {
                            "score": 0.0,
                            "error": str(e)
                        }

                metric_name_union.update(task_evaluation["metrics"].keys())
                all_evaluations.append(task_evaluation)

            except Exception as e:
                print(f"  [오류] {task_id} 평가 실패: {e}")
                continue

        # RRR (Response Return Rate) 계산 로직 추가
        # 기술적 오류 없이 응답을 반환한 비율
        successful_responses = sum(1 for task in all_evaluations if task.get("success", False))
        rrr_score = successful_responses / len(all_evaluations) if all_evaluations else 0.0

        # 메트릭별 평균 계산
        metric_averages = {}
        metric_confidence_intervals = {}
        for metric_name in sorted(metric_name_union):
            scores = [
                task["metrics"].get(metric_name, {}).get("score", 0.0)
                for task in all_evaluations
                if metric_name in task.get("metrics", {})
            ]
            # None 값 필터링
            valid_scores = [score for score in scores if score is not None]
            if valid_scores:
                metric_averages[metric_name] = sum(valid_scores) / len(valid_scores)
                if self.enable_bootstrap_ci:
                    ci = self._bootstrap_mean_ci(valid_scores, self.bootstrap_samples)
                    if ci:
                        metric_confidence_intervals[metric_name] = ci
            else:
                # Not applicable metrics should stay None and be excluded from aggregation.
                metric_averages[metric_name] = None
        metric_averages["RRR"] = rrr_score
        if self.enable_bootstrap_ci:
            rrr_scores = [1.0 if task.get("success", False) else 0.0 for task in all_evaluations]
            rrr_ci = self._bootstrap_mean_ci(rrr_scores, self.bootstrap_samples)
            if rrr_ci:
                metric_confidence_intervals["RRR"] = rrr_ci

        return {
            "level": level_num,
            "file": file_path.name,
            "metadata": metadata,
            "total_tasks": len(tasks),
            "evaluated_tasks": len(all_evaluations),
            "metric_averages": metric_averages,
            "metric_confidence_intervals": metric_confidence_intervals,
            "task_evaluations": all_evaluations
        }

    def evaluate(self) -> Dict[str, Any]:
        """전체 평가 실행"""
        print(f"\n{'='*80}")
        print(f"평가 시작")
        print(f"{'='*80}")
        print(f"모델: {self.model}")
        print(f"날짜: {self.date}")
        if self.levels:
            print(f"평가 레벨: {', '.join(self.levels)}")
        if self.sample_size:
            print(f"샘플링: 레벨당 {self.sample_size}개")
        print(f"{'='*80}\n")

        # 결과 파일 찾기
        level_files = self.find_result_files()

        if not level_files:
            raise FileNotFoundError(
                f"날짜 {self.date}, 모델 {self.model}에 해당하는 결과 파일을 찾을 수 없습니다."
            )

        # 평가할 레벨 필터링
        if self.levels:
            # 대소문자 구분 없이 처리 (L1, l1 모두 허용)
            requested_levels = [lvl.upper() if not lvl.startswith('L') else lvl for lvl in self.levels]
            requested_levels = ['L' + lvl if not lvl.startswith('L') else lvl for lvl in requested_levels]
            
            filtered_files = {k: v for k, v in level_files.items() if k in requested_levels}
            
            if not filtered_files:
                available = ', '.join(sorted(level_files.keys(), key=self._level_sort_key))
                requested = ', '.join(requested_levels)
                raise ValueError(
                    f"요청한 레벨({requested})을 찾을 수 없습니다.\n"
                    f"사용 가능한 레벨: {available}"
                )
            level_files = filtered_files

        print(f"발견된 파일:")
        for level, path in sorted(level_files.items(), key=lambda x: self._level_sort_key(x[0])):
            print(f"  {level}: {path.name}")
        print()

        # 각 레벨 평가
        by_level = {}
        total_tasks = 0
        evaluated_tasks = 0

        for level in sorted(level_files.keys(), key=self._level_sort_key):
            print(f"[{level}] 평가 중...")
            level_result = self.evaluate_level(level_files[level], level)

            by_level[level] = {
                "file": level_result["file"],
                "total_tasks": level_result["total_tasks"],
                "evaluated_tasks": level_result["evaluated_tasks"],
                "metrics": level_result["metric_averages"],
                "metric_confidence_intervals": level_result["metric_confidence_intervals"],
                "metadata": level_result["metadata"],
                "task_evaluations": level_result["task_evaluations"]
            }

            total_tasks += level_result["total_tasks"]
            evaluated_tasks += level_result["evaluated_tasks"]

            print(f"  완료: {level_result['evaluated_tasks']}개 태스크 평가\n")

        v2_scoring = calculate_v2_scores(
            by_level,
            track_config=self.TRACK_CONFIG,
            judge_metrics=set(self.JUDGE_METRICS),
        )

        # 종합 리포트
        report = {
            "summary": {
                "model": self.model,
                "judge_model": ", ".join(self.judge_models),
                "execution_date": self.date,
                "evaluation_date": datetime.now().isoformat(),
                "total_tasks": total_tasks,
                "evaluated_tasks": evaluated_tasks,
                "sample_size": self.sample_size,
                "levels_evaluated": len(by_level),
                "release_scope": self.release_scope,
                "bootstrap_ci_enabled": self.enable_bootstrap_ci,
                "bootstrap_samples": self.bootstrap_samples,
                "v2_scoring": v2_scoring,
            },
            "by_level": by_level
        }

        print(f"{'='*80}")
        print(f"평가 완료")
        print(f"{'='*80}")
        print(f"총 레벨: {len(by_level)}")
        print(f"총 태스크: {evaluated_tasks}/{total_tasks}")
        print(f"{'='*80}\n")

        return report

    @staticmethod
    def _bootstrap_mean_ci(
        scores: List[float],
        bootstrap_samples: int,
        alpha: float = 0.05,
    ) -> Optional[Dict[str, Any]]:
        """Bootstrap mean CI for per-task metric scores."""
        if not scores:
            return None
        n = len(scores)
        mean_score = sum(scores) / n
        if n == 1 or bootstrap_samples <= 1:
            return {
                "mean": mean_score,
                "lower": mean_score,
                "upper": mean_score,
                "alpha": alpha,
                "n": n,
                "bootstrap_samples": bootstrap_samples,
            }

        rng = random.Random(42)
        means: List[float] = []
        for _ in range(bootstrap_samples):
            sample_sum = 0.0
            for _ in range(n):
                sample_sum += scores[rng.randrange(n)]
            means.append(sample_sum / n)

        means.sort()
        lower_idx = max(0, min(bootstrap_samples - 1, int((alpha / 2) * bootstrap_samples)))
        upper_idx = max(0, min(bootstrap_samples - 1, int((1 - alpha / 2) * bootstrap_samples) - 1))
        lower = means[lower_idx]
        upper = means[upper_idx]
        return {
            "mean": mean_score,
            "lower": lower,
            "upper": upper,
            "alpha": alpha,
            "n": n,
            "bootstrap_samples": bootstrap_samples,
        }

    def export_json(self, report: Dict[str, Any], output_dir: Path):
        """JSON 리포트 생성"""
        output_file = output_dir / "evaluation_report.json"
        export_json_report(report, output_file)
        print(f"[저장] JSON: {output_file}")

    def export_csv(self, report: Dict[str, Any], output_dir: Path):
        """CSV 리포트 생성"""
        output_file = output_dir / "evaluation_summary.csv"
        export_csv_summary(report, output_file, level_sort_key=self._level_sort_key)
        print(f"[저장] CSV: {output_file}")

    def export_markdown(self, report: Dict[str, Any], output_dir: Path):
        """Markdown 리포트 생성"""
        output_file = output_dir / "evaluation_report.md"
        export_markdown_report(
            report,
            output_file,
            level_sort_key=self._level_sort_key,
            level_descriptions=self.LEVEL_DESCRIPTIONS,
            metric_descriptions=self.METRIC_DESCRIPTIONS,
        )
        print(f"[저장] Markdown: {output_file}")

    def export_v2_jsonl(self, report: Dict[str, Any], output_dir: Path):
        """v2 evaluation.schema 호환 JSONL 내보내기."""
        output_file = output_dir / "evaluation_v2.jsonl"
        export_result = export_v2_jsonl_from_report(
            report,
            output_file=output_file,
            model_id=self.model,
            date=self.date,
            track_config=self.TRACK_CONFIG,
            judge_metrics=set(self.JUDGE_METRICS),
        )

        records = export_result.get("records", [])
        validation = export_result.get("validation", {})
        print(f"[저장] v2 JSONL: {output_file} ({len(records)} records)")

        if validation.get("validator_available"):
            error_count = len(validation.get("errors", []))
            if error_count == 0:
                print("[검증] v2 evaluation schema 검증 통과")
            else:
                first = validation["errors"][0]
                first_message = "; ".join(first.get("messages", [])[:3])
                print(
                    f"[경고] v2 evaluation schema 검증 실패: errors={error_count} "
                    f"(first={first.get('evaluation_id')} - {first_message})"
                )
                if self.strict_v2_eval_schema:
                    raise ValueError(
                        "strict_v2_eval_schema=True and v2 evaluation schema validation failed."
                    )
        else:
            print("[경고] v2 evaluation schema validator를 사용할 수 없어 검증을 건너뜁니다.")

    def export(self, report: Dict[str, Any], output_dir: str, formats: List[str]):
        """결과 내보내기"""
        output_path = Path(output_dir)

        # 출력 디렉토리 생성
        model_safe = self.model.replace('/', '_')
        final_output_dir = output_path / f"{model_safe}_{self.date}"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"결과 저장")
        print(f"{'='*80}")

        # 형식별 저장
        if 'all' in formats:
            formats = ['json', 'csv', 'markdown', 'v2-jsonl']

        if 'json' in formats:
            self.export_json(report, final_output_dir)

        if 'csv' in formats:
            self.export_csv(report, final_output_dir)

        if 'markdown' in formats:
            self.export_markdown(report, final_output_dir)

        if 'v2-jsonl' in formats:
            self.export_v2_jsonl(report, final_output_dir)

        print(f"{'='*80}\n")


def main():
    """메인 함수"""
    load_dotenv()

    # Set API keys from secrets.py to environment variables
    if AZURE_API_KEY:
        os.environ['AZURE_API_KEY'] = AZURE_API_KEY
    if AZURE_API_BASE:
        os.environ['AZURE_API_BASE'] = AZURE_API_BASE
    if AZURE_API_VERSION:
        os.environ['AZURE_API_VERSION'] = AZURE_API_VERSION
    if ANTHROPIC_API_KEY:
        os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    if GEMINI_API_KEY:
        os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
    if OPENROUTER_API_KEY:
        os.environ['OPENROUTER_API_KEY'] = OPENROUTER_API_KEY
    if OPENROUTER_APP_URL:
        os.environ.setdefault('OPENROUTER_APP_URL', OPENROUTER_APP_URL)
    if OPENROUTER_APP_TITLE:
        os.environ.setdefault('OPENROUTER_APP_TITLE', OPENROUTER_APP_TITLE)

    parser = argparse.ArgumentParser(
        description='Ko-AgentBench 모델 실행 결과 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용 (AB scope: L1-L7, 기본 Judge 앙상블)
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1
  
  # 특정 레벨만 평가
  python evaluate_model_run.py --date 20251021 --model anthropic/claude-sonnet-4-5 --levels L1
  python evaluate_model_run.py --date 20251021 --model anthropic/claude-sonnet-4-5 --levels L1,L3,L5
  
  # Judge 모델 지정
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1 --judge-models openai/gpt-5.1 anthropic/claude-sonnet-4-5 gemini/gemini-2.5-pro
  
  # 빠른 테스트 (각 레벨당 1개)
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1 --quick
  
  # 샘플링
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1 --sample 10
        """
    )

    # 필수 파라미터
    parser.add_argument('--date', required=True,
                        help='실행 날짜 (예: 20251016)')
    parser.add_argument('--model', required=True,
                        help='평가 대상 모델 (예: azure/gpt-4.1)')

    # 평가 설정
    parser.add_argument('--levels', type=str, default=None,
                        help='평가할 레벨 지정 (예: L1, L1,L3,L5). 미지정 시 --release-scope 적용')
    parser.add_argument('--release-scope', choices=['AB', 'ALL'], default='AB',
                        help='평가 릴리스 범위. AB=Track A+B(L1-L7, 기본), ALL=가능한 전체 레벨')
    parser.add_argument('--judge-models', nargs='+', default=None,
                        help='LLM Judge 모델(들). 기본: 3-model 앙상블. 필요 시 단일 모델로 지정 가능')
    parser.add_argument('--sample', type=int, default=None,
                        help='레벨당 평가할 태스크 수 (기본: 전체)')
    parser.add_argument('--quick', action='store_true',
                        help='빠른 테스트 (각 레벨당 1개)')
    parser.add_argument('--bootstrap-samples', type=int, default=1000,
                        help='부트스트랩 샘플 수 (95%% CI 계산용, 기본: 1000)')
    parser.add_argument('--no-bootstrap-ci', action='store_true',
                        help='부트스트랩 CI 계산 비활성화')
    parser.add_argument('--judge-max-retries', type=int, default=5,
                        help='Judge API 호출 최대 재시도 횟수 (429 대응, 기본: 5)')
    parser.add_argument('--judge-backoff-base', type=float, default=2.0,
                        help='Judge 재시도 백오프 시작 초(기본: 2.0)')
    parser.add_argument('--judge-backoff-cap', type=float, default=120.0,
                        help='Judge 재시도 백오프 최대 초(기본: 120.0)')
    parser.add_argument('--strict-v2-eval-schema', action='store_true',
                        help='v2 evaluation.schema 검증 실패 시 즉시 에러로 종료')
    parser.add_argument('--log-file', help='특정 로그 파일 경로 (선택사항)')

    # 출력 설정
    parser.add_argument('--output', default='reports',
                        help='출력 디렉토리 (기본: reports)')
    parser.add_argument('--format', nargs='+',
                        default=['json', 'csv', 'markdown'],
                        choices=['json', 'csv', 'markdown', 'v2-jsonl', 'all'],
                        help='출력 형식 (기본: json csv markdown)')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 로그 출력')

    args = parser.parse_args()

    # Judge 모델 설정
    if args.judge_models:
        judge_models = args.judge_models
    else:
        # 기본값: provider 다양성을 갖춘 Judge 앙상블
        judge_models = [
            "openai/gpt-5.1",
            "anthropic/claude-sonnet-4-5",
            "gemini/gemini-2.5-pro",
        ]

    # 레벨 파싱
    levels = None
    if args.levels:
        # 쉼표로 구분된 레벨 파싱 (예: "L1,L3,L5" 또는 "1,3,5")
        levels = [lvl.strip() for lvl in args.levels.split(',')]
        print(f"[설정] 평가 대상 레벨: {', '.join(levels)}")
    elif args.release_scope.upper() == "AB":
        levels = [f"L{i}" for i in range(1, 8)]
        print(f"[설정] 평가 대상 레벨(AB scope): {', '.join(levels)}")

    # 평가 실행
    try:
        evaluator = ModelRunEvaluator(
            date=args.date,
            model=args.model,
            judge_models=judge_models,
            sample_size=1 if args.quick else args.sample,
            levels=levels,
            verbose=args.verbose,
            release_scope=args.release_scope,
            bootstrap_samples=args.bootstrap_samples,
            enable_bootstrap_ci=(not args.no_bootstrap_ci),
            judge_max_retries=args.judge_max_retries,
            judge_backoff_base=args.judge_backoff_base,
            judge_backoff_cap=args.judge_backoff_cap,
            strict_v2_eval_schema=args.strict_v2_eval_schema,
        )

        report = evaluator.evaluate()
        evaluator.export(report, args.output, formats=args.format)

        print("\n[완료] 평가가 성공적으로 완료되었습니다.\n")

    except Exception as e:
        print(f"\n[오류] 평가 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
