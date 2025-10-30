"""Ko-AgentBench 모델 실행 결과 평가 스크립트

특정 날짜/모델의 벤치마크 결과를 평가하고 종합 리포트를 생성합니다.
"""

import argparse
import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

from bench.runner.metrics import (
    get_metrics_for_level,
    EvalContext,
    METRICS
)
from bench.adapters.litellm_adapter import LiteLLMAdapter

# Import API keys from secrets (환경변수 설정은 main()에서)
try:
    from configs.secrets import (
        AZURE_API_KEY,
        AZURE_API_BASE,
        AZURE_API_VERSION,
        ANTHROPIC_API_KEY,
        GEMINI_API_KEY
    )
except ImportError:
    print("[경고] configs.secrets를 import할 수 없습니다.")
    AZURE_API_KEY = None
    AZURE_API_BASE = None
    AZURE_API_VERSION = None
    ANTHROPIC_API_KEY = None
    GEMINI_API_KEY = None


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
        'pass@k': '반복 안정성. 태스크를 k번 반복 수행했을 때 최소 한 번 이상 성공하는 비율',
        'SelectAcc': '최종 선택 도구 정확도. 여러 후보군 중에서 최종적으로 올바른 도구를 선택했는지 평가',
        'FSM': '정답 경로 완전 일치. 정해진 도구 호출 순서와 완벽하게 일치하는지 평가',
        'PSM': '정답 경로 부분 일치. 정답 경로의 일부를 얼마나 포함하고 있는지 평가',
        'ProvAcc': '데이터 출처 추적 정확도. 이전 단계의 출력값을 다음 단계의 입력값으로 정확히 연결했는지 평가',
        'ΔSteps_norm': '최소 경로 대비 효율. 이론적인 최소 호출 횟수 대비 얼마나 효율적인 경로를 생성했는지 평가',
        'Coverage': '소스 커버리지. 정보를 수집해야 하는 여러 소스를 누락 없이 호출했는지 평가',
        'SourceEPR': '소스별 유효 호출 비율. 병렬적으로 호출한 각 도구가 유효했는지 개별적으로 평가',
        'AdaptiveRoutingScore': '적응형 라우팅 점수. 주입된 도구 실패 이후 얼마나 신속하게 대체 경로로 전환하는지 평가',
        'FallbackSR': '대체 경로 성공률. 특정 도구 실패 시 다른 도구를 활용해 성공하는 비율',
        'EffScore': '효율 점수. 이론적 최소 호출 수와 재사용률을 종합하여 효율성을 점수화',
        'RedundantCallRate': '불필요 호출 비율. 정보를 이미 알고 있음에도 불필요하게 도구를 다시 호출하는 비율',
        'ReuseRate': '재사용 비율. 이전에 호출했던 결과를 재호출 없이 효율적으로 재사용하는 비율',
        'ContextRetention': '맥락 유지율. 여러 턴에 걸친 대화의 핵심 맥락을 답변에 올바르게 유지하는지 평가',
        'RefRecall': '장기 회상 비율. 대화 초반의 정보를 마지막 턴에서 다시 질문했을 때 정확히 기억해내는지 평가'
    }

    def __init__(
            self,
            date: str,
            model: str,
            judge_models: List[str],
            sample_size: Optional[int] = None,
            levels: Optional[List[str]] = None,
            verbose: bool = False
    ):
        self.date = date
        self.model = model
        self.judge_models = judge_models
        self.sample_size = sample_size
        self.levels = levels  # 평가할 레벨 목록
        self.verbose = verbose
        self.judge_adapters = []

        # LLM Judge adapter 초기화
        print(f"[초기화] Judge 모델 로딩: {', '.join(self.judge_models)}")

        for judge_model in self.judge_models:
            try:
                adapter = LiteLLMAdapter(judge_model)
                self.judge_adapters.append(adapter)
                print(f"{judge_model} 초기화 완료")
            except Exception as e:
                print(f"{judge_model} 초기화 실패: {e}")

        if not self.judge_adapters:
            print(f"[경고] Judge 모델 초기화 실패")
        else:
            self._inject_judge_to_metrics()
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



    def find_result_files(self) -> Dict[str, Path]:
        """날짜와 모델로 L1~L7 파일 찾기"""
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
                date_dirs = [d for d in by_model_dir.iterdir()
                             if d.is_dir() and d.name.startswith(self.date)]

                for level_name in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']:
                    for date_dir in date_dirs:
                        level_file = date_dir / f"{level_name}.json"
                        if level_file.exists() and level_name not in level_files:
                            level_files[level_name] = level_file
                            break

        return level_files

    def evaluate_level(self, file_path: Path, level: str) -> Dict[str, Any]:
        """특정 레벨의 결과 파일 평가"""
        level_num = int(level[1])  # "L1" -> 1

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

        # 해당 레벨의 메트릭 가져오기
        metrics = get_metrics_for_level(level_num)

        # 각 태스크 평가
        all_evaluations = []

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

                task_evaluation = {
                    "task_id": task_id,
                    "success": task_result.get("success", False),
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
        for metric_name in metrics.keys():
            scores = [
                task["metrics"].get(metric_name, {}).get("score", 0.0)
                for task in all_evaluations
                if metric_name in task.get("metrics", {})
            ]
            # None 값 필터링
            valid_scores = [score for score in scores if score is not None]
            if valid_scores:
                metric_averages[metric_name] = sum(valid_scores) / len(valid_scores)
            else:
                metric_averages[metric_name] = 0.0
        metric_averages["RRR"] = rrr_score

        return {
            "level": level_num,
            "file": file_path.name,
            "metadata": metadata,
            "total_tasks": len(tasks),
            "evaluated_tasks": len(all_evaluations),
            "metric_averages": metric_averages,
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
                available = ', '.join(sorted(level_files.keys()))
                requested = ', '.join(requested_levels)
                raise ValueError(
                    f"요청한 레벨({requested})을 찾을 수 없습니다.\n"
                    f"사용 가능한 레벨: {available}"
                )
            level_files = filtered_files

        print(f"발견된 파일:")
        for level, path in sorted(level_files.items()):
            print(f"  {level}: {path.name}")
        print()

        # 각 레벨 평가
        by_level = {}
        total_tasks = 0
        evaluated_tasks = 0

        for level in sorted(level_files.keys()):
            print(f"[{level}] 평가 중...")
            level_result = self.evaluate_level(level_files[level], level)

            by_level[level] = {
                "file": level_result["file"],
                "total_tasks": level_result["total_tasks"],
                "evaluated_tasks": level_result["evaluated_tasks"],
                "metrics": level_result["metric_averages"],
                "metadata": level_result["metadata"],
                "task_evaluations": level_result["task_evaluations"]
            }

            total_tasks += level_result["total_tasks"]
            evaluated_tasks += level_result["evaluated_tasks"]

            print(f"  완료: {level_result['evaluated_tasks']}개 태스크 평가\n")

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
                "levels_evaluated": len(by_level)
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

    def export_json(self, report: Dict[str, Any], output_dir: Path):
        """JSON 리포트 생성"""
        output_file = output_dir / "evaluation_report.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"[저장] JSON: {output_file}")

    def export_csv(self, report: Dict[str, Any], output_dir: Path):
        """CSV 리포트 생성"""
        output_file = output_dir / "evaluation_summary.csv"

        # 모든 메트릭 이름 수집
        all_metrics = set()
        for level_data in report['by_level'].values():
            all_metrics.update(level_data['metrics'].keys())

        # CSV 작성
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 헤더
            header = ['Level', 'Total_Tasks', 'Evaluated_Tasks', 'Avg_Exec_Time', 'Avg_Tokens', 'Avg_TPS', 'Avg_TTFT'] + sorted(all_metrics)
            writer.writerow(header)

            # 각 레벨 데이터
            for level in sorted(report['by_level'].keys()):
                level_data = report['by_level'][level]
                metadata = level_data.get('metadata', {})
                
                row = [
                    level,
                    level_data['total_tasks'],
                    level_data['evaluated_tasks'],
                    f"{metadata.get('average_execution_time', 0):.2f}",
                    f"{metadata.get('average_tokens_per_task', 0):.2f}",
                    f"{metadata.get('average_tps', 0):.2f}",
                    f"{metadata.get('ttft', {}).get('average', 0):.4f}",
                ]
                for metric in sorted(all_metrics):
                    score = level_data['metrics'].get(metric, 0.0)
                    row.append(f"{score:.4f}")
                writer.writerow(row)

        print(f"[저장] CSV: {output_file}")

    def export_markdown(self, report: Dict[str, Any], output_dir: Path):
        """Markdown 리포트 생성"""
        output_file = output_dir / "evaluation_report.md"

        with open(output_file, 'w', encoding='utf-8') as f:
            # 헤더
            f.write("# Ko-AgentBench 평가 보고서\n\n")

            # 실행 정보
            summary = report['summary']
            f.write("## 실행 정보\n\n")
            f.write(f"- **평가 대상 모델**: {summary['model']}\n")
            f.write(f"- **Judge 모델**: {summary['judge_model']}\n")
            f.write(f"- **실행 날짜**: {summary['execution_date']}\n")
            f.write(f"- **평가 날짜**: {summary['evaluation_date'][:10]}\n")
            f.write(f"- **총 태스크**: {summary['evaluated_tasks']}/{summary['total_tasks']}\n")
            if summary['sample_size']:
                f.write(f"- **샘플링**: 레벨당 {summary['sample_size']}개\n")
            f.write("\n")

            # 📊 성능 요약 테이블 추가
            f.write("## 📊 성능 요약\n\n")
            f.write("| Level | 태스크 수 | 평균 실행시간 | 평균 TPS | 평균 TTFT | 주요 지표 |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")

            # 각 레벨의 핵심 메트릭 매핑
            key_metrics = {
                'L1': ['ToolAcc', 'ArgAcc'],
                'L2': ['SelectAcc'],
                'L3': ['FSM', 'PSM'],
                'L4': ['Coverage'],
                'L5': ['AdaptiveRoutingScore', 'FallbackSR'],
                'L6': ['EffScore', 'ReuseRate'],
                'L7': ['ContextRetention']
            }

            for level in sorted(report['by_level'].keys()):
                level_data = report['by_level'][level]
                metadata = level_data.get('metadata', {})
                metrics = level_data['metrics']

                task_count = f"{level_data['evaluated_tasks']}/{level_data['total_tasks']}"
                
                exec_time = f"{metadata.get('average_execution_time', 0):.1f}초" if 'average_execution_time' in metadata else "N/A"
                tps = f"{metadata.get('average_tps', 0):.0f}" if 'average_tps' in metadata else "N/A"
                ttft = f"{metadata.get('ttft', {}).get('average', 0):.3f}초" if 'ttft' in metadata else "N/A"

                # 핵심 메트릭 표시
                key_metric_strs = []
                for km in key_metrics.get(level, []):
                    if km in metrics:
                        key_metric_strs.append(f"{km}: {metrics[km]:.3f}")
                key_metric_str = ", ".join(key_metric_strs) if key_metric_strs else "N/A"

                f.write(f"| **{level}** | {task_count} | {exec_time} | {tps} | {ttft} | {key_metric_str} |\n")
            f.write("\n")

            # 레벨별 상세 성능
            f.write("## 레벨별 성능\n\n")

            for level in sorted(report['by_level'].keys()):
                level_data = report['by_level'][level]
                level_num = level[1]

                level_names = {
                    '1': 'Level 1: 단일 도구 호출',
                    '2': 'Level 2: 도구 선택',
                    '3': 'Level 3: 멀티스텝 추론',
                    '4': 'Level 4: 멀티소스 통합',
                    '5': 'Level 5: 오류 처리',
                    '6': 'Level 6: 컨텍스트 재사용',
                    '7': 'Level 7: 멀티턴 대화',
                }

                f.write(f"### {level_names.get(level_num, level)}\n\n")

                # 태스크 의도 설명 추가
                if level_num in self.LEVEL_DESCRIPTIONS:
                    f.write(f"> 태스크 의도: {self.LEVEL_DESCRIPTIONS[level_num]}\n>\n")

                f.write(f"- 태스크 수: {level_data['evaluated_tasks']}/{level_data['total_tasks']}\n")

                metadata = level_data.get('metadata', {})
                if 'average_execution_time' in metadata:
                    f.write(f"- 평균 실행시간: {metadata['average_execution_time']:.2f}초\n")

                f.write("\n**메트릭 점수:**\n\n")

                metrics = level_data['metrics']
                if metrics:
                    # RRR과 SR을 먼저 출력
                    priority_metrics = ['RRR', 'SR']
                    for metric_name in priority_metrics:
                        if metric_name in metrics:
                            score = metrics[metric_name]
                            description = self.METRIC_DESCRIPTIONS.get(metric_name, "설명 없음")
                            f.write(f"- **{metric_name}**: {score:.3f} - {description}\n")
                    
                    # 나머지 메트릭 출력
                    for metric_name in sorted(metrics.keys()):
                        if metric_name not in priority_metrics:
                            score = metrics[metric_name]
                            description = self.METRIC_DESCRIPTIONS.get(metric_name, "설명 없음")
                            f.write(f"- **{metric_name}**: {score:.3f} - {description}\n")
                else:
                    f.write("- (메트릭 없음)\n")

                f.write("\n")

            # 토큰 사용량 및 성능 지표 섹션
            f.write("## 토큰 사용량 및 성능 지표\n\n")
            
            total_tokens = 0
            total_time = 0
            total_tasks = 0
            all_ttft_values = []
            
            for level_data in report['by_level'].values():
                metadata = level_data.get('metadata', {})
                tasks = level_data.get('evaluated_tasks', 0)
                total_tasks += tasks
                total_tokens += metadata.get('total_tokens', 0)
                total_time += metadata.get('total_execution_time', 0)
                
                ttft_avg = metadata.get('ttft', {}).get('average', 0)
                if ttft_avg > 0:
                    all_ttft_values.extend([ttft_avg] * tasks)
            
            # 전체 평균 계산
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
            
            for level in sorted(report['by_level'].keys()):
                level_data = report['by_level'][level]
                metadata = level_data.get('metadata', {})
                
                avg_tokens = metadata.get('average_tokens_per_task', 0)
                avg_prompt = metadata.get('average_prompt_tokens', 0)
                avg_completion = metadata.get('average_completion_tokens', 0)
                tps = metadata.get('average_tps', 0)
                
                ttft_data = metadata.get('ttft', {})
                ttft_avg = ttft_data.get('average', 0)
                ttft_min = ttft_data.get('min', 0)
                ttft_max = ttft_data.get('max', 0)
                ttft_range = f"{ttft_min:.4f} / {ttft_max:.4f}" if ttft_min > 0 else "N/A"
                
                f.write(f"| **{level}** | {avg_tokens:.0f} | {avg_prompt:.0f} | {avg_completion:.0f} | "
                    f"{tps:.0f} | {ttft_avg:.4f}초 | {ttft_range} |\n")
            
            f.write("\n")
            
        print(f"[저장] Markdown: {output_file}")

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
            formats = ['json', 'csv', 'markdown']

        if 'json' in formats:
            self.export_json(report, final_output_dir)

        if 'csv' in formats:
            self.export_csv(report, final_output_dir)

        if 'markdown' in formats:
            self.export_markdown(report, final_output_dir)

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

    parser = argparse.ArgumentParser(
        description='Ko-AgentBench 모델 실행 결과 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용 (모든 레벨, 단일 Judge )
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1
  
  # 특정 레벨만 평가
  python evaluate_model_run.py --date 20251021 --model anthropic/claude-sonnet-4-5 --levels L1
  python evaluate_model_run.py --date 20251021 --model anthropic/claude-sonnet-4-5 --levels L1,L3,L5
  
  # Judge 모델 지정
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1 --judge-models azure/gpt-4o gemini/gemini-2.5-pro-preview-03-25
  
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
                        help='평가할 레벨 지정 (예: L1, L1,L3,L5). 기본: 모든 레벨')
    parser.add_argument('--judge-models', nargs='+', default=None,
                        help='LLM Judge 모델(들). 기본: gpt-4o 단일 모델. 앙상블 원하면 여러 개 지정 (예: gpt-4o claude-sonnet-4-5 gemini-2.5)')
    parser.add_argument('--sample', type=int, default=None,
                        help='레벨당 평가할 태스크 수 (기본: 전체)')
    parser.add_argument('--quick', action='store_true',
                        help='빠른 테스트 (각 레벨당 1개)')
    parser.add_argument('--log-file', help='특정 로그 파일 경로 (선택사항)')

    # 출력 설정
    parser.add_argument('--output', default='reports',
                        help='출력 디렉토리 (기본: reports)')
    parser.add_argument('--format', nargs='+',
                        default=['json', 'csv', 'markdown'],
                        choices=['json', 'csv', 'markdown', 'all'],
                        help='출력 형식 (기본: json csv markdown)')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 로그 출력')

    args = parser.parse_args()

    # Judge 모델 설정
    if args.judge_models:
        judge_models = args.judge_models
    else:
        # 기본값: 단일 Judge 
        judge_models = ["azure/gpt-4o"]

    # 레벨 파싱
    levels = None
    if args.levels:
        # 쉼표로 구분된 레벨 파싱 (예: "L1,L3,L5" 또는 "1,3,5")
        levels = [lvl.strip() for lvl in args.levels.split(',')]
        print(f"[설정] 평가 대상 레벨: {', '.join(levels)}")

    # 평가 실행
    try:
        evaluator = ModelRunEvaluator(
            date=args.date,
            model=args.model,
            judge_models=judge_models,
            sample_size=1 if args.quick else args.sample,
            levels=levels,
            verbose=args.verbose
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
