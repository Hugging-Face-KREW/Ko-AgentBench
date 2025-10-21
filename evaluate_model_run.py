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


class ModelRunEvaluator:
    """모델 실행 결과 평가 클래스"""
    
    def __init__(
        self, 
        date: str, 
        model: str, 
        judge_models: List[str],
        sample_size: Optional[int] = None,
        verbose: bool = False
    ):
        self.date = date
        self.model = model
        self.judge_models = judge_models
        self.sample_size = sample_size
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
            'ArgAcc',           # L1
            'ErrorDetect',      # L5
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
            }
            
            # logs 구성
            logs = {
                "success": task_result.get("success", False),
                "tool_invocations": task_result.get("tool_calls", []),
                "tool_calls": task_result.get("tool_calls", []),  # ErrorDetectMetric용
                "actual_output": task_result.get("final_response", ""),
                "final_response": task_result.get("final_response", ""),
                "conversation_log": task_result.get("conversation_log", {}),
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
        files = self.find_result_files()
        
        if not files:
            raise FileNotFoundError(
                f"결과 파일을 찾을 수 없습니다.\n"
                f"  날짜: {self.date}\n"
                f"  모델: {self.model}\n"
                f"  디렉토리: logs/benchmark_results/"
            )
        
        # 헤더 출력
        print("\n" + "="*80)
        print("Ko-AgentBench 평가")
        print("="*80)
        print(f"평가 대상: {self.model} (실행 날짜: {self.date})")
        print(f"Judge 모델: {self.judge_models}")
        print(f"발견된 레벨: {', '.join(sorted(files.keys()))}")
        if self.sample_size:
            print(f"샘플링: 레벨당 {self.sample_size}개 태스크")
        print("="*80 + "\n")
        
        # 각 레벨 평가
        results = {}
        for level in sorted(files.keys()):
            print(f"\n[{level}] 평가 시작")
            print("-" * 80)
            result = self.evaluate_level(files[level], level)
            results[level] = result
            
            # 간단한 요약 출력
            print(f"  평가 완료: {result['evaluated_tasks']}/{result['total_tasks']} 태스크")
            if result['metric_averages']:
                print(f"  주요 메트릭:")
                for metric_name, score in list(result['metric_averages'].items())[:5]:
                    print(f"    {metric_name}: {score:.3f}")
        
        # 종합 보고서 생성
        return self.generate_report(results)
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 보고서 생성"""
        report = {
            "summary": {
                "model": self.model,
                "judge_model": self.judge_models,
                "execution_date": self.date,
                "evaluation_date": datetime.now().isoformat(),
                "total_levels": len(results),
                "sample_size": self.sample_size,
            },
            "by_level": {}
        }
        
        total_tasks = 0
        total_evaluated = 0
        
        for level, level_result in results.items():
            total_tasks += level_result['total_tasks']
            total_evaluated += level_result['evaluated_tasks']
            
            report["by_level"][level] = {
                "metadata": level_result['metadata'],
                "total_tasks": level_result['total_tasks'],
                "evaluated_tasks": level_result['evaluated_tasks'],
                "metrics": level_result['metric_averages'],
                "task_details": level_result['task_evaluations']
            }
        
        report["summary"]["total_tasks"] = total_tasks
        report["summary"]["evaluated_tasks"] = total_evaluated
        
        return report
    
    def export_json(self, report: Dict[str, Any], output_dir: Path):
        """JSON 형식으로 저장"""
        output_file = output_dir / "evaluation_report.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[저장] JSON: {output_file}")
    
    def export_csv(self, report: Dict[str, Any], output_dir: Path):
        """CSV 형식으로 저장 (메트릭 요약)"""
        output_file = output_dir / "metrics_by_level.csv"
        
        # 모든 메트릭 이름 수집
        all_metrics = set()
        for level_data in report['by_level'].values():
            all_metrics.update(level_data['metrics'].keys())
        
        # CSV 작성
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 헤더
            header = ['Level', 'Total_Tasks', 'Evaluated_Tasks'] + sorted(all_metrics)
            writer.writerow(header)
            
            # 각 레벨 데이터
            for level in sorted(report['by_level'].keys()):
                level_data = report['by_level'][level]
                row = [
                    level,
                    level_data['total_tasks'],
                    level_data['evaluated_tasks']
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
            
            # 요약
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
            
            # 레벨별 상세
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
                f.write(f"- 태스크 수: {level_data['evaluated_tasks']}/{level_data['total_tasks']}\n")
                
                metadata = level_data['metadata']
                if 'success_rate' in metadata:
                    f.write(f"- 성공률: {metadata['success_rate']:.1f}%\n")
                if 'average_execution_time' in metadata:
                    f.write(f"- 평균 실행시간: {metadata['average_execution_time']:.2f}초\n")
                # 토큰 통계 추가
                if 'average_tokens_per_task' in metadata:
                    f.write(f"- 평균 토큰 수: {metadata['average_tokens_per_task']:.2f}\n")
                if 'average_tps' in metadata:
                    f.write(f"- 평균 TPS: {metadata['average_tps']:.2f} tokens/sec\n")
                if 'average_prompt_tokens' in metadata:
                    f.write(f"  - 평균 입력 토큰: {metadata['average_prompt_tokens']:.2f}\n")
                if 'average_completion_tokens' in metadata:
                    f.write(f"  - 평균 출력 토큰: {metadata['average_completion_tokens']:.2f}\n")
                
                f.write("\n**메트릭 점수:**\n\n")
                
                metrics = level_data['metrics']
                if metrics:
                    for metric_name in sorted(metrics.keys()):
                        score = metrics[metric_name]
                        f.write(f"- {metric_name}: {score:.3f}\n")
                else:
                    f.write("- (메트릭 없음)\n")
                
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
    
    # Azure API 키는 환경변수에서 로드됨
    
    parser = argparse.ArgumentParser(
        description='Ko-AgentBench 모델 실행 결과 평가',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 사용
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1
  
  # Judge 모델 지정
  python evaluate_model_run.py --date 20251016 --model azure/gpt-4.1 --judge-model azure/gpt-4o
  
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
    parser.add_argument('--judge-models', nargs='+', default=None,
                    help='LLM Judge 모델들 (예: gpt-4o, claude-sonnet-4-5, gemini-2.5)')
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
        # 기본값: 3개 Judge (변경 가능!)
        judge_models = [
            "azure/gpt-4o",
            "anthropic/claude-sonnet-4-5-20250929",
            "gemini/gemini-2.5-pro-preview-03-25"
        ]
    
    # 평가 실행
    try:
        evaluator = ModelRunEvaluator(
            date=args.date,
            model=args.model,
            judge_models=judge_models,
            sample_size=1 if args.quick else args.sample,
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

