"""
레벨별 벤치마크 결과 평가 스크립트
로그 파일을 읽어와서 레벨별 계산하여 결과 파일을 생성(L6, L7)
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationResult:
    """평가 결과"""
    metric_name: str
    score: float
    details: Dict[str, Any]


class LevelEvaluator:
    """레벨별 벤치마크 결과 평가기 (L6, L7)"""
    
    def __init__(self, log_file_path: str):
        """
        Args:
            log_file_path: 평가할 로그 파일 경로
        """
        self.log_file_path = Path(log_file_path)
        self.log_data = self._load_log()
        
    def _load_log(self) -> Dict[str, Any]:
        """로그 파일 로드"""
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_SR(self, result: Dict[str, Any]) -> EvaluationResult:
        """SR(성공률): success 필드 확인"""
        success = result.get("success", False)
        score = 1.0 if success else 0.0
        
        return EvaluationResult(
            metric_name="SR",
            score=score,
            details={"success": success}
        )
    
    def evaluate_EPR_CVR(self, result: Dict[str, Any]) -> EvaluationResult:
        """EPR/CVR(유효 호출 비율): success=True AND error=None 비율"""
        tool_calls = result.get("tool_calls", [])
        
        if not tool_calls:
            return EvaluationResult(
                metric_name="EPR_CVR",
                score=0.0,
                details={"total_calls": 0, "valid_calls": 0}
            )
        
        valid_calls = sum(
            1 for call in tool_calls 
            if call.get("success") and not call.get("error")
        )
        total_calls = len(tool_calls)
        score = valid_calls / total_calls if total_calls > 0 else 0.0
        
        return EvaluationResult(
            metric_name="EPR_CVR",
            score=score,
            details={
                "total_calls": total_calls,
                "valid_calls": valid_calls
            }
        )
    
    def evaluate_pass_at_k(self, result: Dict[str, Any]) -> EvaluationResult:
        """pass@k(반복 안정성): k번 반복 시 성공 비율"""
        # 일단은 로그에 repetition_results 필드가 없으므로 현재 success만 사용 (추후 수정 필요함!)
        success = result.get("success", False)
        
        return EvaluationResult(
            metric_name="pass@k",
            score=1.0 if success else 0.0,
            details={
                "repetitions": 1,
                "actual_repetitions": 1,
                "success_count": 1 if success else 0
            }
        )
    
    def evaluate_ReuseRate(self, result: Dict[str, Any]) -> EvaluationResult:
        """ReuseRate: 재사용 기회 대비 실제 재사용 비율"""
        golden_action = result.get("golden_action", [])
        tool_calls = result.get("tool_calls", [])
        
        # 재사용 기회: golden_action에서 "reuse" 도구의 개수
        reuse_opportunities = sum(
            1 for action in golden_action 
            if action.get("tool") == "reuse"
        )
        
        if reuse_opportunities == 0:
            return EvaluationResult(
                metric_name="ReuseRate",
                score=0.0,
                details={"reuse_opportunities": 0, "reused": 0, "reason": "No reuse opportunities"}
            )
        
        # 실제 재사용 판단: 동일한 도구를 동일한 파라미터로 재호출하지 않은 경우
        # 유니크한 (tool_name, args) 조합 개수와 전체 호출 개수 비교
        tool_call_signatures = []
        for call in tool_calls:
            if call.get("success"):
                signature = (
                    call.get("tool_name"),
                    json.dumps(call.get("arguments", {}), sort_keys=True)
                )
                tool_call_signatures.append(signature)
        
        total_successful_calls = len(tool_call_signatures)
        unique_calls = len(set(tool_call_signatures))
        
        # 재사용된 횟수 = 전체 호출 - 유니크 호출
        golden_unique_tools = set()
        for action in golden_action:
            tool = action.get("tool")
            if tool != "reuse":
                golden_unique_tools.add(tool)
        
        expected_unique_calls = len(golden_unique_tools)
        
        # 실제 재사용 비율 계산
        if expected_unique_calls == 0:
            reused = 0
        else:
            # 재사용했다면 unique_calls == expected_unique_calls
            # 재사용 안했다면 unique_calls > expected_unique_calls
            if unique_calls <= expected_unique_calls:
                reused = reuse_opportunities
            else:
                # 부분적으로 재사용
                excess_calls = unique_calls - expected_unique_calls
                reused = max(0, reuse_opportunities - excess_calls)
        
        score = reused / reuse_opportunities if reuse_opportunities > 0 else 0.0
        
        return EvaluationResult(
            metric_name="ReuseRate",
            score=score,
            details={
                "reuse_opportunities": reuse_opportunities,
                "reused": reused,
                "unique_calls": unique_calls,
                "expected_unique_calls": expected_unique_calls
            }
        )
    
    def evaluate_RedundantCallRate(self, result: Dict[str, Any]) -> EvaluationResult:
        """RedundantCallRate: 재사용이 정답인 상황에서 불필요한 API 호출 비율"""
        golden_action = result.get("golden_action", [])
        tool_calls = result.get("tool_calls", [])
        
        # 재사용 기회: golden_action에서 "reuse" 도구의 개수
        reuse_opportunities = sum(
            1 for action in golden_action 
            if action.get("tool") == "reuse"
        )
        
        if reuse_opportunities == 0:
            return EvaluationResult(
                metric_name="RedundantCallRate",
                score=0.0,
                details={"reuse_opportunities": 0, "redundant_calls": 0, "reason": "No reuse opportunities"}
            )
        
        # 불필요한 호출: 동일한 도구를 동일한 파라미터로 재호출한 경우
        tool_call_signatures = []
        redundant_calls = 0
        
        for call in tool_calls:
            if call.get("success"):
                signature = (
                    call.get("tool_name"),
                    json.dumps(call.get("arguments", {}), sort_keys=True)
                )
                if signature in tool_call_signatures:
                    redundant_calls += 1
                else:
                    tool_call_signatures.append(signature)
        
        score = redundant_calls / reuse_opportunities if reuse_opportunities > 0 else 0.0
        
        return EvaluationResult(
            metric_name="RedundantCallRate",
            score=score,
            details={
                "reuse_opportunities": reuse_opportunities,
                "redundant_calls": redundant_calls
            }
        )
    
    def evaluate_EffScore(self, result: Dict[str, Any]) -> EvaluationResult:
        """EffScore: 성공 시 최소 호출 수 대비 실제 호출 수의 효율"""
        success = result.get("success", False)
        actual_calls = len(result.get("tool_calls", []))
        
        # minimum_calls는 태스크 스키마에 있거나, golden_action에서 계산
        minimum_calls = result.get("minimum_calls")
        
        if minimum_calls is None:
            # golden_action에서 "reuse"를 제외한 유니크 도구 수로 계산
            golden_action = result.get("golden_action", [])
            unique_tools = set()
            for action in golden_action:
                tool = action.get("tool")
                if tool != "reuse":
                    unique_tools.add(tool)
            minimum_calls = len(unique_tools) if unique_tools else 1
        
        if not success or actual_calls <= 0:
            return EvaluationResult(
                metric_name="EffScore",
                score=0.0,
                details={
                    "success": success,
                    "actual_calls": actual_calls,
                    "minimum_calls": minimum_calls
                }
            )
        
        score = min(1.0, minimum_calls / actual_calls)
        
        return EvaluationResult(
            metric_name="EffScore",
            score=score,
            details={
                "success": success,
                "actual_calls": actual_calls,
                "minimum_calls": minimum_calls
            }
        )
    
    def evaluate_ContextRetention(self, result: Dict[str, Any]) -> EvaluationResult:
        """ContextRetention: 필요한 순간에 과거 맥락을 사용했는지의 비율"""
        context_tests = result.get("context_tests", [])
        
        if not context_tests:
            return EvaluationResult(
                metric_name="ContextRetention",
                score=0.0,
                details={"total": 0, "used": 0, "reason": "No context tests"}
            )
        
        used = sum(1 for t in context_tests if t.get("used") is True)
        score = used / len(context_tests)
        
        return EvaluationResult(
            metric_name="ContextRetention",
            score=score,
            details={"used": used, "total": len(context_tests)}
        )

    def evaluate_RefRecall(self, result: Dict[str, Any]) -> EvaluationResult:
        """RefRecall: 오래된 정보 회상 비율"""
        long_term_tests = result.get("long_term_tests", [])
        
        if not long_term_tests:
            return EvaluationResult(
                metric_name="RefRecall",
                score=0.0,
                details={"total": 0, "recalled": 0, "reason": "No long-term tests"}
            )
        
        recalled = sum(1 for t in long_term_tests if t.get("recalled") is True)
        score = recalled / len(long_term_tests)
        
        return EvaluationResult(
            metric_name="RefRecall",
            score=score,
            details={"recalled": recalled, "total": len(long_term_tests)}
        )
    
    def evaluate_all_tasks(self) -> Dict[str, Any]:
        """모든 태스크 평가"""
        results = self.log_data.get("results", [])
        
        # 레벨 감지
        level = self.log_data.get("metadata", {}).get("level", "L6")
        
        all_evaluations = []
        
        # 레벨별 지표 정의
        if level == "L7":
            metric_names = ["SR", "EPR_CVR", "pass@k", "ContextRetention", "RefRecall"]
        else:  # L6 
            metric_names = ["SR", "EPR_CVR", "pass@k", "ReuseRate", "RedundantCallRate", "EffScore"]
        
        metric_scores = {name: [] for name in metric_names}
        
        for result in results:
            task_id = result.get("task_id")
            
            # 공통 지표 평가
            sr = self.evaluate_SR(result)
            epr_cvr = self.evaluate_EPR_CVR(result)
            pass_at_k = self.evaluate_pass_at_k(result)
            
            # 레벨별 지표 평가
            if level == "L7":
                context_retention = self.evaluate_ContextRetention(result)
                ref_recall = self.evaluate_RefRecall(result)
                
                metric_scores["SR"].append(sr.score)
                metric_scores["EPR_CVR"].append(epr_cvr.score)
                metric_scores["pass@k"].append(pass_at_k.score)
                metric_scores["ContextRetention"].append(context_retention.score)
                metric_scores["RefRecall"].append(ref_recall.score)
                
                task_evaluation = {
                    "task_id": task_id,
                    "instruction": result.get("instruction"),
                    "level": result.get("level"),
                    "metrics": {
                        "SR": {"score": sr.score, "details": sr.details},
                        "EPR_CVR": {"score": epr_cvr.score, "details": epr_cvr.details},
                        "pass@k": {"score": pass_at_k.score, "details": pass_at_k.details},
                        "ContextRetention": {"score": context_retention.score, "details": context_retention.details},
                        "RefRecall": {"score": ref_recall.score, "details": ref_recall.details}
                    }
                }
            else:  # L6
                reuse_rate = self.evaluate_ReuseRate(result)
                redundant_rate = self.evaluate_RedundantCallRate(result)
                eff_score = self.evaluate_EffScore(result)
                
                metric_scores["SR"].append(sr.score)
                metric_scores["EPR_CVR"].append(epr_cvr.score)
                metric_scores["pass@k"].append(pass_at_k.score)
                metric_scores["ReuseRate"].append(reuse_rate.score)
                metric_scores["RedundantCallRate"].append(redundant_rate.score)
                metric_scores["EffScore"].append(eff_score.score)
                
                task_evaluation = {
                    "task_id": task_id,
                    "instruction": result.get("instruction"),
                    "level": result.get("level"),
                    "metrics": {
                        "SR": {"score": sr.score, "details": sr.details},
                        "EPR_CVR": {"score": epr_cvr.score, "details": epr_cvr.details},
                        "pass@k": {"score": pass_at_k.score, "details": pass_at_k.details},
                        "ReuseRate": {"score": reuse_rate.score, "details": reuse_rate.details},
                        "RedundantCallRate": {"score": redundant_rate.score, "details": redundant_rate.details},
                        "EffScore": {"score": eff_score.score, "details": eff_score.details}
                    }
                }
            
            all_evaluations.append(task_evaluation)
        
        # 평균 점수 계산
        avg_metrics = {}
        for metric_name, scores in metric_scores.items():
            if scores:
                avg_metrics[metric_name] = {
                    "average": round(sum(scores) / len(scores), 4),
                    "count": len(scores),
                    "scores": scores
                }
            else:
                avg_metrics[metric_name] = {
                    "average": 0.0,
                    "count": 0,
                    "scores": []
                }
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "level": level,  # 레벨 정보 추가
                "log_file": str(self.log_file_path),
                "original_metadata": self.log_data.get("metadata", {}),
                "total_tasks": len(results)
            },
            "average_metrics": avg_metrics,
            "task_evaluations": all_evaluations
        }
    
    def save_results(self, output_path: str = None) -> str:
        """평가 결과 저장"""
        evaluation_results = self.evaluate_all_tasks()
        
        if output_path is None:
            log_dir = self.log_file_path.parent
            log_filename = self.log_file_path.stem
            output_path = log_dir / f"{log_filename}_evaluation.json"
        else:
            output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        return str(output_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="레벨별 벤치마크 결과 평가")
    parser.add_argument(
        "log_file",
        type=str,
        help="평가할 로그 파일 경로"
    )
    args = parser.parse_args()
    
    evaluator = LevelEvaluator(args.log_file)  
    output_path = evaluator.save_results(args.output)
    
    evaluation_results = evaluator.evaluate_all_tasks()
    level = evaluation_results['metadata']['level']
    
    print(f"{level} 레벨 평가 완료")
    print(f"\n로그 파일: {args.log_file}")
    print(f"결과 파일: {output_path}")
    print(f"\n총 태스크 수: {evaluation_results['metadata']['total_tasks']}")

if __name__ == "__main__":
    main()