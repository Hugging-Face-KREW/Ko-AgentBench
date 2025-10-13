"""
레벨별 벤치마크 결과 평가 스크립트
로그 파일을 읽어와서 레벨별 계산하여 결과 파일을 생성(L6, L7)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    
    def __init__(self, log_file_path: str, llm_model: str = "anthropic/claude-3-7-sonnet-latest"):
        """
        Args:
            log_file_path: 평가할 로그 파일 경로
            llm_model: LLM-as-a-Judge에 사용할 모델 
        """
        self.log_file_path = Path(log_file_path)
        self.log_data = self._load_log()
        self.llm_model = llm_model
        self.llm_client = None
        
        level = self.log_data.get("metadata", {}).get("level", "L7")
        if level == "L7":
            self._init_llm_client()
    
    def _init_llm_client(self):
        """LLM 클라이언트 초기화"""
        from bench.adapters.litellm_adapter import LiteLLMAdapter
        self.llm_client = LiteLLMAdapter(self.llm_model)
        print(f"LLM Judge 초기화 완료: {self.llm_model}")
    
    def _load_log(self) -> Dict[str, Any]:
        """로그 파일 로드"""
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """메시지 목록을 읽기 쉬운 형식으로 변환"""
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"[턴 {i}] 사용자: {content}")
            elif role == "assistant":
                formatted.append(f"[턴 {i}] AI: {content[:200]}{'...' if len(content) > 200 else ''}")
        return "\n".join(formatted)
    
    def _call_llm_judge(self, prompt: str) -> Dict[str, Any]:
        """LLM Judge 호출"""
        if not self.llm_client:
            return {"score": 0.0, "reason": "LLM client not available"}
        
        try:
            messages = [
                {"role": "system", "content": "You are an expert evaluator for AI conversation quality. Always respond in valid JSON format."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm_client.chat_completion(messages, temperature=0.0)
            content = response.get("message", {}).get("content", "{}")
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError:
                import re
                score_match = re.search(r'"score"\s*:\s*([0-9.]+)', content)
                if score_match:
                    return {"score": float(score_match.group(1)), "reason": "Parsed from text"}
                return {"score": 0.0, "reason": "Failed to parse LLM response"}
                
        except Exception as e:
            print(f"LLM Judge 호출 실패: {e}")
            return {"score": 0.0, "reason": f"Error: {str(e)}"}
    
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
    
    # L6 지표
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
        
        minimum_calls = result.get("minimum_calls")
        
        if minimum_calls is None:
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
    
    # L7 지표
    def evaluate_ContextRetention(self, result: Dict[str, Any]) -> EvaluationResult:
        """ContextRetention: LLM-as-a-Judge로 과거 맥락 활용도 평가"""
        conv_preview = result.get("conversation_preview", {})
        
        if not conv_preview:
            return EvaluationResult(
                metric_name="ContextRetention",
                score=0.0,
                details={"reason": "No conversation preview available"}
            )
        
        first_messages = conv_preview.get("first_messages", [])
        last_messages = conv_preview.get("last_messages", [])
        
        if len(first_messages) < 2 or len(last_messages) < 2:
            return EvaluationResult(
                metric_name="ContextRetention",
                score=0.0,
                details={"reason": "Not enough messages for context evaluation"}
            )
        
        # LLM Judge 프롬프트 구성
        prompt = f"""다음 대화에서 AI가 이전 맥락을 적절히 유지하고 활용했는지 평가해주세요.

        대화 내용:
        {self._format_messages(first_messages + last_messages)}

        평가 기준:
        1. AI가 이전 대화의 핵심 정보를 기억하고 있는가?
        2. 사용자가 과거 맥락을 참조할 때("그때", "아까", "전에 말한") 적절히 연결했는가?
        3. 새로운 질문에 이전 정보를 고려하여 답변했는가?
        4. 불필요한 재질문 없이 맥락을 유지했는가?

        점수 기준:
        - 1.0: 모든 맥락을 완벽히 유지하고 적절히 활용
        - 0.7-0.9: 대부분의 맥락을 유지하고 활용
        - 0.4-0.6: 일부 맥락만 유지하거나 부분적으로만 활용
        - 0.1-0.3: 맥락 유지가 미흡
        - 0.0: 맥락을 전혀 유지하지 못함

        JSON 형식으로만 답변해주세요:
        {{"score": 0.0-1.0, "retained": true/false, "reason": "간단한 이유"}}"""

        llm_result = self._call_llm_judge(prompt)
        score = float(llm_result.get("score", 0.0))
        
        return EvaluationResult(
            metric_name="ContextRetention",
            score=score,
            details={
                "retained": llm_result.get("retained", score > 0.5),
                "reason": llm_result.get("reason", "LLM evaluation"),
                "total_messages": len(first_messages) + len(last_messages)
            }
        )
    
    def evaluate_RefRecall(self, result: Dict[str, Any]) -> EvaluationResult:
        """RefRecall: LLM-as-a-Judge로 오래된 정보 회상 능력 평가"""
        conv_preview = result.get("conversation_preview", {})
        
        if not conv_preview:
            return EvaluationResult(
                metric_name="RefRecall",
                score=0.0,
                details={"reason": "No conversation preview available"}
            )
        
        first_messages = conv_preview.get("first_messages", [])
        last_messages = conv_preview.get("last_messages", [])
        total_messages = conv_preview.get("total_messages", len(first_messages) + len(last_messages))
        
        if len(first_messages) < 2 or len(last_messages) < 2:
            return EvaluationResult(
                metric_name="RefRecall",
                score=0.0,
                details={"reason": "Not enough messages for recall evaluation"}
            )
        
        # LLM Judge 프롬프트 구성
        prompt = f"""다음 대화에서 AI가 오래된 정보를 정확히 회상하고 활용했는지 평가해주세요.

        대화 내용 (총 {total_messages}개 메시지 중 일부):
        {self._format_messages(first_messages + last_messages)}

        평가 기준:
        1. 초반 대화의 구체적 정보(숫자, 이름, 특징 등)를 나중에도 정확히 기억하는가?
        2. 시간이 지난 후에도 이전 정보를 정확하게 참조하는가?
        3. 여러 턴이 지난 후에도 맥락의 연속성을 유지하는가?
        4. 혼동 없이 정확한 정보를 회상하는가?

        점수 기준:
        - 1.0: 모든 과거 정보를 정확히 회상
        - 0.7-0.9: 대부분의 정보를 정확히 회상
        - 0.4-0.6: 일부 정보만 회상하거나 부분적으로 정확
        - 0.1-0.3: 회상이 부정확하거나 미흡
        - 0.0: 과거 정보를 전혀 회상하지 못함

        JSON 형식으로만 답변해주세요:
        {{"score": 0.0-1.0, "recalled": true/false, "reason": "간단한 이유"}}"""

        llm_result = self._call_llm_judge(prompt)
        score = float(llm_result.get("score", 0.0))
        
        return EvaluationResult(
            metric_name="RefRecall",
            score=score,
            details={
                "recalled": llm_result.get("recalled", score > 0.5),
                "reason": llm_result.get("reason", "LLM evaluation"),
                "total_messages": total_messages,
                "evaluated_messages": len(first_messages) + len(last_messages)
            }
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
        
        for idx, result in enumerate(results, 1):
            task_id = result.get("task_id")
            print(f"\n[{idx}/{len(results)}] 평가 중: {task_id}")
            
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
                "level": level,
                "log_file": str(self.log_file_path),
                "llm_model": self.llm_model if level == "L7" else None,
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
    parser.add_argument(
        "-o", "--output",  
        type=str,
        default=None,
        help="결과 파일 저장 경로"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="anthropic/claude-3-7-sonnet-latest",
        help="LLM Judge 모델"
    )
    args = parser.parse_args()
    
    evaluator = LevelEvaluator(args.log_file, llm_model=args.llm_model)
    output_path = evaluator.save_results(args.output)
    
    evaluation_results = evaluator.evaluate_all_tasks()
    level = evaluation_results['metadata']['level']
    
    print(f"{level} 레벨 평가 완료")
    print(f"\n로그 파일: {args.log_file}")
    print(f"결과 파일: {output_path}")
    if level == "L7":
        print(f"LLM Judge 모델: {args.llm_model}")
    print(f"\n총 태스크 수: {evaluation_results['metadata']['total_tasks']}")

if __name__ == "__main__":
    main()