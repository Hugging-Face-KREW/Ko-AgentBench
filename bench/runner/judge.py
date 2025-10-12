"""Judge for evaluating benchmark results."""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from ..adapters.base_adapter import BaseAdapter
from ..observability import observe, get_client, is_enabled

@dataclass
class EvaluationResult:
    """메트릭 평가 결과"""
    metric_name: str
    score: float
    details: Dict[str, Any]

@dataclass
class EvalContext:
    """평가 컨텍스트"""
    task_schema: Dict[str, Any]
    logs: Dict[str, Any]
    action_trace: List[Dict[str, Any]] = None 
    
    def __post_init__(self):
        """action_trace가 없으면 tool_invocations에서 자동 생성"""
        if self.action_trace is None:
            self.action_trace = []
            tool_invocations = self.logs.get("tool_invocations", [])
            for inv in tool_invocations:
                self.action_trace.append({
                    "step": inv.get("step"),
                    "tool": inv.get("tool_name"),
                    "args": inv.get("arguments", {}),
                    "result": inv.get("result"),
                    "success": inv.get("success")
                })

class Metric:
    """메트릭 기본 클래스"""
    name = "base_metric"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        raise NotImplementedError

class SRMetric(Metric):
    """SR(성공률): success_condition 충족 여부"""
    name = "SR"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        # task에서 성공 여부 직접 확인
        success = ctx.logs.get("success", False)
        score = 1.0 if success else 0.0
        
        return EvaluationResult(
            self.name, 
            score, 
            {"success": success}
        )

class EPRCVRMetric(Metric):
    """EPR/CVR(유효 호출 비율): accept + schema_valid 비율"""
    name = "EPR_CVR"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        call_logs = ctx.logs.get("tool_invocations", [])
        if not call_logs:
            return EvaluationResult(self.name, 0.0, {
                "total_calls": 0, 
                "valid_calls": 0
            })
        
        # success=True이고 error=None인 호출을 유효한 것으로 간주
        valid_calls = sum(1 for log in call_logs 
                         if log.get("success") and not log.get("error"))
        total_calls = len(call_logs)
        score = valid_calls / total_calls if total_calls > 0 else 0.0
        
        return EvaluationResult(self.name, score, {
            "total_calls": total_calls,
            "valid_calls": valid_calls
        })

class PassAtKMetric(Metric):
    """pass@k(반복 안정성): k번 반복 시 성공 비율"""
    name = "pass@k"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        repetitions = ctx.task_schema.get("repetitions", 1)
        repetition_results = ctx.logs.get("repetition_results", [])
        
        # 반복 실행이 없으면 현재 성공 여부만 사용
        if not repetition_results:
            current_success = ctx.logs.get("success", False)
            repetition_results = [current_success]
        
        success_count = sum(1 for result in repetition_results if result)
        actual_reps = len(repetition_results)
        score = success_count / actual_reps if actual_reps > 0 else 0.0
        
        return EvaluationResult(self.name, score, {
            "repetitions": repetitions,
            "actual_repetitions": actual_reps,
            "success_count": success_count
        })

class FSMMetric(Metric):
    """FSM(Full Sequence Match): 정답 경로와 완전 일치"""
    name = "FSM"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        golden_action = ctx.task_schema.get("golden_action", [])
        action_trace = ctx.action_trace
        
        # 단계 수가 다르면 불일치
        if len(golden_action) != len(action_trace):
            return EvaluationResult(self.name, 0.0, {
                "golden_steps": len(golden_action),
                "actual_steps": len(action_trace),
                "reason": "단계 수 불일치"
            })
        
        # 각 단계의 tool이 모두 일치하는지 확인
        match = all(
            golden.get("tool") == action.get("tool")
            for golden, action in zip(golden_action, action_trace)
        )
        
        golden_sequence = [g.get("tool") for g in golden_action]
        actual_sequence = [a.get("tool") for a in action_trace]
        
        return EvaluationResult(self.name, 1.0 if match else 0.0, {
            "golden_sequence": golden_sequence,
            "actual_sequence": actual_sequence,
            "match": match
        })


class PSMMetric(Metric):
    """PSM(Partial Sequence Match): 일치한 단계 비율"""
    name = "PSM"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        golden_action = ctx.task_schema.get("golden_action", [])
        action_trace = ctx.action_trace
        
        if not golden_action:
            return EvaluationResult(self.name, 0.0, {"reason": "golden_action 없음"})
        
        # 실제 사용된 도구 목록
        actual_tools = [action.get("tool") for action in action_trace]
        
        matched_weight = 0.0
        total_weight = 0.0
        matched_tools = []
        
        for golden_step in golden_action:
            tool = golden_step.get("tool")
            weight = golden_step.get("weight", 1.0)  # 기본 가중치 1.0
            total_weight += weight
            
            # 정답 도구가 실제 실행에 포함되어 있는지 확인
            if tool in actual_tools:
                matched_weight += weight
                matched_tools.append(tool)
        
        score = matched_weight / total_weight if total_weight > 0 else 0.0
        
        return EvaluationResult(self.name, score, {
            "matched_weight": matched_weight,
            "total_weight": total_weight,
            "matched_tools": matched_tools,
            "missing_tools": [g.get("tool") for g in golden_action if g.get("tool") not in actual_tools]
        })


class DeltaStepsNormMetric(Metric):
    """ΔSteps_norm: 최소 경로 대비 효율"""
    name = "ΔSteps_norm"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        minimum_steps = ctx.task_schema.get("minimum_steps", 1)
        actual_steps = len(ctx.action_trace)
        
        if minimum_steps <= 0:
            return EvaluationResult(self.name, 0.0, {"reason": "minimum_steps가 0 이하"})
        
        # delta_norm: 초과 비율 (0이면 최적, 클수록 비효율적)
        delta_norm = (actual_steps - minimum_steps) / minimum_steps
        
        # efficiency: 효율성 점수 (1.0이 최적, 0에 가까울수록 비효율적)
        # delta_norm이 음수면 불가능한 상황이므로 0점
        efficiency = max(0.0, 1.0 - delta_norm) if delta_norm >= 0 else 0.0
        
        return EvaluationResult(self.name, efficiency, {
            "minimum_steps": minimum_steps,
            "actual_steps": actual_steps,
            "delta_norm": round(delta_norm, 4),
            "extra_steps": actual_steps - minimum_steps
        })


class ProvAccMetric(Metric):
    """ProvAcc: 다음 호출 인자의 출처 추적 정확도"""
    name = "ProvAcc"
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        data_flow = ctx.task_schema.get("data_flow", [])
        action_trace = ctx.action_trace
        
        # data_flow가 없으면 측정 불가 (만점 처리)
        if not data_flow:
            return EvaluationResult(self.name, 1.0, {
                "reason": "data_flow 정보 없음",
                "no_data_flow": True
            })
        
        valid_flows = 0
        total_flows = len(data_flow)
        flow_details = []
        
        for flow in data_flow:
            to_step = flow.get("to_step")
            to_param = flow.get("to_parameter")
            from_step = flow.get("from_step")
            from_output = flow.get("from_output")
            
            # to_step에 해당하는 실제 액션 찾기
            target_action = None
            if isinstance(to_step, int) and to_step > 0:
                # step 번호로 찾기
                for action in action_trace:
                    if action.get("step") == to_step:
                        target_action = action
                        break
            
            is_valid = False
            detail = {
                "to_step": to_step,
                "to_parameter": to_param,
                "from_step": from_step,
                "expected_source": from_output
            }
            
            if target_action:
                # 실제 사용된 인자값 확인
                actual_args = target_action.get("args", {})
                actual_value = actual_args.get(to_param)
                
                detail["actual_value"] = actual_value
                
                # from_step이 "user_input"인 경우
                if from_step == "user_input":
                    # 사용자 입력이 포함되어 있으면 유효
                    is_valid = actual_value is not None
                
                # from_step이 이전 단계인 경우
                elif isinstance(from_step, int) and from_step > 0:
                    # 이전 단계의 결과에서 값을 가져왔는지 확인
                    source_action = None
                    for action in action_trace:
                        if action.get("step") == from_step:
                            source_action = action
                            break
                    
                    # 실제로 값이 일치하는지 확인
                    if source_action and source_action.get("success"):
                        source_result = source_action.get("result", {})
                        expected_value = source_result.get(from_output)
                        
                        # 실제 값과 기대 값 비교
                        is_valid = (actual_value is not None and 
                                    expected_value is not None and
                                    str(actual_value) == str(expected_value))
                
                detail["is_valid"] = is_valid
            else:
                detail["is_valid"] = False
                detail["reason"] = f"step {to_step}을 찾을 수 없음"
            
            if is_valid:
                valid_flows += 1
            
            flow_details.append(detail)
        
        score = valid_flows / total_flows if total_flows > 0 else 0.0
        
        return EvaluationResult(self.name, score, {
            "valid_flows": valid_flows,
            "total_flows": total_flows,
            "flow_details": flow_details
        })
    
# 메트릭 레지스트리
METRICS = {
    "SR": SRMetric(),
    "EPR_CVR": EPRCVRMetric(),
    "pass@k": PassAtKMetric(),
    
    "FSM": FSMMetric(),
    "PSM": PSMMetric(),
    "ΔSteps_norm": DeltaStepsNormMetric(),
    "ProvAcc": ProvAccMetric()
}

class Judge:
    """Judge for evaluating task results with multiple evaluation methods."""
    
    def __init__(self, llm_adapter: Optional[BaseAdapter] = None):
        """Initialize judge.
        
        Args:
            llm_adapter: Optional LLM adapter for LLM-based evaluation
        """
        self.llm_adapter = llm_adapter
        self.logger = logging.getLogger(__name__)
        self._oracle_functions = {
            'exact_match': self._exact_match,
            'list_match': self._list_match,
            'numeric_tolerance': self._numeric_tolerance,
            'regex_match': self._regex_match,
            'llm_judge': self._llm_judge,
            'schema_validation': self._schema_validation,
            'golden_action_match': self._golden_action_match,
        }
    
    @observe(name="evaluate")
    def evaluate(self, task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task result.
        
        Args:
            task: Task definition
            result: Task execution result
            
        Returns:
            Evaluation result
        """
        if 'golden_action' in task:
            oracle_type = 'golden_action_match'
            expected_output = task.get('golden_action')
        else:
            oracle_type = task.get('oracle', 'exact_match')
            expected_output = task.get('expected_output')
        
        actual_output = self._extract_output(result)
        
        if is_enabled():
            try:
                langfuse = get_client()
                langfuse.update_current_span(
                    input={
                        "oracle_type": oracle_type,
                        "expected_output": expected_output,
                        "actual_output": actual_output
                    },
                    metadata={"oracle_type": oracle_type}
                )
            except Exception as e:
                self.logger.debug(f"Langfuse update failed: {e}")
        
        if oracle_type not in self._oracle_functions:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
        
        oracle_function = self._oracle_functions[oracle_type]
        
        try:
            evaluation_result = oracle_function(expected_output, actual_output, task)
            
            eval_ctx = EvalContext(
                task_schema=task,
                logs={
                    "success": evaluation_result.get('success', False),
                    "tool_invocations": result.get("tool_invocations", []),
                    "repetition_results": result.get("repetition_results", [])
                }
            )
            
            metrics = {}
            for metric_name, metric in METRICS.items():
                metric_result = metric.evaluate(eval_ctx)
                metrics[metric_name] = {
                    "score": metric_result.score,
                    "details": metric_result.details
                }

            final_result = {
                "success": evaluation_result.get('success', False),
                "score": evaluation_result.get('score', 0.0),
                "oracle_type": oracle_type,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "details": evaluation_result.get('details', {}),
                "metrics": metrics,  # 메트릭 추가
                "error": None
            }
            
            # update current span with result 
            if is_enabled():
                try:
                    langfuse = get_client()
                    langfuse.update_current_span(
                        output=final_result,
                        metadata={
                            "success": final_result["success"],
                            "score": final_result["score"]
                        }
                    )
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            return final_result
            
        except Exception as e:
            error_result = {
                "success": False,
                "score": 0.0,
                "oracle_type": oracle_type,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "details": {},
                "error": str(e)
            }
            
            # update current span with error
            if is_enabled():
                try:
                    langfuse = get_client()
                    langfuse.update_current_span(
                        output=error_result,
                        level="ERROR"
                    )
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            return error_result
    
    def _extract_output(self, result: Dict[str, Any]) -> Any:
        """Extract the final output from execution result.
        
        Args:
            result: Task execution result
            
        Returns:
            Extracted output
        """
        if not result:
            return None
        
        # Try to get final response
        final_response = result.get('final_response', '')
        
        # Try to extract from tool calls if final response is empty
        if not final_response and 'steps' in result:
            for step in reversed(result['steps']):
                if step.get('tool_calls'):
                    for tool_call in step['tool_calls']:
                        if tool_call.get('success') and tool_call.get('result'):
                            return tool_call['result']
        
        return final_response
    
    def _golden_action_match(self, expected: List[Dict], actual: Any, task: Dict) -> Dict[str, Any]:
        """Evaluate based on golden_action - 실제 도구 실행 여부 확인"""
        if not isinstance(expected, list):
            expected = [expected]

        expected_tools = [e.get("tool") for e in expected if isinstance(e, dict)]
        
        # result에서 실제 실행된 도구 추출
        executed_tools = []
        if isinstance(actual, dict):
            # result 구조에서 tool_invocations 확인
            tool_invocations = actual.get('tool_invocations', [])
            executed_tools = [inv.get('tool_name') for inv in tool_invocations if inv.get('success')]
            
            # 또는 steps에서 확인
            if not executed_tools and 'steps' in actual:
                for step in actual.get('steps', []):
                    for tool_call in step.get('tool_calls', []):
                        if tool_call.get('success'):
                            executed_tools.append(tool_call.get('tool_name'))
        
        # 필수 도구가 모두 실행되었는지 확인
        matched = [tool for tool in expected_tools if tool in executed_tools]
        success = len(matched) == len(expected_tools) and len(executed_tools) > 0
        
        # 실패 이유 상세 분석
        missing_tools = [tool for tool in expected_tools if tool not in executed_tools]
        extra_tools = [tool for tool in executed_tools if tool not in expected_tools]
        
        return {
            "success": success,
            "score": 1.0 if success else 0.0,
            "details": {
                "match_type": "golden_action",
                "expected_tools": expected_tools,
                "executed_tools": executed_tools,
                "matched_tools": matched,
                "missing_tools": missing_tools,
                "extra_tools": extra_tools,
                "note": "Success only if all expected tools were actually executed"
            }
        }
    
    def _exact_match(self, expected: Any, actual: Any, task: Dict) -> Dict[str, Any]:
        """Exact match evaluation."""
        success = expected == actual
        return {
            "success": success,
            "score": 1.0 if success else 0.0,
            "details": {
                "match_type": "exact",
                "expected_type": type(expected).__name__,
                "actual_type": type(actual).__name__
            }
        }
    
    def _list_match(self, expected: List, actual: List, task: Dict) -> Dict[str, Any]:
        """List match evaluation (order-independent)."""
        if not isinstance(expected, list) or not isinstance(actual, list):
            return {"success": False, "score": 0.0, "details": {"error": "Both values must be lists"}}
        
        expected_set = set(expected)
        actual_set = set(actual)
        
        intersection = expected_set.intersection(actual_set)
        union = expected_set.union(actual_set)
        
        if len(union) == 0:
            jaccard_score = 1.0
        else:
            jaccard_score = len(intersection) / len(union)
        
        return {
            "success": jaccard_score == 1.0,
            "score": jaccard_score,
            "details": {
                "expected_count": len(expected),
                "actual_count": len(actual),
                "intersection_count": len(intersection),
                "jaccard_score": jaccard_score
            }
        }
    
    def _numeric_tolerance(self, expected: float, actual: float, task: Dict) -> Dict[str, Any]:
        """Numeric tolerance evaluation."""
        tolerance = task.get('tolerance', 0.01)
        
        try:
            expected_num = float(expected)
            actual_num = float(actual)
            
            diff = abs(expected_num - actual_num)
            success = diff <= tolerance
            
            return {
                "success": success,
                "score": max(0.0, 1.0 - (diff / tolerance)) if tolerance > 0 else (1.0 if success else 0.0),
                "details": {
                    "difference": diff,
                    "tolerance": tolerance,
                    "relative_error": diff / abs(expected_num) if expected_num != 0 else float('inf')
                }
            }
            
        except (ValueError, TypeError):
            return {"success": False, "score": 0.0, "details": {"error": "Cannot convert to numeric values"}}
    
    def _regex_match(self, expected_pattern: str, actual: str, task: Dict) -> Dict[str, Any]:
        """Regular expression match evaluation."""
        try:
            pattern = re.compile(expected_pattern)
            match = pattern.search(str(actual))
            
            return {
                "success": match is not None,
                "score": 1.0 if match else 0.0,
                "details": {
                    "pattern": expected_pattern,
                    "match_found": match is not None,
                    "match_groups": match.groups() if match else None
                }
            }
            
        except re.error as e:
            return {"success": False, "score": 0.0, "details": {"error": f"Invalid regex pattern: {str(e)}"}}
    
    @observe(as_type="generation")
    def _llm_judge(self, expected: Any, actual: Any, task: Dict) -> Dict[str, Any]:
        """LLM-based evaluation."""
        if not self.llm_adapter:
            return {"success": False, "score": 0.0, "details": {"error": "No LLM adapter provided for LLM judge"}}
        
        judge_prompt = task.get('judge_prompt', self._default_judge_prompt(task, expected, actual))
        
        messages = [
            {"role": "system", "content": "You are an expert judge evaluating task completion."},
            {"role": "user", "content": judge_prompt}
        ]
        
        try:
            response = self.llm_adapter.chat_completion(messages)
            judgment = response.get('message', {}).get('content', '')
            
            # Extract score (assuming format like "Score: 0.8" or similar)
            score_match = re.search(r'(?:score|rating):\s*([0-9]*\.?[0-9]+)', judgment.lower())
            score = float(score_match.group(1)) if score_match else 0.0
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            
            success = score >= task.get('llm_judge_threshold', 0.7)
            
            return {
                "success": success,
                "score": score,
                "details": {
                    "llm_judgment": judgment,
                    "threshold": task.get('llm_judge_threshold', 0.7)
                }
            }
            
        except Exception as e:
            return {"success": False, "score": 0.0, "details": {"error": f"LLM judge failed: {str(e)}"}}
    
    def _schema_validation(self, expected_schema: Dict, actual: Any, task: Dict) -> Dict[str, Any]:
        """Schema-based validation."""
        try:
            import jsonschema
            jsonschema.validate(actual, expected_schema)
            return {
                "success": True,
                "score": 1.0,
                "details": {"validation": "passed"}
            }
        except ImportError:
            return {"success": False, "score": 0.0, "details": {"error": "jsonschema package required"}}
        except jsonschema.ValidationError as e:
            return {
                "success": False,
                "score": 0.0,
                "details": {
                    "validation_error": str(e),
                    "failed_path": list(e.absolute_path)
                }
            }
    
    def _default_judge_prompt(self, task: Dict, expected: Any, actual: Any) -> str:
        """Generate default judge prompt for LLM evaluation."""
        description = task.get('instruction') or task.get('description', 'No description provided')
        
        return f"""
Task: {description}

Expected Output: {expected}

Actual Output: {actual}

Please evaluate how well the actual output matches the expected output for this task.
Consider both correctness and completeness.

Provide your evaluation in this format:
Score: [0.0 to 1.0]
Reasoning: [Your explanation]
"""