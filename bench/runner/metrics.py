"""Evaluation metrics for Ko-AgentBench."""

import json
import re
import logging
import yaml
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from ..adapters.base_adapter import BaseAdapter

class PromptLoader:
    """YAML 파일에서 프롬프트를 로딩하는 클래스"""
    _instance = None
    _prompts = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._prompts is None:
            self._load_prompts()
    
    def _load_prompts(self):
        """prompts.yaml 파일을 로딩"""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_path = os.path.join(current_dir, 'prompts.yaml')
            
            with open(prompts_path, 'r', encoding='utf-8') as f:
                self._prompts = yaml.safe_load(f)
                
        except Exception as e:
            logging.warning(f"프롬프트 파일 로딩 실패: {e}")
            self._prompts = {}
    
    def get_prompt(self, metric_name: str, template_type: str = 'template') -> str:
        """메트릭별 프롬프트 템플릿 반환"""
        if not self._prompts:
            return ""
        
        metric_prompts = self._prompts.get(metric_name, {})
        return metric_prompts.get(template_type, "")
    
    def get_system_prompt(self, metric_name: str) -> str:
        """메트릭별 시스템 프롬프트 반환"""
        return self.get_prompt(metric_name, 'system')
    
    def get_error_type_mapping(self, error_type: str) -> str:
        """에러 타입 한국어 매핑 반환"""
        mapping = self._prompts.get('error_detect', {}).get('error_type_mapping', {})
        return mapping.get(error_type, error_type)

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
    level = None  # "common", 1, 2, 3, 4, 5, 6, 7 또는 리스트
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        raise NotImplementedError


class LLMJudgeMetric(Metric):
    """LLM Judge를 사용하는 메트릭의 base class"""
    
    def __init__(self, llm_adapter: Optional[BaseAdapter] = None):
        self.llm_adapter = llm_adapter
        self.llm_adapters = []  # 복수 judge용
        self.prompt_loader = PromptLoader()
    
    def _parse_json_response(self, content: str, fallback_pattern: str, fallback_key: str) -> Dict[str, Any]:
        """JSON 응답 파싱"""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(fallback_pattern, content.lower() if 'true|false' in fallback_pattern else content)
            if match:
                if 'true|false' in fallback_pattern:
                    return {fallback_key: match.group(1) == "true", "reason": "Parsed from text"}
                else:
                    return {fallback_key: int(match.group(1)), "reason": "Parsed from text"}
            return {fallback_key: False if 'true|false' in fallback_pattern else 0, 
                    "reason": "Failed to parse LLM response"}
    
    def _call_single_judge(self, adapter: BaseAdapter, system_prompt: str, user_prompt: str,
                          fallback_pattern: str, fallback_key: str) -> Dict[str, Any]:
        """단일 judge 호출"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = adapter.chat_completion(messages, temperature=0.0)
            content = response.get("message", {}).get("content", "{}")
            
            result = self._parse_json_response(content, fallback_pattern, fallback_key)
            
            # reported_error_type 추출 (error_detect 메트릭용)
            if fallback_key == "error_reported":
                try:
                    if isinstance(content, str):
                        import re
                        error_type_match = re.search(r'"reported_error_type"\s*:\s*"([^"]+)"', content)
                        if error_type_match:
                            result["reported_error_type"] = error_type_match.group(1)
                        else:
                            result["reported_error_type"] = "none"
                    else:
                        result["reported_error_type"] = "none"
                except:
                    result["reported_error_type"] = "none"
            
            return result
            
        except Exception as e:
            default_value = False if 'true|false' in fallback_pattern else 0
            result = {fallback_key: default_value, "reason": f"Error: {str(e)}"}
            if fallback_key == "error_reported":
                result["reported_error_type"] = "none"
            return result
    
    def _call_multi_judge_binary(self, prompt: str, metric_key: str, 
                                 result_key: str = "error_reported") -> Dict[str, Any]:
        """복수 judge 호출 - 이진 결정 (다수결)
        
        Args:
            prompt: 평가 프롬프트
            metric_key: 시스템 프롬프트 키 (예: 'error_detect')
            result_key: 결과 키 (예: 'error_reported', 'success')
        """
        if hasattr(self, 'llm_adapters') and self.llm_adapters:
            judges_to_use = self.llm_adapters
        elif self.llm_adapter:
            judges_to_use = [self.llm_adapter]
        else:
            return {result_key: False, "confidence": 0.0, "reason": "No LLM adapter available"}
        
        system_prompt = self.prompt_loader.get_system_prompt(metric_key)
        fallback_pattern = f'"{result_key}"\\s*:\\s*(true|false)'
        
        all_results = []
        
        for idx, adapter in enumerate(judges_to_use, 1):
            result = self._call_single_judge(adapter, system_prompt, prompt, 
                                            fallback_pattern, result_key)
            result['judge_id'] = idx
            result['judge_model'] = getattr(adapter, 'model_name', 'unknown')
            all_results.append(result)
        
        if not all_results:
            return {result_key: False, "confidence": 0.0, "reason": "All judges failed"}
        
        # 다수결
        positive_votes = sum(1 for r in all_results if r.get(result_key, False))
        total_votes = len(all_results)
        final_decision = positive_votes > (total_votes / 2)
        
        # reported_error_type 다수결 (가장 많이 선택된 타입)
        # error_reported가 True인 judge들의 reported_error_type만 고려
        positive_error_types = [r.get("reported_error_type", "none") for r in all_results if r.get(result_key, False)]
        if positive_error_types:
            from collections import Counter
            error_type_counts = Counter(positive_error_types)
            most_common_error_type = error_type_counts.most_common(1)[0][0]
        else:
            most_common_error_type = "none"
        
        # 평균 confidence (있는 경우)
        avg_confidence = sum(r.get("confidence", 0.0) for r in all_results) / total_votes
        
        judge_reasons = [
            f"Judge {r['judge_id']} ({r.get('judge_model', 'unknown')}): "
            f"{'Yes' if r.get(result_key) else 'No'} (type: {r.get('reported_error_type', 'none')})"
            for r in all_results
        ]
        
        return {
            result_key: final_decision,
            "reported_error_type": most_common_error_type,
            "confidence": avg_confidence,
            "reason": f"Multi-judge vote: {positive_votes}/{total_votes}. " + "; ".join(judge_reasons),
            "individual_results": all_results,
            "vote_count": {
                "positive": positive_votes,
                "negative": total_votes - positive_votes,
                "total": total_votes
            }
        }
    
    def _call_multi_judge_score(self, prompt: str, metric_key: str, 
                                min_score: int = 1, max_score: int = 5) -> Dict[str, Any]:
        """복수 judge 호출 - 점수 평가 (평균)
        
        Args:
            prompt: 평가 프롬프트
            metric_key: 시스템 프롬프트 키
            min_score: 최소 점수
            max_score: 최대 점수
        """
        if hasattr(self, 'llm_adapters') and self.llm_adapters:
            judges_to_use = self.llm_adapters
        elif self.llm_adapter:
            judges_to_use = [self.llm_adapter]
        else:
            return {"score": 0, "reason": "No LLM adapter available"}
        
        system_prompt = self.prompt_loader.get_system_prompt(metric_key)
        fallback_pattern = f'"score"\\s*:\\s*([{min_score}-{max_score}])'
        
        all_results = []
        
        for idx, adapter in enumerate(judges_to_use, 1):
            result = self._call_single_judge(adapter, system_prompt, prompt, 
                                            fallback_pattern, "score")
            result['judge_id'] = idx
            result['judge_model'] = getattr(adapter, 'model_name', 'unknown')
            all_results.append(result)
        
        if not all_results:
            return {"score": 0, "reason": "All judges failed"}
        
        # 점수 평균
        avg_score = sum(r.get("score", 0) for r in all_results) / len(all_results)
        
        judge_scores = [
            f"Judge {r['judge_id']} ({r.get('judge_model', 'unknown')}): {r.get('score', 0)}/{max_score}"
            for r in all_results
        ]
        
        return {
            "score": round(avg_score),  # 반올림
            "average_score": avg_score,  # 정확한 평균
            "reason": f"Multi-judge average: {avg_score:.2f}/{max_score}. " + "; ".join(judge_scores),
            "individual_results": all_results
        }


#공통 메트릭
class SRMetric(Metric):
    """SR(성공률): success_condition 충족 여부"""
    name = "SR"
    level = "common"
    
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
    level = "common"
    
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
    level = "common"
    
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
        
#레벨1 메트릭
class ToolAccMetric(Metric):
    """ToolAcc: 첫번째 예측 툴이 golden_action.tool과 일치하는지"""
    name = "ToolAcc"
    level = 1

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:

        golden = ctx.task_schema.get("golden_action", [])
        if isinstance(golden, dict):
            golden = [golden]
        golden_tool = next((g["tool"] for g in golden if isinstance(g, dict) and "tool" in g), None)

        invocations = ctx.logs.get("tool_invocations", []) or []
        first_pred_tool = next((inv.get("tool") or inv.get("tool_name") for inv in invocations if inv.get("tool") or inv.get("tool_name")), None)

        matched = (golden_tool is not None and first_pred_tool == golden_tool)
        score = 1.0 if matched else 0.0

        return EvaluationResult(
            self.name,
            score,
            {"matched": matched}  
        )
        
        
class ArgAccMetric(Metric):
    """
    ArgAcc: 도구 인자 정확도. (P/R/F1)
    """
    name = "ArgAcc"
    level = 1
    DEFAULT_SCORE_KEY = "f1"

    @staticmethod
    def _prf1(tp: int, fp: int, fn: int):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    @staticmethod
    def _get_first_pred_args_for_tool(ctx: EvalContext, golden_tool: str) -> Dict[str, Any]:
        invocations = ctx.logs.get("tool_invocations", []) or []
        for inv in invocations:
            tool = inv.get("tool") or inv.get("tool_name")
            if tool == golden_tool:
                args = inv.get("arguments")
                if args is None:
                    args = inv.get("args")
                return args or {}
        return {}

    @classmethod
    def _compute_prf(cls, ctx: EvalContext) -> Dict[str, Any]:
        golden_actions = ctx.task_schema.get("golden_action", [])
        if isinstance(golden_actions, dict):
            golden_actions = [golden_actions]

        if not golden_actions or not isinstance(golden_actions[0], dict):
            return {"ok": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        golden_tool = golden_actions[0].get("tool")
        golden_args = golden_actions[0].get("args", {}) or {}
        arg_schema = ctx.task_schema.get("arg_schema", {}) or {}

        if not golden_tool:
            return {"ok": False, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        pred_args = cls._get_first_pred_args_for_tool(ctx, golden_tool)

        # TP/FP
        tp = fp = 0
        for k, v in pred_args.items():
            if k in golden_args and v == golden_args[k]:
                tp += 1
            else:
                fp += 1

        # FN 
        fn = 0
        for k, gv in golden_args.items():
            pv = pred_args.get(k, (arg_schema.get(k, {}) or {}).get("default"))
            if pv != gv:
                fn += 1

        precision, recall, f1 = cls._prf1(tp, fp, fn)
        return {"ok": True, "precision": precision, "recall": recall, "f1": f1}

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        r = self._compute_prf(ctx)
        if not r.get("ok"):
            return EvaluationResult(self.name, 0.0, {"ok": False})

        score_key = ctx.task_schema.get("argacc_score_key", self.DEFAULT_SCORE_KEY)
        if score_key not in ("precision", "recall", "f1"):
            score_key = self.DEFAULT_SCORE_KEY

        score = float(r[score_key])

        details = {"ok": True, "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]}
        return EvaluationResult(self.name, score, details)


class CallEMMetric(Metric):
    """CallEM: 첫 호출이 정답 호출(tool+args)과 완전히 동일한지 (EM)"""
    name = "CallEM"
    level = 1

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        golden_actions = ctx.task_schema.get("golden_action", [])
        if isinstance(golden_actions, dict):
            golden_actions = [golden_actions]
        if not golden_actions or not isinstance(golden_actions[0], dict):
            return EvaluationResult(self.name, 0.0, {})

        golden_tool = golden_actions[0].get("tool")
        golden_args = golden_actions[0].get("args", {}) or {}
        if not golden_tool:
            return EvaluationResult(self.name, 0.0, {})


        invocations = ctx.logs.get("tool_invocations", []) or []
        first_pred_tool = None
        first_pred_args = {}

        for inv in invocations:
            tool = inv.get("tool") or inv.get("tool_name")
            if tool:
                first_pred_tool = tool
                # 표준 키: "arguments"
                first_pred_args = inv.get("arguments") or {}
                break

        matched = (
            first_pred_tool is not None
            and first_pred_tool == golden_tool
            and first_pred_args == golden_args
        )
        score = 1.0 if matched else 0.0

        return EvaluationResult(self.name, score, {})

class RespOKMetric(Metric):
    """
    RespOK : actual_output이 resp_schema를 만족하는지 평가
    """
    name = "RespOK"
    level = 1

    @staticmethod
    def _extract_candidate(ctx: EvalContext):
        cand = ctx.logs.get("actual_output")

        if isinstance(cand, (str, dict, list)):
            return cand
        return None

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        schema = ctx.task_schema.get("resp_schema")
        if not isinstance(schema, dict):
            return EvaluationResult(self.name, 0.0, {})

        candidate = self._extract_candidate(ctx)
        if candidate is None:
            return EvaluationResult(self.name, 0.0, {})

        # jsonschema가 있으면 
        try:
            import jsonschema
            jsonschema.validate(instance=candidate, schema=schema)
            return EvaluationResult(self.name, 1.0, {})
        except ImportError:

            expected_type = schema.get("type")
            ok = (expected_type == "string" and isinstance(candidate, str))
            return EvaluationResult(self.name, 1.0 if ok else 0.0, {})
        except Exception:
            return EvaluationResult(self.name, 0.0, {})


#레벨2 메트릭
class SelectAccMetric(Metric):
    name = "SelectAcc"
    level = 2

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        ga = ctx.task_schema.get("golden_action", [])
        if isinstance(ga, dict):
            ga = [ga]
        golden_tool = ga[0].get("tool") if (ga and isinstance(ga[0], dict)) else None

        pred_tool = ctx.logs.get("selected_tool")
        if not isinstance(pred_tool, str) or not pred_tool:
            inv = ctx.logs.get("tool_invocations", []) or []
            for call in inv:
                if isinstance(call, dict):
                    t = call.get("tool") or call.get("tool_name")
                    if isinstance(t, str) and t:
                        pred_tool = t
                        break

        success = (golden_tool is not None and pred_tool == golden_tool)
        score = 1.0 if success else 0.0

        return EvaluationResult(self.name, score, {"success": success})

#레벨3 메트릭   
class FSMMetric(Metric):
    """FSM(Full Sequence Match): 정답 경로와 완전 일치"""
    name = "FSM"
    level = 3
    
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
    level = 3
    
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
    level = 3
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        minimum_steps = ctx.task_schema.get("minimum_steps")
        
        # minimum_steps가 없으면 golden_action 개수로 추정
        if minimum_steps is None:
            golden_action = ctx.task_schema.get("golden_action", [])
            minimum_steps = len(golden_action) if golden_action else 1
        
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
    level = 3
    
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
        
#레벨4 메트릭
class CoverageMetric(Metric):
    """Coverage: 필수 소스를 모두 성공적으로 조회했는지 비율
    - golden_action에 명시된 도구들을 모두 사용했는지 확인
    """
    name = "Coverage"
    level = 4
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        # golden_action에서 필수 도구 추출
        golden_action = ctx.task_schema.get("golden_action", [])
        required_tools = [action.get("tool") for action in golden_action if action.get("tool")]
        
        if not required_tools:
            # 필수 도구가 없으면 만점
            return EvaluationResult(self.name, 1.0, {
                "reason": "golden_action에 도구 정보 없음",
                "required_tools": [],
                "covered_tools": []
            })
        
        # 실제로 성공한 도구 호출 확인
        action_trace = ctx.action_trace
        successful_tools = set()
        
        for action in action_trace:
            tool_name = action.get("tool")
            success = action.get("success", False)
            
            # 성공하고 결과가 있는 경우만 카운트
            if success and tool_name:
                result = action.get("result")
                # 결과가 비어있지 않은지 확인
                has_meaningful_result = False
                
                if isinstance(result, dict):
                    # 검색 결과의 경우 items나 places 확인
                    items = result.get("items") or result.get("places") or []
                    if items:
                        has_meaningful_result = True
                    # total_count가 0보다 크면 의미있는 결과
                    elif result.get("total_count", 0) > 0:
                        has_meaningful_result = True
                elif result:
                    # dict가 아니더라도 결과가 있으면 의미있다고 판단
                    has_meaningful_result = True
                
                if has_meaningful_result:
                    successful_tools.add(tool_name)
        
        # 필수 도구 중 성공한 것들
        covered_tools = [tool for tool in required_tools if tool in successful_tools]
        
        # 중복 제거 (같은 도구를 여러 번 호출할 수 있으므로)
        unique_required = list(set(required_tools))
        unique_covered = list(set(covered_tools))
        
        score = len(unique_covered) / len(unique_required) if unique_required else 0.0
        
        return EvaluationResult(self.name, score, {
            "required_tools": unique_required,
            "covered_tools": unique_covered,
            "missing_tools": [t for t in unique_required if t not in unique_covered],
            "total_required": len(unique_required),
            "total_covered": len(unique_covered)
        })
    
class SourceEPRMetric(Metric):
    """SourceEPR: 소스별 유효 호출 비율의 평균
    
    각 필수 소스(도구)에 대해
    - 유효 호출 = success=True AND error=None AND 의미있는 결과
    - EPR = 유효 호출 / 전체 호출
    - 최종 점수 = 모든 소스의 EPR 평균
    """
    name = "SourceEPR"
    level = 4
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        # golden_action에서 필수 도구 추출
        golden_action = ctx.task_schema.get("golden_action", [])
        required_tools = [action.get("tool") for action in golden_action if action.get("tool")]
        
        if not required_tools:
            return EvaluationResult(self.name, 1.0, {
                "reason": "필수 도구 없음",
                "source_eprs": {}
            })
        
        # 중복 제거
        unique_tools = list(set(required_tools))
        
        # 실제 호출 로그
        action_trace = ctx.action_trace
        
        source_eprs = {}
        all_epr_values = []
        
        for tool_name in unique_tools:
            # 이 도구에 대한 모든 호출
            tool_calls = [
                action for action in action_trace 
                if action.get("tool") == tool_name
            ]
            
            if not tool_calls:
                # 호출하지 않은 도구는 EPR = 0
                source_eprs[tool_name] = {
                    "epr": 0.0,
                    "total_calls": 0,
                    "valid_calls": 0,
                    "reason": "도구 미호출"
                }
                all_epr_values.append(0.0)
                continue
            
            # 유효한 호출 카운트
            valid_calls = 0
            for call in tool_calls:
                success = call.get("success", False)
                error = call.get("error")
                
                if success and not error:
                    # 결과가 의미있는지도 확인
                    result = call.get("result")
                    if isinstance(result, dict):
                        items = result.get("items") or result.get("places") or []
                        if items or result.get("total_count", 0) > 0:
                            valid_calls += 1
                    elif result:
                        valid_calls += 1
            
            epr = valid_calls / len(tool_calls)
            source_eprs[tool_name] = {
                "epr": round(epr, 4),
                "total_calls": len(tool_calls),
                "valid_calls": valid_calls
            }
            all_epr_values.append(epr)
        
        # 모든 소스의 EPR 평균
        score = sum(all_epr_values) / len(all_epr_values) if all_epr_values else 0.0
        
        return EvaluationResult(self.name, score, {
            "source_eprs": source_eprs,
            "average_epr": round(score, 4),
            "total_sources": len(unique_tools)
        })

#레벨5 메트릭
class ErrorDetectMetric(LLMJudgeMetric):
    """
    ErrorDetect : 주입된 오류(tool,error_type)를 모델이 보고하는지의 비율
    LLM-as-a-Judge로 final_response에서 에러 보고 여부를 판단
    """
    name = "ErrorDetect"
    level = 5

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        inj = ctx.task_schema.get("error_injection", {}) or {}
        inj_tool = inj.get("tool")
        inj_type = inj.get("error_type")

        # 에러 주입이 없으면 평가 불가
        if not inj_tool or not inj_type:
            return EvaluationResult(self.name, 0.0, {
                "reason": "No error injection defined",
                "error_occurred": False,
                "error_reported": False
            })

        # 1단계: 실제로 에러가 발생했는지 확인
        error_occurred = False
        tool_calls = ctx.logs.get("tool_calls") or []
        
        for inv in tool_calls:
            if isinstance(inv, dict) and inv.get("error"):
                t = inv.get("tool") or inv.get("tool_name")
                # error 필드에서 에러 타입 추출
                error_msg = inv.get("error", "")
                if t == inj_tool and error_msg:
                    # 에러 메시지에서 타입 매칭 확인
                    if (inj_type == "timeout" and "timeout" in error_msg.lower()) or \
                       (inj_type == "complete_unavailable" and "unavailable" in error_msg.lower()) or \
                       (inj_type == "data_not_available" and "no data" in error_msg.lower()):
                        error_occurred = True
                        break

        if not error_occurred:
            return EvaluationResult(self.name, 0.0, {
                "reason": "Error injection did not trigger",
                "error_occurred": False,
                "error_reported": False
            })

        # 2단계: 모델이 에러를 보고했는지 LLM Judge로 판단
        final_response = ctx.logs.get("actual_output") or ctx.logs.get("final_response", "")
        
        # final_response 검증 강화
        if final_response is None:
            final_response = ""
        elif not isinstance(final_response, str):
            final_response = str(final_response)
        
        # 빈 문자열이거나 너무 짧은 경우 처리
        if not final_response or len(final_response.strip()) < 3:
            return EvaluationResult(self.name, 0.0, {
                "reason": "No final response to evaluate",
                "error_occurred": True,
                "error_reported": False
            })

        # LLM Judge 프롬프트 구성
        error_type_korean = self.prompt_loader.get_error_type_mapping(inj_type)
        prompt_template = self.prompt_loader.get_prompt('error_detect')
        prompt = prompt_template.format(
            injected_tool=inj_tool,
            injected_error_type=inj_type,
            error_type_korean=error_type_korean,
            final_response=final_response
        )

        # Multi-judge 호출
        llm_result = self._call_multi_judge_binary(prompt, 'error_detect', 'error_reported')
        error_reported = llm_result.get("error_reported", False)
        reported_error_type = llm_result.get("reported_error_type", "none")
        confidence = llm_result.get("confidence", 0.0)
        
        # 에러 타입 일치 여부 비교
        type_matches = (reported_error_type == inj_type)
        
        # 점수 계산: 에러를 보고하고 타입도 일치해야 만점
        # 단, 다수결에서 실패했지만 reported_error_type이 유효한 경우도 고려
        if error_reported and type_matches:
            score = 1.0
        elif not error_reported and reported_error_type != "none" and type_matches:
            # 다수결에서 실패했지만 에러 타입이 올바르게 감지된 경우 부분 점수
            score = 0.5
        else:
            score = 0.0
        
        return EvaluationResult(self.name, score, {
            "error_occurred": True,
            "error_reported": error_reported,
            "reported_error_type": reported_error_type,
            "type_matches": type_matches,
            "confidence": confidence,
            "llm_reason": llm_result.get("reason", "No reason provided"),
            "injected_tool": inj_tool,
            "injected_error_type": inj_type,
            "vote_details": llm_result.get("vote_count"),
            "individual_judges": llm_result.get("individual_results")
        })


class GracefulFailMetric(Metric):
    """
    GracefulFail : 전체 실패 케이스 중 환각 없이 안전하게 실패를 보고한 비율
    """
    name = "GracefulFail"
    level = 5

    @staticmethod
    def _has_output(v: Any) -> bool:
        """출력이 유효한지 확인"""
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip() != ""
        if isinstance(v, (list, dict)):
            return len(v) > 0
        return True

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:

        inj = ctx.task_schema.get("error_injection")
        if not isinstance(inj, dict) or not inj:
            return EvaluationResult(self.name, 0.0, {})

        # 반복 실행 결과가 있는 경우
        reps = ctx.logs.get("repetition_results")
        if isinstance(reps, list) and reps and isinstance(reps[0], dict):
            total_fail = sum(1 for r in reps if not bool(r.get("success", False)))
            graceful_fail = sum(
                1 for r in reps
                if (not bool(r.get("success", False))) and (not self._has_output(r.get("actual_output")))
            )
            score = (graceful_fail / total_fail) if total_fail > 0 else 0.0
            return EvaluationResult(self.name, score, {
                "total_fail": total_fail,
                "graceful_fail": graceful_fail,
                "repetitions": len(reps)
            })

        # 반복 정보 없으면 단일 시도
        success = bool(ctx.logs.get("success", False))
        if success:
            return EvaluationResult(self.name, 0.0, {})
        
        out = ctx.logs.get("actual_output", None)
        graceful = not self._has_output(out)
        return EvaluationResult(self.name, 1.0 if graceful else 0.0, {})
    

class FallbackSRMetric(Metric):
    """
    FallbackSR : 주 도구 실패(에러 주입) 시 대체 도구/경로 시도 중 성공 비율
    """
    name = "FallbackSR"
    level = 5

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        # 에러 주입 없는 태스크면 0 (비적용)
        error_injection = ctx.task_schema.get("error_injection")
        if not error_injection:
            return EvaluationResult(self.name, 0.0, {"reason": "No error injection"})

        fallback_opts = ctx.task_schema.get("fallback_options") or []
        fallback_tools = {
            opt.get("tool") for opt in fallback_opts
            if isinstance(opt, dict) and opt.get("tool")
        }
        if not fallback_tools:
            return EvaluationResult(self.name, None, {"reason": "No fallback options - not applicable"})

        invocations = ctx.logs.get("tool_calls", []) or []
        injected_tool = error_injection.get("tool")
        
        # 1단계: 에러 주입된 도구가 실패했는지 확인
        injected_tool_failed = False
        fallback_attempts = 0
        fallback_successes = 0
        
        for call in invocations:
            if not isinstance(call, dict):
                continue
            tool = call.get("tool") or call.get("tool_name")
            
            # 에러 주입된 도구가 실패했는지 확인
            if tool == injected_tool:
                if call.get("success") is False or call.get("error"):
                    injected_tool_failed = True
            
            # 대체 도구 시도 확인
            elif tool in fallback_tools:
                fallback_attempts += 1
                # 대체 도구 성공 여부 확인
                if call.get("success") is True and not call.get("error"):
                    fallback_successes += 1

        # 에러 주입된 도구가 실패하지 않았으면 평가 불가
        if not injected_tool_failed:
            return EvaluationResult(self.name, 0.0, {"reason": "Injected tool did not fail"})
        
        # 대체 도구를 시도하지 않았으면 0점
        if fallback_attempts == 0:
            return EvaluationResult(self.name, 0.0, {"reason": "No fallback attempts"})

        score = (fallback_successes / fallback_attempts) if fallback_attempts > 0 else 0.0
        return EvaluationResult(self.name, score, {
            "injected_tool": injected_tool,
            "injected_tool_failed": injected_tool_failed,
            "fallback_attempts": fallback_attempts,
            "fallback_successes": fallback_successes,
            "fallback_tools": list(fallback_tools)
        })

#레벨6 메트릭
class ReuseRateMetric(Metric):
    """ReuseRate: 재사용 기회 대비 실제 재사용 비율"""
    name = "ReuseRate"
    level = 6
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        golden_action = ctx.task_schema.get("golden_action", [])
        action_trace = ctx.action_trace
        
        # 재사용 기회: golden_action에서 "action": "context_used" 카운트
        reuse_opportunities = sum(
            1 for action in golden_action 
            if isinstance(action, dict) and action.get("action") == "context_used"
        )
        
        if reuse_opportunities == 0:
            return EvaluationResult(
                self.name,
                0.0,
                {"reuse_opportunities": 0, "reused": 0, "reason": "No reuse opportunities"}
            )
        
        # 실제 재사용 판단: 동일한 도구를 동일한 파라미터로 재호출하지 않은 경우
        # golden_action에서 context_used 이전의 도구 호출과 이후 상황 비교
        
        # golden_action의 실제 도구 호출 순서 파악
        golden_tools = []
        for i, action in enumerate(golden_action):
            if isinstance(action, dict):
                if "tool" in action and action.get("tool"):
                    golden_tools.append({
                        "index": i,
                        "tool": action.get("tool"),
                        "args": action.get("args", {})
                    })
                elif action.get("action") == "context_used":
                    golden_tools.append({
                        "index": i,
                        "action": "context_used"
                    })
        
        # 실제 호출된 도구들의 시그니처 생성
        actual_signatures = []
        for action in action_trace:
            if action.get("success"):
                signature = (
                    action.get("tool"),
                    json.dumps(action.get("args", {}), sort_keys=True)
                )
                actual_signatures.append(signature)
        
        # 재사용 판단
        # context_used가 나타나는 위치에서 이전 호출이 재사용되었는지 확인
        reused = 0
        
        for i, golden_item in enumerate(golden_tools):
            if golden_item.get("action") == "context_used":
                # 이전 golden 도구 호출 찾기
                prev_tool_index = None
                for j in range(i-1, -1, -1):
                    if "tool" in golden_tools[j]:
                        prev_tool_index = j
                        break
                
                if prev_tool_index is not None:
                    prev_golden = golden_tools[prev_tool_index]
                    prev_signature = (
                        prev_golden["tool"],
                        json.dumps(prev_golden["args"], sort_keys=True)
                    )
                    
                    # 다음 도구 찾기
                    next_tool_index = None
                    for j in range(i+1, len(golden_tools)):
                        if "tool" in golden_tools[j]:
                            next_tool_index = j
                            break
                    
                    if next_tool_index is not None:
                        next_golden = golden_tools[next_tool_index]
                        next_signature = (
                            next_golden["tool"],
                            json.dumps(next_golden["args"], sort_keys=True)
                        )
                        
                        try:
                            prev_call_indices = [idx for idx, sig in enumerate(actual_signatures) if sig == prev_signature]
                            next_call_indices = [idx for idx, sig in enumerate(actual_signatures) if sig == next_signature]
                            
                            # context_used 위치에서 실제로 재호출 없이 진행되었는지 확인
                            # 이전 호출과 다음 호출 사이에 동일한 이전 호출이 없어야 함
                            if prev_call_indices and next_call_indices:
                                # 가장 마지막 prev 호출과 첫 번째 next 호출 사이에
                                # 추가 prev 호출이 없으면 재사용한 것
                                last_prev_idx = max(prev_call_indices)
                                first_next_idx = min([idx for idx in next_call_indices if idx > last_prev_idx], default=None)
                                
                                if first_next_idx is not None:
                                    between_calls = actual_signatures[last_prev_idx+1:first_next_idx]
                                    if prev_signature not in between_calls:
                                        reused += 1
                        except (ValueError, TypeError):
                            pass
        
        score = reused / reuse_opportunities if reuse_opportunities > 0 else 0.0
        
        return EvaluationResult(
            self.name,
            score,
            {
                "reuse_opportunities": reuse_opportunities,
                "reused": reused,
                "total_actual_calls": len(actual_signatures),
                "golden_tools": [g.get("tool") for g in golden_tools if "tool" in g]
            }
        )


class RedundantCallRateMetric(Metric):
    """RedundantCallRate: 재사용이 정답인 상황에서 불필요한 API 호출 비율"""
    name = "RedundantCallRate"
    level = 6
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        golden_action = ctx.task_schema.get("golden_action", [])
        action_trace = ctx.action_trace
        
        # 재사용 기회: golden_action에서 "action": "context_used" 카운트
        reuse_opportunities = sum(
            1 for action in golden_action 
            if isinstance(action, dict) and action.get("action") == "context_used"
        )
        
        if reuse_opportunities == 0:
            return EvaluationResult(
                self.name,
                1.0,  # 재사용 기회가 없으면 만점
                {"reuse_opportunities": 0, "redundant_calls": 0, "reason": "No reuse opportunities"}
            )
        
        # 불필요한 호출: 동일한 도구를 동일한 파라미터로 재호출한 경우
        tool_call_signatures = []
        redundant_calls = 0
        
        for action in action_trace:
            if action.get("success"):
                signature = (
                    action.get("tool"),
                    json.dumps(action.get("args", {}), sort_keys=True)
                )
                if signature in tool_call_signatures:
                    redundant_calls += 1
                else:
                    tool_call_signatures.append(signature)
        
        redundant_rate = redundant_calls / reuse_opportunities if reuse_opportunities > 0 else 0.0
        score = 1.0 - redundant_rate  # 높을수록 좋음
        
        return EvaluationResult(
            self.name,
            score,
            {
                "reuse_opportunities": reuse_opportunities,
                "redundant_calls": redundant_calls,
                "non_redundant_calls": reuse_opportunities - redundant_calls,
                "total_calls": len(action_trace),
                "unique_calls": len(tool_call_signatures),
                "redundant_rate": redundant_rate,  # 원래 값도 저장
                "non_redundant_rate": score  # 반전된 값
            }
        )


class EffScoreMetric(LLMJudgeMetric):
    """EffScore: 성공 시 최소 호출 수 대비 실제 호출 수의 효율 (LLM Judge)"""
    name = "EffScore"
    level = 6
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        query = ctx.task_schema.get("instruction", "")
        final_answer = ctx.logs.get("actual_output") or ctx.logs.get("final_response", "")
        
        tool_calls = []
        for action in ctx.action_trace:
            tool_calls.append({
                "tool": action.get("tool"),
                "args": action.get("args", {}),
                "success": action.get("success", False)
            })
        
        # LLM Judge 프롬프트
        prompt_template = self.prompt_loader.get_prompt('eff_score')
        prompt = prompt_template.format(
            query=query,
            final_answer=final_answer,
            tool_calls=json.dumps(tool_calls, ensure_ascii=False, indent=2)
        )

        # Multi-judge 호출
        llm_result = self._call_multi_judge_binary(prompt, 'eff_score', 'success')
        success = llm_result.get("success", False)
        
        actual_calls = len(ctx.action_trace)
        minimum_calls = ctx.task_schema.get("minimum_calls")
        
        if minimum_calls is None:
            # golden_action에서 실제 tool 호출만 카운트 (context_used 제외)
            golden_action = ctx.task_schema.get("golden_action", [])
            unique_tools = set()
            for action in golden_action:
                if isinstance(action, dict):
                    tool = action.get("tool")
                    if tool:  # tool이 있는 경우만 (context_used는 제외)
                        args_str = json.dumps(action.get("args", {}), sort_keys=True)
                        unique_tools.add((tool, args_str))
            minimum_calls = len(unique_tools) if unique_tools else 1
        
        if not success or actual_calls <= 0:
            return EvaluationResult(
                self.name,
                0.0,
                {
                    "success": success,
                    "actual_calls": actual_calls,
                    "minimum_calls": minimum_calls,
                    "llm_reason": llm_result.get("reason", ""),
                    "vote_details": llm_result.get("vote_count"),
                    "individual_judges": llm_result.get("individual_results")
                }
            )
        
        score = min(1.0, minimum_calls / actual_calls)
        
        return EvaluationResult(
            self.name,
            score,
            {
                "success": success,
                "actual_calls": actual_calls,
                "minimum_calls": minimum_calls,
                "efficiency": score,
                "llm_reason": llm_result.get("reason", ""),
                "vote_details": llm_result.get("vote_count"),
                "individual_judges": llm_result.get("individual_results")
            }
        )

#레벨7 메트릭
class ContextRetentionMetric(LLMJudgeMetric):
    """ContextRetention: 멀티턴 대화에서 과거 맥락 활용도 (LLM Judge)"""
    name = "ContextRetention"
    level = 7
    
    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """메시지 목록을 읽기 쉬운 형식으로 변환"""
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"[턴 {i}] 사용자: {content}")
            elif role == "assistant":
                formatted.append(f"[턴 {i}] AI: {content[:500]}{'...' if len(content) > 500 else ''}")
        return "\n".join(formatted)
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        conversation_log = ctx.logs.get("conversation_log", {})
        
        if not conversation_log:
            return EvaluationResult(
                self.name,
                0.0,
                {"reason": "No conversation log available"}
            )
        
        messages = conversation_log.get("messages", [])
        total_messages = conversation_log.get("total_messages", len(messages))
        
        # LLM Judge 프롬프트 구성
        prompt_template = self.prompt_loader.get_prompt('context_retention')
        prompt = prompt_template.format(
            formatted_messages=self._format_messages(messages)
        )

        # Multi-judge 호출 (점수 평균)
        llm_result = self._call_multi_judge_score(prompt, 'context_retention', 1, 5)
        raw_score = llm_result.get("score", 0)
        
        # 1-5점을 0-1 스케일로 정규화
        # 1점 -> 0.0, 2점 -> 0.25, 3점 -> 0.5, 4점 -> 0.75, 5점 -> 1.0
        normalized_score = max(0.0, (raw_score - 1) / 4.0)
        
        return EvaluationResult(
            self.name, normalized_score,
            {
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "average_score": llm_result.get("average_score", 0),
                "reason": llm_result.get("reason", ""),
                "total_messages": total_messages,
                "evaluated_messages": len([m for m in messages if m.get("role") != "tool"]),
                "individual_judges": llm_result.get("individual_results")
            }
        )


class RefRecallMetric(LLMJudgeMetric):
    """RefRecall: 오래된 정보 회상 능력 (LLM Judge)"""
    name = "RefRecall"
    level = 7
    
    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> str:
        """메시지 목록을 읽기 쉬운 형식으로 변환"""
        formatted = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted.append(f"[턴 {i}] 사용자: {content}")
            elif role == "assistant":
                formatted.append(f"[턴 {i}] AI: {content[:500]}{'...' if len(content) > 500 else ''}")
        return "\n".join(formatted)
    
    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        conversation_log = ctx.logs.get("conversation_log", {})
        
        if not conversation_log:
            return EvaluationResult(
                self.name,
                0.0,
                {"reason": "No conversation log available"}
            )
        
        messages = conversation_log.get("messages", [])
        total_messages = conversation_log.get("total_messages", len(messages))
        
        # LLM Judge 프롬프트 구성
        prompt_template = self.prompt_loader.get_prompt('ref_recall')
        prompt = prompt_template.format(
            formatted_messages=self._format_messages(messages)
        )

        # Multi-judge 호출 (점수 평균)
        llm_result = self._call_multi_judge_score(prompt, 'context_retention', 1, 5)
        raw_score = llm_result.get("score", 0)
        normalized_score = max(0.0, (raw_score - 1) / 4.0)
        
        return EvaluationResult(
            self.name,
            normalized_score,
            {
                "raw_score": raw_score,
                "normalized_score": normalized_score,
                "average_score": llm_result.get("average_score", 0),
                "reason": llm_result.get("reason", ""),
                "total_messages": total_messages,
                "evaluated_messages": len([m for m in messages if m.get("role") != "tool"]),
                "individual_judges": llm_result.get("individual_results")
            }
        )


# 메트릭 레지스트리
METRICS = {
    #공통 메트릭
    "SR": SRMetric(),
    "EPR_CVR": EPRCVRMetric(),
    "pass@k": PassAtKMetric(),
    #레벨1 메트릭
    "ToolAcc": ToolAccMetric(),
    "ArgAcc": ArgAccMetric(),
    "CallEM": CallEMMetric(),     
    "RespOK": RespOKMetric(),  
    #레벨2 메트릭
    "SelectAcc": SelectAccMetric(),
    #레벨3 메트릭
    "FSM": FSMMetric(),
    "PSM": PSMMetric(),
    "ΔSteps_norm": DeltaStepsNormMetric(),
    "ProvAcc": ProvAccMetric(),
    #레벨4 메트릭
    "Coverage": CoverageMetric(),
    "SourceEPR": SourceEPRMetric(),
    #레벨5 메트릭 
    "ErrorDetect": ErrorDetectMetric(), 
    "GracefulFail": GracefulFailMetric(),
    "FallbackSR": FallbackSRMetric(),
    #레벨6 메트릭
    "ReuseRate": ReuseRateMetric(),
    "RedundantCallRate": RedundantCallRateMetric(),
    "EffScore": EffScoreMetric(),
    #레벨7 메트릭
    "ContextRetention": ContextRetentionMetric(),
    "RefRecall": RefRecallMetric(),
}

def get_metrics_for_level(level: int) -> Dict[str, Metric]:
    """특정 레벨에 해당하는 메트릭들을 반환 (공통 메트릭 포함)
    
    Args:
        level: 레벨 번호 (1-7)
        
    Returns:
        레벨에 맞는 메트릭 딕셔너리
    """
    filtered = {}
    for name, metric in METRICS.items():
        # 공통 메트릭이거나 해당 레벨의 메트릭인 경우
        if metric.level == "common" or metric.level == level:
            filtered[name] = metric
    return filtered

def get_common_metrics() -> Dict[str, Metric]:
    """공통 메트릭만 반환"""
    return {name: metric for name, metric in METRICS.items() if metric.level == "common"}

def get_level_specific_metrics(level: int) -> Dict[str, Metric]:
    """특정 레벨 전용 메트릭만 반환 (공통 메트릭 제외)"""
    return {name: metric for name, metric in METRICS.items() if metric.level == level}


