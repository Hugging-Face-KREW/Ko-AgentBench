"""Evaluation metrics for Ko-AgentBench."""

import json
import logging
import yaml
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from ..adapters.base_adapter import BaseAdapter

# Pydantic 모델 정의
class SRResponse(BaseModel):
    """SR 메트릭 응답 모델 - success/fail 이진 판정 (reason first)"""
    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
            "strict": False  # Gemini 호환성을 위해 strict mode 비활성화
        }
    }

    reason: str = Field(..., description="평가 이유 (먼저 작성)")
    success: bool = Field(..., description="최종 판정 (이유를 바탕으로)")

class ArgAccResponse(BaseModel):
    """ArgAcc 메트릭 응답 모델"""
    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
            "strict": False
        }
    }
    
    score: int = Field(..., ge=1, le=5, description="1-5점 척도 점수")
    confidence: float = Field(..., ge=0.0, le=1.0, description="신뢰도 0.0-1.0")
    reason: str = Field(..., description="각 인수별 유사도 분석 결과")

class EffScoreResponse(BaseModel):
    """EffScore 메트릭 응답 모델"""
    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
            "strict": False
        }
    }
    
    success: bool = Field(..., description="성공 여부")
    reason: str = Field(..., description="간단한 이유")

class ContextRetentionResponse(BaseModel):
    """ContextRetention 메트릭 응답 모델"""
    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
            "strict": False
        }
    }
    
    score: int = Field(..., ge=1, le=5, description="1-5점 척도 점수")
    reason: str = Field(..., description="간단한 이유")

class RefRecallResponse(BaseModel):
    """RefRecall 메트릭 응답 모델"""
    model_config = {
        "json_schema_extra": {
            "additionalProperties": False,
            "strict": False
        }
    }
    
    score: int = Field(..., ge=1, le=5, description="1-5점 척도 점수")
    reason: str = Field(..., description="간단한 이유")

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


def extract_value_by_path(data: Any, path: str) -> Any:
    """
    점 표기법과 배열 인덱싱을 사용하여 중첩된 데이터에서 값 추출.
    
    예시:
    - "items[0].title" -> data["items"][0]["title"]
    - "route.traoptimal[0].summary.duration" -> data["route"]["traoptimal"][0]["summary"]["duration"]
    
    Args:
        data: 중첩된 dict/list 데이터
        path: 점 표기법 경로 (예: "items[0].title")
    
    Returns:
        추출된 값, 실패 시 None
    """
    import re
    
    if data is None:
        return None
    
    try:
        current = data
        # 경로를 파싱: items[0].title -> ["items", "[0]", "title"]
        tokens = re.split(r'\.', path)
        
        for token in tokens:
            if not token:
                continue
                
            # 배열 인덱스 처리: items[0] -> "items", 0
            array_match = re.match(r'^(\w+)\[(\d+)\]$', token)
            if array_match:
                key = array_match.group(1)
                index = int(array_match.group(2))
                
                if isinstance(current, dict) and key in current:
                    current = current[key]
                    if isinstance(current, list) and index < len(current):
                        current = current[index]
                    else:
                        return None
                else:
                    return None
            # 단순 배열 인덱스: [0]
            elif re.match(r'^\[(\d+)\]$', token):
                index = int(token[1:-1])
                if isinstance(current, list) and index < len(current):
                    current = current[index]
                else:
                    return None
            # 일반 키
            else:
                if isinstance(current, dict) and token in current:
                    current = current[token]
                else:
                    return None
        
        return current
    except Exception:
        return None


def extract_golden_context(
    action_trace: List[Dict[str, Any]], 
    golden_fields: List[Dict[str, Any]]
) -> str:
    """
    태스크의 golden_fields 정의에 따라 도구 호출 결과에서 핵심 필드만 추출.
    
    Args:
        action_trace: 도구 호출 기록 리스트
            [{"tool": "WebSearch_naver", "result": {...}, "success": True}, ...]
        golden_fields: golden_fields 정의
            [{"tool": "WebSearch_naver", "fields": ["items[0].title", "items[0].description"]}, ...]
    
    Returns:
        추출된 golden context JSON 문자열
        
    Note:
        - 도구 호출 실패나 필드가 없는 경우에도 에러 없이 graceful하게 처리
        - 모델이 잘못된 도구를 호출했더라도 해당 도구 결과만 제외하고 진행
    """
    if not golden_fields:
        # golden_fields가 없으면 기존 방식(100자 truncation)으로 fallback
        return None
    
    extracted_context = []
    
    # golden_fields를 tool별로 그룹화
    tool_to_fields = {}
    for gf in golden_fields:
        tool_name = gf.get("tool")
        fields = gf.get("fields", [])
        if tool_name:
            if tool_name not in tool_to_fields:
                tool_to_fields[tool_name] = []
            tool_to_fields[tool_name].extend(fields)
    
    # 각 도구 호출에서 관련 필드 추출
    for action in action_trace:
        tool_name = action.get("tool")
        result = action.get("result")
        success = action.get("success", False)
        
        # 실패한 호출이나 결과가 없는 경우 스킵
        if not success or result is None:
            continue
        
        # 해당 도구의 golden_fields가 있는지 확인
        if tool_name not in tool_to_fields:
            continue
        
        fields_to_extract = tool_to_fields[tool_name]
        tool_context = {
            "tool": tool_name,
            "extracted_fields": {}
        }
        
        for field_path in fields_to_extract:
            try:
                value = extract_value_by_path(result, field_path)
                if value is not None:
                    tool_context["extracted_fields"][field_path] = value
            except Exception:
                # 필드 추출 실패 시 무시하고 계속 진행
                continue
        
        # 추출된 필드가 있는 경우에만 추가
        if tool_context["extracted_fields"]:
            extracted_context.append(tool_context)
    
    if not extracted_context:
        # 추출된 컨텍스트가 없으면 None 반환 (fallback 처리)
        return None
    
    return json.dumps(extracted_context, ensure_ascii=False, indent=2)


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
    
    def _call_structured_judge(self, adapter: BaseAdapter, system_prompt: str, user_prompt: str,
                              response_model: BaseModel) -> Dict[str, Any]:
        """구조화된 출력을 사용하는 단일 judge 호출"""
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Gemini 모델 감지
            model_name = getattr(adapter, 'model_name', '').lower()
            is_gemini = 'gemini' in model_name
            
            # Gemini 모델에 최적화된 response_format 사용
            if is_gemini:
                response = adapter.chat_completion(
                    messages, 
                    temperature=0.0,
                    max_tokens=2048,  # Judge 평가는 더 많은 토큰 필요 (특히 한국어)
                    response_format={
                        "type": "json_object",
                        "response_schema": response_model.model_json_schema(),
                        "enforce_validation": True  # Gemini JSON 검증 강화
                    }
                )
            else:
                # OpenAI 스타일 모델용
                response = adapter.chat_completion(
                    messages, 
                    temperature=0.0,
                    max_tokens=2048,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "schema": response_model.model_json_schema(),
                            "strict": False
                        }
                    }
                )
            
            content = response.get("message", {}).get("content", "{}")
            
            # Pydantic 모델로 파싱
            parsed_response = response_model.model_validate_json(content)
            return parsed_response.model_dump()
            
        except Exception as e:
            logging.error(f"Structured judge call failed for {getattr(adapter, 'model_name', 'unknown')}: {str(e)}")
            logging.error(f"Raw response content: {content if 'content' in locals() else 'N/A'}")
            
            # 기본값 반환
            try:
                # Pydantic v2에서 model_fields 사용
                if hasattr(response_model, 'model_fields'):
                    defaults = {}
                    for field_name, field_info in response_model.model_fields.items():
                        if field_name == 'score':
                            defaults[field_name] = 1
                        elif field_name == 'confidence':
                            defaults[field_name] = 0.0
                        elif field_name == 'error_reported' or field_name == 'success':
                            defaults[field_name] = False
                        elif field_name == 'reported_error_type':
                            defaults[field_name] = "none"
                        else:
                            defaults[field_name] = f"Error: {str(e)}"
                    return defaults
                else:
                    # 기본 실패 응답
                    return {"error": str(e)}
            except Exception:
                return {"error": str(e)}
    
    def _call_multi_judge_binary(self, prompt: str, metric_key: str, 
                                 response_model: BaseModel,
                                 result_key: str = "error_reported") -> Dict[str, Any]:
        """복수 judge 호출 - 이진 결정 (다수결)
        
        Args:
            prompt: 평가 프롬프트
            metric_key: 시스템 프롬프트 키 (예: 'error_detect')
            response_model: 응답 Pydantic 모델
            result_key: 결과 키 (예: 'error_reported', 'success')
        """
        if hasattr(self, 'llm_adapters') and self.llm_adapters:
            judges_to_use = self.llm_adapters
        elif self.llm_adapter:
            judges_to_use = [self.llm_adapter]
        else:
            return {result_key: False, "confidence": 0.0, "reason": "No LLM adapter available"}
        
        system_prompt = self.prompt_loader.get_system_prompt(metric_key)
        
        all_results = []
        
        for idx, adapter in enumerate(judges_to_use, 1):
            result = self._call_structured_judge(adapter, system_prompt, prompt, response_model)
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
                                response_model: BaseModel,
                                min_score: int = 1, max_score: int = 5) -> Dict[str, Any]:
        """복수 judge 호출 - 점수 평가 (평균)
        
        Args:
            prompt: 평가 프롬프트
            metric_key: 시스템 프롬프트 키
            response_model: 응답 Pydantic 모델
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
        
        all_results = []
        
        for idx, adapter in enumerate(judges_to_use, 1):
            result = self._call_structured_judge(adapter, system_prompt, prompt, response_model)
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
class SRMetric(LLMJudgeMetric):
    """SR(성공률): LLM Judge로 태스크 완수 여부 평가 (단일 모델 다중 추론 + 하드 보팅)"""
    name = "SR"
    level = "common"
    vote_count: int = 3  # 투표 횟수 (기본값)

    def _call_single_model_voting(self, prompt: str, n: int = 3) -> Dict[str, Any]:
        """단일 모델로 n회 추론 후 하드 보팅으로 최종 판정

        Args:
            prompt: 평가 프롬프트
            n: 투표 횟수 (기본값: 3)

        Returns:
            투표 결과 딕셔너리
        """
        if not self.llm_adapter:
            return {
                "success": False,
                "votes": [],
                "vote_count": {"success": 0, "fail": 0},
                "reasons": [],
                "final_reason": "LLM adapter가 설정되지 않음"
            }

        system_prompt = self.prompt_loader.get_system_prompt('sr')
        votes = []
        reasons = []

        for i in range(n):
            try:
                result = self._call_structured_judge(
                    self.llm_adapter,
                    system_prompt,
                    prompt,
                    SRResponse
                )
                votes.append(result.get("success", False))
                reasons.append(result.get("reason", ""))
            except Exception as e:
                logging.error(f"SR voting round {i+1} failed: {str(e)}")
                votes.append(False)
                reasons.append(f"평가 실패: {str(e)}")

        # 하드 보팅: 과반수 판정
        success_count = sum(1 for v in votes if v)
        fail_count = len(votes) - success_count
        final_success = success_count > fail_count

        return {
            "success": final_success,
            "votes": votes,
            "vote_count": {"success": success_count, "fail": fail_count},
            "reasons": reasons,
            "final_reason": f"{success_count}/{len(votes)} 투표로 {'성공' if final_success else '실패'} 판정"
        }

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        """LLM Judge를 사용하여 태스크 완수 여부 평가 (단일 모델 다중 추론)"""

        # 필요한 정보 추출
        instruction = ctx.task_schema.get("instruction", "")
        final_response = ctx.logs.get("actual_output") or ctx.logs.get("final_response", "")

        # final_response 검증
        if final_response is None:
            final_response = ""
        elif not isinstance(final_response, str):
            final_response = str(final_response)

        # L5 예외 처리: fallback 옵션이 있는데 한 번도 호출하지 않았다면 LLM Judge를 건너뛰고 0점 처리
        def _is_level5(level_value: Any) -> bool:
            if level_value is None:
                return False
            if isinstance(level_value, int):
                return level_value == 5
            if isinstance(level_value, str):
                normalized = level_value.strip().upper()
                return normalized == "L5" or normalized == "5"
            return False

        task_level = ctx.task_schema.get("task_level")
        if task_level is None:
            task_level = ctx.task_schema.get("level")

        fallback_options = ctx.task_schema.get("fallback_options") or []
        fallback_tools = {
            opt.get("tool") for opt in fallback_options
            if isinstance(opt, dict) and opt.get("tool")
        }

        if _is_level5(task_level) and fallback_tools:
            tool_calls = ctx.logs.get("tool_calls") or ctx.action_trace or []
            used_fallback = False
            for call in tool_calls:
                tool_name = None
                if isinstance(call, dict):
                    tool_name = call.get("tool_name") or call.get("tool")
                if tool_name in fallback_tools:
                    used_fallback = True
                    break

            if not used_fallback:
                return EvaluationResult(
                    self.name,
                    0.0,
                    {
                        "success": False,
                        "votes": [],
                        "vote_count": {"success": 0, "fail": 0},
                        "reasons": ["L5 task with fallback options but none were invoked"],
                        "final_reason": "L5 task with fallback options but none were invoked"
                    }
                )

        # 빈 응답 체크
        if not final_response or len(final_response.strip()) < 3:
            return EvaluationResult(
                self.name,
                0.0,
                {
                    "success": False,
                    "votes": [],
                    "vote_count": {"success": 0, "fail": 0},
                    "reasons": ["응답이 생성되지 않음"],
                    "final_reason": "응답이 생성되지 않음"
                }
            )

        # 도구 호출 정보 수집 (응답은 golden_context에서 제공하므로 호출 정보만 포함)
        tool_calls = []
        for action in ctx.action_trace:
            tool_calls.append({
                "tool": action.get("tool"),
                "args": action.get("args", {}),
                "success": action.get("success", False)
            })

        # golden_fields가 있으면 핵심 컨텍스트 추출, 없으면 기본 tool_calls 사용
        golden_fields = ctx.task_schema.get("golden_fields", [])
        golden_context = extract_golden_context(ctx.action_trace, golden_fields)

        # LLM Judge 프롬프트 구성
        prompt_template = self.prompt_loader.get_prompt('sr')

        # golden_context가 있으면 프롬프트에 포함
        if golden_context:
            prompt = prompt_template.format(
                instruction=instruction,
                final_response=final_response,
                tool_calls=json.dumps(tool_calls, ensure_ascii=False, indent=2),
                golden_context=golden_context
            )
        else:
            # golden_context 플레이스홀더가 없는 기존 템플릿과의 호환성
            try:
                prompt = prompt_template.format(
                    instruction=instruction,
                    final_response=final_response,
                    tool_calls=json.dumps(tool_calls, ensure_ascii=False, indent=2),
                    golden_context="[golden_context 없음 - 기본 result_preview 사용]"
                )
            except KeyError:
                # golden_context 플레이스홀더가 템플릿에 없는 경우
                prompt = prompt_template.format(
                    instruction=instruction,
                    final_response=final_response,
                    tool_calls=json.dumps(tool_calls, ensure_ascii=False, indent=2)
                )

        # 단일 모델 다중 추론 + 하드 보팅
        voting_result = self._call_single_model_voting(prompt, n=self.vote_count)

        # success=True면 1.0, False면 0.0
        score = 1.0 if voting_result["success"] else 0.0

        return EvaluationResult(
            self.name,
            score,
            {
                "success": voting_result["success"],
                "votes": voting_result["votes"],
                "vote_count": voting_result["vote_count"],
                "reasons": voting_result["reasons"],
                "final_reason": voting_result["final_reason"]
            }
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
        
        
class ArgAccMetric(LLMJudgeMetric):
    """
    ArgAcc: 도구 인자 정확도. (LLM Judge)
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
        # tool_invocations와 tool_calls 둘 다 확인
        invocations = ctx.logs.get("tool_invocations", []) or []
        tool_calls = ctx.logs.get("tool_calls", []) or []
        
        # tool_invocations에서 먼저 찾기
        for inv in invocations:
            tool = inv.get("tool") or inv.get("tool_name")
            if tool == golden_tool:
                args = inv.get("arguments")
                if args is None:
                    args = inv.get("args")
                return args or {}
        
        # tool_calls에서 찾기
        for call in tool_calls:
            tool = call.get("tool_name")
            if tool == golden_tool:
                args = call.get("arguments")
                if args is None:
                    args = call.get("args")
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
        # 기존 PRF1 계산
        prf_result = self._compute_prf(ctx)
        if not prf_result.get("ok"):
            return EvaluationResult(self.name, 0.0, {"ok": False})

        # LLM Judge를 사용한 의미적 유사성 평가 
        llm_score = self._evaluate_with_llm_judge(ctx)

        details = {
            "ok": True, 
            "precision": prf_result["precision"], 
            "recall": prf_result["recall"], 
            "f1": prf_result["f1"],
            "llm_judge_score": llm_score
        }
        return EvaluationResult(self.name, llm_score, details)
    
    def _evaluate_with_llm_judge(self, ctx: EvalContext) -> float:
        """LLM Judge를 사용하여 인수 유사성 평가"""
        try:
            golden_actions = ctx.task_schema.get("golden_action", [])
            if isinstance(golden_actions, dict):
                golden_actions = [golden_actions]

            if not golden_actions or not isinstance(golden_actions[0], dict):
                return 0.0

            golden_tool = golden_actions[0].get("tool")
            golden_args = golden_actions[0].get("args", {}) or {}
            pred_args = self._get_first_pred_args_for_tool(ctx, golden_tool)

            if not golden_args or not pred_args:
                return 0.0

            # 프롬프트 템플릿 사용
            template = self.prompt_loader.get_prompt("arg_acc")
            prompt = template.format(
                golden_args=golden_args,
                predicted_args=pred_args
            )
            
            result = self._call_multi_judge_score(prompt, 'arg_acc', ArgAccResponse, min_score=1, max_score=5)
            
            # 1-5 점수를 0.0-1.0 범위로 변환 
            score = result.get("score", 1)
            normalized_score = max(0.0, (score - 1) / 4.0)  # 1->0.0, 5->1.0, 음수 방지
            
            print(f"DEBUG ArgAcc: score={score}, normalized_score={normalized_score}, result={result}")
            
            return normalized_score
            
        except Exception as e:
            print(f"LLM Judge 평가 중 오류: {e}")
            return 0.0


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
        # final_response를 확인
        cand = ctx.logs.get("final_response")

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
class AdaptiveRoutingScoreMetric(Metric):
    """
    AdaptiveRoutingScore : 주입 오류 발생 후 대체 경로로 전환하는 속도를 평가
    """
    name = "AdaptiveRoutingScore"
    level = 5

    def evaluate(self, ctx: EvalContext) -> EvaluationResult:
        error_injection = ctx.task_schema.get("error_injection") or {}
        injected_tool = error_injection.get("tool")

        if not injected_tool:
            return EvaluationResult(self.name, None, {"reason": "No error injection"})

        tool_calls = ctx.logs.get("tool_calls") or []
        if not tool_calls:
            return EvaluationResult(self.name, 0.0, {
                "reason": "No tool calls recorded",
                "injected_tool": injected_tool
            })

        fallback_options = ctx.task_schema.get("fallback_options") or []
        fallback_tools = {
            opt.get("tool") for opt in fallback_options
            if isinstance(opt, dict) and opt.get("tool")
        }

        # failure 및 fallback 단계 탐색
        failure_index = None
        failure_step = None
        fallback_index = None
        fallback_step = None
        fallback_tool_used = None

        def _as_int(value: Any) -> Optional[int]:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        for idx, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                continue

            tool_name = call.get("tool_name") or call.get("tool")
            step = _as_int(call.get("step"))
            success = call.get("success")
            error = call.get("error")

            if failure_index is None and tool_name == injected_tool and (success is False or error):
                failure_index = idx
                failure_step = step
                continue

            if failure_index is None:
                continue

            if fallback_tools:
                is_fallback = tool_name in fallback_tools
            else:
                is_fallback = tool_name is not None and tool_name != injected_tool

            if is_fallback:
                fallback_index = idx
                fallback_step = step
                fallback_tool_used = tool_name
                break

        if failure_index is None:
            return EvaluationResult(self.name, 0.0, {
                "reason": "Injected tool did not fail",
                "injected_tool": injected_tool
            })

        if fallback_index is None:
            return EvaluationResult(self.name, 0.0, {
                "reason": "No fallback attempt after failure",
                "failure_step": failure_step,
                "injected_tool": injected_tool
            })

        # 단계 간격 계산 (step 값 우선, 없으면 index 기준)
        if failure_step is not None and fallback_step is not None:
            step_gap = max(0, fallback_step - failure_step - 1)
        else:
            step_gap = max(0, fallback_index - failure_index - 1)

        score = 1.0 / (1 + step_gap)

        return EvaluationResult(self.name, score, {
            "failure_step": failure_step,
            "fallback_step": fallback_step,
            "fallback_tool": fallback_tool_used,
            "step_gap": step_gap,
            "injected_tool": injected_tool,
            "fallback_candidates": list(fallback_tools) if fallback_tools else None
        })


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
        llm_result = self._call_multi_judge_binary(prompt, 'eff_score', EffScoreResponse, 'success')
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
        llm_result = self._call_multi_judge_score(prompt, 'context_retention', ContextRetentionResponse, 1, 5)
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
        llm_result = self._call_multi_judge_score(prompt, 'ref_recall', RefRecallResponse, 1, 5)
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
    #레벨4 메트릭
    "Coverage": CoverageMetric(),
    "SourceEPR": SourceEPRMetric(),
    #레벨5 메트릭 
    "AdaptiveRoutingScore": AdaptiveRoutingScoreMetric(),
    "FallbackSR": FallbackSRMetric(),
    #레벨6 메트릭
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
