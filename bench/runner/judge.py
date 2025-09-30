"""Judge for evaluating benchmark results."""

import json
import re
import logging
from typing import Any, Dict, List, Optional, Callable
from ..adapters.base_adapter import BaseAdapter
from ..observability import observe, get_client, is_enabled


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
            
            final_result = {
                "success": evaluation_result.get('success', False),
                "score": evaluation_result.get('score', 0.0),
                "oracle_type": oracle_type,
                "expected_output": expected_output,
                "actual_output": actual_output,
                "details": evaluation_result.get('details', {}),
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
        """Evaluate based on golden_action(수정 필요)"""

        if not isinstance(expected, list):
            expected = [expected]

        expected_tools = [e.get("tool") for e in expected if isinstance(e, dict)]
        actual_str = str(actual) + " " + str(task)

        matched = [tool for tool in expected_tools if tool and tool in actual_str]
        success = len(matched) == len(expected_tools)

        return {
            "success": success,
            "score": 1.0 if success else 0.0,
            "details": {
                "match_type": "golden_action",
                "expected_tools": expected_tools,
                "matched_tools": matched,
                "note": "Success if expected tool keyword found in actual/task strings"
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