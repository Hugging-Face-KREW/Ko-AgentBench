"""Simple test script for BenchmarkRunner functionality."""

import json
import logging
from typing import Any, Dict, List, Optional

from .run import BenchmarkRunner
from .judge import Judge
from ..adapters.base_adapter import BaseAdapter
from ..tools.base_api import BaseTool
from ..tools.tool_registry import ToolRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Mock Adapter for testing
class MockAdapter(BaseAdapter):
    """Mock LLM adapter for testing."""
    
    def __init__(self, model_name: str = "mock-model", **config):
        super().__init__(model_name, **config)
        self.call_count = 0
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Simulate chat completion."""
        self.call_count += 1
        
        # First call: simulate tool call
        if self.call_count == 1 and tools:
            return {
                "model": self.model_name,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"operation": "add", "a": 5, "b": 3})
                            }
                        }
                    ]
                },
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }
        
        # Second call: simulate final response
        return {
            "model": self.model_name,
            "message": {
                "role": "assistant",
                "content": "계산 결과는 8입니다."
            },
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150
            }
        }
    
    def convert_to_provider_format(self, messages: List[Dict[str, str]], 
                                  tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Mock conversion - no-op."""
        return {"messages": messages, "tools": tools}
    
    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Mock conversion - no-op."""
        return response


# Mock Tool for testing
class CalculatorTool(BaseTool):
    """Simple calculator tool for testing."""
    
    def __init__(self):
        super().__init__(
            name="calculator",
            description="간단한 계산을 수행합니다. 덧셈, 뺄셈, 곱셈, 나눗셈을 지원합니다."
        )
    
    def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """Execute calculation."""
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
        }
        
        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}
        
        result = operations[operation](a, b)
        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": result
        }
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters."""
        required = ["operation", "a", "b"]
        if not all(k in kwargs for k in required):
            return False
        
        valid_ops = ["add", "subtract", "multiply", "divide"]
        if kwargs.get("operation") not in valid_ops:
            return False
        
        try:
            float(kwargs.get("a"))
            float(kwargs.get("b"))
            return True
        except (ValueError, TypeError):
            return False
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "수행할 계산 종류"
                },
                "a": {
                    "type": "number",
                    "description": "첫 번째 숫자"
                },
                "b": {
                    "type": "number",
                    "description": "두 번째 숫자"
                }
            },
            "required": ["operation", "a", "b"]
        }


def create_test_task() -> Dict[str, Any]:
    """Create a simple test task."""
    return {
        "id": "test_001",
        "category": "math",
        "difficulty": "easy",
        "description": "5와 3을 더하세요.",
        "tools": ["calculator"],
        "expected_output": {"operation": "add", "a": 5, "b": 3, "result": 8},
        "oracle": "exact_match"
    }


def run_simple_test():
    """Run a simple test of the BenchmarkRunner."""
    print("=" * 60)
    print("Ko-AgentBench Runner 간단 테스트")
    print("=" * 60)
    
    # 1. Setup components
    print("\n1. 컴포넌트 초기화 중...")
    adapter = MockAdapter(model_name="test-gpt-4")
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool)
    judge = Judge(llm_adapter=adapter)
    
    print("   ✓ Adapter 초기화 완료")
    print("   ✓ Tool Registry 초기화 완료")
    print(f"   ✓ 등록된 도구: {tool_registry.get_available_tools()}")
    print("   ✓ Judge 초기화 완료")
    
    # 2. Create runner
    print("\n2. BenchmarkRunner 생성 중...")
    runner = BenchmarkRunner(
        adapter=adapter,
        tool_registry=tool_registry,
        judge=judge,
        max_steps=5,
        timeout=60,
        max_retries=3
    )
    print("   ✓ Runner 생성 완료")
    
    # 3. Create test task
    print("\n3. 테스트 태스크 생성 중...")
    task = create_test_task()
    print(f"   ✓ Task ID: {task['id']}")
    print(f"   ✓ Description: {task['description']}")
    print(f"   ✓ Expected Output: {task['expected_output']}")
    
    # 4. Run task
    print("\n4. 태스크 실행 중...")
    result = runner.run_task(task)
    
    # 5. Display results
    print("\n" + "=" * 60)
    print("실행 결과")
    print("=" * 60)
    print(f"Task ID: {result['task_id']}")
    print(f"성공 여부: {'✓ 성공' if result['success'] else '✗ 실패'}")
    print(f"실행 시간: {result['execution_time']:.2f}초")
    print(f"수행 단계 수: {result['steps_taken']}")
    
    if result.get('error'):
        print(f"\n에러: {result['error']}")
    
    # Tool invocations
    print("\n도구 호출 내역:")
    if result.get('tool_invocations'):
        for i, invocation in enumerate(result['tool_invocations'], 1):
            print(f"\n  [{i}] {invocation['tool_name']}")
            print(f"      인자: {invocation['arguments']}")
            print(f"      결과: {invocation['result']}")
            print(f"      성공: {'✓' if invocation['success'] else '✗'}")
            if invocation.get('error'):
                print(f"      에러: {invocation['error']}")
    else:
        print("  (없음)")
    
    # Evaluation
    if result.get('evaluation'):
        eval_result = result['evaluation']
        print(f"\n평가 결과:")
        print(f"  성공: {'✓' if eval_result['success'] else '✗'}")
        print(f"  점수: {eval_result['score']:.2f}")
        print(f"  평가 방식: {eval_result['oracle_type']}")
        print(f"  기대값: {eval_result['expected_output']}")
        print(f"  실제값: {eval_result['actual_output']}")
    
    # Final response
    if result.get('result'):
        print(f"\n최종 응답:")
        print(f"  {result['result'].get('final_response', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    
    return result


def run_multiple_tasks_test():
    """Run multiple tasks to test the runner."""
    print("\n\n" + "=" * 60)
    print("다중 태스크 테스트")
    print("=" * 60)
    
    # Setup
    adapter = MockAdapter()
    tool_registry = ToolRegistry()
    tool_registry.register_tool(CalculatorTool)
    judge = Judge(llm_adapter=adapter)
    runner = BenchmarkRunner(
        adapter=adapter,
        tool_registry=tool_registry,
        judge=judge,
        max_steps=5
    )
    
    # Create multiple tasks
    tasks = [
        {
            "id": "test_add",
            "description": "10과 20을 더하세요.",
            "tools": ["calculator"],
            "expected_output": {"result": 30},
            "oracle": "exact_match"
        },
        {
            "id": "test_multiply",
            "description": "7과 8을 곱하세요.",
            "tools": ["calculator"],
            "expected_output": {"result": 56},
            "oracle": "exact_match"
        }
    ]
    
    results = []
    for task in tasks:
        print(f"\n실행 중: {task['id']} - {task['description']}")
        result = runner.run_task(task)
        results.append(result)
        print(f"  결과: {'✓ 성공' if result['success'] else '✗ 실패'}")
    
    # Summary
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    success_count = sum(1 for r in results if r['success'])
    print(f"총 태스크: {len(results)}")
    print(f"성공: {success_count}")
    print(f"실패: {len(results) - success_count}")
    print(f"성공률: {success_count / len(results) * 100:.1f}%")
    
    return results


if __name__ == "__main__":
    # Run simple test
    result = run_simple_test()
    
    # Optionally run multiple tasks test
    # results = run_multiple_tasks_test()
    
    # Save result to file
    output_file = "test_result.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n결과가 '{output_file}'에 저장되었습니다.")
