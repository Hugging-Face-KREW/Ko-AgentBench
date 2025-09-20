import os
from typing import Any
from dotenv import load_dotenv

from bench.tools.tool_registry import ToolRegistry
from bench.tools.naver_search_mock import (
    NaverWebSearchMock,
    NaverBlogSearchMock,
    NaverNewsSearchMock,
)
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.runner import BenchmarkRunner, Judge
from bench.tasks.task_loader import TaskLoader


def create_default_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-registered with default mock tools."""
    registry = ToolRegistry()
    registry.register_tool(NaverWebSearchMock)
    registry.register_tool(NaverBlogSearchMock)
    registry.register_tool(NaverNewsSearchMock)
    return registry


def run_tool_calling_demo():
    """tool calling ruuner demo"""
    load_dotenv()
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("HUGGINGFACE_API_KEY를 .env 파일에 설정")
        return
    print("환경 설정 확인 완료")
    
    # JSONL 샘플 네이버 태스크 로드
    task_loader = TaskLoader()
    tasks = task_loader.load_tasks("sample_naver_tasks.jsonl")
    
    # 컴포넌트 설정
    registry = create_default_tool_registry()
    adapter = LiteLLMAdapter("huggingface/Qwen/Qwen3-4B-Instruct-2507")
    judge = Judge(llm_adapter=adapter)
    runner = BenchmarkRunner(adapter, registry, judge, max_steps=3, timeout=30)

    # 첫 번째 태스크 실행
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)}: {task['id']} ---")
        print(f"설명: {task['description']}")
        
        try:
            result = runner.run_task(task)
            print(f"성공 여부: {result['success']}")
            print(f"실행시간: {result['execution_time']:.2f}초")
            print(f"단계수: {result['steps_taken']}")
            
            if result['result'] and result['result'].get('final_response'):
                response = result['result']['final_response']
                print(f"응답: {response}")
                
        except Exception as e:
            print(f"실행 실패: {e}")


def main() -> None:
    print("Hello from ko-agentbench!")
    registry = create_default_tool_registry()
    print("Registered tools:", registry.get_available_tools())
    # Sample call for quick verification (safe, no network calls)
    sample: Any = registry.execute_tool("naver_web_search", query="테스트", display=2, start=1)
    print("Sample web search result (truncated items count):", len(sample.get("items", [])))
    run_tool_calling_demo()


if __name__ == "__main__":
    main()
