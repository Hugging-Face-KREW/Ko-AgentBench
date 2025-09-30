import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Dict
from dotenv import load_dotenv

from bench.tools.tool_registry import ToolRegistry
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.runner import BenchmarkRunner, Judge
from bench.tasks.task_loader import TaskLoader
from bench.tools.base_api import BaseTool
from bench.models import MODEL_IDS
from bench.tools.tool_catalog import resolve_tool_classes, TOOL_CATALOG
from bench.observability import log_status


def create_default_tool_registry(tool_classes: Optional[List[Type[BaseTool]]] = None) -> ToolRegistry:
    """Create a ToolRegistry and register provided tool classes.

    Note: tool_classes가 None이면 빈 레지스트리를 반환합니다.
    """
    registry = ToolRegistry()
    if tool_classes:
        for tool_class in tool_classes:
            registry.register_tool(tool_class)
    else:
        # 기본값이 없으면 빈 레지스트리로 시작 (호출 측에서 관리)
        pass

    return registry

def save_results_to_json(results: List[Dict[str, Any]], model_name: str, output_dir: str = "logs") -> str:
    """Save benchmark results to JSON file.
    
    Args:
        results: List of task results
        model_name: Name of the model used
        output_dir: Directory to save logs
        
    Returns:
        Path to saved JSON file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("/", "_")
    filename = f"benchmark_{model_safe_name}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r.get('success', False))
    total_time = sum(r.get('execution_time', 0) for r in results)
    total_steps = sum(r.get('steps_taken', 0) for r in results)
    total_tool_calls = sum(len(r.get('tool_invocations', [])) for r in results)
    
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": round(total_time, 2),
            "average_execution_time": round(total_time / total_tasks, 2) if total_tasks > 0 else 0,
            "total_steps": total_steps,
            "total_tool_calls": total_tool_calls,
        },
        "results": results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def run_tool_calling_demo(
    tool_classes: Optional[List[Type[BaseTool]]] = None,
    model_name: str = "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    save_logs: bool = True,  
    log_dir: str = "logs",   
    **adapter_config: Any,
):
    """Tool-calling runner demo with configurable tools and model.

    Args:
        tool_classes: 등록할 툴 클래스 목록. None이면 빈 레지스트리 사용
        model_name: LiteLLM이 인식하는 모델 식별자 (예: 'openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet', 'groq/gemma-7b-it')
        save_logs: JSON 로그 저장 여부 
        log_dir: 로그 저장 디렉토리 
        **adapter_config: LiteLLMAdapter 추가 설정값(temperature, max_tokens 등)
    """
    provider_keys = [
        "HUGGINGFACE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
    ]
    if not any(os.getenv(k) for k in provider_keys):
        print("경고: 주요 LLM API 키가 설정되어 있지 않을 수 있습니다. 선택한 모델에 맞는 키를 .env에 설정하세요.")
    
    # JSONL 샘플 네이버 태스크 로드
    task_loader = TaskLoader()
    tasks = task_loader.load_tasks("sample_naver_tasks.jsonl")
    
    # 태스크에 명시된 tools를 기준으로 필요한 툴 클래스를 자동 등록
    if tool_classes is None:
        requested: List[str] = []
        for t in tasks:
            for name in t.get("tools", []) or []:
                if name not in requested:
                    requested.append(name)

        tool_classes = resolve_tool_classes(requested)
        missing = [n for n in requested if n not in TOOL_CATALOG]
        if missing:
            print(f"경고: 등록할 수 없는 툴 이름이 있습니다: {missing}")

    # 컴포넌트 설정
    registry = create_default_tool_registry(tool_classes)
    adapter = LiteLLMAdapter(model_name, **adapter_config)
    judge = Judge(llm_adapter=adapter)
    runner = BenchmarkRunner(adapter, registry, judge, max_steps=3, timeout=30)
    all_results = []

    # 태스크 실행
    for i, task in enumerate(tasks, 1):
        print(f"\n--- Task {i}/{len(tasks)}: {task['id']} ---")
        print(f"설명: {task['description']}")
        
        try:
            result = runner.run_task(task)
            all_results.append(result)
            print(f"성공 여부: {result['success']}")
            print(f"실행시간: {result['execution_time']:.2f}초")
            print(f"단계수: {result['steps_taken']}")
            # 툴 호출 요약 출력
            inv = result.get('tool_invocations', []) or []
            if inv:
                print(f"툴 호출 수: {len(inv)}")
                for t in inv:
                    print(f"  - step={t.get('step')}, tool={t.get('tool_name')}, success={t.get('success')}")
                    print(f"    args={t.get('arguments')}")
                    # 응답은 길 수 있어 첫 200자만 미리보기
                    preview = str(t.get('result'))
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    print(f"    result_preview={preview}")
            
            if result['result'] and result['result'].get('final_response'):
                response = result['result']['final_response']
                print(f"응답: {response}")
                
        except Exception as e:
            print(f"실행 실패: {e}")
            all_results.append({
                "task_id": task.get('id', 'unknown'),
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "steps_taken": 0,
                "tool_invocations": []
            })
    
    if save_logs and all_results:
        try:
            filepath = save_results_to_json(all_results, model_name, log_dir)
            print(f"\n{'='*60}")
            print(f"결과가 JSON 파일로 저장되었습니다")
            print(f"파일 위치: {filepath}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"\nJSON 저장 실패: {e}\n")


def main() -> None:
    print("Hello from ko-agentbench!")
    load_dotenv()
    log_status()
    print("="*50 + "\n")
    
    registry = create_default_tool_registry()
    print("Registered tools:", registry.get_available_tools())
    # Sample call for quick verification (safe, no network calls)
    # 사용 가능한 모델 id 목록 출력
    print("Available models:", MODEL_IDS)
    
    # 샘플: 레지스트리에 툴이 있을 때만 샘플 호출
    if registry.get_available_tools():
        sample: Any = registry.execute_tool("naver_web_search", query="테스트", display=2, start=1)
        print("Sample web search result (truncated items count):", len(sample.get("items", [])))
    # 예시: 다양한 모델 식별자 사용 가능 (OpenAI/Anthropic/Groq/HF 등)
    # run_tool_calling_demo(model_name="openai/gpt-4o-mini")
    # run_tool_calling_demo(model_name="anthropic/claude-3-5-sonnet")
    # run_tool_calling_demo(model_name="groq/gemma-7b-it")
    selected_model = MODEL_IDS[-1] if MODEL_IDS else ""
    if selected_model:
        print(f"선택된 모델: {selected_model}")
        run_tool_calling_demo(
            model_name=selected_model,
            save_logs=True, 
            log_dir="logs", 
        )


if __name__ == "__main__":
    main()
