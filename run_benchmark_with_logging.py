"""
Ko-AgentBench 데이터셋 기반 벤치마크 실행 및 Tool Call 로깅 스크립트

data 폴더의 L1~L7 데이터셋을 로드하여 각 질문(instruction)을 실행하고,
사용된 tool call 정보를 상세하게 기록합니다.
"""

import os
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Dict
from dotenv import load_dotenv

from bench.tools.tool_registry import ToolRegistry
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.runner import BenchmarkRunner, Judge
from bench.tools.base_api import BaseTool
from bench.models import MODEL_IDS
from bench.tools.tool_catalog import resolve_tool_classes, TOOL_CATALOG, normalize_tool_name


def load_benchmark_datasets(data_dir: str = "data") -> Dict[str, List[Dict]]:
    """Load all benchmark datasets from data directory.
    
    Args:
        data_dir: Directory containing L1.json ~ L7.json files
        
    Returns:
        Dictionary mapping level names to task lists
    """
    datasets = {}
    json_files = sorted(glob.glob(os.path.join(data_dir, "L*.json")))
    
    for filepath in json_files:
        level_name = Path(filepath).stem  # L1, L2, etc.
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                datasets[level_name] = data
                print(f"✓ Loaded {level_name}: {len(data)} tasks")
        except Exception as e:
            print(f"✗ Failed to load {filepath}: {e}")
    
    return datasets


def create_tool_registry(tool_classes: Optional[List[Type[BaseTool]]] = None) -> ToolRegistry:
    """Create a ToolRegistry and register provided tool classes."""
    registry = ToolRegistry()
    if tool_classes:
        for tool_class in tool_classes:
            registry.register_tool(tool_class)
    return registry


def convert_dataset_to_tasks(dataset_tasks: List[Dict]) -> List[Dict]:
    """Convert benchmark dataset format to runner-compatible task format.
    
    Args:
        dataset_tasks: List of tasks from L*.json files
        
    Returns:
        List of tasks in runner format
    """
    converted_tasks = []
    
    # Handle case where dataset_tasks might be wrapped in an extra list
    if isinstance(dataset_tasks, list) and len(dataset_tasks) > 0 and isinstance(dataset_tasks[0], list):
        dataset_tasks = dataset_tasks[0]
    
    for task in dataset_tasks:
        # Skip if task is not a dict
        if not isinstance(task, dict):
            print(f"⚠ Warning: Skipping non-dict task: {type(task)}")
            continue
            
        # Extract required tools from golden_action
        tools_needed = []
        if "golden_action" in task:
            for action in task["golden_action"]:
                tool_name = action.get("tool", "")
                if tool_name and tool_name not in tools_needed:
                    tools_needed.append(tool_name)
        
        converted_task = {
            "id": task.get("task_id", "unknown"),
            "description": task.get("instruction", ""),
            "expected_output": task.get("resp_schema", {}),
            "tools": tools_needed,
            "level": task.get("task_level", 0),
            "category": task.get("task_category", "unknown"),
            "golden_action": task.get("golden_action", []),
            "arg_schema": task.get("arg_schema", {})
        }
        converted_tasks.append(converted_task)
    
    return converted_tasks


def simplify_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify and flatten a single task result for easier analysis.
    
    Args:
        result: Original task result
        
    Returns:
        Simplified, flattened result
    """
    simplified = {
        "task_id": result.get("task_id", "unknown"),
        "instruction": result.get("instruction", ""),
        "level": result.get("level", 0),
        "category": result.get("category", "unknown"),
        "success": result.get("success", False),
        "execution_time": result.get("execution_time", 0),
        "steps_taken": result.get("steps_taken", 0),
        "error": result.get("error"),
        "expected_tools": result.get("expected_tools", []),
        "golden_action": result.get("golden_action", []),
    }
    
    # Extract tool calls in a simplified format with results
    tool_calls = []
    for invocation in result.get("tool_invocations", []):
        tool_call = {
            "step": invocation.get("step"),
            "tool_name": invocation.get("tool_name"),
            "arguments": invocation.get("arguments"),
            "success": invocation.get("success"),
            "error": invocation.get("error")
        }
        
        # Include API result if available and successful
        if invocation.get("success") and invocation.get("result"):
            tool_call["result"] = invocation["result"]
        
        tool_calls.append(tool_call)
    
    simplified["tool_calls"] = tool_calls
    
    # Extract final response
    if result.get("result") and result["result"].get("final_response"):
        simplified["final_response"] = result["result"]["final_response"]
    else:
        simplified["final_response"] = None
    
    return simplified


def save_detailed_results(
    results: List[Dict[str, Any]], 
    model_name: str, 
    level_name: str,
    output_dir: str = "logs/benchmark_results"
) -> str:
    """Save detailed benchmark results including tool call information.
    
    Args:
        results: List of task results with tool call details
        model_name: Name of the model used
        level_name: Dataset level (L1, L2, etc.)
        output_dir: Directory to save logs
        
    Returns:
        Path to saved JSON file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = model_name.replace("/", "_")
    filename = f"{level_name}_{model_safe_name}_{timestamp}.json"
    filepath = Path(output_dir) / filename
    
    # Simplify and flatten results
    simplified_results = [simplify_result(r) for r in results]
    
    # Calculate statistics
    total_tasks = len(simplified_results)
    successful_tasks = sum(1 for r in simplified_results if r.get('success', False))
    total_time = sum(r.get('execution_time', 0) for r in simplified_results)
    total_steps = sum(r.get('steps_taken', 0) for r in simplified_results)
    total_tool_calls = sum(len(r.get('tool_calls', [])) for r in simplified_results)
    
    # Tool usage statistics
    tool_usage_stats = {}
    for result in simplified_results:
        for tool_call in result.get('tool_calls', []):
            tool_name = tool_call.get('tool_name', 'unknown')
            if tool_name not in tool_usage_stats:
                tool_usage_stats[tool_name] = {
                    'count': 0,
                    'success': 0,
                    'failure': 0
                }
            tool_usage_stats[tool_name]['count'] += 1
            if tool_call.get('success', False):
                tool_usage_stats[tool_name]['success'] += 1
            else:
                tool_usage_stats[tool_name]['failure'] += 1
    
    log_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "level": level_name,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "success_rate": round(successful_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0,
            "total_execution_time": round(total_time, 2),
            "average_execution_time": round(total_time / total_tasks, 2) if total_tasks > 0 else 0,
            "total_steps": total_steps,
            "average_steps": round(total_steps / total_tasks, 2) if total_tasks > 0 else 0,
            "total_tool_calls": total_tool_calls,
            "average_tool_calls": round(total_tool_calls / total_tasks, 2) if total_tasks > 0 else 0,
        },
        "tool_usage_statistics": tool_usage_stats,
        "results": simplified_results
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    return str(filepath)


def run_benchmark_on_dataset(
    level_name: str,
    tasks: List[Dict],
    model_name: str = "gpt-4.1",
    max_steps: int = 10,  
    timeout: int = 60,
    save_logs: bool = True,
    log_dir: str = "logs/benchmark_results",
    **adapter_config: Any
) -> List[Dict[str, Any]]:
    """Run benchmark on a specific dataset level.
    
    Args:
        level_name: Dataset level name (L1, L2, etc.)
        tasks: List of tasks to execute
        model_name: LLM model identifier
        max_steps: Maximum steps per task
        timeout: Timeout per task in seconds
        save_logs: Whether to save results to JSON
        log_dir: Directory to save logs
        **adapter_config: Additional LiteLLM adapter configuration
        
    Returns:
        List of detailed task results
    """
    print(f"\n{'='*80}")
    print(f"Running Benchmark: {level_name} ({len(tasks)} tasks)")
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")
    
    # Debug: Check tasks type
    print(f"DEBUG: tasks type = {type(tasks)}")
    if isinstance(tasks, list) and len(tasks) > 0:
        print(f"DEBUG: first task type = {type(tasks[0])}")
    
    # Convert dataset format to runner format
    converted_tasks = convert_dataset_to_tasks(tasks)
    
    # Collect all required tools
    all_required_tools = []
    for task in converted_tasks:
        for tool in task.get("tools", []):
            if tool not in all_required_tools:
                all_required_tools.append(tool)
    
    print(f"Required tools: {all_required_tools}")
    
    # TODO: 데이터셋 수정 후 이 정규화 로직 삭제
    # 별칭을 실제 도구 이름으로 변환
    normalized_tools = [normalize_tool_name(t) for t in all_required_tools]
    print(f"Normalized tools: {normalized_tools}")
    
    # Resolve and register tools
    tool_classes = resolve_tool_classes(all_required_tools)
    missing_tools = [t for t in normalized_tools if t not in TOOL_CATALOG]
    if missing_tools:
        print(f"⚠ Warning: Missing tools in catalog: {missing_tools}")
    
    # Setup components
    registry = create_tool_registry(tool_classes)
    adapter = LiteLLMAdapter(model_name, **adapter_config)
    judge = Judge(llm_adapter=adapter)
    runner = BenchmarkRunner(adapter, registry, judge, max_steps=max_steps, timeout=timeout)
    
    all_results = []
    
    # Execute tasks
    for i, task in enumerate(converted_tasks, 1):
        print(f"\n{'─'*80}")
        print(f"Task {i}/{len(converted_tasks)}: {task['id']}")
        print(f"Level: {task['level']} | Category: {task['category']}")
        print(f"Instruction: {task['description']}")
        print(f"Expected tools: {task['tools']}")
        print(f"{'─'*80}")
        
        try:
            result = runner.run_task(task)
            
            # Add original task information
            result['task_id'] = task['id']
            result['instruction'] = task['description']
            result['level'] = task['level']
            result['category'] = task['category']
            result['expected_tools'] = task['tools']
            result['golden_action'] = task['golden_action']
            
            all_results.append(result)
            
            # Print summary
            print(f"\n✓ Success: {result['success']}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            print(f"  Steps taken: {result['steps_taken']}")
            
            # Print tool call details
            tool_invocations = result.get('tool_invocations', [])
            if tool_invocations:
                print(f"  Tool calls: {len(tool_invocations)}")
                for inv in tool_invocations:
                    print(f"    • Step {inv.get('step')}: {inv.get('tool_name')}")
                    print(f"      Args: {inv.get('arguments')}")
                    print(f"      Success: {inv.get('success')}")
                    if inv.get('error'):
                        print(f"      Error: {inv.get('error')}")
            else:
                print(f"  Tool calls: 0 (no tools used)")
            
            if result.get('result') and result['result'].get('final_response'):
                response = result['result']['final_response']
                preview = response[:200] + "..." if len(response) > 200 else response
                print(f"  Response: {preview}")
                
        except Exception as e:
            print(f"\n✗ Execution failed: {e}")
            all_results.append({
                "task_id": task.get('id', 'unknown'),
                "instruction": task.get('description', ''),
                "level": task.get('level', 0),
                "category": task.get('category', 'unknown'),
                "expected_tools": task.get('tools', []),
                "golden_action": task.get('golden_action', []),
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "steps_taken": 0,
                "tool_invocations": []
            })
    
    # Save results
    if save_logs and all_results:
        try:
            filepath = save_detailed_results(all_results, model_name, level_name, log_dir)
            print(f"\n{'='*80}")
            print(f"Results saved to: {filepath}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n✗ Failed to save results: {e}\n")
    
    return all_results


def main():
    """Main execution function."""
    print("="*80)
    print("Ko-AgentBench Dataset Runner with Tool Call Logging")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    provider_keys = [
        "HUGGINGFACE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
    ]
    found_keys = [k for k in provider_keys if os.getenv(k)]
    if not found_keys:
        print("⚠ Warning: No LLM API keys found in environment")
    else:
        print(f"✓ Found API keys: {found_keys}")
    
    # Load datasets
    print("\nLoading benchmark datasets...")
    datasets = load_benchmark_datasets("data")
    
    if not datasets:
        print("✗ No datasets found in data/ directory")
        return
    
    print(f"\n✓ Loaded {len(datasets)} dataset levels")
    
    # Select model
    if MODEL_IDS:
        selected_model = MODEL_IDS[-1]  # Use last model in list
    else:
        selected_model = "gpt-4.1"  # Default fallback
    
    print(f"\nSelected model: {selected_model}")
    
    # Run benchmarks on each level
    all_level_results = {}
    
    # You can customize which levels to run
    levels_to_run = ["L1","L2", "L3", "L4", "L5", "L6"]  
    # levels_to_run = ["L1"]  # Test only L1 first
    
    for level_name in levels_to_run:
        if level_name in datasets:
            results = run_benchmark_on_dataset(
                level_name=level_name,
                tasks=datasets[level_name],
                model_name=selected_model,
                max_steps=10,  # Only get first tool call, skip final response
                timeout=60,
                save_logs=True,
                log_dir="logs/benchmark_results"
            )
            all_level_results[level_name] = results
        else:
            print(f"⚠ Warning: {level_name} not found in datasets")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    for level_name, results in all_level_results.items():
        total = len(results)
        success = sum(1 for r in results if r.get('success', False))
        success_rate = (success / total * 100) if total > 0 else 0
        print(f"{level_name}: {success}/{total} tasks successful ({success_rate:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
