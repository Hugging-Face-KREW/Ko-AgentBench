"""
Ko-AgentBench 데이터셋 기반 벤치마크 실행 및 Tool Call 로깅 스크립트

data 폴더의 L1~L7 데이터셋을 로드하여 각 질문(instruction)을 실행하고,
사용된 tool call 정보를 상세하게 기록합니다.
"""

import os
import argparse
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Dict
try:
    from dotenv import load_dotenv  # provided by python-dotenv
except Exception:
    def load_dotenv(*args, **kwargs):  # no-op fallback if package not available
        return False

from bench.tools.tool_registry import ToolRegistry
from bench.adapters.litellm_adapter import LiteLLMAdapter
from bench.adapters.transformers_adapter import TransformersAdapter
from bench.runner import BenchmarkRunner
from bench.tools.base_api import BaseTool
from bench.models import MODEL_IDS
from bench.tools.tool_catalog import resolve_tool_classes, TOOL_CATALOG
from bench.config import set_cache_mode

# Import API keys from secrets
from configs.secrets import (
    AZURE_API_KEY, 
    AZURE_API_BASE, 
    AZURE_API_VERSION,
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY
)



def load_benchmark_datasets(data_dir: str = "bench/tasks") -> Dict[str, List[Dict]]:
    """Load all benchmark datasets from bench/tasks directory.
    
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
                print(f"[OK] Loaded {level_name}: {len(data)} tasks")
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    
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
            print(f"[WARNING] Skipping non-dict task: {type(task)}")
            continue
            
        # Extract required tools from golden_action OR conversation_tracking
        tools_needed = []
        
        # First, try golden_action (L1-L6)
        if "golden_action" in task:
            for action in task["golden_action"]:
                tool_name = action.get("tool", "")
                if tool_name and tool_name not in tools_needed:
                    tools_needed.append(tool_name)
        
        # Also check conversation_tracking for additional tools (L6/L7)
        if "conversation_tracking" in task:
            print(f"[DEBUG] Processing conversation_tracking for task: {task.get('task_id')}")
            conversation = task["conversation_tracking"]
            for turn in conversation.get("turns", []):
                # L7 format: turn.actions (list)
                if "actions" in turn:
                    for action in turn.get("actions", []):
                        tool_name = action.get("tool", "")
                        if tool_name and tool_name not in tools_needed:
                            tools_needed.append(tool_name)
                            print(f"  [OK] Found tool in turn.actions: {tool_name}")
                
                # L6 format: turn.action (single object)
                if "action" in turn:
                    action = turn.get("action", {})
                    tool_name = action.get("tool", "")
                    if tool_name and tool_name not in tools_needed:
                        tools_needed.append(tool_name)
                        print(f"  [OK] Found tool in turn.action: {tool_name}")
        
    # 별칭/정규화 제거: 선언된 도구 이름을 그대로 사용
    normalized_tools_needed = tools_needed
    print(f"  [INFO] Task {task.get('task_id')}: tools_needed = {tools_needed}")
        
        converted_task = {
            "id": task.get("task_id", "unknown"),
            "description": task.get("instruction", ""),
            "expected_output": task.get("resp_schema", {}),
            "available_tools": normalized_tools_needed,  # normalized for runner compatibility
            # keep backward-compat with runner that expects 'tools'
            "tools": normalized_tools_needed,
            "level": task.get("task_level", 0),
            "category": task.get("task_category", "unknown"),
            "golden_action": task.get("golden_action", []),
            "conversation_tracking": task.get("conversation_tracking"),
            "arg_schema": task.get("arg_schema", {}),
            # Pass through evaluation helpers if present
            "minimum_steps": task.get("minimum_steps"),
            "data_flow": task.get("data_flow", []),
            "error_injection": task.get("error_injection")
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
        "minimum_steps": result.get("minimum_steps"),
        "data_flow": result.get("data_flow", []),
        "error_injection": result.get("error_injection"),
        "token_usage": result.get("token_usage", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }),
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

    # Store complete conversation log for multi-turn scenarios
    try:
        conv = (result.get("result") or {}).get("conversation") or []
        total_msgs = len(conv)
        
        def _format_message(msg):
            """Format a single message for logging."""
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # For tool messages, include tool_call_id
            if role == "tool":
                tcid = msg.get("tool_call_id")
                # Try to parse and format tool result
                try:
                    tool_data = json.loads(content) if isinstance(content, str) else content
                    return {
                        "role": role,
                        "tool_call_id": tcid,
                        "content": tool_data
                    }
                except:
                    return {
                        "role": role,
                        "tool_call_id": tcid,
                        "content": content
                    }
            
            # For user/assistant messages, return as-is
            return {"role": role, "content": content}

        # Store complete conversation log
        simplified["conversation_log"] = {
            "total_messages": total_msgs,
            "messages": [_format_message(m) for m in conv]
        }
            
    except Exception as e:
        # Non-fatal: skip conversation logging if structure unexpected
        simplified["conversation_log_error"] = str(e)
    
    return simplified


def save_detailed_results(
    results: List[Dict[str, Any]], 
    model_name: str, 
    level_name: str,
    output_dir: str = "logs/benchmark_results",
    run_timestamp: str = None
) -> str:
    """Save detailed benchmark results including tool call information.
    
    Args:
        results: List of task results with tool call details
        model_name: Name of the model used
        level_name: Dataset level (L1, L2, etc.)
        output_dir: Directory to save logs
        run_timestamp: Timestamp for the run (if None, creates new one)
        
    Returns:
        Path to saved JSON file
    """
    # Create timestamp for this run if not provided
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_safe_name = model_name.replace("/", "_")
    
    # Create by_model structure: logs/benchmark_results/by_model/{model}/{timestamp}/
    by_model_path = Path(output_dir) / "by_model" / model_safe_name / run_timestamp
    by_model_path.mkdir(parents=True, exist_ok=True)
    
    # Save result file in the by_model folder
    filename = f"{level_name}.json"
    filepath = by_model_path / filename
    
    # Simplify and flatten results
    simplified_results = [simplify_result(r) for r in results]
    
    # Calculate statistics
    total_tasks = len(simplified_results)
    successful_tasks = sum(1 for r in simplified_results if r.get('success', False))
    total_time = sum(r.get('execution_time', 0) for r in simplified_results)
    total_steps = sum(r.get('steps_taken', 0) for r in simplified_results)
    total_tool_calls = sum(len(r.get('tool_calls', [])) for r in simplified_results)

    # 토큰 통계 추가
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for result in simplified_results:
        token_usage = result.get('token_usage', {})
        total_prompt_tokens += token_usage.get('prompt_tokens', 0)
        total_completion_tokens += token_usage.get('completion_tokens', 0)
        total_tokens += token_usage.get('total_tokens', 0)

    # TPS 계산
    average_tps = total_tokens / total_time if total_time > 0 else 0
    
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
            "total_tokens": total_tokens,
            "average_tokens_per_task": round(total_tokens / total_tasks, 2) if total_tasks > 0 else 0,
            "average_prompt_tokens": round(total_prompt_tokens / total_tasks, 2) if total_tasks > 0 else 0,
            "average_completion_tokens": round(total_completion_tokens / total_tasks, 2) if total_tasks > 0 else 0,
            "average_tps": round(average_tps, 2),
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
    use_local: bool = False,
    max_steps: int = 10,  
    timeout: int = 60,
    save_logs: bool = True,
    log_dir: str = "logs/benchmark_results",
    run_timestamp: str = None,
    **adapter_config: Any
) -> List[Dict[str, Any]]:
    """Run benchmark on a specific dataset level.
    
    Args:
        level_name: Dataset level name (L1, L2, etc.)
        tasks: List of tasks to execute
        model_name: LLM model identifier
        use_local: If True, use TransformersAdapter for local inference
        max_steps: Maximum steps per task
        timeout: Timeout per task in seconds
        save_logs: Whether to save results to JSON
        log_dir: Directory to save logs
        run_timestamp: Timestamp for the entire run (shared across levels)
        **adapter_config: Additional adapter configuration
        
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
    
    # Convert dataset format to runner format (always use full set)
    converted_tasks = convert_dataset_to_tasks(tasks)
    
    # Collect all required tools
    all_required_tools = []
    for task in converted_tasks:
        for tool in task.get("available_tools", []):
            if tool not in all_required_tools:
                all_required_tools.append(tool)
    
    print(f"Required tools: {all_required_tools}")
    
    # 별칭/정규화 제거: 입력된 이름을 그대로 사용
    normalized_tools = all_required_tools
    print(f"Tools: {normalized_tools}")
    
    # Resolve and register tools
    tool_classes = resolve_tool_classes(all_required_tools)
    missing_tools = [t for t in normalized_tools if t not in TOOL_CATALOG]
    if missing_tools:
        print(f"[WARNING] Missing tools in catalog: {missing_tools}")
    
    # Setup components
    registry = create_tool_registry(tool_classes)
    
    # DEBUG: Check registered tools and their schemas
    print(f"\n[DEBUG] Registered tools in registry:")
    registered_tool_names = registry.get_available_tools()
    print(f"  Tool names: {registered_tool_names}")
    
    if registered_tool_names:
        print(f"\n  Tool schemas:")
        for tool_name in registered_tool_names[:3]:  # Show first 3
            tool = registry.get_tool(tool_name)
            if tool:
                schema = tool.get_schema()
                print(f"    - {tool_name}:")
                print(f"      Schema keys: {list(schema.keys())}")
                if 'function' in schema:
                    print(f"      Function name: {schema['function'].get('name')}")
                    print(f"      Description: {schema['function'].get('description')[:50]}...")
    else:
        print("  [WARNING] No tools registered!")
    
    # Create adapter based on use_local flag
    if use_local:
        print(f"\n[LOCAL] Using TransformersAdapter for local inference")
        adapter = TransformersAdapter(model_name, **adapter_config)
    else:
        print(f"\n[API] Using LiteLLMAdapter for API inference")
        adapter = LiteLLMAdapter(model_name, **adapter_config)
    
    runner = BenchmarkRunner(adapter, registry, max_steps=max_steps, timeout=timeout)
    
    all_results = []
    
    # Execute tasks
    for i, task in enumerate(converted_tasks, 1):
        print(f"\n{'─'*80}")
        print(f"Task {i}/{len(converted_tasks)}: {task['id']}")
        print(f"Level: {task['level']} | Category: {task['category']}")
        print(f"Instruction: {task['description']}")
        print(f"Expected tools: {task['available_tools']}")
        
        # Log multi-turn conversation context if present
        if task.get("conversation_tracking") and isinstance(task["conversation_tracking"].get("turns"), list):
            turns = task["conversation_tracking"]["turns"]
            valid_turns = [t for t in turns if t.get("role") in ("user", "assistant") and t.get("content")]
            print(f"\n[MULTI-TURN] Multi-turn conversation context:")
            print(f"   Total turns to seed: {len(valid_turns)}")
            for idx, turn in enumerate(valid_turns, 1):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                content_preview = content[:100] + "..." if len(content) > 100 else content
                print(f"   Turn {idx} [{role}]: {content_preview}")
                
                # Log any tool actions in this turn (for reference)
                if turn.get("actions"):
                    for action in turn["actions"]:
                        tool_name = action.get("tool", "unknown")
                        print(f"      [TOOL] Expected tool: {tool_name}")
        
        print(f"{'─'*80}")
        
        try:
            result = runner.run_task(task)
            
            # Add original task information
            result['task_id'] = task['id']
            result['instruction'] = task['description']
            result['level'] = task['level']
            result['category'] = task['category']
            result['expected_tools'] = task['available_tools']
            result['golden_action'] = task['golden_action']
            # Include dataset guidance fields for analysis
            if 'minimum_steps' in task:
                result['minimum_steps'] = task.get('minimum_steps')
            if 'data_flow' in task:
                result['data_flow'] = task.get('data_flow', [])
            if 'error_injection' in task:
                result['error_injection'] = task.get('error_injection')
            all_results.append(result)
            
            # Print summary
            print(f"\n[RESULT] Success: {result['success']}")
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
            
            # Print conversation summary for multi-turn tasks
            if result.get('result') and result['result'].get('conversation'):
                conversation = result['result']['conversation']
                print(f"\n  [CONVERSATION] Conversation summary:")
                print(f"     Total messages: {len(conversation)}")
                
                # Show last few messages
                last_messages = conversation[-3:] if len(conversation) > 3 else conversation
                for msg in last_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if content:
                        preview = content[:80] + "..." if len(content) > 80 else content
                        print(f"     [{role}]: {preview}")
            
            if result.get('result') and result['result'].get('final_response'):
                response = result['result']['final_response']
                preview = response[:200] + "..." if len(response) > 200 else response
                print(f"  Response: {preview}")
                
        except Exception as e:
            print(f"\n[ERROR] Execution failed: {e}")
            all_results.append({
                "task_id": task.get('id', 'unknown'),
                "instruction": task.get('description', ''),
                "level": task.get('level', 0),
                "category": task.get('category', 'unknown'),
                "expected_tools": task.get('available_tools', []),
                "golden_action": task.get('golden_action', []),
                "minimum_steps": task.get('minimum_steps'),
                "data_flow": task.get('data_flow', []),
                "success": False,
                "error": str(e),
                "execution_time": 0,
                "steps_taken": 0,
                "tool_invocations": []
            })
    
    # Save results
    if save_logs and all_results:
        try:
            filepath = save_detailed_results(all_results, model_name, level_name, log_dir, run_timestamp)
            print(f"\n{'='*80}")
            print(f"Results saved to: {filepath}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\n[ERROR] Failed to save results: {e}\n")
    
    return all_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Ko-AgentBench runner with tool-call logging")
    parser.add_argument("--levels", type=str, default=None,
                        help="Comma-separated levels to run (e.g., L6,L7). Default: all detected")
    # removed --limit: always run full level
    parser.add_argument("--max-steps", type=int, default=10,
                        help="Maximum steps per task")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Timeout (seconds) per task")
    parser.add_argument("--no-save-logs", action="store_true",
                        help="Do not save JSON logs to disk")
    parser.add_argument("--model", type=str, default=None,
                        help="Explicit model id (overrides auto selection)")
    parser.add_argument("--use-local", action="store_true",
                        help="Use local transformers inference instead of API")
    parser.add_argument("--quantization", type=str, default=None, choices=['4bit', '8bit'],
                        help="Quantization method for local models (4bit or 8bit)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for local inference (cuda, cpu, auto)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=['auto', 'float16', 'bfloat16', 'float32', 'fp16', 'bf16', 'fp32'],
                        help="Torch dtype for local models")
    parser.add_argument("--cache-mode", type=str, default="read",
                        choices=['read', 'write'],
                        help="Cache mode for API calls: 'read' = use cached responses only (no real API calls), 'write' = call real APIs and cache results (default: read)")

    args = parser.parse_args()
    
    # Set cache mode from command-line argument
    set_cache_mode(args.cache_mode)
    print(f"Cache mode: {args.cache_mode}")

    print("="*80)
    print("Ko-AgentBench Dataset Runner with Tool Call Logging")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Set API keys environment variables if available from secrets
    if AZURE_API_KEY:
        os.environ['AZURE_API_KEY'] = AZURE_API_KEY
    if AZURE_API_BASE:
        os.environ['AZURE_API_BASE'] = AZURE_API_BASE
    if AZURE_API_VERSION:
        os.environ['AZURE_API_VERSION'] = AZURE_API_VERSION
    if ANTHROPIC_API_KEY:
        os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
    if GEMINI_API_KEY:
        os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
    
    # Check API keys (include Azure/Google for better provider detection)
    provider_keys = [
        "HUGGINGFACE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "AZURE_API_KEY",
        "GOOGLE_API_KEY",
    ]
    found_keys = [k for k in provider_keys if os.getenv(k)]
    if not found_keys:
        print("[WARNING] No LLM API keys found in environment")
    else:
        print(f"[OK] Found API keys: {found_keys}")
    
    # Load datasets
    print("\nLoading benchmark datasets...")
    datasets = load_benchmark_datasets("bench/tasks")
    
    if not datasets:
        print("[ERROR] No datasets found in bench/tasks/ directory")
        return
    
    print(f"\n[OK] Loaded {len(datasets)} dataset levels")
    
    # Select a model compatible with available provider keys
    def _provider_ready(model_id: str) -> bool:
        """Return True if the provider for model_id has required env keys."""
        # Model id format: "<provider>/<model_name>"
        if "/" not in model_id:
            return False
        provider = model_id.split("/", 1)[0].lower()
        if provider == "azure":
            return bool(os.getenv("AZURE_API_KEY") and os.getenv("AZURE_API_BASE") and os.getenv("AZURE_API_VERSION"))
        if provider == "openai":
            return bool(os.getenv("OPENAI_API_KEY"))
        if provider == "anthropic":
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        if provider == "groq":
            return bool(os.getenv("GROQ_API_KEY"))
        if provider == "gemini":
            return bool(os.getenv("GEMINI_API_KEY"))
        if provider == "huggingface":
            return bool(os.getenv("HUGGINGFACE_API_KEY"))
        return False

    selected_model = None
    if MODEL_IDS:
        # Prefer the first model in MODEL_IDS that matches available keys
        for m in MODEL_IDS:
            if _provider_ready(m):
                selected_model = m
                break
        # Fallback: keep existing preference order if none match, but warn
        if selected_model is None:
            selected_model = MODEL_IDS[0]
            print(
                f"[WARNING] No provider credentials matched MODEL_IDS. Falling back to '{selected_model}'.\n"
                f"  Tip: Set provider keys to match one of: {MODEL_IDS}"
            )
    else:
        # Final fallback
        selected_model = "openai/gpt-4.1" if os.getenv("OPENAI_API_KEY") else "azure/gpt-4.1"
    
    # Override model from CLI if provided
    if args.model:
        selected_model = args.model
    print(f"\nSelected model: {selected_model}")
    
    # For local inference, model name doesn't need provider prefix
    if args.use_local:
        # Remove provider prefix if present (e.g., "huggingface/Qwen/..." -> "Qwen/...")
        if "/" in selected_model and selected_model.split("/")[0] in ["huggingface", "openai", "anthropic", "azure", "groq", "gemini"]:
            selected_model = "/".join(selected_model.split("/")[1:])
        print(f"Using local inference mode")
        print(f"Model: {selected_model}")
        if args.quantization:
            print(f"Quantization: {args.quantization}")
        print(f"Device: {args.device}")
        print(f"Dtype: {args.dtype}")
    
    # Prepare adapter config for local models
    adapter_config = {}
    if args.use_local:
        adapter_config['device'] = args.device
        adapter_config['dtype'] = args.dtype
        if args.quantization:
            adapter_config['quantization'] = args.quantization
        # Context management is now handled automatically by TransformersAdapter
        # based on model config (max_position_embeddings, etc.)
    
    # Run benchmarks on each level
    all_level_results = {}
    
    # Create a single timestamp for the entire run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine which levels to run
    if args.levels:
        levels_to_run = [lvl.strip() for lvl in args.levels.split(',') if lvl.strip()]
    else:
        # Default: run all discovered levels (including multi-turn L6/L7)
        levels_to_run = sorted(list(datasets.keys()))
    
    for level_name in levels_to_run:
        if level_name in datasets:
            results = run_benchmark_on_dataset(
                level_name=level_name,
                tasks=datasets[level_name],
                model_name=selected_model,
                use_local=args.use_local,
                max_steps=args.max_steps,
                timeout=args.timeout,
                save_logs=(not args.no_save_logs),
                log_dir="logs/benchmark_results",
                run_timestamp=run_timestamp,
                **adapter_config
            )
            all_level_results[level_name] = results
        else:
            print(f"[WARNING] {level_name} not found in datasets")
    
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