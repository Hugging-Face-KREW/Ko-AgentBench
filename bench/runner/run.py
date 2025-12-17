"""Main benchmark runner for Ko-AgentBench."""

import json
import time
import logging
from typing import Any, Dict, List, Optional

from ..adapters.base_adapter import BaseAdapter
from ..tools.tool_registry import ToolRegistry
from ..observability import observe, get_client, is_enabled
from ..tools.caching_executor import CachingExecutor


class BenchmarkRunner:
    """Main runner for executing benchmarks with LLM-tool loops."""
    
    def __init__(self, 
                 adapter: BaseAdapter,
                 tool_registry: ToolRegistry,
                 max_steps: int = 10,
                 timeout: int = 300,
                 max_retries: int = 3,
                 multiturn_mode: str = "seed_replay"):
        """Initialize benchmark runner.
        
        Args:
            adapter: LLM adapter for API calls
            tool_registry: Registry of available tools
            max_steps: Maximum number of steps per task
            timeout: Timeout in seconds per task
            max_retries: Maximum retries for failed API calls
        """
        self.adapter = adapter
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.timeout = timeout
        self.max_retries = max_retries

        if multiturn_mode not in {"seed_replay", "full_rollout"}:
            raise ValueError(
                "Invalid multiturn_mode. Expected 'seed_replay' or 'full_rollout', "
                f"got: {multiturn_mode!r}"
            )
        self.multiturn_mode = multiturn_mode
        
        self.logger = logging.getLogger(__name__)
    
    @observe(name="run_task")
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark task.

        Args:
            task: Task definition dictionary

        Returns:
            Task execution result including tool invocation summary
        """
        start_time = time.time()
        # Accept both 'task_id' and 'id' from upstream converters
        task_id = task.get('task_id') or task.get('id', 'unknown')

        if is_enabled():
            try:
                langfuse = get_client()
                category = task.get('task_category') or task.get('category', 'unknown')
                difficulty = task.get('task_level') or task.get('difficulty', 'unknown')

                langfuse.update_current_trace(
                    name=f"Task: {task_id}",
                    metadata={
                        "task_id": task_id,
                        "category": category,
                        "difficulty": difficulty,
                    },
                    tags=[str(category), str(difficulty)]
                )
            except Exception as e:
                self.logger.debug(f"Langfuse update failed: {e}")

        self.logger.info(f"Starting task {task_id}")

        try:
            # Seed conversation messages
            messages: List[Dict[str, Any]] = []
            conversation = task.get('conversation_tracking') or task.get('conversation')
            if isinstance(conversation, dict) and isinstance(conversation.get('turns'), list):
                # Build messages from provided multi-turn conversation, trimming to the last user turn
                turns = conversation.get('turns', [])
                for t in turns:
                    role = t.get('role')
                    content = t.get('content', '')
                    if role == "user" and content:
                        messages.append({"role": role, "content": content})
                    elif role == "assistant":
                        # For assistant messages, preserve/convert tool_calls
                        msg = {"role": role}
                        if content:
                            msg["content"] = content
                        
                        # Check for tool_calls (already in correct format)
                        if t.get('tool_calls'):
                            msg["tool_calls"] = t['tool_calls']
                            messages.append(msg)
                        # Or convert from 'action' field (dataset format)
                        elif t.get('action'):
                            action = t['action']
                            if isinstance(action, dict) and action.get('tool') and action.get('result'):
                                # Only convert action to tool_calls if result exists (complete pair)
                                import uuid
                                call_id = f"call_{uuid.uuid4().hex[:24]}"
                                msg["tool_calls"] = [{
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": action['tool'],
                                        "arguments": json.dumps(action.get('args', {}))
                                    }
                                }]
                                messages.append(msg)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": call_id,
                                    "content": json.dumps(action['result'])
                                })
                            elif content:
                                # Text-only assistant message (no tool calls)
                                messages.append(msg)
                        elif content:
                            # Text-only assistant message
                            messages.append(msg)
                    elif role == "tool":
                        # Also preserve tool result messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": t.get('tool_call_id'),
                            "content": content
                        })
                # Trim trailing assistant messages so the next model output is a response to the last user
                if messages:
                    # Find last index of a user message
                    last_user_idx = None
                    for idx in range(len(messages) - 1, -1, -1):
                        if messages[idx].get('role') == 'user':
                            last_user_idx = idx
                            break
                    if last_user_idx is not None:
                        messages = messages[: last_user_idx + 1]
                # Fallback if conversation contained no valid user message
                if not messages:
                    task_description = task.get('instruction') or task.get('description', '')
                    messages = [{"role": "user", "content": task_description}]
            else:
                # Single-turn fallback
                task_description = task.get('instruction') or task.get('description', '')
                messages = [{"role": "user", "content": task_description}]

            # For L5 tasks with fallback_options, use only golden_action tools initially
            if task.get('fallback_options') and task.get('golden_action'):
                # L5 fallback scenario: Start with golden_action tools only
                task_tools = []
                golden_action = task.get('golden_action', [])
                if isinstance(golden_action, dict):
                    golden_action = [golden_action]

                for action in golden_action:
                    if isinstance(action, dict):
                        tool_name = action.get('tool')
                        if tool_name and tool_name != 'reuse' and tool_name not in task_tools:
                            task_tools.append(tool_name)

                self.logger.info(f"[L5] Starting with golden_action tools: {task_tools}")
            else:
                # Get available tools for this task (non-L5 or L5 without fallback)
                task_tools = task.get('available_tools') or task.get('tools', [])

                # If no available_tools specified, extract from golden_action
                if not task_tools and task.get('golden_action'):
                    golden_action = task.get('golden_action', [])
                    if isinstance(golden_action, dict):
                        golden_action = [golden_action]

                    for action in golden_action:
                        if isinstance(action, dict):
                            tool_name = action.get('tool')
                            if tool_name and tool_name != 'reuse' and tool_name not in task_tools:
                                task_tools.append(tool_name)

                    if task_tools:
                        self.logger.info(f"[INFO] Extracted tools from golden_action: {task_tools}")

            available_tools = []
            for tool_name in task_tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    available_tools.append(tool.get_schema())
                else:
                    self.logger.warning(f"Tool '{tool_name}' not found in registry")

            # ...existing code...

            # Check if this is a multi-turn conversation task (L6/L7)
            conversation = task.get('conversation_tracking') or task.get('conversation')
            if isinstance(conversation, dict) and isinstance(conversation.get('turns'), list):
                # Multi-turn conversation: execute each user turn sequentially
                result = self._execute_multiturn_conversation(conversation, available_tools, task)
            else:
                # Single-turn task: execute once
                task_description = task.get('instruction') or task.get('description', '')
                messages = [{"role": "user", "content": task_description}]
                result = self._execute_loop_with_fallback(messages, available_tools, task)
            
            # Aggregate tool invocation summary
            tool_invocations = []
            for step in result.get('steps', []):
                for tool_call in step.get('tool_calls', []):
                    tool_invocations.append({
                        "step": step.get('step'),
                        "tool_call_id": tool_call.get('tool_call_id'),
                        "tool_name": tool_call.get('tool_name'),
                        "arguments": tool_call.get('arguments'),
                        "result": tool_call.get('result'),
                        "success": tool_call.get('success'),
                        "error": tool_call.get('error'),
                        "error_type": tool_call.get('error_type'),  # For ErrorDetect metric
                    })
            
            # Add tool_calls to result (for legacy compatibility)
            result['tool_calls'] = tool_invocations
            
            execution_time = time.time() - start_time
            
            # Determine success based on task completion (no error and has result)
            success = (
                result.get('final_response') is not None 
                and len(result.get('steps', [])) > 0
            )
            
            # 토큰 사용량 및 TTFT 집계
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            ttft_list = []  # ← TTFT 리스트

            for step in result.get('steps', []):
                llm_response = step.get('llm_response', {})
                usage = llm_response.get('usage', {})
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                total_tokens += usage.get('total_tokens', 0)
                
                # TTFT 수집
                ttft = step.get('ttft', 0)
                if ttft > 0:
                    ttft_list.append(ttft)

            # TTFT 통계 계산
            avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
            min_ttft = min(ttft_list) if ttft_list else 0
            max_ttft = max(ttft_list) if ttft_list else 0

            final_result = {
                "task_id": task_id,
                "success": success,
                "result": result,
                "tool_calls": tool_invocations,  # 필드명 변경
                "execution_time": execution_time,
                "steps_taken": len(result.get('steps', [])),
                "token_usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens
                },
                "ttft_stats": {  # ← TTFT 통계 추가
                    "average": avg_ttft,
                    "min": min_ttft,
                    "max": max_ttft,
                    "count": len(ttft_list)
                },
                "error": None
            }
                        
            # update trace with final result 
            if is_enabled():
                try:
                    langfuse = get_client()
                    langfuse.update_current_trace(
                        output=final_result,
                        metadata={
                            "success": final_result["success"],
                            "steps_taken": final_result["steps_taken"],
                            "tool_calls_count": len(tool_invocations),
                        }
                    )
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            # Reset adapter conversation state for next task (important for Assistants API)
            if hasattr(self.adapter, 'reset_conversation'):
                self.adapter.reset_conversation()
            
            return final_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
            error_result = {
                "task_id": task_id,
                "success": False,
                "result": None,
                "evaluation": None,
                "execution_time": execution_time,
                "steps_taken": 0,
                "error": str(e)
            }
            
            # update trace with error 
            if is_enabled():
                try:
                    langfuse = get_client()
                    langfuse.update_current_trace(
                        output=error_result,
                        metadata={"error": str(e)}
                    )
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            # Reset adapter conversation state even on error
            if hasattr(self.adapter, 'reset_conversation'):
                self.adapter.reset_conversation()
            
            return error_result
    
    @observe(name="execute_multiturn_conversation")
    def _execute_multiturn_conversation(self, 
                                       conversation: Dict[str, Any],
                                       tools: List[Dict], 
                                       task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a multi-turn conversation (L6/L7 scenarios).
        
        This method processes each user turn sequentially, maintaining conversation context
        across turns. This is essential for:
        - L6 (Efficiency): Testing context reuse ("방금 찾아준 책 다시 알려줘")
        - L7 (Long-term Context): Testing long-term memory ("아까 처음에 물어봤던 코인")
        
        Args:
            conversation: Conversation tracking object with turns
            tools: Available tools
            task: Task definition
            
        Returns:
            Execution result with all conversation steps
        """
        turns = conversation.get('turns', [])

        # Allow per-task override, otherwise use runner default.
        mode = task.get('multiturn_mode') or self.multiturn_mode
        if mode not in {"seed_replay", "full_rollout"}:
            self.logger.warning(f"[MULTI-TURN] Unknown mode={mode!r}; falling back to 'seed_replay'")
            mode = "seed_replay"

        # Initialize conversation (no system prompt as per requirement)
        messages: List[Dict[str, Any]] = []

        all_steps: List[Dict[str, Any]] = []
        step_counter = 0

        # Extract task level for API routing
        task_level = task.get('level')

        def _drain_tool_calls(initial_response: Dict[str, Any], turn_label: int) -> None:
            """Execute tool calls until the assistant returns no tool_calls.

            This is critical for Azure Responses API: every function_call(call_id)
            must have a matching function_call_output(call_id) before the next
            request when using previous_response_id.
            """
            nonlocal step_counter

            response_local = initial_response
            while True:
                message_local = (response_local or {}).get('message', {}) or {}

                step_counter += 1
                step_data_local = {
                    "step": step_counter,
                    "turn": turn_label,
                    "llm_response": response_local,
                    "tool_calls": [],
                    "timestamp": time.time(),
                    "ttft": (response_local or {}).get('ttft', 0)
                }

                assistant_content_local = message_local.get('content', '')
                if assistant_content_local:
                    self.logger.info(f"[ASSISTANT] {assistant_content_local[:200]}...")

                # Add assistant message to conversation
                messages.append(message_local)

                tool_calls_local = message_local.get('tool_calls') or []
                if tool_calls_local:
                    self.logger.info(f"[TOOL] Tool calls: {len(tool_calls_local)}")

                    for idx, tool_call in enumerate(tool_calls_local, 1):
                        tool_name = tool_call.get('function', {}).get('name', 'unknown')
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        self.logger.info(f"  [{idx}] {tool_name}")
                        self.logger.info(f"      Args: {str(tool_args)[:150]}...")

                        tool_result = self._execute_tool_call(tool_call, task)
                        step_data_local['tool_calls'].append(tool_result)

                        if tool_result.get('success'):
                            result_preview = str(tool_result.get('result', ''))[:150]
                            self.logger.info(f"      [SUCCESS] {result_preview}...")
                        else:
                            self.logger.error(f"      [ERROR] {tool_result.get('error')}")

                        # Always append a tool output (even on failure)
                        result_content = tool_result.get('result')
                        if result_content is None:
                            result_content = {"error": tool_result.get('error', 'Tool execution failed')}

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": json.dumps(result_content)
                        })

                    all_steps.append(step_data_local)

                    if step_counter >= self.max_steps:
                        self.logger.warning(
                            f"[WARN] Reached max_steps={self.max_steps} while draining tool calls; "
                            "stopping further tool execution to avoid infinite loops."
                        )
                        break

                    self.logger.info("[INFO] Getting follow-up response after tool execution...")
                    response_local = self._call_llm_with_retry(messages, tools, task_level)
                    continue

                # No tool calls -> conversation for this request is complete
                all_steps.append(step_data_local)
                break

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[MULTI-TURN] Multi-turn conversation: {len(turns)} turns | mode={mode}")
        self.logger.info(f"{'='*80}\n")

        if mode == "full_rollout":
            # Process each user turn sequentially (original behavior)
            for turn_idx, turn in enumerate(turns, 1):
                role = turn.get('role', '')
                content = turn.get('content', '')

                if role != 'user' or not content:
                    continue

                self.logger.info(f"\n{'─'*80}")
                self.logger.info(f"[TURN {turn_idx}] Processing user message")
                self.logger.info(f"{'─'*80}")
                self.logger.info(f"User: {content[:200]}...")

                messages.append({"role": "user", "content": content})

                try:
                    first_response = self._call_llm_with_retry(messages, tools, task_level)
                    _drain_tool_calls(first_response, turn_label=turn_idx)
                except Exception as e:
                    self.logger.error(f"[ERROR] Error processing turn {turn_idx}: {e}")
                    continue
        else:
            # Seed replay: reuse dataset-provided assistant/tool results up to the evaluation turn,
            # then call the model only on that evaluation user turn.
            evaluation_turn = None
            eval_ctx = conversation.get('evaluation_context') or {}
            context_tests = eval_ctx.get('context_tests') or []
            if isinstance(context_tests, list) and context_tests:
                try:
                    evaluation_turn = max(int(t.get('turn')) for t in context_tests if t.get('turn') is not None)
                except Exception:
                    evaluation_turn = None

            if evaluation_turn is None:
                user_turn_numbers = [
                    int(t.get('turn_number'))
                    for t in turns
                    if t.get('role') == 'user' and t.get('turn_number') is not None
                ]
                evaluation_turn = max(user_turn_numbers) if user_turn_numbers else 1

            def _append_seeded_assistant_and_tools(seed_turn: Dict[str, Any]) -> None:
                assistant_msg: Dict[str, Any] = {"role": "assistant"}
                content_local = seed_turn.get('content', '')
                if content_local:
                    assistant_msg["content"] = content_local

                tool_calls_out: List[Dict[str, Any]] = []
                tool_messages_out: List[Dict[str, Any]] = []

                actions = []
                if isinstance(seed_turn.get('actions'), list):
                    actions = seed_turn.get('actions')
                elif isinstance(seed_turn.get('action'), dict):
                    actions = [seed_turn.get('action')]

                for idx, action in enumerate(actions, 1):
                    if not isinstance(action, dict):
                        continue
                    tool_name = action.get('tool')
                    if not tool_name:
                        continue

                    call_id = f"seed_call_{seed_turn.get('turn_number', 'x')}_{idx}"
                    tool_calls_out.append({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(action.get('args', {}) or {})
                        }
                    })

                    if action.get('result') is not None:
                        tool_messages_out.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": json.dumps(action.get('result'))
                        })

                if tool_calls_out:
                    assistant_msg["tool_calls"] = tool_calls_out

                # Append assistant message (even if it only contains tool_calls)
                if assistant_msg.get("content") or assistant_msg.get("tool_calls"):
                    messages.append(assistant_msg)
                for tool_msg in tool_messages_out:
                    messages.append(tool_msg)

            for seed_turn in turns:
                tn = seed_turn.get('turn_number')
                try:
                    tn_int = int(tn) if tn is not None else None
                except Exception:
                    tn_int = None

                role = seed_turn.get('role', '')
                content = seed_turn.get('content', '')

                if tn_int is None:
                    continue

                if tn_int > evaluation_turn:
                    break

                if role == 'user':
                    if content:
                        messages.append({"role": "user", "content": content})
                    if tn_int == evaluation_turn:
                        break
                elif role == 'assistant':
                    _append_seeded_assistant_and_tools(seed_turn)
                elif role == 'tool':
                    # If the dataset explicitly provides tool messages, preserve them.
                    if content:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": seed_turn.get('tool_call_id'),
                            "content": content
                        })

            self.logger.info(f"[SEED] Seeded messages: {len(messages)} | evaluation_turn={evaluation_turn}")

            # Call the model once on the evaluation turn context.
            try:
                first_response = self._call_llm_with_retry(messages, tools, task_level)
                _drain_tool_calls(first_response, turn_label=evaluation_turn)
            except Exception as e:
                self.logger.error(f"[ERROR] Error processing evaluation turn {evaluation_turn}: {e}")

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[COMPLETE] Multi-turn conversation completed: {len(all_steps)} steps")
        self.logger.info(f"{'='*80}\n")

        final_response_content = ""
        for msg in reversed(messages):
            if msg.get('role') == 'assistant' and msg.get('content'):
                final_response_content = msg.get('content', '')
                break

        return {
            "steps": all_steps,
            "final_response": final_response_content,
            "conversation": messages
        }
    
    @observe(name="execute_loop")
    def _execute_loop(self, messages: List[Dict], 
                     tools: List[Dict], 
                     task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM-tool interaction loop.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            task: Task definition
            
        Returns:
            Execution result
        """
        steps = []
        start_time = time.time()
        
        # Extract task level for API routing
        task_level = task.get('level')
        
        # Log seeded conversation history for multi-turn scenarios
        seeded_messages_count = len(messages)
        if seeded_messages_count > 0:
            self.logger.info(f"[CONTEXT] Multi-turn context: {seeded_messages_count} messages seeded")
            for idx, msg in enumerate(messages):
                role = msg.get('role', 'unknown')
                content_preview = msg.get('content', '')[:100]
                self.logger.info(f"  [{idx+1}] {role}: {content_preview}...")
        
        for step in range(self.max_steps):
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Task exceeded timeout of {self.timeout} seconds")
            
            # Log current turn information
            current_turn = step + 1
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"[STEP {current_turn}/{self.max_steps}]")
            self.logger.info(f"{'='*60}")
            
            # Get LLM response
            response = self._call_llm_with_retry(messages, tools, task_level)
            
            step_data = {
                "step": step + 1,
                "llm_response": response,
                "tool_calls": [],
                "timestamp": time.time(),
                "ttft": response.get('ttft', 0)
            }
            
            # Handle tool calls
            message = response.get('message', {})
            
            # Log assistant response
            assistant_content = message.get('content', '')
            if assistant_content:
                self.logger.info(f"[ASSISTANT] Assistant response: {assistant_content[:200]}...")
            
            # Add assistant message to conversation first
            messages.append(message)
            
            # Then handle tool calls if present
            if 'tool_calls' in message and message['tool_calls']:
                self.logger.info(f"[TOOL] Tool calls in this turn: {len(message['tool_calls'])}")
                for idx, tool_call in enumerate(message['tool_calls'], 1):
                    tool_name = tool_call.get('function', {}).get('name', 'unknown')
                    tool_args = tool_call.get('function', {}).get('arguments', '{}')
                    self.logger.info(f"  [{idx}] Calling: {tool_name}")
                    self.logger.info(f"      Args: {tool_args[:150]}...")
                    
                    tool_result = self._execute_tool_call(tool_call, task)
                    step_data['tool_calls'].append(tool_result)
                    
                    # Log tool execution result
                    if tool_result.get('success'):
                        result_preview = str(tool_result.get('result', ''))[:150]
                        self.logger.info(f"      [SUCCESS] {result_preview}...")
                    else:
                        self.logger.error(f"      [ERROR] {tool_result.get('error')}")
                    
                    # Add tool result to messages for next turn
                    # Include error information if tool call failed
                    if tool_result.get('success'):
                        content = json.dumps(tool_result['result'])
                    else:
                        # Send error information to the model
                        content = json.dumps({
                            "error": tool_result.get('error'),
                            "error_type": tool_result.get('error_type')
                        })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": content
                    })
            else:
                self.logger.info(f"[COMPLETE] No tool calls - conversation complete")
            
            steps.append(step_data)
            
            # Check if task is complete (no more tool calls)
            if 'tool_calls' not in message or not message['tool_calls']:
                break
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[COMPLETE] Conversation completed: {len(steps)} steps taken")
        self.logger.info(f"{'='*60}\n")
        
        return {
            "steps": steps,
            "final_response": message.get('content', ''),
            "conversation": messages
        }

    @observe(name="execute_loop_with_fallback")
    def _execute_loop_with_fallback(self, messages: List[Dict],
                                   tools: List[Dict],
                                   task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM-tool interaction loop with fallback support for L5 tasks.

        Args:
            messages: Conversation messages
            tools: Available tools
            task: Task definition

        Returns:
            Execution result
        """
        # First attempt with original tools
        result = self._execute_loop(messages, tools, task)

        # Check if we need fallback (L5 task with failed tools and fallback_options available)
        fallback_options = task.get('fallback_options')
        if (not result.get('final_response') or
            (fallback_options and self._has_failed_tools(result))):

            self.logger.info(f"[L5 FALLBACK] Original execution failed, trying fallback options")

            # Extract fallback tools from fallback_options
            fallback_tools = []
            for fallback_option in fallback_options:
                tool_name = fallback_option.get('tool')
                if tool_name and tool_name not in fallback_tools:
                    fallback_tools.append(tool_name)
            
            # Also include golden_action tools to give LLM context about original intended tools
            golden_action = task.get('golden_action', [])
            if isinstance(golden_action, dict):
                golden_action = [golden_action]
            
            for action in golden_action:
                if isinstance(action, dict):
                    tool_name = action.get('tool')
                    if tool_name and tool_name != 'reuse' and tool_name not in fallback_tools:
                        fallback_tools.append(tool_name)

            if fallback_tools:
                self.logger.info(f"[L5 FALLBACK] Switching to combined tools (fallback + golden_action): {fallback_tools}")

                # Create new tool schemas for fallback
                available_fallback_tools = []
                for tool_name in fallback_tools:
                    tool = self.tool_registry.get_tool(tool_name)
                    if tool:
                        available_fallback_tools.append(tool.get_schema())
                    else:
                        self.logger.warning(f"Fallback tool '{tool_name}' not found in registry")

                if available_fallback_tools:
                    # Reset conversation to original user message (remove failed attempts)
                    original_user_message = None
                    for msg in messages:
                        if msg.get('role') == 'user':
                            original_user_message = msg
                            break

                    if original_user_message:
                        fallback_messages = [original_user_message]
                        self.logger.info(f"[L5 FALLBACK] Retrying with {len(available_fallback_tools)} fallback tools")

                        # Execute with fallback tools
                        fallback_result = self._execute_loop(fallback_messages, available_fallback_tools, task)

                        # Combine results: original steps + fallback steps
                        combined_steps = result.get('steps', []) + fallback_result.get('steps', [])

                        # Update step numbers for fallback steps
                        original_steps_count = len(result.get('steps', []))
                        for step in fallback_result.get('steps', []):
                            step['step'] += original_steps_count

                        # Combine conversation logs
                        original_conversation = result.get('conversation', messages)
                        fallback_conversation = fallback_result.get('conversation', fallback_messages)

                        # Create combined conversation: original + separator + fallback
                        combined_conversation = list(original_conversation)
                        # Add fallback conversation (skip the repeated user message)
                        for msg in fallback_conversation[1:]:  # Skip first user message as it's duplicate
                            combined_conversation.append(msg)

                        return {
                            "steps": combined_steps,
                            "final_response": fallback_result.get('final_response', ''),
                            "conversation": combined_conversation,
                            "fallback_used": True
                        }

        return result

    def _has_failed_tools(self, result: Dict[str, Any]) -> bool:
        """Check if any tools failed in the execution result."""
        for step in result.get('steps', []):
            for tool_call in step.get('tool_calls', []):
                if not tool_call.get('success', False):
                    return True
        return False
    
    @observe(as_type="generation")
    def _call_llm_with_retry(self, messages: List[Dict], 
                        tools: List[Dict],
                        task_level: Optional[int] = None) -> Dict[str, Any]:
        """Call LLM with retry logic."""
        self.logger.debug(f"Calling LLM with {len(tools)} tools")

        def _is_anthropic_rate_limit(err: Exception) -> bool:
            name = type(err).__name__
            text = str(err).lower()
            model = (getattr(self.adapter, "model_name", "") or "").lower()

            if name == "RateLimitError":
                # LiteLLM raises this for multiple providers; restrict by hints.
                return "anthropic" in text or "claude" in text or "anthropic" in model or "claude" in model

            # Fallback heuristic: Anthropic error payload uses 'rate_limit_error'
            if "rate_limit_error" in text:
                return "anthropic" in text or "claude" in text or "anthropic" in model or "claude" in model

            return False

        def _extract_retry_after_seconds(err: Exception) -> Optional[float]:
            """Best-effort extraction of Retry-After from exception/response headers."""
            # Try common attributes used by httpx/LiteLLM wrappers
            candidates = []
            for attr in ("response", "httpx_response"):
                resp = getattr(err, attr, None)
                if resp is not None:
                    candidates.append(resp)

            headers = getattr(err, "headers", None)
            if headers is None:
                for resp in candidates:
                    headers = getattr(resp, "headers", None)
                    if headers is not None:
                        break

            if not headers:
                return None

            # Normalize dict-like headers
            try:
                def _get(key: str) -> Optional[str]:
                    v = headers.get(key) if hasattr(headers, "get") else None
                    if v is None:
                        v = headers.get(key.lower()) if hasattr(headers, "get") else None
                    return str(v) if v is not None else None

                retry_after = _get("retry-after")
                if retry_after:
                    try:
                        return float(retry_after)
                    except Exception:
                        return None

                # Some providers use epoch reset timestamps
                reset = _get("x-ratelimit-reset") or _get("ratelimit-reset")
                if reset:
                    try:
                        reset_ts = float(reset)
                        now = time.time()
                        if reset_ts > now:
                            return max(0.0, reset_ts - now)
                    except Exception:
                        return None
            except Exception:
                return None

            return None
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # TTFT 측정 시작
                import time
                start_time = time.perf_counter()
                
                result = self.adapter.chat_completion(messages, tools, task_level=task_level)
                
                # TTFT 측정 종료 (API 호출 완료 시점 = 첫 토큰 수신 시점)
                ttft = time.perf_counter() - start_time
                
                # TTFT를 결과에 추가
                result['ttft'] = ttft
                
                if is_enabled() and 'usage' in result:
                    try:
                        langfuse = get_client()
                        langfuse.update_current_span(
                            metadata={
                                "model": result.get('model', self.adapter.model_name),
                                "ttft": ttft,  # ← TTFT 추가
                                "usage": {
                                    "input_tokens": result['usage'].get('prompt_tokens', 0),
                                    "output_tokens": result['usage'].get('completion_tokens', 0),
                                    "total_tokens": result['usage'].get('total_tokens', 0),
                                }
                            }
                        )
                    except Exception as e:
                        self.logger.debug(f"Langfuse update failed: {e}")
                
                return result
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    # Anthropic TPM rate limits are per-minute; short retries amplify failures.
                    if _is_anthropic_rate_limit(e):
                        retry_after = _extract_retry_after_seconds(e)
                        wait_s = max(60.0, float(retry_after)) if retry_after is not None else 60.0
                        self.logger.warning(
                            f"[RATE_LIMIT] Detected Anthropic rate limit; sleeping {wait_s:.1f}s before retry "
                            f"(attempt {attempt + 2}/{self.max_retries})"
                        )
                        time.sleep(wait_s)
                    else:
                        time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"LLM call failed after {self.max_retries} attempts: {str(last_error)}")
    
    @observe(name="execute_tool")
    def _execute_tool_call(self, tool_call: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call with error injection support for Level 5 tasks.
        
        Args:
            tool_call: Tool call information
            task: Task definition (for error_injection check)
            
        Returns:
            Tool execution result
        """
        try:
            function = tool_call.get('function', {})
            tool_name = function.get('name')
            arguments = json.loads(function.get('arguments', '{}'))
            
            # update current span with input 
            if is_enabled():
                try:
                    langfuse = get_client()
                    langfuse.update_current_span(
                        input=arguments,
                        metadata={"tool_name": tool_name}
                    )
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            # Check for error injection (Level 5 robustness testing)
            error_injection = task.get('error_injection')
            if error_injection and error_injection.get('tool') == tool_name:
                error_type = error_injection.get('error_type')
                
                # Simulate different error types
                if error_type == 'timeout':
                    error_message = "Request timeout"
                elif error_type == 'complete_unavailable':
                    error_message = "Service completely unavailable"
                elif error_type == 'data_not_available':
                    error_message = "No data available"
                else:
                    error_message = f"Error: {error_type}"
                
                self.logger.warning(f"[INJECTION] Error injection: {tool_name} - {error_type}")
                
                # Return error result
                error_result = {
                    "tool_call_id": tool_call['id'],
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": None,
                    "success": False,
                    "error": error_message,
                    "error_type": error_type  # For judge evaluation
                }
                
                # update current span with injected error
                if is_enabled():
                    try:
                        langfuse = get_client()
                        langfuse.update_current_span(
                            output=error_result,
                            metadata={"error_injection": True, "error_type": error_type}
                        )
                    except Exception as e:
                        self.logger.debug(f"Langfuse update failed: {e}")
                
                return error_result
            
            # Execute tool normally (may include caching layer under the hood)
            result = self.tool_registry.execute_tool(tool_name, **arguments)
            
            tool_result = {
                "tool_call_id": tool_call['id'],
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": True,
                "error": None
            }
            
            # update current span with output 
            if is_enabled():
                try:
                    langfuse = get_client()
                    # Attach cache metadata if available from wrapper
                    cache_meta = None
                    try:
                        # Access underlying caching executor via tool wrapper if present
                        tool_obj = self.tool_registry.get_tool(tool_name)
                        if hasattr(tool_obj, "_caching_executor"):
                            cache_meta = tool_obj._caching_executor.get_last_meta()
                    except Exception:
                        cache_meta = None
                    langfuse.update_current_span(output=tool_result, metadata={"cache": cache_meta})
                except Exception as e:
                    self.logger.debug(f"Langfuse update failed: {e}")
            
            return tool_result
            
        except Exception as e:
            error_result = {
                "tool_call_id": tool_call['id'],
                "tool_name": function.get('name'),
                "arguments": arguments if 'arguments' in locals() else {},
                "result": None,
                "success": False,
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