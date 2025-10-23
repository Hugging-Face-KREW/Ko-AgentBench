"""Main benchmark runner for Ko-AgentBench."""

import json
import time
import logging
from typing import Any, Dict, List

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
                 max_retries: int = 3):
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
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
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
            
            # Get available tools for this task
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
            
            # DEBUG: Log tools being passed to LLM
            self.logger.info(f"Task tools requested: {task_tools}")
            self.logger.info(f"Available tools count: {len(available_tools)}")
            if available_tools:
                self.logger.info(f"First tool schema keys: {list(available_tools[0].keys())}")
            else:
                self.logger.warning("[WARNING] NO TOOLS AVAILABLE FOR THIS TASK!")
            
            # Check if this is a multi-turn conversation task (L6/L7)
            conversation = task.get('conversation_tracking') or task.get('conversation')
            if isinstance(conversation, dict) and isinstance(conversation.get('turns'), list):
                # Multi-turn conversation: execute each user turn sequentially
                result = self._execute_multiturn_conversation(conversation, available_tools, task)
            else:
                # Single-turn task: execute once
                task_description = task.get('instruction') or task.get('description', '')
                messages = [{"role": "user", "content": task_description}]
                result = self._execute_loop(messages, available_tools, task)
            
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
            
            # Add tool_invocations to result
            result['tool_invocations'] = tool_invocations
            
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
                "tool_invocations": tool_invocations,
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
        
        # Initialize conversation (no system prompt as per requirement)
        messages: List[Dict[str, Any]] = []
        
        all_steps = []
        step_counter = 0
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[MULTI-TURN] Multi-turn conversation: {len(turns)} turns")
        self.logger.info(f"{'='*80}\n")
        
        # Process each user turn sequentially
        for turn_idx, turn in enumerate(turns, 1):
            role = turn.get('role', '')
            content = turn.get('content', '')
            
            # Skip turns without user content (assistant action markers, etc.)
            if role != 'user' or not content:
                continue
            
            self.logger.info(f"\n{'─'*80}")
            self.logger.info(f"[TURN {turn_idx}] Processing user message")
            self.logger.info(f"{'─'*80}")
            self.logger.info(f"User: {content[:200]}...")
            
            # Add user message to conversation
            messages.append({"role": "user", "content": content})
            
            # Call LLM with current context
            try:
                response = self._call_llm_with_retry(messages, tools)
                message = response.get('message', {})
                
                step_counter += 1
                step_data = {
                    "step": step_counter,
                    "turn": turn_idx,
                    "llm_response": response,
                    "tool_calls": [],
                    "timestamp": time.time(),
                    "ttft": response.get('ttft', 0)
                }
                
                # Log assistant response
                assistant_content = message.get('content', '')
                if assistant_content:
                    self.logger.info(f"[ASSISTANT] {assistant_content[:200]}...")
                
                # Add assistant message to conversation
                messages.append(message)
                
                # Handle tool calls if present
                if 'tool_calls' in message and message['tool_calls']:
                    self.logger.info(f"[TOOL] Tool calls: {len(message['tool_calls'])}")
                    
                    for idx, tool_call in enumerate(message['tool_calls'], 1):
                        tool_name = tool_call.get('function', {}).get('name', 'unknown')
                        tool_args = tool_call.get('function', {}).get('arguments', '{}')
                        self.logger.info(f"  [{idx}] {tool_name}")
                        self.logger.info(f"      Args: {tool_args[:150]}...")
                        
                        # Execute tool
                        tool_result = self._execute_tool_call(tool_call, task)
                        step_data['tool_calls'].append(tool_result)
                        
                        # Log result
                        if tool_result.get('success'):
                            result_preview = str(tool_result.get('result', ''))[:150]
                            self.logger.info(f"      [SUCCESS] {result_preview}...")
                        else:
                            self.logger.error(f"      [ERROR] {tool_result.get('error')}")
                        
                        # Add tool result to conversation
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call['id'],
                            "content": json.dumps(tool_result['result'])
                        })
                    
                    # Get final response after tool execution
                    self.logger.info(f"[INFO] Getting final response after tool execution...")
                    final_response = self._call_llm_with_retry(messages, tools)
                    final_message = final_response.get('message', {})
                    
                    # Log final response
                    final_content = final_message.get('content', '')
                    if final_content:
                        self.logger.info(f"[RESPONSE] Final response: {final_content[:200]}...")
                    
                    # Add final response to conversation
                    messages.append(final_message)
                    
                    # Update step data with final response
                    step_counter += 1
                    step_data_final = {
                        "step": step_counter,
                        "turn": turn_idx,
                        "llm_response": final_response,
                        "tool_calls": [],
                        "timestamp": time.time()
                    }
                    all_steps.append(step_data_final)
                
                all_steps.append(step_data)
                
            except Exception as e:
                self.logger.error(f"[ERROR] Error processing turn {turn_idx}: {e}")
                # Continue to next turn even if this one fails
                continue
        
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[COMPLETE] Multi-turn conversation completed: {len(all_steps)} steps")
        self.logger.info(f"{'='*80}\n")
        
        # Get final response from last message
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
        
        # DEBUG: Log tools being used in loop
        self.logger.info(f"_execute_loop: Received {len(tools)} tools")
        if tools:
            tool_names = [t.get('function', {}).get('name', 'unknown') for t in tools]
            self.logger.info(f"Tool names in loop: {tool_names}")
        
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
            response = self._call_llm_with_retry(messages, tools)
            
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
    
    @observe(as_type="generation")
    def _call_llm_with_retry(self, messages: List[Dict], 
                        tools: List[Dict]) -> Dict[str, Any]:
        """Call LLM with retry logic."""
        self.logger.debug(f"Calling LLM with {len(tools)} tools")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # TTFT 측정 시작
                import time
                start_time = time.perf_counter()
                
                result = self.adapter.chat_completion(messages, tools)
                
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