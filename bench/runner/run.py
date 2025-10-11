"""Main benchmark runner for Ko-AgentBench."""

import json
import time
import logging
from typing import Any, Dict, List

from ..adapters.base_adapter import BaseAdapter
from ..tools.tool_registry import ToolRegistry
from .judge import Judge
from ..observability import observe, get_client, is_enabled


class BenchmarkRunner:
    """Main runner for executing benchmarks with LLM-tool loops."""
    
    def __init__(self, 
                 adapter: BaseAdapter,
                 tool_registry: ToolRegistry,
                 judge: Judge,
                 max_steps: int = 10,
                 timeout: int = 300,
                 max_retries: int = 3):
        """Initialize benchmark runner.
        
        Args:
            adapter: LLM adapter for API calls
            tool_registry: Registry of available tools
            judge: Judge for evaluating results
            max_steps: Maximum number of steps per task
            timeout: Timeout in seconds per task
            max_retries: Maximum retries for failed API calls
        """
        self.adapter = adapter
        self.tool_registry = tool_registry
        self.judge = judge
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
        task_id = task.get('id', 'unknown')
        
        if is_enabled():
            try:
                langfuse = get_client()
                langfuse.update_current_trace(
                    name=f"Task: {task_id}",
                    metadata={
                        "task_id": task_id,
                        "category": task.get("category"),
                        "difficulty": task.get("difficulty"),
                    },
                    tags=[task.get("category", "unknown"), task.get("difficulty", "unknown")]
                )
            except Exception as e:
                self.logger.debug(f"Langfuse update failed: {e}")

        self.logger.info(f"Starting task {task_id}")
        
        try:
            # Initialize conversation with task description
            messages = [
                {"role": "user", "content": task.get('description', '')}
            ]
            
            # Get available tools for this task
            task_tools = task.get('tools', [])
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
                self.logger.warning("⚠️ NO TOOLS AVAILABLE FOR THIS TASK!")
            
            # Execute LLM-tool loop
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
                    })
            
            # Judge the result
            evaluation = self.judge.evaluate(task, result)
            
            execution_time = time.time() - start_time
            
            final_result = {
                "task_id": task_id,
                "success": evaluation.get('success', False),
                "result": result,
                "tool_invocations": tool_invocations,
                "evaluation": evaluation,
                "execution_time": execution_time,
                "steps_taken": len(result.get('steps', [])),
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
        
        for step in range(self.max_steps):
            # Check timeout
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Task exceeded timeout of {self.timeout} seconds")
            
            # Get LLM response
            response = self._call_llm_with_retry(messages, tools)
            
            step_data = {
                "step": step + 1,
                "llm_response": response,
                "tool_calls": [],
                "timestamp": time.time()
            }
            
            # Handle tool calls
            message = response.get('message', {})
            
            # Add assistant message to conversation FIRST (before tool results)
            messages.append(message)
            
            if 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    tool_result = self._execute_tool_call(tool_call)
                    step_data['tool_calls'].append(tool_result)
                    
                    # Add tool result to messages AFTER assistant message
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(tool_result['result'])
                    })
            
            steps.append(step_data)
            
            # Check if task is complete (no more tool calls)
            if 'tool_calls' not in message or not message['tool_calls']:
                break
        
        return {
            "steps": steps,
            "final_response": message.get('content', ''),
            "conversation": messages
        }
    
    @observe(as_type="generation")
    def _call_llm_with_retry(self, messages: List[Dict], 
                           tools: List[Dict]) -> Dict[str, Any]:
        """Call LLM with retry logic.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            
        Returns:
            LLM response
        """
        # DEBUG: Log before calling LLM
        self.logger.debug(f"Calling LLM with {len(tools)} tools")
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = self.adapter.chat_completion(messages, tools)
                
                if is_enabled() and 'usage' in result:
                    try:
                        langfuse = get_client()
                        langfuse.update_current_span(
                            metadata={
                                "model": result.get('model', self.adapter.model_name),
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
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool call.
        
        Args:
            tool_call: Tool call information
            
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
                    langfuse.update_current_span(
                        output=tool_result
                    )
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