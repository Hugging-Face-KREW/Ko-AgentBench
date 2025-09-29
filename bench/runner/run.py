"""Main benchmark runner for Ko-AgentBench."""

import json
import time
import logging
from typing import Any, Dict, List

from ..adapters.base_adapter import BaseAdapter
from ..tools.tool_registry import ToolRegistry
from .judge import Judge


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
    
    def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark task.
        
        Args:
            task: Task definition dictionary
            
        Returns:
            Task execution result including tool invocation summary
        """
        start_time = time.time()
        task_id = task.get('id', 'unknown')
        
        self.logger.info(f"Starting task {task_id}")
        
        try:
            # Initialize conversation with task description
            messages = [
                {"role": "system", "content": "You are a helpful assistant that can use tools to complete tasks."},
                {"role": "user", "content": task.get('description', '')}
            ]
            
            # Get available tools for this task
            task_tools = task.get('tools', [])
            available_tools = []
            for tool_name in task_tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    available_tools.append(tool.get_schema())
            
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
            
            return {
                "task_id": task_id,
                "success": evaluation.get('success', False),
                "result": result,
                "tool_invocations": tool_invocations,
                "evaluation": evaluation,
                "execution_time": execution_time,
                "steps_taken": len(result.get('steps', [])),
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task {task_id} failed: {str(e)}")
            
            return {
                "task_id": task_id,
                "success": False,
                "result": None,
                "evaluation": None,
                "execution_time": execution_time,
                "steps_taken": 0,
                "error": str(e)
            }
    
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
            if 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    tool_result = self._execute_tool_call(tool_call)
                    step_data['tool_calls'].append(tool_result)
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(tool_result['result'])
                    })
            
            # Add assistant message to conversation
            messages.append(message)
            steps.append(step_data)
            
            # Check if task is complete (no more tool calls)
            if 'tool_calls' not in message or not message['tool_calls']:
                break
        
        return {
            "steps": steps,
            "final_response": message.get('content', ''),
            "conversation": messages
        }
    
    def _call_llm_with_retry(self, messages: List[Dict], 
                           tools: List[Dict]) -> Dict[str, Any]:
        """Call LLM with retry logic.
        
        Args:
            messages: Conversation messages
            tools: Available tools
            
        Returns:
            LLM response
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.adapter.chat_completion(messages, tools)
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"LLM call failed after {self.max_retries} attempts: {str(last_error)}")
    
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
            
            result = self.tool_registry.execute_tool(tool_name, **arguments)
            
            return {
                "tool_call_id": tool_call['id'],
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "tool_call_id": tool_call['id'],
                "tool_name": function.get('name'),
                "arguments": arguments if 'arguments' in locals() else {},
                "result": None,
                "success": False,
                "error": str(e)
            }