"""Langfuse logging utilities for Ko-AgentBench."""

import os
import time
from typing import Any, Dict, Optional
from langfuse import Langfuse
from langfuse import observe


class LangfuseLogger:
    """Langfuse client for logging LLM interactions and tool calls."""
    
    def __init__(self, 
                 public_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 host: Optional[str] = None):
        """Initialize Langfuse client.
        
        Args:
            public_key: Langfuse public key (defaults to env var LANGFUSE_PUBLIC_KEY)
            secret_key: Langfuse secret key (defaults to env var LANGFUSE_SECRET_KEY)
            host: Langfuse host (defaults to env var LANGFUSE_HOST or https://cloud.langfuse.com)
        """
        self.client = Langfuse(
            public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
            host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        
    def log_task_start(self, task_id: str, task_name: str, input_data: Dict[str, Any]) -> str:
        """Log the start of a task execution.
        
        Args:
            task_id: Unique task identifier
            task_name: Name of the task
            input_data: Task input data
            
        Returns:
            Trace ID for the task
        """
        trace_id = self.client.create_trace_id()
        print(f"📊 Task started: {task_name} (ID: {trace_id})")
        print(f"📥 Input: {input_data}")
        return trace_id
        
    def log_llm_call(self, 
                     trace_id: str,
                     model: str,
                     messages: list,
                     response: str,
                     latency_ms: float,
                     tool_calls: Optional[list] = None) -> str:
        """Log an LLM call.
        
        Args:
            trace_id: Parent trace ID
            model: Model name used
            messages: Input messages
            response: LLM response
            latency_ms: Response latency in milliseconds
            tool_calls: Tool calls made by the LLM (optional)
            
        Returns:
            Generation ID
        """
        generation_id = f"gen_{int(time.time() * 1000)}"
        print(f"🤖 LLM Call: {model} (latency: {latency_ms:.2f}ms)")
        print(f"📤 Messages: {len(messages)} messages")
        print(f"📥 Response: {response[:100]}...")
        if tool_calls:
            print(f"🔧 Tool calls: {len(tool_calls)} calls")
        return generation_id
        
    def log_tool_call(self,
                     trace_id: str,
                     tool_name: str,
                     tool_input: Dict[str, Any],
                     tool_output: Any,
                     latency_ms: float) -> str:
        """Log a tool call.
        
        Args:
            trace_id: Parent trace ID
            tool_name: Name of the tool called
            tool_input: Tool input parameters
            tool_output: Tool output result
            latency_ms: Tool execution latency in milliseconds
            
        Returns:
            Span ID
        """
        span_id = f"span_{int(time.time() * 1000)}"
        print(f"🔧 Tool Call: {tool_name} (latency: {latency_ms:.2f}ms)")
        print(f"📤 Input: {tool_input}")
        print(f"📥 Output: {str(tool_output)[:100]}...")
        return span_id
        
    def log_task_end(self, trace_id: str, output_data: Dict[str, Any], success: bool = True):
        """Log the end of a task execution.
        
        Args:
            trace_id: Task trace ID
            output_data: Task output data
            success: Whether the task completed successfully
        """
        print(f"✅ Task completed: {trace_id} (success: {success})")
        print(f"📤 Output: {output_data}")
        
    def flush(self):
        """Flush all pending logs to Langfuse."""
        self.client.flush()


def get_langfuse_logger() -> LangfuseLogger:
    """Get a configured Langfuse logger instance.
    
    Returns:
        Configured LangfuseLogger instance
    """
    return LangfuseLogger()


@observe()
def log_with_decorator(func):
    """Decorator for automatic logging of function calls."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            latency_ms = (time.time() - start_time) * 1000
            
            # Log successful execution
            logger = get_langfuse_logger()
            logger.log_tool_call(
                trace_id="decorator_trace",
                tool_name=func.__name__,
                tool_input={"args": args, "kwargs": kwargs},
                tool_output=result,
                latency_ms=latency_ms
            )
            
            return result
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            
            # Log error
            logger = get_langfuse_logger()
            logger.log_tool_call(
                trace_id="decorator_trace",
                tool_name=func.__name__,
                tool_input={"args": args, "kwargs": kwargs},
                tool_output={"error": str(e)},
                latency_ms=latency_ms
            )
            
            raise e
    return wrapper
