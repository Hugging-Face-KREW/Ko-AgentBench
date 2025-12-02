"""LiteLLM adapter for Ko-AgentBench."""

import os
import time
import json
import requests
from typing import Any, Dict, List, Optional
from .base_adapter import BaseAdapter
from ..observability import observe

# Lazy import litellm
litellm = None


class LiteLLMAdapter(BaseAdapter):
    """LiteLLM adapter for multiple LLM providers."""
    
    def __init__(self, model_name: str, **config):
        """Initialize LiteLLM adapter.
        
        Args:
            model_name: Model name (e.g., 'gpt-3.5-turbo', 'claude-3-sonnet')
            **config: Configuration parameters
        """
        global litellm
        if litellm is None:
            try:
                import litellm
            except ImportError:
                raise ImportError("litellm package is required for LiteLLMAdapter")
            
        super().__init__(model_name, **config)
        
        # Set default configuration
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1024)
        self.timeout = config.get('timeout', 60)
        
        # GPT-5 reasoning control parameters (default: low to minimize token usage)
        self.effort = config.get('effort', 'low')  # low, medium, high
        self.max_completion_tokens = config.get('max_completion_tokens', 1024)
        
        # Azure Responses API configuration for GPT-5
        # Responses API provides single-call execution with reasoning control
        self.is_gpt5 = 'gpt-5' in model_name.lower()
        self.azure_endpoint = os.getenv('AZURE_API_BASE', '')
        self.azure_api_key = os.getenv('AZURE_API_KEY', '')
        self.azure_api_version = os.getenv('AZURE_API_VERSION', '')
        
        # Session management for multi-turn conversations (Responses API)
        self.previous_response_id = None  # Track previous response for conversation continuity
        
        # Drop unsupported parameters for models that don't support them
        litellm.drop_params = True
        
        # Enable JSON schema validation for better Pydantic/Gemini compatibility
        litellm.enable_json_schema_validation = True
    
    @observe(as_type="generation")
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       task_level: Optional[int] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using LiteLLM Responses API or Chat Completions.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            task_level: Task level (used for reasoning control on GPT-5)
            **kwargs: Additional parameters (including 'n' for multiple completions)
            
        Returns:
            Chat completion response (or list of responses if n > 1)
        """
        # Use Azure Responses API for GPT-5 to control reasoning tokens
        if self.is_gpt5:
            return self._chat_completion_responses_api(messages, tools, task_level, **kwargs)
        
        # Convert to provider format
        provider_request = self.convert_to_provider_format(messages, tools)
        
        # Add kwargs to provider request
        provider_request.update(kwargs)
        
        # Make API call through LiteLLM
        response = litellm.completion(**provider_request)
        # ...existing code...
        
        # Convert back to canonical format
        return self.convert_from_provider_format(response)
    
    def convert_to_provider_format(self, messages: List[Dict[str, str]], 
                                  tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Convert to LiteLLM format.
        
        Args:
            messages: Messages in canonical format
            tools: Tools in canonical format
            
        Returns:
            LiteLLM request format
        """
        request = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout
        }
        
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        
        return request
    
    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Convert LiteLLM response to canonical format.
        
        Args:
            response: LiteLLM response object
            
        Returns:
            Canonical format response (or list of responses if n > 1)
        """
        # Handle multiple choices (when n > 1)
        if len(response.choices) > 1:
            results = []
            for choice in response.choices:
                result = {
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content or ""
                    },
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    },
                    "model": response.model,
                    "finish_reason": choice.finish_reason
                }
                
                # Handle function calls
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    result["message"]["tool_calls"] = []
                    for tool_call in choice.message.tool_calls:
                        result["message"]["tool_calls"].append({
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })
                
                results.append(result)
            
            return {"choices": results, "n": len(results)}
        
        # Single choice (default behavior)
        choice = response.choices[0]
        
        result = {
            "message": {
                "role": choice.message.role,
                "content": choice.message.content or ""
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model,
            "finish_reason": choice.finish_reason
        }
        
        # Handle function calls
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            result["message"]["tool_calls"] = []
            for tool_call in choice.message.tool_calls:
                result["message"]["tool_calls"].append({
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        return result
    
    def reset_conversation(self):
        """Reset conversation state (call between tasks)."""
        # Reset Responses API session
        self.previous_response_id = None
    
    def _chat_completion_responses_api(self, messages: List[Dict[str, str]],
                                       tools: Optional[List[Dict]] = None,
                                       task_level: Optional[int] = None,
                                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Azure Responses API for GPT-5.
        
        Responses API provides:
        - Single REST call (vs 5-10+ calls in Assistants API)
        - Native reasoning control via reasoning parameter
        - Automatic tool calling support
        - Multi-turn conversation via input parameter
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            task_level: Task level for reasoning control
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response in canonical format
        """
        # Convert messages to Responses API input format
        input_data = self._convert_messages_to_responses_input(messages)
        
        # Build Responses API request
        responses_request = {
            "model": self.model_name,
            "input": input_data,
            "max_output_tokens": self.max_completion_tokens,
            "reasoning": {"effort": self.effort},
        }
        
        # Add tools if provided (convert to Responses API format)
        if tools:
            responses_request["tools"] = self._convert_tools_to_responses_format(tools)
            responses_request["tool_choice"] = "auto"
        
        # Add previous response ID for conversation continuity
        if self.previous_response_id:
            responses_request["previous_response_id"] = self.previous_response_id
        
        # Add Azure-specific parameters
        responses_request["api_base"] = self.azure_endpoint
        responses_request["api_key"] = self.azure_api_key
        responses_request["api_version"] = self.azure_api_version
        
        # Make API call through LiteLLM Responses API
        response = litellm.responses(**responses_request)
        
        # Store response ID for next turn (within same task)
        if hasattr(response, 'id'):
            self.previous_response_id = response.id
        
        # Convert Responses API response to canonical format
        return self._convert_responses_to_canonical(response)
    
    def _convert_tools_to_responses_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert Chat Completions tools format to Responses API format.
        
        Chat Completions: {"type": "function", "function": {"name": "...", "parameters": {...}}}
        Responses API: {"type": "function", "name": "...", "parameters": {...}}
        """
        responses_tools = []
        for tool in tools:
            if tool.get('type') == 'function':
                function_def = tool.get('function', {})
                responses_tools.append({
                    "type": "function",
                    "name": function_def.get('name'),
                    "description": function_def.get('description'),
                    "parameters": function_def.get('parameters')
                })
            else:
                # Non-function tools (keep as-is)
                responses_tools.append(tool)
        return responses_tools
    
    def _convert_messages_to_responses_input(self, messages: List[Dict[str, str]]):
        """Convert messages list to Responses API input format.
        
        Responses API accepts:
        - Simple string for single message
        - List of input items for multi-turn conversations with tool calls
        
        IMPORTANT: Only include COMPLETED function_call/function_call_output pairs.
        - function_call without matching output = SKIP (incomplete, will cause API error)
        - Assistant messages with tool_calls = convert only if output exists
        - Assistant text messages = include as regular messages
        """
        input_items = []
        processed_tool_results = set()  # Track which tool results we've already included
        
        for i, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'user':
                input_items.append({
                    "type": "message",
                    "role": "user",
                    "content": content
                })
            elif role == 'assistant':
                tool_calls = msg.get('tool_calls')
                if tool_calls:
                    # Assistant with tool calls - only add if ALL tool calls have results
                    for tool_call in tool_calls:
                        if tool_call.get('type') == 'function':
                            call_id = tool_call['id']
                            
                            # Look ahead for matching tool result
                            found_result = False
                            for j in range(i + 1, len(messages)):
                                if messages[j].get('role') == 'tool' and messages[j].get('tool_call_id') == call_id:
                                    # Found matching result - add the pair
                                    input_items.append({
                                        "type": "function_call",
                                        "call_id": call_id,
                                        "name": tool_call['function']['name'],
                                        "arguments": tool_call['function']['arguments']
                                    })
                                    input_items.append({
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": messages[j].get('content', '')
                                    })
                                    processed_tool_results.add(j)
                                    found_result = True
                                    break
                                # Stop looking if we hit another assistant or user message
                                if messages[j].get('role') in ['assistant', 'user']:
                                    break
                            
                            # If no result found, skip this function_call entirely
                            # (it's incomplete and would cause API error)
                elif content:
                    # Regular assistant message with text
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": content
                    })
            elif role == 'tool':
                # Tool results are handled in the look-ahead above
                # Skip orphaned tool results that weren't matched
                pass
        
        # If only one user message, return as simple string
        if len(input_items) == 1 and input_items[0].get('type') == 'message':
            return input_items[0]['content']
        
        return input_items
    
    def _convert_responses_to_canonical(self, response: Any) -> Dict[str, Any]:
        """Convert Responses API response to canonical format."""
        # Extract output items
        output_items = response.output if hasattr(response, 'output') else []
        
        # Initialize result
        result = {
            "message": {
                "role": "assistant",
                "content": ""
            },
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "model": self.model_name,
            "finish_reason": "stop"
        }
        
        # Extract usage if available
        if hasattr(response, 'usage'):
            usage = response.usage
            result["usage"] = {
                "prompt_tokens": getattr(usage, 'input_tokens', 0),
                "completion_tokens": getattr(usage, 'output_tokens', 0),
                "total_tokens": getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
            }
        
        # Process output items
        tool_calls = []
        text_content = ""
        
        for item in output_items:
            item_type = item.get('type') if isinstance(item, dict) else getattr(item, 'type', None)
            
            if item_type == 'message':
                # ResponseOutputMessage - extract text from content list
                content_list = item.get('content') if isinstance(item, dict) else getattr(item, 'content', [])
                
                # Content is a list of ResponseOutputText objects
                for content_item in content_list:
                    content_type = content_item.get('type') if isinstance(content_item, dict) else getattr(content_item, 'type', None)
                    if content_type == 'output_text':
                        # Extract text from ResponseOutputText
                        if isinstance(content_item, dict):
                            text_content = content_item.get('text', '')
                        else:
                            text_content = getattr(content_item, 'text', '')
                    
            elif item_type == 'function_call':
                # ResponseFunctionToolCall - convert to canonical format
                if isinstance(item, dict):
                    tool_calls.append({
                        "id": item.get('call_id') or item.get('id'),
                        "type": "function",
                        "function": {
                            "name": item.get('name'),
                            "arguments": item.get('arguments')
                        }
                    })
                else:
                    tool_calls.append({
                        "id": getattr(item, 'call_id', None) or getattr(item, 'id', None),
                        "type": "function",
                        "function": {
                            "name": getattr(item, 'name', ''),
                            "arguments": getattr(item, 'arguments', '')
                        }
                    })
            # Ignore 'reasoning' type items
        
        result["message"]["content"] = text_content
        
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
            result["finish_reason"] = "tool_calls"
        
        return result