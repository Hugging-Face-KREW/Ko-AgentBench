"""LiteLLM adapter for Ko-AgentBench."""

from typing import Any, Dict, List, Optional
from .base_adapter import BaseAdapter

try:
    import litellm
except ImportError:
    litellm = None


class LiteLLMAdapter(BaseAdapter):
    """LiteLLM adapter for multiple LLM providers."""
    
    def __init__(self, model_name: str, **config):
        """Initialize LiteLLM adapter.
        
        Args:
            model_name: Model name (e.g., 'gpt-3.5-turbo', 'claude-3-sonnet')
            **config: Configuration parameters
        """
        if litellm is None:
            raise ImportError("litellm package is required for LiteLLMAdapter")
            
        super().__init__(model_name, **config)
        
        # Set default configuration
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 1024)
        self.timeout = config.get('timeout', 60)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using LiteLLM.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        # Convert to provider format
        provider_request = self.convert_to_provider_format(messages, tools)
        
        # Add kwargs to provider request
        provider_request.update(kwargs)
        
        # Make API call through LiteLLM
        response = litellm.completion(**provider_request)
        
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
            Canonical format response
        """
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