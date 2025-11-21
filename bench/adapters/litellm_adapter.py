"""LiteLLM adapter for Ko-AgentBench."""

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
        
        # Drop unsupported parameters for models that don't support them
        litellm.drop_params = True
        
        # Enable JSON schema validation for better Pydantic/Gemini compatibility
        litellm.enable_json_schema_validation = True
    
    @observe(as_type="generation")
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion using LiteLLM.
        
        Args:
            messages: List of chat messages
            tools: Available tools for function calling
            **kwargs: Additional parameters (including 'n' for multiple completions)
            
        Returns:
            Chat completion response (or list of responses if n > 1)
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