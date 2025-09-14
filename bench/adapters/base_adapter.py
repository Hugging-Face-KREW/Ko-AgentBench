"""Base adapter interface for Ko-AgentBench."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    def __init__(self, model_name: str, **config):
        """Initialize adapter.
        
        Args:
            model_name: Name of the model
            **config: Model-specific configuration
        """
        self.model_name = model_name
        self.config = config
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], 
                       tools: Optional[List[Dict]] = None,
                       **kwargs) -> Dict[str, Any]:
        """Generate chat completion.
        
        Args:
            messages: List of chat messages in canonical format
            tools: Available tools for function calling
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response in canonical format
        """
        pass
    
    @abstractmethod  
    def convert_to_provider_format(self, messages: List[Dict[str, str]], 
                                  tools: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Convert canonical format to provider-specific format.
        
        Args:
            messages: Messages in canonical format
            tools: Tools in canonical format
            
        Returns:
            Provider-specific format
        """
        pass
    
    @abstractmethod
    def convert_from_provider_format(self, response: Any) -> Dict[str, Any]:
        """Convert provider-specific response to canonical format.
        
        Args:
            response: Provider-specific response
            
        Returns:
            Response in canonical format
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "config": self.config
        }