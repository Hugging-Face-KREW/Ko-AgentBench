"""Base interfaces for Ko-AgentBench tools and APIs."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseAPI(ABC):
    """Abstract base class for all APIs."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize API.
        
        Args:
            name: API name
            description: API description
        """
        self.name = name
        self.description = description
    
    def get_schema(self) -> Dict[str, Any]:
        """Get API schema for LLM function calling.
        
        Returns:
            API schema dictionary with all available methods
        """
        return {
            "name": self.name,
            "description": self.description
        }


class BaseTool(ABC):
    """Abstract base class for executable tools.
    
    Tools expose a function-calling schema for LLMs and must implement
    parameter validation and execution logic.
    """
    
    def __init__(self, name: str, description: str = ""):
        """Initialize tool.
        
        Args:
            name: Tool name (unique identifier)
            description: Tool description for LLMs/users
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with validated parameters."""
        raise NotImplementedError
    
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters before execution."""
        raise NotImplementedError
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON Schema for input parameters."""
        raise NotImplementedError
    
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI-style tool/function schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema()
            }
        }
