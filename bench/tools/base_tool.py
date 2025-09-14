"""Base tool interface for Ko-AgentBench."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for LLM function calling.
        
        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema.
        
        Returns:
            Parameters schema dictionary
        """
        pass