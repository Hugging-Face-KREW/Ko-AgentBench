"""Base API interface for Ko-AgentBench."""

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