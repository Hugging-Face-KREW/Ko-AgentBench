"""Tool registry for Ko-AgentBench."""

from typing import Dict, List, Optional, Type
from .base_api import BaseTool


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
    
    def register_tool(self, tool_class: Type[BaseTool], **init_kwargs) -> None:
        """Register a tool class.
        
        Args:
            tool_class: Tool class to register
            **init_kwargs: Initialization arguments for the tool
        """
        tool_instance = tool_class(**init_kwargs)
        self._tools[tool_instance.name] = tool_instance
        self._tool_classes[tool_instance.name] = tool_class
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tools_schema(self) -> List[Dict]:
        """Get schema for all registered tools.
        
        Returns:
            List of tool schemas
        """
        return [tool.get_schema() for tool in self._tools.values()]
    
    def execute_tool(self, name: str, **kwargs) -> any:
        """Execute a tool by name.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool not found or invalid input
        """
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        
        if not tool.validate_input(**kwargs):
            raise ValueError(f"Invalid input for tool '{name}'")
        
        return tool.execute(**kwargs)