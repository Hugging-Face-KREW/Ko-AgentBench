"""Sample tool implementations."""

from typing import Any, Dict
from .base_api import BaseTool


class CalculatorTool(BaseTool):
    """Simple calculator tool."""
    
    def __init__(self):
        super().__init__("calculator", "Performs basic arithmetic operations")
    
    def execute(self, operation: str, a: float, b: float) -> float:
        """Execute calculation.
        
        Args:
            operation: Operation type (add, subtract, multiply, divide)
            a: First number
            b: Second number
            
        Returns:
            Calculation result
        """
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    def validate_input(self, **kwargs) -> bool:
        """Validate calculator input."""
        required_params = ['operation', 'a', 'b']
        for param in required_params:
            if param not in kwargs:
                return False
        
        valid_operations = ['add', 'subtract', 'multiply', 'divide']
        return kwargs.get('operation') in valid_operations
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get calculator parameters schema."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "Arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number", 
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        }


class FileReaderTool(BaseTool):
    """File reading tool."""
    
    def __init__(self):
        super().__init__("file_reader", "Reads content from files")
    
    def execute(self, file_path: str) -> str:
        """Read file content.
        
        Args:
            file_path: Path to file to read
            
        Returns:
            File content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {str(e)}")
    
    def validate_input(self, **kwargs) -> bool:
        """Validate file reader input."""
        return 'file_path' in kwargs and isinstance(kwargs['file_path'], str)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get file reader parameters schema."""
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }