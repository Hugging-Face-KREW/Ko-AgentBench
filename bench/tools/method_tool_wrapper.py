"""Lightweight wrapper to expose API class methods as BaseTool tools.

This allows tasks to reference method-backed tools by name while the runner
interacts with them via the existing ToolRegistry and BaseTool interfaces.
"""

from typing import Any, Dict

from .base_api import BaseTool
from .caching_executor import CachingExecutor


class MethodToolWrapper(BaseTool):
    """Wrap an API instance method as a BaseTool-compatible tool."""

    def __init__(
        self,
        name: str,
        description: str,
        api_instance: Any,
        method_name: str,
        parameters_schema: Dict[str, Any],
    ) -> None:
        super().__init__(name=name, description=description)
        self._api_instance = api_instance
        self._method_name = method_name
        self._parameters_schema = parameters_schema or {"type": "object"}
        self._caching_executor = CachingExecutor()

    def validate_input(self, **kwargs) -> bool:
        """Minimal JSON-schema-like validation (required keys and basic types)."""
        schema = self._parameters_schema or {}
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # Required keys check
        for key in required:
            if key not in kwargs:
                return False

        # Basic type checks for 'string' and 'integer'
        for key, prop in properties.items():
            if key not in kwargs:
                continue
            expected_type = prop.get("type")
            if expected_type == "string" and not isinstance(kwargs[key], str):
                return False
            if expected_type == "integer" and not isinstance(kwargs[key], int):
                return False

            # Range checks for integers
            if expected_type == "integer":
                minimum = prop.get("minimum")
                maximum = prop.get("maximum")
                if minimum is not None and kwargs[key] < minimum:
                    return False
                if maximum is not None and kwargs[key] > maximum:
                    return False

        return True

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return self._parameters_schema

    def execute(self, **kwargs) -> Any:
        method = getattr(self._api_instance, self._method_name)
        return self._caching_executor.execute(
            tool_name=self.name,
            description=self.description,
            parameters_schema=self._parameters_schema,
            api_callable=method,
            raw_args=kwargs,
        )


def make_method_tool_class(
    name: str,
    description: str,
    api_instance: Any,
    method_name: str,
    parameters_schema: Dict[str, Any],
):
    """Create a concrete BaseTool class for registry registration."""

    class _MethodTool(MethodToolWrapper):
        def __init__(self) -> None:
            super().__init__(
                name=name,
                description=description,
                api_instance=api_instance,
                method_name=method_name,
                parameters_schema=parameters_schema,
            )

    _MethodTool.__name__ = f"MethodTool_{name}"
    return _MethodTool


