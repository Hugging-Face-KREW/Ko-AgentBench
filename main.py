from typing import Any

from bench.tools.tool_registry import ToolRegistry
from bench.tools.naver_search_mock import (
    NaverWebSearchMock,
    NaverBlogSearchMock,
    NaverNewsSearchMock,
)


def create_default_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry pre-registered with default mock tools."""
    registry = ToolRegistry()
    registry.register_tool(NaverWebSearchMock)
    registry.register_tool(NaverBlogSearchMock)
    registry.register_tool(NaverNewsSearchMock)
    return registry


def main() -> None:
    print("Hello from ko-agentbench!")
    registry = create_default_tool_registry()
    print("Registered tools:", registry.get_available_tools())
    # Sample call for quick verification (safe, no network calls)
    sample: Any = registry.execute_tool("naver_web_search", query="테스트", display=2, start=1)
    print("Sample web search result (truncated items count):", len(sample.get("items", [])))


if __name__ == "__main__":
    main()
