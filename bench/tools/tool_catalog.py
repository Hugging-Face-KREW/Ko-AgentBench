"""Central tool catalog mapping tool names to API methods and schemas.

This minimal catalog enables registering API class methods as tools based on
task-declared tool names, without changing runner logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from .base_tool import BaseTool
from .method_tool_wrapper import make_method_tool_class
from .naver_search_mock import NaverSearchMockAPI


# Catalog entry: tool_name -> (api_class, method_name, description, parameters_schema)
TOOL_CATALOG: Dict[str, Tuple[Type[Any], str, str, Dict[str, Any]]] = {
    "naver_web_search": (
        NaverSearchMockAPI,
        "WebSearch_naver",
        "네이버 웹 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
    "naver_blog_search": (
        NaverSearchMockAPI,
        "BlogSearch_naver",
        "네이버 블로그 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
    "naver_news_search": (
        NaverSearchMockAPI,
        "NewsSearch_naver",
        "네이버 뉴스 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
}


def resolve_tool_classes(tool_names: List[str]) -> List[Type[BaseTool]]:
    """Resolve given tool names to concrete BaseTool classes via catalog.

    For each API class, a single instance is allocated and reused among
    method-backed tools to avoid redundant setup.
    """
    api_instances: Dict[Type[Any], Any] = {}
    resolved: List[Type[BaseTool]] = []
    seen: set[str] = set()

    for name in tool_names:
        if name in seen:
            continue
        seen.add(name)

        entry = TOOL_CATALOG.get(name)
        if not entry:
            continue
        api_class, method_name, description, parameters_schema = entry

        if api_class not in api_instances:
            api_instances[api_class] = api_class()
        api_instance = api_instances[api_class]

        tool_class = make_method_tool_class(
            name=name,
            description=description,
            api_instance=api_instance,
            method_name=method_name,
            parameters_schema=parameters_schema,
        )
        resolved.append(tool_class)

    return resolved


__all__ = ["TOOL_CATALOG", "resolve_tool_classes"]


