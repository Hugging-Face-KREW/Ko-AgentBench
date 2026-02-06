"""Ko-AgentBench: Korean Agent Benchmarking Framework.

A comprehensive benchmarking framework for evaluating AI agents on Korean language tasks.
"""

from .tasks import TaskLoader

# Optional imports for lightweight utility scripts (e.g., schema validation)
# that should not require full runtime dependencies.
try:
    from .tools import ToolRegistry, BaseTool
except Exception:  # pragma: no cover - optional fallback
    ToolRegistry = None
    BaseTool = None

try:
    from .adapters import BaseAdapter, LiteLLMAdapter, OpenRouterAdapter
except Exception:  # pragma: no cover - optional fallback
    BaseAdapter = None
    LiteLLMAdapter = None
    OpenRouterAdapter = None

try:
    from .runner import BenchmarkRunner
except Exception:  # pragma: no cover - optional fallback
    BenchmarkRunner = None

__version__ = "0.1.0"
__all__ = [
    'TaskLoader',
    'ToolRegistry', 
    'BaseTool',
    'BaseAdapter',
    'LiteLLMAdapter',
    'OpenRouterAdapter',
    'BenchmarkRunner'
]
