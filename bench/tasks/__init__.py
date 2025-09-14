"""Tasks module for Ko-AgentBench.

Contains task definitions in JSONL format with:
- Task descriptions
- Ground truth answers  
- Tool graphs
- Oracle functions
"""

from .task_loader import TaskLoader

__all__ = ['TaskLoader']