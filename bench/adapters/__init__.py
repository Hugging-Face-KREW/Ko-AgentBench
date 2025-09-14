"""Adapters module for Ko-AgentBench.

Contains LLM adapter implementations using LiteLLM and other libraries
for converting between canonical and provider-specific formats.
"""

from .base_adapter import BaseAdapter
from .litellm_adapter import LiteLLMAdapter

__all__ = ['BaseAdapter', 'LiteLLMAdapter']