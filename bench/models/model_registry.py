"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', 'huggingface/...').
"""

from typing import List

MODEL_IDS: List[str] = [
    "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    "azure/gpt-4.1"
]

__all__ = ["MODEL_IDS"]