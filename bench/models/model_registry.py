"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', 'huggingface/...').
"""

from typing import List

MODEL_IDS: List[str] = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet",
    "groq/gemma-7b-it",
    "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    "huggingface/Qwen/Qwen2.5-7B-Instruct"
]

__all__ = ["MODEL_IDS"]


