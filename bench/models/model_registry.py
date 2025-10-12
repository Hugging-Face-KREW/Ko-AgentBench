"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', 'huggingface/...').
"""

from typing import List

MODEL_IDS: List[str] = [
    "anthropic/claude-3-7-sonnet-latest",
    "anthropic/claude-3-5-sonnet-latest",
    "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    "huggingface/Qwen/Qwen3-4B-Thinking-2507",
    "huggingface/Qwen/Qwen2.5-7B-Instruct",
    "huggingface/moonshotai/Kimi-K2-Instruct",
    "huggingface/zai-org/GLM-4.5"
]

__all__ = ["MODEL_IDS"]