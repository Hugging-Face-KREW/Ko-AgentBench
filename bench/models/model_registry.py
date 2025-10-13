"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', 'huggingface/...').
"""

from typing import List

MODEL_IDS: List[str] = [
    # "huggingface/Qwen/Qwen3-4B-Instruct-2507",
    # "huggingface/Qwen/Qwen3-4B-Thinking-2507",
    # "huggingface/Qwen/Qwen2.5-7B-Instruct",
    # "huggingface/moonshotai/Kimi-K2-Instruct",
    # "huggingface/zai-org/GLM-4.5"
    "openai/gpt-4.1",  # OpenAI - provider prefix 추가
    "azure/gpt-4.1"
]

__all__ = ["MODEL_IDS"]


