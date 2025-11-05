"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', '...').
"""

from typing import List

MODEL_IDS: List[str] = [
    # Local HuggingFace models
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "skt/A.X-4.0-Light",
    "K-intelligence/Midm-2.0-Base-Instruct",
    "KORMo-Team/KORMo-10B-sft",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "dnotitia/DNA-2.0-14B",
    "trillionlabs/Tri-7B",

    # Cloud models
    "azure/gpt-5",
    "azure/gpt-4.1",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4-20250514",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-2.5-flash",
    "openrouter/anthropic/claude-3.5-sonnet",
    "openrouter/openai/gpt-4o-mini",
]

__all__ = ["MODEL_IDS"]
