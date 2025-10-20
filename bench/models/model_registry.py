"""
model ID list for Ko-AgentBench.
LiteLLM에서 인식 가능한 형식을 사용하세요 (예: 'openai/...', 'anthropic/...', 'groq/...', 'huggingface/...').
"""

from typing import List

MODEL_IDS: List[str] = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-8B",
    "trillionlabs/Tri-7B",
    "skt/A.X-4.0-Light",
    "K-intelligence/Midm-2.0-Base-Instruct",
    "KORMo-Team/KORMo-10B-sft",
    "kakaocorp/kanana-1.5-8b-instruct-2505",
    "dnotitia/DNA-2.0-14B"    
]

__all__ = ["MODEL_IDS"]