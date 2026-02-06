"""API key access via environment variables.

This module loads a local `.env` file when available, then exposes typed
constants used by tool/adapters.
"""

import os
import warnings

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

# Load .env early so module-level constants see updated env vars.
load_dotenv()

# Naver Search API
NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")

# Naver Directions API
Directions_Client_ID = os.environ.get("Directions_Client_ID", "")
Directions_Client_Secret = os.environ.get("Directions_Client_Secret", "")

# Kakao Local API
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")

# Aladin API
ALADIN_API_KEY = os.environ.get("ALADIN_API_KEY", "")

# Korea Investment & Securities API
KIS_APP_KEY = os.environ.get("KIS_APP_KEY", "")
KIS_APP_SECRET = os.environ.get("KIS_APP_SECRET", "")

# Tmap API
TMAP_APP_KEY = os.environ.get("TMAP_APP_KEY", "")

# LS Securities API
LS_APP_KEY = os.environ.get("LS_APP_KEY", "")
LS_APP_SECRET = os.environ.get("LS_APP_SECRET", "")

# Daum API
DAUM_API_KEY = os.environ.get("DAUM_API_KEY", "")

# Azure OpenAI API
AZURE_API_KEY: str = os.environ.get("AZURE_API_KEY", "")
AZURE_API_BASE: str = os.environ.get("AZURE_API_BASE", "")
AZURE_API_VERSION: str = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

# Anthropic API
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Google Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Optional providers
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# OpenRouter API
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_APP_URL: str = os.environ.get("OPENROUTER_APP_URL", "")
OPENROUTER_APP_TITLE: str = os.environ.get("OPENROUTER_APP_TITLE", "")


def check_api_keys():
    """Check if API keys are set and warn if missing."""
    api_keys = {
        "NAVER_CLIENT_ID": NAVER_CLIENT_ID,
        "NAVER_CLIENT_SECRET": NAVER_CLIENT_SECRET,
        "Directions_Client_ID": Directions_Client_ID,
        "Directions_Client_Secret": Directions_Client_Secret,
        "KAKAO_REST_API_KEY": KAKAO_REST_API_KEY,
        "ALADIN_API_KEY": ALADIN_API_KEY,
        "KIS_APP_KEY": KIS_APP_KEY,
        "KIS_APP_SECRET": KIS_APP_SECRET,
        "TMAP_APP_KEY": TMAP_APP_KEY,
        "LS_APP_KEY": LS_APP_KEY,
        "LS_APP_SECRET": LS_APP_SECRET,
        "DAUM_API_KEY": DAUM_API_KEY,
        "AZURE_API_KEY": AZURE_API_KEY,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    }

    missing_keys = [key for key, value in api_keys.items() if not value]

    if missing_keys:
        warnings.warn(
            "다음 API 키가 설정되지 않았습니다: "
            f"{', '.join(missing_keys)}\n"
            "관련 도구를 사용하려면 환경 변수 또는 .env 파일로 설정해주세요.\n"
            f"예: export {missing_keys[0]}='your-api-key-here'",
            UserWarning,
        )


# check_api_keys()  # Optional noisy check
