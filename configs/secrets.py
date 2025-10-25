import os
import warnings

# 네이버 검색 API
# https://developers.naver.com/products/search/
NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')

# 네이버 길찾기 API
# https://www.ncloud.com/product/applicationService/maps
Directions_Client_ID = os.environ.get('Directions_Client_ID', '')
Directions_Client_Secret = os.environ.get('Directions_Client_Secret', '')

# 카카오 로컬 API
# https://developers.kakao.com/
KAKAO_REST_API_KEY = os.environ.get('KAKAO_REST_API_KEY', '')

# 알라딘 도서 API
# https://www.aladin.co.kr/ttb/wblog_manage.aspx
ALADIN_API_KEY = os.environ.get('ALADIN_API_KEY', '')

# 한국투자증권 API
# https://apiportal.koreainvestment.com/
KIS_APP_KEY = os.environ.get('KIS_APP_KEY', '')
KIS_APP_SECRET = os.environ.get('KIS_APP_SECRET', '')

# TMAP 네비게이션 API
# https://tmapapi.sktelecom.com/
TMAP_APP_KEY = os.environ.get('TMAP_APP_KEY', '')

# LS증권 API
# https://openapi.ls-sec.co.kr/
LS_APP_KEY = os.environ.get('LS_APP_KEY', '')
LS_APP_SECRET = os.environ.get('LS_APP_SECRET', '')

# Daum 검색 API (Kakao Developers)
# https://developers.kakao.com/
DAUM_API_KEY = os.environ.get('DAUM_API_KEY', '')

# Azure OpenAI API
# https://portal.azure.com/
AZURE_API_KEY: str = os.environ.get("AZURE_API_KEY", "")
AZURE_API_BASE: str = os.environ.get("AZURE_API_BASE", "")
AZURE_API_VERSION: str = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

# Anthropic API
# https://console.anthropic.com/
ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

# Google Gemini API
# https://ai.google.dev/
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# OpenAI API (optional)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Groq API (optional)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


def check_api_keys():
    """Check if API keys are set and warn if missing."""
    api_keys = {
        'NAVER_CLIENT_ID': NAVER_CLIENT_ID,
        'NAVER_CLIENT_SECRET': NAVER_CLIENT_SECRET,
        'Directions_Client_ID': Directions_Client_ID,
        'Directions_Client_Secret': Directions_Client_Secret,
        'KAKAO_REST_API_KEY': KAKAO_REST_API_KEY,
        'ALADIN_API_KEY': ALADIN_API_KEY,
        'KIS_APP_KEY': KIS_APP_KEY,
        'KIS_APP_SECRET': KIS_APP_SECRET,
        'TMAP_APP_KEY': TMAP_APP_KEY,
        'LS_APP_KEY': LS_APP_KEY,
        'LS_APP_SECRET': LS_APP_SECRET,
        'DAUM_API_KEY': DAUM_API_KEY,
        'AZURE_API_KEY': AZURE_API_KEY,
        'ANTHROPIC_API_KEY': ANTHROPIC_API_KEY,
        'GEMINI_API_KEY': GEMINI_API_KEY,
    }
    
    missing_keys = [key for key, value in api_keys.items() if not value]
    
    if missing_keys:
        warnings.warn(
            f"다음 API 키가 설정되지 않았습니다: {', '.join(missing_keys)}\n"
            f"관련 도구를 사용하려면 secrets.py 파일이나 환경 변수로 설정해주세요.\n"
            f"예: export {missing_keys[0]}='your-api-key-here'",
            UserWarning
        )


# 모듈 로드 시 자동으로 체크 (경고만 출력, 실행은 계속)
# check_api_keys()  # 과도한 로그 출력으로 인해 임시 비활성화
