"""
Deprecated: This file is kept for backward compatibility.
All secrets are now managed in `configs/secrets.py`.

Import from here only if legacy code requires `bench.tools.secrets`.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from configs.secrets import *  # re-export all keys
except ImportError:
    # configs 모듈을 찾을 수 없는 경우, 환경 변수에서 직접 읽기
    import warnings
    warnings.warn(
        "configs.secrets를 찾을 수 없어 환경 변수에서 직접 읽습니다.",
        ImportWarning
    )
    
    
    # 네이버 검색 API
    NAVER_CLIENT_ID = os.environ.get('NAVER_CLIENT_ID', '')
    NAVER_CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '')
    
    # 네이버 길찾기 API
    Directions_Client_ID = os.environ.get('Directions_Client_ID', '')
    Directions_Client_Secret = os.environ.get('Directions_Client_Secret', '')
    
    # 카카오 로컬 API
    KAKAO_REST_API_KEY = os.environ.get('KAKAO_REST_API_KEY', '')
    
    # 알라딘 도서 API
    ALADIN_API_KEY = os.environ.get('ALADIN_API_KEY', '')
    
    # 한국투자증권 API
    KIS_APP_KEY = os.environ.get('KIS_APP_KEY', '')
    KIS_APP_SECRET = os.environ.get('KIS_APP_SECRET', '')
    
    # TMAP 네비게이션 API
    TMAP_APP_KEY = os.environ.get('TMAP_APP_KEY', '')
    
    # LS증권 API
    LS_APP_KEY = os.environ.get('LS_APP_KEY', '')
    LS_APP_SECRET = os.environ.get('LS_APP_SECRET', '')
    
    # Daum 검색 API
    DAUM_API_KEY = os.environ.get('DAUM_API_KEY', '')

