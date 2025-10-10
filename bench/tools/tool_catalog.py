"""Central tool catalog mapping tool names to API methods and schemas.

This minimal catalog enables registering API class methods as tools based on
task-declared tool names, without changing runner logic.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Type

from .base_api import BaseTool
from .method_tool_wrapper import make_method_tool_class
from .festival_search import FestivalSearchAPI
from .naver_directions import NaverMapsAPI
from .ls_stock import LSStock
from .bithumb_stock import BithumbStock
from .naver_search import NaverSearchAPI
from .daum_search import DaumSearchAPI
from .aladin_search import AladinAPI
from .kakao_local import KakaoLocal
from .tmap_navigation import TmapNavigation 
from .kis_stock import KISStock 


# Catalog entry: tool_name -> (api_class, method_name, description, parameters_schema)
TOOL_CATALOG: Dict[str, Tuple[Type[Any], str, str, Dict[str, Any]]] = {
    # ===== Festival Search =====
    "festival_search": (
        FestivalSearchAPI,
        "FestivalSearch_kto",
        "행사/축제 정보를 날짜와 지역으로 검색",
        {
            "type": "object",
            "properties": {
                "eventStartDate": {"type": "string", "pattern": "^[0-9]{8}$", "description": "행사 시작일 (YYYYMMDD)"},
                "eventEndDate": {"type": "string", "pattern": "^[0-9]{8}$", "description": "행사 종료일 (YYYYMMDD)"},
                "location": {"type": "string", "description": "주소나 지역명 (예: '서울 강남')"},
                "num_of_rows": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10, "description": "검색 결과 개수"},
            },
            "required": ["eventStartDate"],
        },
    ),

    # ===== Naver Directions =====
    "_directions": (
        NaverMapsAPI,
        "Directions_naver",
        "입력 정보(출발지, 경유지, 목적지 등)를 기반으로 자동차 경로 조회",
        {
            "type": "object",
            "properties": {
                "start": {"type": "string", "description": "출발지(경도,위도) (예: 127.12345,37.12345)"},
                "goal": {"type": "string", "description": "도착지 좌표 문자열 (예: '123.45678,34.56789')"},
                "waypoints": {"type": "string", "description": "경유지 좌표 문자열. '|'로 구분 (최대 5개)"},
                "option": {
                    "type": "string",
                    "enum": ["trafast", "tracomfort", "traoptimal", "traavoidtoll", "traavoidcaronly", "trafast:traavoidtoll"],
                    "description": "경로 조회 옵션",
                },
            },
            "required": ["start", "goal"],
        },
    ),

    # ===== LS Stock =====
    "StockSearch_ls": (
        LSStock,
        "_stock_search",
        "종목 검색, LS증권 Open API를 활용합니다.",
        {"type": "object", "properties": {"query_index": {"type": "string", "description": "검색할 종목명 또는 코드"}}, "required": ["query_index"]},
    ),
    "MarketIndex_ls": (
        LSStock,
        "_market_index",
        "시장 지수 조회, LS증권 Open API를 활용합니다.",
        {"type": "object", "properties": {"jisu": {"type": "string", "enum": ["KOSPI", "KOSDAQ"], "default": "KOSPI"}}, "required": ["jisu"]},
    ),

    # ===== KIS Stock ===== ✅ 추가
    "StockPrice_kis": (
        KISStock,
        "_stock_price",
        "한국투자증권 국내 주식 현재가 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "종목 코드 (예: 삼성전자 005930)"},
                "market": {"type": "string", "enum": ["KOSPI", "KOSDAQ"], "default": "KOSPI"},
            },
            "required": ["symbol"],
        },
    ),
    "StockChart_kis": (
        KISStock,
        "_stock_chart",
        "한국투자증권 주식 차트 데이터 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "종목 코드 (예: 005930)"},
                "period": {"type": "string", "enum": ["D", "W", "M"], "default": "D"},
                "count": {"type": "integer", "default": 30, "description": "조회할 데이터 개수"},
            },
            "required": ["symbol"],
        },
    ),

    # ===== Bithumb Crypto =====
    "CryptoPrice_bithumb": (
        BithumbStock,
        "_cryptoPrice_bithumb",
        "현재가 정보, 빗썸 Open API를 활용합니다.",
        {"type": "object", "properties": {"markets": {"type": "string", "default": "KRW-BTC"}}, "required": ["markets"]},
    ),

    # ===== Naver Search =====
    "WebSearch_naver": (
        NaverSearchAPI, 
        "_search_web", 
        "네이버 웹 검색 API 호출", 
        {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "start": {"type": "integer", "minimum": 1, "default": 1},
                "sort": {"type": "string", "enum": ["sim", "date"], "default": "sim"}
            }, 
            "required": ["query"]
        }
    ),
    "BlogSearch_naver": (
        NaverSearchAPI, 
        "_search_blog", 
        "네이버 블로그 검색 API 호출", 
        {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "start": {"type": "integer", "minimum": 1, "default": 1},
                "sort": {"type": "string", "enum": ["sim", "date"], "default": "sim"}
            }, 
            "required": ["query"]
        }
    ),
    "NewsSearch_naver": (
        NaverSearchAPI, 
        "_search_news", 
        "네이버 뉴스 검색 API 호출", 
        {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
                "start": {"type": "integer", "minimum": 1, "default": 1},
                "sort": {"type": "string", "enum": ["sim", "date"], "default": "sim"}
            }, 
            "required": ["query"]
        }
    ),

    # ===== Daum Search =====
    "WebSearch_daum": (
        DaumSearchAPI,
        "_search_web",
        "다음 웹 검색 API 호출",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "sort": {"type": "string", "enum": ["accuracy", "recency"], "default": "accuracy"},
                "page": {"type": "integer", "minimum": 1, "default": 1},
                "size": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10}
            },
            "required": ["query"]
        }
    ),
    "VideoSearch_daum": (
        DaumSearchAPI,
        "_search_video",
        "다음 동영상 검색 API 호출",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "sort": {"type": "string", "enum": ["accuracy", "recency"], "default": "accuracy"},
                "page": {"type": "integer", "minimum": 1, "default": 1},
                "size": {"type": "integer", "minimum": 1, "maximum": 30, "default": 10}
            },
            "required": ["query"]
        }
    ),

    # ===== Aladin =====
    "ItemSearch_aladin": (
        AladinAPI, 
        "_search_item", 
        "알라딘 상품 검색 API", 
        {
            "type": "object", 
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "query_type": {"type": "string", "enum": ["Keyword", "Title", "Author", "Publisher"], "default": "Keyword"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "start": {"type": "integer", "minimum": 1, "default": 1},
                "sort": {"type": "string", "enum": ["Accuracy", "PublishTime", "Title", "SalesPoint", "CustomerRating"], "default": "Accuracy"}
            }, 
            "required": ["query"]
        }
    ),
    "ItemList_aladin": (
        AladinAPI,
        "_list_item",
        "알라딘 베스트셀러/신간 리스트 조회",
        {
            "type": "object",
            "properties": {
                "query_type": {"type": "string", "enum": ["Bestseller", "ItemNewAll", "ItemNewSpecial"], "default": "Bestseller"},
                "search_target": {"type": "string", "enum": ["Book", "Foreign", "eBook"], "default": "Book"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "start": {"type": "integer", "minimum": 1, "default": 1}
            },
            "required": ["query_type"]
        }
    ),

    # ===== Kakao Local =====
    "PlaceSearch_kakao": (
        KakaoLocal, 
        "_place_search", 
        "키워드 장소 검색", 
        {
            "type": "object", 
            "properties": {
                "keyword": {"type": "string", "description": "검색 키워드"},
                "x": {"type": "number", "description": "중심 경도"},
                "y": {"type": "number", "description": "중심 위도"},
                "radius": {"type": "integer", "minimum": 0, "maximum": 20000, "description": "반경(m)"},
                "sort": {"type": "string", "enum": ["distance", "accuracy"], "default": "accuracy"}
            }, 
            "required": ["keyword"]
        }
    ),
    "AddressToCoord_kakao": (
        KakaoLocal,
        "_address_to_coord",
        "주소를 좌표로 변환",
        {
            "type": "object",
            "properties": {
                "address": {"type": "string", "description": "변환할 주소"}
            },
            "required": ["address"]
        }
    ),
    "CategorySearch_kakao": (
        KakaoLocal,
        "_category_search",
        "카테고리별 장소 검색",
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "카테고리 코드 (예: CE7=카페)"},
                "x": {"type": "number", "description": "중심 경도"},
                "y": {"type": "number", "description": "중심 위도"},
                "radius": {"type": "integer", "minimum": 0, "maximum": 20000, "default": 1000},
                "size": {"type": "integer", "minimum": 1, "maximum": 15, "default": 15}
            },
            "required": ["category", "x", "y"]
        }
    ),

    # ===== LS Stock (추가) =====
    "StockPrice_ls": (
        LSStock,
        "_stock_price",
        "LS증권 주식 현재가 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "종목 코드"}
            },
            "required": ["symbol"]
        }
    ),

    # ===== Bithumb (추가) =====
    "MarketList_bithumb": (
        BithumbStock,
        "_marketList_bithumb",
        "빗썸 거래 가능 마켓 리스트 조회",
        {
            "type": "object",
            "properties": {},
            "required": []
        }
    ),

    # ===== Tmap Navigation ===== ✅ 추가
    "WalkRoute_tmap": (
        TmapNavigation,
        "WalkRoute_tmap",
        "보행자 경로 안내 (Tmap)",
        {
            "type": "object",
            "properties": {
                "startX": {"type": "number"},
                "startY": {"type": "number"},
                "endX": {"type": "number"},
                "endY": {"type": "number"},
            },
            "required": ["startX", "startY", "endX", "endY"],
        },
    ),
    "CategorySearch_tmap": (
        TmapNavigation,
        "CategorySearch_tmap",
        "카테고리 장소 검색 (Tmap)",
        {
            "type": "object",
            "properties": {
                "categories": {"type": "string"},
                "centerLon": {"type": "number"},
                "centerLat": {"type": "number"},
                "radius": {"type": "integer", "default": 1000},
                "count": {"type": "integer", "default": 10},
            },
            "required": ["categories", "centerLon", "centerLat"],
        },
    ),
}


# TODO: 데이터셋 수정 후 삭제 예정
# 임시 별칭 매핑: 데이터셋의 구 도구 이름 → TOOL_CATALOG 키로 변환
TOOL_ALIAS_MAP: Dict[str, str] = {
    # Naver Search 별칭
    "search_web": "WebSearch_naver",
    "search_blog": "BlogSearch_naver", 
    "search_news": "NewsSearch_naver",
    
    # Daum Search 별칭 (이미 올바른 이름이지만 명시)
    "WebSearch_daum": "WebSearch_daum",
    "VideoSearch_daum": "VideoSearch_daum",
    
    # Aladin 별칭
    "ItemList_aladin": "ItemList_aladin",
    
    # Kakao 별칭
    "AddressToCoord_kakao": "AddressToCoord_kakao",
    "CategorySearch_kakao": "CategorySearch_kakao",
    
    # LS Stock 별칭
    "StockPrice_ls": "StockPrice_ls",
    
    # Bithumb 별칭
    "MarketList_bithumb": "MarketList_bithumb",
}


def normalize_tool_name(tool_name: str) -> str:
    """정규화된 도구 이름 반환 (별칭 → 실제 이름).
    
    TODO: 데이터셋 수정 후 이 함수 삭제 예정
    
    Args:
        tool_name: 원본 도구 이름 (데이터셋에서 온 이름)
        
    Returns:
        TOOL_CATALOG에서 사용하는 실제 이름
    """
    return TOOL_ALIAS_MAP.get(tool_name, tool_name)


def resolve_tool_classes(tool_names: List[str]) -> List[Type[BaseTool]]:
    """Resolve given tool names to concrete BaseTool classes via catalog."""
    api_instances: Dict[Type[Any], Any] = {}
    resolved: List[Type[BaseTool]] = []
    seen: set[str] = set()

    for name in tool_names:
        if name in seen:
            continue
        
        # TODO: 데이터셋 수정 후 이 정규화 로직 삭제
        normalized_name = normalize_tool_name(name)
        
        if normalized_name in seen:
            continue
        seen.add(normalized_name)

        entry = TOOL_CATALOG.get(normalized_name)
        if not entry:
            continue
        api_class, method_name, description, parameters_schema = entry

        if api_class not in api_instances:
            api_instances[api_class] = api_class()
        api_instance = api_instances[api_class]

        tool_class = make_method_tool_class(
            name=normalized_name,  # TODO: 데이터셋 수정 후 name으로 변경
            description=description,
            api_instance=api_instance,
            method_name=method_name,
            parameters_schema=parameters_schema,
        )
        resolved.append(tool_class)

    return resolved


__all__ = ["TOOL_CATALOG", "TOOL_ALIAS_MAP", "normalize_tool_name", "resolve_tool_classes"]
