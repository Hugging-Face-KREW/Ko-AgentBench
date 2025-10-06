"""Central tool catalog mapping tool names to API methods and schemas.

This minimal catalog enables registering API class methods as tools based on
task-declared tool names, without changing runner logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from .base_api import BaseTool
from .method_tool_wrapper import make_method_tool_class
from .naver_search_mock import NaverSearchMockAPI
from .festival_search import FestivalSearchAPI
from .naver_directions import NaverMapsAPI
from .ls_stock import LSStock
from .bithumb_stock import BithumbStock
from .naver_search import NaverSearchAPI
from .daum_search import DaumSearchAPI
from .aladin_search import AladinAPI
from .kakao_local import KakaoLocal


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
                "eventStartDate": {
                    "type": "string",
                    "pattern": "^[0-9]{8}$",
                    "description": "행사 시작일 (YYYYMMDD)"
                },
                "eventEndDate": {
                    "type": "string",
                    "pattern": "^[0-9]{8}$",
                    "description": "행사 종료일 (YYYYMMDD)"
                },
                "location": {
                    "type": "string",
                    "description": "주소나 지역명 (예: '서울 강남')"
                },
                "num_of_rows": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "검색 결과 개수"
                }
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
                "start": {
                    "type": "string",
                    "description": "출발지(경도,위도) (예: 127.12345,37.12345)"
                },
                "goal": {
                    "type": "string",
                    "description": "도착지 좌표 문자열 (예: '123.45678,34.56789')"
                },
                "waypoints": {
                    "type": "string",
                    "description": "경유지 좌표 문자열. '|'로 구분 (최대 5개)"
                },
                "option": {
                    "type": "string",
                    "enum": [
                        "trafast",
                        "tracomfort",
                        "traoptimal",
                        "traavoidtoll",
                        "traavoidcaronly",
                        "trafast:traavoidtoll"
                    ],
                    "description": "경로 조회 옵션"
                },
            },
            "required": ["start", "goal"]
        },
    ),
    
    # ===== LS Stock =====
    "StockSearch_ls": (
        LSStock,
        "_stock_search",
        "종목 검색, LS증권 Open API를 활용합니다.",
        {
            "type": "object",
            "properties": {
                "query_index": {
                    "type": "string",
                    "description": "검색할 종목명 또는 코드",
                }
            },
            "required": ["query_index"]
        },
    ),
    "MarketIndex_ls": (
        LSStock,
        "_market_index",
        "시장 지수 조회, LS증권 Open API를 활용합니다.",
        {
            "type": "object",
            "properties": {
                "jisu": {
                    "type": "string",
                    "description": "조회할 지수 (KOSPI, KOSPI200, KRX100, KOSDAQ)",
                    "enum": ["KOSPI", "KOSPI200", "KRX100", "KOSDAQ"],
                    "default": "KOSPI"
                }
            },
            "required": ["jisu"]
        },
    ),
    
    # ===== Bithumb Crypto =====
    "CryptoPrice_bithumb": (
        BithumbStock,
        "_cryptoPrice_bithumb",
        "현재가 정보, 빗썸 Open API를 활용합니다.",
        {
            "type": "object",
            "properties": {
                "markets": {
                    "type": "string",
                    "description": "반점으로 구분되는 마켓 코드 (ex. KRW-BTC, BTC-ETH)",
                    "pattern": "^[A-Z0-9]+$",
                    "default": "KRW-BTC",
                    "enum": ["KRW-BTC", "BTC-ETH"]
                }
            },
            "required": ["markets"]
        },
    ),
    "MarketList_bithumb": (
        BithumbStock,
        "_marketList_bithumb",
        "거래 가능한 마켓과 가상자산 정보 조회, 빗썸 Open API를 활용합니다.",
        {
            "type": "object",
            "properties": {
                "isDetails": {
                    "type": "boolean",
                    "description": "유의종목 필드과 같은 상세 정보 노출 여부(선택 파라미터)",
                    "default": False
                }
            }
        },
    ),
    
    # ===== Naver Search =====
    "search_web": (
        NaverSearchAPI,
        "_search_web",
        "네이버 웹 검색 API 호출",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어"
                },
                "display": {
                    "type": "integer",
                    "description": "한 번에 표시할 검색 결과 개수",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "start": {
                    "type": "integer",
                    "description": "검색 시작 위치",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 1
                },
                "sort": {
                    "type": "string",
                    "enum": ["sim", "date"],
                    "description": "검색 결과 정렬 방법",
                    "default": "sim"
                }
            },
            "required": ["query"]
        },
    ),
    "search_blog": (
        NaverSearchAPI,
        "_search_blog",
        "네이버 블로그 검색 API 호출",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어"
                },
                "display": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 1
                },
                "sort": {
                    "type": "string",
                    "enum": ["sim", "date"],
                    "default": "sim"
                }
            },
            "required": ["query"]
        },
    ),
    "search_news": (
        NaverSearchAPI,
        "_search_news",
        "네이버 뉴스 검색 API 호출",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어"
                },
                "display": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 1
                },
                "sort": {
                    "type": "string",
                    "enum": ["sim", "date"],
                    "default": "sim"
                }
            },
            "required": ["query"]
        },
    ),
    
    # ===== Aladin =====
    "ItemSearch_aladin": (
        AladinAPI,
        "_search_item",
        "키워드, 카테고리 등 상세 검색 조건으로 상품을 검색합니다.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색어"
                },
                "query_type": {
                    "type": "string",
                    "enum": ["Keyword", "Title", "Author", "Publisher"],
                    "default": "Keyword"
                },
                "search_target": {
                    "type": "string",
                    "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                    "default": "Book"
                },
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "sort": {
                    "type": "string",
                    "enum": ["Accuracy", "PublishTime", "SalesPoint", "CustomerRating", "MyReviewCount"],
                    "default": "Accuracy"
                }
            },
            "required": ["query"]
        },
    ),
    "ItemList_aladin": (
        AladinAPI,
        "_get_item_list",
        "신간, 베스트셀러 등 특정 종류의 상품 리스트를 상세 조건으로 조회합니다.",
        {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["ItemNewAll", "ItemNewSpecial", "ItemEditorChoice", "Bestseller", "BlogBest"],
                },
                "search_target": {
                    "type": "string",
                    "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                    "default": "Book"
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                }
            },
            "required": ["query_type"]
        },
    ),
    
    # ===== Daum Search =====
    "WebSearch_daum": (
        DaumSearchAPI,
        "_search_web",
        "다음 검색 서비스에서 질의어로 웹 문서를 검색합니다.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색을 원하는 질의어"
                },
                "sort": {
                    "type": "string",
                    "enum": ["accuracy", "recency"],
                    "default": "accuracy"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 1
                },
                "size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                }
            },
            "required": ["query"]
        },
    ),
    "VideoSearch_daum": (
        DaumSearchAPI,
        "_search_video",
        "카카오 TV, 유튜브 등 서비스에서 질의어로 동영상을 검색합니다.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색을 원하는 질의어"
                },
                "sort": {
                    "type": "string",
                    "enum": ["accuracy", "recency"],
                    "default": "accuracy"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 15,
                    "default": 1
                },
                "size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "default": 15
                }
            },
            "required": ["query"]
        },
    ),
    
    # ===== Kakao Local =====
    "AddressToCoord_kakao": (
        KakaoLocal,
        "_address_to_coord",
        "주소를 좌표로 변환",
        {
            "type": "object",
            "properties": {
                "address": {
                    "type": "string",
                    "description": "변환할 주소"
                }
            },
            "required": ["address"]
        },
    ),
    "PlaceSearch_kakao": (
        KakaoLocal,
        "_place_search",
        "키워드 장소 검색",
        {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "검색 키워드"
                },
                "x": {
                    "type": "number",
                    "description": "중심 좌표의 경도"
                },
                "y": {
                    "type": "number",
                    "description": "중심 좌표의 위도"
                },
                "radius": {
                    "type": "integer",
                    "description": "검색 반경(미터), 최대 20000"
                },
                "sort": {
                    "type": "string",
                    "enum": ["accuracy", "distance"],
                    "default": "accuracy"
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 45,
                    "default": 15
                }
            },
            "required": ["keyword"]
        },
    ),
    "CategorySearch_kakao": (
        KakaoLocal,
        "_category_search",
        "카테고리 장소 검색",
        {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "카테고리 그룹 코드"
                },
                "x": {
                    "type": "number",
                    "description": "중심 좌표의 경도"
                },
                "y": {
                    "type": "number",
                    "description": "중심 좌표의 위도"
                },
                "radius": {
                    "type": "integer",
                    "description": "검색 반경(미터), 최대 20000",
                    "default": 1000
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1
                },
                "size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 45,
                    "default": 15
                }
            },
            "required": ["category", "x", "y"]
        },
    ),
    
    # ===== Legacy Naver Mock (for backward compatibility) =====
    "naver_web_search": (
        NaverSearchMockAPI,
        "WebSearch_naver",
        "네이버 웹 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
    "naver_blog_search": (
        NaverSearchMockAPI,
        "BlogSearch_naver",
        "네이버 블로그 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
    "naver_news_search": (
        NaverSearchMockAPI,
        "NewsSearch_naver",
        "네이버 뉴스 검색 API",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "display": {"type": "integer", "minimum": 1, "maximum": 100},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["query"],
        },
    ),
}


def resolve_tool_classes(tool_names: List[str]) -> List[Type[BaseTool]]:
    """Resolve given tool names to concrete BaseTool classes via catalog.

    For each API class, a single instance is allocated and reused among
    method-backed tools to avoid redundant setup.
    """
    api_instances: Dict[Type[Any], Any] = {}
    resolved: List[Type[BaseTool]] = []
    seen: set[str] = set()

    for name in tool_names:
        if name in seen:
            continue
        seen.add(name)

        entry = TOOL_CATALOG.get(name)
        if not entry:
            continue
        api_class, method_name, description, parameters_schema = entry

        if api_class not in api_instances:
            api_instances[api_class] = api_class()
        api_instance = api_instances[api_class]

        tool_class = make_method_tool_class(
            name=name,
            description=description,
            api_instance=api_instance,
            method_name=method_name,
            parameters_schema=parameters_schema,
        )
        resolved.append(tool_class)

    return resolved


__all__ = ["TOOL_CATALOG", "resolve_tool_classes"]


