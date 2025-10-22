"""Central tool catalog mapping tool names to API methods and schemas.

This minimal catalog enables registering API class methods as tools based on
task-declared tool names, without changing runner logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from .aladin_search import AladinAPI
from .base_api import BaseTool
from .bithumb_stock import BithumbStock
from .daum_search import DaumSearchAPI
from .kakao_local import KakaoLocal
from .kis_stock import KISStock
from .ls_stock import LSStock
from .method_tool_wrapper import make_method_tool_class
from .naver_directions import NaverMapsAPI
from .naver_search import NaverSearchAPI
from .tmap_navigation import TmapNavigation
from .upbit_crypto import UpbitCrypto

# Catalog entry: tool_name -> (api_class, method_name, description, parameters_schema)
TOOL_CATALOG: Dict[str, Tuple[Type[Any], str, str, Dict[str, Any]]] = {
    # (삭제됨) Festival Search 도구

    # ===== Naver Directions =====
    "Directions_naver": (
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
    "StockPrice_ls": (
        LSStock,
        "_stock_price",
        "주식 현재가 조회, LS증권 Open API를 활용합니다.",
        {"type": "object", "properties": {"shcode": {"type": "string", "description": "주식 종목코드 (6자리, 예: 005930=삼성전자, 000660=SK하이닉스)","pattern": "^[0-9]{6}$"},"exchgubun": {"type": "string", "description": "거래소구분코드(K:KRX,N:NXT,U:통합)", "enum": ["K", "N", "U"], "default": "K"}}, "required": ["shcode"]},
    ),
    "MarketIndex_ls": (
        LSStock,
        "_market_index",
        "시장 지수 조회, LS증권 Open API를 활용합니다.",
        {"type": "object", "properties": {"jisu": {"type": "string", "enum": ["KOSPI", "KOSPI200", "KRX100", "KOSDAQ"], "default": "KOSPI"}}, "required": ["jisu"]},
    ),
    "SectorStock_ls": (
        LSStock,
        "_sector_stock",
        "업종별/테마별 종목 시세 조회 (LS증권)",
        {
            "type": "object",
            "properties": {
                "tmcode": {"type": "string", "description": "테마 코드"}
            },
            "required": ["tmcode"]
        }
    ),
    "OrderBook_ls": (
        LSStock,
        "_order_book",
        "주식 호가 정보 조회 (LS증권)",
        {
            "type": "object",
            "properties": {
                "shcode": {"type": "string", "description": "종목 코드 (6자리)"}
            },
            "required": ["shcode"]
        }
    ),
    "StockTrades_ls": (
        LSStock,
        "_stock_trades",
        "주식 시간대별 체결 내역 조회 (LS증권)",
        {
            "type": "object",
            "properties": {
                "shcode": {"type": "string", "description": "종목 코드"},
                "exchgubun": {"type": "string", "enum": ["K", "N", "U"], "default": "N", "description": "거래소구분코드(K:KRX,N:NXT,U:통합)"}
            },
            "required": ["shcode"]
        }
    ),

    # ===== KIS Stock =====
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
    "USStockPrice_kis": (
        KISStock,
        "_us_stock_price",
        "한국투자증권 미국 주식 현재가 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "종목 심볼 (예: AAPL, TSLA)"},
                "exchange": {"type": "string", "enum": ["NASDAQ", "NYSE"], "default": "NASDAQ", "description": "거래소"}
            },
            "required": ["symbol"]
        }
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
        "빗썸 암호화폐 현재가 정보 조회",
    {
        "type": "object", "properties": {
            "markets": {
                "type": "string",
                "default": "KRW-BTC",
                "description": "쉼표로 구분되는 마켓 코드 목록 (예: KRW-BTC,KRW-ETH)",
                "pattern": "^[A-Z]{2,5}-[A-Z0-9.-]+(,[A-Z]{2,5}-[A-Z0-9.-]+)*$"
            }
        },
            "required": ["markets"]},
    ),
    "OrderBook_bithumb": (
        BithumbStock,
        "_orderBook_bithumb",
        "빗썸 거래소 호가 정보 조회",
        {
            "type": "object",
            "properties": {
                "markets": {
                    "type": "string",
                    "default": "KRW-BTC",
                    "description": "쉼표로 구분되는 마켓 코드 목록 (예: KRW-BTC,KRW-ETH)",
                    "pattern": "^[A-Z]{2,5}-[A-Z0-9.-]+(,[A-Z]{2,5}-[A-Z0-9.-]+)*$"
                }
            },
            "required": ["markets"]
        }
    ),
    "CryptoCandle_bithumb": (
        BithumbStock,
        "_cryptoCandle_bithumb",
        "빗썸 암호화폐 캔들 데이터 조회",
        {
            "type": "object",
            "properties": {
                "time": {"type": "string", "enum": ["minutes", "days", "weeks", "months"], "description": "캔들 단위"},
                "market": {"type": "string", "default": "KRW-BTC", "description": "마켓 코드"},
                "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 1, "description": "캔들 개수"},
                "to": {"type": "string", "description": "마지막 캔들 시각 (yyyy-MM-dd'T'HH:mm:ss'Z')"},
                "unit": {"type": "integer", "enum": [1, 3, 5, 10, 15, 30, 60, 240], "default": 1, "description": "분 단위 (time이 minutes일 때만 사용)"}
            },
            "required": ["time"]
        }
    ),
    "MarketList_bithumb": (
        BithumbStock,
        "_marketList_bithumb",
        "빗썸 거래 가능 마켓 리스트 조회",
        {
            "type": "object",
            "properties": {
                "isDetails": {"type": "boolean", "default": False, "description": "상세 정보 노출 여부"}
            },
            "required": []
        }
    ),

    # ===== Upbit Crypto =====
    "CryptoPrice_upbit": (
        UpbitCrypto,
        "_crypto_price",
        "업비트 암호화폐 현재가 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "암호화폐 심볼 (예: BTC, ETH)"},
                "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT"], "default": "KRW", "description": "기준 통화"}
            },
            "required": ["symbol"]
        }
    ),
    "MarketList_upbit": (
        UpbitCrypto,
        "_market_list",
        "업비트 마켓 목록 조회",
        {
            "type": "object",
            "properties": {
                "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT", "ALL"], "default": "KRW", "description": "기준 통화"},
                "include_event": {"type": "boolean", "default": True, "description": "이벤트 마켓 포함 여부"}
            },
            "required": []
        }
    ),
    "CryptoCandle_upbit": (
        UpbitCrypto,
        "_crypto_candle",
        "업비트 암호화폐 캔들 데이터 조회",
        {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "암호화폐 심볼 (예: BTC, ETH)"},
                "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT"], "default": "KRW", "description": "기준 통화"},
                "candle_type": {"type": "string", "enum": ["minutes", "days", "weeks", "months"], "default": "days", "description": "캔들 타입"},
                "unit": {"type": "integer", "enum": [1, 3, 5, 10, 15, 30, 60, 240], "description": "분 단위 (candle_type이 minutes일 때만 필요)"},
                "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 30, "description": "조회할 캔들 개수"},
                "to": {"type": "string", "description": "마지막 캔들 시각 (YYYY-MM-DD HH:mm:ss)"}
            },
            "required": ["symbol"]
        }
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
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10, "description": "한 번에 표시할 검색 결과 개수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1, "description": "검색 시작 위치"}
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
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10, "description": "한 번에 표시할 검색 결과 개수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1, "description": "검색 시작 위치"},
                "sort": {"type": "string", "enum": ["sim", "date"], "default": "sim", "description": "정렬 방식 (sim: 정확도순, date: 날짜순)"}
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
                "display": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10, "description": "한 번에 표시할 검색 결과 개수"},
                "start": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 1, "description": "검색 시작 위치"},
                "sort": {"type": "string", "enum": ["sim", "date"], "default": "sim", "description": "정렬 방식 (sim: 정확도순, date: 날짜순)"}
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
                "sort": {"type": "string", "enum": ["Accuracy", "PublishTime", "Title", "SalesPoint", "CustomerRating"], "default": "Accuracy"},
                "cover": {"type": "string", "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"], "default": "Mid", "description": "표지 이미지 크기"},
                "category_id": {"type": "integer", "description": "카테고리 ID"},
                "output": {"type": "string", "enum": ["xml", "js"], "default": "js", "description": "출력 형식"},
                "out_of_stock_filter": {"type": "integer", "enum": [0, 1], "default": 0, "description": "품절/절판 상품 필터링 여부 (1: 제외)"},
                "opt_result": {"type": "string", "description": "부가 정보 요청. 쉼표로 구분하여 다중 선택. (예: ebookList, usedList)"}
            }, 
            "required": ["query"]
        }
    ),
    "ItemList_aladin": (
        AladinAPI,
        "_get_item_list",
        "알라딘 베스트셀러/신간 리스트 조회",
        {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["ItemNewAll", "ItemNewSpecial", "ItemEditorChoice", "Bestseller", "BlogBest"],
                    "description": "조회할 리스트 종류"
                },
                "search_target": {
                    "type": "string",
                    "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                    "description": "조회 대상 Mall, 기본값: Book(도서)",
                    "default": "Book"
                },
                "sub_search_target": {
                    "type": "string",
                    "enum": ["Book", "Music", "DVD", ""],
                    "description": "SearchTarget이 Used(중고)일 경우, 서브 Mall 지정",
                    "default": ""
                },
                "start": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "시작 페이지, 기본값: 1",
                    "default": 1
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "한 페이지에 보여질 상품 수, 기본값: 10",
                    "default": 10
                },
                "cover": {
                    "type": "string",
                    "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                    "description": "표지 이미지 크기, 기본값: Mid",
                    "default": "Mid"
                },
                "category_id": {
                    "type": "integer",
                    "description": "분야의 고유 번호로 리스트를 제한합니다. (기본값: 0, 전체)",
                    "default": 0
                },
                "year": {
                    "type": "integer",
                    "description": "Bestseller 조회 시 기준 연도 (생략 시 현재)"
                },
                "month": {
                    "type": "integer",
                    "description": "Bestseller 조회 시 기준 월 (생략 시 현재)"
                },
                "week": {
                    "type": "integer",
                    "description": "Bestseller 조회 시 기준 주 (생략 시 현재)"
                },
                "output": {
                    "type": "string",
                    "enum": ["xml", "js"],
                    "description": "출력 형식, 기본값: js",
                    "default": "js"
                },
                "out_of_stock_filter": {
                    "type": "integer",
                    "enum": [0, 1],
                    "description": "품절/절판 상품 필터링 여부 (1: 제외), 기본값: 0",
                    "default": 0
                }
            },
            "required": ["query_type"]
        }
    ),
    "ItemLookup_aladin": (
        AladinAPI,
        "_get_item_details",
        "알라딘 상품 상세 정보 조회",
        {
            "type": "object",
            "properties": {
                "item_id": {"type": "string", "description": "상품의 고유 ID (ISBN, ISBN13, 또는 알라딘 ItemId)"},
                "item_id_type": {"type": "string", "enum": ["ISBN", "ISBN13", "ItemId"], "default": "ISBN13", "description": "ItemId의 종류"},
                "cover": {"type": "string", "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"], "default": "Mid", "description": "표지 이미지 크기"},
                "output": {"type": "string", "enum": ["xml", "js"], "default": "js", "description": "출력 형식"},
                "opt_result": {"type": "string", "default": "", "description": "부가 정보 요청 (Toc, authors, reviewList, etc)"}
            },
            "required": ["item_id"]
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
                "page": {"type": "integer", "description": "페이지 번호"},
                "size": {"type": "integer", "description": "검색 결과 개수"},
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
    "CoordToAddress_kakao": (
        KakaoLocal,
        "_coord_to_address",
        "위경도 좌표를 주소로 변환 (Kakao Local)",
        {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "위도 (예: 37.4979)"},
                "longitude": {"type": "number", "description": "경도 (예: 127.0276)"},
            },
            "required": ["latitude", "longitude"],
        },
    ),

    # ===== Tmap Navigation =====
    "POISearch_tmap": (
        TmapNavigation,
        "POISearch_tmap",
        "T map을 통해 키워드로 전국의 장소(POI)를 검색합니다. 맛집, 병원, 주유소, 관광지 등 150만 건의 POI 데이터를 검색할 수 있습니다.",
        {
            "type": "object",
            "properties": {
                "searchKeyword": {
                    "type": "string",
                    "description": "검색할 장소명 또는 키워드 (예: 스타벅스, 강남역 병원, 부산 맛집, 서울 이마트). 지역명을 포함하여 검색하세요."
                },
                "count": {
                    "type": "integer",
                    "description": "검색 결과 개수 (기본값: 10, 최대: 200)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 200
                },
                "centerLon": {
                    "type": "number",
                    "description": "검색 중심점 경도 (선택사항, centerLat과 함께 사용하여 해당 위치 근처 결과 우선 표시)"
                },
                "centerLat": {
                    "type": "number",
                    "description": "검색 중심점 위도 (선택사항, centerLon과 함께 사용하여 해당 위치 근처 결과 우선 표시)"
                },
                "page": {
                    "type": "integer",
                    "description": "페이지 번호 (더 많은 결과가 필요할 때 사용)",
                    "default": 1,
                    "minimum": 1
                }
            },
            "required": ["searchKeyword"]
        }
    ),
    "CarRoute_tmap": (
        TmapNavigation,
        "CarRoute_tmap",
        "자동차 경로 안내 (Tmap)",
        {
            "type": "object",
            "properties": {
                "startX": {"type": "number", "description": "출발지 경도"},
                "startY": {"type": "number", "description": "출발지 위도"},
                "endX": {"type": "number", "description": "도착지 경도"},
                "endY": {"type": "number", "description": "도착지 위도"},
                "searchOption": {"type": "integer", "default": 0, "description": "경로 옵션 (0: 추천, 2: 최단거리 등)"}
            },
            "required": ["startX", "startY", "endX", "endY"]
        }
    ),
    "Geocoding_tmap": (
        TmapNavigation,
        "Geocoding_tmap",
        "주소 좌표 변환 (Tmap)",
        {
            "type": "object",
            "properties": {
                "city_do": {"type": "string", "description": "시/도 (예: 서울특별시)"},
                "gu_gun": {"type": "string", "description": "구/군 (예: 강남구)"},
                "dong": {"type": "string", "description": "동 (예: 역삼동)"},
                "bunji": {"type": "string", "description": "번지 (선택)", "default": ""},
                "detailAddress": {"type": "string", "description": "상세주소 (선택)", "default": ""}
            },
            "required": ["city_do", "gu_gun", "dong"]
        }
    ),
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
                "radius": {"type": "integer", "default": 1},
                "count": {"type": "integer", "default": 20},
            },
            "required": ["categories", "centerLon", "centerLat"],
        },
    ),
}


    


def resolve_tool_classes(tool_names: List[str]) -> List[Type[BaseTool]]:
    """Resolve given tool names to concrete BaseTool classes via catalog."""
    api_instances: Dict[Type[Any], Any] = {}
    resolved: List[Type[BaseTool]] = []
    seen: set[str] = set()

    print(f"🔍 DEBUG resolve_tool_classes: Input tool_names = {tool_names}")

    for name in tool_names:
        if name in seen:
            print(f"  ⏭️  Skipping duplicate: {name}")
            continue
        
        # 별칭/정규화 제거: 입력된 이름을 그대로 사용
        seen.add(name)

        entry = TOOL_CATALOG.get(name)
        if not entry:
            print(f"  ❌ Tool '{name}' NOT FOUND in TOOL_CATALOG")
            print(f"     Available tools: {list(TOOL_CATALOG.keys())[:5]}...")
            continue
        
        print(f"  ✅ Found '{name}' in catalog")
        api_class, method_name, description, parameters_schema = entry

        if api_class not in api_instances:
            api_instances[api_class] = api_class()
            print(f"     Created instance of {api_class.__name__}")
        api_instance = api_instances[api_class]

        tool_class = make_method_tool_class(
            name=name,
            description=description,
            api_instance=api_instance,
            method_name=method_name,
            parameters_schema=parameters_schema,
        )
        resolved.append(tool_class)
        print(f"     ✅ Tool class created for '{name}'")

    print(f"🔍 DEBUG resolve_tool_classes: Resolved {len(resolved)} tools")
    return resolved


__all__ = ["TOOL_CATALOG", "resolve_tool_classes"]
