import os
import time
import uuid
from typing import Any, Dict
import requests
from base_api import BaseAPI

class AladinAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="aladin_api",
            description="알라딘 도서 검색 및 정보 조회 API"
        )
        # 환경변수 TTB API 키 우선. (환경변수: ALADIN_TTB_KEY)
        self.api_key = os.getenv("ALADIN_TTB_KEY", "ttbbbangggo1231514001")
        self.base_url = "http://www.aladin.co.kr/ttb/api"
        self.version = "20131101"
        self.default_timeout = 10
        self._last_call_ts = 0.0

    # ========== 내부 공통 헬퍼 ==========
    def _request(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """알라딘 TTB API 공통 호출 헬퍼.

        Args:
            path: 'ItemSearch.aspx' 등 endpoint 파일명
            params: 쿼리 파라미터 (ttbkey 제외 상태로 전달 가능)
        Returns:
            표준화된 dict {error, status, result|message}
        """
        # 간단한 초당 호출 간격 완화
        now = time.time()
        if now - self._last_call_ts < 0.05:
            time.sleep(0.05)
        self._last_call_ts = time.time()

        url = f"{self.base_url}/{path}"
        merged = {
            "ttbkey": self.api_key,
            "Version": self.version,
            **params
        }
        try:
            resp = requests.get(url, params=merged, timeout=self.default_timeout)
        except requests.exceptions.Timeout:
            return {"error": True, "status": "timeout", "message": "요청 시간이 초과되었습니다."}
        except requests.exceptions.RequestException as e:
            return {"error": True, "status": "network_error", "message": str(e)}

        if resp.status_code != 200:
            return {"error": True, "status": resp.status_code, "message": resp.text[:500]}

        # 알라딘은 XML 혹은 JS(JSONP 형태) 지원. 여기서는 raw text 반환 후 상위 소비자에서 파싱하도록 둠.
        # output=JS 인 경우 JSON 처리를 시도.
        content_type = resp.headers.get("Content-Type", "")
        body_text = resp.text
        parsed: Any = body_text
        if "json" in content_type.lower() or merged.get("output", "").upper() == "JS":
            try:
                parsed = resp.json()
            except Exception:
                # JSON 파싱 실패 시 raw 유지
                pass

        return {"error": False, "status": 200, "result": parsed, "raw": body_text}

    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========

    def _search_item(self, query: str, query_type: str = "Keyword", search_target: str = "Book",
                     start: int = 1, max_results: int = 10, sort: str = "Accuracy",
                     cover: str = "Mid", category_id: int = 0, output: str = "XML",
                     out_of_stock_filter: int = 0, opt_result: str = "") -> dict:
        """알라딘 상품 검색 API 호출 (ItemSearch)"""
        if not query:
            return {"error": True, "status": "invalid_param", "message": "query는 필수입니다."}
        max_results = max(1, min(max_results, 100))
        start = max(1, start)
        params = {
            "Query": query,
            "QueryType": query_type,
            "SearchTarget": search_target,
            "Start": start,
            "MaxResults": max_results,
            "Sort": sort,
            "Cover": cover,
            "CategoryId": category_id,
            "output": output.lower(),  # xml/js
            "outofStockFilter": out_of_stock_filter,
        }
        if opt_result:
            params["OptResult"] = opt_result
        return self._request("ItemSearch.aspx", params)

    def _get_item_list(self, query_type: str, search_target: str = "Book",
                       sub_search_target: str = "", start: int = 1, max_results: int = 10,
                       cover: str = "Mid", category_id: int = 0, year: int | None = None,
                       month: int | None = None, week: int | None = None, output: str = "XML",
                       out_of_stock_filter: int = 0) -> dict:
        """알라딘 상품 리스트 조회 API 호출 (ItemList)"""
        if not query_type:
            return {"error": True, "status": "invalid_param", "message": "query_type은 필수입니다."}
        max_results = max(1, min(max_results, 100))
        start = max(1, start)
        params: Dict[str, Any] = {
            "QueryType": query_type,
            "SearchTarget": search_target,
            "Start": start,
            "MaxResults": max_results,
            "Cover": cover,
            "CategoryId": category_id,
            "output": output.lower(),
            "outofStockFilter": out_of_stock_filter,
        }
        if sub_search_target:
            params["SubSearchTarget"] = sub_search_target
        if year:
            params["Year"] = year
        if month:
            params["Month"] = month
        if week:
            params["Week"] = week
        return self._request("ItemList.aspx", params)

    def _get_item_details(self, item_id: str, item_id_type: str = "ISBN",
                          cover: str = "Mid", output: str = "XML", opt_result: str = "") -> dict:
        """알라딘 상품 상세 정보 조회 API 호출 (ItemLookUp)"""
        if not item_id:
            return {"error": True, "status": "invalid_param", "message": "item_id는 필수입니다."}
        params: Dict[str, Any] = {
            "ItemId": item_id,
            "ItemIdType": item_id_type,
            "Cover": cover,
            "output": output.lower(),
        }
        if opt_result:
            params["OptResult"] = opt_result
        return self._request("ItemLookUp.aspx", params)


    def _purchase_item(self, item_id: str, quantity: int = 1,
                       delivery_address: str = "", payment_method: str = "card") -> dict:
        """실제 구매 API가 없으므로 모의 구현.

        간단히 item_id와 수량만 검증 후 주문 ID 생성.
        """
        if not item_id:
            return {"error": True, "status": "invalid_param", "message": "item_id 필수"}
        quantity = max(1, min(quantity, 99))
        order_id = f"mock-order-{uuid.uuid4().hex[:12]}"
        return {
            "error": False,
            "status": 200,
            "order_id": order_id,
            "item_id": item_id,
            "quantity": quantity,
            "payment_method": payment_method,
            "delivery_address": delivery_address or "DEFAULT",
            "message": "모의 구매 성공 (실제 결제 아님)"
        }

    # ========== Tool Calling 스키마 메서드들 ==========
    
    def ItemSearch_aladin(self) -> dict:
        """상품 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "ItemSearch_aladin",
                "description": "키워드, 카테고리 등 상세 검색 조건으로 상품을 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Query": {
                            "type": "string",
                            "description": "검색어"
                        },
                        "QueryType": {
                            "type": "string",
                            "enum": ["Keyword", "Title", "Author", "Publisher"],
                            "description": "검색어 종류, 기본값: Keyword(제목+저자)",
                            "default": "Keyword"
                        },
                        "SearchTarget": {
                            "type": "string",
                            "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                            "description": "검색 대상 Mall, 기본값: Book(도서)",
                            "default": "Book"
                        },
                        "Start": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "검색 시작 페이지, 기본값: 1",
                            "default": 1
                        },
                        "MaxResults": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "한 페이지에 보여질 상품 수, 기본값: 10",
                            "default": 10
                        },
                        "Sort": {
                            "type": "string",
                            "enum": ["Accuracy", "PublishTime", "SalesPoint", "CustomerRating", "MyReviewCount"],
                            "description": "정렬 방식, 기본값: Accuracy(관련도순)",
                            "default": "Accuracy"
                        },
                        "Cover": {
                            "type": "string",
                            "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                            "description": "표지 이미지 크기, 기본값: Mid",
                            "default": "Mid"
                        },
                        "CategoryId": {
                            "type": "integer",
                            "description": "분야의 고유 번호로 검색 결과를 제한합니다. (기본값: 0, 전체)",
                            "default": 0
                        },
                        "Output": {
                            "type": "string",
                            "enum": ["XML", "JS"],
                            "description": "출력 형식, 기본값: XML",
                            "default": "XML"
                        },
                        "outofStockFilter": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "품절/절판 상품 필터링 여부 (1: 제외), 기본값: 0",
                            "default": 0
                        },
                        "OptResult": {
                            "type": "string",
                            "description": "부가 정보 요청. 쉼표로 구분하여 다중 선택. (예: ebookList, usedList)",
                            "default": ""
                        }
                    },
                    "required": ["Query"]
                }
            }
        }
    
    def ItemList_aladin(self) -> dict:
        """상품 리스트 조회 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "ItemList_aladin",
                "description": "신간, 베스트셀러 등 특정 종류의 상품 리스트를 상세 조건으로 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "QueryType": {
                            "type": "string",
                            "enum": ["ItemNewAll", "ItemNewSpecial", "ItemEditorChoice", "Bestseller", "BlogBest"],
                            "description": "조회할 리스트 종류"
                        },
                        "SearchTarget": {
                            "type": "string",
                            "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                            "description": "조회 대상 Mall, 기본값: Book(도서)",
                            "default": "Book"
                        },
                        "SubSearchTarget": {
                            "type": "string",
                            "enum": ["Book", "Music", "DVD"],
                            "description": "SearchTarget이 Used(중고)일 경우, 서브 Mall 지정",
                            "default": ""
                        },
                        "Start": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "시작 페이지, 기본값: 1",
                            "default": 1
                        },
                        "MaxResults": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "한 페이지에 보여질 상품 수, 기본값: 10",
                            "default": 10
                        },
                        "Cover": {
                            "type": "string",
                            "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                            "description": "표지 이미지 크기, 기본값: Mid",
                            "default": "Mid"
                        },
                        "CategoryId": {
                            "type": "integer",
                            "description": "분야의 고유 번호로 리스트를 제한합니다. (기본값: 0, 전체)",
                            "default": 0
                        },
                        "Year": {
                            "type": "integer",
                            "description": "Bestseller 조회 시 기준 연도 (생략 시 현재)"
                        },
                        "Month": {
                            "type": "integer",
                            "description": "Bestseller 조회 시 기준 월 (생략 시 현재)"
                        },
                        "Week": {
                            "type": "integer",
                            "description": "Bestseller 조회 시 기준 주 (생략 시 현재)"
                        },
                        "Output": {
                            "type": "string",
                            "enum": ["XML", "JS"],
                            "description": "출력 형식, 기본값: XML",
                            "default": "XML"
                        },
                        "outofStockFilter": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "품절/절판 상품 필터링 여부 (1: 제외), 기본값: 0",
                            "default": 0
                        }
                    },
                    "required": ["QueryType"]
                }
            }
        }
    
    def ItemLookup_aladin(self) -> dict:
        """상품 상세 정보 조회 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "ItemLookup_aladin",
                "description": "특정 상품의 상세 정보와 함께 원하는 부가 정보(목차, 리뷰 등)를 함께 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ItemId": {
                            "type": "string",
                            "description": "상품의 고유 ID (ISBN, ISBN13, 또는 알라딘 ItemId)"
                        },
                        "ItemIdType": {
                            "type": "string",
                            "enum": ["ISBN", "ISBN13", "ItemId"],
                            "description": "ItemId의 종류. 가급적 'ISBN13' 사용 권장, 기본값: ISBN",
                            "default": "ISBN"
                        },
                        "Cover": {
                            "type": "string",
                            "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                            "description": "표지 이미지 크기, 기본값: Mid",
                            "default": "Mid"
                        },
                        "Output": {
                            "type": "string",
                            "enum": ["XML", "JS"],
                            "description": "출력 형식, 기본값: XML",
                            "default": "XML"
                        },
                        "OptResult": {
                            "type": "string",
                            "description": "부가 정보 요청. 쉼표로 구분하여 다중 선택 가능. (예: Toc, authors, reviewList, usedList, ebookList, fulldescription, ratingInfo 등)",
                            "default": ""
                        }
                    },
                    "required": ["ItemId"]
                }
            }
        }
    
    def Purchase_aladin(self) -> dict:
        """알라딘 상품구매 tool calling 스키마 (모의)"""
        return {
            "type": "function",
            "function": {
                "name": "Purchase_aladin",
                "description": "모의 상품 구매를 수행합니다. (로그인/결제 실제 수행 안 함)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "구매할 상품의 고유 ID (알라딘 ItemId 또는 ISBN13)"
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 99,
                            "description": "구매 수량, 기본값: 1",
                            "default": 1
                        },
                        "delivery_address": {
                            "type": "string",
                            "description": "배송 주소 (생략 시 기본 배송지 사용)",
                            "default": ""
                        },
                        "payment_method": {
                            "type": "string",
                            "enum": ["card", "bank", "point"],
                            "description": "결제 방법 (card: 신용카드, bank: 계좌이체, point: 적립금), 기본값: card",
                            "default": "card"
                        }
                    },
                    "required": ["item_id"]
                }
            }
        }
    
    # ========== Tool Call 실행기 ==========
    
    def execute_tool(self, tool_name: str, **kwargs) -> dict:
        """Tool call 실행
        
        Args:
            tool_name: 실행할 tool 이름 (ItemSearch_aladin, ItemList_aladin, ItemLookup_aladin, Login_aladin, Purchase_aladin)
            **kwargs: tool별 매개변수
            
        Returns:
            tool 실행 결과
        """
        tool_map = {
            "ItemSearch_aladin": self._search_item,
            "ItemList_aladin": self._get_item_list,
            "ItemLookup_aladin": self._get_item_details,
            "Purchase_aladin": self._purchase_item
        }
        
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
            
        return tool_map[tool_name](**kwargs)
    
    def get_all_tool_schemas(self) -> list[dict]:
        """모든 tool 스키마 반환"""
        return [
            self.ItemSearch_aladin(),
            self.ItemList_aladin(),
            self.ItemLookup_aladin(),
            self.Purchase_aladin()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트 메서드"""
        # ItemSearch API 테스트
        search_url = f"{self.base_url}/ItemSearch.aspx"
        search_params = {
            "ttbkey": self.api_key,
            "Query": "테스트",
            "QueryType": "Keyword",
            "MaxResults": 1,
            "start": 1,
            "SearchTarget": "Book",
            "output": "xml",
            "Version": "20131101"
        }
        
        # ItemLookUp API 테스트
        lookup_url = f"{self.base_url}/ItemLookUp.aspx"
        lookup_params = {
            "ttbkey": self.api_key,
            "ItemId": "9788932473901",  # 테스트용 ISBN13
            "ItemIdType": "ISBN13",
            "output": "xml",
            "Version": "20131101"
        }
        
        try:
            # ItemSearch API 테스트
            print("알라딘 ItemSearch API 연결 테스트 중...")
            search_response = requests.get(search_url, params=search_params, timeout=10)
            print(f"ItemSearch API - 상태 코드: {search_response.status_code}")
            
            if search_response.status_code != 200:
                print(f"ItemSearch API 호출 실패: {search_response.status_code} - {search_response.text}")
                return False
            
            # ItemLookUp API 테스트
            print("알라딘 ItemLookUp API 연결 테스트 중...")
            lookup_response = requests.get(lookup_url, params=lookup_params, timeout=10)
            print(f"ItemLookUp API - 상태 코드: {lookup_response.status_code}")
            
            if lookup_response.status_code != 200:
                print(f"ItemLookUp API 호출 실패: {lookup_response.status_code} - {lookup_response.text}")
                return False
            
            print("모든 알라딘 API 연결 성공!")
            return True
                
        except requests.exceptions.RequestException as e:
            print(f"알라딘 API 네트워크 오류: {e}")
            return False
        except Exception as e:
            print(f"알라딘 API 예상치 못한 오류: {e}")
            return False


if __name__ == "__main__":
    api = AladinAPI()
    print("[알라딘 API 연결 테스트]")
    if not api.test_connection():
        print("❌ 알라딘 API 연결 실패 - 데모 중단")
        raise SystemExit(1)
    print("✅ 알라딘 API 연결 성공\n")

    # 1. 검색 데모
    print("[ItemSearch 데모] '파이썬' 관련 검색 (상위 3개) ...")
    search_res = api._search_item("파이썬", max_results=3, output="JS")
    if search_res.get("error"):
        print("검색 오류:", search_res)
    else:
        print(str(search_res.get("result"))[:500], "...\n")

    # 2. 리스트 데모 (신간)
    print("[ItemList 데모] 신간 전체 (ItemNewAll) 상위 3개 ...")
    list_res = api._get_item_list("ItemNewAll", max_results=3, output="JS")
    if list_res.get("error"):
        print("리스트 오류:", list_res)
    else:
        print(list_res.get("raw", "")[:500], "...\n")

    # 3. 상세조회 (ISBN13 예시 하나 사용)
    sample_isbn13 = "9788932473901"
    print(f"[ItemLookUp 데모] ISBN13={sample_isbn13} ...")
    detail_res = api._get_item_details(sample_isbn13, item_id_type="ISBN13", output="JS")
    if detail_res.get("error"):
        print("상세 오류:", detail_res)
    else:
        print(detail_res.get("raw", "")[:400], "...\n")

    # 4. 구매 (모의)
    print("[Purchase 모의 데모] quantity=2")
    purchase_res = api._purchase_item(sample_isbn13, quantity=2)
    print(purchase_res)

