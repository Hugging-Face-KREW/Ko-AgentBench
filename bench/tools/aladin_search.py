import requests
from .base_api import BaseAPI
from .secrets import ALADIN_API_KEY

class AladinAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="aladin_api",
            description="알라딘 도서 검색 및 정보 조회 API"
        )
        self.api_key = ALADIN_API_KEY
        self.base_url = "http://www.aladin.co.kr/ttb/api"

    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========

    def _search_item(self, query: str, query_type: str = "Keyword", search_target: str = "Book", 
                    start: int = 1, max_results: int = 10, sort: str = "Accuracy", 
                    cover: str = "Mid", category_id: int = 0, output: str = "js", 
                    out_of_stock_filter: int = 0, opt_result: str = "") -> dict:
        """알라딘 상품 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            query_type: 검색어 종류 (Keyword, Title, Author, Publisher)
            search_target: 검색 대상 (Book, Foreign, Music, DVD, Used, eBook, All)
            start: 검색 시작 페이지
            max_results: 한 페이지에 보여질 상품 수 (1~100)
            sort: 정렬 방식 (Accuracy, PublishTime, SalesPoint, CustomerRating, MyReviewCount)
            cover: 표지 이미지 크기 (Big, MidBig, Mid, Small, Mini, None)
            category_id: 분야의 고유 번호 (0: 전체)
            output: 출력 형식 (xml, js)
            out_of_stock_filter: 품절/절판 상품 필터링 여부 (1: 제외)
            opt_result: 부가 정보 요청
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        url = f"{self.base_url}/ItemSearch.aspx"
        params = {
            "ttbkey": self.api_key,
            "Query": query,
            "QueryType": query_type,
            "SearchTarget": search_target,
            "start": start,
            "MaxResults": max_results,
            "Sort": sort,
            "Cover": cover,
            "CategoryId": category_id,
            "output": output,
            "outofStockFilter": out_of_stock_filter,
            "Version": "20131101"
        }
        
        if opt_result:
            params["OptResult"] = opt_result
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() if output == "js" else response.text

    def _get_item_list(self, query_type: str, search_target: str = "Book", 
                      sub_search_target: str = "", start: int = 1, max_results: int = 10,
                      cover: str = "Mid", category_id: int = 0, year: int = None,
                      month: int = None, week: int = None, output: str = "js",
                      out_of_stock_filter: int = 0) -> dict:
        """알라딘 상품 리스트 조회 API 호출 (내부 구현)
        
        Args:
            query_type: 조회할 리스트 종류 (ItemNewAll, ItemNewSpecial, ItemEditorChoice, Bestseller, BlogBest)
            search_target: 조회 대상 Mall (Book, Foreign, Music, DVD, Used, eBook, All)
            sub_search_target: SearchTarget이 Used일 경우 서브 Mall 지정
            start: 시작 페이지
            max_results: 한 페이지에 보여질 상품 수 (1~100)
            cover: 표지 이미지 크기
            category_id: 분야의 고유 번호
            year: Bestseller 조회 시 기준 연도
            month: Bestseller 조회 시 기준 월
            week: Bestseller 조회 시 기준 주
            output: 출력 형식
            out_of_stock_filter: 품절/절판 상품 필터링 여부
            
        Returns:
            상품 리스트를 포함한 딕셔너리
        """
        url = f"{self.base_url}/ItemList.aspx"
        params = {
            "ttbkey": self.api_key,
            "QueryType": query_type,
            "SearchTarget": search_target,
            "start": start,
            "MaxResults": max_results,
            "Cover": cover,
            "CategoryId": category_id,
            "output": output,
            "outofStockFilter": out_of_stock_filter,
            "Version": "20131101"
        }
        
        if sub_search_target:
            params["SubSearchTarget"] = sub_search_target
        if year:
            params["Year"] = year
        if month:
            params["Month"] = month
        if week:
            params["Week"] = week
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() if output == "js" else response.text

    def _get_item_details(self, item_id: str, item_id_type: str = "ISBN13", 
                         cover: str = "Mid", output: str = "js", opt_result: str = "") -> dict:
        """알라딘 상품 상세 정보 조회 API 호출 (내부 구현)
        
        Args:
            item_id: 상품의 고유 ID (ISBN, ISBN13, 또는 알라딘 ItemId)
            item_id_type: ItemId의 종류 (ISBN, ISBN13, ItemId)
            cover: 표지 이미지 크기
            output: 출력 형식
            opt_result: 부가 정보 요청 (Toc, authors, reviewList, usedList, ebookList, fulldescription, ratingInfo 등)
            
        Returns:
            상품 상세 정보를 포함한 딕셔너리
        """
        url = f"{self.base_url}/ItemLookUp.aspx"
        params = {
            "ttbkey": self.api_key,
            "ItemId": item_id,
            "ItemIdType": item_id_type,
            "Cover": cover,
            "output": output,
            "Version": "20131101"
        }
        
        if opt_result:
            params["OptResult"] = opt_result
            
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json() if output == "js" else response.text

    # ========== Tool Calling 스키마 메서드들 ==========
    
    def ItemSearch_aladin(self) -> dict:
        """상품 검색 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "ItemSearch_aladin",
                "description": "키워드, 카테고리 등 상세 검색 조건으로 상품을 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색어"
                        },
                        "query_type": {
                            "type": "string",
                            "enum": ["Keyword", "Title", "Author", "Publisher"],
                            "description": "검색어 종류, 기본값: Keyword(제목+저자)",
                            "default": "Keyword"
                        },
                        "search_target": {
                            "type": "string",
                            "enum": ["Book", "Foreign", "Music", "DVD", "Used", "eBook", "All"],
                            "description": "검색 대상 Mall, 기본값: Book(도서)",
                            "default": "Book"
                        },
                        "start": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "검색 시작 페이지, 기본값: 1",
                            "default": 1
                        },
                        "max_results": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "description": "한 페이지에 보여질 상품 수, 기본값: 10",
                            "default": 10
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["Accuracy", "PublishTime", "SalesPoint", "CustomerRating", "MyReviewCount"],
                            "description": "정렬 방식, 기본값: Accuracy(관련도순)",
                            "default": "Accuracy"
                        },
                        "cover": {
                            "type": "string",
                            "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                            "description": "표지 이미지 크기, 기본값: Mid",
                            "default": "Mid"
                        },
                        "category_id": {
                            "type": "integer",
                            "description": "분야의 고유 번호로 검색 결과를 제한합니다. (기본값: 0, 전체)",
                            "default": 0
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
                        },
                        "opt_result": {
                            "type": "string",
                            "description": "부가 정보 요청. 쉼표로 구분하여 다중 선택. (예: ebookList, usedList)",
                            "default": ""
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def ItemList_aladin(self) -> dict:
        """상품 리스트 조회 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "ItemList_aladin",
                "description": "신간, 베스트셀러 등 특정 종류의 상품 리스트를 상세 조건으로 조회합니다.",
                "parameters": {
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
            }
        }
    
    def Lookup_aladin_item(self) -> dict:
        """상품 상세 정보 조회 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "Lookup_aladin_item",
                "description": "특정 상품의 상세 정보와 함께 원하는 부가 정보(목차, 리뷰 등)를 함께 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "상품의 고유 ID (ISBN, ISBN13, 또는 알라딘 ItemId)"
                        },
                        "item_id_type": {
                            "type": "string",
                            "enum": ["ISBN", "ISBN13", "ItemId"],
                            "description": "ItemId의 종류. 가급적 'ISBN13' 사용 권장, 기본값: ISBN13",
                            "default": "ISBN13"
                        },
                        "cover": {
                            "type": "string",
                            "enum": ["Big", "MidBig", "Mid", "Small", "Mini", "None"],
                            "description": "표지 이미지 크기, 기본값: Mid",
                            "default": "Mid"
                        },
                        "output": {
                            "type": "string",
                            "enum": ["xml", "js"],
                            "description": "출력 형식, 기본값: js",
                            "default": "js"
                        },
                        "opt_result": {
                            "type": "string",
                            "description": "부가 정보 요청. 쉼표로 구분하여 다중 선택 가능. (예: Toc, authors, reviewList, usedList, ebookList, fulldescription, ratingInfo 등)",
                            "default": ""
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
            tool_name: 실행할 tool 이름 (ItemSearch_aladin, ItemList_aladin, Lookup_aladin_item)
            **kwargs: tool별 매개변수
            
        Returns:
            tool 실행 결과
        """
        tool_map = {
            "ItemSearch_aladin": self._search_item,
            "ItemList_aladin": self._get_item_list,
            "Lookup_aladin_item": self._get_item_details,
            # Backward compatibility
            "Search_aladin_item": self._search_item,
            "List_aladin_item": self._get_item_list,
        }
        
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
            
        return tool_map[tool_name](**kwargs)
    
    def get_all_tool_schemas(self) -> list[dict]:
        """모든 tool 스키마 반환"""
        return [
            self.ItemSearch_aladin(),
            self.ItemList_aladin(),
            self.Lookup_aladin_item()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트 메서드
        
        Returns:
            True if connection is successful, False otherwise
        """
        print("=" * 50)
        print("알라딘 API 연결 테스트")
        print("=" * 50)
        
        # ItemSearch API 테스트
        search_url = f"{self.base_url}/ItemSearch.aspx"
        search_params = {
            "ttbkey": self.api_key,
            "Query": "테스트",
            "QueryType": "Keyword",
            "MaxResults": 1,
            "start": 1,
            "SearchTarget": "Book",
            "output": "js",
            "Version": "20131101"
        }
        
        # ItemLookUp API 테스트
        lookup_url = f"{self.base_url}/ItemLookUp.aspx"
        lookup_params = {
            "ttbkey": self.api_key,
            "ItemId": "9788932473901",  # 테스트용 ISBN13
            "ItemIdType": "ISBN13",
            "output": "js",
            "Version": "20131101"
        }
        
        try:
            # ItemSearch API 테스트
            search_response = requests.get(search_url, params=search_params, timeout=10)
            
            if search_response.status_code != 200:
                print(f"❌ ItemSearch API - 실패 (상태 코드: {search_response.status_code})")
                print(f"   응답: {search_response.text}")
                return False
            
            print(f"✅ ItemSearch API - 성공 (상태 코드: {search_response.status_code})")
            
            # ItemLookUp API 테스트
            lookup_response = requests.get(lookup_url, params=lookup_params, timeout=10)
            
            if lookup_response.status_code != 200:
                print(f"❌ ItemLookUp API - 실패 (상태 코드: {lookup_response.status_code})")
                print(f"   응답: {lookup_response.text}")
                return False
            
            print(f"✅ ItemLookUp API - 성공 (상태 코드: {lookup_response.status_code})")
            
            print("=" * 50)
            print("✅ 모든 알라딘 API 연결 성공!")
            print("=" * 50)
            return True
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 알라딘 API - 네트워크 오류: {e}")
            return False
        except Exception as e:
            print(f"❌ 알라딘 API - 예상치 못한 오류: {e}")
            return False


if __name__ == "__main__":
    api = AladinAPI()
    if api.test_connection():
        print("알라딘 API 연결 성공")
    else:
        print("알라딘 API 연결 실패")

