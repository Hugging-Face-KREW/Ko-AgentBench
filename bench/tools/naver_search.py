import requests

# 상대 임포트와 절대 임포트 모두 지원
try:
    # 패키지 내부에서 임포트할 때 (외부에서 사용)
    from .base_api import BaseAPI
    from .secrets import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET
except ImportError:
    # 직접 실행할 때 (스크립트로 실행)
    from base_api import BaseAPI
    from secrets import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

class NaverSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="naver_search_api",
            description="네이버 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        self.client_id = NAVER_CLIENT_ID
        self.client_secret = NAVER_CLIENT_SECRET
    
    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========
    
    def _search_web(self, query: str, display: int = 10, start: int = 1, sort: str = "sim") -> dict:
        """네이버 웹 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            sort: 검색 결과 정렬 방법 (sim: 정확도순, date: 날짜순)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        url = "https://openapi.naver.com/v1/search/webkr.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    def _search_blog(self, query: str, display: int = 10, start: int = 1, sort: str = "sim") -> dict:
        """네이버 블로그 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            sort: 검색 결과 정렬 방법 (sim: 정확도순, date: 날짜순)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        url = "https://openapi.naver.com/v1/search/blog.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def _search_news(self, query: str, display: int = 10, start: int = 1, sort: str = "sim") -> dict:
        """네이버 뉴스 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            sort: 검색 결과 정렬 방법 (sim: 정확도순, date: 날짜순)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        url = "https://openapi.naver.com/v1/search/news.json"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        params = {
            "query": query,
            "display": display,
            "start": start,
            "sort": sort
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    
    # ========== Tool Calling 스키마 메서드들 ==========
    
    def Search_naver_web(self) -> dict:
        """웹 검색 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "Search_naver_web",
                "description": "네이버 웹 검색 API 호출",
                "parameters": {
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
                            "description": "검색 결과 정렬 방법, sim: 정확도순으로 내림차순 정렬(기본값) date: 날짜순으로 내림차순 정렬",
                            "default": "sim"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def Search_naver_blog(self) -> dict:
        """블로그 검색 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "Search_naver_blog",
                "description": "네이버 블로그 검색 API 호출",
                "parameters": {
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
                            "description": "검색 결과 정렬 방법, sim: 정확도순으로 내림차순 정렬(기본값) date: 날짜순으로 내림차순 정렬",
                            "default": "sim"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def Search_naver_news(self) -> dict:
        """뉴스 검색 tool calling 스키마
        
        Returns:
            OpenAI function calling 형식의 스키마
        """
        return {
            "type": "function",
            "function": {
                "name": "Search_naver_news",
                "description": "네이버 뉴스 검색 API 호출", 
                "parameters": {
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
                            "description": "검색 결과 정렬 방법, sim: 정확도순으로 내림차순 정렬(기본값) date: 날짜순으로 내림차순 정렬",
                            "default": "sim"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    # ========== Tool Call 실행기 ==========
    
    def execute_tool(self, tool_name: str, **kwargs) -> dict:
        """Tool call 실행
        
        Args:
            tool_name: 실행할 tool 이름 (Search_naver_web, Search_naver_blog, Search_naver_news, search_web, search_blog, search_news)
            **kwargs: tool별 매개변수
            
        Returns:
            tool 실행 결과
        """
        tool_map = {
            "Search_naver_web": self._search_web,
            "Search_naver_blog": self._search_blog,
            "Search_naver_news": self._search_news,
            # Alias for dataset compatibility
            "search_web": self._search_web,
            "search_blog": self._search_blog,
            "search_news": self._search_news,
        }
        
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
            
        return tool_map[tool_name](**kwargs)
    
    def get_all_tool_schemas(self) -> list[dict]:
        """모든 tool 스키마 반환"""
        return [
            self.Search_naver_web(),
            self.Search_naver_blog(),
            self.Search_naver_news()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트 메서드
        
        Returns:
            True if connection is successful, False otherwise
        """
        endpoints = [
            ("웹 검색", "https://openapi.naver.com/v1/search/webkr.json"),
            ("블로그 검색", "https://openapi.naver.com/v1/search/blog.json"),
            ("뉴스 검색", "https://openapi.naver.com/v1/search/news.json")
        ]
        
        print("=" * 50)
        print("네이버 검색 API 연결 테스트")
        print("=" * 50)
        
        for name, url in endpoints:
            try:
                headers = {
                    "X-Naver-Client-Id": self.client_id,
                    "X-Naver-Client-Secret": self.client_secret
                }
                params = {
                    "query": "김치볶음밥 만드는 법",  # 간단한 테스트 쿼리
                    "display": 1  # 최소한의 결과만 요청
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                
                # 200 상태 코드가 아니면 실패
                if response.status_code != 200:
                    print(f"❌ {name} - 실패 (상태 코드: {response.status_code})")
                    print(f"   응답: {response.text}")
                    return False
                
                print(f"✅ {name} - 성공 (상태 코드: {response.status_code})")
                print(f"\n📋 {name} API 응답 결과:")
                print("-" * 50)
                import json
                print(json.dumps(response.json(), ensure_ascii=False, indent=2))
                print("-" * 50)
                print()
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ {name} - 네트워크 오류: {e}")
                return False
            except Exception as e:
                print(f"❌ {name} - 예상치 못한 오류: {e}")
                return False
        
        print("=" * 50)
        print("✅ 모든 네이버 검색 API 연결 성공!")
        print("=" * 50)
        # 모든 API가 성공하면 True 반환
        return True
        
        
if __name__ == "__main__":
    api = NaverSearchAPI()
    if api.test_connection():
        print("네이버 검색 API 연결 성공")
    else:
        print("네이버 검색 API 연결 실패")
