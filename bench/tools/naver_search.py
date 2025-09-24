import requests
from base_api import BaseAPI

class NaverSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="naver_search_api",
            description="네이버 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        self.client_id = "nfe9e3rPKhRY5G3qwzuf"
        self.client_secret = "Il8nrlEM3r"
    
    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========
    
    def _search_web(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 웹 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        # TODO: 실제 API 호출 로직 구현
        pass
    
    def _search_blog(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 블로그 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        # TODO: 실제 API 호출 로직 구현
        pass

    def _search_news(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 뉴스 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        # TODO: 실제 API 호출 로직 구현
        pass
    
    # ========== Tool Calling 스키마 메서드들 ==========
    
    def WebSearch_naver(self) -> dict:
        """웹 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "WebSearch_naver",
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
                            "minimum": 10,
                            "maximum": 100
                        },
                        "start": {
                            "type": "integer",
                            "description": "검색 시작 위치",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 1
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def BlogSearch_naver(self) -> dict:
        """블로그 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "BlogSearch_naver",
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
                        },
                    },
                    "required": ["query"]
                }
            }
        }
    
    def search_news_tool(self) -> dict:
        """뉴스 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "search_news",
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
                        },
                    },
                    "required": ["query"]
                }
            }
        }
    
    # ========== Tool Call 실행기 ==========
    
    def execute_tool(self, tool_name: str, **kwargs) -> dict:
        """Tool call 실행
        
        Args:
            tool_name: 실행할 tool 이름 (search_web, search_blog, search_news)
            **kwargs: tool별 매개변수
            
        Returns:
            tool 실행 결과
        """
        tool_map = {
            "search_web": self._search_web,
            "search_blog": self._search_blog,
            "search_news": self._search_news
        }
        
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
            
        return tool_map[tool_name](**kwargs)
    
    def get_all_tool_schemas(self) -> list[dict]:
        """모든 tool 스키마 반환"""
        return [
            self.search_web_tool(),
            self.search_blog_tool(),
            self.search_news_tool()
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
        
        for name, url in endpoints:
            try:
                headers = {
                    "X-Naver-Client-Id": self.client_id,
                    "X-Naver-Client-Secret": self.client_secret
                }
                params = {
                    "query": "테스트",  # 간단한 테스트 쿼리
                    "display": 1  # 최소한의 결과만 요청
                }
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                print(f"{name} - 상태 코드: {response.status_code}")
                
                # 200 상태 코드가 아니면 실패
                if response.status_code != 200:
                    print(f"{name} API 호출 실패: {response.status_code} - {response.text}")
                    return False
                    
            except requests.exceptions.RequestException as e:
                print(f"{name} 네트워크 오류: {e}")
                return False
            except Exception as e:
                print(f"{name} 예상치 못한 오류: {e}")
                return False
        
        # 모든 API가 성공하면 True 반환
        return True
        
        
if __name__ == "__main__":
    api = NaverSearchAPI()
    if api.test_connection():
        print("네이버 검색 API 연결 성공")
    else:
        print("네이버 검색 API 연결 실패")
