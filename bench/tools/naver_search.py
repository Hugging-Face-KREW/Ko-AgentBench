import os
import time
from typing import Any, Dict
import requests
from base_api import BaseAPI

class NaverSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="naver_search_api",
            description="네이버 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        # 환경 변수 우선, 없으면 기존 하드코딩 값 (향후 제거 권장)
        self.client_id = os.getenv("NAVER_CLIENT_ID", "nfe9e3rPKhRY5G3qwzuf")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "Il8nrlEM3r")
        self.base_url = "https://openapi.naver.com/v1/search"
        self.default_timeout = 10
        self._last_call_ts = 0.0  # 간단한 rate-limit 제어 (120 req/min 등 고려시 슬립)

    # ========== 내부 공통 헬퍼 ==========
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """네이버 검색 REST API 공통 호출 헬퍼

        Args:
            endpoint: '/webkr' 등 엔드포인트 suffix
            params: 쿼리 파라미터
        Returns:
            파싱된 JSON(dict). 실패 시 {'error': True, 'status': code, 'message': str}
        """
        # 간단한 rate-limit 방지 (짧은 연속 호출시 0.05s sleep)
        now = time.time()
        if now - self._last_call_ts < 0.05:
            time.sleep(0.05)
        self._last_call_ts = time.time()

        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
        }
        url = f"{self.base_url}{endpoint}.json"
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=self.default_timeout)
        except requests.exceptions.Timeout:
            return {"error": True, "status": "timeout", "message": "요청 시간이 초과되었습니다."}
        except requests.exceptions.RequestException as e:
            return {"error": True, "status": "network_error", "message": str(e)}

        if resp.status_code != 200:
            # 네이버는 오류시 JSON을 반환하는 경우가 많으므로 시도
            try:
                data = resp.json()
            except Exception:
                data = {"errorMessage": resp.text}
            return {
                "error": True,
                "status": resp.status_code,
                "message": data.get("errorMessage") or data.get("message") or "알 수 없는 오류"
            }

        try:
            data = resp.json()
        except ValueError:
            return {"error": True, "status": "invalid_json", "message": "JSON 파싱 실패"}

        # 성공 결과 표준화 wrapper
        return {
            "error": False,
            "status": 200,
            "result": data
        }
    
    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========
    
    def _search_web(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 웹 검색 API 호출 (내부 구현)

        Args:
            query: 검색어
            display: 1~100
            start: 1~1000
        Returns:
            통합 결과 dict (오류 형식 포함)
        """
        if not query or not isinstance(query, str):
            return {"error": True, "status": "invalid_param", "message": "query는 비어있지 않은 문자열이어야 합니다."}
        display = max(1, min(display, 100))
        start = max(1, min(start, 1000))
        params = {"query": query, "display": display, "start": start}
        return self._request("/webkr", params)
    
    def _search_blog(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 블로그 검색 API 호출 (내부 구현)"""
        if not query or not isinstance(query, str):
            return {"error": True, "status": "invalid_param", "message": "query는 비어있지 않은 문자열이어야 합니다."}
        display = max(1, min(display, 100))
        start = max(1, min(start, 1000))
        params = {"query": query, "display": display, "start": start}
        return self._request("/blog", params)

    def _search_news(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 뉴스 검색 API 호출 (내부 구현)"""
        if not query or not isinstance(query, str):
            return {"error": True, "status": "invalid_param", "message": "query는 비어있지 않은 문자열이어야 합니다."}
        display = max(1, min(display, 100))
        start = max(1, min(start, 1000))
        params = {"query": query, "display": display, "start": start}
        return self._request("/news", params)
    
    # ========== Tool Calling 스키마 메서드들 ==========
    
    def search_web(self) -> dict:
        """웹 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "search_web",
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
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def search_blog(self) -> dict:
        """블로그 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "search_blog", 
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
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def search_news(self) -> dict:
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
            self.search_web(),
            self.search_blog(),
            self.search_news()
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
    print("[네이버 검색 API 연결 테스트]")
    if api.test_connection():
        print("✅ 네이버 검색 API 연결 성공\n")
    else:
        print("❌ 네이버 검색 API 연결 실패 - 검색 예시는 시도하지 않습니다.")
        raise SystemExit(1)

    sample_query = "파이썬"  # 예시 검색어
    print(f"[웹 검색 예시] query='{sample_query}'")
    web_res = api._search_web(sample_query, display=3)
    print(web_res if web_res.get("error") else web_res["result"].get("items", [])[:3])

    print(f"\n[블로그 검색 예시] query='{sample_query}'")
    blog_res = api._search_blog(sample_query, display=3)
    print(blog_res if blog_res.get("error") else blog_res["result"].get("items", [])[:3])

    print(f"\n[뉴스 검색 예시] query='{sample_query}'")
    news_res = api._search_news(sample_query, display=3)
    print(news_res if news_res.get("error") else news_res["result"].get("items", [])[:3])
