import os
import time
from typing import Any, Dict
import requests
from base_api import BaseAPI

class DaumSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="daum_search_api",
            description="다음 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        # 환경변수 KAKAO_REST_API_KEY 우선 사용
        self.api_key = os.getenv("KAKAO_REST_API_KEY", "783ca97170d40abe26ba1f7e8a6678a0")
        self.base_url = "https://dapi.kakao.com/v2/search"
        self.default_timeout = 10
        self._last_call_ts = 0.0

    # ========== 내부 공통 헬퍼 ==========
    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """카카오(다음) 검색 API 공통 호출.

        Args:
            endpoint: '/web', '/vclip' 등
            params: 쿼리 파라미터
        Returns:
            dict: {error, status, result|message}
        """
        now = time.time()
        if now - self._last_call_ts < 0.05:
            time.sleep(0.05)
        self._last_call_ts = time.time()

        headers = {"Authorization": f"KakaoAK {self.api_key}"}
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=self.default_timeout)
        except requests.exceptions.Timeout:
            return {"error": True, "status": "timeout", "message": "요청 시간이 초과되었습니다."}
        except requests.exceptions.RequestException as e:
            return {"error": True, "status": "network_error", "message": str(e)}

        if resp.status_code != 200:
            # 카카오 에러 JSON 시도 파싱
            try:
                data = resp.json()
            except Exception:
                data = {"message": resp.text}
            return {"error": True, "status": resp.status_code, "message": data.get("message", "알 수 없는 오류")}

        try:
            data = resp.json()
        except ValueError:
            return {"error": True, "status": "invalid_json", "message": "JSON 파싱 실패"}

        return {"error": False, "status": 200, "result": data}

    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========

    def _web_search_daum(self, query: str, sort: str = "accuracy", page: int = 1, size: int = 10) -> dict:
        """다음 웹 검색 API 호출"""
        if not query:
            return {"error": True, "status": "invalid_param", "message": "query는 필수입니다."}
        sort = sort if sort in ("accuracy", "recency") else "accuracy"
        page = max(1, min(page, 50))  # 카카오 웹검색 page 최대 50 (문서 참고)
        size = max(1, min(size, 50))
        params = {"query": query, "sort": sort, "page": page, "size": size}
        return self._request("/web", params)

    def _video_search_daum(self, query: str, sort: str = "accuracy", page: int = 1, size: int = 15) -> dict:
        """다음 비디오 검색 API 호출"""
        if not query:
            return {"error": True, "status": "invalid_param", "message": "query는 필수입니다."}
        sort = sort if sort in ("accuracy", "recency") else "accuracy"
        page = max(1, min(page, 15))
        size = max(1, min(size, 30))
        params = {"query": query, "sort": sort, "page": page, "size": size}
        return self._request("/vclip", params)

    def test_connection(self):
        """연결 테스트를 위한 메서드"""
        url = "https://dapi.kakao.com/v2/search/web"
        headers = {
            "Authorization": f"KakaoAK {self.api_key}"
        }
        params = {
            "query": "이효리"
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            print(f"연결 성공! 상태 코드: {response.status_code}")
            print(f"응답 데이터: {response.json()}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"연결 실패: {e}")
            return False

    # ========== Tool Calling 스키마 메서드들 ==========
    
    def web_search_daum(self) -> dict:
        """웹 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "web_search_daum",
                "description": "다음 검색 서비스에서 질의어로 웹 문서를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색을 원하는 질의어"
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["accuracy", "recency"],
                            "description": "결과 문서 정렬 방식 (accuracy: 정확도순, recency: 최신순), 기본 값 accuracy",
                            "default": "accuracy"
                        },
                        "page": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "결과 페이지 번호",
                            "default": 1
                        },
                        "size": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "한 페이지에 보여질 문서 수 (1~50), 기본 값 10",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    def search_video(self) -> dict:
        """비디오 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "search_video",
                "description": "카카오 TV, 유튜브 등 서비스에서 질의어로 동영상을 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "검색을 원하는 질의어"
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["accuracy", "recency"],
                            "description": "결과 문서 정렬 방식 (accuracy: 정확도순, recency: 최신순), 기본 값 accuracy",
                            "default": "accuracy"
                        },
                        "page": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 15,
                            "description": "결과 페이지 번호 (1~15)",
                            "default": 1
                        },
                        "size": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "description": "한 페이지에 보여질 문서 수 (1~30), 기본 값 15",
                            "default": 15
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
            tool_name: 실행할 tool 이름 (web_search_daum, search_video)
            **kwargs: tool별 매개변수
            
        Returns:
            tool 실행 결과
        """
        tool_map = {
            "web_search_daum": self._web_search_daum,
            "search_video": self._video_search_daum
        }
        
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
            
        return tool_map[tool_name](**kwargs)
    
    def get_all_tool_schemas(self) -> list[dict]:
        """모든 tool 스키마 반환"""
        return [
            self.web_search_daum(),
            self.search_video()
        ]


if __name__ == "__main__":
    api = DaumSearchAPI()
    print("[다음 검색 API 연결 테스트]")
    if not api.test_connection():
        print("❌ 연결 실패 - 데모 중단")
        raise SystemExit(1)
    print("✅ 연결 성공\n")

    sample_query = "파이썬"
    print(f"[웹 검색 데모] query='{sample_query}' size=3")
    web_res = api._web_search_daum(sample_query, size=3)
    if web_res.get("error"):
        print("웹 검색 오류:", web_res)
    else:
        docs = web_res["result"].get("documents", [])
        for i, d in enumerate(docs, 1):
            print(f"{i}. {d.get('title')} -> {d.get('url')}")

    print(f"\n[비디오 검색 데모] query='{sample_query}' size=3")
    video_res = api._video_search_daum(sample_query, size=3)
    if video_res.get("error"):
        print("비디오 검색 오류:", video_res)
    else:
        vids = video_res["result"].get("documents", [])
        for i, v in enumerate(vids, 1):
            print(f"{i}. {v.get('title')} -> {v.get('url')}")