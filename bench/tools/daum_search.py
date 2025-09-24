import requests
from base_api import BaseAPI

class DaumSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="daum_search_api",
            description="다음 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        self.api_key = "783ca97170d40abe26ba1f7e8a6678a0"

    # ========== 실제 API 호출 메서드들 (비즈니스 로직) ==========

    def _web_search_daum(self, query: str, sort: str = "accuracy", page: int = 1, size: int = 10) -> dict:
        """다음 웹 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            sort: 결과 문서 정렬 방식 (accuracy: 정확도순, recency: 최신순)
            page: 결과 페이지 번호
            size: 한 페이지에 보여질 문서 수 (1~50)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        # TODO: 실제 API 호출 로직 구현
        pass

    def _video_search_daum(self, query: str, sort: str = "accuracy", page: int = 1, size: int = 15) -> dict:
        """다음 비디오 검색 API 호출 (내부 구현)
        
        Args:
            query: 검색어
            sort: 결과 문서 정렬 방식 (accuracy: 정확도순, recency: 최신순)
            page: 결과 페이지 번호 (1~15)
            size: 한 페이지에 보여질 문서 수 (1~30)
            
        Returns:
            검색 결과를 포함한 딕셔너리
        """
        # TODO: 실제 API 호출 로직 구현
        pass

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
    
    def WebSearch_daum(self) -> dict:
        """웹 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "WebSearch_daum",
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
                            "description": "결과 문서 정렬 방식, accuracy(정확도순) 또는 recency(최신순), 기본 값 accuracy",
                            "default": "accuracy"
                        },
                        "page": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "결과 페이지 번호, 1~50 사이의 값, 기본 값 1	",
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
    
    def VideoSearch_daum(self) -> dict:
        """비디오 검색 tool calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "VideoSearch_daum",
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
                            "description": "결과 문서 정렬 방식, accuracy(정확도순) 또는 recency(최신순), 기본 값 accuracy",
                            "default": "accuracy"
                        },
                        "page": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 15,
                            "description": "결과 페이지 번호, 1~15 사이의 값",
                            "default": 1
                        },
                        "size": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 30,
                            "description": "한 페이지에 보여질 문서 수, 1~30 사이의 값, 기본 값 15",
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
            self.web_search_daum_tool(),
            self.video_search_daum_tool()
        ]


if __name__ == "__main__":
    api = DaumSearchAPI()
    if api.test_connection():
        print("다음 검색 API 연결 성공")
    else:
        print("다음 검색 API 연결 실패")