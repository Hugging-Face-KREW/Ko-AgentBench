import requests
from base_tool import BaseAPI

class NaverSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="naver_search_api",
            description="네이버 검색 API를 통한 웹, 블로그, 뉴스 검색 도구"
        )
        self.client_id = "nfe9e3rPKhRY5G3qwzuf"
        self.client_secret = "Il8nrlEM3r"
        
    def get_available_methods(self) -> list[str]:
        return ["search_web", "search_blog", "search_news"]
    
    def WebSearch_naver(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 웹 검색 API 호출
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
        """
        
        pass
    
    def BlogSearch_naver(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 블로그 검색 API 호출
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
        """

        pass

    def NewsSearch_naver(self, query: str, display: int = 10, start: int = 1) -> dict:
        """네이버 뉴스 검색 API 호출
        
        Args:
            query: 검색어
            display: 한 번에 표시할 검색 결과 개수 (기본값: 10, 최대값: 100)
            start: 검색 시작 위치 (기본값: 1, 최대값: 1000)
        """

        pass

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
