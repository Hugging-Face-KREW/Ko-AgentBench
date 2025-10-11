import requests
from typing import Dict, List, Any

# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
    from .secrets import KAKAO_REST_API_KEY
except ImportError:
    from base_api import BaseAPI
    from secrets import KAKAO_REST_API_KEY

class KakaoLocal(BaseAPI):
    def __init__(self):
        super().__init__(
            name="kakao_local",
            description="카카오 로컬 API - 주소변환, 장소검색, 카테고리 검색"
        )
        self.base_url = "https://dapi.kakao.com"
        self.rest_api_key = KAKAO_REST_API_KEY

    # ========== 실제 API 호출 메서드들 ==========

    def _address_to_coord(self, address: str) -> Dict[str, Any]:
        """주소-좌표 변환"""
        endpoint = "/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
        params = {"query": address}

        response = requests.get(f"{self.base_url}{endpoint}",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if not data.get("documents"):
            raise ValueError(f"주소 '{address}'를 찾을 수 없습니다")

        result = data["documents"][0]
        return {
            "address": address,
            "latitude": float(result["y"]),
            "longitude": float(result["x"]),
            "address_name": result.get("address_name", "")
        }

    def _place_search(self, keyword: str, x: float = None, y: float = None,
                      radius: int = None, sort: str = "accuracy",
                      page: int = 1, size: int = 15) -> Dict[str, Any]:
        """키워드 장소 검색"""
        endpoint = "/v2/local/search/keyword.json"
        headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
        params = {
            "query": keyword,
            "sort": sort,
            "page": page,
            "size": size
        }
        if x and y:
            params["x"] = x
            params["y"] = y
        if radius:
            params["radius"] = min(radius, 20000)

        response = requests.get(f"{self.base_url}{endpoint}",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        places = []
        for place in data.get("documents", []):
            places.append({
                "name": place.get("place_name"),
                "latitude": float(place.get("y")),
                "longitude": float(place.get("x")),
                "address": place.get("address_name"),
                "road_address": place.get("road_address_name"),
                "phone": place.get("phone"),
                "category": place.get("category_name"),
                "distance": place.get("distance", "")
            })

        return {
            "keyword": keyword,
            "total_count": data.get("meta", {}).get("total_count", 0),
            "places": places
        }

    def _category_search(self, category: str, x: float, y: float,
                         radius: int = 1000, page: int = 1, size: int = 15) -> Dict[str, Any]:
        """카테고리 장소 검색"""
        endpoint = "/v2/local/search/category.json"
        headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
        params = {
            "category_group_code": category,
            "x": x,
            "y": y,
            "radius": min(radius, 20000),
            "page": page,
            "size": size
        }

        response = requests.get(f"{self.base_url}{endpoint}",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        places = []
        for place in data.get("documents", []):
            places.append({
                "name": place.get("place_name"),
                "latitude": float(place.get("y")),
                "longitude": float(place.get("x")),
                "address": place.get("address_name"),
                "phone": place.get("phone"),
                "distance": place.get("distance", "")
            })

        return {
            "category": category,
            "total_count": data.get("meta", {}).get("total_count", 0),
            "places": places
        }

    # ========== Tool Calling 스키마 메서드들 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "AddressToCoord_kakao": self._address_to_coord,
            "PlaceSearch_kakao": self._place_search,
            "CategorySearch_kakao": self._category_search
        }

        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.address_to_coord_tool(),
            self.place_search_tool(),
            self.category_search_tool()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
            response = requests.get(f"{self.base_url}/v2/local/search/address.json?query=테스트",
                                    headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False