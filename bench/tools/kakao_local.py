import requests
from typing import Dict, List, Any
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시 필요)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
except ImportError:
    from base_api import BaseAPI

from configs.secrets import KAKAO_REST_API_KEY

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

    def _coord_to_address(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """좌표(위도/경도) → 주소 변환

        Kakao Local API: GET /v2/local/geo/coord2address.json
        - x: 경도(longitude)
        - y: 위도(latitude)
        """
        endpoint = "/v2/local/geo/coord2address.json"
        headers = {"Authorization": f"KakaoAK {self.rest_api_key}"}
        params = {
            "x": longitude,
            "y": latitude,
        }

        response = requests.get(f"{self.base_url}{endpoint}", headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        if not data.get("documents"):
            raise ValueError(f"좌표({latitude}, {longitude})에 대한 주소를 찾을 수 없습니다")

        doc = data["documents"][0]
        road = doc.get("road_address") or {}
        addr = doc.get("address") or {}

        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "road_address": road.get("address_name", ""),
            "jibun_address": addr.get("address_name", ""),
            "building_name": road.get("building_name", ""),
            "region_1depth": (road.get("region_1depth_name") or addr.get("region_1depth_name") or ""),
            "region_2depth": (road.get("region_2depth_name") or addr.get("region_2depth_name") or ""),
            "region_3depth": (road.get("region_3depth_name") or addr.get("region_3depth_name") or ""),
            "zone_no": road.get("zone_no", ""),
        }

    # ========== Tool Calling 스키마 메서드들 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "AddressToCoord_kakao": self._address_to_coord,
            "PlaceSearch_kakao": self._place_search,
            "CategorySearch_kakao": self._category_search,
            "CoordToAddress_kakao": self._coord_to_address,
        }

        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.address_to_coord_tool(),
            self.place_search_tool(),
            self.category_search_tool(),
            self.coord_to_address_tool(),
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

    # ========== Tool 스키마 (TC용) ==========
    def coord_to_address_tool(self) -> Dict[str, Any]:
        """CoordToAddress_kakao Tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CoordToAddress_kakao",
                "description": "위경도 좌표를 지번/도로명 주소로 변환합니다. GPS 위치를 실제 주소로 확인할 때 사용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "number", "description": "위도 (예: 37.4979)"},
                        "longitude": {"type": "number", "description": "경도 (예: 127.0276)"},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }

    def address_to_coord_tool(self) -> Dict[str, Any]:
        """AddressToCoord_kakao Tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "AddressToCoord_kakao",
                "description": "주소를 위경도 좌표로 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {"type": "string", "description": "검색할 주소 (예: 서울 강남구 역삼동)"},
                    },
                    "required": ["address"],
                },
            },
        }

    def place_search_tool(self) -> Dict[str, Any]:
        """PlaceSearch_kakao Tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "PlaceSearch_kakao",
                "description": "키워드로 장소를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "검색 키워드"},
                        "x": {"type": "number", "description": "중심 경도 (선택)"},
                        "y": {"type": "number", "description": "중심 위도 (선택)"},
                        "radius": {"type": "integer", "description": "반경(m, 최대 20000)"},
                        "sort": {"type": "string", "description": "정렬 방식 (accuracy|distance)", "default": "accuracy"},
                        "page": {"type": "integer", "description": "페이지 번호", "default": 1},
                        "size": {"type": "integer", "description": "한 페이지 결과 수", "default": 15}
                    },
                    "required": ["keyword"],
                },
            },
        }

    def category_search_tool(self) -> Dict[str, Any]:
        """CategorySearch_kakao Tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CategorySearch_kakao",
                "description": "카테고리로 장소를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "카테고리 그룹 코드 (MT1, CS2, PS3 등)"},
                        "x": {"type": "number", "description": "중심 경도"},
                        "y": {"type": "number", "description": "중심 위도"},
                        "radius": {"type": "integer", "description": "반경(m, 최대 20000)", "default": 1000},
                        "page": {"type": "integer", "description": "페이지 번호", "default": 1},
                        "size": {"type": "integer", "description": "한 페이지 결과 수", "default": 15}
                    },
                    "required": ["category", "x", "y"],
                },
            },
        }


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("KakaoLocal API 테스트")
    print("=" * 60)
    
    api = KakaoLocal()
    
    # 1. API 연결 테스트
    print("\n[1] API 연결 테스트")
    if api.test_connection():
        print("✓ API 연결 성공")
    else:
        print("✗ API 연결 실패")
        exit(1)
    
    # 2. 주소 → 좌표 변환 테스트
    print("\n[2] 주소 → 좌표 변환 (AddressToCoord_kakao)")
    try:
        result = api.execute_tool("AddressToCoord_kakao", address="서울 강남구 역삼동 737")
        print(f"주소: {result['address']}")
        print(f"위도: {result['latitude']}")
        print(f"경도: {result['longitude']}")
        print(f"정확한 주소: {result['address_name']}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 3. 좌표 → 주소 변환 테스트
    print("\n[3] 좌표 → 주소 변환 (CoordToAddress_kakao)")
    try:
        result = api.execute_tool("CoordToAddress_kakao", latitude=37.4979, longitude=127.0276)
        print(f"좌표: ({result['latitude']}, {result['longitude']})")
        print(f"도로명 주소: {result['road_address']}")
        print(f"지번 주소: {result['jibun_address']}")
        print(f"건물명: {result['building_name']}")
        print(f"우편번호: {result['zone_no']}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 4. 키워드 장소 검색 테스트
    print("\n[4] 키워드 장소 검색 (PlaceSearch_kakao)")
    try:
        result = api.execute_tool("PlaceSearch_kakao", keyword="카카오프렌즈", size=5)
        print(f"검색어: {result['keyword']}")
        print(f"전체 결과 수: {result['total_count']}")
        print(f"검색된 장소 {len(result['places'])}개:")
        for i, place in enumerate(result['places'][:3], 1):
            print(f"  {i}. {place['name']}")
            print(f"     주소: {place['address']}")
            print(f"     전화: {place['phone']}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 5. 카테고리 검색 테스트
    print("\n[5] 카테고리 검색 (CategorySearch_kakao)")
    try:
        # 강남역 근처 편의점(CS2) 검색
        result = api.execute_tool("CategorySearch_kakao", 
                                 category="CS2", 
                                 x=127.0276, 
                                 y=37.4979, 
                                 radius=500,
                                 size=5)
        print(f"카테고리: {result['category']}")
        print(f"전체 결과 수: {result['total_count']}")
        print(f"검색된 장소 {len(result['places'])}개:")
        for i, place in enumerate(result['places'][:3], 1):
            print(f"  {i}. {place['name']}")
            print(f"     주소: {place['address']}")
            print(f"     거리: {place['distance']}m")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 6. Tool 스키마 출력
    print("\n[6] 사용 가능한 Tool 스키마")
    schemas = api.get_all_tool_schemas()
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description']}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)