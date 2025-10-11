import requests
import os
from typing import Dict, List, Any
<<<<<<< HEAD

# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
    from .secrets import TMAP_APP_KEY
except ImportError:
    from base_api import BaseAPI
    from secrets import TMAP_APP_KEY
=======
from .base_api import BaseAPI
from .secrets import TMAP_APP_KEY
>>>>>>> e899c6aa717f6ad4a22e0f4f343ce676421236b0

class TmapNavigation(BaseAPI):
    def __init__(self):
        super().__init__(
            name="tmap_navigation",
            description="T맵 API - POI 검색, 경로 안내, 좌표 변환"
        )
        self.base_url = "https://apis.openapi.sk.com"
        self.app_key = TMAP_APP_KEY

    # ========== 실제 API 호출 메서드들 ==========

    def POISearch_tmap(self, searchKeyword: str, count: int = 10, centerLon: float = None,
                    centerLat: float = None, radius: int = None, page: int = 1) -> Dict[str, Any]:
        """POI 통합검색 (내부 구현)"""
        try:
            endpoint = "/tmap/pois"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "appKey": self.app_key
            }
            
            params = {
                "searchKeyword": searchKeyword,
                "count": count,
                "page": page,
            }
            
            # 선택적 파라미터 추가
            if centerLon is not None:
                params["centerLon"] = centerLon
            if centerLat is not None:
                params["centerLat"] = centerLat
            if radius is not None:
                params["radius"] = radius
                
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"POI 검색 API 호출 실패: {e}",
                "suggested_actions": ["API 키 확인", "네트워크 연결 확인"]
            }


    def CarRoute_tmap(self, startX: float, startY: float, endX: float, endY: float,
                   searchOption: int = 0) -> Dict[str, Any]:
        """자동차 경로안내 (내부 구현)"""
        try:
            endpoint = "/tmap/routes"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "appKey": self.app_key
            }
            
            data = {
                "startX": str(startX),
                "startY": str(startY),
                "endX": str(endX),
                "endY": str(endY),
                "searchOption": str(searchOption),
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"자동차 경로 API 호출 실패: {e}",
                "suggested_actions": ["좌표 범위 확인", "경로 검색 옵션 확인"]
            }
        

    def WalkRoute_tmap(self, startX: float, startY: float, endX: float, endY: float) -> Dict[str, Any]:
        """보행자 경로안내 (내부 구현)"""
        try:
            endpoint = "/tmap/routes/pedestrian"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "appKey": self.app_key
            }
            
            data = {
                "startX": str(startX),
                "startY": str(startY),
                "endX": str(endX),
                "endY": str(endY)
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"보행자 경로 API 호출 실패: {e}",
                "suggested_actions": ["좌표 범위 확인", "도보 가능 거리 확인"]
            }


    def Geocoding_tmap(self, fullAddr: str) -> Dict[str, Any]:
        """주소 좌표 변환 (내부 구현)"""
        try:
            endpoint = "/tmap/geo/geocoding"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "appKey": self.app_key
            }
            
            params = {
                "fullAddr": fullAddr
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"주소 변환 API 호출 실패: {e}",
                "suggested_actions": ["주소 형식 확인", "도로명주소 또는 지번주소 사용"]
            }


    def CategorySearch_tmap(self, categories: str, centerLon: float, centerLat: float,
                         radius: int = 1000, count: int = 10) -> Dict[str, Any]:
        """카테고리별 장소 검색 (내부 구현)"""
        try:
            endpoint = "/tmap/pois/categories"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "appKey": self.app_key
            }
            
            params = {
                "categories": categories,
                "centerLon": centerLon,
                "centerLat": centerLat,
                "radius": radius,
                "count": count
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"카테고리 검색 API 호출 실패: {e}",
                "suggested_actions": ["카테고리명 확인", "검색 반경 조정", "좌표 확인"]
            }



    # ========== Tool Calling 스키마 메서드들 ==========

    def poi_search_tool(self) -> Dict:
        """POISearch_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "POISearch_tmap",
                "description": "T map을 통해 키워드로 전국의 장소(POI)를 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "searchKeyword": {
                            "type": "string",
                            "description": "검색할 장소명 또는 키워드 (예: 스타벅스, 강남역 병원, 부산 맛집)"
                        },
                        "count": {
                            "type": "integer",
                            "description": "검색 결과 개수",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 200
                        },
                        "centerLon": {
                            "type": "number",
                            "description": "검색 중심 경도"
                        },
                        "centerLat": {
                            "type": "number",
                            "description": "검색 중심 위도"
                        },
                        "radius": {
                            "type": "integer",
                            "description": "검색 반경(미터)"
                        },
                        "page": {
                            "type": "integer",
                            "description": "페이지 번호",
                            "default": 1,
                            "minimum": 1
                        }
                    },
                    "required": ["searchKeyword"]
                }
            }
        }

    def car_route_tool(self) -> Dict:
        """CarRoute_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CarRoute_tmap",
                "description": "T map을 통해 자동차 최적 경로를 제공합니다. 실시간 교통정보가 반영됩니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "startX": {
                            "type": "number",
                            "description": "출발지 경도 (WGS84 좌표계)"
                        },
                        "startY": {
                            "type": "number",
                            "description": "출발지 위도 (WGS84 좌표계)"
                        },
                        "endX": {
                            "type": "number",
                            "description": "도착지 경도 (WGS84 좌표계)"
                        },
                        "endY": {
                            "type": "number",
                            "description": "도착지 위도 (WGS84 좌표계)"
                        },
                        "searchOption": {
                            "type": "integer",
                            "description": "경로 검색 옵션 (0:추천, 1:무료우선, 2:최소시간, 3:초보운전, 4:고속도로우선, 10:최단거리)",
                            "enum": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            "default": 0
                        }
                    },
                    "required": ["startX", "startY", "endX", "endY"]
                }
            }
        }

    def walk_route_tool(self) -> Dict:
        """WalkRoute_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "WalkRoute_tmap",
                "description": "T map을 통해 보행자 최적 경로를 제공합니다. 도보 이용 시 안전하고 편리한 경로를 안내합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "startX": {
                            "type": "number",
                            "description": "출발지 경도 (WGS84 좌표계)"
                        },
                        "startY": {
                            "type": "number",
                            "description": "출발지 위도 (WGS84 좌표계)"
                        },
                        "endX": {
                            "type": "number",
                            "description": "도착지 경도 (WGS84 좌표계)"
                        },
                        "endY": {
                            "type": "number",
                            "description": "도착지 위도 (WGS84 좌표계)"
                        }
                    },
                    "required": ["startX", "startY", "endX", "endY"]
                }
            }
        }

    def geocoding_tool(self) -> Dict:
        """Geocoding_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "Geocoding_tmap",
                "description": "T map을 통해 주소를 GPS 좌표(위도, 경도)로 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fullAddr": {
                            "type": "string",
                            "description": "변환할 주소 (도로명주소 또는 지번주소, 예: 서울시 강남구 역삼동 825)"
                        }
                    },
                    "required": ["fullAddr"]
                }
            }
        }

    def category_search_tool(self) -> Dict:
        """CategorySearch_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CategorySearch_tmap",
                "description": "T map 카테고리 체계를 이용한 업종별 장소 검색입니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "categories": {
                            "type": "string",
                            "description": "T map 업종 카테고리 (예: 음식점, 병원, 주유소, 편의점, 은행, 카페, 마트, 약국, 학교, 숙박업소)"
                        },
                        "centerLon": {
                            "type": "number",
                            "description": "검색 중심 경도"
                        },
                        "centerLat": {
                            "type": "number",
                            "description": "검색 중심 위도"
                        },
                        "radius": {
                            "type": "integer",
                            "description": "검색 반경(미터)",
                            "default": 1000,
                            "minimum": 100,
                            "maximum": 20000
                        },
                        "count": {
                            "type": "integer",
                            "description": "검색 결과 개수",
                            "default": 10,
                            "maximum": 100
                        }
                    },
                    "required": ["categories", "centerLon", "centerLat"]
                }
            }
        }

    # ========== Tool Call 실행기 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "POISearch_tmap": self.POISearch_tmap,
            "CarRoute_tmap": self.CarRoute_tmap,
            "WalkRoute_tmap": self.WalkRoute_tmap,
            "Geocoding_tmap": self.Geocoding_tmap,
            "CategorySearch_tmap": self.CategorySearch_tmap
        }

        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.poi_search_tool(),
            self.car_route_tool(),
            self.walk_route_tool(),
            self.geocoding_tool(),
            self.category_search_tool()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            headers = {"appKey": self.app_key}
            response = requests.get(f"{self.base_url}/tmap/pois?searchKeyword=테스트&count=1",
                                    headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False