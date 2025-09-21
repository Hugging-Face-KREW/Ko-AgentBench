import requests
import os
from typing import Dict, List, Any
from base_api import BaseAPI

class TmapNavigation(BaseAPI):
    def __init__(self):
        super().__init__(
            name="tmap_navigation",
            description="T맵 API - POI 검색, 경로 안내, 좌표 변환"
        )
        self.base_url = "https://apis.openapi.sk.com"
        self.app_key = os.getenv("TMAP_APP_KEY")

    # ========== 실제 API 호출 메서드들 ==========

    def _poi_search(self, searchKeyword: str, count: int = 10, centerLon: float = None,
                    centerLat: float = None, radius: int = None, page: int = 1) -> Dict[str, Any]:
        """POI 통합검색 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _car_route(self, startX: float, startY: float, endX: float, endY: float,
                   searchOption: int = 0) -> Dict[str, Any]:
        """자동차 경로안내 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _walk_route(self, startX: float, startY: float, endX: float, endY: float) -> Dict[str, Any]:
        """보행자 경로안내 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _geocoding(self, fullAddr: str) -> Dict[str, Any]:
        """주소 좌표 변환 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _category_search(self, categories: str, centerLon: float, centerLat: float,
                         radius: int = 1000, count: int = 10) -> Dict[str, Any]:
        """카테고리별 장소 검색 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

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
            "POISearch_tmap": self._poi_search,
            "CarRoute_tmap": self._car_route,
            "WalkRoute_tmap": self._walk_route,
            "Geocoding_tmap": self._geocoding,
            "CategorySearch_tmap": self._category_search
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