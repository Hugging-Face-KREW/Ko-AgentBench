import requests
import os
from typing import Dict, List, Any
from base_api import BaseAPI

class KakaoLocal(BaseAPI):
    def __init__(self):
        super().__init__(
            name="kakao_local",
            description="카카오 로컬 API - 주소변환, 장소검색, 카테고리 검색"
        )
        self.base_url = "https://dapi.kakao.com"
        self.rest_api_key = os.getenv("KAKAO_REST_API_KEY")

    # ========== 실제 API 호출 메서드들 ==========

    def _address_to_coord(self, address: str) -> Dict[str, Any]:
        """주소-좌표 변환 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _coord_to_address(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """좌표-주소 변환 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _place_search(self, keyword: str, x: float = None, y: float = None,
                      radius: int = None, sort: str = "accuracy",
                      page: int = 1, size: int = 15) -> Dict[str, Any]:
        """키워드 장소 검색 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _category_search(self, category: str, x: float, y: float,
                         radius: int = 1000, page: int = 1, size: int = 15) -> Dict[str, Any]:
        """카테고리 장소 검색 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    # ========== Tool Calling 스키마 메서드들 ==========

    def address_to_coord_tool(self) -> Dict:
        """AddressToCoord_kakao tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "AddressToCoord_kakao",
                "description": "지번 주소 또는 도로명 주소를 위경도 좌표로 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "address": {
                            "type": "string",
                            "description": "변환할 주소 (예: 서울 강남구 테헤란로 231, 서울 강남구 삼성동 159)"
                        }
                    },
                    "required": ["address"]
                }
            }
        }

    def coord_to_address_tool(self) -> Dict:
        """CoordToAddress_kakao tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CoordToAddress_kakao",
                "description": "위경도 좌표를 지번/도로명 주소로 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {
                            "type": "number",
                            "description": "위도 (예: 37.4979)"
                        },
                        "longitude": {
                            "type": "number",
                            "description": "경도 (예: 127.0276)"
                        }
                    },
                    "required": ["latitude", "longitude"]
                }
            }
        }

    def place_search_tool(self) -> Dict:
        """PlaceSearch_kakao tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "PlaceSearch_kakao",
                "description": "키워드로 장소를 검색합니다. 장소명, 업종, 지역명 등으로 검색 가능합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string",
                            "description": "검색할 장소명 또는 키워드 (예: 스타벅스, 강남역 맛집, 이태원 카페)"
                        },
                        "x": {
                            "type": "number",
                            "description": "중심 좌표의 경도"
                        },
                        "y": {
                            "type": "number",
                            "description": "중심 좌표의 위도"
                        },
                        "radius": {
                            "type": "integer",
                            "description": "검색 반경(미터, 최대 20000m)",
                            "maximum": 20000
                        },
                        "sort": {
                            "type": "string",
                            "enum": ["accuracy", "distance"],
                            "description": "정렬 방식",
                            "default": "accuracy"
                        },
                        "page": {
                            "type": "integer",
                            "description": "페이지 번호 (1~45)",
                            "minimum": 1,
                            "maximum": 45,
                            "default": 1
                        },
                        "size": {
                            "type": "integer",
                            "description": "한 페이지 결과 수 (1~15)",
                            "minimum": 1,
                            "maximum": 15,
                            "default": 15
                        }
                    },
                    "required": ["keyword"]
                }
            }
        }

    def category_search_tool(self) -> Dict:
        """CategorySearch_kakao tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CategorySearch_kakao",
                "description": "특정 카테고리의 장소를 좌표 기준 반경 내에서 검색합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["MT1", "CS2", "PS3", "SC4", "AC5", "PK6", "OL7", "SW8", "BK9", "CT1", "AG2", "PO3", "AT4", "AD5", "FD6", "CE7", "HP8", "PM9"],
                            "description": "카테고리 코드 (MT1:대형마트, CS2:편의점, FD6:음식점, CE7:카페, HP8:병원, PM9:약국, SW8:지하철역, BK9:은행 등)"
                        },
                        "x": {
                            "type": "number",
                            "description": "중심 좌표의 경도"
                        },
                        "y": {
                            "type": "number",
                            "description": "중심 좌표의 위도"
                        },
                        "radius": {
                            "type": "integer",
                            "description": "검색 반경(미터)",
                            "default": 1000,
                            "minimum": 0,
                            "maximum": 20000
                        },
                        "page": {
                            "type": "integer",
                            "description": "페이지 번호",
                            "minimum": 1,
                            "default": 1
                        },
                        "size": {
                            "type": "integer",
                            "description": "한 페이지 결과 수 (1~15)",
                            "minimum": 1,
                            "maximum": 15,
                            "default": 15
                        }
                    },
                    "required": ["category"]
                }
            }
        }

    # ========== Tool Call 실행기 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "AddressToCoord_kakao": self._address_to_coord,
            "CoordToAddress_kakao": self._coord_to_address,
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
            self.coord_to_address_tool(),
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