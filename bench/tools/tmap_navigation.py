import requests
from typing import Dict, Any
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

from configs.secrets import TMAP_APP_KEY


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
                    centerLat: float = None, page: int = 1,
                    reqCoordType: str = "WGS84GEO", resCoordType: str = "WGS84GEO") -> Dict[str, Any]:
        """POI 통합검색 (내부 구현)
        
        주의: radius 파라미터는 API에서 지원하지 않음
        centerLon/centerLat만으로 중심점 기반 검색 가능
        """
        try:
            endpoint = "/tmap/pois"
            url = f"{self.base_url}{endpoint}"
            
            headers = {
                "accept": "application/json",
                "appKey": self.app_key
            }
            
            params = {
                "searchKeyword": searchKeyword,
                "count": min(count, 200),  # 최대 200
                "page": page,
                "reqCoordType": reqCoordType,
                "resCoordType": resCoordType,
            }
            
            # centerLon과 centerLat은 함께 사용해야 함
            if centerLon is not None and centerLat is not None:
                params["centerLon"] = centerLon
                params["centerLat"] = centerLat
                
            response = requests.get(url, headers=headers, params=params, timeout=10)
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
            
            response = requests.post(url, headers=headers, json=data, timeout=10)
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
        """보행자 경로안내 (최신 스펙, 안정버전)"""
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
                "endY": str(endY),
                "reqCoordType": "WGS84GEO",
                "resCoordType": "WGS84GEO",
                "startName": "출발지",
                "endName": "도착지"
            }

            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"보행자 경로 API 호출 실패: {e}",
                "suggested_actions": ["좌표 범위 확인", "도보 가능 거리 확인"]
            }


    def Geocoding_tmap(self,
                       city_do: str,
                       gu_gun: str,
                       dong: str,
                       bunji: str = "",
                       detailAddress: str = "",
                       addressFlag: str = "F01",
                       coordType: str = "WGS84GEO") -> Dict[str, Any]:
        """주소를 GPS 좌표(위도, 경도)로 변환"""
        try:
            endpoint = "/tmap/geo/geocoding"
            url = f"{self.base_url}{endpoint}"
            headers = {"accept": "application/json"}

            params = {
                "version": "1.0",
                "city_do": city_do,
                "gu_gun": gu_gun,
                "dong": dong,
                "addressFlag": addressFlag,
                "coordType": coordType,
                "appKey": self.app_key
            }

            if bunji:
                params["bunji"] = bunji
            if detailAddress:
                params["detailAddress"] = detailAddress

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            print(f"요청 URL: {response.url}")
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"주소 변환 API 호출 실패: {e}"
            }


    def CategorySearch_tmap(self,
                            categories: str,
                            centerLon: float,
                            centerLat: float,
                            radius: int = 1,
                            page: int = 1,
                            count: int = 20,
                            reqCoordType: str = "WGS84GEO",
                            resCoordType: str = "WGS84GEO",
                            sort: str = "distance",
                            multiPoint: str = "N") -> Dict[str, Any]:
        """명칭(POI) 주변 카테고리 검색 (최신 공식문서 기반)"""
        try:
            endpoint = "/tmap/pois/search/around"
            url = f"{self.base_url}{endpoint}"

            headers = {"accept": "application/json", "appKey": self.app_key}

            params = {
                "version": 1,
                "page": page,
                "count": count,
                "categories": categories,
                "centerLon": centerLon,
                "centerLat": centerLat,
                "radius": radius,
                "reqCoordType": reqCoordType,
                "resCoordType": resCoordType,
                "multiPoint": multiPoint,
                "sort": sort
            }

            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if "searchPoiInfo" in data:
                pois = data["searchPoiInfo"].get("pois", {}).get("poi", [])
                parsed = [
                    {
                        "name": p.get("name"),
                        "telNo": p.get("telNo"),
                        "latitude": p.get("noorLat"),
                        "longitude": p.get("noorLon"),
                        "address": f"{p.get('upperAddrName', '')} {p.get('middleAddrName', '')} {p.get('lowerAddrName', '')}".strip(),
                        "roadName": p.get("roadName"),
                        "buildingNo": f"{p.get('firstBuildNo', '')}-{p.get('secondBuildNo', '')}",
                        "radius_km": p.get("radius"),
                        "dataKind": p.get("dataKind"),
                        "stId": p.get("stId"),
                        "parkFlag": p.get("parkFlag"),
                        "merchantFlag": p.get("merchantFlag")
                    } for p in pois
                ]
                return {
                    "totalCount": data["searchPoiInfo"].get("totalCount", 0),
                    "count": data["searchPoiInfo"].get("count", 0),
                    "page": data["searchPoiInfo"].get("page", 1),
                    "pois": parsed
                }

            return data

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error_code": "API_REQUEST_FAILED",
                "error_message": f"카테고리 검색 API 호출 실패: {e}",
                "suggested_actions": [
                    "categories 명칭 확인 (예: 카페, 병원, 주유소 등)",
                    "좌표(centerLon/centerLat) 확인",
                    "반경(radius) 1~33 범위 내 설정",
                    "appKey 권한 확인"
                ]
            }

    # ========== Tool Calling 스키마 메서드들 ==========

    def poi_search_tool(self) -> Dict:
        """POISearch_tmap tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "POISearch_tmap",
                "description": "T map을 통해 키워드로 전국의 장소(POI)를 검색합니다. 맛집, 병원, 주유소, 관광지 등 150만 건의 POI 데이터를 검색할 수 있습니다. 중심점 기반 검색 시 centerLon, centerLat을 사용하세요 (radius는 지원 안 됨).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "searchKeyword": {
                            "type": "string",
                            "description": "검색할 장소명 또는 키워드 (예: 스타벅스, 강남역 병원, 부산 맛집, 서울 이마트). 지역명을 포함하여 검색하세요."
                        },
                        "count": {
                            "type": "integer",
                            "description": "검색 결과 개수 (기본값: 10, 최대: 200)",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 200
                        },
                        "centerLon": {
                            "type": "number",
                            "description": "검색 중심점 경도 (선택사항, centerLat과 함께 사용하여 해당 위치 근처 결과 우선 표시)"
                        },
                        "centerLat": {
                            "type": "number",
                            "description": "검색 중심점 위도 (선택사항, centerLon과 함께 사용하여 해당 위치 근처 결과 우선 표시)"
                        },
                        "page": {
                            "type": "integer",
                            "description": "페이지 번호 (더 많은 결과가 필요할 때 사용)",
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
                "description": "자동차 경로 안내 (Tmap)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "startX": {"type": "number", "description": "출발지 경도"},
                        "startY": {"type": "number", "description": "출발지 위도"},
                        "endX": {"type": "number", "description": "도착지 경도"},
                        "endY": {"type": "number", "description": "도착지 위도"},
                        "searchOption": {"type": "integer", "default": 0, "description": "경로 옵션 (0: 추천, 2: 최단거리 등)"}
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
                        "startX": {"type": "number", "description": "출발지 경도 (WGS84 좌표계)"},
                        "startY": {"type": "number", "description": "출발지 위도 (WGS84 좌표계)"},
                        "endX": {"type": "number", "description": "도착지 경도 (WGS84 좌표계)"},
                        "endY": {"type": "number", "description": "도착지 위도 (WGS84 좌표계)"}
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
                "description": "T map을 통해 시/도, 구/군, 동 입력을 기반으로 GPS 좌표(위도, 경도)로 변환합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city_do": {"type": "string", "description": "시/도 (예: 서울특별시)"},
                        "gu_gun": {"type": "string", "description": "구/군 (예: 강남구)"},
                        "dong": {"type": "string", "description": "동 (예: 역삼동)"},
                        "bunji": {"type": "string", "description": "번지 (선택)", "default": ""},
                        "detailAddress": {"type": "string", "description": "상세주소 (선택)", "default": ""}
                    },
                    "required": ["city_do", "gu_gun", "dong"]
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
                        "categories": {"type": "string", "description": "카테고리명 (예: 카페, 음식점 등)"},
                        "centerLon": {"type": "number", "description": "검색 중심 경도"},
                        "centerLat": {"type": "number", "description": "검색 중심 위도"},
                        "radius": {"type": "integer", "description": "검색 반경(1=1km)", "default": 1},
                        "count": {"type": "integer", "description": "검색 결과 개수", "default": 20}
                    },
                    "required": ["categories", "centerLon", "centerLat"]
                }
            }
        }

    # ========== Tool Call 실행기 & 테스트 ==========

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

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            headers = {"appKey": self.app_key}
            res = requests.get(
                f"{self.base_url}/tmap/pois?searchKeyword=테스트&count=1",
                headers=headers,
                timeout=10
            )
            return res.status_code == 200
        except Exception:
            return False

    def get_all_tool_schemas(self):
        """모든 tool 스키마 반환"""
        return [
            self.poi_search_tool(),
            self.car_route_tool(),
            self.walk_route_tool(),
            self.geocoding_tool(),
            self.category_search_tool(),
        ]


if __name__ == "__main__":
    import json
    
    print("=" * 60)
    print("TmapNavigation API 테스트")
    print("=" * 60)
    
    api = TmapNavigation()
    
    # 1. API 연결 테스트
    print("\n[1] API 연결 테스트")
    if api.test_connection():
        print("✓ API 연결 성공")
    else:
        print("✗ API 연결 실패")
        exit(1)
    
    # 2. POI 검색 테스트
    print("\n[2] POI 통합 검색 (POISearch_tmap)")
    try:
        result = api.execute_tool("POISearch_tmap", searchKeyword="강남역", count=5)
        if result.get("success") == False:
            print(f"✗ 오류: {result.get('error_message')}")
        else:
            print(f"검색어: 강남역")
            pois = result.get("searchPoiInfo", {}).get("pois", {}).get("poi", [])
            print(f"검색된 POI {len(pois)}개:")
            for i, poi in enumerate(pois[:3], 1):
                print(f"  {i}. {poi.get('name')}")
                print(f"     주소: {poi.get('upperAddrName')} {poi.get('middleAddrName')} {poi.get('lowerAddrName')}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 3. 주소 → 좌표 변환 테스트 (Geocoding)
    print("\n[3] 주소 → 좌표 변환 (Geocoding_tmap)")
    try:
        result = api.execute_tool("Geocoding_tmap", 
                                 city_do="서울특별시",
                                 gu_gun="강남구",
                                 dong="역삼동")
        if result.get("success") == False:
            print(f"✗ 오류: {result.get('error_message')}")
        else:
            coord_info = result.get("coordinateInfo", {})
            coordinate = coord_info.get("coordinate", [{}])[0]
            print(f"주소: 서울특별시 강남구 역삼동")
            print(f"위도: {coordinate.get('lat', coordinate.get('newLat'))}")
            print(f"경도: {coordinate.get('lon', coordinate.get('newLon'))}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 4. 보행자 경로 안내 테스트
    print("\n[4] 보행자 경로 안내 (WalkRoute_tmap)")
    try:
        # 강남역(127.0276, 37.4979) → 역삼역(127.0366, 37.5006)
        result = api.execute_tool("WalkRoute_tmap",
                                 startX=127.0276,
                                 startY=37.4979,
                                 endX=127.0366,
                                 endY=37.5006)
        if result.get("success") == False:
            print(f"✗ 오류: {result.get('error_message')}")
        else:
            features = result.get("features", [])
            if features:
                # 전체 경로 요약 정보 추출
                total_distance = 0
                total_time = 0
                for feature in features:
                    props = feature.get("properties", {})
                    total_distance += props.get("distance", 0)
                    total_time += props.get("time", 0)
                
                print(f"출발: (127.0276, 37.4979)")
                print(f"도착: (127.0366, 37.5006)")
                print(f"총 거리: {total_distance}m")
                print(f"예상 시간: {total_time // 60}분 {total_time % 60}초")
                print(f"경로 구간 수: {len(features)}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 5. 자동차 경로 안내 테스트
    print("\n[5] 자동차 경로 안내 (CarRoute_tmap)")
    try:
        # 강남역 → 역삼역
        result = api.execute_tool("CarRoute_tmap",
                                 startX=127.0276,
                                 startY=37.4979,
                                 endX=127.0366,
                                 endY=37.5006,
                                 searchOption=0)
        if result.get("success") == False:
            print(f"✗ 오류: {result.get('error_message')}")
        else:
            features = result.get("features", [])
            if features:
                # 전체 경로 요약 정보
                total_distance = 0
                total_time = 0
                for feature in features:
                    props = feature.get("properties", {})
                    total_distance += props.get("distance", 0)
                    total_time += props.get("time", 0)
                
                print(f"출발: (127.0276, 37.4979)")
                print(f"도착: (127.0366, 37.5006)")
                print(f"총 거리: {total_distance / 1000:.2f}km")
                print(f"예상 시간: {total_time // 60}분")
                print(f"경로 구간 수: {len(features)}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 6. 카테고리 검색 테스트
    print("\n[6] 카테고리 검색 (CategorySearch_tmap)")
    try:
        # 강남역 근처 카페 검색
        result = api.execute_tool("CategorySearch_tmap",
                                 categories="카페",
                                 centerLon=127.0276,
                                 centerLat=37.4979,
                                 radius=1,
                                 count=5)
        if result.get("success") == False:
            print(f"✗ 오류: {result.get('error_message')}")
        else:
            print(f"카테고리: 카페")
            print(f"중심 좌표: (127.0276, 37.4979)")
            print(f"검색 반경: 1km")
            pois = result.get("pois", [])
            print(f"검색된 장소 {len(pois)}개:")
            for i, poi in enumerate(pois[:3], 1):
                print(f"  {i}. {poi.get('name')}")
                print(f"     주소: {poi.get('address')}")
                print(f"     전화: {poi.get('telNo', '정보없음')}")
    except Exception as e:
        print(f"✗ 오류: {e}")
    
    # 7. Tool 스키마 출력
    print("\n[7] 사용 가능한 Tool 스키마")
    schemas = api.get_all_tool_schemas()
    for schema in schemas:
        print(f"  - {schema['function']['name']}: {schema['function']['description']}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
