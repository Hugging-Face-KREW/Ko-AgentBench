import requests
from typing import Dict, List, Optional
<<<<<<< HEAD

# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
    from .secrets import KTO_SERVICE_KEY
except ImportError:
    from base_api import BaseAPI
    from secrets import KTO_SERVICE_KEY
=======
from .base_api import BaseAPI
from .secrets import KTO_SERVICE_KEY
>>>>>>> e899c6aa717f6ad4a22e0f4f343ce676421236b0

class FestivalSearchAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="festival_search_api",
            description="한국관광공사 API를 활용한 행사/축제 정보 검색 도구"
        )
        self.service_key = KTO_SERVICE_KEY
        self.base_url = "http://apis.data.go.kr/B551011/KorService2/searchFestival2"
        self.location_mapping = {
            "서울": "1", "인천": "2", "대전": "3", "대구": "4", "광주": "5",
            "부산": "6", "울산": "7", "세종": "8", "경기": "31", "강원": "32",
            "충북": "33", "충남": "34", "경북": "35", "경남": "36",
            "전북": "37", "전남": "38", "제주": "39"
        }


    # ===== 실제 API 호출 메서드 (비즈니스 로직) =====
    """축제 검색 (내부 구현)"""
    def FestivalSearch_kto(self, eventStartDate: str = None,
                                 eventEndDate: Optional[str] = None,
                                 location: Optional[str] = None,
                                 num_of_rows: int = 10) -> Dict:
        url = self.base_url
        params = {
            "serviceKey": self.service_key,
            "num_of_rows": num_of_rows,        # 한페이지 결과 수
            "pageNo": 1,
            "_type": "json"
        }

        if not eventStartDate:
            return {"error": "eventStartDate는 필수입니다.", "status_code": 400}

        params["eventStartDate"] = eventStartDate

        if eventEndDate:
            params["eventEndDate"] = eventEndDate

        if location:
            area_code = self._extract_area_code(location)
            if area_code:
                params["areaCode"] = area_code

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            raw_data = response.json()
            return self._format_festival_response(raw_data)
        except Exception as e:
            return {
                "error": f"API 호출 실패: {str(e)}",
                "status": "error"
            }
        

    # area_code 매핑 함수 : 필요시 시군구코드(sigunguCode)매핑 로직 확장 예정
    def _extract_area_code(self, location: str) -> Optional[str]:
        location_lower = location.lower()
        for region, code in self.location_mapping.items():
            if region in location or region.lower() in location_lower:
                return code
        return None

    # response
    def _format_festival_response(self, raw_data: Dict) -> Dict:
        try:
            response_body = raw_data.get("response", {}).get("body", {})
            items = response_body.get("items", {}).get("item", [])
            total_count = response_body.get("totalCount", 0)

            if isinstance(items, dict):
                items = [items]

            formatted_items = []
            for item in items:
                formatted_items.append({
                    "title": item.get("title", "").strip(),
                    "addr": item.get("addr1", ""),
                    "eventStartDate": item.get("eventstartdate"),
                    "eventEndDate": item.get("eventenddate")
                })

            return {
                "totalCount": total_count,
                "items": formatted_items,
                "status": "success"
            }

        except Exception as e:
            return {
                "error": f"응답 데이터 파싱 오류: {str(e)}",
                "raw_data": raw_data,
                "status": "error"
            }


    # ===== Tool Calling용 스키마 정의 =====

    def festival_search_tool(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "FestivalSearch_kto",
                "description": "행사/축제 정보를 날짜와 지역으로 검색",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "eventStartDate": {
                            "type": "string",
                            "pattern": "^[0-9]{8}$",
                            "description": "행사 시작일 (YYYYMMDD)"
                        },
                        "eventEndDate": {
                            "type": "string",
                            "pattern": "^[0-9]{8}$",
                            "description": "행사 종료일 (YYYYMMDD)"
                        },
                        "location": {
                            "type": "string",
                            "description": "주소나 지역명 (예: '서울 강남')"
                        },
                        "num_of_rows": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 10,
                            "description": "검색 결과 개수"
                        }
                    },
                    # 필수 조건 정의
                    "required": ["eventStartDate"],
                }
            }
        }

    # ======================= Tool Call 실행기 ========================

    def execute_tool(self, tool_name: str, **kwargs) -> Dict:
        if tool_name in ["FestivalSearch_kto", "festival_search"]:
            return self.FestivalSearch_kto(**kwargs)
        else:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

    def get_all_tool_schemas(self) -> List[Dict]:
        return [self.festival_search_tool()]



    # ========== 실제 API 연결 테스트 ==========
    def test_connection(self) -> bool:
        url = self.base_url
        params = {
            "serviceKey": self.service_key,
            "numOfRows": 1,
            "pageNo": 1,
            "MobileOS": "ETC",
            "MobileApp": "TestApp",
            "_type": "json",
            "eventStartDate": "20250101"   # 필수 파라미터 테스트용
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                res_json = response.json()
                result_code = res_json.get("response", {}).get("header", {}).get("resultCode")
                if result_code == "0000":
                    total_count = res_json.get("response", {}).get("body", {}).get("totalCount", 0)
                    items = res_json.get("response", {}).get("body", {}).get("items", {}).get("item", [])
                    first_title = items[0].get("title", "제목 없음") if items else "데이터 없음"

                    print(f"[INFO] 전체 건수: {total_count}") #test용 출력
                    print(f"[INFO] 첫 축제 제목: {first_title}") #test용 출력
                    return True
            return False
        except Exception as e:
            print(f"[ERROR] 예외 발생: {e}")
            return False
        


# ========== 연결 테스트 ==========
if __name__ == "__main__":
    # 인스턴스 생성
    api = FestivalSearchAPI(service_key=KTO_SERVICE_KEY)

    # 연결 테스트
    success = api.test_connection()
    print("연결 성공!" if success else "❌ 연결 실패")


# 테스트 결과
"""[INFO] 전체 건수: 1029
[INFO] 첫 축제 제목: 가락몰 빵축제 전국빵지자랑
연결 성공! """