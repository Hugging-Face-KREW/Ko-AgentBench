import requests
from typing import Dict, List, Optional
from .base_api import BaseAPI
from .secrets import Directions_Client_ID, Directions_Client_Secret

class NaverMapsAPI(BaseAPI):
    def __init__(self):
        super().__init__(
            name="naver_maps_api",
            description="Naver Maps API를 활용한 경로 탐색 도구"
        )
        self.client_id = Directions_Client_ID
        self.secret_key = Directions_Client_Secret
        self.base_url = "https://maps.apigw.ntruss.com"

    # ===== 실제 API 호출 메서드 (비즈니스 로직) =====
    def Directions_naver(
        self,
        start: str,
        goal: str,
        waypoints: Optional[List[str]] = None,
        option: Optional[List[str]] = None,
    ) -> Dict:

        if not start or not goal:
            return {"error": "start와 goal은 필수입니다.", "status_code": 400}

        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.secret_key,
        }

        url = self.base_url + "/map-direction/v1/driving"
        params = {
            "start": start,
            "goal": goal,
            "waypoints": "|".join(waypoints) if waypoints else None,
            "option": ":".join(option) if option else None
        }

        try:
            response = requests.get(url, headers=headers, params={k:v for k,v in params.items() if v}, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "error"}
        

    # ===== Tool Calling용 스키마 정의 =====
    def directions_tool(self) -> Dict:
        return {
            "type": "function",
            "function": {
                "name": "Directions_naver",
                "description": "입력 정보(출발지, 경유지, 목적지 등)를 기반으로 자동차 경로 및 통행 정보(소요 시간, 거리, 예상 유류비, 통행 요금 정보, 분기점 안내) 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start": {
                            "type": "string",
                            "description": "출발지(경도,위도) (예: 127.12345,37.12345)"
                        },
                        "goal": {
                            "type": "string",
                            "description": "도착지 좌표 문자열. 여러 좌표는 ':'로 구분 (예: '123.45678,34.56789:124.56789,35.67890')"
                        },
                        "waypoints": {
                            "type": "string",
                            "pattern": "^([0-9.]+,[0-9.]+(:[0-9.]+,[0-9.]+)?)(\\|[0-9.]+,[0-9.]+(:[0-9.]+,[0-9.]+)?){0,4}$",
                            "description": "경유지 좌표 문자열. '|'로 경유지를 구분하고, 각 경유지는 '경도,위도' 또는 '경도,위도:경도,위도' 형식 (최대 5개)"
                        },
                        "option": {
                            "type": "string",
                            "enum": [
                                "trafast",
                                "tracomfort",
                                "traoptimal",
                                "traavoidtoll",
                                "traavoidcaronly",
                                "trafast:traavoidtoll"
                            ],
                            "description": "경로 조회 옵션 (':'로 최대 3개까지 조합 가능), 예: 'trafast:traavoidtoll'"
                        },
                    },
                    "required": ["start", "goal"]
                }
            }
        }


    # =================== Tool 호출 실행기 ===================
    def execute_tool(self, tool_name: str, **kwargs) -> dict:
        tool_map = {
            "Directions_naver": self.Directions_naver,
            # 필요 시 추가
        }
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[dict]:
        return [
            self.directions_tool(),
        ]

    # ========== 실제 API 연결 테스트 ==========

    def test_connection(self) -> bool:
        url = self.base_url + "/map-direction/v1/driving"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.secret_key,
        }
        params = {
            "start": "127.1058342,37.359708",   # 출발지 (경도,위도) - 분당
            "goal": "127.075986,37.517992",     # 도착지 (경도,위도) - 서울 근처
            "option": "traoptimal",             # 최적 경로 옵션
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            print(f"[Status] {response.status_code}")
            if response.status_code == 200:
                res_json = response.json()
                print("[응답 예시]")
                print(res_json)  # 전체 응답 출력
                route = res_json.get("route", {}).get("traoptimal", [])
                if route:
                    summary = route[0].get("summary", {})
                    print(f"총 거리: {summary.get('distance')}m")
                    print(f"예상 시간: {summary.get('duration')}초")
                    return True
                else:
                    print("⚠️ 경로 결과 없음")
            else:
                print(f"[ERROR] {response.text}")
            return False
        except Exception as e:
            print(f"[Exception] {e}")
            return False


# ========== 연결 테스트 ==========
if __name__ == "__main__":
    api = NaverMapsAPI()
    if api.test_connection():
        print("✅ API 연결 성공")
    else:
        print("❌ API 연결 실패")


# 테스트 결과
"""[Status] 200
[응답 예시]
{'code': 0, 'message': '길찾기를 성공하였습니다.', 'currentDateTime': '2025-09-23T10:56:48', 'route': {'traoptimal': [{'guide': [{'distance': 698, 'duration': 192578, 'instructions': "정자일로3사거리에서 '분당수서로' 방면으로 우회전", 'pointIndex': 35, 'type': 3}, {'distance': 139, 'duration': 28076, 'instructions': "'서울, 판교IC' 방면으로 우회전", 'pointIndex': 39, 'type': 3}, ... """