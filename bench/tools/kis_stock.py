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

from configs.secrets import KIS_APP_KEY, KIS_APP_SECRET

class KISStock(BaseAPI):
    def __init__(self):
        super().__init__(
            name="kis_stock",
            description="한국투자증권 API - 국내외 주식 현재가, 종목 검색, 차트 데이터 조회"
        )
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = KIS_APP_KEY
        self.app_secret = KIS_APP_SECRET
        self._token = None

    # ========== 실제 API 호출 메서드들 ==========

    def _get_access_token(self) -> str:
        """OAuth 토큰 발급"""
        if self._token:
            return self._token

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        self._token = response.json()["access_token"]
        return self._token

    def _stock_price(self, symbol: str, market: str = "KOSPI") -> Dict[str, Any]:
        """국내 주식 현재가 조회"""
        token = self._get_access_token()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010100"
        }
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()["output"]
        return {
            "symbol": symbol,
            "market": market,
            "name": data.get("prdt_name"),
            "current_price": int(data.get("stck_prpr")),
            "change_rate": float(data.get("prdy_ctrt")),
            "volume": int(data.get("acml_vol"))
        }

    def _us_stock_price(self, symbol: str, exchange: str = "NASDAQ") -> Dict[str, Any]:
        """미국 주식 현재가 조회"""
        token = self._get_access_token()
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "HHDFS00000300"
        }
        params = {
            "AUTH": "",
            "EXCD": "NAS" if exchange == "NASDAQ" else "NYS",
            "SYMB": symbol
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()["output"]
        return {
            "symbol": symbol,
            "exchange": exchange,
            "current_price": float(data.get("last")),
            "change_rate": float(data.get("rate")),
            "volume": int(data.get("tvol"))
        }

    def _stock_chart(self, symbol: str, period: str = "D", count: int = 30,
                     start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """주식 차트 데이터 조회"""
        token = self._get_access_token()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-price"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST01010400"
        }

        period_code = {"D": "D", "W": "W", "M": "M", "Y": "Y"}
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_PERIOD_DIV_CODE": period_code.get(period, "D"),
            "FID_ORG_ADJ_PRC": "0"
        }

        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        chart_data = []
        for item in response.json()["output"][:count]:
            chart_data.append({
                "date": item.get("stck_bsop_date"),
                "open": int(item.get("stck_oprc")),
                "high": int(item.get("stck_hgpr")),
                "low": int(item.get("stck_lwpr")),
                "close": int(item.get("stck_clpr")),
                "volume": int(item.get("acml_vol"))
            })

        return {
            "symbol": symbol,
            "period": period,
            "data": chart_data
        }

    # ========== Tool Calling 스키마 메서드들 ==========

    def stock_price_tool(self) -> Dict:
        """StockPrice_kis tool 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "StockPrice_kis",
                "description": "한국투자증권 국내 주식 현재가 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "종목 코드 (예: 삼성전자 005930)"},
                        "market": {"type": "string", "enum": ["KOSPI", "KOSDAQ"], "default": "KOSPI"},
                    },
                    "required": ["symbol"]
                }
            }
        }

    def us_stock_price_tool(self) -> Dict:
        """USStockPrice_kis tool 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "USStockPrice_kis",
                "description": "한국투자증권 미국 주식 현재가 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "종목 심볼 (예: AAPL, TSLA)"},
                        "exchange": {"type": "string", "enum": ["NASDAQ", "NYSE"], "default": "NASDAQ", "description": "거래소"}
                    },
                    "required": ["symbol"]
                }
            }
        }

    def stock_chart_tool(self) -> Dict:
        """StockChart_kis tool 스키마"""
        return {
            "type": "function",
            "function": {
                "name": "StockChart_kis",
                "description": "한국투자증권 주식 차트 데이터 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "종목 코드 (예: 005930)"},
                        "period": {"type": "string", "enum": ["D", "W", "M"], "default": "D"},
                        "count": {"type": "integer", "default": 30, "description": "조회할 데이터 개수"},
                    },
                    "required": ["symbol"]
                }
            }
        }

    # ========== Tool Call 실행기 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "StockPrice_kis": self._stock_price,
            "USStockPrice_kis": self._us_stock_price,
            "StockChart_kis": self._stock_chart
        }

        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.stock_price_tool(),
            self.us_stock_price_tool(),
            self.stock_chart_tool()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트: 실패 시 이유를 콘솔에 출력합니다."""
        # 사전 점검: 환경 변수/시크릿 누락 여부
        missing = []
        if not self.app_key:
            missing.append("KIS_APP_KEY")
        if not self.app_secret:
            missing.append("KIS_APP_SECRET")
        if missing:
            print(
                f"[ERROR] 누락된 환경 변수: {', '.join(missing)}.\n"
                f"        configs/secrets.py 또는 환경 변수 설정을 확인하세요."
            )
            return False

        print("[INFO] KIS API 토큰 발급 시도...")
        try:
            token = self._get_access_token()
            if not token:
                print("[ERROR] 토큰 발급 실패: access_token이 비어 있습니다.")
                return False
            print("[INFO] KIS API 토큰 발급 성공")
            return True
        except requests.exceptions.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", "N/A")
            # 응답 본문을 가능한 한 축약하여 표시
            body_text = ""
            if getattr(e, "response", None) is not None:
                try:
                    body = e.response.json()
                    # 일반적으로 메시지 필드가 존재하면 우선 출력
                    if isinstance(body, dict):
                        msg = body.get("msg") or body.get("message") or body
                        body_text = str(msg)
                    else:
                        body_text = str(body)
                except Exception:
                    try:
                        body_text = (e.response.text or "").strip()
                    except Exception:
                        body_text = ""
            print(f"[ERROR] HTTPError: status={status}")
            if body_text:
                # 너무 긴 응답은 앞부분만 출력
                preview = body_text if len(body_text) < 500 else body_text[:500] + "..."
                print(f"[ERROR] Response: {preview}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 네트워크 오류: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] 예외 발생: {e.__class__.__name__}: {e}")
            return False
        
if __name__ == "__main__":
    api = KISStock()
    success = api.test_connection()
    print(f"KISStock API 연결 테스트: {'성공' if success else '실패'}")

    if success:
        print("\n[국내 주식 현재가 조회]")
        result = api._stock_price(symbol="005930")  # 삼성전자
        print(result)

        print("\n[미국 주식 현재가 조회]")
        result = api._us_stock_price(symbol="AAPL")  # 애플
        print(result)

        print("\n[국내 주식 차트 데이터 조회]")
        result = api._stock_chart(symbol="005930", period="D", count=5)  # 삼성전자 최근 5일
        print(result)