import requests
import os
from typing import Dict, List, Any
from base_api import BaseAPI

class KISStock(BaseAPI):
    def __init__(self):
        super().__init__(
            name="kis_stock",
            description="한국투자증권 API - 국내외 주식 현재가, 종목 검색, 차트 데이터 조회"
        )
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.app_key = os.getenv("KIS_APP_KEY")
        self.app_secret = os.getenv("KIS_APP_SECRET")
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
        """API 연결 테스트"""
        try:
            self._get_access_token()
            return True
        except:
            return False