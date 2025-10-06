import requests
import os
from typing import Dict, List, Any, Optional
from .base_api import BaseAPI

class UpbitCrypto(BaseAPI):
    def __init__(self):
        super().__init__(
            name="upbit_crypto",
            description="업비트 암호화폐 거래소 API - 현재가, 마켓 목록, 캔들 데이터 조회"
        )
        self.base_url = "https://api.upbit.com"

    # ========== 실제 API 호출 메서드들 ==========

    def _crypto_price(self, symbol: str, quote: str = "KRW") -> Dict[str, Any]:
        """암호화폐 현재가 조회"""
        market = f"{quote}-{symbol}"
        url = f"{self.base_url}/v1/ticker"
        params = {"markets": market}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()[0]
        return {
            "symbol": symbol,
            "quote": quote,
            "market": market,
            "current_price": data.get("trade_price"),
            "change_rate": data.get("change_rate") * 100,
            "volume": data.get("acc_trade_volume_24h"),
            "high_price": data.get("high_price"),
            "low_price": data.get("low_price")
        }

    def _market_list(self, quote: str = "KRW", include_event: bool = True) -> Dict[str, Any]:
        """마켓 목록 조회"""
        url = f"{self.base_url}/v1/market/all"
        params = {"isDetails": "true"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        markets = []
        for market in response.json():
            if quote != "ALL" and not market["market"].startswith(quote):
                continue
            markets.append({
                "market": market.get("market"),
                "korean_name": market.get("korean_name"),
                "english_name": market.get("english_name")
            })

        return {
            "quote": quote,
            "markets": markets,
            "count": len(markets)
        }

    def _crypto_candle(self, symbol: str, quote: str = "KRW",
                       candle_type: str = "days", unit: Optional[int] = None,
                       count: int = 30, to: Optional[str] = None) -> Dict[str, Any]:
        """캔들 데이터 조회"""
        market = f"{quote}-{symbol}"

        if candle_type == "minutes":
            url = f"{self.base_url}/v1/candles/minutes/{unit}"
        else:
            url = f"{self.base_url}/v1/candles/{candle_type}"

        params = {"market": market, "count": min(count, 200)}
        if to:
            params["to"] = to

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        candles = []
        for candle in response.json():
            candles.append({
                "timestamp": candle.get("candle_date_time_kst"),
                "open": candle.get("opening_price"),
                "high": candle.get("high_price"),
                "low": candle.get("low_price"),
                "close": candle.get("trade_price"),
                "volume": candle.get("candle_acc_trade_volume")
            })

        return {
            "symbol": symbol,
            "market": market,
            "candle_type": candle_type,
            "data": candles
        }

    # ========== Tool Calling 스키마 메서드들 ==========

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Tool call 실행"""
        tool_map = {
            "CryptoPrice_upbit": self._crypto_price,
            "MarketList_upbit": self._market_list,
            "CryptoCandle_upbit": self._crypto_candle
        }

        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")

        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.crypto_price_tool(),
            self.market_list_tool(),
            self.crypto_candle_tool()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/v1/market/all", timeout=10)
            return response.status_code == 200
        except:
            return False