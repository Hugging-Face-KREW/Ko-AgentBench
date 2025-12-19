from typing import Any, Dict, List, Optional

import requests

# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
except ImportError:
    from base_api import BaseAPI

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
                       candle_type: str = "days", unit: Optional[int | str] = None,
                       count: int = 30, to: Optional[str] = None) -> Dict[str, Any]:
        """캔들 데이터 조회"""
        market = f"{quote}-{symbol}"

        if candle_type == "minutes":
            try:
                unit = int(unit) if unit is not None else 1
            except (TypeError, ValueError):
                return {"error": "unit must be one of 1,3,5,10,15,30,60,240"}
            valid_units = {1, 3, 5, 10, 15, 30, 60, 240}
            if unit not in valid_units:
                return {"error": "unit must be one of 1,3,5,10,15,30,60,240"}
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

    def crypto_price_tool(self) -> dict:
        "CryptoPrice_upbit tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "CryptoPrice_upbit",
                "description": "업비트 암호화폐 현재가 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "암호화폐 심볼 (예: BTC, ETH)"},
                        "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT"], "default": "KRW", "description": "기준 통화"}
                    },
                    "required": ["symbol"]
                }
            }
        }

    def market_list_tool(self) -> dict:
        "MarketList_upbit tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "MarketList_upbit",
                "description": "업비트 마켓 목록 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT", "ALL"], "default": "KRW", "description": "기준 통화"},
                        "include_event": {"type": "boolean", "default": True, "description": "이벤트 마켓 포함 여부"}
                    },
                    "required": []
                }
            }
        }

    def crypto_candle_tool(self) -> dict:
        "CryptoCandle_upbit tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "CryptoCandle_upbit",
                "description": "업비트 암호화폐 캔들 데이터 조회",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "암호화폐 심볼 (예: BTC, ETH)"},
                        "quote": {"type": "string", "enum": ["KRW", "BTC", "USDT"], "default": "KRW", "description": "기준 통화"},
                        "candle_type": {"type": "string", "enum": ["minutes", "days", "weeks", "months"], "default": "days", "description": "캔들 타입"},
                        "unit": {"type": "integer", "enum": [1, 3, 5, 10, 15, 30, 60, 240], "description": "분 단위 (candle_type이 minutes일 때만 필요)"},
                        "count": {"type": "integer", "minimum": 1, "maximum": 200, "default": 30, "description": "조회할 캔들 개수"},
                        "to": {"type": "string", "description": "마지막 캔들 시각 (YYYY-MM-DD HH:mm:ss)"}
                    },
                    "required": ["symbol"]
                }
            }
        }

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
