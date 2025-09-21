import requests
import os
from typing import Dict, List, Any, Optional
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
        """암호화폐 현재가 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _market_list(self, quote: str = "KRW", include_event: bool = True) -> Dict[str, Any]:
        """마켓 목록 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _crypto_candle(self, symbol: str, quote: str = "KRW",
                       candle_type: str = "days", unit: Optional[int] = None,
                       count: int = 30, to: Optional[str] = None) -> Dict[str, Any]:
        """캔들 데이터 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    # ========== Tool Calling 스키마 메서드들 ==========

    def crypto_price_tool(self) -> Dict:
        """CryptoPrice_upbit tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CryptoPrice_upbit",
                "description": "업비트 거래소에서 암호화폐의 실시간 시세 정보를 조회합니다. (Rate Limit: 초당 10회, IP 기준)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "암호화폐 심볼 (예: BTC, ETH, XRP, ADA, SOL, DOGE, AVAX, DOT, LINK, UNI)",
                            "pattern": "^[A-Z0-9]+$"
                        },
                        "quote": {
                            "type": "string",
                            "enum": ["KRW", "BTC", "USDT"],
                            "default": "KRW",
                            "description": "기준 통화"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }

    def market_list_tool(self) -> Dict:
        """MarketList_upbit tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "MarketList_upbit",
                "description": "업비트에서 거래 가능한 암호화폐 마켓 목록을 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "quote": {
                            "type": "string",
                            "enum": ["KRW", "BTC", "USDT", "ALL"],
                            "default": "KRW",
                            "description": "기준 통화 필터"
                        },
                        "include_event": {
                            "type": "boolean",
                            "default": True,
                            "description": "시장 경고/주의 정보 포함 여부"
                        }
                    },
                    "required": []
                }
            }
        }

    def crypto_candle_tool(self) -> Dict:
        """CryptoCandle_upbit tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "CryptoCandle_upbit",
                "description": "업비트에서 암호화폐의 캔들(OHLCV) 데이터를 조회합니다. 차트 분석과 기술적 분석에 활용됩니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "암호화폐 심볼 (예: BTC, ETH)",
                            "pattern": "^[A-Z0-9]+$"
                        },
                        "quote": {
                            "type": "string",
                            "enum": ["KRW", "BTC", "USDT"],
                            "default": "KRW",
                            "description": "기준 통화"
                        },
                        "candle_type": {
                            "type": "string",
                            "enum": ["minutes", "days", "weeks", "months"],
                            "default": "days",
                            "description": "캔들 타입"
                        },
                        "unit": {
                            "type": "integer",
                            "enum": [1, 3, 5, 10, 15, 30, 60, 240],
                            "description": "candle_type이 minutes일 때만 필요한 분 단위"
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 30,
                            "description": "조회할 캔들 개수"
                        },
                        "to": {
                            "type": "string",
                            "description": "마지막 캔들 시점 (ISO8601 형식)"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }

    # ========== Tool Call 실행기 ==========

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