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

    # ========== 실제 API 호출 메서드들 ==========

    def _stock_price(self, symbol: str, market: str = "KOSPI") -> Dict[str, Any]:
        """국내 주식 현재가 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _stock_search(self, keyword: str, market: str = "ALL") -> Dict[str, Any]:
        """종목 검색 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _us_stock_price(self, symbol: str, exchange: str = "NASDAQ") -> Dict[str, Any]:
        """미국 주식 현재가 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    def _stock_chart(self, symbol: str, period: str = "D", count: int = 30,
                     start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """주식 차트 데이터 조회 (내부 구현)"""
        # TODO: 실제 API 호출 로직 구현
        pass

    # ========== Tool Calling 스키마 메서드들 ==========

    def stock_price_tool(self) -> Dict:
        """StockPrice_kis tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "StockPrice_kis",
                "description": "한국 주식의 현재가, 등락률, 거래량 정보를 조회합니다. 한국투자증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "주식 종목코드 (6자리, 예: 005930=삼성전자, 000660=SK하이닉스)",
                            "pattern": "^[0-9]{6}$"
                        },
                        "market": {
                            "type": "string",
                            "description": "시장구분",
                            "enum": ["KOSPI", "KOSDAQ"],
                            "default": "KOSPI"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }

    def us_stock_price_tool(self) -> Dict:
        """USStockPrice_kis tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "USStockPrice_kis",
                "description": "미국 주식의 현재가와 기본 정보를 조회합니다. NASDAQ, NYSE 등 주요 거래소를 지원합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "미국 주식 심볼 (예: AAPL, MSFT, TSLA, GOOGL, BRK.B)",
                            "pattern": "^[A-Z.-]{1,10}$"
                        },
                        "exchange": {
                            "type": "string",
                            "description": "거래소",
                            "enum": ["NASDAQ", "NYSE", "AMEX"],
                            "default": "NASDAQ"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }

    def stock_chart_tool(self) -> Dict:
        """StockChart_kis tool schema"""
        return {
            "type": "function",
            "function": {
                "name": "StockChart_kis",
                "description": "한국 주식의 일봉, 주봉, 월봉 차트 데이터를 조회합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "주식 종목코드 (6자리)",
                            "pattern": "^[0-9]{6}$"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["D", "W", "M"],
                            "default": "D",
                            "description": "조회 기간 (D=일봉, W=주봉, M=월봉)"
                        },
                        "count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 30,
                            "description": "조회할 데이터 개수"
                        },
                        "start_date": {
                            "type": "string",
                            "description": "시작일자 (YYYYMMDD)",
                            "pattern": "^[0-9]{8}$"
                        },
                        "end_date": {
                            "type": "string",
                            "description": "종료일자 (YYYYMMDD)",
                            "pattern": "^[0-9]{8}$"
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
            "StockPrice_kis": self._stock_price,
            "StockSearch_kis": self._stock_search,
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
            self.stock_search_tool(),
            self.us_stock_price_tool(),
            self.stock_chart_tool()
        ]

    def test_connection(self) -> bool:
        """API 연결 테스트"""
        # TODO: OAuth 토큰 발급 테스트 구현
        return True