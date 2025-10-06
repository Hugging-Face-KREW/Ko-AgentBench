import os
from typing import Any, Dict, List

import requests
from .base_api import BaseAPI


class LSStock(BaseAPI):
    """
    LS증권 API를 사용하여 주식 정보를 조회하는 클래스.
    """
    def __init__(self):
        super().__init__(
            name="ls_stock",
            description="LS증권 API - 국내외 주식 현재가, 종목 검색, 차트 데이터 조회"
        )
        self.base_url = "https://openapi.ls-sec.co.kr:8080"
        self.app_key = os.getenv("LS_APP_KEY")  
        self.app_secret = os.getenv("LS_APP_SECRET")
        self.access_token = None

    def _get_access_token(self) -> bool:
        """
        API 액세스 토큰 발급
        발급시 self.access_token에 저장하고 True 반환
        """
        header = {"content-type": "application/x-www-form-urlencoded"}
        param = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecretkey": self.app_secret,
            "scope": "oob"
        }
        endpoint = "oauth2/token"
        URL = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(URL, verify=True, headers=header, params=param)
            response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
            self.access_token = response.json()["access_token"]
            return True
        except requests.exceptions.RequestException as e:
            print(f"액세스 토큰 발급 실패: {e}")
            return False

    # ========== 실제 API 호출 메서드들 ========== 

    def _stock_price(self, shcode: str, exchgubun: str = "K") -> Dict[str, Any]:
        """
        주식 현재가 조회 (t1102)

        Args:
            shcode (str): 종목 코드 (6자리)
            exchgubun (str): 거래소 구분 (K: KRX, N: NXT, U: 통합)
        Returns:
            해당 종목의 현재가 정보 (Dict)
        """
        endpoint = "/stock/market-data"
        URL = f"{self.base_url}/{endpoint}"
        header = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": "t1102",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body = {"t1102InBlock": {"shcode": shcode}}
        response = requests.post(URL, headers=header, json=body)
        return response.json()

    def _stock_search(self, query_index: str) -> Dict[str, Any]:
        """
        종목 검색 (내부 구현)
        """
        pass

    def _sector_stock(self, tmcode: str) -> Dict[str, Any]:
        """
        업종별 시세 조회 (테마 종목별 시세조회, t1537)

        Args:
            tmcode (str): 테마 코드
        Returns:
            해당 테마의 종목 정보 (Dict)
        """
        endpoint = "/stock/sector"
        URL = f"{self.base_url}/{endpoint}"
        header = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": "t1537",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body = {"t1537InBlock": {"tmcode": tmcode}}
        response = requests.post(URL, headers=header, json=body)
        return response.json()

    def _market_index(self, jisu: str = "KOSPI") -> Dict[str, Any]:
        """
        시장 지수 조회 (t1511)

        Args:
            jisu (str): 지수명 (KOSPI, KOSPI200, KRX100, KOSDAQ)
        Returns:
            해당 지수의 정보 (Dict)
        """
        endpoint = "/indtp/market-data"
        URL = f"{self.base_url}/{endpoint}"
        jisu_to_upcode = {
            "KOSPI": "001", "코스피": "001",
            "KOSPI200": "101", "코스피200": "101",
            "KRX100": "501",
            "KOSDAQ": "301", "코스닥": "301"
        }
        upcode = jisu_to_upcode.get(jisu, "001")  # Default to KOSPI
        header = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": "t1511",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body = {"t1511InBlock": {"upcode": upcode}}
        response = requests.post(URL, headers=header, json=body)
        return response.json()

    def _order_book(self, shcode: str) -> Dict[str, Any]:
        """
        주식 호가 조회 (t1101)

        Args:
            shcode (str): 종목 코드 (6자리)
        Returns:
            해당 종목의 호가 정보 (Dict)
        """
        endpoint = "/stock/market-data"
        URL = f"{self.base_url}/{endpoint}"
        header = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": "t1101",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body = {"t1101InBlock": {"shcode": shcode}}
        response = requests.post(URL, headers=header, json=body)
        return response.json()

    def _stock_trades(self, shcode: str, exchgubun: str = "N") -> Dict[str, Any]:
        """
        주식 시간대별 체결 내역 조회 (t8454)

        Args:
            shcode (str): 종목 코드
            exchgubun (str): 거래소 구분
        Returns:
            시간대별 체결 내역 (Dict)
        """
        endpoint = "/stock/market-data"
        URL = f"{self.base_url}/{endpoint}"
        header = {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.access_token}",
            "tr_cd": "t8454",
            "tr_cont": "N",
            "tr_cont_key": "",
        }
        body = {
            "t8454InBlock": {
                "shcode": shcode,
                "starttime": "",
                "endtime": "",
                "bun_term": "",
                "exchgubun": exchgubun
            }
        }
        response = requests.post(URL, headers=header, json=body)
        return response.json()

    # ========== Tool Calling 스키마 메서드들 ========== 

    def stock_price_tool(self) -> Dict:
        "StockPrice_ls tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "StockPrice_ls",
                "description": "주식 현재가 조회, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shcode": {
                            "type": "string",
                            "description": "주식 종목코드 (6자리, 예: 005930=삼성전자, 000660=SK하이닉스)",
                            "pattern": "^[0-9]{6}$"
                        },
                        "exchgubun": {
                            "type": "string",
                            "description": "거래소구분코드(K:KRX,N:NXT,U:통합)",
                            "enum": ["K", "N", "U"],
                            "default": "K"
                        }
                    },
                    "required": ["shcode"]
                }
            }
        }

    def stock_search_tool(self) -> Dict:
        "StockSearch_ls tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "StockSearch_ls",
                "description": "종목 검색, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_index": {
                            "type": "string",
                            "description": "검색할 종목명 또는 코드",
                        }
                    },
                    "required": ["query_index"]
                }
            }
        }

    def sector_stock_tool(self) -> Dict:
        "SectorStock_ls tool 스키마."
        return {
            "type": "function",
            "function": {
                "name": "SectorStock_ls",
                "description": "테마별 종목 시세 조회, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tmcode": {
                            "type": "string",
                            "description": "테마코드 (4자리)",
                            "pattern": "^[0-9]{4}$"
                        }
                    },
                    "required": ["tmcode"]
                }
            }
        }

    def market_index_tool(self) -> Dict:
        "MarketIndex_ls tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "MarketIndex_ls",
                "description": "시장 지수 조회, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "jisu": {
                            "type": "string",
                            "description": "조회할 지수 (KOSPI, KOSPI200, KRX100, KOSDAQ)",
                            "enum": ["KOSPI", "KOSPI200", "KRX100", "KOSDAQ"],
                            "default": "KOSPI"
                        }
                    },
                    "required": ["jisu"]
                }
            }
        }

    def order_book_tool(self) -> Dict:
        "OrderBook_ls tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "OrderBook_ls",
                "description": "개별 종목 호가 조회, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shcode": {
                            "type": "string",
                            "description": "주식 종목코드 (6자리, 예: 005930=삼성전자, 000660=SK하이닉스)",
                            "pattern": "^[0-9]{6}$"
                        }
                    },
                    "required": ["shcode"]
                }
            }
        }

    def stock_trades_tool(self) -> Dict:
        "StockTrades_ls tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "StockTrades_ls",
                "description": "주식 시간대별 체결 조회, LS증권 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "shcode": {
                            "type": "string",
                            "description": "주식 종목코드 (6자리, 예: 005930=삼성전자, 000660=SK하이닉스)",
                            "pattern": "^[0-9]{6}$"
                        },
                        "exchgubun": {
                            "type": "string",
                            "description": "거래소구분코드(K:KRX,N:NXT,U:통합)",
                            "enum": ["K", "N", "U"],
                            "default": "K"
                        }
                    },
                    "required": ["shcode"]
                }
            }
        }

    # ========== Tool Call 실행기 ========== 

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        tool_map = {
            "StockPrice_ls": self._stock_price,
            "StockSearch_ls": self._stock_search,
            "SectorStock_ls": self._sector_stock,
            "MarketIndex_ls": self._market_index,
            "OrderBook_ls": self._order_book,
            "StockTrades_ls": self._stock_trades
        }
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.stock_price_tool(),
            self.stock_search_tool(),
            self.sector_stock_tool(),
            self.market_index_tool(),
            self.order_book_tool(),
            self.stock_trades_tool()
        ]

    def test_connection(self) -> bool:
        """
        API 연결 상태를 테스트합니다. (현재는 항상 True를 반환)
        """
        # TODO: 실제 API 호출을 통해 연결 상태 확인 로직 추가
        return True

if __name__ == "__main__":
    api = LSStock()
    
    # 액세스 토큰 발급 테스트
    access_token = api._get_access_token()
    print(f"Access token: {api.access_token[:30]}")