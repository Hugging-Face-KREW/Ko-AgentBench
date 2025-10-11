
import hashlib
import time
import uuid
from typing import Any, Dict, List
from urllib.parse import urlencode

import jwt
import requests

<<<<<<< HEAD
# 상대 임포트와 절대 임포트 모두 지원
try:
    from .base_api import BaseAPI
except ImportError:
    from base_api import BaseAPI
=======
from .base_api import BaseAPI
>>>>>>> e899c6aa717f6ad4a22e0f4f343ce676421236b0


class BithumbStock(BaseAPI):
    """
    빗썸 API를 사용하여 주식 정보를 조회하는 클래스.
    """
    def __init__(self):
        super().__init__(
            name="bithumb_stock",
            description="빗썸 API - 국내외 주식 현재가, 종목 검색, 차트 데이터 조회"
        )
        self.base_url = "https://api.bithumb.com/v1"
        # 빗썸은 API요청시 API Key 발급 필요 없음
        # self.access_key = BITHUMB_ACCESS_KEY
        # self.secret_key = BITHUMB_SECRET_KEY
        self.access_token = None   # jwt token

    def _get_access_token(self) -> bool:
        """
        jwt 액세스 토큰 발급
        발급시 self.access_token에 저장하고 True 반환
        """
        
        try:
            query = urlencode(dict( string="abc", number=123 )).encode()

            hash = hashlib.sha512()
            hash.update(query)
            query_hash = hash.hexdigest()

            payload = {
                'access_key': self.access_key,
                'nonce': str(uuid.uuid4()),
                'timestamp': round(time.time() * 1000), 
                'query_hash': query_hash,
                'query_hash_alg': 'SHA512',
            }

            print(payload, "\n")
                
            jwt_token = jwt.encode(payload, self.secret_key)
            self.access_token = 'Bearer {}'.format(jwt_token)

            print(self.access_token)

            return True
        except requests.exceptions.RequestException as e:
            print(f"액세스 토큰 발급 실패: {e}")
            return False

    # ========== 실제 API 호출 메서드들 ========== 

    def _cryptoPrice_bithumb(self, markets: str = "KRW-BTC") -> Dict[str, Any]:
        """
        현재가 조회

        Args:
            markets (str): 반점으로 구분되는 마켓 코드 (ex. KRW-BTC, BTC-ETH)
        Returns:
            현재가 정보 (Dict)
        """
        endpoint = f"ticker?markets={markets}"
        URL = f"{self.base_url}/{endpoint}"
        headers = {"accept": "application/json"}
        response = requests.get(URL, headers=headers)
        return response.json()

    def _orderBook_bithumb(self, markets: str = "KRW-BTC") -> Dict[str, Any]:
        """
        거래소 호가 정보 조회

        Args:
            markets (str): 마켓 코드 목록 (ex. KRW-BTC,BTC-ETH)

        Returns:
            거래소 호가 정보 (Dict)
        """
        endpoint = f"orderbook?markets={markets}"
        URL = f"{self.base_url}/{endpoint}"
        headers = {"accept": "application/json"}
        response = requests.get(URL, headers=headers)
        return response.json()

    def _marketList_bithumb(self, isDetails: bool = False) -> Dict[str, Any]:
        """
        빗썸에서 거래 가능한 마켓 목록을 조회합니다.

        Args:
            isDetails (bool) : 유의종목 필드과 같은 상세 정보 노출 여부(선택 파라미터)

        Returns:
            거래 가능한 마켓과 가상자산 정보 (Dict)
        """
        endpoint = f"market/all?isDetails={isDetails}"
        URL = f"{self.base_url}/{endpoint}"
        headers = {"accept": "application/json"}
        response = requests.get(URL, headers=headers)
        return response.json()


    def _cryptoCandle_bithumb(self, time: str, count: int = 1, to: str = None, market: str = "KRW-BTC", unit: int = 1) -> Dict[str, Any]:
        """
        시간 및 구간 별 빗썸 거래소 가상자산 가격, 거래량 정보 제공

        Args:
            time (str) : 캔들 단위
            market (str) : 마켓 코드 (ex. KRW-BTC)
            to (str) : 마지막 캔들 시각 (exclusive). 비워서 요청시 가장 최근 캔들,  (yyyy-MM-dd'T'HH:mm:ss'Z' or yyyy-MM-dd HH:mm:ss)
            count (int) :  캔들 개수(최대 200개까지 요청 가능) 
            unit (int) : 분 단위. 가능한 값 : 1, 3, 5, 10, 15, 30, 60, 240
        Returns:
            캔들 스틱 정보 (Dict)
        """

        # 입력값 검증
        valid_time = ["minutes", "days", "weeks", "months"]
        if time not in valid_time:
            return {"error": f"Invalid time. Must be one of {valid_time}"}
        elif time == "minutes":
            valid_units = [1, 3, 5, 10, 15, 30, 60, 240]
            if unit not in valid_units:
                return {"error": f"Invalid unit. Must be one of {valid_units}"}
            endpoint = f"candles/{time}/{unit}"
        else:
            endpoint = f"candles/{time}"
            
        if count <= 0 or count > 200:
            return {"error": "Count must be between 1 and 200"}
        
        if not market or not isinstance(market, str):
            return {"error": "Market must be a valid string"}
        
        # URL 파라미터 구성
        params = {
            "market": market,
            "count": min(count, 200),  # 최대값 보장
        }
        if to:
            params["to"] = to
        
        # 엔드포인트 구성 (쿼리 파라미터는 requests가 자동 처리)
        URL = f"{self.base_url}/{endpoint}"

        headers = {"accept": "application/json"}
        response = requests.get(URL, headers=headers, params=params)
        return response.json()


        


    # ========== Tool Calling 스키마 메서드들 ========== 

    def cryptoPrice_bithumb_tool(self) -> Dict:
        "cryptoPrice_bithumb tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "CryptoPrice_bithumb",
                "description": "현재가 정보, 빗썸 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "markets": {
                            "type": "string",
                            "description": "반점으로 구분되는 마켓 코드 (ex. KRW-BTC, BTC-ETH)",
                            "pattern": "^[A-Z0-9]+$",
                            "defaults" : "KRW-BTC",
                            "enum" : [ "KRW-BTC", "BTC-ETH"]
                        }
                    
                    },
                    "required": ["markets"]
                }
            }
        }

    def orderBook_bithumb_tool(self) -> Dict:
        "OrderBook_bithumb tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "OrderBook_bithumb",
                "description": "호가 정보 조회, 빗썸 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                         "markets": {
                            "type": "string",
                            "description": "반점으로 구분되는 마켓 코드 (ex. KRW-BTC, BTC-ETH)",
                            "pattern": "^[A-Z0-9]+$",
                            "defaults" : "KRW-BTC",
                            "enum" : [ "KRW-BTC", "BTC-ETH"]
                        }
                    
                    },
                    "required": ["markets"]
                }
            }
        }

    def marketList_bithumb_tool(self) -> Dict:
        "MarketList_bithumb tool 스키마."
        return {
            "type": "function",
            "function": {
                "name": "MarketList_bithumb",
                "description": "거래 가능한 마켓과 가상자산 정보 조회, 빗썸 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "isDetails": {
                            "type": "boolean",
                            "description": "유의종목 필드과 같은 상세 정보 노출 여부(선택 파라미터)",
                            "default": False 
                        }
                    }
                }
            }
        }

    def cryptoCandle_bithumb_tool(self) -> Dict:
        "CryptoCandle_bithumb tool 스키마"
        return {
            "type": "function",
            "function": {
                "name": "CryptoCandle_bithumb",
                "description": "캔들스틱 조회, 빗썸 Open API를 활용합니다.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "integer",
                            "description": "캔들 단위. 가능한 값 : minutes, days, weeks, months",
                            "enum": ["minutes", "days", "weeks", "months"]
                        },
                        "unit": {
                            "type": "integer",
                            "description": "분 단위. 가능한 값 : 1, 3, 5, 10, 15, 30, 60, 240",
                            "enum": [1, 3, 5, 10, 15, 30, 60, 240]
                        },
                        "market": {
                            "type": "string",
                            "description": "마켓 코드 (ex. KRW-BTC, BTC-ETH)",
                            "pattern": "^[A-Z0-9]+$",
                            "default": "KRW-BTC"
                        },
                        "to": {
                            "type": "string",
                            "description": "마지막 캔들 시각 (yyyy-MM-dd'T'HH:mm:ss'Z' or yyyy-MM-dd HH:mm:ss). 기본적으로 KST 기준 시간이며 비워서 요청시 가장 최근 캔들",
                            "default": "1"
                        },
                        "count": {
                            "type": "integer",
                            "description": "캔들개수",
                            "maximum": 200
                        }
                    },
                    "required": ["time"]
                }
            }
        }


    # ========== Tool Call 실행기 ========== 

    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        tool_map = {
            "CryptoPrice_bithumb": self._cryptoPrice_bithumb,
            "OrderBook_bithumb": self._orderBook_bithumb,
            "MarketList_bithumb": self._marketList_bithumb,
            "CryptoCandle_bithumb": self._cryptoCandle_bithumb
        }
        if tool_name not in tool_map:
            raise ValueError(f"지원하지 않는 tool: {tool_name}")
        return tool_map[tool_name](**kwargs)

    def get_all_tool_schemas(self) -> List[Dict]:
        """모든 tool 스키마 반환"""
        return [
            self.cryptoPrice_bithumb_tool(),
            self.orderBook_bithumb_tool(),
            self.marketList_bithumb_tool(),
            self.cryptoCandle_bithumb_tool()
        ]

    def test_connection(self) -> bool:
        """
        API 연결 상태를 테스트합니다. (마켓 목록 조회 및 주요 API 호출 테스트)
        """
        try:
            # 1. 마켓 목록 조회 테스트 (기본 연결 확인)
            market_list_response = self._marketList_bithumb()
            print(f"Debug: market_list_response type: {type(market_list_response)}, content: {market_list_response}")
            # _marketList_bithumb은 리스트를 반환하므로, 리스트가 비어있지 않은지 확인합니다.
            if not (isinstance(market_list_response, list) and len(market_list_response) > 0):
                print(f"Bithumb API connection test (market list) failed. Response: {market_list_response}")
                return False
            print("Bithumb API connection (market list) successful.")

            # 2. 현재가 조회 테스트
            crypto_price_response = self._cryptoPrice_bithumb(markets="KRW-BTC")
            print(f"Debug: crypto_price_response type: {type(crypto_price_response)}, content: {crypto_price_response}")
            # _cryptoPrice_bithumb도 리스트를 반환하므로, 리스트가 비어있지 않은지 확인합니다.
            if not (isinstance(crypto_price_response, list) and len(crypto_price_response) > 0):
                print(f"Bithumb API connection test (crypto price) failed. Response: {crypto_price_response}")
                return False
            print("Bithumb API connection (crypto price) successful.")

            # 3. 호가 정보 조회 테스트
            order_book_response = self._orderBook_bithumb(markets="KRW-BTC")
            print(f"Debug: order_book_response type: {type(order_book_response)}, content: {order_book_response}")
            # _orderBook_bithumb도 리스트를 반환하므로, 리스트가 비어있지 않은지 확인합니다.
            if not (isinstance(order_book_response, list) and len(order_book_response) > 0):
                print(f"Bithumb API connection test (order book) failed. Response: {order_book_response}")
                return False
            print("Bithumb API connection (order book) successful.")

            # 4. 캔들스틱 조회 테스트 (예시: 1분봉, KRW-BTC)
            crypto_candle_response = self._cryptoCandle_bithumb(time="minutes", unit=1, market="KRW-BTC", count=1)
            print(f"Debug: crypto_candle_response type: {type(crypto_candle_response)}, content: {crypto_candle_response}")
            # _cryptoCandle_bithumb도 리스트를 반환하므로, 리스트가 비어있지 않은지 확인합니다.
            if not (isinstance(crypto_candle_response, list) and len(crypto_candle_response) > 0):
                print(f"Bithumb API connection test (crypto candle) failed. Response: {crypto_candle_response}")
                return False
            print("Bithumb API connection (crypto candle) successful.")

            print("All Bithumb API connection tests passed.")
            return True
        except Exception as e:
            print(f"An error occurred during the Bithumb API connection test: {e}")
            return False

if __name__ == "__main__":
    api = BithumbStock()
