# BE/trading_core/data_fetcher/adapters/kucoin.py
"""
KuCoin adapter
──────────────
Lightweight HTTP client for KuCoin REST API endpoints:

- /api/v1/symbols                    → Get list of available symbols
- /api/v1/market/orderbook/level1    → Get best bid/ask prices  
- /api/v1/market/stats               → Get 24hr stats for symbol
- /api/v1/market/candles             → Get kline data
- /api/v1/market/histories           → Get trade histories
- /api/v1/currencies                 → Get currency list
- /api/v1/timestamp                  → Get server timestamp

Design goals
------------
• No hardcoded symbols. Callers specify which symbols to request.
• Conservative timeouts + retries with exponential backoff.
• Normalized, minimal return payloads for consistent upstream usage.
• Works without API key for public market data endpoints.
• Optional API key support for private account data and trading.
• Handles KuCoin's specific response format and pagination.

Usage
-----
from trading_core.data_fetcher.adapters.kucoin import KuCoinAdapter

kucoin = KuCoinAdapter()
symbols = kucoin.get_symbols()  # All trading symbols
ticker = kucoin.get_ticker("BTC-USDT")  # Single symbol stats
candles = kucoin.get_candles("ETH-USDT", type="1day")  # Daily candles
orderbook = kucoin.get_orderbook("BTC-USDT")  # Level 1 order book
"""

from __future__ import annotations

import time
import base64
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests


DEFAULT_BASE_URL = "https://api.kucoin.com"
DEFAULT_TIMEOUT = 12
MAX_RETRIES = 3
BACKOFF_START_S = 0.7  # exponential: 0.7, 1.4, 2.8 ...


class KuCoinHTTPError(RuntimeError):
    """Raised when KuCoin API returns an error response"""
    pass


class KuCoinAdapter:
    """
    KuCoin REST API adapter for cryptocurrency market data and trading.
    
    Supports both public endpoints (no API key required) and 
    private endpoints (requires API key, secret, and passphrase).
    """
    
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        passphrase: Optional[str] = None,
        sandbox: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize KuCoin adapter.
        
        Args:
            base_url: KuCoin API base URL
            timeout: Request timeout in seconds
            api_key: KuCoin API key (optional)
            api_secret: KuCoin API secret (optional)
            passphrase: KuCoin API passphrase (optional)
            sandbox: Use sandbox environment for testing
            session: Optional requests session
        """
        if sandbox:
            self.base_url = "https://openapi-sandbox.kucoin.com"
        else:
            self.base_url = base_url.rstrip("/")
            
        self.timeout = timeout
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session = session or requests.Session()
        
        # Set reasonable headers
        self.session.headers.update({
            "User-Agent": "TradingCore/1.0 KuCoin-Adapter",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

    def _sign_request(self, timestamp: str, method: str, endpoint: str, body: str = "") -> Dict[str, str]:
        """
        Create signature for authenticated requests.
        
        Args:
            timestamp: Request timestamp
            method: HTTP method
            endpoint: Request endpoint
            body: Request body
            
        Returns:
            Headers with authentication signature
        """
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise KuCoinHTTPError("API key, secret, and passphrase required for authenticated requests")
        
        # Create the message to sign
        str_to_sign = timestamp + method.upper() + endpoint + body
        
        # Create signature
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode(),
                str_to_sign.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        # Create passphrase signature
        passphrase_signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode(),
                self.passphrase.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        return {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase_signature,
            "KC-API-KEY-VERSION": "2"  # Version 2 for stronger security
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """
        Make HTTP request to KuCoin API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL parameters
            json_data: JSON request body
            authenticated: Whether to authenticate the request
            
        Returns:
            JSON response data
            
        Raises:
            KuCoinHTTPError: On API errors or network failures
        """
        url = urljoin(self.base_url, endpoint)
        headers = self.session.headers.copy()
        body = ""
        
        if authenticated:
            timestamp = str(int(time.time() * 1000))  # Millisecond precision
            if json_data:
                import json
                body = json.dumps(json_data, separators=(',', ':'))
            auth_headers = self._sign_request(timestamp, method, endpoint, body)
            headers.update(auth_headers)
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data if not authenticated else None,
                    data=body if authenticated and json_data else None,
                    headers=headers,
                    timeout=self.timeout
                )
                
                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        sleep_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    raise KuCoinHTTPError("Rate limit exceeded")
                
                # Handle server errors with retry
                if response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        sleep_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                
                # Check for client errors
                if not response.ok:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("msg", f"HTTP {response.status_code}")
                        error_code = error_data.get("code", "")
                        full_msg = f"{error_code}: {error_msg}" if error_code else error_msg
                    except:
                        full_msg = f"HTTP {response.status_code}"
                    raise KuCoinHTTPError(f"KuCoin API error: {full_msg}")
                
                # Parse response
                data = response.json()
                
                # Check KuCoin-specific error format
                if data.get("code") != "200000":
                    error_msg = data.get("msg", "Unknown error")
                    error_code = data.get("code", "")
                    raise KuCoinHTTPError(f"KuCoin API error {error_code}: {error_msg}")
                
                return data.get("data", {})
                
            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    sleep_time = BACKOFF_START_S * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                raise KuCoinHTTPError(f"Network error: {e}") from e
        
        raise KuCoinHTTPError("Max retries exceeded")

    # ─────────────────────────────────────────────────────────────────────
    # Public Market Data Methods
    # ─────────────────────────────────────────────────────────────────────

    def get_timestamp(self) -> Dict[str, Any]:
        """
        Get server timestamp.
        
        Returns:
            Server timestamp information
        """
        return self._request("GET", "/api/v1/timestamp")

    def get_symbols(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available symbols.
        
        Args:
            market: Market type filter (optional)
            
        Returns:
            List of symbol information
        """
        params = {}
        if market:
            params["market"] = market
            
        result = self._request("GET", "/api/v1/symbols", params)
        return result if isinstance(result, list) else []

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr stats for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            
        Returns:
            24hr ticker statistics
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v1/market/stats", params)

    def get_all_tickers(self) -> Dict[str, Any]:
        """
        Get 24hr stats for all symbols.
        
        Returns:
            All symbols 24hr statistics
        """
        return self._request("GET", "/api/v1/market/allTickers")

    def get_orderbook(self, symbol: str) -> Dict[str, Any]:
        """
        Get level 1 order book (best bid/ask) for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            
        Returns:
            Level 1 order book data
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v1/market/orderbook/level1", params)

    def get_full_orderbook(self, symbol: str) -> Dict[str, Any]:
        """
        Get full order book for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            
        Returns:
            Full order book data
        """
        params = {"symbol": symbol.upper()}
        return self._request("GET", "/api/v1/market/orderbook/level2_100", params)

    def get_trade_histories(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recent trade histories for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            
        Returns:
            List of recent trades
        """
        params = {"symbol": symbol.upper()}
        result = self._request("GET", "/api/v1/market/histories", params)
        return result if isinstance(result, list) else []

    def get_candles(
        self,
        symbol: str,
        type: str = "1day",
        start_at: Optional[int] = None,
        end_at: Optional[int] = None
    ) -> List[List[str]]:
        """
        Get kline data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USDT")
            type: Kline type (1min, 3min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 6hour, 8hour, 12hour, 1day, 1week)
            start_at: Start time as Unix timestamp in seconds
            end_at: End time as Unix timestamp in seconds
            
        Returns:
            List of kline data arrays [time, open, close, high, low, volume, turnover]
        """
        params = {
            "symbol": symbol.upper(),
            "type": type
        }
        
        if start_at:
            params["startAt"] = start_at
        if end_at:
            params["endAt"] = end_at
            
        result = self._request("GET", "/api/v1/market/candles", params)
        return result if isinstance(result, list) else []

    def get_currencies(self) -> List[Dict[str, Any]]:
        """
        Get list of available currencies.
        
        Returns:
            List of currency information
        """
        result = self._request("GET", "/api/v1/currencies")
        return result if isinstance(result, list) else []

    def get_currency(self, currency: str) -> Dict[str, Any]:
        """
        Get details of a specific currency.
        
        Args:
            currency: Currency code (e.g., "BTC")
            
        Returns:
            Currency details
        """
        return self._request("GET", f"/api/v1/currencies/{currency.upper()}")

    def get_fiat_prices(self, base: str = "USD", currencies: Optional[str] = None) -> Dict[str, Any]:
        """
        Get fiat prices for cryptocurrencies.
        
        Args:
            base: Base fiat currency (default: USD)
            currencies: Comma-separated list of currencies to get prices for
            
        Returns:
            Fiat price data
        """
        params = {"base": base.upper()}
        if currencies:
            params["currencies"] = currencies.upper()
            
        return self._request("GET", "/api/v1/prices", params)

    # ─────────────────────────────────────────────────────────────────────
    # Private Account Methods (require authentication)
    # ─────────────────────────────────────────────────────────────────────

    def get_accounts(self, currency: Optional[str] = None, type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get account list. Requires authentication.
        
        Args:
            currency: Currency filter (optional)
            type: Account type filter (main, trade, margin, isolated) (optional)
            
        Returns:
            List of account information
        """
        params = {}
        if currency:
            params["currency"] = currency.upper()
        if type:
            params["type"] = type
            
        result = self._request("GET", "/api/v1/accounts", params, authenticated=True)
        return result if isinstance(result, list) else []

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """
        Get single account info. Requires authentication.
        
        Args:
            account_id: Account ID
            
        Returns:
            Account information
        """
        return self._request("GET", f"/api/v1/accounts/{account_id}", authenticated=True)

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def normalize_ticker_data(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize 24hr ticker data to standard format.
        
        Args:
            ticker_data: Raw ticker data from KuCoin
            
        Returns:
            Normalized ticker data
        """
        try:
            symbol = ticker_data.get("symbol", "")
            last = float(ticker_data.get("last", 0))
            high = float(ticker_data.get("high", 0))
            low = float(ticker_data.get("low", 0))
            vol = float(ticker_data.get("vol", 0))
            change_rate = float(ticker_data.get("changeRate", 0))
            
            # Calculate day range percentage: (high - low) / last * 100
            day_range_pct = ((high - low) / last * 100) if last > 0 else 0
            
            # Convert change rate to percentage
            price_change_pct = change_rate * 100
            
            return {
                "symbol": symbol,
                "price": last,
                "volume": vol,
                "high": high,
                "low": low,
                "change": float(ticker_data.get("change", 0)),
                "price_change_pct": price_change_pct,
                "day_range_pct": day_range_pct,
                "vol_value": float(ticker_data.get("volValue", 0)),  # Volume in quote currency
                "avg_price": float(ticker_data.get("averagePrice", 0)),
                "taker_fee": float(ticker_data.get("takerFeeRate", 0)),
                "maker_fee": float(ticker_data.get("makerFeeRate", 0)),
            }
        except (ValueError, TypeError, KeyError) as e:
            raise KuCoinHTTPError(f"Failed to normalize ticker data: {e}") from e

    def normalize_candle_data(self, candle_data: List[List[str]]) -> List[Dict[str, Any]]:
        """
        Normalize candle data to standard OHLCV format.
        
        Args:
            candle_data: Raw candle data from KuCoin
            
        Returns:
            List of normalized OHLCV dictionaries
        """
        normalized = []
        
        for candle in candle_data:
            try:
                # KuCoin candle format: [time, open, close, high, low, volume, turnover]
                normalized.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "close": float(candle[2]),
                    "high": float(candle[3]),
                    "low": float(candle[4]),
                    "volume": float(candle[5]),
                    "turnover": float(candle[6])  # Volume in quote currency
                })
            except (ValueError, TypeError, IndexError):
                # Skip malformed candles
                continue
                
        return normalized

    def normalize_orderbook_data(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize order book data to standard format.
        
        Args:
            orderbook_data: Raw order book data from KuCoin
            
        Returns:
            Normalized order book data
        """
        try:
            bid_price = float(orderbook_data.get("bestBid", 0))
            ask_price = float(orderbook_data.get("bestAsk", 0))
            bid_size = float(orderbook_data.get("bestBidSize", 0))
            ask_size = float(orderbook_data.get("bestAskSize", 0))
            
            # Calculate spread
            spread = ask_price - bid_price
            spread_pct = (spread / ask_price * 100) if ask_price > 0 else 0
            
            return {
                "bid": bid_price,
                "ask": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "spread": spread,
                "spread_pct": spread_pct,
                "timestamp": int(orderbook_data.get("time", 0))
            }
        except (ValueError, TypeError, KeyError) as e:
            raise KuCoinHTTPError(f"Failed to normalize orderbook data: {e}") from e

    def get_active_symbols(self, market: Optional[str] = None, quote_currency: Optional[str] = None) -> List[str]:
        """
        Get list of active trading symbols, optionally filtered.
        
        Args:
            market: Market type filter (optional)
            quote_currency: Filter by quote currency (e.g., "USDT")
            
        Returns:
            List of active symbol names
        """
        symbols_data = self.get_symbols(market=market)
        active_symbols = []
        
        for symbol_info in symbols_data:
            # Only include enabled symbols
            if not symbol_info.get("enableTrading", False):
                continue
                
            symbol = symbol_info.get("symbol", "")
            quote = symbol_info.get("quoteCurrency", "")
            
            # Apply quote currency filter if specified
            if quote_currency and quote.upper() != quote_currency.upper():
                continue
                
            active_symbols.append(symbol)
            
        return active_symbols

    def get_crypto_symbols(self) -> List[str]:
        """
        Get list of cryptocurrency trading symbols.
        
        Returns:
            List of crypto symbol names
        """
        symbols_data = self.get_symbols()
        crypto_symbols = []
        
        # Common fiat currencies
        fiat_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"}
        
        for symbol_info in symbols_data:
            if not symbol_info.get("enableTrading", False):
                continue
                
            base = symbol_info.get("baseCurrency", "")
            symbol = symbol_info.get("symbol", "")
            
            # Include if base is not fiat (likely crypto)
            if base not in fiat_currencies:
                crypto_symbols.append(symbol)
                
        return crypto_symbols

    def health_check(self) -> bool:
        """
        Check if the KuCoin API is accessible.
        
        Returns:
            True if API is responding, False otherwise
        """
        try:
            self.get_timestamp()
            return True
        except Exception:
            return False
