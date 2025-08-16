# BE/trading_core/data_fetcher/adapters/binance.py
"""
Binance adapter
───────────────
Lightweight HTTP client for Binance REST API endpoints:

- /api/v3/ticker/24hr          → 24hr ticker statistics for all symbols
- /api/v3/ticker/price         → Latest price for a symbol or all symbols
- /api/v3/klines               → Kline/candlestick data
- /api/v3/exchangeInfo         → Exchange trading rules and symbol information

Design goals
------------
• No hardcoded symbols. Callers specify which symbols to request.
• Conservative timeouts + retries with exponential backoff.
• Normalized, minimal return payloads for consistent upstream usage.
• Works without API key for public endpoints (respects rate limits).
• Optional API key support for higher rate limits and account data.

Usage
-----
from trading_core.data_fetcher.adapters.binance import BinanceAdapter

binance = BinanceAdapter()
tickers = binance.get_24hr_ticker()  # All symbols
price = binance.get_price("BTCUSDT")  # Single symbol price
klines = binance.get_klines("BTCUSDT", interval="1d", limit=30)
"""

from __future__ import annotations

import time
import hmac
import hashlib
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import requests


DEFAULT_BASE_URL = "https://api.binance.com"
DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_START_S = 0.5  # exponential: 0.5, 1.0, 2.0 ...


class BinanceHTTPError(RuntimeError):
    """Raised when Binance API returns an error response"""
    pass


class BinanceAdapter:
    """
    Binance REST API adapter for cryptocurrency market data.
    
    Supports both public endpoints (no API key required) and 
    private endpoints (requires API key and secret).
    """
    
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        """
        Initialize Binance adapter.
        
        Args:
            base_url: Binance API base URL
            timeout: Request timeout in seconds
            api_key: Binance API key (optional)
            api_secret: Binance API secret (optional)
            session: Optional requests session
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = session or requests.Session()
        
        # Set reasonable headers
        self.session.headers.update({
            "User-Agent": "TradingCore/1.0 Binance-Adapter",
            "Accept": "application/json"
        })
        
        if self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

    def _sign_request(self, params: Dict[str, Any]) -> str:
        """
        Create HMAC SHA256 signature for signed endpoints.
        
        Args:
            params: Request parameters
            
        Returns:
            HMAC signature
        """
        if not self.api_secret:
            raise BinanceHTTPError("API secret required for signed requests")
            
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Binance API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            JSON response data
            
        Raises:
            BinanceHTTPError: On API errors or network failures
        """
        params = params or {}
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp for signed requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign_request(params)
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params if method == "GET" else None,
                    json=params if method != "GET" else None,
                    timeout=self.timeout
                )
                
                # Handle rate limiting (HTTP 429)
                if response.status_code == 429:
                    if attempt < MAX_RETRIES:
                        sleep_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    raise BinanceHTTPError("Rate limit exceeded")
                
                # Handle server errors with retry
                if response.status_code >= 500:
                    if attempt < MAX_RETRIES:
                        sleep_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                
                # Check for client errors
                if not response.ok:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("msg", f"HTTP {response.status_code}")
                    raise BinanceHTTPError(f"Binance API error: {error_msg}")
                
                return response.json()
                
            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    sleep_time = BACKOFF_START_S * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                raise BinanceHTTPError(f"Network error: {e}") from e
        
        raise BinanceHTTPError("Max retries exceeded")

    # ─────────────────────────────────────────────────────────────────────
    # Public Market Data Methods
    # ─────────────────────────────────────────────────────────────────────

    def get_24hr_ticker(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get 24hr ticker price change statistics.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT"). If None, returns all symbols.
            
        Returns:
            Ticker data for single symbol (dict) or all symbols (list of dicts)
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
            
        return self._request("GET", "/api/v3/ticker/24hr", params)

    def get_price(self, symbol: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get latest price for a symbol or all symbols.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT"). If None, returns all symbols.
            
        Returns:
            Price data for single symbol (dict) or all symbols (list of dicts)
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
            
        return self._request("GET", "/api/v3/ticker/price", params)

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[List[Any]]:
        """
        Get kline/candlestick data for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            start_time: Start time as Unix timestamp in milliseconds
            end_time: End time as Unix timestamp in milliseconds
            limit: Number of klines to return (max 1000)
            
        Returns:
            List of kline data arrays
        """
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(limit, 1000)  # Binance max limit
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
            
        return self._request("GET", "/api/v3/klines", params)

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.
        
        Args:
            symbol: Trading symbol (optional). If provided, returns info for that symbol only.
            
        Returns:
            Exchange information
        """
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
            
        return self._request("GET", "/api/v3/exchangeInfo", params)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book depth for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            limit: Number of entries to return (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Order book data
        """
        valid_limits = [5, 10, 20, 50, 100, 500, 1000, 5000]
        if limit not in valid_limits:
            limit = min(valid_limits, key=lambda x: abs(x - limit))
            
        params = {
            "symbol": symbol.upper(),
            "limit": limit
        }
        
        return self._request("GET", "/api/v3/depth", params)

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def normalize_ticker_data(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize 24hr ticker data to standard format.
        
        Args:
            ticker_data: Raw ticker data from Binance
            
        Returns:
            Normalized ticker data
        """
        try:
            price = float(ticker_data.get("lastPrice", 0))
            volume = float(ticker_data.get("volume", 0))
            price_change_pct = float(ticker_data.get("priceChangePercent", 0))
            high = float(ticker_data.get("highPrice", 0))
            low = float(ticker_data.get("lowPrice", 0))
            
            # Calculate day range percentage: (high - low) / price * 100
            day_range_pct = ((high - low) / price * 100) if price > 0 else 0
            
            return {
                "symbol": ticker_data.get("symbol", ""),
                "price": price,
                "volume": int(volume),
                "price_change_pct": price_change_pct,
                "day_range_pct": day_range_pct,
                "high": high,
                "low": low,
                "open": float(ticker_data.get("openPrice", 0)),
                "close": price,  # lastPrice is essentially the close
                "count": int(ticker_data.get("count", 0)),  # Trade count
            }
        except (ValueError, TypeError, KeyError) as e:
            raise BinanceHTTPError(f"Failed to normalize ticker data: {e}") from e

    def normalize_kline_data(self, kline_data: List[List[Any]]) -> List[Dict[str, Any]]:
        """
        Normalize kline data to standard OHLCV format.
        
        Args:
            kline_data: Raw kline data from Binance
            
        Returns:
            List of normalized OHLCV dictionaries
        """
        normalized = []
        
        for kline in kline_data:
            try:
                # Binance kline format:
                # [timestamp, open, high, low, close, volume, close_time, 
                #  quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, 
                #  taker_buy_quote_asset_volume, ignore]
                
                normalized.append({
                    "timestamp": int(kline[0]),  # Open time
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                    "close_time": int(kline[6]),
                    "quote_volume": float(kline[7]),
                    "trades": int(kline[8]),
                    "taker_buy_volume": float(kline[9]),
                    "taker_buy_quote_volume": float(kline[10])
                })
            except (ValueError, TypeError, IndexError) as e:
                # Skip malformed klines
                continue
                
        return normalized

    def get_trading_symbols(self, base_asset: Optional[str] = None, quote_asset: Optional[str] = None) -> List[str]:
        """
        Get list of trading symbols, optionally filtered by base or quote asset.
        
        Args:
            base_asset: Filter by base asset (e.g., "BTC")
            quote_asset: Filter by quote asset (e.g., "USDT")
            
        Returns:
            List of trading symbols
        """
        exchange_info = self.get_exchange_info()
        symbols = []
        
        for symbol_info in exchange_info.get("symbols", []):
            if symbol_info.get("status") != "TRADING":
                continue
                
            symbol = symbol_info.get("symbol", "")
            base = symbol_info.get("baseAsset", "")
            quote = symbol_info.get("quoteAsset", "")
            
            # Apply filters if specified
            if base_asset and base.upper() != base_asset.upper():
                continue
            if quote_asset and quote.upper() != quote_asset.upper():
                continue
                
            symbols.append(symbol)
            
        return symbols
