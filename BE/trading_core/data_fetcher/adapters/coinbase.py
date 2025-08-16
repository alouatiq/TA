# BE/trading_core/data_fetcher/adapters/coinbase.py
"""
Coinbase adapter
────────────────
Lightweight HTTP client for Coinbase Exchange API (formerly Coinbase Pro) endpoints:

- /products                    → Get trading pairs information
- /products/{product-id}/ticker → Get current market ticker
- /products/{product-id}/stats  → Get 24hr stats for a product
- /products/{product-id}/candles → Get historic rates (OHLCV)
- /currencies                  → Get list of known currencies
- /exchange-rates              → Get current exchange rates

Design goals
------------
• No hardcoded symbols. Callers specify which products/pairs to request.
• Conservative timeouts + retries with exponential backoff.
• Normalized, minimal return payloads for consistent upstream usage.
• Works without API key for public market data endpoints.
• Optional API key support for private account data and trading.

Usage
-----
from trading_core.data_fetcher.adapters.coinbase import CoinbaseAdapter

coinbase = CoinbaseAdapter()
products = coinbase.get_products()  # All trading pairs
ticker = coinbase.get_ticker("BTC-USD")  # Single product ticker
stats = coinbase.get_24hr_stats("ETH-USD")  # 24hr statistics
candles = coinbase.get_candles("BTC-USD", granularity=86400)  # Daily candles
"""

from __future__ import annotations

import time
import base64
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import requests


DEFAULT_BASE_URL = "https://api.exchange.coinbase.com"
DEFAULT_TIMEOUT = 12
MAX_RETRIES = 3
BACKOFF_START_S = 0.6  # exponential: 0.6, 1.2, 2.4 ...


class CoinbaseHTTPError(RuntimeError):
    """Raised when Coinbase API returns an error response"""
    pass


class CoinbaseAdapter:
    """
    Coinbase Exchange (Pro) REST API adapter for cryptocurrency market data.
    
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
        Initialize Coinbase adapter.
        
        Args:
            base_url: Coinbase API base URL
            timeout: Request timeout in seconds
            api_key: Coinbase API key (optional)
            api_secret: Coinbase API secret (optional)
            passphrase: Coinbase API passphrase (optional)
            sandbox: Use sandbox environment for testing
            session: Optional requests session
        """
        if sandbox:
            self.base_url = "https://api-public.sandbox.exchange.coinbase.com"
        else:
            self.base_url = base_url.rstrip("/")
            
        self.timeout = timeout
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.session = session or requests.Session()
        
        # Set reasonable headers
        self.session.headers.update({
            "User-Agent": "TradingCore/1.0 Coinbase-Adapter",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> Dict[str, str]:
        """
        Create signature for authenticated requests.
        
        Args:
            timestamp: Request timestamp
            method: HTTP method
            path: Request path
            body: Request body
            
        Returns:
            Headers with authentication signature
        """
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise CoinbaseHTTPError("API key, secret, and passphrase required for authenticated requests")
        
        # Create the prehash string
        message = timestamp + method.upper() + path + body
        
        # Decode the secret
        secret = base64.b64decode(self.api_secret)
        
        # Create signature
        signature = hmac.new(secret, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()
        
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self.passphrase
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        authenticated: bool = False
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Make HTTP request to Coinbase API with retry logic.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: URL parameters
            json_data: JSON request body
            authenticated: Whether to authenticate the request
            
        Returns:
            JSON response data
            
        Raises:
            CoinbaseHTTPError: On API errors or network failures
        """
        url = urljoin(self.base_url, endpoint)
        path = endpoint
        
        # Prepare request
        headers = self.session.headers.copy()
        body = ""
        
        if authenticated:
            timestamp = str(time.time())
            if json_data:
                import json
                body = json.dumps(json_data, separators=(',', ':'))
            auth_headers = self._sign_request(timestamp, method, path, body)
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
                    raise CoinbaseHTTPError("Rate limit exceeded")
                
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
                        error_msg = error_data.get("message", f"HTTP {response.status_code}")
                    except:
                        error_msg = f"HTTP {response.status_code}"
                    raise CoinbaseHTTPError(f"Coinbase API error: {error_msg}")
                
                # Parse response
                return response.json()
                
            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    sleep_time = BACKOFF_START_S * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                raise CoinbaseHTTPError(f"Network error: {e}") from e
        
        raise CoinbaseHTTPError("Max retries exceeded")

    # ─────────────────────────────────────────────────────────────────────
    # Public Market Data Methods
    # ─────────────────────────────────────────────────────────────────────

    def get_products(self) -> List[Dict[str, Any]]:
        """
        Get a list of available currency pairs for trading.
        
        Returns:
            List of product/trading pair information
        """
        return self._request("GET", "/products")

    def get_product(self, product_id: str) -> Dict[str, Any]:
        """
        Get information about a single product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            
        Returns:
            Product information
        """
        return self._request("GET", f"/products/{product_id}")

    def get_ticker(self, product_id: str) -> Dict[str, Any]:
        """
        Get current market ticker for a product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            
        Returns:
            Current ticker data including price, volume, spread
        """
        return self._request("GET", f"/products/{product_id}/ticker")

    def get_24hr_stats(self, product_id: str) -> Dict[str, Any]:
        """
        Get 24hr statistics for a product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            
        Returns:
            24hr trading statistics
        """
        return self._request("GET", f"/products/{product_id}/stats")

    def get_candles(
        self,
        product_id: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        granularity: int = 86400
    ) -> List[List[Any]]:
        """
        Get historic rates (OHLCV) for a product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            start: Start time in ISO 8601 format (optional)
            end: End time in ISO 8601 format (optional)
            granularity: Time slice in seconds (60, 300, 900, 3600, 21600, 86400)
            
        Returns:
            List of candle data arrays [timestamp, low, high, open, close, volume]
        """
        params = {"granularity": granularity}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
            
        return self._request("GET", f"/products/{product_id}/candles", params=params)

    def get_order_book(self, product_id: str, level: int = 1) -> Dict[str, Any]:
        """
        Get order book for a product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            level: Level of detail (1=best bid/ask, 2=top 50, 3=full book)
            
        Returns:
            Order book data
        """
        params = {"level": level}
        return self._request("GET", f"/products/{product_id}/book", params=params)

    def get_trades(self, product_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for a product.
        
        Args:
            product_id: Trading pair ID (e.g., "BTC-USD")
            limit: Number of trades to return (max 1000)
            
        Returns:
            List of recent trades
        """
        params = {"limit": min(limit, 1000)}
        return self._request("GET", f"/products/{product_id}/trades", params=params)

    def get_currencies(self) -> List[Dict[str, Any]]:
        """
        Get list of known currencies.
        
        Returns:
            List of currency information
        """
        return self._request("GET", "/currencies")

    def get_exchange_rates(self, currency: str = "USD") -> Dict[str, Any]:
        """
        Get current exchange rates.
        
        Args:
            currency: Base currency for rates (default: USD)
            
        Returns:
            Exchange rates data
        """
        params = {"currency": currency}
        return self._request("GET", "/exchange-rates", params=params)

    def get_time(self) -> Dict[str, Any]:
        """
        Get the API server time.
        
        Returns:
            Server time information
        """
        return self._request("GET", "/time")

    # ─────────────────────────────────────────────────────────────────────
    # Private Account Methods (require authentication)
    # ─────────────────────────────────────────────────────────────────────

    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get account information. Requires authentication.
        
        Returns:
            List of account balances
        """
        return self._request("GET", "/accounts", authenticated=True)

    def get_account(self, account_id: str) -> Dict[str, Any]:
        """
        Get information for a single account. Requires authentication.
        
        Args:
            account_id: Account UUID
            
        Returns:
            Account information
        """
        return self._request("GET", f"/accounts/{account_id}", authenticated=True)

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def normalize_ticker_data(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize ticker data to standard format.
        
        Args:
            ticker_data: Raw ticker data from Coinbase
            
        Returns:
            Normalized ticker data
        """
        try:
            price = float(ticker_data.get("price", 0))
            volume = float(ticker_data.get("volume", 0))
            bid = float(ticker_data.get("bid", 0))
            ask = float(ticker_data.get("ask", 0))
            
            # Calculate spread percentage
            spread_pct = ((ask - bid) / price * 100) if price > 0 and ask > bid else 0
            
            return {
                "price": price,
                "volume": volume,
                "bid": bid,
                "ask": ask,
                "spread_pct": spread_pct,
                "size": float(ticker_data.get("size", 0)),  # Size of last trade
                "time": ticker_data.get("time", ""),  # Time of last trade
            }
        except (ValueError, TypeError, KeyError) as e:
            raise CoinbaseHTTPError(f"Failed to normalize ticker data: {e}") from e

    def normalize_24hr_stats(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize 24hr stats data to standard format.
        
        Args:
            stats_data: Raw 24hr stats from Coinbase
            
        Returns:
            Normalized 24hr statistics
        """
        try:
            open_price = float(stats_data.get("open", 0))
            high = float(stats_data.get("high", 0))
            low = float(stats_data.get("low", 0))
            last = float(stats_data.get("last", 0))
            volume = float(stats_data.get("volume", 0))
            
            # Calculate day range percentage: (high - low) / last * 100
            day_range_pct = ((high - low) / last * 100) if last > 0 else 0
            
            # Calculate price change percentage
            price_change_pct = ((last - open_price) / open_price * 100) if open_price > 0 else 0
            
            return {
                "open": open_price,
                "high": high,
                "low": low,
                "last": last,
                "volume": volume,
                "volume_30day": float(stats_data.get("volume_30day", 0)),
                "day_range_pct": day_range_pct,
                "price_change_pct": price_change_pct,
            }
        except (ValueError, TypeError, KeyError) as e:
            raise CoinbaseHTTPError(f"Failed to normalize 24hr stats: {e}") from e

    def normalize_candle_data(self, candle_data: List[List[Any]]) -> List[Dict[str, Any]]:
        """
        Normalize candle data to standard OHLCV format.
        
        Args:
            candle_data: Raw candle data from Coinbase
            
        Returns:
            List of normalized OHLCV dictionaries
        """
        normalized = []
        
        for candle in candle_data:
            try:
                # Coinbase candle format: [timestamp, low, high, open, close, volume]
                normalized.append({
                    "timestamp": int(candle[0]),
                    "low": float(candle[1]),
                    "high": float(candle[2]), 
                    "open": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })
            except (ValueError, TypeError, IndexError):
                # Skip malformed candles
                continue
                
        return normalized

    def get_active_products(self, quote_currency: Optional[str] = None) -> List[str]:
        """
        Get list of active trading products, optionally filtered by quote currency.
        
        Args:
            quote_currency: Filter by quote currency (e.g., "USD", "BTC")
            
        Returns:
            List of active product IDs
        """
        products = self.get_products()
        active_products = []
        
        for product in products:
            # Only include active products
            if product.get("status") != "online":
                continue
                
            product_id = product.get("id", "")
            
            # Apply quote currency filter if specified
            if quote_currency:
                quote = product.get("quote_currency", "")
                if quote.upper() != quote_currency.upper():
                    continue
                    
            active_products.append(product_id)
            
        return active_products

    def get_crypto_products(self) -> List[str]:
        """
        Get list of cryptocurrency trading products (excludes forex pairs).
        
        Returns:
            List of crypto product IDs
        """
        products = self.get_products()
        crypto_products = []
        
        # Common fiat currencies to exclude
        fiat_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"}
        
        for product in products:
            if product.get("status") != "online":
                continue
                
            base = product.get("base_currency", "")
            quote = product.get("quote_currency", "")
            
            # Include if base is not fiat (likely crypto)
            if base not in fiat_currencies:
                crypto_products.append(product.get("id", ""))
                
        return crypto_products

    def health_check(self) -> bool:
        """
        Check if the Coinbase API is accessible.
        
        Returns:
            True if API is responding, False otherwise
        """
        try:
            self.get_time()
            return True
        except Exception:
            return False
