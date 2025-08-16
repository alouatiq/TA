# BE/trading_core/data_fetcher/adapters/kraken.py
"""
Kraken adapter
──────────────
Lightweight HTTP client for Kraken REST API endpoints:

- /0/public/Assets             → Get asset info
- /0/public/AssetPairs         → Get tradable asset pairs
- /0/public/Ticker             → Get ticker information
- /0/public/OHLC               → Get OHLC data
- /0/public/Depth              → Get order book
- /0/public/Trades             → Get recent trades
- /0/public/Spread             → Get recent spread data
- /0/public/SystemStatus       → Get system status

Design goals
------------
• No hardcoded symbols. Callers specify which pairs to request.
• Conservative timeouts + retries with exponential backoff.
• Normalized, minimal return payloads for consistent upstream usage.
• Works without API key for public market data endpoints.
• Optional API key support for private account data and trading.
• Handles Kraken's unique symbol naming conventions.

Usage
-----
from trading_core.data_fetcher.adapters.kraken import KrakenAdapter

kraken = KrakenAdapter()
assets = kraken.get_assets()  # All assets
pairs = kraken.get_asset_pairs()  # All trading pairs
ticker = kraken.get_ticker("XBTUSD")  # Single pair ticker
ohlc = kraken.get_ohlc("ETHUSDT", interval=1440)  # Daily OHLC
"""

from __future__ import annotations

import time
import base64
import hashlib
import hmac
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, urljoin

import requests


DEFAULT_BASE_URL = "https://api.kraken.com"
DEFAULT_TIMEOUT = 15
MAX_RETRIES = 3
BACKOFF_START_S = 0.8  # exponential: 0.8, 1.6, 3.2 ...


class KrakenHTTPError(RuntimeError):
    """Raised when Kraken API returns an error response"""
    pass


class KrakenAdapter:
    """
    Kraken REST API adapter for cryptocurrency market data and trading.
    
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
        Initialize Kraken adapter.
        
        Args:
            base_url: Kraken API base URL
            timeout: Request timeout in seconds
            api_key: Kraken API key (optional)
            api_secret: Kraken API secret (optional)
            session: Optional requests session
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = session or requests.Session()
        
        # Set reasonable headers
        self.session.headers.update({
            "User-Agent": "TradingCore/1.0 Kraken-Adapter",
            "Accept": "application/json"
        })

    def _sign_request(self, endpoint: str, data: Dict[str, Any], nonce: int) -> str:
        """
        Create signature for authenticated requests.
        
        Args:
            endpoint: API endpoint path
            data: Request data
            nonce: Unique nonce value
            
        Returns:
            Base64 encoded signature
        """
        if not self.api_secret:
            raise KrakenHTTPError("API secret required for authenticated requests")
        
        # Create the message to sign
        postdata = urlencode(data)
        encoded = (str(nonce) + postdata).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        
        # Create signature
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        authenticated: bool = False
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Kraken API with retry logic.
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            authenticated: Whether to authenticate the request
            
        Returns:
            JSON response data
            
        Raises:
            KrakenHTTPError: On API errors or network failures
        """
        url = urljoin(self.base_url, endpoint)
        params = params or {}
        headers = self.session.headers.copy()
        
        # Add authentication if required
        if authenticated:
            if not self.api_key:
                raise KrakenHTTPError("API key required for authenticated requests")
            
            nonce = int(time.time() * 1000000)  # Microsecond precision
            params["nonce"] = nonce
            
            headers.update({
                "API-Key": self.api_key,
                "API-Sign": self._sign_request(endpoint, params, nonce)
            })
        
        for attempt in range(MAX_RETRIES + 1):
            try:
                if authenticated:
                    # Use POST for authenticated requests
                    response = self.session.post(
                        url,
                        data=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                else:
                    # Use GET for public requests
                    response = self.session.get(
                        url,
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )
                
                # Handle rate limiting and server errors
                if response.status_code in (429, 500, 502, 503, 504):
                    if attempt < MAX_RETRIES:
                        sleep_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(sleep_time)
                        continue
                    raise KrakenHTTPError(f"HTTP {response.status_code}: Server error")
                
                # Check for client errors
                if not response.ok:
                    raise KrakenHTTPError(f"HTTP {response.status_code}: {response.text}")
                
                # Parse response
                data = response.json()
                
                # Check for Kraken API errors
                if "error" in data and data["error"]:
                    error_msgs = ", ".join(data["error"])
                    raise KrakenHTTPError(f"Kraken API error: {error_msgs}")
                
                return data.get("result", {})
                
            except requests.RequestException as e:
                if attempt < MAX_RETRIES:
                    sleep_time = BACKOFF_START_S * (2 ** attempt)
                    time.sleep(sleep_time)
                    continue
                raise KrakenHTTPError(f"Network error: {e}") from e
        
        raise KrakenHTTPError("Max retries exceeded")

    # ─────────────────────────────────────────────────────────────────────
    # Public Market Data Methods
    # ─────────────────────────────────────────────────────────────────────

    def get_server_time(self) -> Dict[str, Any]:
        """
        Get server time.
        
        Returns:
            Server time information
        """
        return self._request("/0/public/Time")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status.
        
        Returns:
            System status information
        """
        return self._request("/0/public/SystemStatus")

    def get_assets(self, assets: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get asset info.
        
        Args:
            assets: List of assets to get info for (optional)
            
        Returns:
            Asset information dictionary
        """
        params = {}
        if assets:
            params["asset"] = ",".join(assets)
            
        return self._request("/0/public/Assets", params)

    def get_asset_pairs(self, pairs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get tradable asset pairs.
        
        Args:
            pairs: List of asset pairs to get info for (optional)
            
        Returns:
            Asset pair information dictionary
        """
        params = {}
        if pairs:
            params["pair"] = ",".join(pairs)
            
        return self._request("/0/public/AssetPairs", params)

    def get_ticker(self, pairs: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Get ticker information for asset pairs.
        
        Args:
            pairs: Single pair string or list of pairs
            
        Returns:
            Ticker information dictionary
        """
        if isinstance(pairs, str):
            pairs = [pairs]
            
        params = {"pair": ",".join(pairs)}
        return self._request("/0/public/Ticker", params)

    def get_ohlc(
        self,
        pair: str,
        interval: int = 1,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get OHLC data for a pair.
        
        Args:
            pair: Asset pair to get data for
            interval: Time frame interval in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Return data since given timestamp
            
        Returns:
            OHLC data dictionary
        """
        params = {
            "pair": pair,
            "interval": interval
        }
        
        if since:
            params["since"] = since
            
        return self._request("/0/public/OHLC", params)

    def get_order_book(self, pair: str, count: int = 100) -> Dict[str, Any]:
        """
        Get order book for a pair.
        
        Args:
            pair: Asset pair to get order book for
            count: Maximum number of asks/bids to return (default 100)
            
        Returns:
            Order book data
        """
        params = {
            "pair": pair,
            "count": count
        }
        
        return self._request("/0/public/Depth", params)

    def get_recent_trades(self, pair: str, since: Optional[int] = None) -> Dict[str, Any]:
        """
        Get recent trades for a pair.
        
        Args:
            pair: Asset pair to get trades for
            since: Return trade data since given timestamp
            
        Returns:
            Recent trades data
        """
        params = {"pair": pair}
        
        if since:
            params["since"] = since
            
        return self._request("/0/public/Trades", params)

    def get_recent_spreads(self, pair: str, since: Optional[int] = None) -> Dict[str, Any]:
        """
        Get recent spread data for a pair.
        
        Args:
            pair: Asset pair to get spread data for
            since: Return spread data since given timestamp
            
        Returns:
            Recent spread data
        """
        params = {"pair": pair}
        
        if since:
            params["since"] = since
            
        return self._request("/0/public/Spread", params)

    # ─────────────────────────────────────────────────────────────────────
    # Private Account Methods (require authentication)
    # ─────────────────────────────────────────────────────────────────────

    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance. Requires authentication.
        
        Returns:
            Account balance information
        """
        return self._request("/0/private/Balance", authenticated=True)

    def get_trade_balance(self, asset: str = "ZUSD") -> Dict[str, Any]:
        """
        Get trade balance. Requires authentication.
        
        Args:
            asset: Base asset used to determine balance
            
        Returns:
            Trade balance information
        """
        params = {"asset": asset}
        return self._request("/0/private/TradeBalance", params, authenticated=True)

    # ─────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────

    def normalize_ticker_data(self, pair: str, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize ticker data to standard format.
        
        Args:
            pair: Trading pair name
            ticker_data: Raw ticker data from Kraken
            
        Returns:
            Normalized ticker data
        """
        try:
            # Kraken ticker format: [ask_price, ask_lot_volume, ask_volume],
            # [bid_price, bid_lot_volume, bid_volume], [last_price, last_volume],
            # [volume_today, volume_24h], [vwap_today, vwap_24h], [trades_today, trades_24h],
            # [low_today, low_24h], [high_today, high_24h], [open_price]
            
            ask = float(ticker_data["a"][0])  # Ask price
            bid = float(ticker_data["b"][0])  # Bid price
            last = float(ticker_data["c"][0])  # Last trade price
            volume_24h = float(ticker_data["v"][1])  # 24h volume
            high_24h = float(ticker_data["h"][1])  # 24h high
            low_24h = float(ticker_data["l"][1])  # 24h low
            open_price = float(ticker_data["o"])  # Open price
            
            # Calculate derived metrics
            spread_pct = ((ask - bid) / last * 100) if last > 0 and ask > bid else 0
            day_range_pct = ((high_24h - low_24h) / last * 100) if last > 0 else 0
            price_change_pct = ((last - open_price) / open_price * 100) if open_price > 0 else 0
            
            return {
                "pair": pair,
                "ask": ask,
                "bid": bid,
                "last": last,
                "volume_24h": volume_24h,
                "high_24h": high_24h,
                "low_24h": low_24h,
                "open": open_price,
                "spread_pct": spread_pct,
                "day_range_pct": day_range_pct,
                "price_change_pct": price_change_pct,
                "vwap_24h": float(ticker_data["p"][1]),  # 24h VWAP
                "trades_24h": int(ticker_data["t"][1]),  # 24h trade count
            }
        except (ValueError, TypeError, KeyError) as e:
            raise KrakenHTTPError(f"Failed to normalize ticker data for {pair}: {e}") from e

    def normalize_ohlc_data(self, ohlc_data: List[List[Any]]) -> List[Dict[str, Any]]:
        """
        Normalize OHLC data to standard format.
        
        Args:
            ohlc_data: Raw OHLC data from Kraken
            
        Returns:
            List of normalized OHLCV dictionaries
        """
        normalized = []
        
        for candle in ohlc_data:
            try:
                # Kraken OHLC format: [time, open, high, low, close, vwap, volume, count]
                normalized.append({
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "vwap": float(candle[5]),
                    "volume": float(candle[6]),
                    "count": int(candle[7])  # Trade count
                })
            except (ValueError, TypeError, IndexError):
                # Skip malformed candles
                continue
                
        return normalized

    def get_active_pairs(self, quote_currency: Optional[str] = None) -> List[str]:
        """
        Get list of active trading pairs, optionally filtered by quote currency.
        
        Args:
            quote_currency: Filter by quote currency (e.g., "USD", "EUR")
            
        Returns:
            List of active pair names (Kraken format)
        """
        pairs_data = self.get_asset_pairs()
        active_pairs = []
        
        for pair_name, pair_info in pairs_data.items():
            # Skip if pair is not active
            if pair_info.get("status") != "online":
                continue
            
            # Apply quote currency filter if specified
            if quote_currency:
                quote = pair_info.get("quote", "")
                # Handle Kraken's currency naming (e.g., ZUSD for USD)
                if not (quote.endswith(quote_currency.upper()) or quote == f"Z{quote_currency.upper()}"):
                    continue
                    
            active_pairs.append(pair_name)
            
        return active_pairs

    def get_crypto_pairs(self) -> List[str]:
        """
        Get list of cryptocurrency trading pairs (excludes pure forex pairs).
        
        Returns:
            List of crypto pair names
        """
        pairs_data = self.get_asset_pairs()
        crypto_pairs = []
        
        # Common fiat currency prefixes in Kraken
        fiat_prefixes = {"Z", "X"}  # Z for fiat, X for crypto (though not always consistent)
        fiat_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF"}
        
        for pair_name, pair_info in pairs_data.items():
            if pair_info.get("status") != "online":
                continue
                
            base = pair_info.get("base", "")
            quote = pair_info.get("quote", "")
            
            # Remove Kraken prefixes for checking
            base_clean = base.lstrip("XZ")
            quote_clean = quote.lstrip("XZ")
            
            # Include if base is not a pure fiat currency (likely crypto)
            if base_clean not in fiat_currencies:
                crypto_pairs.append(pair_name)
                
        return crypto_pairs

    def convert_symbol_to_kraken(self, symbol: str) -> str:
        """
        Convert common symbol formats to Kraken format.
        
        Args:
            symbol: Symbol in common format (e.g., "BTC-USD", "BTCUSD")
            
        Returns:
            Symbol in Kraken format (e.g., "XBTUSD")
        """
        # Remove common separators
        symbol = symbol.replace("-", "").replace("/", "").replace("_", "").upper()
        
        # Common symbol mappings for Kraken
        symbol_map = {
            "BTC": "XBT",  # Kraken uses XBT for Bitcoin
            "DOGE": "XXDG",  # Kraken uses XXDG for Dogecoin
        }
        
        # Apply mappings if needed
        for standard, kraken in symbol_map.items():
            symbol = symbol.replace(standard, kraken)
            
        return symbol

    def convert_symbol_from_kraken(self, kraken_symbol: str) -> str:
        """
        Convert Kraken symbol format to common format.
        
        Args:
            kraken_symbol: Symbol in Kraken format
            
        Returns:
            Symbol in common format
        """
        # Reverse mappings
        reverse_map = {
            "XBT": "BTC",
            "XXDG": "DOGE",
        }
        
        symbol = kraken_symbol
        for kraken, standard in reverse_map.items():
            symbol = symbol.replace(kraken, standard)
            
        return symbol

    def health_check(self) -> bool:
        """
        Check if the Kraken API is accessible.
        
        Returns:
            True if API is responding, False otherwise
        """
        try:
            status = self.get_system_status()
            return status.get("status") == "online"
        except Exception:
            return False
