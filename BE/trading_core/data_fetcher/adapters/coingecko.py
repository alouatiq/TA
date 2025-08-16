"""
CoinGecko adapter
─────────────────
Lightweight HTTP client for a few reliable CoinGecko v3 endpoints we need:

- /coins/markets             → discover + quote top coins by volume/market cap
- /coins/{id}/market_chart   → OHLC/price history for indicators (daily/hourly)
- /simple/price              → quick spot prices for specific ids
- /ping                      → health check

Design goals
------------
• No hardcoded coin symbols or ids. Callers decide which ids to request.
• Conservative timeouts + retries with exponential backoff.
• Normalized, minimal return payloads that upstream code can depend on.
• Safe to use without an API key (CoinGecko public). If they rate limit,
  we gracefully back off and surface partial results rather than crash.

Usage
-----
from trading_core.data_fetcher.adapters.coingecko import CoinGeckoAdapter

cg = CoinGeckoAdapter()
coins = cg.get_markets(vs_currency="usd", per_page=25)  # discovery + quotes
hist  = cg.get_market_chart("bitcoin", vs_currency="usd", days=15, interval="daily")
spot  = cg.simple_price(ids=["bitcoin","ethereum"], vs_currency="usd")
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_TIMEOUT  = 12   # seconds
MAX_RETRIES      = 3
BACKOFF_START_S  = 0.8  # exponential: 0.8, 1.6, 3.2 ...


class CoinGeckoHTTPError(RuntimeError):
    pass


class CoinGeckoAdapter:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self.session  = session or requests.Session()
        # sane UA reduces chance of anti-bot blocks
        self.session.headers.update(
            {"User-Agent": "TA/1.0 (+https://example.local) python-requests"}
        )

    # ────────────────────────────────────────────────────────────────────────────
    # Internal: request with retries/backoff
    # ────────────────────────────────────────────────────────────────────────────
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        HTTP request with exponential backoff on rate limits.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path (e.g., "/coins/markets")
            params: URL query parameters
            json_data: JSON request body
            
        Returns:
            Parsed JSON response
            
        Raises:
            CoinGeckoHTTPError: On persistent HTTP errors
        """
        url = f"{self.base_url}{path}"
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    json=json_data,
                    timeout=self.timeout,
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        backoff_time = BACKOFF_START_S * (2 ** attempt)
                        time.sleep(backoff_time)
                        continue
                    else:
                        raise CoinGeckoHTTPError(f"Rate limited after {MAX_RETRIES} attempts")
                
                # Handle other HTTP errors
                if not response.ok:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(BACKOFF_START_S)
                        continue
                    else:
                        raise CoinGeckoHTTPError(
                            f"HTTP {response.status_code}: {response.text[:200]}"
                        )
                
                # Success - parse JSON
                try:
                    return response.json()
                except ValueError as e:
                    raise CoinGeckoHTTPError(f"Invalid JSON response: {e}")
                    
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_START_S * (2 ** attempt))
                    continue
                else:
                    raise CoinGeckoHTTPError(f"Request failed after {MAX_RETRIES} attempts: {e}")
        
        # Should never reach here, but just in case
        raise CoinGeckoHTTPError("Unexpected error in request handling")

    # ────────────────────────────────────────────────────────────────────────────
    # Public API methods
    # ────────────────────────────────────────────────────────────────────────────
    
    def ping(self) -> bool:
        """
        Health check endpoint.
        
        Returns:
            True if CoinGecko API is responsive, False otherwise
        """
        try:
            response = self._request("GET", "/ping")
            return response.get("gecko_says") == "(V3) To the Moon!"
        except Exception:
            return False

    def get_markets(
        self,
        *,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: str = "24h",
    ) -> List[Dict[str, Any]]:
        """
        Get cryptocurrency market data (prices, market cap, volume, etc.)
        
        This is the main discovery endpoint that replaces the missing method.
        
        Args:
            vs_currency: Target currency for prices (default: "usd")
            order: Sort order (market_cap_desc, volume_desc, etc.)
            per_page: Number of results per page (1-250)
            page: Page number
            sparkline: Include 7-day price sparkline data
            price_change_percentage: Include price change % for timeframes
            
        Returns:
            List of coin market data dictionaries
        """
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": min(per_page, 250),  # API limit
            "page": page,
            "sparkline": "true" if sparkline else "false",
            "price_change_percentage": price_change_percentage,
        }
        
        response = self._request("GET", "/coins/markets", params=params)
        
        # Ensure we return a list even if API returns something unexpected
        if isinstance(response, list):
            return response
        else:
            return []

    def top_coins(
        self,
        *,
        vs_currency: str = "usd",
        per_page: int = 50,
        order: str = "volume_desc",
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to get top coins by volume.
        
        Args:
            vs_currency: Target currency for prices
            per_page: Number of coins to return
            order: Sort order for results
            
        Returns:
            List of top cryptocurrency market data
        """
        return self.get_markets(
            vs_currency=vs_currency,
            per_page=per_page,
            order=order,
            sparkline=False,
        )

    def get_market_chart(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: int = 30,
        interval: str = "daily",
        want: str = "prices",
    ) -> Dict[str, Any]:
        """
        Historical market data for a specific coin.
        
        Args:
            coin_id: CoinGecko coin identifier (e.g., "bitcoin")
            vs_currency: Target currency
            days: Number of days of historical data
            interval: Data interval ("daily", "hourly")
            want: Data type to extract ("prices", "market_caps", "total_volumes")
            
        Returns:
            Dictionary with coin_id, vs_currency, interval, and time series data
        """
        params = {
            "vs_currency": vs_currency,
            "days": max(1, days),
            "interval": interval,
        }
        
        data = self._request("GET", f"/coins/{coin_id}/market_chart", params=params)
        series = data.get(want, [])
        
        # Keep only last N points if caller passed non-sense days (server may return more)
        return {
            "coin_id": coin_id,
            "vs": vs_currency,
            "interval": interval,
            "series": [(int(t), float(v)) for t, v in series if isinstance(t, (int, float))],
        }

    def market_chart(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: int = 30,
        interval: str = "daily",
    ) -> Dict[str, Any]:
        """
        Alias for get_market_chart for backward compatibility.
        
        Args:
            coin_id: CoinGecko coin identifier
            vs_currency: Target currency
            days: Number of days of historical data
            interval: Data interval
            
        Returns:
            Historical price data
        """
        return self.get_market_chart(
            coin_id=coin_id,
            vs_currency=vs_currency,
            days=days,
            interval=interval,
            want="prices"
        )

    def simple_price(
        self,
        *,
        ids: List[str],
        vs_currency: str = "usd",
        include_market_cap: bool = False,
        include_24hr_vol: bool = False,
        include_24hr_change: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Quick spot price lookup for specific CoinGecko ids.
        
        Args:
            ids: List of CoinGecko coin IDs
            vs_currency: Target currency
            include_market_cap: Include market cap data
            include_24hr_vol: Include 24h volume data
            include_24hr_change: Include 24h price change data
            
        Returns:
            Dictionary mapping coin_id to price data
        """
        if not ids:
            return {}
            
        params = {
            "ids": ",".join(ids),
            "vs_currencies": vs_currency,
            "include_market_cap": "true" if include_market_cap else "false",
            "include_24hr_vol": "true" if include_24hr_vol else "false",
            "include_24hr_change": "true" if include_24hr_change else "false",
        }
        
        raw = self._request("GET", "/simple/price", params=params)
        out: Dict[str, Dict[str, Any]] = {}
        
        for cid, payload in (raw or {}).items():
            rec: Dict[str, Any] = {"price": payload.get(vs_currency)}
            if include_market_cap:
                rec["market_cap"] = payload.get(f"{vs_currency}_market_cap")
            if include_24hr_vol:
                rec["volume_24h"] = payload.get(f"{vs_currency}_24h_vol")
            if include_24hr_change:
                rec["change_24h_pct"] = payload.get(f"{vs_currency}_24h_change")
            out[cid] = rec
            
        return out

    def get_coin(
        self,
        coin_id: str,
        *,
        localization: bool = False,
        tickers: bool = False,
        market_data: bool = True,
        community_data: bool = False,
        developer_data: bool = False,
        sparkline: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific coin.
        
        Args:
            coin_id: CoinGecko coin identifier
            localization: Include localized text
            tickers: Include ticker data
            market_data: Include market data
            community_data: Include community statistics
            developer_data: Include developer statistics
            sparkline: Include price sparkline
            
        Returns:
            Detailed coin information or None if not found
        """
        params = {
            "localization": "true" if localization else "false",
            "tickers": "true" if tickers else "false",
            "market_data": "true" if market_data else "false",
            "community_data": "true" if community_data else "false",
            "developer_data": "true" if developer_data else "false",
            "sparkline": "true" if sparkline else "false",
        }
        
        try:
            return self._request("GET", f"/coins/{coin_id}", params=params)
        except CoinGeckoHTTPError:
            return None

    def search(self, query: str) -> Dict[str, Any]:
        """
        Search for coins, exchanges, and categories.
        
        Args:
            query: Search query string
            
        Returns:
            Search results with coins, exchanges, and categories
        """
        params = {"query": query}
        return self._request("GET", "/search", params=params)

    def get_coins_list(self, include_platform: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of all supported coins with id, name, and symbol.
        
        Args:
            include_platform: Include platform contract addresses
            
        Returns:
            List of all supported coins
        """
        params = {"include_platform": "true" if include_platform else "false"}
        return self._request("GET", "/coins/list", params=params)

    # ────────────────────────────────────────────────────────────────────────────
    # Utility methods
    # ────────────────────────────────────────────────────────────────────────────
    
    def is_healthy(self) -> bool:
        """
        Check if the CoinGecko API is accessible and responsive.
        
        Returns:
            True if API is healthy, False otherwise
        """
        return self.ping()

    def get_supported_currencies(self) -> List[str]:
        """
        Get list of supported vs_currencies.
        
        Returns:
            List of supported currency codes
        """
        try:
            response = self._request("GET", "/simple/supported_vs_currencies")
            return response if isinstance(response, list) else []
        except Exception:
            # Return common currencies as fallback
            return ["usd", "eur", "btc", "eth", "jpy", "gbp", "aud", "cad", "chf", "cny"]

    def normalize_coin_id(self, symbol_or_id: str) -> Optional[str]:
        """
        Attempt to normalize a symbol or partial ID to a valid CoinGecko coin ID.
        
        Args:
            symbol_or_id: Coin symbol (e.g., "BTC") or partial ID
            
        Returns:
            Valid CoinGecko coin ID if found, None otherwise
        """
        # Common symbol to CoinGecko ID mappings
        common_mappings = {
            "btc": "bitcoin",
            "eth": "ethereum", 
            "usdt": "tether",
            "bnb": "binancecoin",
            "sol": "solana",
            "ada": "cardano",
            "xrp": "ripple",
            "dot": "polkadot",
            "doge": "dogecoin",
            "avax": "avalanche-2",
            "shib": "shiba-inu",
            "link": "chainlink",
            "matic": "matic-network",
            "uni": "uniswap",
            "ltc": "litecoin",
            "atom": "cosmos",
            "xlm": "stellar",
            "algo": "algorand",
            "vet": "vechain",
            "icp": "internet-computer",
            "fil": "filecoin",
            "trx": "tron",
            "etc": "ethereum-classic",
            "ftt": "ftx-token",
            "hbar": "hedera-hashgraph",
            "aave": "aave",
            "xmr": "monero",
            "eos": "eos",
            "bch": "bitcoin-cash",
            "flow": "flow",
            "mana": "decentraland",
            "sand": "the-sandbox",
            "cake": "pancakeswap-token",
            "gala": "gala",
            "axs": "axie-infinity",
            "theta": "theta-token",
        }
        
        normalized = symbol_or_id.lower().strip()
        
        # Check if it's a known symbol mapping
        if normalized in common_mappings:
            return common_mappings[normalized]
        
        # If it looks like it might already be a CoinGecko ID, return as-is
        if len(normalized) > 3 and "-" in normalized:
            return normalized
        
        # Try search API as last resort (but don't fail if it doesn't work)
        try:
            search_result = self.search(symbol_or_id)
            coins = search_result.get("coins", [])
            if coins:
                # Return the first match
                return coins[0].get("id")
        except Exception:
            pass
        
        return None
