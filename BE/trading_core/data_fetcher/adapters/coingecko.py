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
coins = cg.top_coins(vs_currency="usd", per_page=25)  # discovery + quotes
hist  = cg.market_chart("bitcoin", vs_currency="usd", days=15, interval="daily")
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

    # ──────────────────────────────────────────────────────
    # Internal: request with retries/backoff
    # ──────────────────────────────────────────────────────
    def _request(self, method: str, path: str, *, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        backoff = BACKOFF_START_S
        last_exc: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.request(
                    method.upper(), url, params=params, timeout=self.timeout
                )
                # Basic retry on 429 / 5xx
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    # Try to respect Retry-After when present
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait_s = float(ra)
                        except Exception:
                            wait_s = backoff
                    else:
                        wait_s = backoff
                    time.sleep(wait_s)
                    backoff *= 2
                    continue

                if resp.status_code != 200:
                    raise CoinGeckoHTTPError(
                        f"HTTP {resp.status_code} on {path} (params={params})"
                    )

                # OK
                return resp.json()

            except (requests.Timeout, requests.ConnectionError) as e:
                last_exc = e
                time.sleep(backoff)
                backoff *= 2
                continue
            except ValueError as e:
                # JSON decode error → don't retry infinitely
                raise CoinGeckoHTTPError(f"Invalid JSON from {path}: {e}") from e

        if last_exc:
            raise CoinGeckoHTTPError(
                f"Failed after {MAX_RETRIES} attempts for {path}: {last_exc}"
            )
        raise CoinGeckoHTTPError(f"Failed after {MAX_RETRIES} attempts for {path}")

    # ──────────────────────────────────────────────────────
    # Public endpoints
    # ──────────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            data = self._request("GET", "/ping")
            return bool(data and data.get("gecko_says"))
        except Exception:
            return False

    def top_coins(
        self,
        *,
        vs_currency: str = "usd",
        order: str = "volume_desc",   # also: "market_cap_desc"
        per_page: int = 30,
        page: int = 1,
        sparkline: bool = False,
        price_change_percentage: Optional[str] = None,  # e.g., "1h,24h,7d"
    ) -> List[Dict[str, Any]]:
        """
        Discover + quote top coins. Normalized keys:
        - id, symbol, name
        - price (current_price), volume (total_volume), market_cap
        - high_24h, low_24h, day_range_pct (computed)
        - price_change_pct_* (when requested)
        """
        params = {
            "vs_currency": vs_currency,
            "order": order,
            "per_page": max(1, min(per_page, 250)),
            "page": max(1, page),
            "sparkline": "true" if sparkline else "false",
        }
        if price_change_percentage:
            params["price_change_percentage"] = price_change_percentage

        raw = self._request("GET", "/coins/markets", params=params)
        out: List[Dict[str, Any]] = []
        for row in raw or []:
            price = row.get("current_price")
            high  = row.get("high_24h")
            low   = row.get("low_24h")
            try:
                drp = round(((high - low) / price) * 100, 2) if price and high and low and price != 0 else None
            except Exception:
                drp = None

            rec: Dict[str, Any] = {
                "id": row.get("id"),
                "symbol": (row.get("symbol") or "").upper(),
                "name": row.get("name"),
                "price": price,
                "volume": row.get("total_volume", 0),
                "market_cap": row.get("market_cap"),
                "high_24h": high,
                "low_24h": low,
                "day_range_pct": drp,
            }
            if price_change_percentage:
                # CoinGecko returns e.g. "price_change_percentage_24h_in_currency"
                for horizon in (h.strip() for h in price_change_percentage.split(",") if h.strip()):
                    key = f"price_change_percentage_{horizon}_in_currency"
                    if key in row:
                        rec[f"price_change_pct_{horizon}"] = row.get(key)
            out.append(rec)
        return out

    def market_chart(
        self,
        coin_id: str,
        *,
        vs_currency: str = "usd",
        days: int = 30,
        interval: str = "daily",
        want: str = "prices",  # "prices" | "market_caps" | "total_volumes"
    ) -> Dict[str, Any]:
        """
        Get simple arrays of [timestamp, value] for the desired series.
        - interval: "daily" is great for indicators (RSI-14 etc.)
        - days: 1..max
        Returns: {"series": [(ts_ms, value), ...], "coin_id": coin_id, "vs": vs_currency, "interval": interval}
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
        Returns a dict: {id: {"price": float, "market_cap":?, "volume_24h":?, "change_24h_pct":?}}
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
