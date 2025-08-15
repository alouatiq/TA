from __future__ import annotations

"""
crypto.py
─────────
Discovery + quotes for cryptocurrencies with layered fallbacks:

Order:
  1) CoinGecko (primary; discovery + price + history)
  2) CoinPaprika (fallback; discovery + price; history best-effort)
  3) CoinCap (fallback; discovery + price; history best-effort)
  4) CryptoCompare (fallback; discovery + price + history; requires API key)
  5) TwelveData (final fallback; price + history; requires API key and symbol normalization)

Outputs a list[dict] for top liquid assets:
  {
    "asset": "<coingecko_id or symbol>",
    "symbol": "BTC-USD",            # standardized display pair
    "price":  62850.24,
    "volume": 123456789.0,
    "day_range_pct": 4.21,          # (high - low) / price * 100 when available; else 24h pct change
    "price_history": [ ... ]        # optional; daily closes (oldest→newest); len ≈ history_days+1
  }

Usage:
  rows = fetch_crypto_data(include_history=True, limit=20, history_days=60)

Diagnostics:
  LAST_CRYPTO_SOURCE: str
  FAILED_CRYPTO_SOURCES: list[str]
  SKIPPED_CRYPTO_SOURCES: list[str]

Notes:
  • No hard-coded coin lists – discovery comes from providers.
  • History is best-effort; depths vary by provider and plan limits.
  • TwelveData expects symbols like "BTC/USD" (we normalize to "BTC-USD" for display).
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone

import requests

# Adapters (we built these in adapters/)
try:
    from .adapters.coingecko import CoinGeckoAdapter as CoinGeckoClient
except ImportError:
    CoinGeckoClient = None

try:
    from .adapters.cryptocompare import CryptoCompareAdapter as CryptoCompareClient
except ImportError:
    CryptoCompareClient = None

try:
    from .adapters.twelvedata import TwelveDataAdapter as TwelveDataClient  
except ImportError:
    TwelveDataClient = None

REQUEST_TIMEOUT = 15
DEFAULT_LIMIT = 20

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_CRYPTO_SOURCE: str = "None"
FAILED_CRYPTO_SOURCES: List[str] = []
SKIPPED_CRYPTO_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _std_pair(symbol: str, quote: str = "USD") -> str:
    """Standardize display pair as 'BTC-USD'."""
    return f"{symbol.upper()}-{quote.upper()}"

def _as_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _now_ms() -> int:
    return int(time.time() * 1000)

def _ms_days_ago(days: int) -> int:
    return _now_ms() - days * 24 * 3600 * 1000


# ────────────────────────────────────────────────────────────
# 1) CoinGecko – primary
# ────────────────────────────────────────────────────────────
def _from_coingecko(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    cg = CoinGeckoClient()
    markets = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit, page=1, sparkline=False)
    rows: List[Dict[str, Any]] = []

    for coin in markets:
        price = _as_float(coin.get("current_price"))
        high  = _as_float(coin.get("high_24h"))
        low   = _as_float(coin.get("low_24h"))
        vol   = _as_float(coin.get("total_volume")) or 0.0
        sym   = (coin.get("symbol") or "").upper()
        cid   = coin.get("id") or sym

        day_range_pct: float = 0.0
        if price and high is not None and low is not None and price != 0:
            try:
                day_range_pct = round(((high - low) / price) * 100.0, 2)
            except Exception:
                day_range_pct = 0.0

        item: Dict[str, Any] = {
            "asset": cid,
            "symbol": _std_pair(sym),
            "price": price or 0.0,
            "volume": vol,
            "day_range_pct": day_range_pct,
        }

        if include_history and cid:
            try:
                chart = cg.get_market_chart(cid, vs_currency="usd", days=history_days + 1, interval="daily")
                prices = [p[1] for p in chart.get("prices", [])]
                if prices:
                    item["price_history"] = prices[-(history_days + 1):]
            except Exception:
                # history optional; ignore
                pass

        rows.append(item)

    return rows


# ────────────────────────────────────────────────────────────
# 2) CoinPaprika – fallback
# ────────────────────────────────────────────────────────────
def _from_coinpaprika(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    url = "https://api.coinpaprika.com/v1/tickers"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # sort by USD volume if available
    def usd_vol(x: Dict[str, Any]) -> float:
        return float((((x.get("quotes") or {}).get("USD") or {}).get("volume_24h")) or 0.0)

    top = sorted(data, key=usd_vol, reverse=True)[:limit]
    rows: List[Dict[str, Any]] = []

    for c in top:
        q = (c.get("quotes") or {}).get("USD") or {}
        price = _as_float(q.get("price")) or 0.0
        vol   = _as_float(q.get("volume_24h")) or 0.0
        pct24 = _as_float(q.get("percent_change_24h"))
        sym   = (c.get("symbol") or "").upper()
        cid   = c.get("id") or sym

        day_range_pct = 0.0
        if pct24 is not None:
            # if we don't have H/L, use 24h pct move as a proxy (not identical)
            day_range_pct = round(float(pct24), 2)

        item: Dict[str, Any] = {
            "asset": cid,
            "symbol": _std_pair(sym),
            "price": price,
            "volume": vol,
            "day_range_pct": day_range_pct,
        }

        if include_history and cid:
            try:
                # best-effort historical endpoint
                start = (datetime.now(timezone.utc) - timedelta(days=history_days + 2)).strftime("%Y-%m-%d")
                hurl = f"https://api.coinpaprika.com/v1/tickers/{cid}/historical?start={start}&interval=1d"
                hresp = requests.get(hurl, timeout=REQUEST_TIMEOUT)
                if hresp.ok:
                    series = hresp.json()
                    closes = [float(x["price"]) for x in series if "price" in x]
                    if closes:
                        item["price_history"] = closes[-(history_days + 1):]
            except Exception:
                pass

        rows.append(item)

    return rows


# ────────────────────────────────────────────────────────────
# 3) CoinCap – fallback
# ────────────────────────────────────────────────────────────
def _from_coincap(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    base = "https://api.coincap.io/v2"
    resp = requests.get(f"{base}/assets", params={"limit": limit}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = (resp.json() or {}).get("data", [])
    rows: List[Dict[str, Any]] = []

    for c in data:
        sym = (c.get("symbol") or "").upper()
        cid = (c.get("id") or sym)
        price = _as_float(c.get("priceUsd")) or 0.0
        vol   = _as_float(c.get("volumeUsd24Hr")) or 0.0
        pct24 = _as_float(c.get("changePercent24Hr"))
        day_range_pct = round(pct24, 2) if pct24 is not None else 0.0

        item: Dict[str, Any] = {
            "asset": cid,
            "symbol": _std_pair(sym),
            "price": price,
            "volume": vol,
            "day_range_pct": day_range_pct,
        }

        if include_history and cid:
            try:
                end = _now_ms()
                start = _ms_days_ago(history_days + 2)
                hresp = requests.get(
                    f"{base}/assets/{cid}/history",
                    params={"interval": "d1", "start": start, "end": end},
                    timeout=REQUEST_TIMEOUT,
                )
                if hresp.ok:
                    series = (hresp.json() or {}).get("data", [])
                    closes = [float(x["priceUsd"]) for x in series if "priceUsd" in x]
                    if closes:
                        item["price_history"] = closes[-(history_days + 1):]
            except Exception:
                pass

        rows.append(item)

    return rows


# ────────────────────────────────────────────────────────────
# 4) CryptoCompare – fallback (API key)
# ────────────────────────────────────────────────────────────
def _from_cryptocompare(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    key = os.getenv("CRYPTOCOMPARE_API_KEY")
    if not key:
        raise RuntimeError("CRYPTOCOMPARE_API_KEY not set")

    cc = CryptoCompareClient(api_key=key)

    # Discovery: top by total volume (vs USD)
    top = cc.top_by_total_volume_full(tsym="USD", limit=limit)
    # top entries include coin info under 'CoinInfo' (Name symbol, FullName, Id, ...)

    symbols: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for e in top:
        info = e.get("CoinInfo") or {}
        name = (info.get("Name") or "").upper()  # e.g., BTC
        if not name:
            continue
        symbols.append(name)
        meta[name] = info

    if not symbols:
        return []

    # Prices (RAW)
    quotes = cc.price_multi_full(fsyms=",".join(symbols), tsyms="USD")  # RAW + DISPLAY

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            raw = (quotes.get("RAW") or {}).get(sym, {}).get("USD", {})
            price = _as_float(raw.get("PRICE")) or 0.0
            vol   = _as_float(raw.get("TOTALVOLUME24H")) or 0.0
            high  = _as_float(raw.get("HIGH24HOUR"))
            low   = _as_float(raw.get("LOW24HOUR"))
            day_range_pct = 0.0
            if price and high is not None and low is not None and price != 0:
                day_range_pct = round(((high - low) / price) * 100.0, 2)

            item: Dict[str, Any] = {
                "asset": sym.lower(),
                "symbol": _std_pair(sym),
                "price": price,
                "volume": vol,
                "day_range_pct": day_range_pct,
            }

            if include_history:
                try:
                    # histoday returns last N daily candles (limit ≈ count - 1); we add buffer
                    series = cc.histoday(fsym=sym, tsym="USD", limit=history_days + 2)
                    closes = [float(x["close"]) for x in series if "close" in x]
                    if closes:
                        item["price_history"] = closes[-(history_days + 1):]
                except Exception:
                    pass

            rows.append(item)
        except Exception:
            continue

    return rows


# ────────────────────────────────────────────────────────────
# 5) TwelveData – final fallback (API key)
# ────────────────────────────────────────────────────────────
def _from_twelvedata(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    td_key = os.getenv("TWELVEDATA_API_KEY")
    if not td_key:
        raise RuntimeError("TWELVEDATA_API_KEY not set")

    td = TwelveDataClient(api_key=td_key)

    # Discovery on TwelveData is not as straightforward. As a pragmatic fallback:
    #  - use a small, *dynamic* top set from CoinGecko discovery first (symbols only),
    #    then fetch via TwelveData. If CoinGecko discovery fails too, fallback to commonly
    #    traded majors as last resort (kept to a tiny set to avoid "hard coding" bias).
    symbols: List[str] = []
    try:
        cg = CoinGeckoClient()
        mk = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit, page=1, sparkline=False)
        symbols = [(c.get("symbol") or "").upper() for c in mk if c.get("symbol")]
    except Exception:
        pass

    if not symbols:
        # tiny safe default set if everything else fails (still discovery-light)
        symbols = ["BTC", "ETH", "XRP", "SOL", "ADA"][:limit]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        pair_td = f"{sym}/USD"       # TwelveData format
        pair_ui = _std_pair(sym)     # display format

        try:
            q = td.get_quote(pair_td)
            price = _as_float(q.get("price")) or 0.0
            vol   = _as_float(q.get("volume")) or 0.0

            # No explicit high/low for day in /quote in all cases; estimate day_range_pct from last 1d series
            day_range_pct = 0.0

            item: Dict[str, Any] = {
                "asset": sym.lower(),
                "symbol": pair_ui,
                "price": price,
                "volume": vol,
                "day_range_pct": day_range_pct,
            }

            if include_history:
                try:
                    ts = td.get_time_series(pair_td, interval="1day", outputsize=min(200, history_days + 50))
                    closes = [float(x["close"]) for x in ts][- (history_days + 1):]
                    if closes:
                        item["price_history"] = closes
                    # compute day_range_pct from the last bar if we have OHLC
                    if ts:
                        last = ts[-1]
                        h = _as_float(last.get("high"))
                        l = _as_float(last.get("low"))
                        c = _as_float(last.get("close"))
                        if h and l and c and c != 0:
                            item["day_range_pct"] = round(((h - l) / c) * 100.0, 2)
                except Exception:
                    pass

            rows.append(item)
        except Exception:
            continue

    return rows


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_crypto_data(
    include_history: bool = False,
    *,
    limit: int = DEFAULT_LIMIT,
    history_days: int = 60
) -> List[Dict[str, Any]]:
    """
    Fetch top liquid cryptocurrencies with robust fallbacks and optional daily history.

    Args:
        include_history: whether to include 'price_history' (daily closes, ~history_days+1 points).
        limit: number of assets to return (best-effort per provider).
        history_days: number of daily bars to fetch for indicators.

    Returns:
        list of asset dicts (see module docstring).
    """
    global LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
    FAILED_CRYPTO_SOURCES = []
    SKIPPED_CRYPTO_SOURCES = []
    LAST_CRYPTO_SOURCE = "None"

    # Try providers in order, capturing diagnostics
    # 1) CoinGecko
    try:
        rows = _from_coingecko(limit, include_history, history_days)
        if rows:
            LAST_CRYPTO_SOURCE = "CoinGecko"
            SKIPPED_CRYPTO_SOURCES = ["CoinPaprika", "CoinCap", "CryptoCompare", "TwelveData"]
            return rows
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"CoinGecko ({e})")

    # 2) CoinPaprika
    try:
        rows = _from_coinpaprika(limit, include_history, history_days)
        if rows:
            LAST_CRYPTO_SOURCE = "CoinPaprika"
            SKIPPED_CRYPTO_SOURCES = ["CoinCap", "CryptoCompare", "TwelveData"]
            return rows
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"CoinPaprika ({e})")

    # 3) CoinCap
    try:
        rows = _from_coincap(limit, include_history, history_days)
        if rows:
            LAST_CRYPTO_SOURCE = "CoinCap"
            SKIPPED_CRYPTO_SOURCES = ["CryptoCompare", "TwelveData"]
            return rows
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"CoinCap ({e})")

    # 4) CryptoCompare (API key)
    try:
        rows = _from_cryptocompare(limit, include_history, history_days)
        if rows:
            LAST_CRYPTO_SOURCE = "CryptoCompare"
            SKIPPED_CRYPTO_SOURCES = ["TwelveData"]
            return rows
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"CryptoCompare ({e})")

    # 5) TwelveData (API key)
    try:
        rows = _from_twelvedata(limit, include_history, history_days)
        if rows:
            LAST_CRYPTO_SOURCE = "TwelveData"
            return rows
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"TwelveData ({e})")

    return []
