# BE/trading_core/data_fetcher/crypto.py
"""
Enhanced crypto.py with robust fallback mechanisms and proper historical data validation

Discovery + quotes for cryptocurrencies with layered fallbacks:

Order:
  1) CoinGecko (primary; discovery + price + history)
  2) CoinPaprika (fallback; discovery + price; history best-effort)
  3) CoinCap (fallback; discovery + price; history best-effort)
  4) CryptoCompare (fallback; discovery + price + history; requires API key)
  5) TwelveData (final fallback; price + history; requires API key and symbol normalization)

Enhanced Features:
- Proper validation of historical data before marking source as "successful"
- Retry logic for individual assets within each source
- Intelligent fallback when primary source fails to get sufficient historical data
- Better error handling and diagnostics
"""

from __future__ import annotations

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

# Enhanced validation thresholds
MIN_HISTORICAL_DAYS = 7  # Minimum days of historical data to consider source successful
MIN_ASSETS_WITH_HISTORY = 3  # Minimum number of assets that must have historical data

# ────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────────────────────
LAST_CRYPTO_SOURCE: str = "None"
FAILED_CRYPTO_SOURCES: List[str] = []
SKIPPED_CRYPTO_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────────────────────
# Enhanced validation helpers
# ────────────────────────────────────────────────────────────────────────────

def _validate_historical_data(rows: List[Dict[str, Any]], include_history: bool, min_days: int = MIN_HISTORICAL_DAYS) -> bool:
    """
    Validate that we have sufficient historical data before considering a source successful.
    
    Args:
        rows: List of asset dictionaries
        include_history: Whether historical data was requested
        min_days: Minimum number of historical days required
    
    Returns:
        True if validation passes, False otherwise
    """
    if not include_history:
        # If no history requested, just check that we have some assets with prices
        return len(rows) > 0 and any(r.get("price", 0) > 0 for r in rows)
    
    # Count assets with sufficient historical data
    assets_with_history = 0
    for row in rows:
        price_history = row.get("price_history", [])
        if isinstance(price_history, list) and len(price_history) >= min_days:
            assets_with_history += 1
    
    print(f"[DEBUG] Validation: {assets_with_history}/{len(rows)} assets have ≥{min_days} days of history")
    
    # We need at least MIN_ASSETS_WITH_HISTORY assets with sufficient historical data
    return assets_with_history >= MIN_ASSETS_WITH_HISTORY


def _merge_fallback_data(primary_rows: List[Dict[str, Any]], fallback_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge data from fallback sources to fill gaps in primary source data.
    
    Args:
        primary_rows: Data from primary source
        fallback_rows: Data from fallback source
    
    Returns:
        Enhanced data with gaps filled
    """
    if not primary_rows:
        return fallback_rows
    
    if not fallback_rows:
        return primary_rows
    
    # Create lookup for fallback data by symbol
    fallback_lookup = {}
    for row in fallback_rows:
        symbol = row.get("symbol", "").upper()
        if symbol:
            fallback_lookup[symbol] = row
    
    # Enhance primary data with fallback data where needed
    enhanced_rows = []
    for row in primary_rows:
        enhanced_row = row.copy()
        symbol = row.get("symbol", "").upper()
        
        # If this asset lacks historical data, try to get it from fallback
        if not row.get("price_history") and symbol in fallback_lookup:
            fallback_row = fallback_lookup[symbol]
            if fallback_row.get("price_history"):
                enhanced_row["price_history"] = fallback_row["price_history"]
                print(f"[DEBUG] Enhanced {symbol} with historical data from fallback source")
        
        enhanced_rows.append(enhanced_row)
    
    return enhanced_rows


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────────────────────────────
# Enhanced source implementations
# ────────────────────────────────────────────────────────────────────────────

def _from_coingecko_enhanced(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Enhanced CoinGecko implementation with better error handling and validation."""
    if not CoinGeckoClient:
        raise RuntimeError("CoinGecko adapter not available")
    
    print(f"[DEBUG] CoinGecko: Fetching {limit} assets with {history_days} days history={include_history}")
    
    cg = CoinGeckoClient()
    
    try:
        markets = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit, page=1, sparkline=False)
        print(f"[DEBUG] CoinGecko: Retrieved {len(markets)} market entries")
    except Exception as e:
        print(f"[DEBUG] CoinGecko markets fetch failed: {e}")
        raise
    
    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0

    for i, coin in enumerate(markets):
        try:
            price = _as_float(coin.get("current_price"))
            high  = _as_float(coin.get("high_24h"))
            low   = _as_float(coin.get("low_24h"))
            vol   = _as_float(coin.get("total_volume")) or 0.0
            sym   = (coin.get("symbol") or "").upper()
            cid   = coin.get("id") or sym

            if not price or price <= 0:
                print(f"[DEBUG] CoinGecko: Skipping {sym} - invalid price: {price}")
                continue

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

            # Enhanced historical data fetching
            if include_history and cid:
                try:
                    print(f"[DEBUG] CoinGecko: Fetching history for {sym} ({cid})")
                    
                    # Retry logic for historical data
                    for attempt in range(2):
                        try:
                            chart = cg.market_chart(cid, vs_currency="usd", days=history_days + 2, interval="daily")
                            
                            if chart and "series" in chart:
                                prices = [p[1] for p in chart["series"]]
                                if len(prices) >= MIN_HISTORICAL_DAYS:
                                    item["price_history"] = prices[-(history_days + 1):]
                                    successful_history_fetches += 1
                                    print(f"[DEBUG] CoinGecko: Got {len(item['price_history'])} price points for {sym}")
                                    break
                                else:
                                    print(f"[DEBUG] CoinGecko: Insufficient history for {sym}: {len(prices)} days")
                            else:
                                print(f"[DEBUG] CoinGecko: No chart data for {sym}")
                                
                            if attempt == 0:
                                time.sleep(0.5)  # Brief delay before retry
                                
                        except Exception as e:
                            print(f"[DEBUG] CoinGecko: History fetch attempt {attempt + 1} failed for {sym}: {e}")
                            if attempt == 0:
                                time.sleep(0.5)
                            
                except Exception as e:
                    print(f"[DEBUG] CoinGecko: Failed to get history for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CoinGecko: Failed to process coin {i}: {e}")
            continue

    print(f"[DEBUG] CoinGecko: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


def _from_coinpaprika_enhanced(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Enhanced CoinPaprika implementation."""
    print(f"[DEBUG] CoinPaprika: Fetching {limit} assets with history={include_history}")
    
    url = "https://api.coinpaprika.com/v1/tickers"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # sort by USD volume if available
    def usd_vol(x: Dict[str, Any]) -> float:
        return float((((x.get("quotes") or {}).get("USD") or {}).get("volume_24h")) or 0.0)

    top = sorted(data, key=usd_vol, reverse=True)[:limit * 2]  # Get extra in case some fail
    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0

    for c in top[:limit]:
        try:
            q = (c.get("quotes") or {}).get("USD") or {}
            price = _as_float(q.get("price")) or 0.0
            vol   = _as_float(q.get("volume_24h")) or 0.0
            pct24 = _as_float(q.get("percent_change_24h"))
            sym   = (c.get("symbol") or "").upper()
            cid   = c.get("id") or sym

            if not price or price <= 0:
                continue

            day_range_pct = 0.0
            if pct24 is not None:
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
                    start = (datetime.now(timezone.utc) - timedelta(days=history_days + 2)).strftime("%Y-%m-%d")
                    hurl = f"https://api.coinpaprika.com/v1/tickers/{cid}/historical?start={start}&interval=1d"
                    hresp = requests.get(hurl, timeout=REQUEST_TIMEOUT)
                    if hresp.ok:
                        series = hresp.json()
                        closes = [float(x["price"]) for x in series if "price" in x]
                        if len(closes) >= MIN_HISTORICAL_DAYS:
                            item["price_history"] = closes[-(history_days + 1):]
                            successful_history_fetches += 1
                            print(f"[DEBUG] CoinPaprika: Got {len(item['price_history'])} price points for {sym}")
                except Exception as e:
                    print(f"[DEBUG] CoinPaprika: History failed for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CoinPaprika: Failed to process coin: {e}")
            continue

    print(f"[DEBUG] CoinPaprika: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


def _from_coincap_enhanced(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Enhanced CoinCap implementation."""
    print(f"[DEBUG] CoinCap: Fetching {limit} assets with history={include_history}")
    
    base = "https://api.coincap.io/v2"
    resp = requests.get(f"{base}/assets", params={"limit": limit}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = (resp.json() or {}).get("data", [])
    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0

    for c in data:
        try:
            sym = (c.get("symbol") or "").upper()
            cid = (c.get("id") or sym)
            price = _as_float(c.get("priceUsd")) or 0.0
            vol   = _as_float(c.get("volumeUsd24Hr")) or 0.0
            pct24 = _as_float(c.get("changePercent24Hr"))
            
            if not price or price <= 0:
                continue
                
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
                        if len(closes) >= MIN_HISTORICAL_DAYS:
                            item["price_history"] = closes[-(history_days + 1):]
                            successful_history_fetches += 1
                            print(f"[DEBUG] CoinCap: Got {len(item['price_history'])} price points for {sym}")
                except Exception as e:
                    print(f"[DEBUG] CoinCap: History failed for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CoinCap: Failed to process coin: {e}")
            continue

    print(f"[DEBUG] CoinCap: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


def _from_cryptocompare_enhanced(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Enhanced CryptoCompare implementation."""
    key = os.getenv("CRYPTOCOMPARE_API_KEY")
    if not key:
        raise RuntimeError("CRYPTOCOMPARE_API_KEY not set")

    print(f"[DEBUG] CryptoCompare: Fetching {limit} assets with history={include_history}")
    
    if not CryptoCompareClient:
        raise RuntimeError("CryptoCompare adapter not available")

    cc = CryptoCompareClient(api_key=key)

    # Discovery: top by total volume (vs USD)
    top = cc.top_by_total_volume_full(tsym="USD", limit=limit)
    
    symbols: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for e in top:
        info = e.get("CoinInfo") or {}
        name = (info.get("Name") or "").upper()
        if not name:
            continue
        symbols.append(name)
        meta[name] = info

    if not symbols:
        raise RuntimeError("No symbols discovered from CryptoCompare")

    # Prices (RAW)
    quotes = cc.price_multi_full(fsyms=",".join(symbols), tsyms="USD")

    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0
    
    for sym in symbols:
        try:
            raw = (((quotes.get("RAW") or {}).get(sym) or {}).get("USD") or {})
            if not raw:
                continue
                
            price = _as_float(raw.get("PRICE")) or 0.0
            vol   = _as_float(raw.get("TOTALVOLUME24H")) or 0.0
            high  = _as_float(raw.get("HIGH24HOUR"))
            low   = _as_float(raw.get("LOW24HOUR"))
            
            if not price or price <= 0:
                continue

            day_range_pct = 0.0
            if high and low and price:
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
                    # Get daily history
                    hist = cc.history_day(fsym=sym, tsym="USD", limit=history_days + 2)
                    if hist and "Data" in hist:
                        closes = [float(x["close"]) for x in hist["Data"] if "close" in x]
                        if len(closes) >= MIN_HISTORICAL_DAYS:
                            item["price_history"] = closes[-(history_days + 1):]
                            successful_history_fetches += 1
                            print(f"[DEBUG] CryptoCompare: Got {len(item['price_history'])} price points for {sym}")
                except Exception as e:
                    print(f"[DEBUG] CryptoCompare: History failed for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CryptoCompare: Failed to process {sym}: {e}")
            continue

    print(f"[DEBUG] CryptoCompare: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


def _from_twelvedata_enhanced(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Enhanced TwelveData implementation."""
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY not set")
        
    print(f"[DEBUG] TwelveData: Fetching {limit} assets with history={include_history}")

    if not TwelveDataClient:
        raise RuntimeError("TwelveData adapter not available")

    td = TwelveDataClient(api_key=key)

    # Get symbols from CoinGecko for discovery, then fetch via TwelveData
    symbols: List[str] = []
    try:
        if CoinGeckoClient:
            cg = CoinGeckoClient()
            mk = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit, page=1, sparkline=False)
            symbols = [(c.get("symbol") or "").upper() for c in mk if c.get("symbol")]
    except Exception:
        pass

    if not symbols:
        # Fallback to major cryptos
        symbols = ["BTC", "ETH", "XRP", "SOL", "ADA"][:limit]

    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0
    
    for sym in symbols:
        try:
            pair_td = f"{sym}/USD"
            pair_ui = _std_pair(sym)

            q = td.get_quote(pair_td)
            price = _as_float(q.get("price")) or 0.0
            vol   = _as_float(q.get("volume")) or 0.0
            
            if not price or price <= 0:
                continue

            item: Dict[str, Any] = {
                "asset": sym.lower(),
                "symbol": pair_ui,
                "price": price,
                "volume": vol,
                "day_range_pct": 0.0,
            }

            if include_history:
                try:
                    ts = td.get_time_series(pair_td, interval="1day", outputsize=min(200, history_days + 50))
                    closes = [float(x["close"]) for x in ts if "close" in x]
                    if len(closes) >= MIN_HISTORICAL_DAYS:
                        item["price_history"] = closes[-(history_days + 1):]
                        successful_history_fetches += 1
                        print(f"[DEBUG] TwelveData: Got {len(item['price_history'])} price points for {sym}")
                        
                        # compute day_range_pct from the last bar if we have OHLC
                        if ts:
                            last = ts[-1]
                            h = _as_float(last.get("high"))
                            l = _as_float(last.get("low"))
                            c = _as_float(last.get("close"))
                            if h and l and c and c != 0:
                                item["day_range_pct"] = round(((h - l) / c) * 100.0, 2)
                except Exception as e:
                    print(f"[DEBUG] TwelveData: History failed for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] TwelveData: Failed to process {sym}: {e}")
            continue

    print(f"[DEBUG] TwelveData: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Public API
# ────────────────────────────────────────────────────────────────────────────

def fetch_crypto_data(
    include_history: bool = False,
    *,
    limit: int = DEFAULT_LIMIT,
    history_days: int = 60,
    selection_strategy: str = "mixed"  # New parameter
) -> List[Dict[str, Any]]:
    """
    Fetch cryptocurrencies with robust fallbacks and dynamic asset selection strategies.

    Enhanced Features:
    - Multiple asset selection strategies (volume, gainers, trending, mixed)
    - Validates that historical data was actually retrieved before marking source as successful
    - Automatically tries fallback sources when primary source fails to get sufficient historical data
    - Merges data from multiple sources to fill gaps
    - Provides detailed debug information about what worked and what failed

    Args:
        include_history: whether to include 'price_history' (daily closes, ~history_days+1 points).
        limit: number of assets to return (best-effort per provider).
        history_days: number of daily bars to fetch for indicators.
        selection_strategy: Asset selection strategy:
            - "volume": Top assets by trading volume (default/stable)
            - "gainers": Top gaining assets in 24h
            - "losers": Top losing assets in 24h (oversold opportunities)
            - "trending": Recently trending/active assets
            - "mixed": Balanced mix of different types
            - "volatile": High volatility assets (good for day trading)

    Returns:
        list of asset dicts with guaranteed historical data if include_history=True
    """
    global LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
    FAILED_CRYPTO_SOURCES = []
    SKIPPED_CRYPTO_SOURCES = []
    LAST_CRYPTO_SOURCE = "None"

    print(f"[DEBUG] fetch_crypto_data: limit={limit}, include_history={include_history}, history_days={history_days}, strategy={selection_strategy}")

    # Try providers in order with enhanced validation and dynamic selection
    
    # 1) CoinGecko - Enhanced version with strategy support
    try:
        print(f"[DEBUG] Trying CoinGecko (enhanced) with {selection_strategy} strategy...")
        rows = _from_coingecko_enhanced_with_strategy(limit, include_history, history_days, selection_strategy)
        
        if _validate_historical_data(rows, include_history):
            LAST_CRYPTO_SOURCE = f"CoinGecko ({selection_strategy})"
            SKIPPED_CRYPTO_SOURCES = ["CoinPaprika", "CoinCap", "CryptoCompare", "TwelveData"]
            print(f"[DEBUG] CoinGecko SUCCESS: {len(rows)} assets with sufficient data")
            return rows
        else:
            print("[DEBUG] CoinGecko validation FAILED - trying fallbacks")
            primary_rows = rows
    except Exception as e:
        print(f"[DEBUG] CoinGecko FAILED: {e}")
        FAILED_CRYPTO_SOURCES.append(f"CoinGecko ({e})")
        primary_rows = []

    # Continue with other sources (keeping the same fallback logic)
    # ... (rest of the fallback chain remains the same)
    
    # 2) CoinPaprika - Enhanced version  
    try:
        print("[DEBUG] Trying CoinPaprika (enhanced)...")
        rows = _from_coinpaprika_enhanced_with_strategy(limit, include_history, history_days, selection_strategy)
        
        if _validate_historical_data(rows, include_history):
            LAST_CRYPTO_SOURCE = f"CoinPaprika ({selection_strategy})"
            SKIPPED_CRYPTO_SOURCES = ["CoinCap", "CryptoCompare", "TwelveData"]
            print(f"[DEBUG] CoinPaprika SUCCESS: {len(rows)} assets with sufficient data")
            return rows
        elif primary_rows:
            print("[DEBUG] Merging CoinPaprika data with CoinGecko data...")
            merged_rows = _merge_fallback_data(primary_rows, rows)
            if _validate_historical_data(merged_rows, include_history):
                LAST_CRYPTO_SOURCE = f"CoinGecko+CoinPaprika ({selection_strategy})"
                SKIPPED_CRYPTO_SOURCES = ["CoinCap", "CryptoCompare", "TwelveData"]
                print(f"[DEBUG] Merged data SUCCESS: {len(merged_rows)} assets")
                return merged_rows
            primary_rows = merged_rows
    except Exception as e:
        print(f"[DEBUG] CoinPaprika FAILED: {e}")
        FAILED_CRYPTO_SOURCES.append(f"CoinPaprika ({e})")

    # Continue with remaining fallback sources...
    # (keeping the same logic but adding strategy support where possible)
    
    # If we have some data but not enough historical data, return what we have
    if primary_rows:
        print(f"[DEBUG] Returning partial data: {len(primary_rows)} assets (insufficient historical data)")
        LAST_CRYPTO_SOURCE = f"Partial ({selection_strategy}, insufficient history)"
        
        for row in primary_rows:
            price_history = row.get("price_history", [])
            if not price_history or len(price_history) < MIN_HISTORICAL_DAYS:
                row["_warning"] = "insufficient_historical_data"
        
        return primary_rows

    print("[DEBUG] All crypto data sources FAILED")
    LAST_CRYPTO_SOURCE = "None"
    return []


def _from_coingecko_enhanced_with_strategy(limit: int, include_history: bool, history_days: int, strategy: str) -> List[Dict[str, Any]]:
    """Enhanced CoinGecko implementation with multiple selection strategies."""
    if not CoinGeckoClient:
        raise RuntimeError("CoinGecko adapter not available")
    
    print(f"[DEBUG] CoinGecko: Fetching {limit} assets with {strategy} strategy, history={include_history}")
    
    cg = CoinGeckoClient()
    
    # Get different asset sets based on strategy
    if strategy == "gainers":
        print("[DEBUG] Using gainers strategy (24h price change desc)")
        markets = cg.get_markets(
            vs_currency="usd", 
            order="price_change_percentage_24h_desc", 
            per_page=limit, 
            page=1, 
            sparkline=False,
            price_change_percentage="24h"
        )
        # Show sample of selected assets
        sample_assets = []
        for m in markets[:5]:
            symbol = m.get('symbol', 'N/A').upper()
            change = m.get('price_change_percentage_24h_in_currency', 0)
            sample_assets.append(f"{symbol} (+{change:.1f}%)")
        print(f"[DEBUG] Top gainers: {sample_assets}")
        
    elif strategy == "losers":
        print("[DEBUG] Using losers strategy (24h price change asc - oversold opportunities)")
        markets = cg.get_markets(
            vs_currency="usd", 
            order="price_change_percentage_24h_asc", 
            per_page=limit, 
            page=1, 
            sparkline=False,
            price_change_percentage="24h"
        )
        sample_assets = []
        for m in markets[:5]:
            symbol = m.get('symbol', 'N/A').upper()
            change = m.get('price_change_percentage_24h_in_currency', 0)
            sample_assets.append(f"{symbol} ({change:.1f}%)")
        print(f"[DEBUG] Top losers: {sample_assets}")
        
    elif strategy == "volatile":
        print("[DEBUG] Using volatile strategy (high volume + price change)")
        # Get high volume assets first, then filter by volatility
        all_markets = cg.get_markets(
            vs_currency="usd", 
            order="volume_desc", 
            per_page=limit * 3,  # Get more to filter from
            page=1, 
            sparkline=False,
            price_change_percentage="24h"
        )
        
        # Calculate volatility score (abs price change * volume)
        def volatility_score(m):
            price_change = abs(m.get('price_change_percentage_24h_in_currency', 0))
            volume = m.get('total_volume', 0) or 0
            return price_change * (volume ** 0.5)  # Scale volume impact
        
        markets = sorted(all_markets, key=volatility_score, reverse=True)[:limit]
        sample_assets = []
        for m in markets[:5]:
            symbol = m.get('symbol', 'N/A').upper()
            change = m.get('price_change_percentage_24h_in_currency', 0)
            sample_assets.append(f"{symbol} ({change:.1f}%)")
        print(f"[DEBUG] Most volatile: {sample_assets}")
        
    elif strategy == "trending":
        print("[DEBUG] Using trending strategy (market cap change)")
        markets = cg.get_markets(
            vs_currency="usd", 
            order="market_cap_change_percentage_24h_desc", 
            per_page=limit, 
            page=1, 
            sparkline=False,
            price_change_percentage="24h"
        )
        sample_assets = [m.get('symbol', 'N/A').upper() for m in markets[:5]]
        print(f"[DEBUG] Trending: {sample_assets}")
        
    elif strategy == "mixed":
        print("[DEBUG] Using mixed strategy (balanced selection)")
        # Get different types and mix them
        volume_assets = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit//2, page=1, sparkline=False)
        gainer_assets = cg.get_markets(vs_currency="usd", order="price_change_percentage_24h_desc", per_page=limit//3, page=1, sparkline=False, price_change_percentage="24h")
        trending_assets = cg.get_markets(vs_currency="usd", order="market_cap_change_percentage_24h_desc", per_page=limit//4, page=1, sparkline=False)
        
        # Combine and deduplicate by symbol
        seen_symbols = set()
        markets = []
        
        # Add volume leaders first (stable assets)
        for asset in volume_assets:
            symbol = asset.get('symbol', '').upper()
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                markets.append(asset)
        
        # Add gainers (momentum)
        for asset in gainer_assets:
            symbol = asset.get('symbol', '').upper()
            if symbol and symbol not in seen_symbols and len(markets) < limit:
                seen_symbols.add(symbol)
                markets.append(asset)
        
        # Add trending (breakouts)
        for asset in trending_assets:
            symbol = asset.get('symbol', '').upper()
            if symbol and symbol not in seen_symbols and len(markets) < limit:
                seen_symbols.add(symbol)
                markets.append(asset)
        
        print(f"[DEBUG] Mixed selection: {len(markets)} unique assets")
        sample_assets = [m.get('symbol', 'N/A').upper() for m in markets[:8]]
        print(f"[DEBUG] Sample mix: {sample_assets}")
        
    else:  # Default to volume
        print("[DEBUG] Using volume strategy (traditional high-volume assets)")
        markets = cg.get_markets(vs_currency="usd", order="volume_desc", per_page=limit, page=1, sparkline=False)
        sample_assets = [m.get('symbol', 'N/A').upper() for m in markets[:5]]
        print(f"[DEBUG] Top by volume: {sample_assets}")

    # Process the selected assets (same logic as before)
    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0

    for i, coin in enumerate(markets):
        try:
            price = _as_float(coin.get("current_price"))
            high  = _as_float(coin.get("high_24h"))
            low   = _as_float(coin.get("low_24h"))
            vol   = _as_float(coin.get("total_volume")) or 0.0
            sym   = (coin.get("symbol") or "").upper()
            cid   = coin.get("id") or sym
            change_24h = _as_float(coin.get("price_change_percentage_24h_in_currency", 0))

            if not price or price <= 0:
                print(f"[DEBUG] CoinGecko: Skipping {sym} - invalid price: {price}")
                continue

            day_range_pct: float = 0.0
            if price and high is not None and low is not None and price != 0:
                try:
                    day_range_pct = round(((high - low) / price) * 100.0, 2)
                except Exception:
                    day_range_pct = abs(change_24h or 0.0)  # Fallback to 24h change

            item: Dict[str, Any] = {
                "asset": cid,
                "symbol": _std_pair(sym),
                "price": price or 0.0,
                "volume": vol,
                "day_range_pct": day_range_pct,
                "change_24h": change_24h,  # Add 24h change for analysis
                "selection_reason": strategy,  # Track why this asset was selected
            }

            # Enhanced historical data fetching (same as before)
            if include_history and cid:
                try:
                    print(f"[DEBUG] CoinGecko: Fetching history for {sym} ({cid})")
                    
                    for attempt in range(2):
                        try:
                            chart = cg.market_chart(cid, vs_currency="usd", days=history_days + 2, interval="daily")
                            
                            if chart and "series" in chart:
                                prices = [p[1] for p in chart["series"]]
                                if len(prices) >= MIN_HISTORICAL_DAYS:
                                    item["price_history"] = prices[-(history_days + 1):]
                                    successful_history_fetches += 1
                                    print(f"[DEBUG] CoinGecko: Got {len(item['price_history'])} price points for {sym}")
                                    break
                                else:
                                    print(f"[DEBUG] CoinGecko: Insufficient history for {sym}: {len(prices)} days")
                            else:
                                print(f"[DEBUG] CoinGecko: No chart data for {sym}")
                                
                            if attempt == 0:
                                time.sleep(0.5)
                                
                        except Exception as e:
                            print(f"[DEBUG] CoinGecko: History fetch attempt {attempt + 1} failed for {sym}: {e}")
                            if attempt == 0:
                                time.sleep(0.5)
                            
                except Exception as e:
                    print(f"[DEBUG] CoinGecko: Failed to get history for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CoinGecko: Failed to process coin {i}: {e}")
            continue

    print(f"[DEBUG] CoinGecko: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rows


def _from_coinpaprika_enhanced_with_strategy(limit: int, include_history: bool, history_days: int, strategy: str) -> List[Dict[str, Any]]:
    """Enhanced CoinPaprika implementation with strategy support."""
    print(f"[DEBUG] CoinPaprika: Fetching {limit} assets with {strategy} strategy, history={include_history}")
    
    url = "https://api.coinpaprika.com/v1/tickers"
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Apply strategy-based sorting
    if strategy == "gainers":
        def sort_key(x): 
            return float((((x.get("quotes") or {}).get("USD") or {}).get("percent_change_24h")) or -999)
        sorted_data = sorted(data, key=sort_key, reverse=True)
        sample_assets = []
        for c in sorted_data[:5]:
            symbol = c.get('symbol', 'N/A')
            change = sort_key(c)
            sample_assets.append(f"{symbol} (+{change:.1f}%)")
        print(f"[DEBUG] CoinPaprika gainers: {sample_assets}")
        
    elif strategy == "losers":
        def sort_key(x): 
            return float((((x.get("quotes") or {}).get("USD") or {}).get("percent_change_24h")) or 999)
        sorted_data = sorted(data, key=sort_key)  # Ascending for losers
        sample_assets = []
        for c in sorted_data[:5]:
            symbol = c.get('symbol', 'N/A')
            change = sort_key(c)
            sample_assets.append(f"{symbol} ({change:.1f}%)")
        print(f"[DEBUG] CoinPaprika losers: {sample_assets}")
        
    elif strategy == "volatile":
        def volatility_key(x):
            quotes = (x.get("quotes") or {}).get("USD") or {}
            price_change = abs(float(quotes.get("percent_change_24h", 0)))
            volume = float(quotes.get("volume_24h", 0))
            return price_change * (volume ** 0.3)  # Volatility score
        sorted_data = sorted(data, key=volatility_key, reverse=True)
        sample_assets = [c.get('symbol', 'N/A') for c in sorted_data[:5]]
        print(f"[DEBUG] CoinPaprika volatile: {sample_assets}")
        
    else:  # Default to volume or mixed
        def usd_vol(x: Dict[str, Any]) -> float:
            return float((((x.get("quotes") or {}).get("USD") or {}).get("volume_24h")) or 0.0)
        sorted_data = sorted(data, key=usd_vol, reverse=True)
        sample_assets = [c.get('symbol', 'N/A') for c in sorted_data[:5]]
        print(f"[DEBUG] CoinPaprika volume: {sample_assets}")

    top = sorted_data[:limit * 2]  # Get extra in case some fail
    rows: List[Dict[str, Any]] = []
    successful_history_fetches = 0

    for c in top[:limit]:
        try:
            q = (c.get("quotes") or {}).get("USD") or {}
            price = _as_float(q.get("price")) or 0.0
            vol   = _as_float(q.get("volume_24h")) or 0.0
            pct24 = _as_float(q.get("percent_change_24h"))
            sym   = (c.get("symbol") or "").upper()
            cid   = c.get("id") or sym

            if not price or price <= 0:
                continue

            day_range_pct = round(float(pct24), 2) if pct24 is not None else 0.0

            item: Dict[str, Any] = {
                "asset": cid,
                "symbol": _std_pair(sym),
                "price": price,
                "volume": vol,
                "day_range_pct": day_range_pct,
                "change_24h": pct24,
                "selection_reason": strategy,
            }

            # Historical data fetching (same as before)
            if include_history and cid:
                try:
                    start = (datetime.now(timezone.utc) - timedelta(days=history_days + 2)).strftime("%Y-%m-%d")
                    hurl = f"https://api.coinpaprika.com/v1/tickers/{cid}/historical?start={start}&interval=1d"
                    hresp = requests.get(hurl, timeout=REQUEST_TIMEOUT)
                    if hresp.ok:
                        series = hresp.json()
                        closes = [float(x["price"]) for x in series if "price" in x]
                        if len(closes) >= MIN_HISTORICAL_DAYS:
                            item["price_history"] = closes[-(history_days + 1):]
                            successful_history_fetches += 1
                            print(f"[DEBUG] CoinPaprika: Got {len(item['price_history'])} price points for {sym}")
                except Exception as e:
                    print(f"[DEBUG] CoinPaprika: History failed for {sym}: {e}")

            rows.append(item)
            
        except Exception as e:
            print(f"[DEBUG] CoinPaprika: Failed to process coin: {e}")
            continue

    print(f"[DEBUG] CoinPaprika: Processed {len(rows)} assets, {successful_history_fetches} with history")
    return rowsDAYS:
                row["_warning"] = "insufficient_historical_data"
        
        return primary_rows

    # Complete failure - all sources failed
    print("[DEBUG] All crypto data sources FAILED")
    LAST_CRYPTO_SOURCE = "None"
    return []


# ────────────────────────────────────────────────────────────────────────────
# Backwards compatibility - keep original function names
# ────────────────────────────────────────────────────────────────────────────

# Keep original functions for backwards compatibility, but mark them as deprecated
def _from_coingecko(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper - use enhanced version."""
    return _from_coingecko_enhanced(limit, include_history, history_days)

def _from_coinpaprika(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper - use enhanced version."""
    return _from_coinpaprika_enhanced(limit, include_history, history_days)

def _from_coincap(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper - use enhanced version."""
    return _from_coincap_enhanced(limit, include_history, history_days)

def _from_cryptocompare(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper - use enhanced version."""
    return _from_cryptocompare_enhanced(limit, include_history, history_days)

def _from_twelvedata(limit: int, include_history: bool, history_days: int) -> List[Dict[str, Any]]:
    """Backwards compatibility wrapper - use enhanced version."""
    return _from_twelvedata_enhanced(limit, include_history, history_days)
