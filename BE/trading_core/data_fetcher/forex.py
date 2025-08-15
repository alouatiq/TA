# BE/trading_core/data_fetcher/forex.py
"""
FX data acquisition (quotes + optional history).

Usage:
    from trading_core.data_fetcher.forex import fetch_forex_data
    rows = fetch_forex_data(include_history=True, pairs=["EURUSD", "USDJPY"])
    # Each row: {
    #   "asset": "EURUSD", "symbol": "EURUSD",
    #   "price": 1.0923, "volume": 0, "day_range_pct": 0.42,
    #   "price_history": [ ... ]  # if include_history=True and backend supports it
    # }

Design:
- Backend order:
    1) TwelveData (if API key set)
    2) Yahoo Finance (yfinance; symbol 'EURUSD=X')
    3) Stooq (CSV; symbol 'eurusd')
- No region/market hard binds: FX is global. If you pass `pairs`, we only use those.
  If you don't pass pairs, we use a compact majors default as a fallback universe.
- Robust diagnostics via LAST/FAILED/SKIPPED variables (for CLI status panel).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import os

# optional: not all environments have every adapter
try:
    from .adapters import twelvedata as tda  # type: ignore
except Exception:
    tda = None  # type: ignore

try:
    from .adapters import yahoo as yfa  # type: ignore
except Exception:
    yfa = None  # type: ignore

try:
    from .adapters import stooq as stq  # type: ignore
except Exception:
    stq = None  # type: ignore


AssetRow = Dict[str, Any]
AssetRows = List[AssetRow]

# Diagnostics mirrors other fetchers
LAST_FOREX_SOURCE: str = "None"
FAILED_FOREX_SOURCES: List[str] = []
SKIPPED_FOREX_SOURCES: List[str] = []

# Default lookback for indicators that need ~14 periods (+1 cushion)
PRICE_HISTORY_DAYS = 16

# Compact majors fallback (only used when user doesn't supply `pairs`)
_DEFAULT_MAJORS: Tuple[str, ...] = (
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "AUDJPY",
)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _norm_pair(p: str) -> str:
    """
    Accepts 'eurusd', 'EURUSD', 'EUR/USD' -> returns 'EURUSD'.
    """
    if not isinstance(p, str):
        return ""
    p = p.strip().upper().replace("/", "")
    return p


def _pairs_universe(pairs: Optional[Iterable[str]]) -> List[str]:
    if pairs:
        out: List[str] = []
        seen = set()
        for p in pairs:
            np = _norm_pair(p)
            if np and np not in seen:
                seen.add(np)
                out.append(np)
        return out
    # fallback default universe
    return list(_DEFAULT_MAJORS)


def _pct_range(high: Optional[float], low: Optional[float], price: Optional[float]) -> float:
    try:
        if price and high is not None and low is not None and price != 0:
            return round(((float(high) - float(low)) / float(price)) * 100.0, 2)
    except Exception:
        pass
    return 0.0


def _append_history(row: AssetRow, closes: Optional[List[float]], *, want_history: bool) -> None:
    if want_history and closes and len(closes) >= 2:
        row["price_history"] = [float(x) for x in closes[-(PRICE_HISTORY_DAYS + 1):]]


# ────────────────────────────────────────────────────────────
# Backend fetchers (per symbol)
# ────────────────────────────────────────────────────────────
def _fetch_td_fx(pair: str, *, include_history: bool) -> Optional[AssetRow]:
    """
    TwelveData path. Symbol format: 'EUR/USD'.
    """
    if tda is None:
        return None
    if not os.getenv("TWELVEDATA_API_KEY") and not os.getenv("TWELVE_DATA_API_KEY"):
        return None

    td_symbol = f"{pair[:3]}/{pair[3:]}"  # EURUSD -> EUR/USD
    try:
        q = tda.get_quote(td_symbol)  # expected keys: price, high, low (best-effort)
        if not q or "price" not in q:
            return None

        price = float(q.get("price"))
        high = q.get("high")
        low = q.get("low")
        drp = _pct_range(high, low, price)

        row: AssetRow = {
            "asset": pair,
            "symbol": pair,
            "price": price,
            "volume": 0,  # OTC FX: no consolidated volume
            "day_range_pct": drp,
        }

        if include_history:
            hist = tda.get_history(td_symbol, interval="1day", outputsize=60)
            closes = [float(h["close"]) for h in hist] if hist else []
            _append_history(row, closes, want_history=True)

        return row
    except Exception:
        return None


def _fetch_yf_fx(pair: str, *, include_history: bool) -> Optional[AssetRow]:
    """
    Yahoo path via yfinance. Symbol format: 'EURUSD=X'.
    """
    if yfa is None:
        return None
    yf_symbol = f"{pair}=X"
    try:
        # We use the unified adapter to get a short history window
        hist = yfa.get_quote_history(yf_symbol, period=f"{PRICE_HISTORY_DAYS + 2}d")
        if not hist or hist.empty:
            # try plain quote if no history
            q = yfa.get_quote(yf_symbol)
            if not q or "price" not in q:
                return None
            price = float(q.get("price"))
            high = q.get("high")
            low = q.get("low")
            drp = _pct_range(high, low, price)
            return {
                "asset": pair,
                "symbol": pair,
                "price": price,
                "volume": 0,
                "day_range_pct": drp,
            }

        # last row quote fields
        last = hist.iloc[-1]
        price = float(last["Close"])
        high = float(last["High"]) if not math.isnan(last["High"]) else None
        low = float(last["Low"]) if not math.isnan(last["Low"]) else None
        drp = _pct_range(high, low, price)

        row: AssetRow = {
            "asset": pair,
            "symbol": pair,
            "price": price,
            "volume": 0,
            "day_range_pct": drp,
        }

        if include_history:
            closes = hist["Close"].tail(PRICE_HISTORY_DAYS + 1).tolist()
            _append_history(row, closes, want_history=True)

        return row
    except Exception:
        return None


def _fetch_stooq_fx(pair: str, *, include_history: bool) -> Optional[AssetRow]:
    """
    Stooq CSV. Symbol format: 'eurusd' (lowercase).
    We only get last quote (daily); no reliable multi-day history via the lightweight endpoint.
    """
    if stq is None:
        return None
    st_symbol = pair.lower()
    try:
        csvrow = stq.get_csv_quote(st_symbol)
        if not csvrow:
            return None

        # csvrow keys: symbol,date,time,open,high,low,close,volume
        close = csvrow.get("close")
        price = float(close) if close not in (None, "", "N/A") else None
        if price is None:
            return None
        high = csvrow.get("high")
        low = csvrow.get("low")
        vol = csvrow.get("volume")
        drp = _pct_range(
            float(high) if high not in (None, "", "N/A") else None,
            float(low) if low not in (None, "", "N/A") else None,
            price,
        )

        row: AssetRow = {
            "asset": pair,
            "symbol": pair,
            "price": float(price),
            "volume": int(float(vol)) if vol not in (None, "", "N/A") else 0,
            "day_range_pct": drp,
        }
        # No robust multi-day series from this endpoint
        if include_history:
            # leave missing; RSI/SMA will be gated by caller anyway
            pass

        return row
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# Public: fetch_forex_data
# ────────────────────────────────────────────────────────────
def fetch_forex_data(
    include_history: bool = False,
    *,
    pairs: Optional[Iterable[str]] = None,
    market: Optional[str] = None,   # accepted but ignored (FX is global)
    region: Optional[str] = None,   # accepted but ignored (FX is global)
) -> AssetRows:
    """
    Fetch a small liquid FX universe or the user-provided pairs.

    Parameters
    ----------
    include_history : bool
        If True, attach 'price_history' (last ~15 daily closes) when backend supports it.
    pairs : Optional[Iterable[str]]
        Specific pairs to fetch, e.g., ["EURUSD","USDJPY"] or ["EUR/USD"].
        If not provided, uses a compact majors fallback (_DEFAULT_MAJORS).
    market, region : optional
        Accepted for a consistent signature across fetchers; FX is treated as global here.

    Returns
    -------
    List[dict]
        Rows contain: asset, symbol, price, volume (0 for FX), day_range_pct, and optional price_history.
    """
    global LAST_FOREX_SOURCE, FAILED_FOREX_SOURCES, SKIPPED_FOREX_SOURCES
    FAILED_FOREX_SOURCES = []
    SKIPPED_FOREX_SOURCES = []
    LAST_FOREX_SOURCE = "None"

    universe = _pairs_universe(pairs)
    results: AssetRows = []

    prefer_td = bool(os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY"))
    backends_order: List[str] = []
    if prefer_td and tda is not None:
        backends_order.append("twelvedata")
    if yfa is not None:
        backends_order.append("yahoo")
    if stq is not None:
        backends_order.append("stooq")

    if not backends_order:
        FAILED_FOREX_SOURCES.extend(["twelvedata", "yahoo", "stooq"])
        return results

    # try backends in order, collect rows that are still missing
    remaining = set(universe)

    for i, backend in enumerate(backends_order):
        if not remaining:
            break

        fetched_now: List[AssetRow] = []

        if backend == "twelvedata":
            if not prefer_td:
                SKIPPED_FOREX_SOURCES.append("twelvedata (no API key)")
            else:
                for p in list(remaining):
                    row = _fetch_td_fx(p, include_history=include_history)
                    if row:
                        fetched_now.append(row)
                        remaining.discard(p)
                if fetched_now:
                    LAST_FOREX_SOURCE = "TwelveData"
                else:
                    FAILED_FOREX_SOURCES.append("TwelveData")

        elif backend == "yahoo":
            for p in list(remaining):
                row = _fetch_yf_fx(p, include_history=include_history)
                if row:
                    fetched_now.append(row)
                    remaining.discard(p)
            if fetched_now and LAST_FOREX_SOURCE == "None":
                LAST_FOREX_SOURCE = "Yahoo Finance"
            elif not fetched_now:
                FAILED_FOREX_SOURCES.append("Yahoo Finance")

        elif backend == "stooq":
            for p in list(remaining):
                row = _fetch_stooq_fx(p, include_history=include_history)
                if row:
                    fetched_now.append(row)
                    remaining.discard(p)
            if fetched_now and LAST_FOREX_SOURCE == "None":
                LAST_FOREX_SOURCE = "Stooq"
            elif not fetched_now:
                FAILED_FOREX_SOURCES.append("Stooq")

        results.extend(fetched_now)

        # mark subsequent backends as skipped (advisory only)
        if fetched_now and (i + 1) < len(backends_order):
            for b in backends_order[i + 1 :]:
                SKIPPED_FOREX_SOURCES.append(f"{b} (not needed)")

    # If nothing fetched, list all as failed
    if not results and LAST_FOREX_SOURCE == "None":
        if "TwelveData" not in FAILED_FOREX_SOURCES and prefer_td:
            FAILED_FOREX_SOURCES.append("TwelveData")
        if "Yahoo Finance" not in FAILED_FOREX_SOURCES:
            FAILED_FOREX_SOURCES.append("Yahoo Finance")
        if "Stooq" not in FAILED_FOREX_SOURCES:
            FAILED_FOREX_SOURCES.append("Stooq")

    # stable order as input
    order_map = {p: i for i, p in enumerate(universe)}
    results.sort(key=lambda r: order_map.get(str(r.get("asset")), 1_000_000))
    return results


__all__ = [
    "fetch_forex_data",
    "LAST_FOREX_SOURCE",
    "FAILED_FOREX_SOURCES",
    "SKIPPED_FOREX_SOURCES",
]
