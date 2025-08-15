# BE/trading_core/data_fetcher/commodities.py
"""
Commodities (metals) – market‑aware fetcher.

Universe
--------
• Global front‑month futures (Yahoo symbols): GC=F, SI=F, HG=F, PL=F, PA=F, ALI=F
• Optional *local* ETF proxies per exchange (e.g., LSE, HKEX, TSX, XETRA, ASX, SGX)
  These are best‑effort liquid trackers users can actually trade on local venues.

Backends & Fallbacks
--------------------
Primary path tries:
  1) TwelveData (if API key present and prefer_twelvedata() is True)
  2) Yahoo history (yfinance) for price & 15 closes (indicators-ready)
  3) Yahoo Quote JSON (price/vol/day range only; no history)
  4) Stooq CSV (price/vol/day range only; no history)

Inputs
------
include_history : bool
    If True, attach 'price_history' (≥15 closes when available).
market : Optional[str]
    Exchange key (e.g., "LSE","HKEX","TSX","XETRA","ASX","SGX"). Enables local ETF proxies.
region : Optional[str]
    Advisory only; currently not required since futures are global.
symbols : Optional[List[str]]
    Manual override – use exactly these fetch symbols (Yahoo/TwelveData), skipping discovery.

Output rows (per asset)
-----------------------
{
  "asset": str,            # display symbol (same as 'symbol')
  "symbol": str,           # fetch/display symbol (Yahoo/TwelveData)
  "price": float,
  "volume": int,
  "day_range_pct": float,  # (high - low)/price * 100 when available, else 0
  "price_history": List[float]  # only when include_history=True and available
}

Diagnostics
-----------
LAST_COMMODITIES_SOURCE : str
FAILED_COMMODITIES_SOURCES : List[str]
SKIPPED_COMMODITIES_SOURCES : List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .adapters import yahoo, stooq, twelvedata
from . import prefer_twelvedata

AssetRow = Dict[str, Any]

# ────────────────────────────────────────────────────────────
# Base universe: front‑month metals futures (universal Yahoo symbols)
# ────────────────────────────────────────────────────────────
_DEFAULT_METAL_FUTURES = [
    "GC=F",  # Gold
    "SI=F",  # Silver
    "HG=F",  # Copper
    "PL=F",  # Platinum
    "PA=F",  # Palladium
    "ALI=F", # Aluminum (LME)
]

# ────────────────────────────────────────────────────────────
# Optional local ETF proxies by market (extend as needed)
# These are *examples* of liquid trackers for users trading locally.
# Add/adjust symbols as you validate availability in your environment.
# ────────────────────────────────────────────────────────────
_LOCAL_ETF_BY_MARKET = {
    # London Stock Exchange
    "LSE": [
        "SGLN.L",  # iShares Physical Gold
        "PHAG.L",  # WisdomTree Physical Silver
    ],
    # Hong Kong
    "HKEX": [
        "2840.HK", # SPDR Gold Shares (HK)
        # "2828.HK"   # Example silver/China proxy – comment/uncomment if confirmed in your setup
    ],
    # Canada
    "TSX": [
        "CGL.TO",  # iShares Gold Bullion ETF
        "SVR.TO",  # iShares Silver Bullion ETF
    ],
    # Germany (XETRA)
    "XETRA": [
        "8PSG.DE",  # Xtrackers IE Physical Gold (example; verify in your region)
    ],
    # Australia
    "ASX": [
        "GOLD.AX", # ETFS Physical Gold
        "ETPMAG.AX", # ETFS Physical Silver (ticker may vary; keep best-effort)
    ],
    # Singapore
    "SGX": [
        # SGX ETF tickers for gold/silver are limited; leave empty by default.
    ],
    # Paris / Euronext (examples)
    "EN_PA": [
        # Add verified FR tickers if desired, e.g., "PAU.PA" for palladium proxy if liquid enough.
    ],
}

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_COMMODITIES_SOURCE: str = "None"
FAILED_COMMODITIES_SOURCES: List[str] = []
SKIPPED_COMMODITIES_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _compute_day_range_pct(price: Optional[float], high: Optional[float], low: Optional[float]) -> float:
    if not price or price == 0 or high is None or low is None:
        return 0.0
    try:
        return round(((float(high) - float(low)) / float(price)) * 100.0, 2)
    except Exception:
        return 0.0


def _resolve_symbols(market: Optional[str], region: Optional[str], symbols: Optional[List[str]]) -> List[str]:
    """
    Decide which symbols to fetch:
      - If symbols override is provided, use it as‑is.
      - Else start with local ETF proxies (if market provided), then add global futures.
    """
    if symbols:
        # caller specified exactly what to fetch
        seen, out = set(), []
        for s in symbols:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    base: List[str] = []
    if market and market in _LOCAL_ETF_BY_MARKET:
        base.extend(_LOCAL_ETF_BY_MARKET[market])

    # Always append global metals futures – reliable & universal
    base.extend(_DEFAULT_METAL_FUTURES)

    # Deduplicate preserving order
    seen, out = set(), []
    for s in base:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _append_row(results: List[AssetRow], symbol: str, price: float, volume: int,
                day_range_pct: float, price_history: Optional[List[float]]) -> None:
    row: AssetRow = {
        "asset": symbol,
        "symbol": symbol,
        "price": float(price),
        "volume": int(volume or 0),
        "day_range_pct": float(day_range_pct or 0.0),
    }
    if price_history:
        row["price_history"] = list(price_history)
    results.append(row)


# ────────────────────────────────────────────────────────────
# Core fetch
# ────────────────────────────────────────────────────────────
def fetch_commodities_data(*,
                           include_history: bool = False,
                           market: Optional[str] = None,
                           region: Optional[str] = None,
                           symbols: Optional[List[str]] = None) -> List[AssetRow]:
    """
    Fetch commodities universe (metals futures + optional local proxies) with robust fallbacks.

    Parameters
    ----------
    include_history : bool
        If True, attempt to attach ≥15 closes for indicators (RSI/SMA).
    market : Optional[str]
        Market key (e.g., "LSE","HKEX","TSX","XETRA","ASX","SGX") – enables local ETF proxies.
    region : Optional[str]
        Advisory only; futures are global (kept for symmetry with other fetchers).
    symbols : Optional[List[str]]
        Manual override – fetch exactly these symbols.

    Returns
    -------
    List[AssetRow]
    """
    global LAST_COMMODITIES_SOURCE, FAILED_COMMODITIES_SOURCES, SKIPPED_COMMODITIES_SOURCES
    FAILED_COMMODITIES_SOURCES = []
    SKIPPED_COMMODITIES_SOURCES = []
    LAST_COMMODITIES_SOURCE = "None"

    fetch_syms = _resolve_symbols(market, region, symbols)
    results: List[AssetRow] = []

    use_td = prefer_twelvedata()
    used_yahoo_hist = False
    used_yahoo_quote = False
    used_stooq = False
    used_td = False

    for sym in fetch_syms:
        # 1) TwelveData (if available & preferred)
        if use_td:
            try:
                q = twelvedata.fetch_quote(sym)
                if q and q.price is not None:
                    price = float(q.price)
                    volume = int(q.volume or 0)
                    # TwelveData quote may already expose high/low; if not, default to 0% range
                    drp = _compute_day_range_pct(price, q.high, q.low)
                    price_hist = None
                    if include_history:
                        series = twelvedata.fetch_history(sym, interval="1day", outputsize=16)
                        if series:
                            price_hist = [float(x.close) for x in series][-16:]  # ensure ≥15 closes
                    _append_row(results, sym, price, volume, drp, price_hist)
                    used_td = True
                    LAST_COMMODITIES_SOURCE = "TwelveData"
                    continue
            except Exception as e:
                FAILED_COMMODITIES_SOURCES.append(f"TwelveData:{sym} ({e})")

        # 2) Yahoo history (best for indicators)
        try:
            price, vol, drp, price_hist = yahoo.fetch_history_for_symbol(sym, period_days=16)
            if price is not None:
                _append_row(results, sym, price, int(vol or 0), float(drp or 0.0),
                            price_hist if include_history else None)
                used_yahoo_hist = True
                if LAST_COMMODITIES_SOURCE == "None":
                    LAST_COMMODITIES_SOURCE = "Yahoo Finance (history)"
                continue
        except Exception as e:
            FAILED_COMMODITIES_SOURCES.append(f"Yahoo history:{sym} ({e})")

        # 3) Yahoo Quote JSON
        try:
            qp = yahoo.fetch_yahoo_quote_json(sym)
            if qp and qp.price is not None:
                price = float(qp.price)
                volume = int(qp.volume or 0)
                drp = _compute_day_range_pct(price, qp.high, qp.low)
                _append_row(results, sym, price, volume, drp, None)  # no history from this path
                used_yahoo_quote = True
                if LAST_COMMODITIES_SOURCE == "None":
                    LAST_COMMODITIES_SOURCE = "Yahoo Finance (quote)"
                continue
        except Exception as e:
            FAILED_COMMODITIES_SOURCES.append(f"Yahoo quote:{sym} ({e})")

        # 4) Stooq CSV
        try:
            sp = stooq.fetch_stooq_quote(sym)
            if sp and sp.price is not None:
                price = float(sp.price)
                volume = int(sp.volume or 0)
                drp = _compute_day_range_pct(price, sp.high, sp.low)
                _append_row(results, sym, price, volume, drp, None)  # no history from this path
                used_stooq = True
                if LAST_COMMODITIES_SOURCE == "None":
                    LAST_COMMODITIES_SOURCE = "Stooq"
                continue
        except Exception as e:
            FAILED_COMMODITIES_SOURCES.append(f"Stooq:{sym} ({e})")
            # give up on this symbol; skip

    # Diagnostics “skipped” reasoning
    if used_td:
        if used_yahoo_hist or used_yahoo_quote or used_stooq:
            SKIPPED_COMMODITIES_SOURCES.append("Yahoo/Stooq (TwelveData preferred)")
    else:
        # If we didn’t use TwelveData but it’s available, note it as skipped
        if prefer_twelvedata():
            SKIPPED_COMMODITIES_SOURCES.append("TwelveData (not used for some symbols)")

        if used_yahoo_hist and (used_yahoo_quote or used_stooq):
            SKIPPED_COMMODITIES_SOURCES.append("Fallbacks used for some symbols")

    # If absolutely nothing worked
    if not results:
        LAST_COMMODITIES_SOURCE = "None"
        if not FAILED_COMMODITIES_SOURCES:
            FAILED_COMMODITIES_SOURCES.append("No provider returned data")

    # Deduplicate diagnostics
    def _dedup(seq: List[str]) -> List[str]:
        seen = set(); out: List[str] = []
        for s in seq:
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out

    FAILED_COMMODITIES_SOURCES[:] = _dedup(FAILED_COMMODITIES_SOURCES)
    SKIPPED_COMMODITIES_SOURCES[:] = _dedup(SKIPPED_COMMODITIES_SOURCES)

    return results
