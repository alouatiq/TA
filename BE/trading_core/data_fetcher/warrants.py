# BE/trading_core/data_fetcher/warrants.py
"""
Leveraged ETP proxy layer (market‑aware)

Purpose
-------
Provides a practical universe of liquid leveraged ETPs/ETFs that can be used
as "warrants & structured products" proxies across regions. We prioritize
local listings where we know liquid tickers; otherwise we fall back to a
compact, US‑listed leveraged set.

Discovery:
  • If `symbols=[...]` is supplied, use it directly.
  • Else pick from a market‑aware map (LSE etc.), else fallback to US set.

Quotes/History:
  • Primary: Yahoo adapter (price + optional recent close history for indicators)
  • Fallback: Stooq adapter (price only for many tickers)

Output row per symbol:
  {
    "asset":  <symbol>,
    "symbol": <symbol>,
    "price":  float,
    "volume": int (0 if not available),
    "day_range_pct": float (0.0 if unavailable),
    "price_history": [floats]  # only if include_history=True and available
  }

Diagnostics (mirrors other fetchers):
  LAST_WARRANTS_SOURCE: str
  FAILED_WARRANTS_SOURCES: List[str]
  SKIPPED_WARRANTS_SOURCES: List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .adapters import yahoo as yq
from .adapters import stooq as sq

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_WARRANTS_SOURCE: str = "None"
FAILED_WARRANTS_SOURCES: List[str] = []
SKIPPED_WARRANTS_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Universes
# ────────────────────────────────────────────────────────────
# US leveraged ETPs (liquid core)
_US_WARRANTS: List[str] = [
    "TQQQ", "SQQQ",
    "SPXL", "SPXS",
    "SOXL", "SOXS",
    "LABU", "LABD",
    "UDOW", "SDOW",
    "TNA", "TZA",
    "UVXY", "SVXY",
]

# LSE leveraged ETPs (WisdomTree/Leverage Shares examples; liquidity varies by day)
# NOTE: This is a best‑effort seed list; you can extend via data/seeds.yml later.
_LSE_WARRANTS: List[str] = [
    "3USL.L",  # 3x Long S&P 500
    "3USS.L",  # 3x Short S&P 500
    "3UKL.L",  # 3x Long FTSE 100
    "3UKS.L",  # 3x Short FTSE 100
    "3QQQ.L",  # 3x Long NASDAQ‑100
    # Add more when you validate liquidity (e.g., TSL3.L, TSLI.L, NAS3.L, etc.)
]

# DE/XETRA examples (verify availability in your region; else rely on US set)
_XETRA_WARRANTS: List[str] = [
    # Many leveraged products list on XETRA under issuer‑specific tickers.
    # Keep conservative seed examples; extend via seeds.yml as you validate.
    # "LQQ.DE",   # Example: leveraged NASDAQ‑100 (issuer dependent; may differ)
]

# Region fallback mapping (best‑effort)
_REGION_DEFAULTS: Dict[str, List[str]] = {
    "Americas": _US_WARRANTS,
    "Europe": _LSE_WARRANTS + _US_WARRANTS,
    "MEA": _US_WARRANTS,
    "Asia": _US_WARRANTS,
}


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _resolve_symbols(market: Optional[str], region: Optional[str], symbols: Optional[List[str]]) -> List[str]:
    """
    Resolve the warrant ETP universe:
      1) explicit `symbols` if provided
      2) exchange‑specific list where known (e.g., LSE)
      3) region defaults
      4) fallback to US set
    """
    if symbols:
        return _dedup([s.strip().upper() for s in symbols if s])

    mk = (market or "").strip().upper()
    rg = (region or "").strip()

    if mk == "LSE":
        return _dedup(list(_LSE_WARRANTS) + list(_US_WARRANTS))
    if mk == "XETRA":
        return _dedup(list(_XETRA_WARRANTS) + list(_US_WARRANTS))

    # Region fallback
    if rg:
        key = (
            "Americas" if rg.lower().startswith("amer")
            else "Europe" if rg.lower().startswith("euro")
            else "MEA" if rg.lower() in {"mea", "middle east", "africa", "middle east & africa"}
            else "Asia" if rg.lower().startswith("asia")
            else None
        )
        if key and key in _REGION_DEFAULTS:
            return _dedup(list(_REGION_DEFAULTS[key]))

    # Final fallback
    return list(_US_WARRANTS)


def _build_row_from_series(sym: str, series: Dict[str, Any], include_history: bool) -> Dict[str, Any]:
    price = float(series.get("close") or series.get("price") or 0.0)
    high = series.get("high")
    low = series.get("low")
    vol = series.get("volume")

    try:
        volume = int(vol) if vol is not None else 0
    except Exception:
        volume = 0

    # intraday day‑range (% of price)
    drp = 0.0
    try:
        if price and high is not None and low is not None and price != 0:
            drp = round(((float(high) - float(low)) / float(price)) * 100.0, 2)
    except Exception:
        drp = 0.0

    row: Dict[str, Any] = {
        "asset": sym,
        "symbol": sym,
        "price": price,
        "volume": volume,
        "day_range_pct": drp,
    }
    if include_history:
        hist = series.get("history") or []
        if hist:
            row["price_history"] = hist
    return row


def _fetch_one_symbol(sym: str, *, include_history: bool) -> Optional[Dict[str, Any]]:
    """
    Try Yahoo first (price + optional history) → Stooq fallback (price only).
    """
    ya = yq.get_history(sym, lookback_days=16, with_intraday=False)
    if ya and ya.get("close") is not None:
        return _build_row_from_series(sym, ya, include_history)

    st = sq.get_quote(sym)
    if st and st.get("close") is not None:
        return _build_row_from_series(sym, st, include_history=False)

    return None


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_warrants_data(
    include_history: bool = False,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    min_assets: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch a market‑aware universe of leveraged ETPs.

    Parameters
    ----------
    include_history : bool
        If True, attach last ~15 closes where possible (Yahoo path).
    market : Optional[str]
        Exchange key (e.g., "LSE", "XETRA"); used to prefer local listings.
    region : Optional[str]
        Region hint for fallback universes: "Americas", "Europe", "MEA", "Asia".
    symbols : Optional[List[str]]
        Explicit override universe.
    min_assets : int
        Minimum rows to consider a successful fetch.

    Returns
    -------
    List[dict]
        Normalized rows suitable for the indicator pipeline.
    """
    global LAST_WARRANTS_SOURCE, FAILED_WARRANTS_SOURCES, SKIPPED_WARRANTS_SOURCES
    FAILED_WARRANTS_SOURCES = []
    SKIPPED_WARRANTS_SOURCES = []
    LAST_WARRANTS_SOURCE = "None"

    universe = _resolve_symbols(market, region, symbols)
    rows: List[Dict[str, Any]] = []

    used_yahoo = False
    used_stooq = False

    for sym in universe:
        row = _fetch_one_symbol(sym, include_history=include_history)
        if row:
            rows.append(row)
            if "price_history" in row or sym in yq._YF_QUOTE_CACHE:  # type: ignore[attr-defined]
                used_yahoo = True
            else:
                used_stooq = True

    if rows:
        if used_yahoo and used_stooq:
            LAST_WARRANTS_SOURCE = "Yahoo + Stooq"
        elif used_yahoo:
            LAST_WARRANTS_SOURCE = "Yahoo"
        elif used_stooq:
            LAST_WARRANTS_SOURCE = "Stooq"
        else:
            LAST_WARRANTS_SOURCE = "Unknown"
    else:
        FAILED_WARRANTS_SOURCES.append("Yahoo")
        FAILED_WARRANTS_SOURCES.append("Stooq")

    # Not enough data? signal failure to upstream
    if len(rows) < max(1, min_assets):
        return []

    return rows
