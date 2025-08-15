# BE/trading_core/data_fetcher/futures.py
"""
Index futures / index cash proxies (market-aware)

Discovery:
  • Prefer a market’s primary cash indices (Yahoo: ^GSPC, ^FTSE, ^GDAXI, …)
  • Add a compact global futures set (ES=F, NQ=F, YM=F, RTY=F, NK=F, HSI=F, VX=F)
  • Allow explicit symbol overrides via `symbols=[...]`

Quotes/History:
  • Primary: Yahoo adapter (reliable for indices & index futures)
  • Fallback: Stooq adapter (best-effort for cash indices & many regions)
  • (TwelveData usually doesn’t carry index futures; not used here)

Output row per symbol:
  {
    "asset":  <symbol>,
    "symbol": <symbol>,
    "price":  float,
    "volume": int (0 if not available),
    "day_range_pct": float (0.0 if unavailable),
    "price_history": [floats]  # only if include_history=True and available
  }

Diagnostics:
  LAST_FUTURES_SOURCE: str
  FAILED_FUTURES_SOURCES: List[str]
  SKIPPED_FUTURES_SOURCES: List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .adapters import yahoo as yq
from .adapters import stooq as sq

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_FUTURES_SOURCE: str = "None"
FAILED_FUTURES_SOURCES: List[str] = []
SKIPPED_FUTURES_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Universes
# ────────────────────────────────────────────────────────────
# Primary cash indices per market (safe Yahoo tickers)
_INDEX_BY_MARKET: Dict[str, List[str]] = {
    # Americas
    "NYSE":   ["^GSPC", "^DJI", "^RUT"],
    "NASDAQ": ["^NDX", "^IXIC"],
    "TSX":    ["^GSPTSE"],

    # Europe
    "LSE":    ["^FTSE"],
    "XETRA":  ["^GDAXI"],
    "EN_PA":  ["^FCHI"],
    "BME":    ["^IBEX"],
    "SIX":    ["^SSMI"],
    "WSE":    ["^WIG"],   # Warsaw broad index
    "VIE":    ["^ATX"],
    "BVB":    ["^BETI"],  # Bucharest BET-TR or BET; BETI (total return) often present
    "ATHEX":  ["^ATG"],   # Athens Composite
    "OSLO":   ["^OSEAX"],
    "OMX_STO":["^OMX"],   # Stockholm — Yahoo sometimes exposes ^OMX Stockholm 30 via ^OMXS30
    "OMX_HEL":["^OMXH25"],# Helsinki
    "OMX_CPH":["^OMXC25"],# Copenhagen
    "EN_AM":  ["^AEX"],
    "EN_BR":  ["^BFX"],   # BEL 20 proxy (may vary)
    "EN_LI":  ["^PSI20"], # Lisbon PSI
    "EN_DU":  ["^ISEQ"],  # Dublin ISEQ

    # Middle East & Africa (best-effort availability)
    "TADAWUL":["^TASI.SR"],
    "JSE":    ["^J200"],  # FTSE/JSE All Share
    "EGX":    ["^EGX30"],
    "CSE_MA": ["^MASI"],  # Morocco All Shares
    "NGX":    ["^NGSEINDEX"],  # NGX ASI; availability may vary
    "NSE_KE": ["^NSE20"],

    # Asia-Pacific
    "TSE":    ["^N225"],
    "HKEX":   ["^HSI"],
    "ASX":    ["^AXJO"],
    "SGX":    ["^STI"],
    "SSE":    ["^SSEC"],
    "SZSE":   ["^SZSC"],   # Shenzhen composite proxy (availability varies)
    "KOSPI":  ["^KS11"],
    "TWSE":   ["^TWII"],
    "NSE_IN": ["^NSEI"],   # Nifty 50
    "BSE_IN": ["^BSESN"],  # Sensex
}

# Regional defaults if a market has no direct mapping
_INDEX_BY_REGION: Dict[str, List[str]] = {
    "Americas": ["^GSPC", "^NDX", "^DJI"],
    "Europe":   ["^STOXX50E", "^FTSE", "^GDAXI"],
    "MEA":      ["^J200", "^TASI.SR", "^MASI"],
    "Asia":     ["^N225", "^HSI", "^AXJO", "^STI"],
}

# Compact global futures set (Yahoo)
_GLOBAL_INDEX_FUTURES: List[str] = [
    "ES=F",   # S&P 500
    "NQ=F",   # Nasdaq-100
    "YM=F",   # Dow
    "RTY=F",  # Russell 2000
    "NK=F",   # Nikkei 225
    "HSI=F",  # Hang Seng
    "VX=F",   # VIX futures
]


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


def _resolve_index_symbols(
    *, market: Optional[str], region: Optional[str], symbols: Optional[List[str]]
) -> List[str]:
    """
    Resolve a symbol universe:
      1) explicit `symbols` if provided
      2) market-specific cash indices (if available)
      3) region fallback list (if provided)
      4) always append a compact global futures set
    """
    if symbols:
        base = list(symbols)
    else:
        base = []
        mk = (market or "").strip().upper()
        if mk and mk in _INDEX_BY_MARKET:
            base.extend(_INDEX_BY_MARKET[mk])
        else:
            rg = (region or "").strip()
            # normalize region to our keys
            if rg.lower().startswith("amer"):
                base.extend(_INDEX_BY_REGION["Americas"])
            elif rg.lower().startswith("euro"):
                base.extend(_INDEX_BY_REGION["Europe"])
            elif rg.lower() in {"mea", "middle east", "africa", "middle east & africa"}:
                base.extend(_INDEX_BY_REGION["MEA"])
            elif rg.lower().startswith("asia"):
                base.extend(_INDEX_BY_REGION["Asia"])

    # Always add a small global futures set
    base.extend(_GLOBAL_INDEX_FUTURES)
    return _dedup(base)


def _build_row_from_series(sym: str, series: Dict[str, Any], include_history: bool) -> Dict[str, Any]:
    """
    Convert a Yahoo/Stooq result series -> normalized row.
    """
    price = float(series.get("close") or series.get("price") or 0.0)
    high  = series.get("high")
    low   = series.get("low")
    vol   = series.get("volume")
    try:
        volume = int(vol) if vol is not None else 0
    except Exception:
        volume = 0

    # day range (%)
    drp = 0.0
    try:
        if price and high is not None and low is not None and price != 0:
            drp = round(((float(high) - float(low)) / float(price)) * 100.0, 2)
    except Exception:
        drp = 0.0

    row: Dict[str, Any] = {
        "asset": sym,
        "symbol": sym,
        "price": float(price),
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
    Try Yahoo first (supports indices & futures) → Stooq fallback.
    """
    # Yahoo
    ya = yq.get_history(sym, lookback_days=16, with_intraday=False)
    if ya and ya.get("close") is not None:
        return _build_row_from_series(sym, ya, include_history)

    # Stooq fallback (often works for cash indices; futures coverage varies)
    st = sq.get_quote(sym)
    if st and st.get("close") is not None:
        # Stooq doesn't provide history in our helper; row will be price-only unless Yahoo worked
        return _build_row_from_series(sym, st, include_history=False)

    return None


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_futures_data(
    include_history: bool = False,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    min_assets: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch a market-aware universe of index cash proxies + index futures.

    Parameters
    ----------
    include_history : bool
        If True, attach last ~15 closes where possible (Yahoo path).
    market : Optional[str]
        Exchange key (e.g., "LSE", "NYSE", "TSE"); used to select cash indices.
    region : Optional[str]
        Region hint for fallback universes: "Americas", "Europe", "MEA", "Asia".
    symbols : Optional[List[str]]
        Explicit override universe.
    min_assets : int
        Minimum rows to consider a successful fetch.

    Returns
    -------
    List[dict]
        Normalized rows suitable for indicator pipeline.
    """
    global LAST_FUTURES_SOURCE, FAILED_FUTURES_SOURCES, SKIPPED_FUTURES_SOURCES
    FAILED_FUTURES_SOURCES = []
    SKIPPED_FUTURES_SOURCES = []
    LAST_FUTURES_SOURCE = "None"

    universe = _resolve_index_symbols(market=market, region=region, symbols=symbols)
    rows: List[Dict[str, Any]] = []

    used_yahoo = False
    used_stooq = False

    for sym in universe:
        row = _fetch_one_symbol(sym, include_history=include_history)
        if row:
            rows.append(row)
            # mark which provider succeeded (best-effort via simple probe)
            # Yahoo path returns history field when include_history True
            if "price_history" in row or sym in yq._YF_QUOTE_CACHE:  # type: ignore[attr-defined]
                used_yahoo = True
            else:
                used_stooq = True

    if rows:
        # Label the primary used source(s)
        if used_yahoo and used_stooq:
            LAST_FUTURES_SOURCE = "Yahoo + Stooq"
        elif used_yahoo:
            LAST_FUTURES_SOURCE = "Yahoo"
        elif used_stooq:
            LAST_FUTURES_SOURCE = "Stooq"
        else:
            LAST_FUTURES_SOURCE = "Unknown"
    else:
        FAILED_FUTURES_SOURCES.append("Yahoo")
        FAILED_FUTURES_SOURCES.append("Stooq")

    # If too few assets, signal failure to upstream
    if len(rows) < max(1, min_assets):
        return []

    return rows
