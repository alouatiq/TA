# BE/trading_core/data_fetcher/funds.py
"""
Funds (ETFs) – market-aware discovery + robust quotes/history.

Discovery:
  • If `symbols` provided -> use them.
  • Else prefer local ETFs by `market` (small, liquid seed lists per exchange).
  • Append a compact US core-ETF set to give baseline coverage.

Quotes/History:
  • Primary: Yahoo adapter (works broadly for ETFs; can return history).
  • Optional: TwelveData (if API key set) – first for intraday/daily history on equities ETFs.
  • Fallback: Stooq adapter for price-only if Yahoo history not present.

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
  LAST_FUNDS_SOURCE: str
  FAILED_FUNDS_SOURCES: List[str]
  SKIPPED_FUNDS_SOURCES: List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .adapters import yahoo as yq
from .adapters import stooq as sq
from .adapters import twelvedata as td

# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_FUNDS_SOURCE: str = "None"
FAILED_FUNDS_SOURCES: List[str] = []
SKIPPED_FUNDS_SOURCES: List[str] = []

# ────────────────────────────────────────────────────────────
# Seed universes
# (Conservative, liquid examples; expand later or load from data/seeds.yml)
# ────────────────────────────────────────────────────────────

# Compact US core ETFs (used everywhere as a safety fallback)
_FUNDS_US_CORE: List[str] = [
    "SPY", "IVV", "VOO", "QQQ", "VTI", "IWM", "EFA", "EEM", "TLT", "GLD", "SLV", "HYG", "LQD"
]

# Local ETFs by market (examples; safe, widely-available tickers)
_FUNDS_BY_MARKET: Dict[str, List[str]] = {
    # UK (LSE)
    "LSE":   ["ISF.L", "CSP1.L", "EQQQ.L", "SGLN.L"],

    # Canada (TSX)
    "TSX":   ["XIU.TO", "ZCN.TO"],

    # Japan (TSE)
    "TSE":   ["1306.T", "1321.T"],  # TOPIX, Nikkei 225

    # Hong Kong
    "HKEX":  ["2800.HK", "2828.HK"],

    # India
    "NSE_IN": ["NIFTYBEES.NS", "BANKBEES.NS"],
    "BSE_IN": ["NIFTYBEES.BO"],

    # Australia
    "ASX":   ["STW.AX", "VAS.AX"],

    # Singapore
    "SGX":   ["ES3.SI"],

    # Germany
    "XETRA": ["EXS1.DE"],  # iShares Core DAX (example; providers vary)

    # France / Euronext Paris
    "EN_PA": ["CAC.PA"],  # placeholder example; replace with your local preference later if needed

    # Switzerland
    "SIX":   ["CSSMI.SW"],  # SMI ETF (example symbol; providers vary)
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


def _resolve_fund_symbols(
    *, market: Optional[str], region: Optional[str], symbols: Optional[List[str]]
) -> List[str]:
    """
    Resolve a symbol universe:
      1) explicit `symbols` if provided
      2) local ETFs by market (if available)
      3) append US core set
    """
    if symbols:
        return _dedup(list(symbols))

    base: List[str] = []
    mk = (market or "").strip().upper()
    if mk and mk in _FUNDS_BY_MARKET:
        base.extend(_FUNDS_BY_MARKET[mk])

    # Always append US core baseline
    base.extend(_FUNDS_US_CORE)
    return _dedup(base)


def _build_row(symbol: str, quote: Dict[str, Any], include_history: bool) -> Dict[str, Any]:
    """
    Convert a provider quote dict into our normalized row.
    """
    price = float(quote.get("close") or quote.get("price") or 0.0)
    high = quote.get("high")
    low = quote.get("low")
    vol = quote.get("volume")

    try:
        volume = int(vol) if vol is not None else 0
    except Exception:
        volume = 0

    drp = 0.0
    try:
        if price and high is not None and low is not None and price != 0:
            drp = round(((float(high) - float(low)) / float(price)) * 100.0, 2)
    except Exception:
        drp = 0.0

    row: Dict[str, Any] = {
        "asset": symbol,
        "symbol": symbol,
        "price": price,
        "volume": volume,
        "day_range_pct": drp,
    }

    if include_history:
        hist = quote.get("history")
        if hist:
            row["price_history"] = hist

    return row


def _fetch_one_symbol(symbol: str, *, include_history: bool) -> Optional[Dict[str, Any]]:
    """
    Try TwelveData (if available) for history first (ETFs often supported),
    then Yahoo, then Stooq (price-only).
    """
    used_any = False

    # TwelveData – prefer for history if available
    td_quote = td.get_history(symbol, lookback_days=16, interval="1day")
    if td_quote and td_quote.get("close") is not None:
        used_any = True
        return _build_row(symbol, td_quote, include_history=True)

    # Yahoo – reliable for ETFs globally; may include history
    yq_quote = yq.get_history(symbol, lookback_days=16, with_intraday=False)
    if yq_quote and yq_quote.get("close") is not None:
        used_any = True
        return _build_row(symbol, yq_quote, include_history=include_history)

    # Stooq – price-only fallback
    sq_quote = sq.get_quote(symbol)
    if sq_quote and sq_quote.get("close") is not None:
        used_any = True
        return _build_row(symbol, sq_quote, include_history=False)

    if not used_any:
        return None
    return None


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_funds_data(
    include_history: bool = False,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    min_assets: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch a market-aware universe of ETFs (local-first, then US core).

    Parameters
    ----------
    include_history : bool
        If True, attach last ~15 closes for indicators when available.
    market : Optional[str]
        Exchange key (e.g., "LSE", "XETRA", "TSX") for local-first discovery.
    region : Optional[str]
        (Currently unused; reserved for future expansion.)
    symbols : Optional[List[str]]
        Explicit universe override.
    min_assets : int
        Minimum rows to consider a successful fetch.

    Returns
    -------
    List[dict]
        Normalized rows suitable for indicator pipeline.
    """
    global LAST_FUNDS_SOURCE, FAILED_FUNDS_SOURCES, SKIPPED_FUNDS_SOURCES
    FAILED_FUNDS_SOURCES = []
    SKIPPED_FUNDS_SOURCES = []
    LAST_FUNDS_SOURCE = "None"

    universe = _resolve_fund_symbols(market=market, region=region, symbols=symbols)
    rows: List[Dict[str, Any]] = []

    used_td = False
    used_yq = False
    used_sq = False

    for sym in universe:
        row = _fetch_one_symbol(sym, include_history=include_history)
        if row:
            rows.append(row)
            # best-effort provider detection: TwelveData returns 'provider' field in our adapter
            # and Yahoo sets history often; Stooq is price-only via our adapter naming
            prov = None
            # inspect slightly to tag source
            if "price_history" in row and len(row["price_history"]) >= 10:
                # could be TwelveData or Yahoo; try to refetch tiny hint (cheap)
                if td.get_history(sym, lookback_days=1) is not None:
                    used_td = True
                else:
                    used_yq = True
            else:
                # price-only likely Stooq or Yahoo JSON
                # probe order: Yahoo first, else Stooq
                yprobe = yq.get_quote(sym)
                if yprobe:
                    used_yq = True
                else:
                    used_sq = True

    # Diagnostics source label
    if rows:
        sources = []
        if used_td:
            sources.append("TwelveData")
        if used_yq:
            sources.append("Yahoo")
        if used_sq:
            sources.append("Stooq")
        LAST_FUNDS_SOURCE = " + ".join(sources) if sources else "Unknown"
    else:
        FAILED_FUNDS_SOURCES.extend(["TwelveData", "Yahoo", "Stooq"])

    if len(rows) < max(1, min_assets):
        return []

    return rows
