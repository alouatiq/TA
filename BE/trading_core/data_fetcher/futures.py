# BE/trading_core/data_fetcher/futures.py
"""
Index futures / index cash proxies (market-aware, config-driven)

Discovery order
---------------
1) If `symbols=[...]` provided → use exactly those.
2) If `market` provided:
   - Try trading_core.config.get_market_info(market) and look for:
       • "primary_indices" (preferred) OR "indices" (list of Yahoo cash indices)
   - If not found there, try to read BE/trading_core/data/markets.yml
     under markets[<market>].primary_indices / .indices
3) If still empty, use a region default ("Americas" | "Europe" | "MEA" | "Asia").
4) Always append a compact global futures set (ES=F, NQ=F, YM=F, RTY=F, NK=F, HSI=F, VX=F).

Quotes/History
--------------
• Primary: Yahoo adapter (reliable for indices & many index futures)
• Fallback: Stooq adapter (best-effort for cash indices)
• TwelveData is typically not used for index futures; omitted here.

Row schema
----------
{
  "asset":  <symbol>,
  "symbol": <symbol>,
  "price":  float,
  "volume": int (0 if not available),
  "day_range_pct": float (0.0 if unavailable),
  "price_history": [floats]  # only if include_history=True and available
}

Diagnostics
-----------
LAST_FUTURES_SOURCE: str
FAILED_FUTURES_SOURCES: List[str]
SKIPPED_FUTURES_SOURCES: List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import sys

# Adapters
from .adapters import yahoo as yq
from .adapters import stooq as sq

# Optional config/YAML loading
try:
    # Preferred: centralized config helper if present in your project
    from trading_core.config import get_market_info  # type: ignore
except Exception:  # pragma: no cover
    get_market_info = None  # type: ignore

try:
    # Fallback: read markets.yml directly if available
    import importlib.resources as ilres
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    ilres = None  # type: ignore
    yaml = None   # type: ignore


# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_FUTURES_SOURCE: str = "None"
FAILED_FUTURES_SOURCES: List[str] = []
SKIPPED_FUTURES_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Small region defaults (used only if config/YAML give nothing)
# ────────────────────────────────────────────────────────────
_REGION_DEFAULTS: Dict[str, List[str]] = {
    "americas": ["^GSPC", "^NDX", "^DJI"],
    "europe":   ["^STOXX50E", "^FTSE", "^GDAXI"],
    "mea":      ["^J200", "^TASI.SR", "^MASI"],   # best-effort availability
    "asia":     ["^N225", "^HSI", "^AXJO", "^STI"],
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


def _markets_yaml_indices(market: str) -> List[str]:
    """
    Try to open trading_core/data/markets.yml and pull
    markets[market].primary_indices or .indices.
    Returns [] if not found/unavailable.
    """
    if ilres is None or yaml is None:
        return []
    try:
        pkg = "trading_core.data"
        with ilres.files(pkg).joinpath("markets.yml").open("rb") as fh:
            data = yaml.safe_load(fh) or {}
        markets = data.get("markets", {})
        node = markets.get(market, {}) if isinstance(markets, dict) else {}
        for key in ("primary_indices", "indices"):
            vals = node.get(key)
            if isinstance(vals, list) and vals:
                syms = [str(v).strip() for v in vals if v]
                return [s for s in syms if s.startswith("^") or s.endswith("=F")]
    except Exception:
        return []
    return []


def _config_indices(market: str) -> List[str]:
    """
    Ask trading_core.config.get_market_info(market) for primary indices.
    Expected keys:
      • "primary_indices" (preferred) OR
      • "indices"
    """
    if get_market_info is None:
        return []
    try:
        mi = get_market_info(market) or {}
        for key in ("primary_indices", "indices"):
            vals = mi.get(key)
            if isinstance(vals, list) and vals:
                syms = [str(v).strip() for v in vals if v]
                return [s for s in syms if s.startswith("^") or s.endswith("=F")]
    except Exception:
        return []
    return []


def _region_defaults(region: Optional[str]) -> List[str]:
    if not region:
        return []
    key = region.strip().lower()
    # normalize a few aliases
    if key.startswith("amer"):
        key = "americas"
    elif key.startswith("euro"):
        key = "europe"
    elif key in {"middle east", "africa", "middle east & africa"}:
        key = "mea"
    elif key.startswith("asia"):
        key = "asia"
    return list(_REGION_DEFAULTS.get(key, []))


def _resolve_index_symbols(
    *, market: Optional[str], region: Optional[str], symbols: Optional[List[str]]
) -> List[str]:
    """
    Resolve universe using config → YAML → region defaults, then add global futures.
    """
    base: List[str] = []
    if symbols:
        base = list(symbols)
    else:
        mk = (market or "").strip().upper()
        if mk:
            # config first
            idx = _config_indices(mk)
            if not idx:
                idx = _markets_yaml_indices(mk)
            base.extend(idx)

        if not base:
            base.extend(_region_defaults(region))

    # Always add a small global futures set
    base.extend(_GLOBAL_INDEX_FUTURES)
    return _dedup(base)


def _build_row_from_series(sym: str, series: Dict[str, Any], include_history: bool) -> Dict[str, Any]:
    """
    Convert an adapter result dict -> normalized row.
    Adapters (yahoo/stooq) return keys: price/close, high, low, volume, history?
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
    Try Yahoo (supports indices & many futures) → Stooq fallback (cash indices).
    """
    # Yahoo first
    ya = yq.get_history(sym, lookback_days=16, with_intraday=False)
    if ya and (ya.get("close") is not None or ya.get("price") is not None):
        return _build_row_from_series(sym, ya, include_history)

    # Stooq fallback (history not provided in our helper)
    st = sq.get_quote(sym)
    if st and (st.get("close") is not None or st.get("price") is not None):
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
        If True, attach ~15 closes where possible (Yahoo path).
    market : Optional[str]
        Exchange key (e.g., "LSE", "NYSE", "TSE"); used to select cash indices from config/YAML.
    region : Optional[str]
        Region hint for fallback universes: "Americas", "Europe", "MEA", "Asia".
    symbols : Optional[List[str]]
        Explicit override universe.
    min_assets : int
        Minimum rows to consider a successful fetch.

    Returns
    -------
    List[dict]
        Normalized rows suitable for downstream indicators.
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
            # Heuristic: if we have price_history on include_history, it likely came from Yahoo
            if include_history and "price_history" in row:
                used_yahoo = True
            else:
                # Could still be Yahoo (price-only); mark both best-effort
                used_yahoo = True if not used_yahoo else used_yahoo
                used_stooq = True  # some may have come from Stooq

    if rows:
        if used_yahoo and used_stooq:
            LAST_FUTURES_SOURCE = "Yahoo + Stooq"
        elif used_yahoo:
            LAST_FUTURES_SOURCE = "Yahoo"
        elif used_stooq:
            LAST_FUTURES_SOURCE = "Stooq"
        else:
            LAST_FUTURES_SOURCE = "Unknown"
    else:
        FAILED_FUTURES_SOURCES.extend(["Yahoo", "Stooq"])

    # Enforce minimum asset count
    if len(rows) < max(1, min_assets):
        return []

    return rows
