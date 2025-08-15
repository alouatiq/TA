# BE/trading_core/data_fetcher/commodities.py
"""
Metals & commodity exposure (market-aware)

Discovery
---------
• If `symbols` are provided -> use them directly.
• Else read market-specific proxies from BE/trading_core/data/markets.yml
  under: markets.<MARKET_KEY>.proxies.commodities (e.g., local gold/silver ETFs).
• Always append a compact global futures set (Yahoo: GC=F, SI=F, HG=F, PL=F, PA=F, ALI=F).

Quotes/History
--------------
• For futures symbols (=F), prefer Yahoo adapter (robust for futures) with Stooq fallback.
• For ETF proxies (no "=F"), prefer TwelveData (if API key set), then Yahoo, then Stooq.
• When include_history=True, attach ~15 daily closes when the backend provides them.

Output row shape
----------------
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
LAST_COMMODITIES_SOURCE: str
FAILED_COMMODITIES_SOURCES: List[str]
SKIPPED_COMMODITIES_SOURCES: List[str]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # We'll handle absence gracefully.

from .adapters import yahoo as yq
from .adapters import stooq as sq
from .adapters import twelvedata as td


# ────────────────────────────────────────────────────────────
# Diagnostics
# ────────────────────────────────────────────────────────────
LAST_COMMODITIES_SOURCE: str = "None"
FAILED_COMMODITIES_SOURCES: List[str] = []
SKIPPED_COMMODITIES_SOURCES: List[str] = []


# ────────────────────────────────────────────────────────────
# Defaults
# ────────────────────────────────────────────────────────────
# Compact global metals futures (Yahoo)
_GLOBAL_METALS_FUTURES: List[str] = [
    "GC=F",   # Gold
    "SI=F",   # Silver
    "HG=F",   # Copper
    "PL=F",   # Platinum
    "PA=F",   # Palladium
    "ALI=F",  # Aluminum (LME)
]


# ────────────────────────────────────────────────────────────
# Config loader
# ────────────────────────────────────────────────────────────
def _load_markets_yaml() -> Dict[str, Any]:
    """
    Load BE/trading_core/data/markets.yml if available.
    Returns {} if missing or YAML unavailable.
    """
    try:
        here = Path(__file__).resolve()
        markets_path = here.parents[2] / "data" / "markets.yml"
        if not markets_path.exists() or yaml is None:
            return {}
        with markets_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _proxies_for_market(market: Optional[str]) -> List[str]:
    """
    Read market-specific commodity proxies (e.g., local ETFs) from markets.yml:
      markets.<MARKET_KEY>.proxies.commodities: [ "SGLN.L", "PHAG.L", ... ]
    """
    if not market:
        return []
    cfg = _load_markets_yaml()
    # expected structure: { markets: { KEY: { proxies: { commodities: [...] } } } }
    try:
        mk = str(market).strip().upper()
        markets = cfg.get("markets", {}) or {}
        node = markets.get(mk, {}) or {}
        proxies = ((node.get("proxies", {}) or {}).get("commodities", [])) or []
        return [str(s).strip().upper() for s in proxies if s]
    except Exception:
        return []


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


def _resolve_symbols(
    *,
    market: Optional[str],
    region: Optional[str],  # region currently unused for commodities; reserved for future
    symbols: Optional[List[str]],
) -> List[str]:
    """
    Universe resolution:
      1) Explicit symbols if provided.
      2) Market-specific proxies from markets.yml (if any).
      3) Always append compact global metals futures.
    """
    if symbols:
        base = [str(s).strip().upper() for s in symbols if s]
    else:
        base = _proxies_for_market(market)
    base.extend(_GLOBAL_METALS_FUTURES)
    return _dedup(base)


def _build_row(sym: str, series: Dict[str, Any], include_history: bool) -> Dict[str, Any]:
    price = series.get("close", series.get("price"))
    high = series.get("high")
    low = series.get("low")
    vol = series.get("volume")

    try:
        price_f = float(price) if price is not None else 0.0
    except Exception:
        price_f = 0.0

    try:
        vol_i = int(vol) if vol is not None else 0
    except Exception:
        vol_i = 0

    drp = 0.0
    try:
        if price_f and high is not None and low is not None and float(price_f) != 0.0:
            drp = round(((float(high) - float(low)) / float(price_f)) * 100.0, 2)
    except Exception:
        drp = 0.0

    row: Dict[str, Any] = {
        "asset": sym,
        "symbol": sym,
        "price": price_f,
        "volume": vol_i,
        "day_range_pct": drp,
    }
    if include_history:
        hist = series.get("history") or []
        if hist:
            row["price_history"] = hist
    return row


def _has_twelvedata_key() -> bool:
    return bool(os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY"))


def _fetch_one_symbol(sym: str, *, include_history: bool) -> Optional[Dict[str, Any]]:
    """
    For futures (=F): Yahoo → Stooq
    For ETF proxies: TwelveData (if key) → Yahoo → Stooq
    """
    is_future = "=F" in sym or sym.startswith("^")

    # FUTURES PATH: Yahoo first
    if is_future:
        ya = yq.get_history(sym, lookback_days=16, with_intraday=False)
        if ya and ya.get("close") is not None:
            return _build_row(sym, ya, include_history)
        st = sq.get_quote(sym)
        if st and st.get("close") is not None:
            # Stooq path doesn't add history in our helper
            return _build_row(sym, st, include_history=False)
        return None

    # ETF/PROXY PATH
    # 1) TwelveData (if configured)
    if _has_twelvedata_key():
        tdq = td.get_quote(sym)
        if tdq and tdq.get("close") is not None:
            if include_history:
                tdh = td.get_history(sym, interval="1day", outputsize=20)
                if tdh:
                    tdq["history"] = tdh
            return _build_row(sym, tdq, include_history)

    # 2) Yahoo
    ya = yq.get_history(sym, lookback_days=16, with_intraday=False)
    if ya and ya.get("close") is not None:
        return _build_row(sym, ya, include_history)

    # 3) Stooq
    st = sq.get_quote(sym)
    if st and st.get("close") is not None:
        return _build_row(sym, st, include_history=False)

    return None


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_commodities_data(
    include_history: bool = False,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    min_assets: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetch market-aware commodities exposure (futures + optional local ETF proxies).

    Parameters
    ----------
    include_history : bool
        If True, attach last ~15 daily closes when the backend returns them.
    market : Optional[str]
        Exchange key (e.g., "LSE", "HKEX") to look up local ETF proxies in markets.yml.
    region : Optional[str]
        Reserved for future regional logic; currently unused in commodities.
    symbols : Optional[List[str]]
        Explicit override universe.
    min_assets : int
        Minimum rows required for a successful fetch.

    Returns
    -------
    List[dict]
        Normalized rows suitable for indicator pipeline.
    """
    global LAST_COMMODITIES_SOURCE, FAILED_COMMODITIES_SOURCES, SKIPPED_COMMODITIES_SOURCES
    FAILED_COMMODITIES_SOURCES = []
    SKIPPED_COMMODITIES_SOURCES = []
    LAST_COMMODITIES_SOURCE = "None"

    universe = _resolve_symbols(market=market, region=region, symbols=symbols)
    rows: List[Dict[str, Any]] = []

    used_td = False
    used_yahoo = False
    used_stooq = False

    for sym in universe:
        row = _fetch_one_symbol(sym, include_history=include_history)
        if row:
            rows.append(row)
            # crude provider attribution:
            if "=F" in sym or sym.startswith("^"):
                # futures path: Yahoo if price_history exists or cached; else Stooq
                if "price_history" in row:
                    used_yahoo = True
                else:
                    used_stooq = True
            else:
                if _has_twelvedata_key():
                    # If TD provided history/quote, our helper would have attached "history" when include_history
                    # We can't perfectly attribute; assume TD used first if key present and row has history
                    if "price_history" in row:
                        used_td = True
                    else:
                        # Could still be Yahoo; mark both best-effort
                        used_yahoo = True
                else:
                    # No TD key -> Yahoo or Stooq
                    if "price_history" in row:
                        used_yahoo = True
                    else:
                        used_stooq = True

    if rows:
        # Label the primary used source(s)
        labels = []
        if used_td:
            labels.append("TwelveData")
        if used_yahoo:
            labels.append("Yahoo")
        if used_stooq:
            labels.append("Stooq")
        LAST_COMMODITIES_SOURCE = " + ".join(labels) if labels else "Unknown"
    else:
        # Mark failures to help diagnostics
        if _has_twelvedata_key():
            FAILED_COMMODITIES_SOURCES.append("TwelveData")
        FAILED_COMMODITIES_SOURCES.append("Yahoo")
        FAILED_COMMODITIES_SOURCES.append("Stooq")

    # If too few assets, treat as failure (upstream can decide to switch category/market)
    if len(rows) < max(1, min_assets):
        return []

    return rows
