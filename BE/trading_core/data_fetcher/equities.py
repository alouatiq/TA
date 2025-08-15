"""
equities.py
───────────
Equities discovery + quotes with global/regional awareness.

Discovery order
  1) Localized Yahoo Finance screeners for the selected market (if known)
  2) US Yahoo Finance screeners (Most Active / Gainers / Losers)
  3) Curated seed tickers from data/seeds.yml (per market), if discovery fails
     or when `force_seeds=True`

Quote/history fallback chain (per symbol)
  1) yfinance daily history (last close + last 15 closes for indicators)
  2) TwelveData quote/history (if TWELVEDATA_API_KEY configured)
  3) Yahoo Quote JSON (price/vol/day-range%)
  4) Stooq CSV last quote (price/vol/day-range%)

Returns rows shaped like:
  {
    "asset": "<ticker>",         # same as 'symbol'
    "symbol": "<ticker>",        # Yahoo-style exchange suffixes kept (e.g., RDSA.AS)
    "price": 123.45,
    "volume": 987654,
    "day_range_pct": 2.31,       # (high - low) / price * 100
    "price_history": [ ... ]     # optional; ~15 closes when available
  }

Diagnostics (module globals; read by CLI):
  - LAST_STOCK_SOURCE: "Yahoo Finance (regional: X)", "Yahoo Finance (US)", "Seeds (X)", or "None"
  - FAILED_STOCK_SOURCES: [str,...]   # errors encountered
  - SKIPPED_STOCK_SOURCES: [str,...]  # what we bypassed due to earlier success
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# provider adapters
from .adapters import yahoo as yf_adp
from .adapters import stooq as stq_adp
from .adapters import twelvedata as td_adp

# shared IO loader for seeds.yml
try:
    # prefer our project utility if present
    from ..utils.io import load_yaml
except Exception:
    # minimal local loader fallback
    import yaml  # type: ignore

    def load_yaml(p: str | Path) -> dict:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

# ────────────────────────────────────────────────────────────
# module diagnostics (kept compatible with your existing CLI)
# ────────────────────────────────────────────────────────────
LAST_STOCK_SOURCE: str = "None"
FAILED_STOCK_SOURCES: List[str] = []
SKIPPED_STOCK_SOURCES: List[str] = []

# how many days we want for technical indicators (RSI-14 etc.)
PRICE_HISTORY_DAYS = 14

# cap to keep LLM/token usage reasonable; discovery may yield more
MAX_UNIVERSE = 80


# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _data_dir() -> Path:
    # trading_core/data/
    return Path(__file__).resolve().parents[1] / "data"


def _load_market_seeds(market: Optional[str]) -> List[str]:
    """
    Load curated seed tickers for a given market from data/seeds.yml
    Structure (example):
      equities:
        LSE: ["HSBA.L","BP.L","AZN.L",...]
        XETRA: ["SAP.DE","DTE.DE",...]
        ...
    """
    if not market:
        return []
    seeds_path = _data_dir() / "seeds.yml"
    data = load_yaml(seeds_path)
    eq = (data or {}).get("equities", {})
    syms = list(eq.get(market, []) or [])
    return _dedup(syms)[:MAX_UNIVERSE]


def _calc_day_range_pct(price: Optional[float], high: Optional[float], low: Optional[float]) -> Optional[float]:
    try:
        if price and high is not None and low is not None and price != 0:
            return round(((float(high) - float(low)) / float(price)) * 100, 2)
    except Exception:
        pass
    return None


def _row_from_yfinance(sym: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Try yfinance first (best path; also gives us recent closes).
    Returns (row_dict, error_msg_if_any).
    """
    try:
        last = yf_adp.yfinance_history(sym, PRICE_HISTORY_DAYS + 2)
        if not last or not last.get("price"):
            return None, "yfinance: empty"
        row = {
            "asset": sym,
            "symbol": sym,
            "price": float(last["price"]),
            "volume": int(last.get("volume") or 0),
            "day_range_pct": float(last.get("day_range_pct") or 0.0),
        }
        hist = last.get("price_history") or []
        if hist:
            row["price_history"] = hist
        return row, None
    except Exception as e:
        return None, f"yfinance: {e}"


def _row_from_twelvedata(sym: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    TwelveData (if configured) — history first, else quote.
    """
    if not td_adp.is_configured():
        return None, "twelvedata: not configured"
    try:
        h = td_adp.history(sym, interval="1day", outputsize=PRICE_HISTORY_DAYS + 2)
        if h and h.get("price") is not None:
            row = {
                "asset": sym,
                "symbol": sym,
                "price": float(h["price"]),
                "volume": int(h.get("volume") or 0),
                "day_range_pct": float(h.get("day_range_pct") or 0.0),
            }
            ph = h.get("price_history") or []
            if ph:
                row["price_history"] = ph
            return row, None

        q = td_adp.quote(sym)
        if q and q.get("price") is not None:
            drp = _calc_day_range_pct(q.get("price"), q.get("high"), q.get("low"))
            row = {
                "asset": sym,
                "symbol": sym,
                "price": float(q["price"]),
                "volume": int(q.get("volume") or 0),
                "day_range_pct": float(drp or 0.0),
            }
            return row, None

        return None, "twelvedata: empty"
    except Exception as e:
        return None, f"twelvedata: {e}"


def _row_from_yahoo_quote(sym: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Yahoo Quote JSON — quick price/volume/range fallback.
    """
    try:
        q = yf_adp.quote_json(sym)
        if not q or q.get("price") is None:
            return None, "yahoo quote: empty"
        drp = _calc_day_range_pct(q.get("price"), q.get("high"), q.get("low"))
        row = {
            "asset": sym,
            "symbol": sym,
            "price": float(q["price"]),
            "volume": int(q.get("volume") or 0),
            "day_range_pct": float(drp or 0.0),
        }
        return row, None
    except Exception as e:
        return None, f"yahoo quote: {e}"


def _row_from_stooq(sym: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Stooq — last-resort CSV quote.
    """
    try:
        q = stq_adp.csv_quote(sym)
        if not q or q.get("price") is None:
            return None, "stooq: empty"
        row = {
            "asset": sym,
            "symbol": sym,
            "price": float(q["price"]),
            "volume": int(q.get("volume") or 0),
            "day_range_pct": float(q.get("day_range_pct") or 0.0),
        }
        return row, None
    except Exception as e:
        return None, f"stooq: {e}"


def _fetch_one_symbol(sym: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Apply the per-symbol fallback chain and collect error notes.
    """
    errors: List[str] = []

    row, err = _row_from_yfinance(sym)
    if row:
        return row, errors
    if err:
        errors.append(err)

    row, err = _row_from_twelvedata(sym)
    if row:
        return row, errors
    if err:
        errors.append(err)

    row, err = _row_from_yahoo_quote(sym)
    if row:
        return row, errors
    if err:
        errors.append(err)

    row, err = _row_from_stooq(sym)
    if row:
        return row, errors
    if err:
        errors.append(err)

    return None, errors


# ────────────────────────────────────────────────────────────
# discovery
# ────────────────────────────────────────────────────────────
def _discover_symbols(market: Optional[str], *, max_n: int = MAX_UNIVERSE) -> Tuple[List[str], str, List[str]]:
    """
    Returns (symbols, source_label, skipped_reasons)
    """
    skipped: List[str] = []

    # 1) localized yahoo for the chosen market
    if market and market in yf_adp.LOCALIZED_HOSTS:
        try:
            syms_local = yf_adp.discover_from_localized_host(market, max_rows=max_n)
            syms_local = _dedup(syms_local)
            if syms_local:
                return syms_local[:max_n], f"Yahoo Finance (regional: {market})", skipped
            skipped.append(f"regional screeners ({market}) — empty")
        except Exception as e:
            skipped.append(f"regional screeners ({market}) — error: {e}")

    # 2) US yahoo screeners
    try:
        syms_us = yf_adp.discover_from_us_screeners(max_rows=max_n)
        syms_us = _dedup(syms_us)
        if syms_us:
            return syms_us[:max_n], "Yahoo Finance (US screeners)", skipped
        skipped.append("US screeners — empty")
    except Exception as e:
        skipped.append(f"US screeners — error: {e}")

    # 3) seeds.yml (market-specific)
    if market:
        seeds = _load_market_seeds(market)
        if seeds:
            return seeds[:max_n], f"Seeds ({market})", skipped
        skipped.append(f"seeds.yml — none for {market}")

    # no symbols
    return [], "None", skipped


# ────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────
def fetch_equities_data(
    include_history: bool = False,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,      # reserved for future refinements
    min_assets: int = 8,
    force_seeds: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build a liquid stock universe and fetch quotes/history with robust fallbacks.

    Args
    ----
    include_history : bool
        When True, attach ~15 closes for technical indicators when available.
    market : Optional[str]
        Exchange key (e.g., "LSE", "XETRA", "TSE"). Improves localized discovery.
    region : Optional[str]
        Currently not used directly here; present for API symmetry/future filters.
    min_assets : int
        Minimum rows to return (try seeds if discovery results are too small).
    force_seeds : bool
        If True, skip discovery and use curated seeds for the market immediately.

    Returns
    -------
    List[Dict[str, Any]] : normalized rows
    """
    global LAST_STOCK_SOURCE, FAILED_STOCK_SOURCES, SKIPPED_STOCK_SOURCES
    FAILED_STOCK_SOURCES = []
    SKIPPED_STOCK_SOURCES = []
    LAST_STOCK_SOURCE = "None"

    # Discovery
    symbols: List[str] = []
    source_label = "None"

    if force_seeds and market:
        symbols = _load_market_seeds(market)
        source_label = f"Seeds ({market})"
        if not symbols:
            SKIPPED_STOCK_SOURCES.append(f"forced seeds ({market}) — empty")
    if not symbols:
        symbols, source_label, skipped = _discover_symbols(market, max_n=MAX_UNIVERSE)
        SKIPPED_STOCK_SOURCES.extend(skipped)

    # If still thin and market known, try seeds as booster
    if market and len(symbols) < min_assets:
        seeds = _load_market_seeds(market)
        booster = [s for s in seeds if s not in symbols]
        if booster:
            symbols.extend(booster)
            symbols = _dedup(symbols)[:MAX_UNIVERSE]
            if "Seeds" not in source_label:
                SKIPPED_STOCK_SOURCES.append("added market seeds as booster")

    if not symbols:
        LAST_STOCK_SOURCE = "None"
        FAILED_STOCK_SOURCES.append("Discovery: no symbols")
        return []

    LAST_STOCK_SOURCE = source_label

    # Per-symbol quotes
    rows: List[Dict[str, Any]] = []
    used_td = False
    used_yq = False
    used_stooq = False

    for sym in symbols:
        row, errors = _fetch_one_symbol(sym)
        if row:
            # keep history only if requested
            if not include_history and "price_history" in row:
                row.pop("price_history", None)
            rows.append(row)
        else:
            # map errors to diagnostics flags
            for e in errors:
                if e.startswith("twelvedata:") and not used_td:
                    used_td = True
                if e.startswith("yahoo quote:") and not used_yq:
                    used_yq = True
                if e.startswith("stooq:") and not used_stooq:
                    used_stooq = True
            # collect last error for reference
            if errors:
                FAILED_STOCK_SOURCES.append(f"{sym}: {errors[-1]}")

    # Tag which fallbacks we ended up using successfully for *some* symbols
    # (We can't know per-call success cleanly here without verbose tracking;
    #  use a simple hint so the CLI can print “skipped/used fallback” messages.)
    if td_adp.is_configured():
        SKIPPED_STOCK_SOURCES.append("yfinance history (TwelveData used where needed)")
    SKIPPED_STOCK_SOURCES.append("yfinance history (Yahoo Quote JSON/Stooq used where needed)")

    # Final hygiene
    rows = rows[:MAX_UNIVERSE]
    FAILED_STOCK_SOURCES = _dedup(FAILED_STOCK_SOURCES)
    SKIPPED_STOCK_SOURCES = _dedup(SKIPPED_STOCK_SOURCES)

    # Ensure we meet minimum assets if requested
    if len(rows) < min_assets:
        FAILED_STOCK_SOURCES.append(f"Min-assets shortfall: need ≥{min_assets}, have {len(rows)}")

    return rows
