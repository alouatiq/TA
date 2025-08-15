# BE/trading_core/data_fetcher/equities.py
"""
Equities discovery + quotes (market/region aware) with layered fallbacks.

Discovery order
---------------
1) Yahoo localized screeners (if market is known & supported by adapters.yahoo)
2) Seeds from `trading_core/data/seeds.yml` (if available for that market)
3) Yahoo US screeners (as last resort)

Quote/History order
-------------------
Preferred (if TWELVEDATA_API_KEY set):
    TwelveData (quote+history) → Yahoo (history) → Yahoo Quote JSON → Stooq CSV
Default:
    Yahoo (history) → Yahoo Quote JSON → Stooq CSV

Public API
----------
fetch_equities_data(include_history: bool = False,
                    market: Optional[str] = None,
                    region: Optional[str] = None,
                    symbols: Optional[List[str]] = None,
                    max_universe: int = 60,
                    min_assets: int = 8,
                    force_seeds: bool = False) -> List[dict]

Diagnostics
-----------
LAST_EQUITIES_SOURCE: str
FAILED_EQUITIES_SOURCES: List[str]
SKIPPED_EQUITIES_SOURCES: List[str]

Row schema
----------
{
  "asset": "AAPL",          # display symbol
  "symbol": "AAPL",         # same as asset (for now)
  "price": 212.34,          # float
  "volume": 123456789,      # int
  "day_range_pct": 1.87,    # float, intraday (high-low)/price * 100 or close-based if provider supplies
  "price_history": [...],   # optional, last 15 closes when include_history=True
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

# Adapters
from .adapters import yahoo as yf_adapter
from .adapters import stooq as stq_adapter
from .adapters import twelvedata as td_adapter

# Optional: we won't fail if these aren’t present; we’ll just skip them
try:
    from ..utils.io import load_yaml_safe  # type: ignore
except Exception:
    load_yaml_safe = None  # graceful fallback

# ────────────────────────────────────────────────────────────
# Diagnostics (module-level)
# ────────────────────────────────────────────────────────────
LAST_EQUITIES_SOURCE: str = "None"
FAILED_EQUITIES_SOURCES: List[str] = []
SKIPPED_EQUITIES_SOURCES: List[str] = []

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _prefer_twelvedata() -> bool:
    key = os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY")
    force = (os.getenv("FORCE_TWELVEDATA", "").strip().lower() in {"1", "true", "yes"})
    return bool(key) or force


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _load_market_seeds(market: Optional[str]) -> List[str]:
    """Read a few liquid local listings for `market` from seeds.yml, if available."""
    if not market or not load_yaml_safe:
        return []
    try:
        # search default path within the repo structure
        data = load_yaml_safe("trading_core/data/seeds.yml")
        if not isinstance(data, dict):
            return []
        eq = data.get("equities", {})
        if not isinstance(eq, dict):
            return []
        syms = eq.get(market, []) or []
        return [s for s in syms if isinstance(s, str)]
    except Exception:
        return []


# ────────────────────────────────────────────────────────────
# Discovery
# ────────────────────────────────────────────────────────────
def _discover_symbols(market: Optional[str],
                      region: Optional[str],
                      *,
                      max_universe: int,
                      force_seeds: bool) -> Tuple[List[str], str, List[str], List[str]]:
    """
    Return (symbols, used_label, failed_sources, skipped_sources)
    """
    failed: List[str] = []
    skipped: List[str] = []

    # 0) If forcing seeds, try them first
    if force_seeds and market:
        seeds = _load_market_seeds(market)
        if seeds:
            return (seeds[:max_universe], f"Seed list ({market})", failed, skipped)
        # If no seeds available, proceed with normal discovery but note this
        skipped.append(f"Seed list ({market}) not found")

    # 1) Try localized Yahoo screeners when market is known
    if market and yf_adapter.market_supported(market):
        try:
            syms = yf_adapter.discover_symbols(market, max_rows=max_universe)
            if syms:
                return (syms[:max_universe], f"Yahoo Finance (regional: {market})", failed, skipped)
            else:
                failed.append(f"Yahoo regional screener ({market})")
        except Exception as e:
            failed.append(f"Yahoo regional screener ({market}): {e}")

    # 2) Seeds for that market
    if market:
        seeds = _load_market_seeds(market)
        if seeds:
            skipped.append("Yahoo regional screener (fallback to seeds)")
            return (seeds[:max_universe], f"Seed list ({market})", failed, skipped)

    # 3) US screeners (broad liquid universe) as last resort
    try:
        syms_us = yf_adapter.discover_symbols(None, max_rows=max_universe)
        if syms_us:
            src = "Yahoo Finance (US screeners)"
            if market:
                skipped.append(f"Regional discovery ({market})")
            return (syms_us[:max_universe], src, failed, skipped)
        else:
            failed.append("Yahoo US screeners (empty)")
    except Exception as e:
        failed.append(f"Yahoo US screeners: {e}")

    return ([], "None", failed, skipped)


# ────────────────────────────────────────────────────────────
# Quote & history
# ────────────────────────────────────────────────────────────
def _build_row(symbol: str,
               price: Optional[float],
               volume: Optional[int],
               day_range_pct: Optional[float],
               price_history: Optional[List[float]]) -> Optional[Dict[str, Any]]:
    try:
        if price is None:
            return None
        row: Dict[str, Any] = {
            "asset": symbol,
            "symbol": symbol,
            "price": float(price),
            "volume": int(volume or 0),
            "day_range_pct": float(day_range_pct if day_range_pct is not None else 0.0),
        }
        if price_history:
            row["price_history"] = price_history
        return row
    except Exception:
        return None


def _fetch_one_symbol(symbol: str, include_history: bool, prefer_td: bool,
                      failed: List[str], skipped: List[str]) -> Optional[Dict[str, Any]]:
    """
    Try providers in layered order to populate a single asset row.
    """
    # Prefer TwelveData when available
    if prefer_td:
        try:
            if include_history:
                ph = td_adapter.fetch_history(symbol, interval="1day", outputsize=20)  # last ~20 to ensure ≥15 closes
                if ph and len(ph) >= 15:
                    price = ph[-1]
                    # Approximate day range pct from last 1-2 bars if needed (TwelveData returns closes only here)
                    # We'll try Yahoo history next for richer intraday range if available
                    vol = td_adapter.fetch_quote(symbol).get("volume") if hasattr(td_adapter, "fetch_quote") else None  # type: ignore
                    # attempt Yahoo for better day range if available (optional)
                    try:
                        y_price, y_vol, y_drp, _ = yf_adapter.fetch_history(symbol)
                        if y_drp is not None:
                            day_range_pct = y_drp
                            if vol is None:
                                vol = y_vol
                        else:
                            day_range_pct = None
                    except Exception:
                        day_range_pct = None
                    return _build_row(symbol, price, vol, day_range_pct, ph[-15:])
            # Quote only
            q = td_adapter.fetch_quote(symbol)
            if q and q.get("price") is not None:
                return _build_row(
                    symbol,
                    float(q["price"]),
                    int(q.get("volume") or 0),
                    float(q.get("day_range_pct")) if q.get("day_range_pct") is not None else None,
                    None if not include_history else None,
                )
        except Exception as e:
            failed.append(f"TwelveData({symbol})")
            # fall through to Yahoo chain

    # Yahoo history (best when available)
    try:
        y_price, y_vol, y_drp, y_hist = yf_adapter.fetch_history(symbol)
        if y_price is not None:
            ph = y_hist[-15:] if (include_history and y_hist) else None
            return _build_row(symbol, y_price, y_vol, y_drp, ph)
    except Exception:
        # Try Yahoo Quote JSON
        try:
            q_price, q_vol, q_drp = yf_adapter.fetch_quote_json(symbol)
            if q_price is not None:
                return _build_row(symbol, q_price, q_vol, q_drp, None)
            else:
                failed.append(f"Yahoo quote JSON({symbol})")
        except Exception:
            failed.append(f"Yahoo quote JSON({symbol})")

    # Stooq CSV last resort
    try:
        s_price, s_vol, s_drp = stq_adapter.fetch_quote(symbol)
        if s_price is not None:
            return _build_row(symbol, s_price, s_vol, s_drp, None)
        else:
            failed.append(f"Stooq({symbol})")
    except Exception:
        failed.append(f"Stooq({symbol})")

    return None


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────
def fetch_equities_data(include_history: bool = False,
                        *,
                        market: Optional[str] = None,
                        region: Optional[str] = None,
                        symbols: Optional[List[str]] = None,
                        max_universe: int = 60,
                        min_assets: int = 8,
                        force_seeds: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch a liquid universe of equities + quotes, optionally with 15 bars of history.

    Args:
        include_history: Include 'price_history' when possible (last 15 closes).
        market: Optional exchange key (e.g., 'LSE', 'EN_PA', 'XETRA'). Improves discovery.
        region: Optional region hint (not required; discovery mainly uses market).
        symbols: If provided, skip discovery and fetch exactly these.
        max_universe: Cap the number of discovered symbols (default 60).
        min_assets: Desired minimum output rows (CLI will gate on this for indicators).
        force_seeds: Force using seeds.yml (when you know screeners are flaky).

    Returns:
        List[AssetRow] rows (possibly empty if all providers failed).
    """
    global LAST_EQUITIES_SOURCE, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES
    LAST_EQUITIES_SOURCE = "None"
    FAILED_EQUITIES_SOURCES = []
    SKIPPED_EQUITIES_SOURCES = []

    prefer_td = _prefer_twelvedata()

    # 1) Discover universe
    if symbols:
        universe = symbols[:max_universe]
        used_label = "User-specified symbols"
    else:
        universe, used_label, disc_failed, disc_skipped = _discover_symbols(
            market, region, max_universe=max_universe, force_seeds=force_seeds
        )
        FAILED_EQUITIES_SOURCES.extend(disc_failed)
        SKIPPED_EQUITIES_SOURCES.extend(disc_skipped)

    if not symbols and not universe:
        LAST_EQUITIES_SOURCE = "None"
        if not FAILED_EQUITIES_SOURCES:
            FAILED_EQUITIES_SOURCES.append("Discovery")
        return []

    if not symbols:
        LAST_EQUITIES_SOURCE = used_label
    else:
        LAST_EQUITIES_SOURCE = f"{used_label}"

    # 2) Fetch quotes/history per symbol with layered fallbacks
    rows: List[Dict[str, Any]] = []
    used_td = False
    used_yq = False
    used_stq = False

    for sym in universe:
        before_failed = set(FAILED_EQUITIES_SOURCES)
        row = _fetch_one_symbol(sym, include_history, prefer_td, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES)
        if row:
            rows.append(row)
        # mark which fallbacks got used by looking at new failures (best-effort signal)
        after_failed = set(FAILED_EQUITIES_SOURCES)
        new_fails = after_failed - before_failed
        # we can infer which path succeeded by absence of failure; this is soft.
        # For a clearer signal, you could instrument _fetch_one_symbol to return a tag.
        # For now, we conservatively set 'used_*' flags when earlier providers failed for this symbol.
        if any("TwelveData(" in x for x in new_fails):
            # TwelveData failed → likely used Yahoo or Stooq
            pass
        else:
            if prefer_td:
                used_td = True

        if any("Yahoo quote JSON" in x for x in new_fails):
            used_yq = True
        if any("Stooq" in x for x in new_fails):
            used_stq = True

        if len(rows) >= max_universe:
            break

    if used_yq:
        SKIPPED_EQUITIES_SOURCES.append("yfinance history (used Yahoo Quote JSON fallback)")
    if used_stq:
        SKIPPED_EQUITIES_SOURCES.append("yfinance history (used Stooq fallback)")
    if used_td:
        SKIPPED_EQUITIES_SOURCES.append("Yahoo/TwelveData secondary fallbacks not used for some symbols")

    # 3) Dedup diagnostics
    FAILED_EQUITIES_SOURCES = _dedup(FAILED_EQUITIES_SOURCES)
    SKIPPED_EQUITIES_SOURCES = _dedup(SKIPPED_EQUITIES_SOURCES)

    # 4) If output is too small and seeds exist (equities often hit screener quirks),
    #    try to supplement from seeds once (unless user explicitly passed symbols)
    if not symbols and len(rows) < min_assets and market:
        seeds = _load_market_seeds(market)
        if seeds:
            SKIPPED_EQUITIES_SOURCES.append("Supplemented with seeds to reach min_assets")
            supplement = [s for s in seeds if s not in universe]
            for sym in supplement:
                row = _fetch_one_symbol(sym, include_history, prefer_td, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES)
                if row:
                    rows.append(row)
                if len(rows) >= min_assets:
                    break
            rows = rows[:max_universe]

    return rows
