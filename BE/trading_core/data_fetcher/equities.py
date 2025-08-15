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

# Adapters - handle gracefully if they don't exist yet
try:
    from .adapters import yahoo as yf_adapter  # type: ignore
except Exception:
    yf_adapter = None  # type: ignore

try:
    from .adapters import stooq as stq_adapter  # type: ignore
except Exception:
    stq_adapter = None  # type: ignore

try:
    from .adapters import twelvedata as td_adapter  # type: ignore
except Exception:
    td_adapter = None  # type: ignore

# Optional: we won't fail if these aren't present; we'll just skip them
try:
    from ..utils.io import load_yaml_safe  # type: ignore
except Exception:
    load_yaml_safe = None  # graceful fallback

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics (module-level)
# ──────────────────────────────────────────────────────────────────────────────
LAST_EQUITIES_SOURCE: str = "None"
FAILED_EQUITIES_SOURCES: List[str] = []
SKIPPED_EQUITIES_SOURCES: List[str] = []

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
PRICE_HISTORY_DAYS = 16  # Default lookback for indicators that need ~14 periods (+1 cushion)

# No hardcoded fallback symbols - if discovery fails, it should fail properly

# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────
def _reset_diagnostics():
    """Reset module-level diagnostics before a new fetch operation."""
    global LAST_EQUITIES_SOURCE, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES
    LAST_EQUITIES_SOURCE = "None"
    FAILED_EQUITIES_SOURCES.clear()
    SKIPPED_EQUITIES_SOURCES.clear()


def _build_row(symbol: str, price: float, volume: Optional[int] = None, 
               day_range_pct: Optional[float] = None, 
               price_history: Optional[List[float]] = None) -> Dict[str, Any]:
    """Build a standardized asset row with the required schema."""
    return {
        "asset": symbol,
        "symbol": symbol,
        "price": price,
        "volume": volume or 0,
        "day_range_pct": day_range_pct or 0.0,
        "price_history": price_history or []
    }


def _discover_symbols(market: Optional[str] = None, 
                     region: Optional[str] = None,
                     max_universe: int = 60,
                     force_seeds: bool = False) -> List[str]:
    """
    Discover equity symbols using the layered fallback approach.
    
    Returns:
        List of symbol strings to fetch quotes for
    """
    global FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES
    
    symbols = []
    
    # Strategy 1: Yahoo localized screeners (if market supported)
    if not force_seeds and market and yf_adapter:
        try:
            # Check if adapter has discovery function
            if hasattr(yf_adapter, 'discover_equities'):
                yahoo_symbols = yf_adapter.discover_equities(market=market, limit=max_universe)
                if yahoo_symbols:
                    symbols.extend(yahoo_symbols)
                    return symbols[:max_universe]  # Early return if successful
            else:
                SKIPPED_EQUITIES_SOURCES.append(f"Yahoo screener ({market}) - function not available")
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append(f"Yahoo screener ({market})")
    elif not yf_adapter:
        SKIPPED_EQUITIES_SOURCES.append("Yahoo adapter not available")
    
    # Strategy 2: Seeds from YAML (if available)
    if load_yaml_safe and market:
        try:
            # Try to load seeds for the specific market
            seeds_data = load_yaml_safe("trading_core/data/seeds.yml")
            if seeds_data and "equities" in seeds_data:
                market_seeds = seeds_data["equities"].get(market, [])
                if market_seeds:
                    symbols.extend(market_seeds)
                    return symbols[:max_universe]  # Early return if successful
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append(f"Seeds YAML ({market})")
    elif not load_yaml_safe:
        SKIPPED_EQUITIES_SOURCES.append("YAML loader not available")
    
    # Strategy 3: Yahoo US screeners (last resort)
    if not symbols and yf_adapter:
        try:
            if hasattr(yf_adapter, 'discover_equities'):
                yahoo_us_symbols = yf_adapter.discover_equities(market="US", limit=max_universe)
                if yahoo_us_symbols:
                    symbols.extend(yahoo_us_symbols)
            else:
                SKIPPED_EQUITIES_SOURCES.append("Yahoo US screener - function not available")
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append("Yahoo US screener")
    
    # If no symbols found through any discovery method, return empty list
    # This allows the calling code to handle the failure appropriately
    if not symbols:
        FAILED_EQUITIES_SOURCES.append("All discovery methods failed")
    
    return symbols[:max_universe]


def _fetch_quote_with_history(symbol: str, include_history: bool = False) -> Optional[Dict[str, Any]]:
    """
    Fetch quote and optionally history for a single symbol using the fallback chain.
    
    Returns:
        Asset row dict or None if all methods failed
    """
    global LAST_EQUITIES_SOURCE, FAILED_EQUITIES_SOURCES
    
    # Check if TwelveData API key is available
    has_twelvedata_key = bool(os.getenv("TWELVEDATA_API_KEY"))
    
    # Method 1: TwelveData (if API key available and adapter exists)
    if has_twelvedata_key and td_adapter:
        try:
            if hasattr(td_adapter, 'fetch_quote_with_history'):
                td_data = td_adapter.fetch_quote_with_history(symbol, days=PRICE_HISTORY_DAYS if include_history else 0)
                if td_data and td_data.get("price") is not None:
                    LAST_EQUITIES_SOURCE = "TwelveData"
                    price_history = td_data.get("price_history", []) if include_history else []
                    return _build_row(
                        symbol=symbol,
                        price=td_data["price"],
                        volume=td_data.get("volume"),
                        day_range_pct=td_data.get("day_range_pct"),
                        price_history=price_history
                    )
            else:
                SKIPPED_EQUITIES_SOURCES.append(f"TwelveData({symbol}) - function not available")
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append(f"TwelveData({symbol})")
    elif not td_adapter:
        SKIPPED_EQUITIES_SOURCES.append("TwelveData adapter not available")
    
    # Method 2: Yahoo Finance (with history if requested)
    if yf_adapter:
        try:
            if include_history and hasattr(yf_adapter, 'fetch_quote_with_history'):
                yf_data = yf_adapter.fetch_quote_with_history(symbol, days=PRICE_HISTORY_DAYS)
            elif hasattr(yf_adapter, 'fetch_quote'):
                yf_data = yf_adapter.fetch_quote(symbol)
            else:
                yf_data = None
                SKIPPED_EQUITIES_SOURCES.append(f"Yahoo({symbol}) - functions not available")
            
            if yf_data and yf_data.get("price") is not None:
                LAST_EQUITIES_SOURCE = "Yahoo Finance"
                price_history = yf_data.get("price_history", []) if include_history else []
                return _build_row(
                    symbol=symbol,
                    price=yf_data["price"],
                    volume=yf_data.get("volume"),
                    day_range_pct=yf_data.get("day_range_pct"),
                    price_history=price_history
                )
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append(f"Yahoo({symbol})")
    else:
        SKIPPED_EQUITIES_SOURCES.append("Yahoo adapter not available")
    
    # Method 3: Stooq CSV (price only, no history)
    if stq_adapter:
        try:
            if hasattr(stq_adapter, 'fetch_quote'):
                stq_data = stq_adapter.fetch_quote(symbol)
                if stq_data and stq_data.get("price") is not None:
                    LAST_EQUITIES_SOURCE = "Stooq"
                    return _build_row(
                        symbol=symbol,
                        price=stq_data["price"],
                        volume=stq_data.get("volume"),
                        day_range_pct=stq_data.get("day_range_pct"),
                        price_history=[]  # Stooq doesn't provide history
                    )
            else:
                SKIPPED_EQUITIES_SOURCES.append(f"Stooq({symbol}) - function not available")
        except Exception as e:
            FAILED_EQUITIES_SOURCES.append(f"Stooq({symbol})")
    else:
        SKIPPED_EQUITIES_SOURCES.append("Stooq adapter not available")
    
    # All methods failed
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
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
    # Reset diagnostics for this fetch operation
    _reset_diagnostics()
    
    # Step 1: Symbol Discovery (unless symbols are provided)
    if symbols:
        target_symbols = symbols[:max_universe]  # Respect max_universe even for provided symbols
    else:
        target_symbols = _discover_symbols(
            market=market, 
            region=region, 
            max_universe=max_universe,
            force_seeds=force_seeds
        )
    
    # If discovery failed completely, return empty list with clear error message
    if not target_symbols:
        print(f"✗ Symbol discovery failed for market '{market}' - no symbols found")
        print(f"  Failed sources: {', '.join(FAILED_EQUITIES_SOURCES)}")
        if SKIPPED_EQUITIES_SOURCES:
            print(f"  Skipped sources: {', '.join(SKIPPED_EQUITIES_SOURCES)}")
        return []
    
    # Step 2: Fetch quotes and optionally history for each symbol
    results = []
    for symbol in target_symbols:
        row = _fetch_quote_with_history(symbol, include_history=include_history)
        if row:
            results.append(row)
        
        # Stop if we have enough results and meet maximum requirements
        if len(results) >= max_universe:
            break
    
    # Log final diagnostics
    if results:
        print(f"✓ Fetched {len(results)} equities (last source: {LAST_EQUITIES_SOURCE})")
        if SKIPPED_EQUITIES_SOURCES:
            print(f"  Note: Some sources were skipped: {', '.join(set(SKIPPED_EQUITIES_SOURCES))}")
    else:
        print(f"✗ No equities data available")
        print(f"  Failed sources: {', '.join(FAILED_EQUITIES_SOURCES)}")
        if SKIPPED_EQUITIES_SOURCES:
            print(f"  Skipped sources: {', '.join(SKIPPED_EQUITIES_SOURCES)}")
    
    return results
