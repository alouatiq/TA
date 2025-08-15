"""
trading_core.data_fetcher
─────────────────────────
Unified import surface for all category fetchers:

- equities.fetch_equities_data
- crypto.fetch_crypto_data
- forex.fetch_forex_data
- commodities.fetch_commodities_data
- futures.fetch_futures_data
- warrants.fetch_warrants_data
- funds.fetch_funds_data

Design:
- Each category file exposes a single public function named fetch_<category>_data(...)
- All functions follow a consistent signature:
    fetch_<category>_data(
        include_history: bool = False,
        *,
        market: str | None = None,
        region: str | None = None,
        min_assets: int | None = None,
        force_seeds: bool = False,
    ) -> list[dict]

Return shape (per asset dict — superset, fields may be missing if unavailable):
{
    "asset": "<display symbol/ticker>",
    "symbol": "<fetch symbol used on the data source>",
    "price": float,
    "volume": int | None,
    "day_range_pct": float | None,
    # Optional if include_history=True and available
    "price_history": list[float],
    # Optional extra context
    "currency": "USD" | "EUR" | ...,
    "exchange": "NYSE" | "LSE" | ...,
}

Notes:
- Fallback chains are handled *inside* each category.
- Adapters live in trading_core.data_fetcher.adapters.*
"""

from __future__ import annotations

# Re-export public fetchers for ergonomic imports
from .equities import fetch_equities_data
from .crypto import fetch_crypto_data
from .forex import fetch_forex_data
from .commodities import fetch_commodities_data
from .futures import fetch_futures_data
from .warrants import fetch_warrants_data
from .funds import fetch_funds_data


def diagnostics_for(category: str) -> dict:
    """
    Get diagnostic information for a specific data fetcher category.

    Args:
        category: One of 'crypto', 'forex', 'equities', 'commodities', 'futures', 'warrants', 'funds'

    Returns:
        Dict with keys: 'used', 'failed', 'skipped'
    """
    try:
        if category == "crypto":
            from .crypto import LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
            return {
                "used": LAST_CRYPTO_SOURCE,
                "failed": FAILED_CRYPTO_SOURCES,
                "skipped": SKIPPED_CRYPTO_SOURCES
            }
        elif category == "forex":
            from .forex import LAST_FOREX_SOURCE, FAILED_FOREX_SOURCES, SKIPPED_FOREX_SOURCES
            return {
                "used": LAST_FOREX_SOURCE,
                "failed": FAILED_FOREX_SOURCES,
                "skipped": SKIPPED_FOREX_SOURCES
            }
        elif category == "equities":
            from .equities import LAST_EQUITIES_SOURCE, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES
            return {
                "used": LAST_EQUITIES_SOURCE,
                "failed": FAILED_EQUITIES_SOURCES,
                "skipped": SKIPPED_EQUITIES_SOURCES
            }
        elif category == "commodities":
            from .commodities import LAST_COMMODITIES_SOURCE, FAILED_COMMODITIES_SOURCES, SKIPPED_COMMODITIES_SOURCES
            return {
                "used": LAST_COMMODITIES_SOURCE,
                "failed": FAILED_COMMODITIES_SOURCES,
                "skipped": SKIPPED_COMMODITIES_SOURCES
            }
        elif category == "futures":
            from .futures import LAST_FUTURES_SOURCE, FAILED_FUTURES_SOURCES, SKIPPED_FUTURES_SOURCES
            return {
                "used": LAST_FUTURES_SOURCE,
                "failed": FAILED_FUTURES_SOURCES,
                "skipped": SKIPPED_FUTURES_SOURCES
            }
        elif category == "warrants":
            from .warrants import LAST_WARRANTS_SOURCE, FAILED_WARRANTS_SOURCES, SKIPPED_WARRANTS_SOURCES
            return {
                "used": LAST_WARRANTS_SOURCE,
                "failed": FAILED_WARRANTS_SOURCES,
                "skipped": SKIPPED_WARRANTS_SOURCES
            }
        elif category == "funds":
            from .funds import LAST_FUNDS_SOURCE, FAILED_FUNDS_SOURCES, SKIPPED_FUNDS_SOURCES
            return {
                "used": LAST_FUNDS_SOURCE,
                "failed": FAILED_FUNDS_SOURCES,
                "skipped": SKIPPED_FUNDS_SOURCES
            }
        else:
            return {"used": "Unknown", "failed": [], "skipped": []}
    except ImportError:
        return {"used": "Not available", "failed": [], "skipped": []}


def fetch_single_symbol_quote(symbol: str, asset_class: str = "equities") -> dict | None:
    """
    Fetch a single symbol quote using the appropriate category fetcher.
    
    Args:
        symbol: The symbol to fetch (e.g., "AAPL", "BTCUSD", "EURUSD")
        asset_class: The asset class category ("equities", "crypto", "forex", etc.)
    
    Returns:
        Single asset dict or None if fetch failed
    """
    try:
        if asset_class in ["equities", "stocks"]:
            results = fetch_equities_data(symbols=[symbol], max_universe=1)
        elif asset_class in ["crypto", "cryptocurrencies"]:
            results = fetch_crypto_data(symbols=[symbol], max_universe=1)
        elif asset_class in ["forex", "fx"]:
            results = fetch_forex_data(pairs=[symbol], max_universe=1)
        elif asset_class == "commodities":
            results = fetch_commodities_data(symbols=[symbol], max_universe=1)
        elif asset_class == "futures":
            results = fetch_futures_data(symbols=[symbol], max_universe=1)
        elif asset_class == "warrants":
            results = fetch_warrants_data(symbols=[symbol], max_universe=1)
        elif asset_class == "funds":
            results = fetch_funds_data(symbols=[symbol], max_universe=1)
        else:
            # Default to equities
            results = fetch_equities_data(symbols=[symbol], max_universe=1)
        
        return results[0] if results else None
    except Exception:
        return None


__all__ = [
    "fetch_equities_data",
    "fetch_crypto_data", 
    "fetch_forex_data",
    "fetch_commodities_data",
    "fetch_futures_data",
    "fetch_warrants_data",
    "fetch_funds_data",
    "diagnostics_for",
    "fetch_single_symbol_quote",
]
