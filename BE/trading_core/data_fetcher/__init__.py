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

__all__ = [
    "fetch_equities_data",
    "fetch_crypto_data",
    "fetch_forex_data",
    "fetch_commodities_data",
    "fetch_futures_data",
    "fetch_warrants_data",
    "fetch_funds_data",
]
