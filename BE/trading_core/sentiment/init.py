"""
Regional+fallback sentiment facade.

Exports category-specific fetchers that accept:
- region: "Americas" | "Europe" | "Middle East & Africa" | "Asia-Pacific" | None
- market: exchange key (optional)
- tickers: optional list to prioritize mentions
- limit: cap

Each returns (headlines, diagnostics).
"""

from .equities import fetch_equities_sentiment
from .commodities import fetch_commodities_sentiment
from .futures import fetch_futures_sentiment
from .warrants import fetch_warrants_sentiment
from .funds import fetch_funds_sentiment

__all__ = [
    "fetch_equities_sentiment",
    "fetch_commodities_sentiment",
    "fetch_futures_sentiment",
    "fetch_warrants_sentiment",
    "fetch_funds_sentiment",
]
