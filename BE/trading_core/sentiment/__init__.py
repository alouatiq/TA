"""
trading_core.sentiment
──────────────────────
Raw headline fetchers → normalized signals (package entrypoint).

This module exposes:
- fetch_equities_sentiment(market: Optional[str], limit: int) -> Tuple[List[str], List[dict]]
- fetch_crypto_sentiment(market: Optional[str], limit: int)   -> Tuple[List[str], List[dict]]
- fetch_sentiment(category: str, market: Optional[str], limit: int) -> Tuple[List[str], List[dict]]

Design notes
------------
• Returns a pair: (unique_headlines, diagnostics), where diagnostics is a list of dicts:
    {
      "source": "<name>",
      "success": bool,
      "headline_count": int,
      "error": Optional[str]
    }
• `market` is optional and may be used by category fetchers for market-aware feeds.
• No assets are hardcoded here; category modules own their sources and logic.
• Safe to import from both CLI and future web API layers.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional

# Re-export category fetchers
from .equities import fetch_equities_sentiment as _fetch_equities_sentiment
from .crypto import fetch_crypto_sentiment as _fetch_crypto_sentiment

__all__ = [
    "fetch_equities_sentiment",
    "fetch_crypto_sentiment",
    "fetch_sentiment",
]


def fetch_equities_sentiment(
    market: Optional[str] = None,
    *,
    limit: int = 10,
) -> Tuple[List[str], List[Dict]]:
    """
    Thin wrapper around sentiment.equities.fetch_equities_sentiment to keep
    a consistent import surface for callers.
    """
    return _fetch_equities_sentiment(market=market, limit=limit)


def fetch_crypto_sentiment(
    market: Optional[str] = None,
    *,
    limit: int = 10,
) -> Tuple[List[str], List[Dict]]:
    """
    Thin wrapper around sentiment.crypto.fetch_crypto_sentiment to keep
    a consistent import surface for callers.
    """
    return _fetch_crypto_sentiment(market=market, limit=limit)


def fetch_sentiment(
    category: str,
    *,
    market: Optional[str] = None,
    limit: int = 10,
) -> Tuple[List[str], List[Dict]]:
    """
    Unified sentiment fetcher used by higher-level orchestration.

    Parameters
    ----------
    category : str
        One of: "equities", "stocks", "crypto", "cryptocurrencies".
        (Aliases are handled leniently.)
    market : Optional[str]
        Optional market key (e.g., "LSE", "NYSE", "TADAWUL") that downstream
        fetchers may use to prefer local sources.
    limit : int
        Cap on unique headlines returned.

    Returns
    -------
    Tuple[List[str], List[Dict]]
        (unique_headlines, diagnostics)
    """
    key = (category or "").strip().lower()

    if key in {"equities", "stocks", "stock"}:
        return fetch_equities_sentiment(market=market, limit=limit)

    if key in {"crypto", "cryptos", "cryptocurrencies", "coin"}:
        return fetch_crypto_sentiment(market=market, limit=limit)

    # Unknown category → empty but well-formed diagnostics
    return (
        [],
        [
            {
                "source": "aggregate",
                "success": False,
                "headline_count": 0,
                "error": f"unsupported category '{category}'",
            }
        ],
    )
