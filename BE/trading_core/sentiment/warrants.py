"""
sentiment.warrants
──────────────────
There are few warrant/ETP-specific news feeds; we approximate using regional
equities + markets feeds, then bias headlines with leveraged ETP tickers if provided.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

from .equities import fetch_equities_sentiment

def fetch_warrants_sentiment(
    *,
    region: Optional[str] = None,
    market: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    limit: int = 30
) -> Tuple[List[str], List[Dict[str, Any]]]:
    # Reuse equities sentiment (regional), which best reflects warrant underlyings.
    return fetch_equities_sentiment(region=region, market=market, tickers=tickers, limit=limit)
