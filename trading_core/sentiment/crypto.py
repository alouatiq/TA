from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any, Optional

from .utils import (
    fetch_rss_titles,
    fetch_json,
    dedup_preserve_order,
    diagnostics_entry,
)

# Core crypto news feeds
CRYPTO_FEEDS = {
    "CoinDesk":      "https://feeds.feedburner.com/CoinDesk",
    "CoinTelegraph": "https://cointelegraph.com/rss",
}

# Optional: Fear & Greed Index (alternative.me)
FNG_URL = "https://api.alternative.me/fng/"

def _fetch_fear_greed() -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (headline_snippet, error).
    """
    data, err = fetch_json(FNG_URL)
    if err or not data:
        return None, err or "no data"
    try:
        # pick the latest value/classification
        latest = (data.get("data") or [])[0]
        value = latest.get("value")
        cls = latest.get("value_classification")
        headline = f"Crypto Fear & Greed: {value} ({cls})"
        return headline, None
    except Exception as e:
        return None, str(e)

def fetch_crypto_sentiment(limit: int = 40) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (headlines, diagnostics).
    Uses RSS + Fear & Greed snapshot. Social APIs can be plugged later.
    """
    diagnostics: List[Dict[str, Any]] = []
    merged: List[str] = []

    # 1) News feeds
    for name, url in CRYPTO_FEEDS.items():
        titles, err = fetch_rss_titles(url, limit=limit)
        diagnostics.append(diagnostics_entry(name, bool(titles), len(titles), err))
        merged.extend(titles)

    # 2) Fear & Greed
    fng_title, fng_err = _fetch_fear_greed()
    diagnostics.append(diagnostics_entry("Fear&Greed", bool(fng_title), 1 if fng_title else 0, fng_err))
    if fng_title:
        merged.append(fng_title)

    # 3) Dedup & cap
    headlines = dedup_preserve_order(merged, max_items=limit)

    return headlines, diagnostics
