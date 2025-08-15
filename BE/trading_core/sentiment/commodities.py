"""
sentiment.commodities
─────────────────────
Region-aware sentiment for commodities (metals focus). Falls back to global metals/news.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

from .utils import fetch_rss_titles, merge_source_batches, rank_and_trim, filter_by_tickers

MAX_TOTAL = 30

REGIONAL_FEEDS: Dict[str, List[str]] = {
    "Americas": [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/commodities.rss",
        "https://www.kitco.com/rss/news",  # Kitco metals
    ],
    "Europe": [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/commodities.rss",
    ],
    "Middle East & Africa": [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/commodities.rss",
        # Reuters business as broad fallback
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    ],
    "Asia-Pacific": [
        "https://www.mining.com/feed/",
        "https://www.investing.com/rss/commodities.rss",
        "https://asia.nikkei.com/rss/feed/nar",
    ],
}

GLOBAL_FEEDS = [
    "https://www.mining.com/feed/",
    "https://www.investing.com/rss/commodities.rss",
    "https://www.kitco.com/rss/news",
]

def _choose_feeds(region: Optional[str]) -> List[str]:
    if region and region in REGIONAL_FEEDS:
        return REGIONAL_FEEDS[region]
    return GLOBAL_FEEDS

def fetch_commodities_sentiment(
    *,
    region: Optional[str] = None,
    market: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    limit: int = MAX_TOTAL
) -> Tuple[List[str], List[Dict[str, Any]]]:
    batches: List[Tuple[str, List[str], Optional[str]]] = []

    for url in _choose_feeds(region):
        titles, err = fetch_rss_titles(url, limit=limit)
        batches.append(("regional" if region else "global", titles, err))

    # Always add global fallbacks to enrich
    for url in GLOBAL_FEEDS:
        titles, err = fetch_rss_titles(url, limit=limit)
        batches.append(("global", titles, err))

    merged, diags = merge_source_batches(batches, max_total=limit)
    if not merged:
        return [], diags

    merged = filter_by_tickers(merged, tickers)
    merged = rank_and_trim(merged, limit=limit)
    return merged, diags
