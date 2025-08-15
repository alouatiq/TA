"""
sentiment.futures
─────────────────
Region-aware sentiment for equity index futures (via broad market news).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

from .utils import fetch_rss_titles, merge_source_batches, rank_and_trim, filter_by_tickers

MAX_TOTAL = 30

REGIONAL_FEEDS: Dict[str, List[str]] = {
    "Americas": [
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://feeds2.feedburner.com/marketwatch/marketpulse",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    ],
    "Europe": [
        "https://www.euronews.com/rss?level=theme&name=business",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ESTOXX50E&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EFTSE&region=US&lang=en-US",
    ],
    "Middle East & Africa": [
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "Asia-Pacific": [
        "https://asia.nikkei.com/rss/feed/nar",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EN225&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EHSI&region=US&lang=en-US",
    ],
}

GLOBAL_FEEDS = [
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds2.feedburner.com/marketwatch/marketpulse",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
]

def _choose_feeds(region: Optional[str]) -> List[str]:
    if region and region in REGIONAL_FEEDS:
        return REGIONAL_FEEDS[region]
    return GLOBAL_FEEDS

def fetch_futures_sentiment(
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

    if region:
        for url in GLOBAL_FEEDS:
            titles, err = fetch_rss_titles(url, limit=limit)
            batches.append(("global", titles, err))

    merged, diags = merge_source_batches(batches, max_total=limit)
    if not merged:
        return [], diags

    merged = filter_by_tickers(merged, tickers)
    merged = rank_and_trim(merged, limit=limit)
    return merged, diags
