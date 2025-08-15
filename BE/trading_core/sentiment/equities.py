"""
sentiment.equities
──────────────────
Region-aware sentiment for equities with tiered fallbacks.

Inputs:
- region: one of {"Americas","Europe","Middle East & Africa","Asia-Pacific"} or None
- market: optional exchange key (e.g., "LSE","XETRA"); used to bias index picks
- tickers: optional list to prioritize in ranking
- limit: cap total headlines

Output:
- (headlines, diagnostics)
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional

from .utils import fetch_rss_titles, merge_source_batches, rank_and_trim, filter_by_tickers

MAX_TOTAL = 30

# --- Tier 1: regional finance/news feeds (best effort; may partially overlap globally)
REGIONAL_FEEDS: Dict[str, List[str]] = {
    "Americas": [
        # Yahoo US indices headlines (stable endpoint format)
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI&region=US&lang=en-US",
        # CNBC Markets
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        # MarketWatch
        "https://feeds2.feedburner.com/marketwatch/marketpulse",
    ],
    "Europe": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5ESTOXX50E&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EFTSE&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGDAXI&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EFCHI&region=US&lang=en-US",
        # Euronews business
        "https://www.euronews.com/rss?level=theme&name=business",
    ],
    "Middle East & Africa": [
        # Broad finance/business region feeds (fallback if local portals not available)
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.aljazeera.com/xml/rss/all.xml",
    ],
    "Asia-Pacific": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EN225&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EHSI&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EAXJO&region=US&lang=en-US",
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EBSESN&region=US&lang=en-US",
        # Nikkei Asia business (limited headlines)
        "https://asia.nikkei.com/rss/feed/nar",
    ],
}

# --- Tier 2: global finance feeds (used if regional is thin)
GLOBAL_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EIXIC&region=US&lang=en-US",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds2.feedburner.com/marketwatch/marketpulse",
]

# --- Tier 3: social market chatter (kept last; optional)
SOCIAL_FEEDS = [
    "https://www.reddit.com/r/stocks/.rss",
]

def _choose_feeds(region: Optional[str]) -> List[str]:
    if region and region in REGIONAL_FEEDS:
        return REGIONAL_FEEDS[region]
    return GLOBAL_FEEDS

def fetch_equities_sentiment(
    *,
    region: Optional[str] = None,
    market: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    limit: int = MAX_TOTAL
) -> Tuple[List[str], List[Dict[str, Any]]]:
    batches: List[Tuple[str, List[str], Optional[str]]] = []

    # Tier 1
    for url in _choose_feeds(region):
        titles, err = fetch_rss_titles(url, limit=limit)
        batches.append(("regional" if region else "global", titles, err))

    # Tier 2 (global fallbacks)
    if region:
        for url in GLOBAL_FEEDS:
            titles, err = fetch_rss_titles(url, limit=limit)
            batches.append(("global", titles, err))

    # Tier 3 (social last)
    for url in SOCIAL_FEEDS:
        titles, err = fetch_rss_titles(url, limit=limit)
        batches.append(("social", titles, err))

    merged, diags = merge_source_batches(batches, max_total=limit)
    if not merged:
        return [], diags

    merged = filter_by_tickers(merged, tickers)
    merged = rank_and_trim(merged, limit=limit)
    return merged, diags
