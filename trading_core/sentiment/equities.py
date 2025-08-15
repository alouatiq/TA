from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

from .utils import (
    fetch_rss_titles,
    dedup_preserve_order,
    diagnostics_entry,
)

# Core RSS sources (no tickers hard-coded; broad market)
SOURCES_GLOBAL = {
    "Reddit r/stocks": "https://www.reddit.com/r/stocks/.rss",
    "MarketWatch":     "https://www.marketwatch.com/feeds/topstories",
    "Yahoo Finance":   "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
    # add more if needed: Bloomberg free feeds (if available), Reuters RSS mirrors, FT (if accessible)
}

# Region/market-aware extras (best-effort; use when market hint is provided)
SOURCES_BY_MARKET = {
    # Examples; safe to extend
    "LSE":    {"Yahoo UK": "https://uk.finance.yahoo.com/rss/topstories"},
    "XETRA":  {"Yahoo DE": "https://de.finance.yahoo.com/rss/topstories"},
    "EN_PA":  {"Yahoo FR": "https://fr.finance.yahoo.com/rss/topstories"},
    "SIX":    {"Yahoo CH": "https://ch.finance.yahoo.com/rss/topstories"},
    "TSX":    {"Yahoo CA": "https://ca.finance.yahoo.com/rss/topstories"},
    "TSE":    {"Yahoo JP": "https://finance.yahoo.co.jp/rss/topstories"},
    "HKEX":   {"Yahoo HK": "https://hk.finance.yahoo.com/rss/topstories"},
}

def fetch_equities_sentiment(market: Optional[str] = None, limit: int = 40) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Returns (headlines, diagnostics).
    - headlines: unique cleaned titles (≤ limit)
    - diagnostics: per-source status
    """
    diagnostics: List[Dict[str, Any]] = []
    merged: List[str] = []

    # 1) Global sources
    for name, url in SOURCES_GLOBAL.items():
        titles, err = fetch_rss_titles(url, limit=limit)
        diagnostics.append(diagnostics_entry(name, bool(titles), len(titles), err))
        merged.extend(titles)

    # 2) Optional market-specific flavor
    if market and market in SOURCES_BY_MARKET:
        for name, url in SOURCES_BY_MARKET[market].items():
            titles, err = fetch_rss_titles(url, limit=limit)
            diagnostics.append(diagnostics_entry(name, bool(titles), len(titles), err))
            merged.extend(titles)

    # 3) Deduplicate and cap
    headlines = dedup_preserve_order(merged, max_items=limit)

    return headlines, diagnostics
