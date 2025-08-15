"""
crypto.py
─────────
Raw crypto sentiment fetchers → headlines + diagnostics.

Design
• Global-first sources (crypto trades 24/7, not exchange-bound).
• Multiple RSS feeds with graceful fallbacks.
• Optional “social” feeds (Reddit subreddits) you can toggle.
• Crypto Fear & Greed Index (alternative.me) included as a diagnostic item.
• Output shape mirrors equities.py:
    -> returns (headlines, diagnostics)
       - headlines: de-duplicated list of strings (titles only)
       - diagnostics: [{source, success, headline_count, error, extra?}, ...]

Dependencies
• trading_core/sentiment/utils.py  (get_rss_titles, dedup_titles)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import os
import requests

from .utils import get_rss_titles, dedup_titles

# Tuning
MAX_PER_SOURCE   = 12     # hard cap per feed (before de-dup)
MAX_HEADLINES    = 30     # final cap we return
REQUEST_TIMEOUT  = 12

# Primary crypto news feeds (broad + liquid coverage)
NEWS_FEEDS: Dict[str, str] = {
    "CoinDesk":      "https://feeds.feedburner.com/CoinDesk",
    "CoinTelegraph": "https://cointelegraph.com/rss",
    "Decrypt":       "https://decrypt.co/feed",
}

# Optional social/reddit feeds
SOCIAL_FEEDS: Dict[str, str] = {
    "Reddit r/CryptoCurrency": "https://www.reddit.com/r/CryptoCurrency/.rss",
    "Reddit r/Bitcoin":        "https://www.reddit.com/r/Bitcoin/.rss",
    "Reddit r/Ethereum":       "https://www.reddit.com/r/ethereum/.rss",
}

# Alternative.me – Crypto Fear & Greed Index
FNG_URL = "https://api.alternative.me/fng/?limit=1"


def _fetch_fear_greed_index() -> Tuple[Optional[int], Dict[str, object]]:
    """
    Fetch the latest Crypto Fear & Greed Index value.
    Returns:
        (value_0_100 or None, diagnostic dict)
    """
    diag: Dict[str, object] = {
        "source": "FearGreedIndex",
        "success": False,
        "headline_count": 0,
        "error": None,
        "extra": {},
    }
    try:
        resp = requests.get(FNG_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        js = resp.json()
        data = js.get("data", [])
        if data:
            v = data[0]
            value_str = v.get("value")
            classification = v.get("value_classification")
            # risky: value could be str, convert safely
            value = int(value_str) if value_str is not None and str(value_str).isdigit() else None
            diag["success"] = True
            diag["extra"] = {
                "value": value,
                "classification": classification,
                "timestamp": v.get("timestamp"),
            }
            return value, diag  # we don't contribute to "headline_count"
        diag["error"] = "empty data"
        return None, diag
    except Exception as e:
        diag["error"] = str(e)
        return None, diag


def _collect_from_feeds(feed_map: Dict[str, str]) -> Tuple[List[str], List[Dict[str, object]]]:
    """
    Iterate over given feeds and gather titles with per-source diagnostics.
    """
    all_titles: List[str] = []
    diags: List[Dict[str, object]] = []
    for name, url in feed_map.items():
        try:
            titles = get_rss_titles(url, MAX_PER_SOURCE)
            success = len(titles) > 0
            diags.append({
                "source": name,
                "success": success,
                "headline_count": len(titles),
                "error": None if success else "empty",
            })
            all_titles.extend(titles)
        except Exception as e:
            diags.append({
                "source": name,
                "success": False,
                "headline_count": 0,
                "error": str(e),
            })
    return all_titles, diags


def fetch_crypto_sentiment(
    *,
    include_social: bool = True,
    region: Optional[str] = None,     # accepted for API compatibility; not used (crypto is global)
    market: Optional[str] = None,     # accepted for API compatibility; not used (crypto is global)
) -> Tuple[List[str], List[Dict[str, object]]]:
    """
    Aggregate crypto sentiment headlines + diagnostics.

    Args
    ----
    include_social : bool
        If True, also include Reddit feeds (r/CryptoCurrency/Bitcoin/Ethereum).
    region, market : Optional[str]
        Unused for crypto (global market) but accepted to keep CLI/API uniform.

    Returns
    -------
    (headlines, diagnostics)
      - headlines: de-duplicated, capped at MAX_HEADLINES
      - diagnostics: per source info + Fear/Greed diagnostic item
    """
    # News first
    news_titles, news_diags = _collect_from_feeds(NEWS_FEEDS)

    # Optional social
    social_titles: List[str] = []
    social_diags: List[Dict[str, object]] = []
    if include_social:
        social_titles, social_diags = _collect_from_feeds(SOCIAL_FEEDS)

    # Fear & Greed
    fng_value, fng_diag = _fetch_fear_greed_index()

    # Combine and de-dup
    all_titles = dedup_titles(news_titles + social_titles)[:MAX_HEADLINES]
    diagnostics: List[Dict[str, object]] = news_diags + social_diags + [fng_diag]

    return all_titles, diagnostics


# Convenience, used by some codebases that call category-specific names:
def fetch_crypto_headlines_only(max_items: int = MAX_HEADLINES) -> List[str]:
    """
    Return just a de-duplicated list of titles (news+social), capped.
    """
    titles, _ = fetch_crypto_sentiment(include_social=True)
    return titles[:max_items]


__all__ = [
    "fetch_crypto_sentiment",
    "fetch_crypto_headlines_only",
]
