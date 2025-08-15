"""
sentiment.utils
───────────────
Lightweight helpers for RSS fetching, deduplication, naive sentiment scoring,
and filtering headlines by tickers/keywords.

Dependencies: requests, feedparser, beautifulsoup4
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable, Optional
import time
import re

import requests
import feedparser
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 12
MAX_PER_FEED = 30

_WORD_SPLIT = re.compile(r"[^\w$]+", re.UNICODE)

# Very small lexicon (avoid heavy deps). These are *naive* and only for rough scoring.
POS_WORDS = {
    "beats","beat","surge","rally","rallies","record","growth","upgrade","upgrades",
    "strong","accelerate","bullish","tops","soars","solid","profit","profits","raises",
    "raise","positive","optimistic","expands","expansion","gains","higher","rebound"
}
NEG_WORDS = {
    "miss","misses","plunge","plunges","fall","falls","cuts","cut","downgrade","downgrades",
    "weak","slowdown","bearish","lower","drop","drops","decline","declines","loss","losses",
    "warns","warning","negative","recall","lawsuit","default","bankruptcy","fraud","frauds"
}

def _clean_html(text: str) -> str:
    return BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)

def fetch_rss_titles(url: str, limit: int = MAX_PER_FEED) -> Tuple[List[str], Optional[str]]:
    """Return (titles, error)."""
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        titles: List[str] = []
        for e in parsed.entries[:limit]:
            t = _clean_html(getattr(e, "title", "") or getattr(e, "summary", ""))
            if t:
                titles.append(t.strip())
        return titles, None
    except Exception as e:
        return [], str(e)

def dedup_preserve(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def naive_sentiment_score(headline: str) -> float:
    """
    Very naive sentiment: (#pos - #neg) / max(1, tokens) in [-1, +1].
    Intended only for rough ranking/weighting.
    """
    if not headline:
        return 0.0
    tokens = [w.lower() for w in _WORD_SPLIT.split(headline) if w]
    if not tokens:
        return 0.0
    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    score = (pos - neg) / max(1, len(tokens))
    return max(-1.0, min(1.0, score))

def rank_and_trim(headlines: List[str], limit: int = 30) -> List[str]:
    # Sort by absolute sentiment signal strength as a proxy for relevance
    ranked = sorted(headlines, key=lambda h: abs(naive_sentiment_score(h)), reverse=True)
    return ranked[:limit]

def filter_by_tickers(headlines: List[str], tickers: Optional[List[str]] = None) -> List[str]:
    """
    If tickers provided, prioritize headlines that mention any of them (case-insensitive).
    """
    if not tickers:
        return headlines
    pats = [re.compile(rf"\b{re.escape(t)}\b", re.I) for t in tickers if t]
    if not pats:
        return headlines

    preferred, others = [], []
    for h in headlines:
        if any(p.search(h) for p in pats):
            preferred.append(h)
        else:
            others.append(h)
    return preferred + others

def merge_source_batches(batches: List[Tuple[str, List[str], Optional[str]]], max_total: int = 30) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    batches: list of (source_name, titles, error)
    Produces a unique list of headlines (capped) + diagnostics per source.
    """
    diags: List[Dict[str, Any]] = []
    all_titles: List[str] = []
    for name, titles, err in batches:
        ok = bool(titles)
        diags.append({
            "source": name,
            "success": ok,
            "headline_count": len(titles),
            "error": None if ok else err
        })
        all_titles.extend(titles)

    unique = dedup_preserve(all_titles)
    return unique[:max_total], diags
