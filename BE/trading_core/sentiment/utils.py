from __future__ import annotations

import re
import time
from typing import List, Tuple, Optional, Dict
import requests
import feedparser
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 12  # seconds
USER_AGENT = "TA-Sentiment/1.0 (+https://example.local)"

# very small, transparent lexicon (can be extended or swapped with VADER later)
_POS_WORDS = {
    "beat", "beats", "surge", "soar", "soars", "gain", "gains", "bullish", "record",
    "strong", "upgrade", "upgrades", "buy", "outperform", "positive", "rally",
}
_NEG_WORDS = {
    "miss", "misses", "fall", "falls", "plunge", "plunges", "drop", "drops",
    "bearish", "downgrade", "downgrades", "sell", "underperform", "negative",
    "lawsuit", "probe", "fraud", "ban", "hack", "exploit",
}

_HTML_TAG = re.compile(r"<[^>]+>")

def clean_text(s: str) -> str:
    s = s or ""
    s = BeautifulSoup(s, "html.parser").get_text(" ")
    s = _HTML_TAG.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fetch_rss_titles(url: str, limit: int = 50) -> Tuple[List[str], Optional[str]]:
    """
    Returns (titles, error). Titles are cleaned; dedup left to caller if merging feeds.
    """
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        titles: List[str] = []
        for e in feed.entries[:limit]:
            title = clean_text(getattr(e, "title", "") or getattr(e, "summary", ""))
            if title:
                titles.append(title)
        return titles, None
    except Exception as e:
        return [], str(e)

def dedup_preserve_order(items: List[str], max_items: Optional[int] = None) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if max_items and len(out) >= max_items:
            break
    return out

def simple_sentiment_score(text: str) -> float:
    """
    Crude lexicon-based score in [-1, 1].
    Positive if more positive tokens; negative if more negative tokens.
    """
    if not text:
        return 0.0
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    pos = sum(1 for t in tokens if t in _POS_WORDS)
    neg = sum(1 for t in tokens if t in _NEG_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / float(pos + neg)

def annotate_with_scores(headlines: List[str]) -> List[Dict[str, float]]:
    """
    Turn headlines into a list of dicts: {"text": ..., "score": float}
    (Kept optional; your pipeline primarily needs raw titles, but this can help debugging.)
    """
    return [{"text": h, "score": simple_sentiment_score(h)} for h in headlines]

def fetch_json(url: str, params: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
    try:
        resp = requests.get(url, params=params or {}, timeout=REQUEST_TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)

def diagnostics_entry(source: str, success: bool, count: int, error: Optional[str]) -> Dict[str, Optional[str]]:
    return {
        "source": source,
        "success": bool(success),
        "headline_count": int(count),
        "error": error if not success else None,
    }
