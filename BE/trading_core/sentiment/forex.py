"""
forex.py
────────
Pair-agnostic FX sentiment fetcher.

Key properties
• No hard-coded currency pairs.
• If `pairs` are provided, we filter headlines to those pairs.
• If `pairs` are not provided, we AUTO-DISCOVER trending pairs by scanning FX news feeds
  and extracting pair tokens (EUR/USD, USDJPY, GBPJPY, etc) using ISO-4217 validation.
• Region/market arguments are accepted for future routing, but not required.

Returns:
  (headlines: List[str], diagnostics: List[dict])

Diagnostics entries look like:
  {"source": "DailyFX", "success": True, "headline_count": 10, "error": None}
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple, Dict, Any, Optional

import feedparser
from bs4 import BeautifulSoup

# Keep this small; we’re not fixing/forcing pairs here.
MAX_HEADLINES = 10
REQUEST_TIMEOUT = 10  # feedparser handles internally; kept for symmetry with others

# Reputable FX news feeds (global). We DO NOT fix any pairs here.
FEEDS: Dict[str, str] = {
    "DailyFX":   "https://www.dailyfx.com/feeds/all",
    "ForexLive": "https://www.forexlive.com/feed/",
    # FXStreet generic RSS; if unavailable, it will just fail gracefully.
    "FXStreet":  "https://www.fxstreet.com/rss",
    # Economic calendar (macro) as a weak signal; might have fewer pair-specific titles.
    "ForexFactory": "https://www.forexfactory.com/ffcal_week_this.xml",
}

# ISO‑4217 currency codes (static standard list, not pairs).
# This is NOT a list of hard-coded pairs; it’s used only to validate tokens seen in headlines.
# (You can move this into a shared constants module later if you prefer.)
ISO_CCY = {
    "USD","EUR","JPY","GBP","AUD","CAD","CHF","NZD","CNH","CNY","SEK","NOK","DKK",
    "ZAR","MXN","BRL","RUB","HKD","SGD","TRY","INR","KRW","PLN","HUF","CZK","ILS",
    "AED","SAR","QAR","KWD","THB","MYR","IDR","PHP","TWD"
}

_PAIR_RE = re.compile(r"\b([A-Z]{3})[\/]?\s?([A-Z]{3})\b")

def _clean_text(htmlish: str) -> str:
    return BeautifulSoup(htmlish or "", "html.parser").get_text(" ", strip=True)

def _fetch_feed_titles(url: str, limit: int) -> Tuple[List[str], Optional[str]]:
    try:
        feed = feedparser.parse(url)
        titles: List[str] = []
        for e in feed.entries[:limit]:
            title = _clean_text(getattr(e, "title", "") or getattr(e, "summary", ""))
            if title:
                titles.append(title)
        return titles, None
    except Exception as e:
        return [], str(e)

def _normalize_pairs(pairs: Iterable[str]) -> List[str]:
    """
    Normalize input pairs to uppercase compact format (e.g., 'EURUSD'), and
    also track alternate 'EUR/USD' representation for matching.
    """
    out: List[str] = []
    seen = set()
    for p in pairs:
        if not p:
            continue
        p_up = p.upper().replace(" ", "")
        p_compact = p_up.replace("/", "")
        if len(p_compact) == 6 and p_compact.isalpha():
            if p_compact not in seen:
                seen.add(p_compact)
                out.append(p_compact)
    return out

def _pairs_from_headlines(all_titles: List[str], max_pairs: int = 12) -> List[str]:
    """
    Discover pairs by scanning titles for patterns like 'EUR/USD' or 'USDJPY',
    then validate each side against ISO currency codes. Returns compact 'EURUSD' form.
    """
    found: Dict[str, int] = {}
    for t in all_titles:
        for m in _PAIR_RE.finditer(t.upper()):
            a, b = m.group(1), m.group(2)
            if a in ISO_CCY and b in ISO_CCY:
                compact = f"{a}{b}"
                found[compact] = found.get(compact, 0) + 1
    # sort by frequency (descending), then alphabetically to stabilize
    ranked = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    return [p for p, _score in ranked[:max_pairs]]

def _filter_by_pairs(titles: Iterable[str], pairs_compact: List[str]) -> List[str]:
    """
    Keep titles mentioning any of the requested pairs (in compact or slash form).
    """
    out: List[str] = []
    seen = set()
    # also consider 'EUR/USD' form for each requested pair
    slash_forms = {f"{p[:3]}/{p[3:]}" for p in pairs_compact}
    for title in titles:
        t_up = title.upper()
        hit = False
        for p in pairs_compact:
            if p in t_up:
                hit = True
                break
        if not hit:
            for s in slash_forms:
                if s in t_up:
                    hit = True
                    break
        if hit and title not in seen:
            seen.add(title)
            out.append(title)
    return out

def fetch_forex_sentiment(
    pairs: Optional[List[str]] = None,
    *,
    market: Optional[str] = None,
    region: Optional[str] = None,
    limit: int = MAX_HEADLINES
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Pair-agnostic FX sentiment.
    - If `pairs` are provided, filter to those pairs only.
    - If `pairs` are None/empty, auto-discover popular pairs from current headlines (no hard-coded list).
    - `market` and `region` are accepted for future routing (e.g., source selection by locale).

    Returns: (headlines, diagnostics)
    """
    diagnostics: List[Dict[str, Any]] = []
    all_titles: List[str] = []
    per_source_titles: Dict[str, List[str]] = {}

    # 1) Pull a small batch from each source
    for name, url in FEEDS.items():
        titles, err = _fetch_feed_titles(url, limit*2)  # over-fetch a bit; we'll filter later
        success = bool(titles)
        per_source_titles[name] = titles
        all_titles.extend(titles)
        diagnostics.append({
            "source": name,
            "success": success,
            "headline_count": len(titles),
            "error": None if success else err
        })

    # 2) Decide which pairs to care about
    target_pairs = _normalize_pairs(pairs or [])
    if not target_pairs:
        # Auto-discover from headlines (frequency-ranked)
        target_pairs = _pairs_from_headlines(all_titles, max_pairs=12)

    # 3) Filter titles by selected pairs (if we discovered any)
    final_titles: List[str] = []
    if target_pairs:
        for name, ts in per_source_titles.items():
            final_titles.extend(_filter_by_pairs(ts, target_pairs))
    else:
        # If we discovered no pairs (rare), just return top global FX headlines
        final_titles = all_titles[: limit * 2]

    # 4) Deduplicate and cap
    seen = set()
    unique: List[str] = []
    for t in final_titles:
        if t not in seen:
            seen.add(t)
            unique.append(t)
        if len(unique) >= limit:
            break

    return unique, diagnostics
