"""
BE/trading_core/indicators/sentiment.py
───────────────────────────────────────
Lightweight, dependency-tolerant sentiment feature extraction and aggregation.

This module turns raw text inputs (news headlines, social snippets) and optional
numeric gauges (e.g., Fear & Greed Index) into normalized sentiment signals
in the range [0, 1], where:
    0.00 = very negative
    0.50 = neutral
    1.00 = very positive

It is designed to run *even if advanced NLP libraries are missing*:
- Prefers VADER (if installed) for robust English sentiment scoring.
- Falls back to a simple lexicon/keyword approach when VADER is unavailable.
- Works with any language, but the fallback lexicon is English-focused.

Outputs are suitable inputs to the broader scoring/weights engine:
- `score_headlines(...)` : headline sentiment (with per-headline diagnostics)
- `score_social(...)`    : social/media snippet sentiment (same shape)
- `normalize_fear_greed(...)` : maps raw FGI values to [0, 1]
- `aggregate_sentiment(...)`  : blends components into a composite sentiment score

Typical usage (CLI or strategy layer):
--------------------------------------
    from trading_core.indicators.sentiment import SentimentScorer

    scorer = SentimentScorer()
    heads_summary = scorer.score_headlines(headlines)
    social_summary = scorer.score_social(tweets_or_reddit)
    fg = scorer.normalize_fear_greed(crypto_fgi_value)  # 0..100 or dict/str known formats

    agg = scorer.aggregate_sentiment(
        headlines=heads_summary,
        social=social_summary,
        fear_greed=fg,
        weights={"headlines": 0.5, "social": 0.3, "fear_greed": 0.2},
        regime_hint="volatile",  # optional: 'bull'|'bear'|'range'|'volatile'
    )

    # agg = {
    #   "score": 0.63,
    #   "components": {...},
    #   "diagnostics": {...}
    # }

Design goals:
- Deterministic, explainable signal extraction with graceful degradation.
- Minimal assumptions about upstream fetchers; accepts raw strings and lists.
- Ready for fusion with technical/fundamental/microstructure signals.

Author’s note:
- If you add `vaderSentiment` to requirements, you’ll get more nuanced scores.
- The fallback lexicon is conservative to reduce false positives from “clickbait”.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import math
import statistics

# Optional VADER import (preferred)
_VADER_AVAILABLE = False
try:
    # First try the modern name
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER_AVAILABLE = True
except Exception:
    try:
        # Some distros package under nltk.sentiment
        from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
        _VADER_AVAILABLE = True
    except Exception:
        _VADER_AVAILABLE = False


# ────────────────────────────────────────────────────────────
#  Sanitization & Fallback Lexicon
# ────────────────────────────────────────────────────────────

_CLEAN_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")
_TICKER_RE = re.compile(r"\$?[A-Z]{1,5}(\.[A-Z]{1,3})?")  # e.g., AAPL, TSLA, RY.TO
_NUM_RE = re.compile(r"[\+\-]?\d+(\.\d+)?%?")

# Conservative lexicon (English); extend as needed.
_POS_WORDS = {
    "beat", "beats", "beat-estimates", "outperform", "outperforms",
    "surge", "soar", "rally", "bullish", "upgrade", "upgraded",
    "record", "growth", "profit", "profits", "strong", "robust",
    "positive", "optimistic", "buyback", "dividend", "topline",
    "guidance-raise", "raise-guidance", "oversubscribed", "breakthrough",
    "approval", "partnership", "contract-win", "whale-buy",
}

_NEG_WORDS = {
    "miss", "misses", "miss-estimates", "underperform", "downgrade", "downgraded",
    "drop", "plunge", "tank", "bearish", "lawsuit", "probe", "investigation",
    "fraud", "loss", "losses", "weak", "negative", "pessimistic", "profit-warning",
    "cut-guidance", "guidance-cut", "default", "bankruptcy", "layoffs", "recall",
    "halt", "delist", "whale-sell",
}

# Phrases that magnify or dampen polarity in the fallback approach
_EMPHASIS_WORDS = {"huge", "massive", "sharp", "unexpected", "shocking", "record", "historic"}
_HEDGE_WORDS = {"might", "may", "could", "possibly", "rumor", "reportedly", "unconfirmed"}


def _clean_text(s: str) -> str:
    """Basic sanitizer for headlines or social snippets."""
    if not s:
        return ""
    s = s.strip()
    s = _URL_RE.sub("", s)            # strip URLs
    s = _CLEAN_RE.sub(" ", s)         # normalize whitespace
    # keep tickers/numbers to preserve context; do not lowercase tickers entirely (false negs with 'US')
    return s


def _fallback_polarity(s: str) -> float:
    """
    Very simple lexicon-based polarity in [-1, +1].
    Counts positive/negative cues and calibrates with emphasis/hedges.
    """
    if not s:
        return 0.0

    txt = s.lower()
    # normalize variants like "beat estimates"
    txt = txt.replace("beat estimates", "beat-estimates")
    txt = txt.replace("miss estimates", "miss-estimates")
    txt = txt.replace("raise guidance", "raise-guidance")
    txt = txt.replace("cut guidance", "cut-guidance")
    txt = txt.replace("out performs", "outperforms")
    txt = txt.replace("under performs", "underperform")
    txt = txt.replace("whale buy", "whale-buy").replace("whale sell", "whale-sell")

    pos = sum(1 for w in _POS_WORDS if w in txt)
    neg = sum(1 for w in _NEG_WORDS if w in txt)
    total = pos + neg

    if total == 0:
        return 0.0

    base = (pos - neg) / total  # in [-1, 1]

    # Emphasis/hedge modifiers
    if any(w in txt for w in _EMPHASIS_WORDS):
        base *= 1.15
    if any(w in txt for w in _HEDGE_WORDS):
        base *= 0.85

    # clamp
    return max(-1.0, min(1.0, base))


def _polarity_to_unit_interval(p: float) -> float:
    """Map polarity in [-1, 1] → [0, 1]."""
    return max(0.0, min(1.0, 0.5 * (p + 1.0)))


# ────────────────────────────────────────────────────────────
#  VADER Wrapper (if available)
# ────────────────────────────────────────────────────────────

@dataclass
class _VaderWrapper:
    analyzer: Any

    def score(self, s: str) -> float:
        """
        Return compound score in [-1, 1] using VADER.
        """
        try:
            return float(self.analyzer.polarity_scores(s).get("compound", 0.0))
        except Exception:
            return 0.0


# ────────────────────────────────────────────────────────────
#  Public Scorer
# ────────────────────────────────────────────────────────────

class SentimentScorer:
    """
    High-level sentiment scorer with graceful fallback.
    """

    def __init__(self, use_vader: Optional[bool] = None) -> None:
        """
        Args:
            use_vader: force enable/disable VADER. If None, auto-detect.
        """
        self._use_vader = _VADER_AVAILABLE if use_vader is None else bool(use_vader)
        self._vader: Optional[_VaderWrapper] = None
        if self._use_vader and _VADER_AVAILABLE:
            try:
                self._vader = _VaderWrapper(SentimentIntensityAnalyzer())
            except Exception:
                self._vader = None
                self._use_vader = False

    # ────────────────────────────────────────────────────────
    # Core sentiment scorers
    # ────────────────────────────────────────────────────────

    def _score_text(self, text: str) -> float:
        """
        Score a single text to [0,1]. Uses VADER if available, else fallback.
        """
        t = _clean_text(text)
        if not t:
            return 0.5  # neutral for empty
        if self._vader:
            pol = self._vader.score(t)  # [-1,1]
        else:
            pol = _fallback_polarity(t)  # [-1,1]
        return _polarity_to_unit_interval(pol)  # [0,1]

    def _summarize_scores(self, entries: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Build a compact summary + useful diagnostics.
        """
        if not entries:
            return {
                "score": 0.5,
                "count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "avg": 0.5,
                "stdev": 0.0,
                "top_positive": [],
                "top_negative": [],
                "items": [],
            }

        scores = [sc for _, sc in entries]
        avg = statistics.fmean(scores)
        stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        pos_mask = [sc > 0.55 for sc in scores]  # mild threshold above neutral
        neg_mask = [sc < 0.45 for sc in scores]

        pos_pct = sum(pos_mask) / len(scores)
        neg_pct = sum(neg_mask) / len(scores)

        # Choose median-ish as summary to reduce outlier impact
        summary = float(sorted(scores)[len(scores)//2])

        # Rank extremes for explainability
        ranked_pos = sorted(entries, key=lambda kv: kv[1], reverse=True)[:3]
        ranked_neg = sorted(entries, key=lambda kv: kv[1])[:3]

        return {
            "score": round(summary, 4),
            "count": len(scores),
            "positive_pct": round(pos_pct, 4),
            "negative_pct": round(neg_pct, 4),
            "avg": round(avg, 4),
            "stdev": round(stdev, 4),
            "top_positive": [{"text": t, "score": round(s, 4)} for t, s in ranked_pos],
            "top_negative": [{"text": t, "score": round(s, 4)} for t, s in ranked_neg],
            "items": [{"text": t, "score": round(s, 4)} for t, s in entries],
        }

    # ────────────────────────────────────────────────────────
    # Public APIs
    # ────────────────────────────────────────────────────────

    def score_headlines(self, headlines: Optional[List[str]]) -> Dict[str, Any]:
        """
        Score news headlines to produce a compact sentiment summary.

        Args:
            headlines: list of headline strings
        Returns:
            dict with keys: score (median), count, positive_pct, negative_pct,
            avg, stdev, top_positive/negative, items (per-headline).
        """
        entries: List[Tuple[str, float]] = []
        if headlines:
            for h in headlines:
                if not h or not isinstance(h, str):
                    continue
                sc = self._score_text(h)
                entries.append((h, sc))
        return self._summarize_scores(entries)

    def score_social(self, posts: Optional[List[str]]) -> Dict[str, Any]:
        """
        Score social snippets (tweets, Reddit comments, etc.).

        Args:
            posts: list of short texts
        Returns:
            same shape as score_headlines
        """
        entries: List[Tuple[str, float]] = []
        if posts:
            # mild dedup by normalized string (lowercase, strip punctuation)
            seen = set()
            for p in posts:
                if not p or not isinstance(p, str):
                    continue
                key = re.sub(r"[^a-z0-9 ]+", "", p.lower()).strip()
                if key in seen:
                    continue
                seen.add(key)
                sc = self._score_text(p)
                entries.append((p, sc))
        return self._summarize_scores(entries)

    # ────────────────────────────────────────────────────────
    # External gauges
    # ────────────────────────────────────────────────────────

    def normalize_fear_greed(self, raw: Union[None, int, float, str, Dict[str, Any]]) -> float:
        """
        Normalize Fear & Greed-like gauges to [0,1].

        Accepts:
          - numeric 0..100 (typical FGI scale)
          - strings containing a number (e.g., "FGI=34")
          - dicts with likely keys ("value", "score")
          - None → neutral (0.5)

        Returns:
          float in [0,1]
        """
        if raw is None:
            return 0.5
        # dict with common keys
        if isinstance(raw, dict):
            for k in ("value", "score", "index", "fgi"):
                if k in raw:
                    try:
                        v = float(raw[k])
                        return max(0.0, min(1.0, v / 100.0))
                    except Exception:
                        pass
            # fallback: any numeric in dict
            for v in raw.values():
                try:
                    f = float(v)
                    return max(0.0, min(1.0, f / 100.0))
                except Exception:
                    continue

        # numeric
        if isinstance(raw, (int, float)):
            return max(0.0, min(1.0, float(raw) / 100.0))

        # string with a number in it
        if isinstance(raw, str):
            m = re.search(r"(\d+(\.\d+)?)", raw)
            if m:
                try:
                    v = float(m.group(1))
                    return max(0.0, min(1.0, v / 100.0))
                except Exception:
                    pass

        # default neutral
        return 0.5

    # ────────────────────────────────────────────────────────
    # Aggregation
    # ────────────────────────────────────────────────────────

    def aggregate_sentiment(
        self,
        *,
        headlines: Optional[Dict[str, Any]] = None,
        social: Optional[Dict[str, Any]] = None,
        fear_greed: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
        regime_hint: Optional[str] = None,  # 'bull'|'bear'|'range'|'volatile'
    ) -> Dict[str, Any]:
        """
        Blend sentiment components into a composite score with explainability.

        Weighting guidance (defaults):
            headlines: 0.5
            social:    0.3
            fear_greed:0.2

        Regime adaptation (heuristic):
            - 'volatile': dampen extremes by 15%
            - 'bear': tilt towards conservatism (negative bias +0.05)
            - 'bull': tilt towards optimism (positive bias +0.05)
            - 'range': slight damping (5%)

        Returns:
            {
              "score": float in [0,1],
              "components": {
                 "headlines": {"score":..., "weight":..., "count":...},
                 "social":    {"score":..., "weight":..., "count":...},
                 "fear_greed":{"score":..., "weight":...}
              },
              "diagnostics": {
                 "regime_hint": "...",
                 "damping": ...,
                 "bias": ...,
                 "notes": [...]
              }
            }
        """
        w = {"headlines": 0.5, "social": 0.3, "fear_greed": 0.2}
        if weights:
            w.update({k: float(v) for k, v in weights.items() if k in w})

        # Normalize weights to sum 1 (robust to zeros/missing)
        s = sum(w.values())
        if s <= 0:
            w = {k: (1.0/3.0) for k in w}  # uniform
        else:
            w = {k: v / s for k, v in w.items()}

        h_score = float(headlines.get("score", 0.5)) if headlines else 0.5
        h_count = int(headlines.get("count", 0)) if headlines else 0

        s_score = float(social.get("score", 0.5)) if social else 0.5
        s_count = int(social.get("count", 0)) if social else 0

        fgi_score = 0.5 if fear_greed is None else max(0.0, min(1.0, float(fear_greed)))

        # Weighted sum (pre-regime)
        base = (w["headlines"] * h_score) + (w["social"] * s_score) + (w["fear_greed"] * fgi_score)

        # Regime adaptation
        damping = 1.0
        bias = 0.0
        notes: List[str] = []

        if regime_hint:
            rh = regime_hint.lower()
            if rh == "volatile":
                damping = 0.85
                notes.append("Volatile regime: damping extremes by 15%")
            elif rh == "range":
                damping = 0.95
                notes.append("Range-bound regime: mild damping")
            elif rh == "bear":
                bias = -0.05
                notes.append("Bearish regime: conservative tilt (-5%)")
            elif rh == "bull":
                bias = +0.05
                notes.append("Bullish regime: optimistic tilt (+5%)")

        adjusted = (base - 0.5) * damping + 0.5 + bias
        final = max(0.0, min(1.0, adjusted))

        return {
            "score": round(final, 4),
            "components": {
                "headlines": {"score": round(h_score, 4), "weight": round(w["headlines"], 3), "count": h_count},
                "social":    {"score": round(s_score, 4), "weight": round(w["social"], 3), "count": s_count},
                "fear_greed":{"score": round(fgi_score, 4), "weight": round(w["fear_greed"], 3)},
            },
            "diagnostics": {
                "regime_hint": regime_hint,
                "damping": damping,
                "bias": bias,
                "notes": notes,
            }
        }


# ────────────────────────────────────────────────────────────
#  Convenience free functions (optional)
# ────────────────────────────────────────────────────────────

_DEFAULT_SCORER: Optional[SentimentScorer] = None

def _scorer() -> SentimentScorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = SentimentScorer()
    return _DEFAULT_SCORER

def score_headlines(headlines: Optional[List[str]]) -> Dict[str, Any]:
    """Module-level convenience wrapper."""
    return _scorer().score_headlines(headlines)

def score_social(posts: Optional[List[str]]) -> Dict[str, Any]:
    """Module-level convenience wrapper."""
    return _scorer().score_social(posts)

def normalize_fear_greed(raw: Union[None, int, float, str, Dict[str, Any]]) -> float:
    """Module-level convenience wrapper."""
    return _scorer().normalize_fear_greed(raw)

def aggregate_sentiment(
    *,
    headlines: Optional[Dict[str, Any]] = None,
    social: Optional[Dict[str, Any]] = None,
    fear_greed: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    regime_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Module-level convenience wrapper."""
    return _scorer().aggregate_sentiment(
        headlines=headlines,
        social=social,
        fear_greed=fear_greed,
        weights=weights,
        regime_hint=regime_hint,
    )
