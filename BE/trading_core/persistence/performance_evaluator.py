# BE/trading_core/persistence/performance_evaluator.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

from .history_tracker import last_session


def _extract_day_high(rec: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort: if the recommendation carried a short 'price_history',
    use the max of the most recent window (~1 trading day).
    If not present, return None (we avoid hitting network here).
    """
    ph = rec.get("price_history") or rec.get("history")
    if isinstance(ph, list) and ph:
        # Use last 2–3 points as a proxy for “next day’s high” is inaccurate.
        # Instead, we just report the max of the available tail.
        tail = ph[-5:] if len(ph) >= 5 else ph
        try:
            return float(max(tail))
        except Exception:
            return None
    return None


def evaluate_previous_session(days_back: int = 1) -> List[Dict[str, Any]]:
    """
    Produce a simple score-card for the most recent session (default “yesterday”).
    We DO NOT fetch live data here to keep it fast/offline-friendly.

    Returns a list of rows like:
        {
          "asset": "AAPL",
          "target": 190.0,
          "day_high": 191.2 or None,
          "hit": "HIT"/"MISS"/"?"
        }
    """
    sess = last_session()
    if not sess:
        return []

    # sanity: only consider sessions that are at least `days_back` days old
    try:
        ts = sess.get("ts_utc")
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))  # type: ignore[union-attr]
    except Exception:
        dt = datetime.now(timezone.utc)
    if (datetime.now(timezone.utc) - dt) < timedelta(days=days_back):
        # too fresh; skip evaluation to avoid confusion
        return []

    out: List[Dict[str, Any]] = []
    for r in sess.get("recommendations", []):
        asset = str(r.get("asset", r.get("symbol", "?")))
        target = r.get("sell_target") or r.get("target") or None
        try:
            target_f = float(target) if target is not None else None
        except Exception:
            target_f = None

        day_high = _extract_day_high(r)
        # classify
        if target_f is None or day_high is None:
            hit = "?"
        else:
            hit = "HIT" if day_high >= target_f else "MISS"

        out.append({
            "asset": asset,
            "target": float(target_f) if target_f is not None else None,
            "day_high": float(day_high) if day_high is not None else None,
            "hit": hit,
        })

    return out
