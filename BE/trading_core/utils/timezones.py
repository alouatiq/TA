# BE/trading_core/utils/timezones.py
"""
TZ + session utilities with **no dependency** on the app's config module.
All functions operate on a light `market_info` dict (tz, sessions, trading_days)
so callers can supply data from YAML, DB, or mock fixtures.

market_info schema (minimal):
{
  "timezone": "Europe/Paris",         # IANA tz
  "sessions": [ ["09:00","17:30"], ... ],  # local HH:MM ranges
  "trading_days": [0,1,2,3,4],        # 0=Mon ... 6=Sun
}

Why isolated?
- prevents circular imports (config ↔ utils)
- makes unit testing trivial
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone as dt_timezone
from typing import Iterable, List, Tuple, Dict, Optional

try:
    # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


# ────────────────────────────────────────────────────────────
# Parsing & formatting
# ────────────────────────────────────────────────────────────
def parse_hhmm(hhmm: str) -> time:
    """Parse 'HH:MM' → datetime.time. Raises ValueError on bad input."""
    hhmm = (hhmm or "").strip()
    if len(hhmm) != 5 or hhmm[2] != ":":
        raise ValueError(f"Invalid HH:MM string: {hhmm!r}")
    h = int(hhmm[:2])
    m = int(hhmm[3:])
    if not (0 <= h <= 23 and 0 <= m <= 59):
        raise ValueError(f"Invalid clock values: {hhmm!r}")
    return time(hour=h, minute=m)


def localize_hhmm_in_market_tz(hhmm: str, *, market_tz: str, reference_dt: Optional[datetime] = None) -> datetime:
    """
    Take 'HH:MM' assumed in market_tz and return a timezone-aware datetime on the same date
    as reference_dt (default: now in that tz). Useful for converting session strings.
    """
    tz = ZoneInfo(market_tz)
    now_in_tz = (reference_dt.astimezone(tz) if reference_dt else datetime.now(tz))
    t = parse_hhmm(hhmm)
    return datetime(now_in_tz.year, now_in_tz.month, now_in_tz.day, t.hour, t.minute, tzinfo=tz)


def convert_hhmm_between_tz(hhmm: str, *, from_tz: str, to_tz: str, reference_dt: Optional[datetime] = None) -> str:
    """
    Convert 'HH:MM' from from_tz to to_tz for today's date (in from_tz).
    Returns 'HH:MM TZ' (e.g., '10:30 CET').
    """
    src = localize_hhmm_in_market_tz(hhmm, market_tz=from_tz, reference_dt=reference_dt)
    dst = src.astimezone(ZoneInfo(to_tz))
    abbr = dst.tzname() or to_tz
    return f"{dst.strftime('%H:%M')} {abbr}"


# ────────────────────────────────────────────────────────────
# Sessions & Market clock
# ────────────────────────────────────────────────────────────
def _ensure_schema(mi: Dict[str, object]) -> None:
    if "timezone" not in mi or not mi["timezone"]:
        raise KeyError("market_info missing 'timezone'")
    mi.setdefault("sessions", [])
    mi.setdefault("trading_days", [0, 1, 2, 3, 4])


def sessions_today(market_info: Dict[str, object], *, as_aware_intervals: bool = True) -> List[Tuple[datetime, datetime]]:
    """
    Return today's trading sessions as list of (start_dt, end_dt) in the market's timezone.
    If today is not a trading day, returns [].
    """
    _ensure_schema(market_info)
    tz = ZoneInfo(str(market_info["timezone"]))
    today = datetime.now(tz).date()
    weekday = today.weekday()
    trading_days = list(market_info.get("trading_days", [0, 1, 2, 3, 4]))  # type: ignore[assignment]
    if weekday not in trading_days:
        return []
    out: List[Tuple[datetime, datetime]] = []
    for span in market_info.get("sessions", []):  # type: ignore[assignment]
        try:
            start_hhmm, end_hhmm = span
            s = localize_hhmm_in_market_tz(start_hhmm, market_tz=str(market_info["timezone"]))
            e = localize_hhmm_in_market_tz(end_hhmm, market_tz=str(market_info["timezone"]))
            # Handle overnight (rare for cash equities but safe)
            if e <= s:
                e = e + timedelta(days=1)
            out.append((s, e))
        except Exception:
            continue
    return out


def market_is_open_now(market_info: Dict[str, object]) -> bool:
    """
    Return True if current time in market tz is within any session today.
    """
    intervals = sessions_today(market_info)
    if not intervals:
        return False
    tz = ZoneInfo(str(market_info["timezone"]))
    now = datetime.now(tz)
    return any(s <= now <= e for s, e in intervals)


# Convenience: detect next session boundary (useful for timers/UX)
def next_session_event(market_info: Dict[str, object]) -> Optional[Tuple[str, datetime]]:
    """
    Returns ("open" or "close", datetime) for the next boundary today, else None.
    """
    intervals = sessions_today(market_info)
    if not intervals:
        return None
    tz = ZoneInfo(str(market_info["timezone"]))
    now = datetime.now(tz)

    # If before first open
    first_open = intervals[0][0]
    if now < first_open:
        return ("open", first_open)
    # During any session -> next close
    for s, e in intervals:
        if s <= now <= e:
            return ("close", e)
    # After last close
    return None
