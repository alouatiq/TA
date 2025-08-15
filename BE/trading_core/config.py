"""
config.py
─────────
Central configuration layer:
• Loads static market metadata (`data/markets.yml`) and seeds (`data/seeds.yml`)
• Provides market-time helpers (sessions_today, is_market_open)
• Normalizes region labels and market keys
• Centralizes environment/API key access (TwelveData, CryptoCompare, OpenAI, etc.)

All session times in markets.yml are LOCAL to each exchange.
`trading_days` use Python weekday integers: 0=Mon … 6=Sun.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import os
import sys
import yaml
import zoneinfo


# ────────────────────────────────────────────────────────────
# Paths & YAML loading
# ────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _repo_root() -> Path:
    # Resolve based on this file’s location: <repo>/BE/trading_core/config.py
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _data_dir() -> Path:
    return _repo_root() / "trading_core" / "data"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def _markets_yaml() -> Dict[str, Any]:
    return _load_yaml(_data_dir() / "markets.yml")


@lru_cache(maxsize=1)
def _seeds_yaml() -> Dict[str, Any]:
    return _load_yaml(_data_dir() / "seeds.yml")


# ────────────────────────────────────────────────────────────
# Public: market metadata access
# ────────────────────────────────────────────────────────────
def list_markets() -> List[str]:
    """Return all market keys from markets.yml."""
    return list(_markets_yaml().keys())


def get_market_info(market_key: str) -> Dict[str, Any]:
    """
    Return a dict with fields: label, region, timezone, trading_days, sessions.
    If missing/unknown, returns a safe default with UTC tz and empty sessions.
    """
    mk = (market_key or "").strip()
    data = _markets_yaml().get(mk, {})
    if not data:
        return {
            "label": mk or "Unknown",
            "region": None,
            "timezone": "UTC",
            "trading_days": [0, 1, 2, 3, 4],
            "sessions": [],
        }
    return data


def get_region_for_market(market_key: str) -> Optional[str]:
    """Return the region string for a market key, or None."""
    return get_market_info(market_key).get("region")


# ────────────────────────────────────────────────────────────
# Time helpers
# ────────────────────────────────────────────────────────────
def _parse_hhmm(s: str) -> time:
    h, m = (s or "00:00").split(":")
    return time(int(h), int(m))


def _today_in_tz(tz: zoneinfo.ZoneInfo) -> datetime:
    now = datetime.now(tz)
    return datetime(now.year, now.month, now.day, tzinfo=tz)


def sessions_today(market_key: str) -> List[Tuple[datetime, datetime]]:
    """
    Convert sessions from markets.yml into concrete (start_dt, end_dt) *today*
    in the market's local timezone. Handles split sessions (e.g., HKEX, TSE).
    """
    mi = get_market_info(market_key)
    tzname = mi.get("timezone", "UTC")
    try:
        mtz = zoneinfo.ZoneInfo(tzname)
    except Exception:
        mtz = zoneinfo.ZoneInfo("UTC")

    base = _today_in_tz(mtz)
    out: List[Tuple[datetime, datetime]] = []

    for sess in mi.get("sessions", []):
        try:
            start_s, end_s = sess
            start_dt = base.replace(hour=_parse_hhmm(start_s).hour, minute=_parse_hhmm(start_s).minute)
            end_dt = base.replace(hour=_parse_hhmm(end_s).hour, minute=_parse_hhmm(end_s).minute)
            # If end < start (rare), assume next-day rollover
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            out.append((start_dt, end_dt))
        except Exception:
            continue

    return out


def is_market_open(market_key: str, *, at: Optional[datetime] = None) -> bool:
    """
    Return True if the market is currently open (within any session window today).
    Respects market-local timezone and trading_days.
    """
    mi = get_market_info(market_key)
    tzname = mi.get("timezone", "UTC")
    try:
        mtz = zoneinfo.ZoneInfo(tzname)
    except Exception:
        mtz = zoneinfo.ZoneInfo("UTC")

    now = at.astimezone(mtz) if at else datetime.now(mtz)
    weekday_ok = (now.weekday() in set(mi.get("trading_days", [0, 1, 2, 3, 4])))

    if not weekday_ok:
        return False

    for start_dt, end_dt in sessions_today(market_key):
        if start_dt <= now <= end_dt:
            return True
    return False


# ────────────────────────────────────────────────────────────
# Seeds access (used by equities/funds discovery)
# ────────────────────────────────────────────────────────────
def seed_tickers_for(market_key: str) -> List[str]:
    """Return a *copy* of the seed tickers list for a given market, or empty list."""
    seeds = _seeds_yaml()
    arr = seeds.get(market_key, []) or []
    return list(arr)


# ────────────────────────────────────────────────────────────
# Environment / API keys
# ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ApiKeys:
    twelve_data: Optional[str]
    crypto_compare: Optional[str]
    openai: Optional[str]


def load_api_keys() -> ApiKeys:
    """
    Centralized env key discovery. Keeps naming flexible:
      - TWELVEDATA_API_KEY or TWELVE_DATA_API_KEY
      - CRYPTOCOMPARE_API_KEY
      - OPENAI_API_KEY
    """
    td = os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY")
    cc = os.getenv("CRYPTOCOMPARE_API_KEY")
    oa = os.getenv("OPENAI_API_KEY")
    return ApiKeys(twelve_data=td, crypto_compare=cc, openai=oa)
