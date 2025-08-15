# BE/trading_core/config.py
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
    # Resolve based on this file's location: TA/BE/trading_core/config.py
    # We want to return the BE directory (parents[1])
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _data_dir() -> Path:
    return _repo_root() / "trading_core" / "data"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file safely, returning empty dict if file doesn't exist or has errors."""
    try:
        if not path.exists():
            print(f"[W] YAML file not found: {path}")
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            print(f"[I] Successfully loaded: {path}")
            return data
    except Exception as e:
        print(f"[E] Error loading YAML file {path}: {e}")
        return {}


@lru_cache(maxsize=1)
def _markets_yaml() -> Dict[str, Any]:
    """Load markets.yml configuration."""
    markets_path = _data_dir() / "markets.yml"
    print(f"[D] Looking for markets.yml at: {markets_path}")
    return _load_yaml(markets_path)


@lru_cache(maxsize=1)
def _seeds_yaml() -> Dict[str, Any]:
    """Load seeds.yml configuration."""
    seeds_path = _data_dir() / "seeds.yml"
    print(f"[D] Looking for seeds.yml at: {seeds_path}")
    return _load_yaml(seeds_path)


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
    info = get_market_info(market_key)
    return info.get("region")


def load_markets_config() -> Dict[str, Any]:
    """
    Public function to load markets configuration.
    Called by main.py during warm-up.
    """
    try:
        markets = _markets_yaml()
        if markets:
            print(f"[I] Loaded {len(markets)} markets from configuration")
        else:
            print("[W] No markets found in configuration")
        return markets
    except Exception as e:
        print(f"[E] Failed to load markets config: {e}")
        return {}


def load_yaml_safe(relative_path: str) -> Dict[str, Any]:
    """
    Load a YAML file relative to the repo root.
    Used by other modules (like data_fetcher) to load configuration files.
    """
    try:
        full_path = _repo_root() / relative_path
        return _load_yaml(full_path)
    except Exception as e:
        print(f"[E] Error loading {relative_path}: {e}")
        return {}


# ────────────────────────────────────────────────────────────
# Market grouping and organization
# ────────────────────────────────────────────────────────────
def group_markets_by_region(markets_data: Optional[Dict] = None) -> Dict[str, List[Tuple[str, Dict]]]:
    """Group markets by region for UI display."""
    if markets_data is None:
        markets_data = _markets_yaml()
    
    grouped: Dict[str, List[Tuple[str, Dict]]] = {}
    
    for market_key, market_info in markets_data.items():
        region = market_info.get("region", "Other")
        if region not in grouped:
            grouped[region] = []
        grouped[region].append((market_key, market_info))
    
    # Sort markets within each region
    for region_markets in grouped.values():
        region_markets.sort(key=lambda x: x[1].get("label", x[0]))
    
    return grouped


def get_region_order(grouped_markets: Dict[str, List]) -> List[str]:
    """Return preferred order of regions for display."""
    preferred_order = ["Americas", "Europe", "Asia", "MEA"]
    regions = list(grouped_markets.keys())
    
    # Put preferred regions first, then others alphabetically
    ordered = []
    for region in preferred_order:
        if region in regions:
            ordered.append(region)
    
    for region in sorted(regions):
        if region not in ordered:
            ordered.append(region)
    
    return ordered


# ────────────────────────────────────────────────────────────
# Time and session helpers
# ────────────────────────────────────────────────────────────
@dataclass
class MarketSession:
    """Represents a market trading session."""
    start_time: time
    end_time: time
    timezone: zoneinfo.ZoneInfo


def _parse_time(time_str: str) -> time:
    """Parse HH:MM format time string."""
    try:
        hour, minute = map(int, time_str.split(":"))
        return time(hour, minute)
    except Exception:
        return time(0, 0)  # Default to midnight


def sessions_today(market_key: str) -> List[Tuple[datetime, datetime]]:
    """
    Return today's trading sessions as (start_dt, end_dt) tuples in UTC.
    Returns empty list if market is closed today or unknown.
    """
    info = get_market_info(market_key)
    today = datetime.now().weekday()  # 0=Monday, 6=Sunday
    
    # Check if market trades today
    trading_days = info.get("trading_days", [])
    if today not in trading_days:
        return []
    
    # Get timezone
    tz_name = info.get("timezone", "UTC")
    try:
        market_tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        market_tz = zoneinfo.ZoneInfo("UTC")
    
    # Parse sessions
    sessions = info.get("sessions", [])
    today_dt = datetime.now(market_tz).date()
    
    session_times = []
    for session in sessions:
        if len(session) >= 2:
            start_time = _parse_time(session[0])
            end_time = _parse_time(session[1])
            
            start_dt = datetime.combine(today_dt, start_time, tzinfo=market_tz)
            end_dt = datetime.combine(today_dt, end_time, tzinfo=market_tz)
            
            # Convert to UTC
            start_utc = start_dt.astimezone(zoneinfo.ZoneInfo("UTC"))
            end_utc = end_dt.astimezone(zoneinfo.ZoneInfo("UTC"))
            
            session_times.append((start_utc, end_utc))
    
    return session_times


def is_market_open(market_key: str) -> bool:
    """Check if the market is currently open."""
    sessions = sessions_today(market_key)
    if not sessions:
        return False
    
    now_utc = datetime.now(zoneinfo.ZoneInfo("UTC"))
    
    for start_dt, end_dt in sessions:
        if start_dt <= now_utc <= end_dt:
            return True
    
    return False


# ────────────────────────────────────────────────────────────
# Environment and API key management
# ────────────────────────────────────────────────────────────
def load_api_keys() -> Dict[str, Optional[str]]:
    """Load API keys from environment variables."""
    return {
        "TWELVEDATA_API_KEY": os.getenv("TWELVEDATA_API_KEY"),
        "CRYPTOCOMPARE_API_KEY": os.getenv("CRYPTOCOMPARE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY"),
    }


def get_api_key(service: str) -> Optional[str]:
    """Get a specific API key."""
    keys = load_api_keys()
    return keys.get(f"{service.upper()}_API_KEY")


# ────────────────────────────────────────────────────────────
# Debug helpers
# ────────────────────────────────────────────────────────────
def debug_paths() -> None:
    """Print path resolution for debugging."""
    print(f"[D] Config file location: {Path(__file__).resolve()}")
    print(f"[D] Repo root: {_repo_root()}")
    print(f"[D] Data directory: {_data_dir()}")
    print(f"[D] Markets.yml path: {_data_dir() / 'markets.yml'}")
    print(f"[D] Markets.yml exists: {(_data_dir() / 'markets.yml').exists()}")


if __name__ == "__main__":
    # Debug mode when run directly
    debug_paths()
    try:
        markets = load_markets_config()
        print(f"[D] Successfully loaded {len(markets)} markets")
        if markets:
            print("[D] Available markets:", list(markets.keys())[:5], "...")
    except Exception as e:
        print(f"[E] Error: {e}")
