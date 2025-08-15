# BE/app_cli/terminal_ui.py
"""
Terminal UI helpers for the Trading Assistant CLI.

This module centralizes ALL console I/O so the rest of the app remains
testable and headless.

Public functions (used by app_cli/main.py):
  • ask_use_all_features() -> bool
  • ask_use_rsi() -> bool
  • ask_use_sma() -> bool
  • ask_use_sentiment() -> bool
  • get_user_budget() -> float
  • get_user_choice() -> Literal[
        'crypto','forex','equities','commodities','futures','warrants','funds'
    ]
  • ask_region_choice() -> str
  • ask_market_choice(region: str) -> str
  • get_market_selection() -> tuple[str, str]
  • get_market_selection_details() -> dict
  • ask_stock_symbol(candidates: list[str] | None = None) -> str
  • ask_stock_region_market(candidates: list[str] | None = None)
        -> tuple[str, str, str]

Notes
-----
- Region & market menus show status dots:
      🟢 all open   🟠 some open   🔴 all closed
- Market sessions are displayed **in the user's local timezone**.
- If market metadata can't be loaded, we gracefully degrade and return an
  empty dict from get_market_selection_details(), allowing callers to
  default to non–market-aware behavior (e.g., UTC / US).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import tzlocal

# Try to import config helpers. If anything fails, we keep fallbacks.
try:
    from trading_core.config import (
        load_markets_yaml,        # () -> dict
        group_markets_by_region,  # (markets: dict) -> dict[str, list[tuple[key, info]]]
        get_region_order,         # (grouped: dict) -> list[str]
        get_market_info,          # (market_key: str) -> dict
        sessions_today,           # (market_key: str) -> list[tuple[dt_start, dt_end]]
        is_market_open,           # (market_key: str) -> bool
    )
except Exception:  # pragma: no cover - defensive fallback
    load_markets_yaml = None
    group_markets_by_region = None
    get_region_order = None
    get_market_info = None
    sessions_today = None
    is_market_open = None

# Detect USER's local timezone (fallback to Europe/Paris)
try:
    LOCAL_TZ = ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = ZoneInfo("Europe/Paris")


# ────────────────────────────────────────────────────────────
# Categories (canonical keys + labels + aliases)
# ────────────────────────────────────────────────────────────

_CATEGORIES: List[Tuple[str, str]] = [
    ("crypto",      "Cryptocurrencies"),
    ("forex",       "Forex (FX)"),
    ("equities",    "Equities / Stocks"),
    ("commodities", "Commodities (Metals)"),
    ("futures",     "Index Futures / Proxies"),
    ("warrants",    "Warrants & Leveraged ETPs"),
    ("funds",       "Funds / ETFs"),
]

_ALIAS_TO_KEY = {
    # crypto
    "crypto": "crypto", "cryptos": "crypto", "coin": "crypto", "coins": "crypto",
    "cryptocurrency": "crypto", "cryptocurrencies": "crypto",
    # fx
    "fx": "forex", "forex": "forex", "currency": "forex", "currencies": "forex", "fx pairs": "forex",
    # equities
    "equity": "equities", "equities": "equities", "stock": "equities", "stocks": "equities",
    # commodities
    "commodity": "commodities", "commodities": "commodities", "minerals": "commodities", "metals": "commodities",
    # futures
    "future": "futures", "futures": "futures", "index": "futures", "indices": "futures",
    # warrants
    "warrant": "warrants", "warrants": "warrants", "structured": "warrants",
    "structured products": "warrants", "etp": "warrants", "leveraged": "warrants",
    # funds
    "fund": "funds", "funds": "funds", "etf": "funds", "etfs": "funds",
}


# ────────────────────────────────────────────────────────────
# Basic yes/no prompts
# ────────────────────────────────────────────────────────────

def _yn(prompt: str, *, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        ans = input(prompt + suffix).strip().lower()
        if not ans:
            return default
        if ans in {"y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please answer y or n.")


def ask_use_all_features() -> bool:
    return _yn("Enable ALL optional features (RSI, SMA, Sentiment)?", default=True)


def ask_use_rsi() -> bool:
    return _yn("Enable RSI analysis?", default=True)


def ask_use_sma() -> bool:
    return _yn("Enable SMA indicator?", default=True)


def ask_use_sentiment() -> bool:
    return _yn("Enable sentiment analysis?", default=True)


# ────────────────────────────────────────────────────────────
# Budget
# ────────────────────────────────────────────────────────────

def get_user_budget() -> float:
    while True:
        raw = input("Enter your daily trading budget ($): ").strip()
        try:
            val = float(raw)
            if val <= 0:
                raise ValueError
            return val
        except Exception:
            print("Please enter a positive number (e.g., 1000).")


# ────────────────────────────────────────────────────────────
# Category selection
# ────────────────────────────────────────────────────────────

def get_user_choice() -> str:
    """
    Interactive category chooser (supports numbers and free‑text aliases).
    Returns one of:
      'crypto','forex','equities','commodities','futures','warrants','funds'
    """
    print("\nSelect a TRADE CATEGORY:")
    for i, (_, label) in enumerate(_CATEGORIES, start=1):
        print(f"  {i}. {label}")

    while True:
        ans = input("Enter number or name: ").strip().lower()

        # numeric path
        if ans.isdigit():
            idx = int(ans)
            if 1 <= idx <= len(_CATEGORIES):
                return _CATEGORIES[idx - 1][0]

        # alias path
        k = _ALIAS_TO_KEY.get(ans)
        if k:
            return k

        # try matching label exactly
        for key, label in _CATEGORIES:
            if ans == label.lower():
                return key

        print("Invalid choice. Try again (e.g., '1', 'stocks', 'fx', or 'crypto').")


# ────────────────────────────────────────────────────────────
# Region & market helpers (colored dots + local sessions)
# ────────────────────────────────────────────────────────────

_DEFAULT_REGIONS = ["Americas", "Europe", "Middle East & Africa", "Asia-Pacific"]

def _load_grouped_markets() -> Tuple[List[str], Dict[str, List[Tuple[str, Dict]]]]:
    """
    Returns:
      (ordered_regions, grouped_markets)
      grouped_markets[region] = [(market_key, market_info_dict), ...]
    Gracefully degrades on any error.
    """
    try:
        markets = load_markets_yaml() if load_markets_yaml else {}
        grouped = group_markets_by_region(markets) if group_markets_by_region else {}
        regions = get_region_order(grouped) if get_region_order else (
            [r for r in _DEFAULT_REGIONS if r in grouped] or list(grouped.keys())
        )
        return regions, grouped
    except Exception:
        return [], {}


def _market_is_open(market_key: str) -> Optional[bool]:
    try:
        if is_market_open:
            return bool(is_market_open(market_key))
    except Exception:
        return None
    return None


def _region_dot(region: str, grouped: Dict[str, List[Tuple[str, Dict]]]) -> str:
    """
    🟢 all open   🟠 some open   🔴 all closed   ⚪ unknown
    """
    mkts = grouped.get(region, [])
    if not mkts:
        return "⚪"
    open_cnt = 0
    total = 0
    for mkey, _ in mkts:
        st = _market_is_open(mkey)
        if st is None:
            # unknown state – skip from counts
            continue
        total += 1
        if st:
            open_cnt += 1
    if total == 0:
        return "⚪"
    if open_cnt == 0:
        return "🔴"
    if open_cnt == total:
        return "🟢"
    return "🟠"


def _sessions_in_user_tz(market_key: str) -> str:
    """
    Render today's exchange sessions converted to the user's local timezone.
    Example -> "10:00-16:30" or "03:30-06:00, 07:00-10:00"
    """
    try:
        if not sessions_today:
            return "n/a"
        spans = []
        for start_dt, end_dt in sessions_today(market_key):
            spans.append(
                f"{start_dt.astimezone(LOCAL_TZ).strftime('%H:%M')}-"
                f"{end_dt.astimezone(LOCAL_TZ).strftime('%H:%M')}"
            )
        return ", ".join(spans) if spans else "n/a"
    except Exception:
        return "n/a"


def ask_region_choice() -> str:
    """
    Ask the user to choose a broad region first and show a colored status dot:
      🟢 all markets open   🟠 some open   🔴 all closed
    Returns the region string. Falls back to a static list if metadata fails.
    """
    regions, grouped = _load_grouped_markets()
    if not regions:
        # fallback to static order if config not available
        regions = list(_DEFAULT_REGIONS)
        grouped = {r: [] for r in regions}

    print("\nSelect a REGION:")
    for i, r in enumerate(regions, start=1):
        dot = _region_dot(r, grouped)
        print(f"  {i}. {dot} {r}")

    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(regions):
                return regions[idx - 1]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(regions)}.")


def ask_market_choice(region: str) -> str:
    """
    After choosing a region, ask for the specific market in that region.
    For each market, show a green/red dot (open/closed) and the sessions
    converted to the user's local time.
    Returns the MARKET KEY (e.g., 'LSE', 'TADAWUL', 'NYSE').
    """
    regions, grouped = _load_grouped_markets()
    mkts = grouped.get(region, [])

    if not mkts:
        raise ValueError(f"No markets configured for region: {region}")

    # Sort by display label for a neat menu
    mkts_sorted = sorted(mkts, key=lambda kv: kv[1].get("label", kv[0]))

    print(f"\nSelect a MARKET in {region}:")
    for i, (mkey, info) in enumerate(mkts_sorted, start=1):
        label = info.get("label", mkey)
        dot = "🟢" if _market_is_open(mkey) else "🔴"
        sess_txt = _sessions_in_user_tz(mkey)
        print(f"  {i}. {dot} {label} [{mkey}]  —  Hours (your time): {sess_txt}")

    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(mkts_sorted):
                return mkts_sorted[idx - 1][0]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(mkts_sorted)}.")


def get_market_selection() -> Tuple[str, str]:
    """
    Convenience wrapper to get both region and market in one call.
    Returns (region, market_key).
    """
    region = ask_region_choice()
    market = ask_market_choice(region)
    return region, market


def get_market_selection_details() -> Dict[str, str]:
    """
    Returns a dict with normalized market details for main():
      {
        "market": "LSE",
        "market_name": "London Stock Exchange (UK)",
        "market_region": "Europe",
        "market_type": "equity",
        "timezone": "Europe/London",
      }

    If metadata cannot be loaded, prints a note and returns {} so callers
    can proceed without market awareness (default to UTC/US logic).
    """
    try:
        region, market = get_market_selection()
        mi = get_market_info(market) if get_market_info else {}
        return {
            "market": market,
            "market_name": mi.get("label", market),
            "market_region": mi.get("region", region),
            "market_type": "equity",
            "timezone": mi.get("timezone", "UTC"),
        }
    except Exception:
        print("\n(Region/market list unavailable; proceeding without a specific market.)")
        return {}


# ────────────────────────────────────────────────────────────
# Optional symbol prompts (for single‑asset flow)
# ────────────────────────────────────────────────────────────

def ask_stock_symbol(candidates: Optional[List[str]] = None) -> str:
    """
    Ask for a stock/asset symbol. If `candidates` are provided, show a menu
    with a '0 = custom' option.
    """
    if candidates:
        print("\nChoose a symbol (or type a different one):")
        for i, s in enumerate(candidates, start=1):
            print(f"  {i}. {s}")
        print("  0. Type a different symbol")
        while True:
            raw = input("Enter number (or 0): ").strip()
            try:
                idx = int(raw)
                if idx == 0:
                    break
                if 1 <= idx <= len(candidates):
                    return candidates[idx - 1].upper()
            except Exception:
                pass
            print(f"Choose 0 or a number between 1 and {len(candidates)}.")
    # custom entry path
    sym = input("Enter symbol: ").strip().upper()
    while not sym:
        sym = input("Symbol cannot be empty. Enter symbol: ").strip().upper()
    return sym


def ask_stock_region_market(candidates: Optional[List[str]] = None) -> Tuple[str, str, str]:
    """
    Combined convenience flow:
      1) choose a symbol (or enter manually)
      2) choose a region
      3) choose a market in that region
    Returns (symbol, region, market_key).
    """
    symbol = ask_stock_symbol(candidates)
    region, market = get_market_selection()
    return symbol, region, market
