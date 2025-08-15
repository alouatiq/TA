"""
Terminal UI helpers for the Trading Assistant CLI.

This module exposes ONLY simple input/print helpers so the rest of the app
can stay testable and headless. It does NOT fetch data or compute indicators.

Functions expected by app_cli/main.py:
- ask_use_all_features() -> bool
- ask_use_rsi() -> bool
- ask_use_sma() -> bool
- ask_use_sentiment() -> bool
- get_user_budget() -> float
- get_user_choice() -> Literal['crypto','forex','equities','commodities','futures','warrants','funds']
- get_market_selection_details() -> dict | {}  (returns {'market': <MARKET_KEY>} when chosen)

This file *reads* market metadata via trading_core.config to render
region/market menus. If config loading fails, we degrade gracefully and
return {} so the caller can default to non–market-aware behavior.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# We only import lightweight helpers from config; if anything fails,
# the UI will still run with safe defaults.
try:
    from trading_core.config import (
        load_markets_yaml,
        group_markets_by_region,
        get_region_order,
    )
except Exception:  # pragma: no cover - defensive
    load_markets_yaml = None
    group_markets_by_region = None
    get_region_order = None


# ────────────────────────────────────────────────────────────
# Basic yes/no prompts
# ────────────────────────────────────────────────────────────

def _yn(prompt: str, default: bool = True) -> bool:
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
    return _yn("Enable all features (RSI, SMA, Sentiment)?", default=True)


def ask_use_rsi() -> bool:
    return _yn("Enable RSI?", default=True)


def ask_use_sma() -> bool:
    return _yn("Enable SMA?", default=True)


def ask_use_sentiment() -> bool:
    return _yn("Enable Sentiment?", default=True)


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

_CATEGORIES = [
    ("crypto", "Cryptocurrencies"),
    ("forex", "Forex (FX)"),
    ("equities", "Equities / Stocks"),
    ("commodities", "Commodities (Metals)"),
    ("futures", "Index Futures / Proxies"),
    ("warrants", "Warrants & Leveraged ETPs"),
    ("funds", "Funds / ETFs"),
]

def get_user_choice() -> str:
    print("\nSelect a CATEGORY:")
    for i, (_, label) in enumerate(_CATEGORIES, start=1):
        print(f"  {i}. {label}")
    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(_CATEGORIES):
                return _CATEGORIES[idx - 1][0]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(_CATEGORIES)}.")


# ────────────────────────────────────────────────────────────
# Market selection (for market-aware categories)
# ────────────────────────────────────────────────────────────

def _load_grouped_markets() -> Tuple[List[str], Dict[str, List[Tuple[str, Dict]]]]:
    """
    Returns:
      (ordered_regions, grouped_markets)
      grouped_markets[region] = [(market_key, market_info_dict), ...]
    """
    try:
        markets = load_markets_yaml() if load_markets_yaml else {}
        grouped = group_markets_by_region(markets) if group_markets_by_region else {}
        regions = get_region_order(grouped) if get_region_order else list(grouped.keys())
        return regions, grouped
    except Exception:
        return [], {}


def _present_region_menu(regions: List[str]) -> Optional[str]:
    if not regions:
        return None
    print("\nSelect a REGION:")
    badges = {"Americas": "🟢", "Europe": "🟢", "Middle East & Africa": "🟠", "Asia-Pacific": "🔴"}
    for i, r in enumerate(regions, start=1):
        b = badges.get(r, "🟢")
        print(f"  {i}. {b} {r}")
    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(regions):
                return regions[idx - 1]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(regions)}.")


def _present_market_menu(region: str, markets: List[Tuple[str, Dict]]) -> Optional[str]:
    if not markets:
        return None

    print(f"\nSelect a MARKET in {region}:")
    # We don't recompute open/closed state here; just show sessions in the user's local tz
    # (conversion is handled/presented by main() after selection).
    for i, (mkey, mi) in enumerate(markets, start=1):
        label = mi.get("label", mkey)
        sessions = mi.get("sessions", [])
        # render sessions as "HH:MM-HH:MM, HH:MM-HH:MM"
        if sessions:
            parts = [f"{s}-{e}" for s, e in sessions]
            sess_txt = ", ".join(parts)
            print(f"  {i}. {label} [{mkey}]  —  Hours (local exchange time): {sess_txt}")
        else:
            print(f"  {i}. {label} [{mkey}]")

    while True:
        raw = input("Enter number: ").strip()
        try:
            idx = int(raw)
            if 1 <= idx <= len(markets):
                return markets[idx - 1][0]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(markets)}.")


def get_market_selection_details() -> Dict[str, str]:
    """
    Interactive region -> market selection. Returns:
        {'market': MARKET_KEY}
    If markets metadata can't be loaded, returns {} and lets callers default to US/UTC.
    """
    regions, grouped = _load_grouped_markets()
    if not regions or not grouped:
        # Graceful degradation
        print("\n(Region/market list unavailable; proceeding without a specific market.)")
        return {}

    region = _present_region_menu(regions)
    if not region:
        return {}

    market_key = _present_market_menu(region, grouped.get(region, []))
    if not market_key:
        return {}

    return {"market": market_key}
