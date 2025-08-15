# BE/app_cli/terminal_ui.py
"""
Terminal UI helpers for the Trading Assistant CLI.

This module centralizes ALL console I/O so the rest of the app remains
testable and headless.

Public functions used by app_cli/main.py (backward compatible):
  • prompt_main_mode() -> "category" | "single_asset"
  • ask_use_all_features() -> bool
  • ask_use_rsi() -> bool
  • ask_use_sma() -> bool
  • ask_use_sentiment() -> bool
  • get_user_budget() -> float
  • get_user_choice() -> Literal[
        'crypto','forex','equities','commodities','futures','warrants','funds'
    ]
  • get_market_selection_details() -> dict
  • prompt_single_asset_input() -> dict
  • NEW: prompt_category_indicator_selection() -> list[str]  (optional in old flow)

Region & market menus show status dots:
    🟢 all open   🟠 some open   🔴 all closed   ⚪ unknown
Market sessions are displayed **in the user's local timezone**.

If market metadata can't be loaded, we gracefully degrade and return an
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
# Pretty output helpers (used by main.py)
# ────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + title)
    print("─" * max(12, len(title)))


def print_line() -> None:
    print("—" * 36)


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def print_table(headers: List[str], rows: List[List[object]]) -> None:
    # very small, dependency‑free table
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(r: List[object]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))

    print(fmt_row(headers))
    print(fmt_row(["-" * w for w in widths]))
    for r in rows:
        print(fmt_row(r))


# ────────────────────────────────────────────────────────────
# Mode selection (classic 7‑category flow vs single‑asset)
# ────────────────────────────────────────────────────────────

def prompt_main_mode() -> str:
    print("\nChoose mode:")
    print("  1) Analyze a CATEGORY (classic workflow)")
    print("  2) Analyze a SINGLE ASSET (symbol/coin/pair)")
    while True:
        ans = input("Enter number [1/2]: ").strip()
        if ans in {"1", ""}:
            return "category"
        if ans == "2":
            return "single_asset"
        print("Please enter 1 or 2.")


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
# Category selection (classic flow)
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

        # exact label match
        for key, label in _CATEGORIES:
            if ans == label.lower():
                return key

        print("Invalid choice. Try again (e.g., '1', 'stocks', 'fx', or 'crypto').")


# ────────────────────────────────────────────────────────────
# Indicator selection (classic flow – optional)
# ────────────────────────────────────────────────────────────

# Canonical indicator names used by the rules engine
_ALL_INDICATORS = [
    "SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR",
]

def prompt_category_indicator_selection() -> List[str]:
    """
    Optional menu to choose which indicators the classic flow should run.
    If the user chooses "All core indicators", returns the full list.
    If they choose "Custom selection", returns only the checked ones (can be empty).

    NOTE: If your current main.py doesn't consume this yet, you can ignore
    the result for now. It’s non‑breaking.
    """
    print("\nIndicator set for CATEGORY analysis:")
    print("  1) All core indicators")
    print("  2) Custom selection")
    while True:
        ans = input("Enter number [1/2]: ").strip()
        if ans in {"1", ""}:
            return list(_ALL_INDICATORS)
        if ans == "2":
            break
        print("Please enter 1 or 2.")

    chosen: List[str] = []
    print("\nSelect indicators (y/n):")
    for name in _ALL_INDICATORS:
        if _yn(f"  - {name}?", default=True):
            chosen.append(name)
    return chosen


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
# Single‑asset prompts (symbol + indicators + optional context)
# ────────────────────────────────────────────────────────────

_SINGLE_ASSET_CLASSES = [
    ("equity", "Stock / Equity"),
    ("crypto", "Crypto / Coin"),
    ("forex",  "FX Pair"),
]

_TIMEFRAME_CHOICES = [
    "1h", "4h", "1d", "1w"
]

def _prompt_from_list(title: str, options: List[Tuple[str, str]], default_idx: int = 1) -> str:
    print(f"\n{title}")
    for i, (_, label) in enumerate(options, start=1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"Enter number [{default_idx}]: ").strip()
        if raw == "":
            return options[default_idx - 1][0]
        try:
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1][0]
        except Exception:
            pass
        print(f"Choose a number between 1 and {len(options)} (or press Enter for default).")


def prompt_single_asset_input() -> Dict[str, object]:
    """
    Collects single‑asset analysis parameters:
      {
        symbol: str,
        asset_class: 'equity'|'crypto'|'forex',
        market: Optional[str],
        region: Optional[str],
        timeframes: list[str],         # e.g., ["1d","4h"]
        indicators: list[str],         # subset of _ALL_INDICATORS
        budget: float,
        use_rsi: bool,
        use_sma: bool,
        use_sentiment: bool,
      }
    """
    symbol = input("\nEnter symbol / pair / coin (e.g., AAPL, BTC, EURUSD): ").strip().upper()
    while not symbol:
        symbol = input("Symbol cannot be empty. Enter symbol: ").strip().upper()

    asset_class = _prompt_from_list("Choose ASSET CLASS:", _SINGLE_ASSET_CLASSES, default_idx=1)

    # Optional market/region (useful for equities)
    region = None
    market = None
    if asset_class == "equity":
        if _yn("Select a market for this stock?", default=True):
            try:
                region, market = get_market_selection()
            except Exception:
                print("(Could not load markets; continuing without a specific market.)")

    # Timeframes (simple multi‑select)
    print("\nChoose TIMEFRAMES to consider (press Enter to keep default: 1d):")
    chosen_tf = []
    for tf in _TIMEFRAME_CHOICES:
        if _yn(f"  - {tf}?", default=(tf == "1d" or tf == "4h")):
            chosen_tf.append(tf)
    if not chosen_tf:
        chosen_tf = ["1d"]

    # Indicators selection
    print("\nIndicator set for SINGLE ASSET:")
    print("  1) All core indicators")
    print("  2) Custom selection")
    while True:
        ans = input("Enter number [1/2]: ").strip()
        if ans in {"1", ""}:
            indicators = list(_ALL_INDICATORS)
            break
        if ans == "2":
            indicators = []
            print("\nSelect indicators (y/n):")
            for name in _ALL_INDICATORS:
                if _yn(f"  - {name}?", default=True):
                    indicators.append(name)
            break
        print("Please enter 1 or 2.")

    # Budget + feature toggles (compatible with earlier engine flags)
    try:
        budget = float(input("Budget for this trade ($) [1000]: ").strip() or "1000")
    except Exception:
        budget = 1000.0

    use_rsi = "RSI" in indicators
    use_sma = ("SMA" in indicators) or ("EMA" in indicators)
    use_sentiment = _yn("Enable sentiment analysis for this asset?", default=False)

    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "market": market,
        "region": region,
        "timeframes": chosen_tf,
        "indicators": indicators,
        "budget": budget,
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
    }
