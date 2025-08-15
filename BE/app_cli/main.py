# BE/app_cli/terminal_ui.py
"""
Terminal UI helpers for the Trading Assistant CLI.

This module keeps ALL user I/O in one place so the rest of the app stays
headless and testable.

Public API (used by app_cli/main.py):
  • prompt_main_mode() -> "category" | "single_asset"
  • ask_use_all_features() -> bool
  • ask_use_rsi() -> bool
  • ask_use_sma() -> bool
  • ask_use_sentiment() -> bool
  • get_user_choice() -> one of:
        'crypto','forex','equities','commodities','futures','warrants','funds'
  • get_user_budget() -> float
  • get_market_selection_details() -> dict | {}   (e.g., {"market":"LSE"})
  • prompt_single_asset_input() -> dict           (symbol, asset_class, indicators, etc.)
  • print_header(), print_table(), print_kv(), print_line()  (pretty output)

Region/market menus:
  - Show colored dots for status: 🟢 open, 🟠 some open, 🔴 closed
  - Show each market’s session hours converted to the USER’s local timezone

This file *reads* market metadata via trading_core.config. If that fails for
any reason, the UI degrades gracefully and uses neutral/defaults.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Iterable
from datetime import datetime
from zoneinfo import ZoneInfo
import tzlocal

# ────────────────────────────────────────────────────────────
# Try to import config helpers, but stay robust if anything fails
# ────────────────────────────────────────────────────────────
try:
    from trading_core.config import (
        load_markets_config,     # -> Dict[str, Dict]
        get_market_info,         # (market_key) -> Dict
        sessions_today,          # (market_key) -> List[Tuple[dt_start, dt_end]]
        is_market_open,          # (market_key) -> bool
    )
except Exception:  # pragma: no cover
    load_markets_config = None
    get_market_info = None
    sessions_today = None
    is_market_open = None

# ────────────────────────────────────────────────────────────
# Local timezone detection
# ────────────────────────────────────────────────────────────
try:
    LOCAL_TZ = ZoneInfo(tzlocal.get_localzone_name())
except Exception:  # pragma: no cover
    LOCAL_TZ = ZoneInfo("Europe/Paris")

# ────────────────────────────────────────────────────────────
# Categories (canonical keys + labels)
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

# Fallback region order if config grouping not available
_REGION_ORDER_DEFAULT = [
    "Americas",
    "Europe",
    "Middle East & Africa",
    "Asia-Pacific",
]


# ═══════════════════════════════════════════════════════════
# Printing helpers (shared by the CLI)
# ═══════════════════════════════════════════════════════════
def print_line(char: str = "─", width: int = 70) -> None:
    print(char * width)


def print_header(title: str) -> None:
    print()
    print_line("─")
    print(title)
    print_line("─")


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def _col_widths(rows: Iterable[Iterable[str]]) -> List[int]:
    widths: List[int] = []
    for row in rows:
        for i, cell in enumerate(row):
            w = len(str(cell))
            if i >= len(widths):
                widths.append(w)
            else:
                widths[i] = max(widths[i], w)
    return widths


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    # compute widths with headers + rows
    all_rows = [headers] + rows
    widths = _col_widths(all_rows)
    # header
    hdr = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(hdr)
    print("-" * len(hdr))
    # rows
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


# ═══════════════════════════════════════════════════════════
# Basic yes/no prompts
# ═══════════════════════════════════════════════════════════
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
    return _yn("Enable ALL optional features (RSI, SMA, Sentiment)?", default=True)


def ask_use_rsi() -> bool:
    return _yn("Enable RSI indicator?", default=True)


def ask_use_sma() -> bool:
    return _yn("Enable SMA indicator?", default=True)


def ask_use_sentiment() -> bool:
    return _yn("Enable sentiment analysis?", default=True)


# ═══════════════════════════════════════════════════════════
# Budget
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# Main mode
# ═══════════════════════════════════════════════════════════
def prompt_main_mode() -> str:
    """
    Choose between:
      1) Explore by category (legacy 7‑category flow)
      2) Analyze a specific asset (new flow)
    """
    print("\nChoose a mode:")
    print("  1. 📦 Explore by category (crypto / stocks / …)")
    print("  2. 🎯 Analyze a specific asset (symbol/coin/pair)")
    while True:
        ans = input("Enter number: ").strip()
        if ans == "1":
            return "category"
        if ans == "2":
            return "single_asset"
        print("Please choose 1 or 2.")


# ═══════════════════════════════════════════════════════════
# Category selection
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# Region / Market menus (with colored status + sessions in local time)
# ═══════════════════════════════════════════════════════════
def _load_grouped_markets() -> Tuple[List[str], Dict[str, List[Tuple[str, Dict]]]]:
    """
    Returns:
      (ordered_regions, grouped_markets)
      grouped_markets[region] = [(market_key, market_info_dict), ...]
    """
    try:
        markets = load_markets_config() if load_markets_config else {}
    except Exception:
        markets = {}

    # group by market["region"]
    grouped: Dict[str, List[Tuple[str, Dict]]] = {}
    for mkey, mi in (markets or {}).items():
        region = mi.get("region", "Other")
        grouped.setdefault(region, []).append((mkey, mi))

    # sort each region's markets by label
    for r in grouped:
        grouped[r].sort(key=lambda kv: kv[1].get("label", kv[0]))

    # region order: our canonical sequence, then any leftovers
    regions: List[str] = []
    for r in _REGION_ORDER_DEFAULT:
        if r in grouped:
            regions.append(r)
    for r in grouped.keys():
        if r not in regions:
            regions.append(r)

    return regions, grouped


def _region_status_dot(region: str, grouped: Dict[str, List[Tuple[str, Dict]]]) -> str:
    """
    Region dot:
      🟢 all markets open   🟠 some open   🔴 all closed
    """
    mkts = grouped.get(region, [])
    if not mkts:
        return "⚪"
    if not is_market_open:
        return "🟠"  # unknown — neutral/some open
    total = len(mkts)
    open_cnt = 0
    for mkey, _ in mkts:
        try:
            if is_market_open(mkey):  # type: ignore[misc]
                open_cnt += 1
        except Exception:
            pass
    if open_cnt == 0:
        return "🔴"
    if open_cnt == total:
        return "🟢"
    return "🟠"


def _sessions_user_local_str(market_key: str, mi: Optional[Dict] = None) -> str:
    """
    Convert today's sessions for this market into the USER's local time.
    Fallback to static sessions if sessions_today() is not available.
    """
    spans: List[str] = []
    # live conversion via sessions_today()
    if sessions_today:
        try:
            for start_dt, end_dt in sessions_today(market_key):  # type: ignore[misc]
                spans.append(
                    f"{start_dt.astimezone(LOCAL_TZ).strftime('%H:%M')}-"
                    f"{end_dt.astimezone(LOCAL_TZ).strftime('%H:%M')}"
                )
        except Exception:
            spans = []
    # static fallback (exchange-local times)
    if not spans and mi:
        sess = mi.get("sessions") or []
        if sess:
            spans = [f"{s}-{e}" for s, e in sess]
    return ", ".join(spans) if spans else "n/a"


def _market_status_dot(market_key: str) -> str:
    if not is_market_open:
        return "🟠"
    try:
        return "🟢" if is_market_open(market_key) else "🔴"  # type: ignore[misc]
    except Exception:
        return "⚪"


def _present_region_menu(regions: List[str], grouped: Dict[str, List[Tuple[str, Dict]]]) -> Optional[str]:
    if not regions:
        return None
    print("\nSelect a REGION:")
    for i, r in enumerate(regions, start=1):
        dot = _region_status_dot(r, grouped)
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


def _present_market_menu(region: str, markets: List[Tuple[str, Dict]]) -> Optional[str]:
    if not markets:
        return None

    print(f"\nSelect a MARKET in {region}:")
    for i, (mkey, mi) in enumerate(markets, start=1):
        label = mi.get("label", mkey)
        dot = _market_status_dot(mkey)
        sess_local = _sessions_user_local_str(mkey, mi)
        print(f"  {i}. {dot} {label} [{mkey}]  —  Hours (your time): {sess_local}")

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
    Interactive region -> market selection.
    Returns {'market': MARKET_KEY} or {} if markets config unavailable.
    """
    regions, grouped = _load_grouped_markets()
    if not regions or not grouped:
        print("\n(Region/market list unavailable; proceeding without a specific market.)")
        return {}

    region = _present_region_menu(regions, grouped)
    if not region:
        return {}

    market_key = _present_market_menu(region, grouped.get(region, []))
    if not market_key:
        return {}

    return {"market": market_key}


# ═══════════════════════════════════════════════════════════
# Single-asset prompts (symbol + which indicators to run)
# ═══════════════════════════════════════════════════════════
_INDICATOR_CHOICES = [
    ("SMA",     "Simple Moving Average"),
    ("EMA",     "Exponential Moving Average"),
    ("MACD",    "MACD (12,26,9)"),
    ("ADX",     "Average Directional Index"),
    ("RSI",     "Relative Strength Index (14)"),
    ("STOCH",   "Stochastic Oscillator"),
    ("OBV",     "On-Balance Volume"),
    ("BBANDS",  "Bollinger Bands (20,2)"),
    ("ATR",     "Average True Range"),
]

_ASSET_CLASS_CHOICES = [
    ("equity", "Stock / ETF"),
    ("crypto", "Crypto asset"),
    ("forex",  "FX pair (e.g., EURUSD)"),
    ("fund",   "Fund / ETF"),
    ("warrant","Warrant / Leveraged ETP"),
]


def _ask_multiselect(name: str, choices: List[Tuple[str, str]]) -> List[str]:
    """
    Show a 1..N list and accept comma-separated selections.
    Returns list of selected KEYS.
    """
    print(f"\nSelect {name} (comma-separated, or press Enter to skip):")
    for i, (k, desc) in enumerate(choices, start=1):
        print(f"  {i}. {k:<7} — {desc}")
    raw = input("Your choice(s): ").strip()
    if not raw:
        return []
    picked: List[str] = []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        if p.isdigit():
            idx = int(p)
            if 1 <= idx <= len(choices):
                picked.append(choices[idx - 1][0])
        else:
            # allow typing the key, e.g., "RSI"
            keys = {k.lower(): k for k, _ in choices}
            if p.lower() in keys:
                picked.append(keys[p.lower()])
    # dedup, keep order
    seen = set()
    out: List[str] = []
    for k in picked:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def prompt_single_asset_input() -> Dict[str, object]:
    """
    Collect inputs for the single-asset analysis flow.
    Returns a dict with keys:
      symbol, asset_class, market?, region?, timeframes, indicators, budget,
      use_rsi, use_sma, use_sentiment
    """
    print_header("Single-Asset Analysis")

    # Asset class
    print("Choose asset class:")
    for i, (_, label) in enumerate(_ASSET_CLASS_CHOICES, start=1):
        print(f"  {i}. {label}")
    asset_class = "equity"
    while True:
        raw = input("Enter number [1]: ").strip() or "1"
        try:
            idx = int(raw)
            if 1 <= idx <= len(_ASSET_CLASS_CHOICES):
                asset_class = _ASSET_CLASS_CHOICES[idx - 1][0]
                break
        except Exception:
            pass
        print(f"Pick a number between 1 and {len(_ASSET_CLASS_CHOICES)}.")

    # Symbol
    ex = {"equity": "AAPL", "crypto": "BTC", "forex": "EURUSD", "fund": "SPY", "warrant": "TSLA3L.L"}
    symbol = input(f"Enter symbol (e.g., {ex.get(asset_class)}): ").strip().upper()
    while not symbol:
        symbol = input("Symbol cannot be empty. Enter symbol: ").strip().upper()

    # Market selection (needed mostly for equities/funds/warrants)
    market = None
    region = None
    if asset_class in {"equity", "fund", "warrant"}:
        use_market = _yn("Select a specific market/exchange for this symbol?", default=False)
        if use_market:
            sel = get_market_selection_details()
            market = sel.get("market") if sel else None

    # Timeframes (future MTF usage; keep simple)
    timeframes = ["1d"]
    tf_raw = input("Timeframes (comma-separated; default: 1d): ").strip()
    if tf_raw:
        parts = [p.strip() for p in tf_raw.split(",") if p.strip()]
        timeframes = parts or ["1d"]

    # Indicators
    indicators = _ask_multiselect("indicators to run", _INDICATOR_CHOICES)

    # Feature toggles — infer from indicators but allow overrides
    use_rsi = "RSI" in indicators or ask_use_rsi()
    use_sma = "SMA" in indicators or "EMA" in indicators or ask_use_sma()
    use_sentiment = ask_use_sentiment()

    # Budget (optional for single-asset — still useful for qty sizing)
    budget = get_user_budget()

    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "market": market,
        "region": region,
        "timeframes": timeframes,
        "indicators": indicators,
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "budget": budget,
    }
