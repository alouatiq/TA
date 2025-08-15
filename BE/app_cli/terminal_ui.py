# BE/app_cli/terminal_ui.py
"""
Terminal UI layer for the Trading Assistant CLI.

This module handles all user interaction, input validation, and pretty printing.
It integrates with trading_core.config for market metadata and session detection.

If market metadata can't be loaded, we gracefully degrade and return an
empty dict from get_market_selection_details(), allowing callers to
default to non-market-aware behavior (e.g., UTC / US).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
import tzlocal

# Try to import config helpers. If anything fails, we keep fallbacks.
try:
    from trading_core.config import (
        load_markets_config,         # () -> dict
        group_markets_by_region,     # (markets: dict) -> dict[str, list[tuple[key, info]]]
        get_region_order,            # (grouped: dict) -> list[str]
        get_market_info,             # (market_key: str) -> dict
        sessions_today,              # (market_key: str) -> list[tuple[dt_start, dt_end]]
        is_market_open,              # (market_key: str) -> bool
        load_api_keys,               # () -> dict
    )
except Exception:  # pragma: no cover - defensive fallback
    load_markets_config = None
    group_markets_by_region = None
    get_region_order = None
    get_market_info = None
    sessions_today = None
    is_market_open = None
    load_api_keys = None

# Detect USER's local timezone (fallback to Europe/Paris)
try:
    LOCAL_TZ = ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = ZoneInfo("Europe/Paris")


# ──────────────────────────────────────────────────────────────────────────────
# Pretty output helpers (used by main.py)
# ──────────────────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + title)
    print("─" * max(12, len(title)))


def print_line() -> None:
    print("—" * 36)


def print_kv(key: str, value) -> None:
    print(f"{key}: {value}")


def print_table(headers: List[str], rows: List[List[object]]) -> None:
    # very small, dependency-free table
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


# ──────────────────────────────────────────────────────────────────────────────
# API Key Management
# ──────────────────────────────────────────────────────────────────────────────

def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    if not load_api_keys:
        return {}
    
    keys = load_api_keys()
    return {
        "TwelveData": bool(keys.get("TWELVEDATA_API_KEY")),
        "CryptoCompare": bool(keys.get("CRYPTOCOMPARE_API_KEY")),
        "OpenAI": bool(keys.get("OPENAI_API_KEY")),
        "Anthropic": bool(keys.get("ANTHROPIC_API_KEY")),
        "Alpha Vantage": bool(keys.get("ALPHA_VANTAGE_API_KEY")),
    }


def print_api_status() -> None:
    """Print API key status with recommendations."""
    print_header("API Configuration Status")
    
    key_status = check_api_keys()
    if not key_status:
        print("⚠️  Could not check API keys (config module unavailable)")
        return
    
    print("📋 API Key Status:")
    for service, available in key_status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {service}: {'Configured' if available else 'Not configured'}")
    
    missing_keys = [service for service, available in key_status.items() if not available]
    
    if missing_keys:
        print("\n🔧 To configure missing API keys, run:")
        print("   make setup-api")
        print("\n📌 Note: The app will work with limited functionality without API keys,")
        print("   but you'll get better data coverage with them configured.")
    else:
        print("\n✅ All API keys are configured!")


# ──────────────────────────────────────────────────────────────────────────────
# Mode selection (classic 7-category flow vs single-asset)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Feature selection helpers
# ──────────────────────────────────────────────────────────────────────────────

def ask_use_all_features() -> bool:
    print_header("Feature Configuration")
    print("🚀 By default, ALL features and indicators are enabled for maximum analysis power!")
    print("   This includes: RSI, SMA, Sentiment Analysis, and ALL Technical Indicators")
    print("   Technical Indicators: SMA, EMA, MACD, ADX, RSI, STOCH, OBV, BBANDS, ATR")
    print()
    while True:
        ans = input("✅ Use ALL features and indicators? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_rsi() -> bool:
    while True:
        ans = input("📊 Enable RSI (Relative Strength Index)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sma() -> bool:
    while True:
        ans = input("📈 Enable SMA (Simple Moving Averages)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sentiment() -> bool:
    while True:
        ans = input("💭 Enable Sentiment Analysis? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def prompt_indicator_bundle() -> Dict[str, any]:
    """Legacy function for compatibility. Returns dict with all/selected indicators."""
    return {"all": True, "selected": []}


# ──────────────────────────────────────────────────────────────────────────────
# Market selection with proper status indicators
# ──────────────────────────────────────────────────────────────────────────────

def _get_region_status_icon(region_markets: List[Tuple[str, Dict]]) -> str:
    """Get colored circle based on market status in the region."""
    if not region_markets or not is_market_open:
        return "⚪"  # Gray if we can't determine status
    
    open_count = 0
    total_count = len(region_markets)
    
    for market_key, _ in region_markets:
        if is_market_open(market_key):
            open_count += 1
    
    if open_count == 0:
        return "🔴"  # Red - all markets closed
    elif open_count == total_count:
        return "🟢"  # Green - all markets open
    else:
        return "🟠"  # Orange - some markets open, some closed


def _format_market_time(market_key: str) -> str:
    """Format current time in market's timezone."""
    if not get_market_info:
        return ""
    
    info = get_market_info(market_key)
    tz_name = info.get("timezone", "UTC")
    
    try:
        market_tz = ZoneInfo(tz_name)
        now = datetime.now(market_tz)
        return f"({now.strftime('%H:%M %Z')})"
    except Exception:
        return ""


def get_user_choice() -> str:
    """Get category choice from user."""
    print("\nSelect a TRADE CATEGORY:")
    print("  1. Cryptocurrencies")
    print("  2. Forex (FX)")
    print("  3. Equities / Stocks")
    print("  4. Commodities (Metals)")
    print("  5. Index Futures / Proxies")
    print("  6. Warrants & Leveraged ETPs")
    print("  7. Funds / ETFs")
    
    while True:
        ans = input("Enter number or name: ").strip()
        category_map = {
            "1": "crypto", "crypto": "crypto", "cryptocurrencies": "crypto",
            "2": "forex", "forex": "forex", "fx": "forex",
            "3": "equities", "equities": "equities", "stocks": "equities",
            "4": "commodities", "commodities": "commodities", "metals": "commodities",
            "5": "futures", "futures": "futures", "index": "futures",
            "6": "warrants", "warrants": "warrants", "leveraged": "warrants",
            "7": "funds", "funds": "funds", "etfs": "funds"
        }
        
        normalized = ans.lower()
        if normalized in category_map:
            return category_map[normalized]
        
        print("Please enter a number (1-7) or category name.")


def get_user_budget() -> float:
    """Get budget from user with validation."""
    while True:
        try:
            ans = input("Enter your daily trading budget ($): ").strip()
            budget = float(ans)
            if budget <= 0:
                print("Budget must be positive.")
                continue
            return budget
        except ValueError:
            print("Please enter a valid number.")


def get_market_selection_details() -> Dict[str, any]:
    """
    Enhanced market selection with proper status indicators and market-specific choices.
    """
    try:
        markets_data = load_markets_config() if load_markets_config else {}
        if not markets_data:
            print("\n(Market metadata unavailable; proceeding with default settings.)")
            return {}
        
        grouped = group_markets_by_region(markets_data) if group_markets_by_region else {}
        if not grouped:
            print("\n(Region grouping unavailable; proceeding with default settings.)")
            return {}
        
        region_order = get_region_order(grouped) if get_region_order else list(grouped.keys())
        
        # Step 1: Select Region with status indicators
        print("\nSelect a REGION:")
        for i, region in enumerate(region_order, 1):
            region_markets = grouped[region]
            status_icon = _get_region_status_icon(region_markets)
            print(f"  {i}. {status_icon} {region}")
        
        while True:
            try:
                choice = input("Enter number: ").strip()
                region_idx = int(choice) - 1
                if 0 <= region_idx < len(region_order):
                    selected_region = region_order[region_idx]
                    break
                print(f"Please enter a number between 1 and {len(region_order)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Step 2: Select specific market within the region
        region_markets = grouped[selected_region]
        
        print(f"\nSelect a MARKET in {selected_region}:")
        for i, (market_key, market_info) in enumerate(region_markets, 1):
            status_icon = "🟢" if (is_market_open and is_market_open(market_key)) else "🔴"
            label = market_info.get("label", market_key)
            time_str = _format_market_time(market_key)
            print(f"  {i}. {status_icon} {label} {time_str}")
        
        while True:
            try:
                choice = input("Enter number: ").strip()
                market_idx = int(choice) - 1
                if 0 <= market_idx < len(region_markets):
                    selected_market_key, selected_market_info = region_markets[market_idx]
                    break
                print(f"Please enter a number between 1 and {len(region_markets)}.")
            except ValueError:
                print("Please enter a valid number.")
        
        return {
            "region": selected_region,
            "market": selected_market_key,
            "market_info": selected_market_info,
            "timezone": selected_market_info.get("timezone", "UTC"),
            "sessions": selected_market_info.get("sessions", []),
            "trading_days": selected_market_info.get("trading_days", [0, 1, 2, 3, 4]),
        }
        
    except Exception as e:
        print(f"\n(Market selection error: {e}; proceeding with default settings.)")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Single asset flow
# ──────────────────────────────────────────────────────────────────────────────

def prompt_single_asset_input() -> Dict[str, any]:
    """Collect input for single asset analysis."""
    print_header("Single Asset Analysis")
    
    # Get symbol
    symbol = input("Enter symbol/ticker (e.g., AAPL, BTC-USD, EURUSD): ").strip().upper()
    if not symbol:
        return {"symbol": ""}
    
    # Get asset class
    print("\nSelect asset class:")
    print("  1. Equity/Stock")
    print("  2. Cryptocurrency")
    print("  3. Forex pair")
    print("  4. Commodity")
    print("  5. Future")
    print("  6. Fund/ETF")
    
    asset_class_map = {
        "1": "equity", "2": "crypto", "3": "forex",
        "4": "commodity", "5": "future", "6": "fund"
    }
    
    while True:
        choice = input("Enter number [1-6]: ").strip()
        if choice in asset_class_map:
            asset_class = asset_class_map[choice]
            break
        print("Please enter a number between 1 and 6.")
    
    # Get budget
    budget = get_user_budget()
    
    # For now, use default indicators
    indicators = ["RSI", "SMA", "MACD", "EMA"]
    
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "budget": budget,
        "indicators": indicators,
        "timeframes": ["1d"],  # Default timeframe
    }
