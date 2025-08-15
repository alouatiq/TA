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


def ask_individual_indicators() -> List[str]:
    """Ask user to select individual technical indicators."""
    print("\n📊 Select Technical Indicators:")
    print("Choose which indicators to enable (you can select multiple):")
    
    indicators = [
        ("SMA", "Simple Moving Average - trend following"),
        ("EMA", "Exponential Moving Average - responsive trend"),
        ("MACD", "Moving Average Convergence Divergence - momentum"),
        ("ADX", "Average Directional Index - trend strength"),
        ("RSI", "Relative Strength Index - momentum oscillator"),
        ("STOCH", "Stochastic Oscillator - momentum"),
        ("OBV", "On-Balance Volume - volume analysis"),
        ("BBANDS", "Bollinger Bands - volatility"),
        ("ATR", "Average True Range - volatility measure"),
    ]
    
    selected = []
    
    for i, (code, description) in enumerate(indicators, 1):
        print(f"  {i}. {code} - {description}")
    
    print("\nSelect indicators:")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter 'all' for all indicators")
    print("  • Enter 'none' for no technical indicators")
    
    while True:
        choice = input("Your selection: ").strip().lower()
        
        if choice == "all":
            selected = [code for code, _ in indicators]
            break
        elif choice == "none":
            selected = []
            break
        elif choice:
            try:
                # Parse comma-separated numbers
                numbers = [int(x.strip()) for x in choice.split(",")]
                selected = []
                for num in numbers:
                    if 1 <= num <= len(indicators):
                        selected.append(indicators[num-1][0])
                    else:
                        raise ValueError(f"Number {num} is out of range")
                break
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid selection: {e}")
                print("Please enter numbers 1-9 separated by commas, 'all', or 'none'")
        else:
            print("Please make a selection.")
    
    if selected:
        print(f"✅ Selected indicators: {', '.join(selected)}")
    else:
        print("ℹ️  No technical indicators selected")
    
    return selected


def ask_sentiment_components() -> List[str]:
    """Ask user to select individual sentiment components."""
    print("\n💭 Select Sentiment Analysis Components:")
    
    components = [
        ("news", "News Headlines Analysis"),
        ("social", "Social Media Sentiment"),
        ("fear_greed", "Fear & Greed Index"),
    ]
    
    selected = []
    
    for i, (code, description) in enumerate(components, 1):
        print(f"  {i}. {description}")
    
    print("\nSelect sentiment components:")
    print("  • Enter numbers separated by commas (e.g., 1,3)")
    print("  • Enter 'all' for all components")
    print("  • Enter 'none' for no sentiment analysis")
    
    while True:
        choice = input("Your selection: ").strip().lower()
        
        if choice == "all":
            selected = [code for code, _ in components]
            break
        elif choice == "none":
            selected = []
            break
        elif choice:
            try:
                numbers = [int(x.strip()) for x in choice.split(",")]
                selected = []
                for num in numbers:
                    if 1 <= num <= len(components):
                        selected.append(components[num-1][0])
                    else:
                        raise ValueError(f"Number {num} is out of range")
                break
            except (ValueError, IndexError) as e:
                print(f"❌ Invalid selection: {e}")
                print("Please enter numbers 1-3 separated by commas, 'all', or 'none'")
        else:
            print("Please make a selection.")
    
    if selected:
        print(f"✅ Selected sentiment components: {', '.join(selected)}")
    else:
        print("ℹ️  No sentiment analysis selected")
    
    return selected


def configure_individual_features() -> Dict[str, any]:
    """Configure features individually with detailed selection."""
    print("🔧 Configuring individual features...")
    
    # Get technical indicators
    selected_indicators = ask_individual_indicators()
    
    # Get sentiment components
    sentiment_components = ask_sentiment_components()
    
    # Legacy compatibility flags
    use_rsi = "RSI" in selected_indicators
    use_sma = "SMA" in selected_indicators or "EMA" in selected_indicators
    use_sentiment = len(sentiment_components) > 0
    
    # Show summary
    print("\n" + "─" * 50)
    print("📋 Configuration Summary:")
    if selected_indicators:
        print(f"   📊 Technical Indicators: {', '.join(selected_indicators)}")
    else:
        print("   📊 Technical Indicators: None")
    
    if sentiment_components:
        print(f"   💭 Sentiment Components: {', '.join(sentiment_components)}")
    else:
        print("   💭 Sentiment Components: None")
    
    print("─" * 50)
    
    return {
        "use_all": False,
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "selected_indicators": selected_indicators,
        "sentiment_components": sentiment_components,
    }


def get_feature_configuration() -> Dict[str, any]:
    """Main function to get feature configuration from user."""
    use_all = ask_use_all_features()
    
    if use_all:
        # All features enabled
        selected_indicators = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
        sentiment_components = ["news", "social", "fear_greed"]
        
        print("🎯 Using ALL features and indicators for comprehensive analysis!")
        
        # Show detailed feature status
        print("─" * 44)
        print("💭 Sentiment Analysis Components:")
        print("   News Headlines: ✅ Enabled")
        print("   Social Media: ✅ Enabled") 
        print("   Fear & Greed Index: ✅ Enabled")
        print("─" * 44)
        print("📊 Technical Indicators Status:")
        for indicator in selected_indicators:
            print(f"   {indicator}: ✅ Enabled")
        print("─" * 44)
        print(f"Total Indicators Active: {len(selected_indicators)} of {len(selected_indicators)} technical")
        print("Sentiment Components Active: 3 of 3 components")
        print("─" * 44)
        
        return {
            "use_all": True,
            "use_rsi": True,
            "use_sma": True,
            "use_sentiment": True,
            "selected_indicators": selected_indicators,
            "sentiment_components": sentiment_components,
        }
    else:
        # Individual configuration
        return configure_individual_features()


# Legacy compatibility functions (kept for backward compatibility)
def ask_use_rsi() -> bool:
    """Legacy function - kept for compatibility."""
    while True:
        ans = input("📊 Enable RSI (Relative Strength Index)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sma() -> bool:
    """Legacy function - kept for compatibility."""
    while True:
        ans = input("📈 Enable SMA (Simple Moving Averages)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sentiment() -> bool:
    """Legacy function - kept for compatibility."""
    while True:
        ans = input("💭 Enable Sentiment Analysis? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


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
