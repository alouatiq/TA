# BE/app_cli/terminal_ui.py
"""
Enhanced Terminal UI layer for the Trading Assistant CLI.

This module handles all user interaction, input validation, and pretty printing.
It includes enhanced AI selection, sentiment configuration, and technical indicator choices.
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
        validate_api_keys,           # () -> dict
    )
except Exception:  # pragma: no cover - defensive fallback
    load_markets_config = None
    group_markets_by_region = None
    get_region_order = None
    get_market_info = None
    sessions_today = None
    is_market_open = None
    load_api_keys = None
    validate_api_keys = None

# Detect USER's local timezone (fallback to Europe/Paris)
try:
    LOCAL_TZ = ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = ZoneInfo("Europe/Paris")


# ────────────────────────────────────────────────────────────────────────────
# Pretty output helpers (used by main.py)
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# API Key Management
# ────────────────────────────────────────────────────────────────────────────

def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    if not validate_api_keys:
        return {}
    
    try:
        return validate_api_keys()
    except Exception:
        return {}


def get_available_ai_engines() -> Dict[str, bool]:
    """Get available AI engines with their status using robust detection."""
    try:
        import os
        
        # Direct environment variable access for reliability
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        print(f"[DEBUG] Terminal UI - Raw key lengths: OpenAI={len(openai_key) if openai_key else 0}, Anthropic={len(anthropic_key) if anthropic_key else 0}")
        
        def is_valid_key(key: str) -> bool:
            if not key or key.strip() == "":
                return False
            
            # Check for common placeholder values
            invalid_values = [
                "your_key_here", "YOUR_API_KEY", "your_openai_key_here", 
                "your_anthropic_key_here", "sk-your_openai_key_here", 
                "sk-ant-your_anthropic_key_here", "openai_key", "anthropic_key"
            ]
            
            if key.strip() in invalid_values:
                return False
            
            # Generic check - has reasonable length and not obviously placeholder
            return len(key.strip()) > 10
        
        openai_available = is_valid_key(openai_key)
        anthropic_available = is_valid_key(anthropic_key)
        
        print(f"[DEBUG] Terminal UI - Key validation: OpenAI={openai_available}, Anthropic={anthropic_available}")
        
        return {
            "OpenAI": openai_available,
            "Anthropic": anthropic_available
        }
        
    except Exception as e:
        print(f"[DEBUG] Terminal UI - get_available_ai_engines error: {e}")
        return {"OpenAI": False, "Anthropic": False}


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


# ────────────────────────────────────────────────────────────────────────────
# Mode selection (classic 7-category flow vs single-asset)
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# Enhanced AI Engine Selection
# ────────────────────────────────────────────────────────────────────────────

def ask_ai_engine_selection() -> List[str]:
    """Ask user which AI engines to use for analysis."""
    available_engines = get_available_ai_engines()
    
    if not any(available_engines.values()):
        print("⚠️  No AI engines available. Please configure API keys first.")
        return []
    
    print("\n🤖 AI ENGINE SELECTION:")
    print("Choose which AI engines to use for analysis:")
    
    available_options = []
    if available_engines["OpenAI"]:
        available_options.append(("OpenAI", "OpenAI GPT-4 - Advanced reasoning and analysis"))
    if available_engines["Anthropic"]:
        available_options.append(("Anthropic", "Anthropic Claude - Nuanced market understanding"))
    
    if len(available_options) == 1:
        engine_name, description = available_options[0]
        print(f"   ✅ {engine_name} (Only available engine)")
        print(f"      {description}")
        return [engine_name]
    
    # Multiple engines available
    print("   1. Use ALL available AI engines (Multi-AI analysis)")
    for i, (engine_name, description) in enumerate(available_options, 2):
        print(f"   {i}. {engine_name} only")
        print(f"      {description}")
    
    while True:
        try:
            choice = input("Enter number: ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                # All engines
                return [opt[0] for opt in available_options]
            elif 2 <= choice_num <= len(available_options) + 1:
                # Single engine
                selected_engine = available_options[choice_num - 2][0]
                return [selected_engine]
            else:
                print(f"Please enter a number between 1 and {len(available_options) + 1}.")
        except ValueError:
            print("Please enter a valid number.")


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Sentiment Analysis Selection
# ────────────────────────────────────────────────────────────────────────────

def ask_sentiment_components() -> List[str]:
    """Ask user which sentiment components to include."""
    print("\n💭 SENTIMENT ANALYSIS SELECTION:")
    print("Choose which sentiment data to include in analysis:")
    
    sentiment_options = [
        ("news", "News Headlines & Market News"),
        ("social", "Social Media Sentiment"),
        ("fear_greed", "Fear & Greed Index"),
        ("institutional", "Institutional Sentiment"),
        ("technical", "Technical Sentiment Indicators"),
    ]
    
    print("   1. Use ALL sentiment components (Comprehensive analysis)")
    for i, (code, description) in enumerate(sentiment_options, 2):
        print(f"   {i}. {description}")
    print(f"   {len(sentiment_options) + 2}. No sentiment analysis")
    
    while True:
        try:
            choice = input("Enter number: ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                # All components
                return [opt[0] for opt in sentiment_options]
            elif 2 <= choice_num <= len(sentiment_options) + 1:
                # Single component
                selected_component = sentiment_options[choice_num - 2][0]
                return [selected_component]
            elif choice_num == len(sentiment_options) + 2:
                # No sentiment
                return []
            else:
                print(f"Please enter a number between 1 and {len(sentiment_options) + 2}.")
        except ValueError:
            print("Please enter a valid number.")


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Technical Indicator Selection
# ────────────────────────────────────────────────────────────────────────────

def ask_individual_indicators() -> List[str]:
    """Ask user to select individual technical indicators."""
    print("\n📊 TECHNICAL INDICATOR SELECTION:")
    print("Choose which technical indicators to enable:")
    
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
    
    print("   1. Use ALL technical indicators (Maximum analysis power)")
    for i, (code, description) in enumerate(indicators, 2):
        print(f"   {i}. {code} - {description}")
    print(f"   {len(indicators) + 2}. Custom selection (pick multiple)")
    
    while True:
        try:
            choice = input("Enter number: ").strip()
            choice_num = int(choice)
            
            if choice_num == 1:
                # All indicators
                return [ind[0] for ind in indicators]
            elif 2 <= choice_num <= len(indicators) + 1:
                # Single indicator
                selected_indicator = indicators[choice_num - 2][0]
                return [selected_indicator]
            elif choice_num == len(indicators) + 2:
                # Custom selection
                return ask_custom_indicator_selection(indicators)
            else:
                print(f"Please enter a number between 1 and {len(indicators) + 2}.")
        except ValueError:
            print("Please enter a valid number.")


def ask_custom_indicator_selection(indicators: List[Tuple[str, str]]) -> List[str]:
    """Allow user to select multiple indicators."""
    print("\n📊 CUSTOM INDICATOR SELECTION:")
    print("Enter the numbers of indicators you want (e.g., 1,3,5 or 1-4):")
    
    for i, (code, description) in enumerate(indicators, 1):
        print(f"   {i}. {code} - {description}")
    
    while True:
        choice = input("Enter selection: ").strip()
        try:
            selected_indices = parse_number_selection(choice, len(indicators))
            if selected_indices:
                selected = [indicators[i-1][0] for i in selected_indices]
                print(f"✅ Selected: {', '.join(selected)}")
                return selected
            else:
                print("No indicators selected. Please try again.")
        except ValueError as e:
            print(f"Invalid selection: {e}")


def parse_number_selection(selection: str, max_num: int) -> List[int]:
    """Parse user selection like '1,3,5' or '1-4' into list of numbers."""
    if not selection:
        return []
    
    indices = []
    parts = selection.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range like "1-4"
            start, end = part.split('-', 1)
            start_num = int(start.strip())
            end_num = int(end.strip())
            if 1 <= start_num <= end_num <= max_num:
                indices.extend(range(start_num, end_num + 1))
            else:
                raise ValueError(f"Range {part} is out of bounds (1-{max_num})")
        else:
            # Single number
            num = int(part)
            if 1 <= num <= max_num:
                indices.append(num)
            else:
                raise ValueError(f"Number {num} is out of bounds (1-{max_num})")
    
    return sorted(list(set(indices)))  # Remove duplicates and sort


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Feature Configuration
# ────────────────────────────────────────────────────────────────────────────

def ask_use_all_features() -> bool:
    """Ask if user wants to use all features with enhanced display."""
    print_header("🚀 ANALYSIS CONFIGURATION")
    print("Configure your trading analysis setup:")
    print()
    
    # Show what "ALL" includes with smart AI detection
    available_ai = get_available_ai_engines()
    ai_engines = [name for name, available in available_ai.items() if available]
    
    print("🎯 MAXIMUM POWER MODE includes:")
    print("   📊 Technical Indicators: All 9 indicators (SMA, EMA, MACD, ADX, RSI, STOCH, OBV, BBANDS, ATR)")
    print("   💭 Sentiment Analysis: All 5 components (News, Social, Fear/Greed, Institutional, Technical)")
    
    # Smart AI status display
    if ai_engines:
        if len(ai_engines) == 1:
            print(f"   🤖 AI Analysis: {ai_engines[0]} (Single-AI)")
        else:
            print(f"   🤖 AI Analysis: {' + '.join(ai_engines)} (Multi-AI)")
    else:
        print("   🤖 AI Analysis: ❌ Not available (configure API keys)")
        print("      💡 Run 'make setup-api' to configure OpenAI or Anthropic")
    
    print()
    
    while True:
        ans = input("✅ Use MAXIMUM POWER mode (all features)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def configure_individual_features() -> Dict[str, any]:
    """Configure features individually with detailed selection."""
    print("🔧 CUSTOM CONFIGURATION MODE")
    print("Configure each component individually:")
    
    # Get AI engines
    selected_ai_engines = ask_ai_engine_selection()
    
    # Get technical indicators
    selected_indicators = ask_individual_indicators()
    
    # Get sentiment components
    sentiment_components = ask_sentiment_components()
    
    # Legacy compatibility flags
    use_rsi = "RSI" in selected_indicators
    use_sma = "SMA" in selected_indicators or "EMA" in selected_indicators
    use_sentiment = len(sentiment_components) > 0
    use_ai = len(selected_ai_engines) > 0
    
    # Show comprehensive summary
    print("\n" + "─" * 60)
    print("📋 CUSTOM CONFIGURATION SUMMARY")
    print("─" * 60)
    
    if selected_ai_engines:
        print(f"🤖 AI Engines: {', '.join(selected_ai_engines)}")
    else:
        print("🤖 AI Engines: None selected")
    
    if selected_indicators:
        print(f"📊 Technical Indicators ({len(selected_indicators)}): {', '.join(selected_indicators)}")
    else:
        print("📊 Technical Indicators: None selected")
    
    if sentiment_components:
        print(f"💭 Sentiment Components ({len(sentiment_components)}): {', '.join(sentiment_components)}")
    else:
        print("💭 Sentiment Components: None selected")
    
    print("─" * 60)
    
    return {
        "use_all": False,
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "use_ai": use_ai,
        "selected_indicators": selected_indicators,
        "sentiment_components": sentiment_components,
        "ai_engines": selected_ai_engines,
    }


def get_feature_configuration() -> Dict[str, any]:
    """Main function to get complete feature configuration from user."""
    use_all = ask_use_all_features()
    
    if use_all:
        # All features enabled
        available_ai = get_available_ai_engines()
        ai_engines = [name for name, available in available_ai.items() if available]
        
        selected_indicators = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
        sentiment_components = ["news", "social", "fear_greed", "institutional", "technical"]
        
        print("🎯 MAXIMUM POWER MODE ACTIVATED!")
        print("─" * 50)
        
        if ai_engines:
            print(f"🤖 AI Analysis: {' + '.join(ai_engines)} ({'Multi-AI' if len(ai_engines) > 1 else 'Single-AI'})")
        else:
            print("🤖 AI Analysis: ❌ Not available")
        
        print(f"📊 Technical Indicators: ✅ All {len(selected_indicators)} enabled")
        for ind in selected_indicators:
            print(f"   • {ind}")
        
        print(f"💭 Sentiment Analysis: ✅ All {len(sentiment_components)} components")
        for comp in sentiment_components:
            print(f"   • {comp.replace('_', ' ').title()}")
        
        print("─" * 50)
        
        return {
            "use_all": True,
            "use_rsi": True,
            "use_sma": True,
            "use_sentiment": True,
            "use_ai": len(ai_engines) > 0,
            "selected_indicators": selected_indicators,
            "sentiment_components": sentiment_components,
            "ai_engines": ai_engines,
        }
    else:
        # Individual configuration
        return configure_individual_features()


# ────────────────────────────────────────────────────────────────────────────
# Legacy compatibility functions
# ────────────────────────────────────────────────────────────────────────────

def ask_use_rsi() -> bool:
    """Legacy function for RSI selection."""
    while True:
        ans = input("📊 Enable RSI (Relative Strength Index)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sma() -> bool:
    """Legacy function for SMA selection."""
    while True:
        ans = input("📈 Enable SMA (Simple Moving Averages)? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def ask_use_sentiment() -> bool:
    """Legacy function for sentiment selection."""
    while True:
        ans = input("💭 Enable Sentiment Analysis? [Y/n]: ").strip().lower()
        if ans in {"", "y", "yes"}:
            return True
        if ans in {"n", "no"}:
            return False
        print("Please enter 'y' for yes or 'n' for no.")


def prompt_indicator_bundle() -> Dict[str, any]:
    """Legacy function for compatibility. Returns dict with all/selected indicators."""
    return {"all": True, "selected": ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]}


# ────────────────────────────────────────────────────────────────────────────
# Category and budget selection
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# Market selection helpers
# ────────────────────────────────────────────────────────────────────────────

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
        return "🟡"  # Orange - some markets open, some closed


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


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Individual Selection Functions
# ────────────────────────────────────────────────────────────────────────────

def ask_individual_selection_by_comma(options: List[Tuple[str, str]], category_name: str) -> List[str]:
    """
    Ask user to select items by entering comma-separated choices.
    
    Args:
        options: List of (code, description) tuples
        category_name: Name of category for display (e.g., "Technical Indicators")
    
    Returns:
        List of selected option codes
    """
    print(f"\n📋 {category_name.upper()} SELECTION:")
    print(f"Select {category_name.lower()} you want to use:")
    print("You can:")
    print("  • Enter 'all' for all options")
    print("  • Enter 'none' for no options")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter ranges with dash (e.g., 1-4)")
    print("  • Combine methods (e.g., 1,3,5-7)")
    print()
    
    for i, (code, description) in enumerate(options, 1):
        print(f"  {i}. {code} - {description}")
    
    while True:
        choice = input(f"\nEnter your selection for {category_name.lower()}: ").strip().lower()
        
        if choice == "all":
            selected = [opt[0] for opt in options]
            print(f"✅ Selected ALL {category_name.lower()}: {', '.join(selected)}")
            return selected
        
        elif choice == "none":
            print(f"❌ No {category_name.lower()} selected")
            return []
        
        else:
            try:
                selected_indices = parse_number_selection(choice, len(options))
                if selected_indices:
                    selected = [options[i-1][0] for i in selected_indices]
                    print(f"✅ Selected {category_name.lower()}: {', '.join(selected)}")
                    return selected
                else:
                    print(f"No {category_name.lower()} selected. Try again or enter 'none'.")
            except ValueError as e:
                print(f"Invalid selection: {e}")
                print("Please try again or enter 'help' for examples.")


def ask_individual_ai_selection() -> List[str]:
    """Ask user to select AI engines individually."""
    available_engines = get_available_ai_engines()
    available_options = []
    
    if available_engines["OpenAI"]:
        available_options.append(("OpenAI", "OpenAI GPT-4 - Advanced reasoning and analysis"))
    if available_engines["Anthropic"]:
        available_options.append(("Anthropic", "Anthropic Claude - Nuanced market understanding"))
    
    if not available_options:
        print("❌ No AI engines available. Please configure API keys first.")
        return []
    
    if len(available_options) == 1:
        engine_name, description = available_options[0]
        print(f"\n🤖 Only {engine_name} is available and will be used automatically.")
        return [engine_name]
    
    return ask_individual_selection_by_comma(available_options, "AI Engines")


def ask_individual_technical_indicators() -> List[str]:
    """Ask user to select technical indicators individually."""
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
    
    return ask_individual_selection_by_comma(indicators, "Technical Indicators")


def ask_individual_sentiment_components() -> List[str]:
    """Ask user to select sentiment components individually."""
    sentiment_options = [
        ("news", "News Headlines & Market News"),
        ("social", "Social Media Sentiment"),
        ("fear_greed", "Fear & Greed Index"),
        ("institutional", "Institutional Sentiment"),
        ("technical_sentiment", "Technical Sentiment Indicators"),
    ]
    
    return ask_individual_selection_by_comma(sentiment_options, "Sentiment Components")


# ────────────────────────────────────────────────────────────────────────────
# Updated Enhanced Feature Configuration
# ────────────────────────────────────────────────────────────────────────────

def configure_individual_features() -> Dict[str, any]:
    """Configure features individually with detailed selection."""
    print("🔧 CUSTOM CONFIGURATION MODE")
    print("Configure each component individually:")
    
    # Get AI engines selection
    selected_ai_engines = ask_individual_ai_selection()
    
    # Get technical indicators selection
    selected_indicators = ask_individual_technical_indicators()
    
    # Get sentiment components selection
    sentiment_components = ask_individual_sentiment_components()
    
    # Legacy compatibility flags
    use_rsi = "RSI" in selected_indicators
    use_sma = "SMA" in selected_indicators or "EMA" in selected_indicators
    use_sentiment = len(sentiment_components) > 0
    use_ai = len(selected_ai_engines) > 0
    
    # Show comprehensive summary
    print("\n" + "─" * 70)
    print("📋 CUSTOM CONFIGURATION SUMMARY")
    print("─" * 70)
    
    # AI Engines Summary
    if selected_ai_engines:
        if len(selected_ai_engines) > 1:
            print(f"🤖 AI Engines ({len(selected_ai_engines)}): {', '.join(selected_ai_engines)} (Multi-AI Analysis)")
        else:
            print(f"🤖 AI Engines: {selected_ai_engines[0]} (Single-AI Analysis)")
    else:
        print("🤖 AI Engines: ❌ None selected (Technical analysis only)")
    
    # Technical Indicators Summary
    if selected_indicators:
        print(f"📊 Technical Indicators ({len(selected_indicators)}/{9}): {', '.join(selected_indicators)}")
    else:
        print("📊 Technical Indicators: ❌ None selected")
    
    # Sentiment Analysis Summary
    if sentiment_components:
        print(f"💭 Sentiment Components ({len(sentiment_components)}/{5}): {', '.join(sentiment_components)}")
    else:
        print("💭 Sentiment Components: ❌ None selected")
    
    # Power Level Assessment
    total_possible = 9 + 5 + len(get_available_ai_engines())  # indicators + sentiment + ai
    total_selected = len(selected_indicators) + len(sentiment_components) + len(selected_ai_engines)
    power_level = (total_selected / total_possible) * 100 if total_possible > 0 else 0
    
    print(f"⚡ Analysis Power Level: {power_level:.0f}% ({total_selected}/{total_possible} components)")
    print("─" * 70)
    
    return {
        "use_all": False,
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "use_ai": use_ai,
        "selected_indicators": selected_indicators,
        "sentiment_components": sentiment_components,
        "ai_engines": selected_ai_engines,
    }


def get_feature_configuration() -> Dict[str, any]:
    """Main function to get complete feature configuration from user."""
    use_all = ask_use_all_features()
    
    if use_all:
        # All features enabled
        available_ai = get_available_ai_engines()
        ai_engines = [name for name, available in available_ai.items() if available]
        
        selected_indicators = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
        sentiment_components = ["news", "social", "fear_greed", "institutional", "technical_sentiment"]
        
        print("🎯 MAXIMUM POWER MODE ACTIVATED!")
        print("─" * 60)
        
        # AI Analysis Status
        if ai_engines:
            if len(ai_engines) > 1:
                print(f"🤖 AI Analysis: {' + '.join(ai_engines)} (Multi-AI Analysis)")
            else:
                print(f"🤖 AI Analysis: {ai_engines[0]} (Single-AI Analysis)")
        else:
            print("🤖 AI Analysis: ❌ Not available (configure API keys)")
        
        # Technical Indicators Status
        print(f"📊 Technical Indicators: ✅ All {len(selected_indicators)}/9 enabled")
        print("   " + ", ".join(selected_indicators))
        
        # Sentiment Analysis Status
        print(f"💭 Sentiment Analysis: ✅ All {len(sentiment_components)}/5 components")
        print("   " + ", ".join([comp.replace('_', ' ').title() for comp in sentiment_components]))
        
        print("⚡ Analysis Power Level: 100% (Maximum)")
        print("─" * 60)
        
        return {
            "use_all": True,
            "use_rsi": True,
            "use_sma": True,
            "use_sentiment": True,
            "use_ai": len(ai_engines) > 0,
            "selected_indicators": selected_indicators,
            "sentiment_components": sentiment_components,
            "ai_engines": ai_engines,
        }
    else:
        # Individual configuration
        return configure_individual_features()


# ────────────────────────────────────────────────────────────────────────────
# Helper function for backward compatibility
# ────────────────────────────────────────────────────────────────────────────

def prompt_single_asset_input() -> Dict[str, any]:
    """Placeholder for single asset input (future implementation)."""
    return {
        "symbol": "BTC-USD",
        "asset_class": "crypto",
        "budget": 1000.0,
        "indicators": ["RSI", "SMA", "MACD"],
    }
