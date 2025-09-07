# BE/app_cli/terminal_ui.py
"""
Enhanced Terminal UI layer for the Trading Assistant CLI.

This module handles all user interaction, input validation, and pretty printing.
It includes enhanced AI selection, sentiment configuration, technical indicator choices,
and user-configurable profit targets with confidence level display.
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


def print_api_status(api_status: Optional[Dict[str, bool]] = None) -> None:
    """Print API key status with recommendations."""
    if api_status is None:
        api_status = check_api_keys()
    
    print_header("API Configuration Status")
    
    if not api_status:
        print("⚠️  Could not check API keys (config module unavailable)")
        return
    
    print("📋 API Key Status:")
    for service, available in api_status.items():
        status_icon = "✅" if available else "❌"
        print(f"   {status_icon} {service}: {'Configured' if available else 'Not configured'}")
    
    missing_keys = [service for service, available in api_status.items() if not available]
    
    if missing_keys:
        print("\n🔧 To configure missing API keys, run:")
        print("   make setup-api")
        print("\n📌 Note: The app will work with limited functionality without API keys,")
        print("   but you'll get better data coverage with them configured.")
    else:
        print("\n✅ All API keys are configured!")


def get_available_ai_engines() -> Dict[str, bool]:
    """Get available AI engines with their status using robust detection."""
    try:
        import os
        
        # Direct environment variable access for reliability
        openai_key = os.getenv("OPENAI_API_KEY", "")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        
        print(f"[DEBUG] Terminal UI - Raw key lengths: OpenAI={len(openai_key) if openai_key else 0}, Anthropic={len(anthropic_key) if anthropic_key else 0}")
        
        def is_valid_key(key: str, key_type: str) -> bool:
            if not key or key.strip() == "":
                return False
            
            # Check for common placeholder values
            invalid_values = [
                "your_key_here", "YOUR_API_KEY", "your_openai_key_here", 
                "your_anthropic_key_here", "sk-your_openai_key_here", 
                "sk-ant-your_anthropic_key_here", "openai_key", "anthropic_key",
                "your_api_key", "api_key_here", "insert_key_here"
            ]
            
            if key.strip().lower() in [v.lower() for v in invalid_values]:
                return False
            
            # Specific validation by key type
            if key_type == "openai":
                # OpenAI keys should start with "sk-" and be reasonably long
                return key.startswith("sk-") and len(key) > 40
            elif key_type == "anthropic":
                # Anthropic keys should start with "sk-ant-" and be reasonably long
                return key.startswith("sk-ant-") and len(key) > 50
            
            # Generic check - has reasonable length and not obviously placeholder
            return len(key.strip()) > 20
        
        openai_available = is_valid_key(openai_key, "openai")
        anthropic_available = is_valid_key(anthropic_key, "anthropic")
        
        print(f"[DEBUG] Terminal UI - Key validation: OpenAI={openai_available}, Anthropic={anthropic_available}")
        
        return {
            "OpenAI": openai_available,
            "Anthropic": anthropic_available
        }
        
    except Exception as e:
        print(f"[DEBUG] Terminal UI - get_available_ai_engines error: {e}")
        return {"OpenAI": False, "Anthropic": False}


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
# Profit Target Selection
# ────────────────────────────────────────────────────────────────────────────

def get_profit_target_selection() -> Tuple[float, str]:
    """
    Ask user to select their profit target with clear options and custom input.
    
    Returns:
        Tuple of (profit_percentage, description)
    """
    print(f"\n🎯 PROFIT TARGET CONFIGURATION")
    print("─" * 40)
    print("Choose your minimum profit target:")
    print("  1) Conservative (1-2% minimum) - Lower risk, higher probability")
    print("  2) Moderate (2-4% minimum) - Balanced risk/reward")
    print("  3) Aggressive (3-5% minimum) - Higher risk, higher reward")
    print("  4) Very Aggressive (5%+ minimum) - High risk, high reward")
    print("  5) Custom target - Set your own percentage")
    
    while True:
        profit_choice = input("Enter choice [1-5]: ").strip()
        
        if profit_choice == "1":
            return 1.0, "Conservative (1-2%)"
        elif profit_choice == "2":
            return 2.0, "Moderate (2-4%)"
        elif profit_choice == "3":
            return 3.0, "Aggressive (3-5%)"
        elif profit_choice == "4":
            return 5.0, "Very Aggressive (5%+)"
        elif profit_choice == "5":
            # Custom target selection
            return get_custom_profit_target()
        else:
            print("Please enter a number between 1 and 5.")


def get_custom_profit_target() -> Tuple[float, str]:
    """
    Get custom profit target from user with validation.
    
    Returns:
        Tuple of (profit_percentage, description)
    """
    print("\n🎯 CUSTOM PROFIT TARGET")
    print("Enter your desired minimum profit percentage:")
    print("  • Recommended range: 0.5% to 15%")
    print("  • Conservative: 0.5% - 2%")
    print("  • Moderate: 2% - 5%")
    print("  • Aggressive: 5% - 10%")
    print("  • Very Aggressive: 10%+")
    
    while True:
        try:
            target_input = input("Enter profit percentage (e.g., 2.5): ").strip()
            custom_target = float(target_input)
            
            # Validate range
            if custom_target <= 0:
                print("❌ Profit target must be greater than 0%")
                continue
            elif custom_target > 50:
                print("⚠️  Very high target (>50%)! This may be unrealistic for intraday trading.")
                confirm = input("Are you sure? [y/N]: ").strip().lower()
                if confirm not in {"y", "yes"}:
                    continue
            elif custom_target > 20:
                print("⚠️  High target (>20%)! This requires very volatile markets.")
                confirm = input("Continue with high target? [y/N]: ").strip().lower()
                if confirm not in {"y", "yes"}:
                    continue
            
            # Clamp to reasonable bounds and create description
            final_target = max(0.1, min(50.0, custom_target))
            
            # Generate risk assessment description
            if final_target <= 1:
                risk_level = "Very Conservative"
            elif final_target <= 2:
                risk_level = "Conservative"
            elif final_target <= 5:
                risk_level = "Moderate"
            elif final_target <= 10:
                risk_level = "Aggressive"
            else:
                risk_level = "Very Aggressive"
            
            description = f"Custom {risk_level} ({final_target}%+)"
            
            print(f"✅ Custom profit target set: {final_target}% ({risk_level})")
            return final_target, description
            
        except ValueError:
            print("❌ Please enter a valid number (e.g., 2.5)")


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
        ("technical_sentiment", "Technical Sentiment Indicators"),
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
        sentiment_components = ["news", "social", "fear_greed", "institutional", "technical_sentiment"]
        
        print("🎯 MAXIMUM POWER MODE ACTIVATED!")
        print("─" * 60)
        
        if ai_engines:
            if len(ai_engines) > 1:
                print(f"🤖 AI Analysis: {' + '.join(ai_engines)} (Multi-AI Analysis)")
            else:
                print(f"🤖 AI Analysis: {ai_engines[0]} (Single-AI Analysis)")
        else:
            print("🤖 AI Analysis: ❌ Not available")
        
        print(f"📊 Technical Indicators: ✅ All {len(selected_indicators)}/9 enabled")
        print("   " + ", ".join(selected_indicators))
        
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
# Updated Enhanced Feature Configuration with AI Status Integration
# ────────────────────────────────────────────────────────────────────────────

def ask_ai_engines_with_status(ai_options: List[Tuple[str, str]], available_status: Dict[str, bool]) -> List[str]:
    """
    Ask user to select AI engines with clear status indicators.
    
    Args:
        ai_options: List of (engine_name, description) tuples
        available_status: Dict mapping engine names to availability
    
    Returns:
        List of selected engine names that are actually available
    """
    print("\n📋 AI ENGINES SELECTION:")
    print("Select ai engines you want to use:")
    print("You can:")
    print("  • Enter 'all' for all available engines")
    print("  • Enter 'none' for no AI analysis")
    print("  • Enter numbers separated by commas (e.g., 1,2)")
    print("  • Enter ranges with dash (e.g., 1-2)")
    print("  • Combine methods (e.g., 1,2)")
    print()
    
    available_engines = []
    for i, (engine_name, description) in enumerate(ai_options, 1):
        is_available = available_status.get(engine_name, False)
        if is_available:
            print(f"  {i}. ✅ {engine_name} - {description}")
            available_engines.append((engine_name, description))
        else:
            print(f"  {i}. ❌ {engine_name} - {description} (API key not configured)")
    
    if not available_engines:
        print("\n❌ No AI engines available. Please configure API keys first.")
        print("💡 Run 'make setup-api' to configure your API keys.")
        return []
    
    while True:
        choice = input(f"\nEnter your selection for ai engines: ").strip().lower()
        
        if choice == "all":
            selected = [engine[0] for engine in available_engines]
            if len(selected) > 1:
                print(f"✅ Selected ALL available ai engines: {', '.join(selected)}")
            else:
                print(f"✅ Selected ai engines: {selected[0]}")
            return selected
        
        elif choice == "none":
            print("❌ No ai engines selected")
            return []
        
        else:
            try:
                selected_indices = parse_number_selection(choice, len(ai_options))
                selected = []
                
                for idx in selected_indices:
                    if 1 <= idx <= len(ai_options):
                        engine_name = ai_options[idx-1][0]
                        if available_status.get(engine_name, False):
                            if engine_name not in selected:
                                selected.append(engine_name)
                        else:
                            print(f"⚠️  {engine_name} is not available (API key not configured)")
                    else:
                        print(f"⚠️  Invalid number: {idx}")
                
                if selected:
                    if len(selected) > 1:
                        print(f"✅ Selected ai engines: {', '.join(selected)}")
                    else:
                        print(f"✅ Selected ai engines: {selected[0]}")
                    return selected
                else:
                    print("❌ No valid ai engines selected. Try again or enter 'none'.")
                    
            except ValueError as e:
                print(f"Invalid selection: {e}")
                print("Please try again or enter 'help' for examples.")


# ────────────────────────────────────────────────────────────────────────────
# Single Asset Analysis Input
# ────────────────────────────────────────────────────────────────────────────

def prompt_single_asset_input() -> Dict[str, any]:
    """
    Get user input for single asset analysis with enhanced validation.
    
    Returns:
        Dictionary with asset analysis parameters
    """
    print_header("🔍 Single Asset Analysis Configuration")
    
    # Step 1: Get symbol
    while True:
        symbol = input("Enter symbol/ticker (e.g., AAPL, BTC, EURUSD): ").strip().upper()
        if symbol:
            break
        print("Please enter a valid symbol.")
    
    # Step 2: Get asset class
    print("\nSelect ASSET CLASS:")
    print("  1. Cryptocurrencies")
    print("  2. Forex (FX)")
    print("  3. Equities / Stocks")
    print("  4. Commodities")
    print("  5. Futures")
    print("  6. Warrants")
    print("  7. Funds / ETFs")
    
    asset_classes = {
        "1": "crypto", "2": "forex", "3": "equities", 
        "4": "commodities", "5": "futures", "6": "warrants", "7": "funds"
    }
    
    while True:
        choice = input("Enter number [1-7]: ").strip()
        if choice in asset_classes:
            asset_class = asset_classes[choice]
            break
        print("Please enter a number between 1 and 7.")
    
    # Step 3: Get market/region if needed (for regional assets)
    market = None
    region = None
    
    if asset_class in ["equities", "commodities", "futures", "warrants", "funds"]:
        print(f"\n{asset_class.title()} require market selection:")
        try:
            market_details = get_market_selection_details()
            market = market_details.get("market")
            region = market_details.get("region")
        except Exception:
            print("Using default market settings.")
    
    # Step 4: Get technical indicators
    print(f"\nSelect TECHNICAL INDICATORS for {symbol}:")
    indicators = ask_individual_technical_indicators()
    
    # Step 5: Get budget
    budget = get_user_budget()
    
    # Step 6: Get profit target
    profit_target, target_desc = get_profit_target_selection()
    
    # Summary
    print(f"\n✅ SINGLE ASSET ANALYSIS SUMMARY:")
    print(f"   🎯 Symbol: {symbol}")
    print(f"   📊 Asset Class: {asset_class.title()}")
    if market:
        print(f"   🏛 Market: {market}")
    if region:
        print(f"   🌍 Region: {region}")
    print(f"   📈 Indicators ({len(indicators)}): {', '.join(indicators) if indicators else 'None'}")
    print(f"   💰 Budget: ${budget:,.2f}")
    print(f"   🎯 Profit Target: {target_desc}")
    
    return {
        "symbol": symbol,
        "asset_class": asset_class,
        "market": market,
        "region": region,
        "indicators": indicators,
        "budget": budget,
        "profit_target": profit_target,
        "target_desc": target_desc,
    }


# ────────────────────────────────────────────────────────────────────────────
# Confidence Level Display Helpers
# ────────────────────────────────────────────────────────────────────────────

def format_confidence_display(confidence: float) -> Tuple[str, str, str]:
    """
    Format confidence level with appropriate emoji and description.
    
    Args:
        confidence: Confidence percentage (0-100)
        
    Returns:
        Tuple of (emoji, color_description, text_description)
    """
    if confidence >= 85:
        return "🟢", "Green", "Very High - Strong recommendation"
    elif confidence >= 70:
        return "🟡", "Yellow", "High - Good opportunity"
    elif confidence >= 50:
        return "🟠", "Orange", "Medium - Consider carefully"
    else:
        return "🔴", "Red", "Low - High risk"


def print_confidence_summary(recommendations: List[Dict[str, any]]) -> None:
    """
    Print a summary of confidence levels across all recommendations.
    
    Args:
        recommendations: List of recommendation dictionaries
    """
    if not recommendations:
        return
    
    confidence_levels = {
        "very_high": 0,  # 85%+
        "high": 0,       # 70-84%
        "medium": 0,     # 50-69%
        "low": 0         # <50%
    }
    
    for rec in recommendations:
        confidence = rec.get("confidence", 0)
        if confidence >= 85:
            confidence_levels["very_high"] += 1
        elif confidence >= 70:
            confidence_levels["high"] += 1
        elif confidence >= 50:
            confidence_levels["medium"] += 1
        else:
            confidence_levels["low"] += 1
    
    print(f"\n📊 CONFIDENCE LEVEL BREAKDOWN:")
    if confidence_levels["very_high"]:
        print(f"   🟢 Very High (85%+): {confidence_levels['very_high']} opportunities")
    if confidence_levels["high"]:
        print(f"   🟡 High (70-84%): {confidence_levels['high']} opportunities")
    if confidence_levels["medium"]:
        print(f"   🟠 Medium (50-69%): {confidence_levels['medium']} opportunities")
    if confidence_levels["low"]:
        print(f"   🔴 Low (<50%): {confidence_levels['low']} opportunities")
