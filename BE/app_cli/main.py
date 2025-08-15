# BE/app_cli/main.py
"""
CLI runner for the Trading Assistant (Backend).

Features
--------
• Original 7-category workflow (crypto, forex, equities, commodities, futures, warrants, funds)
• New "Analyze a specific asset" workflow with selectable indicators
• Region/market selection with colored open/closed dots (via terminal_ui)
• Rules-based strategy (deterministic), indicator fusion, risk controls
• Diagnostics (used/failed/skipped sources), simple P&L table, persistence
• API key validation and setup guidance

This file stays thin; most logic lives in trading_core/* modules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import zoneinfo
import tzlocal
from tqdm import tqdm

# Terminal UX (prompts and menus)
from .terminal_ui import (
    # main-mode & formatting
    prompt_main_mode,                 # -> "category" | "single_asset"
    print_header, print_table, print_kv, print_line,
    print_api_status, check_api_keys,  # API status functions

    # feature toggles & indicator selection
    ask_use_all_features, ask_use_rsi, ask_use_sma, ask_use_sentiment,
    prompt_indicator_bundle,          # -> {"all":bool,"selected":[...]} for old flow

    # category flow
    get_user_choice, get_user_budget, get_market_selection_details,

    # single-asset flow
    prompt_single_asset_input,        # -> {symbol, asset_class, market?, region?, timeframes?, indicators?, budget?, ...}
)

# Config (for warm-up/validation)
from trading_core.config import load_markets_config, get_market_info

# Data access facade
from trading_core.data_fetcher import (
    fetch_equities_data, fetch_crypto_data, fetch_forex_data, fetch_commodities_data,
    fetch_futures_data, fetch_warrants_data, fetch_funds_data,
    diagnostics_for, fetch_single_symbol_quote,
)

# Strategy (deterministic rules engine)
from trading_core.strategy.rules_engine import (
    analyze_market_batch,      # (rows, market_ctx, feature_flags, budget) -> [recs]
    analyze_single_asset,      # (row, asset_class, market_ctx, feature_flags, budget) -> rec
)

# Persistence
from trading_core.persistence.history_tracker import log_trade
from trading_core.persistence.performance_evaluator import evaluate_previous_session

# Logging
from trading_core.utils.logging import get_logger

log = get_logger(__name__)

# Detect local timezone (fallback to EU/Paris)
try:
    LOCAL_TZ = zoneinfo.ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = zoneinfo.ZoneInfo("Europe/Paris")


# ──────────────────────────────────────────────────────────────────────────────
# Startup validation and API check
# ──────────────────────────────────────────────────────────────────────────────

def validate_startup_environment() -> None:
    """Validate environment and show API status at startup."""
    print_header("🚀 Trading Assistant - Backend")
    
    # Check API keys and show status
    api_status = check_api_keys()
    
    if not api_status:
        print("\n⚠️  Could not check API configuration")
        print("   This might affect data source availability.")
    else:
        # Count configured vs missing APIs
        financial_apis = ["TwelveData", "CryptoCompare", "Alpha Vantage"]
        ai_apis = ["OpenAI", "Anthropic"]
        
        financial_configured = sum(1 for api in financial_apis if api_status.get(api, False))
        ai_configured = sum(1 for api in ai_apis if api_status.get(api, False))
        
        total_configured = financial_configured + ai_configured
        
        if total_configured == 0:
            print("\n🔧 API Configuration Status:")
            print("   ❌ No API keys configured")
            print("   📊 Using free data sources only (limited coverage)")
            print("   💡 For better data: run 'make setup-api' to configure APIs")
            
        elif financial_configured == 0:
            print("\n🔧 API Configuration Status:")
            print(f"   ⚠️  {ai_configured} AI API(s) configured, but no financial data APIs")
            print("   📊 Using free data sources for market data")
            print("   💡 Consider adding financial APIs: run 'make setup-api'")
            
        elif total_configured < 5:
            print("\n🔧 API Configuration Status:")
            print(f"   ✅ {financial_configured} financial API(s) configured")
            if ai_configured > 0:
                print(f"   ✅ {ai_configured} AI API(s) configured")
            print("   📊 Good data coverage available")
            if financial_configured + ai_configured < 5:
                print("   💡 For full coverage: run 'make setup-api' to add more APIs")
                
        else:
            print("\n🔧 API Configuration Status:")
            print("   ✅ All APIs configured - maximum data coverage!")
    
    # Show quick API breakdown if any are missing
    if api_status:
        missing_apis = [api for api, configured in api_status.items() if not configured]
        if missing_apis:
            print(f"\n📋 Missing APIs: {', '.join(missing_apis)}")
            print("   ℹ️  Run 'make show-api-info' for registration links")


def show_data_source_disclaimer() -> None:
    """Show disclaimer about data sources and their limitations."""
    print("\n" + "─" * 60)
    print("📊 Data Source Information")
    print("─" * 60)
    print("• The app uses multiple fallback data sources")
    print("• Free sources may have rate limits or delayed data")
    print("• API keys provide better coverage and real-time data")
    print("• All data is for informational purposes only")
    print("─" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _merge_market_context(selection: Optional[Dict[str, Any]], category_label: str) -> Dict[str, Any]:
    """
    Normalize a market context dict for downstream strategy/rules.
    """
    if not selection:
        return {"category": category_label, "market": None, "region": None}
    
    return {
        "category": category_label,
        "market": selection.get("market"),
        "region": selection.get("region"),
        "timezone": selection.get("timezone", "UTC"),
        "sessions": selection.get("sessions", []),
        "trading_days": selection.get("trading_days", [0, 1, 2, 3, 4]),
    }


def _feature_flags(use_rsi: bool, use_sma: bool, use_sentiment: bool, 
                   selected_indicators: List[str]) -> Dict[str, Any]:
    """Build feature flags dict for strategy engine."""
    return {
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "selected_indicators": selected_indicators,
        "use_all_features": use_rsi and use_sma and use_sentiment,
    }


def _print_recommendations(recs: List[Dict], title: str = "Recommendations") -> None:
    """Print recommendations in a formatted table."""
    if not recs:
        print(f"\n❌ No {title.lower()} available.")
        return
    
    print_header(title)
    
    # Prepare table data
    headers = ["Asset", "Action", "Confidence", "Price $", "Target $", "Stop $", "Key Reasons"]
    rows = []
    
    for rec in recs[:10]:  # Limit to top 10
        asset = str(rec.get("asset", "N/A"))
        action = rec.get("action", "Hold")
        confidence = f"{rec.get('confidence', 0)}%"
        price = f"{rec.get('price', 0.0):.2f}"
        target = f"{rec.get('sell_target', 0.0):.2f}" if rec.get('sell_target', 0.0) > 0 else "-"
        stop = f"{rec.get('stop_loss', 0.0):.2f}" if rec.get('stop_loss', 0.0) > 0 else "-"
        reasons = str(rec.get("reasons", ""))[:50] + ("..." if len(str(rec.get("reasons", ""))) > 50 else "")
        
        rows.append([asset, action, confidence, price, target, stop, reasons])
    
    print_table(headers, rows)


def _print_diagnostics(category: str) -> None:
    """Print data source diagnostics."""
    try:
        diag = diagnostics_for(category)
        if diag:
            print_header("Data Source Diagnostics")
            print(f"✅ Used: {diag.get('used', 'Unknown')}")
            if diag.get('failed'):
                print(f"❌ Failed: {', '.join(diag['failed'])}")
            if diag.get('skipped'):
                print(f"⏭️  Skipped: {', '.join(diag['skipped'])}")
    except Exception as e:
        log.warning(f"Could not get diagnostics for {category}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Category flow (original 7-category workflow)
# ──────────────────────────────────────────────────────────────────────────────

def run_category_flow() -> None:
    """
    Main category-based analysis workflow.
    """
    # Validate startup environment first
    validate_startup_environment()
    
    # Show data source disclaimer
    show_data_source_disclaimer()
    
    # Feature selection
    use_all = ask_use_all_features()
    
    if use_all:
        use_rsi = use_sma = use_sentiment = True
        selected_inds = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
        print("🎯 Using ALL features and indicators for comprehensive analysis!")
        
        # Show detailed feature status
        print("─" * 44)
        print("💭 Sentiment Analysis Components:")
        print("   News Headlines: ✅ Enabled")
        print("   Social Media: ✅ Enabled") 
        print("   Fear & Greed Index: ✅ Enabled")
        print("─" * 44)
        print("📊 Technical Indicators Status:")
        for indicator in selected_inds:
            print(f"   {indicator}: ✅ Enabled")
        print("─" * 44)
        print(f"Total Indicators Active: {len(selected_inds)} of {len(selected_inds)} technical")
        print("Sentiment Components Active: 3 of 3 components")
        print("─" * 44)
        
    else:
        print("🔧 Configuring individual features...")
        use_rsi = ask_use_rsi()
        use_sma = ask_use_sma()
        use_sentiment = ask_use_sentiment()
        selected_inds = []
        
        if use_rsi:
            selected_inds.extend(["RSI"])
        if use_sma:
            selected_inds.extend(["SMA", "EMA"])
        
        print(f"📊 Selected indicators: {', '.join(selected_inds) if selected_inds else 'None'}")

    # Category and budget selection
    category = get_user_choice()
    budget = get_user_budget()
    
    # Market/region selection
    market_selection = get_market_selection_details()
    market_ctx = _merge_market_context(market_selection, category)
    
    # Data fetching with progress bar
    pbar = tqdm(total=3, desc="⏳ Processing")
    rows = []
    
    try:
        # Step 1: Fetch market data
        fetcher_map = {
            "crypto": fetch_crypto_data,
            "forex": fetch_forex_data,
            "equities": fetch_equities_data,
            "commodities": fetch_commodities_data,
            "futures": fetch_futures_data,
            "warrants": fetch_warrants_data,
            "funds": fetch_funds_data,
        }
        
        fetcher = fetcher_map.get(category)
        if fetcher:
            fetch_kwargs = {"include_history": True}
            if market_selection and market_selection.get("market"):
                fetch_kwargs["market"] = market_selection["market"]
            if market_selection and market_selection.get("region"):
                fetch_kwargs["region"] = market_selection["region"]
                
            rows = fetcher(**fetch_kwargs)
        else:
            log.error(f"Unknown category: {category}")
            
    except Exception as e:
        log.exception("Data fetching failed")
        print(f"\n❌ Data fetch error: {e}")
    finally:
        pbar.update(1)

    if not rows:
        print("\n❌ No market data fetched. Try a different market/region or check API configuration.")
        print("💡 Tip: Run 'make setup-api' to configure data APIs for better coverage.")
        pbar.close()
        return

    # Step 2: Analyze via rules engine (deterministic strategy)
    feat = _feature_flags(
        use_rsi=use_rsi,
        use_sma=use_sma,
        use_sentiment=use_sentiment,
        selected_indicators=selected_inds,
    )
    try:
        recs = analyze_market_batch(rows, market_ctx=market_ctx, feature_flags=feat, budget=budget)
    except Exception as e:
        log.exception("rules engine failed")
        print(f"\n❌ Strategy error: {e}")
        pbar.close()
        return
    pbar.update(1)

    # Step 3: Print outputs + diagnostics
    try:
        now = datetime.now(LOCAL_TZ)
        _print_recommendations(recs, title=f"Recommendations for Today ({now.strftime('%H:%M %Z')})")
        _print_diagnostics(category)
    finally:
        pbar.update(1)
        pbar.close()

    # Persist session data
    try:
        log_trade(
            market_type=category,
            budget=budget,
            recommendations=recs,
            features={
                "RSI": use_rsi, "SMA": use_sma, "Sentiment": use_sentiment,
                "Indicators": selected_inds
            },
        )
    except Exception as e:
        log.warning(f"Could not log trade: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Single-asset flow
# ──────────────────────────────────────────────────────────────────────────────

def run_single_asset_flow() -> None:
    """
    Analyze one specific symbol/coin/pair with user-selected indicators.
    """
    # Validate startup environment first
    validate_startup_environment()
    
    # Show data source disclaimer
    show_data_source_disclaimer()
    
    # Collect input (symbol, asset_class, optional market/region, chosen indicators)
    params = prompt_single_asset_input()
    symbol = params.get("symbol", "").strip()
    asset_class = params.get("asset_class", "equity").strip().lower()
    market = params.get("market")
    region = params.get("region")
    timeframes = params.get("timeframes", ["1d"])      # reserved for MTF extensions
    indicators = params.get("indicators", [])          # e.g., ["EMA","MACD","RSI","BBANDS","ATR"]
    budget = params.get("budget", 1000.0)

    if not symbol:
        print("❌ No symbol provided.")
        return

    # Build market context for single asset
    market_ctx = {
        "category": asset_class,
        "market": market,
        "region": region,
        "timezone": "UTC",  # Default for single asset
        "sessions": [],
        "trading_days": [0, 1, 2, 3, 4],
    }

    # Feature flags for single asset (enable all for comprehensive analysis)
    feat = _feature_flags(
        use_rsi=True,
        use_sma=True,
        use_sentiment=True,
        selected_indicators=indicators,
    )

    # Progress tracking
    pbar = tqdm(total=3, desc="⏳ Processing")
    
    # Step 1: Fetch single asset data
    try:
        row = fetch_single_symbol_quote(symbol, asset_class)
        if not row:
            print(f"\n❌ Could not fetch data for {symbol}")
            print("💡 Tip: Check symbol format or run 'make setup-api' for better data coverage.")
            pbar.close()
            return
    except Exception as e:
        log.exception("Single asset fetch failed")
        print(f"\n❌ Data fetch error: {e}")
        pbar.close()
        return
    finally:
        pbar.update(1)

    # Step 2: Analyze single asset
    try:
        rec = analyze_single_asset(row, asset_class=asset_class, market_ctx=market_ctx,
                                   feature_flags=feat, budget=budget)
    except Exception as e:
        log.exception("single-asset engine failed")
        print(f"\n❌ Strategy error: {e}")
        pbar.close()
        return
    pbar.update(1)

    # Step 3: Print recommendation
    try:
        now = datetime.now(LOCAL_TZ)
        print_header(f"Single-Asset Recommendation ({now.strftime('%H:%M %Z')})")
        
        # Format single asset recommendation
        headers = ["Asset", "Action", "Confidence", "Price $", "Target $", "Stop $", "Key Reasons"]
        reasons = rec.get("reasons", "")[:60] + ("..." if len(str(rec.get("reasons", ""))) > 60 else "")
        
        rows = [[
            str(rec.get("asset", symbol)),
            rec.get("action", "Hold"),
            f"{rec.get('confidence', 0)}%",
            f"{rec.get('price', 0.0):.2f}",
            f"{rec.get('sell_target', 0.0):.2f}" if rec.get('sell_target', 0.0) > 0 else "-",
            f"{rec.get('stop_loss', 0.0):.2f}" if rec.get('stop_loss', 0.0) > 0 else "-",
            reasons,
        ]]
        
        print_table(headers, rows)
        
    finally:
        pbar.update(1)
        pbar.close()

    # Persist single asset analysis
    try:
        log_trade(
            market_type=f"single:{asset_class}",
            budget=budget,
            recommendations=[rec],
            features={
                "RSI": True, "SMA": True, "Sentiment": True,
                "Indicators": indicators, "Timeframes": timeframes
            },
        )
    except Exception as e:
        log.warning(f"Could not log trade: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Trading Assistant CLI."""
    try:
        # Warm up markets config (for tz/sessions/labels); non-fatal on error
        try:
            markets = load_markets_config()
            if markets:
                sample_market = list(markets.keys())[0]
                sample_info = markets[sample_market]
                print(f"[I] Markets config loaded successfully: {len(markets)} markets available")
                print(f"[I] Sample market '{sample_market}': {sample_info.get('label', 'Unknown')}")
            else:
                print("[W] No markets loaded from configuration")
        except Exception as e:
            log.warning("Could not warm up markets config: %s", e)
            print(f"[W] Could not load markets config: {e}")

        # Check for previous session and show performance
        try:
            evaluate_previous_session()
        except Exception as e:
            log.warning(f"Could not evaluate previous session: {e}")

        # Select mode and run appropriate flow
        mode = prompt_main_mode()  # "category" or "single_asset"
        if mode == "single_asset":
            return run_single_asset_flow()
        return run_category_flow()
        
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Trading Assistant!")
        print("💡 Tip: Run 'make setup-api' to configure APIs for better data coverage")
    except Exception as e:
        log.exception("Unexpected error in main")
        print(f"\n❌ Unexpected error: {e}")
        print("💡 If this persists, please check the logs or run 'make debug-paths'")


if __name__ == "__main__":
    main()
