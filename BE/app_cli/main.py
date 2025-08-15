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

    # feature toggles & indicator selection
    ask_use_all_features, ask_use_rsi, ask_use_sma, ask_use_sentiment,
    prompt_category_indicator_selection,  # -> list[str] for indicator selection

    # category flow
    get_user_choice, get_user_budget, get_market_selection_details,

    # single-asset flow
    prompt_single_asset_input,        # -> {symbol, asset_class, market?, region?, timeframes?, indicators?, budget?, ...}
)

# Config (for warm-up/validation) - FIXED: Use correct imports
from trading_core.config import get_market_info, list_markets

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
# Constants: ALL available indicators
# ──────────────────────────────────────────────────────────────────────────────
ALL_TECHNICAL_INDICATORS = [
    "SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"
]

ALL_SENTIMENT_COMPONENTS = [
    "News Headlines", "Social Media", "Fear & Greed Index"
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _merge_market_context(selection: Optional[Dict[str, Any]], category_label: str) -> Dict[str, Any]:
    """
    Normalize a market context dict for downstream strategy/rules.
    """
    if not selection:
        return {"market": None, "region": None, "category": category_label}
    return {
        "market": selection.get("market"),
        "region": selection.get("region"),
        "category": category_label,
        "market_label": selection.get("market_label"),
        "region_label": selection.get("region_label"),
    }


def _feature_flags(
    use_rsi: bool = True,           # Default to True for ALL features
    use_sma: bool = True,           # Default to True for ALL features
    use_sentiment: bool = True,     # Default to True for ALL features
    selected_indicators: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Bundle feature flags for the strategy engines.
    By default, ALL indicators and features are enabled.
    """
    # If no indicators specified, use ALL available technical indicators
    if not selected_indicators:
        selected_indicators = ALL_TECHNICAL_INDICATORS.copy()
    
    return {
        "use_rsi": use_rsi,
        "use_sma": use_sma,
        "use_sentiment": use_sentiment,
        "selected_indicators": selected_indicators,
    }


def _print_recommendations(recommendations: List[Dict[str, Any]], *, title: str) -> None:
    """
    Pretty-print recommendations in a uniform table.
    """
    print_header(title)
    rows = []
    total_invest = 0.0
    gross_profit = 0.0
    for r in recommendations:
        qty = int(r.get("quantity", 0))
        price = float(r.get("price", 0.0))
        cost = qty * price
        total_invest += cost
        gross_profit += float(r.get("estimated_profit", 0.0))
        rows.append([
            str(r.get("asset", "")),
            qty,
            f"{price:.2f}",
            f"{cost:.2f}",
            f"{float(r.get('sell_target', 0.0)):.2f}",
            r.get("sell_time_disp", r.get("sell_time", "")),
            f"{float(r.get('estimated_profit', 0.0)):.2f}",
        ])
    print_table(
        headers=["Asset", "Qty", "Buy $", "Cost $", "Target $", "Sell@", "Profit $"],
        rows=rows
    )
    print_line()
    print_kv("Total Capital", f"{total_invest:.2f}")
    print_kv("Gross Profit", f"{gross_profit:.2f}")


def _print_diagnostics(category: str) -> None:
    di = diagnostics_for(category)
    used = di.get("used")
    failed = di.get("failed") or []
    skipped = di.get("skipped") or []
    print_header("Feature Check Summary")
    if used is not None:
        print_kv("Market data source", used or "Not available")
    if failed:
        print_kv("Failed sources", ", ".join(map(str, failed)))
    if skipped:
        print_kv("Skipped sources", ", ".join(map(str, skipped)))


def _include_history_needed(selected_indicators: List[str], use_rsi: bool, use_sma: bool) -> bool:
    """
    Decide whether to pull price history. If any technicals are on, we want it.
    Since we're using ALL indicators by default, this will almost always return True.
    """
    techs = {"SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"}
    on = set(i.upper() for i in (selected_indicators or []))
    return bool(use_rsi or use_sma or (techs & on))


def _warm_up_config() -> None:
    """
    Warm up markets configuration. Safe to call - handles errors gracefully.
    """
    try:
        # Test if we can load market info
        markets = list_markets()
        log.info(f"Markets config loaded successfully: {len(markets)} markets available")
        if markets:
            # Test loading first market info
            first_market = markets[0]
            info = get_market_info(first_market)
            log.info(f"Sample market '{first_market}': {info.get('label', 'N/A')}")
    except Exception as e:
        log.warning(f"Could not warm up markets config: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Category (multi-asset) flow - MODIFIED TO USE ALL INDICATORS BY DEFAULT
# ──────────────────────────────────────────────────────────────────────────────
def run_category_flow() -> None:
    """
    Original 7-category workflow (now uses ALL indicators by default).
    """
    # Yesterday's report
    prev = evaluate_previous_session()
    if prev:
        print_header("Yesterday's Score-card")
        rows = []
        for r in prev:
            high_txt = "?" if r.get("day_high") is None else f"{float(r['day_high']):.2f}"
            rows.append([r.get("asset",""), f"{float(r.get('target',0.0)):.2f}", high_txt, r.get("hit","")])
        print_table(["Asset","Target$","High$","Result"], rows)

    # NEW: Simplified flow - use ALL features by default with option to customize
    print_header("Feature Configuration")
    print("🚀 By default, ALL features and indicators are enabled for maximum analysis power!")
    print("   This includes: RSI, SMA, Sentiment Analysis, and ALL Technical Indicators")
    print("   Technical Indicators: SMA, EMA, MACD, ADX, RSI, STOCH, OBV, BBANDS, ATR")
    
    # Ask if user wants to use all features or customize
    use_all_defaults = input("\n✅ Use ALL features and indicators? [Y/n]: ").strip().lower()
    
    if use_all_defaults in {"", "y", "yes"}:
        # Use ALL features and indicators
        use_rsi = True
        use_sma = True
        use_sentiment = True
        selected_inds = ALL_TECHNICAL_INDICATORS.copy()
        print("🎯 Using ALL features and indicators for comprehensive analysis!")
    else:
        # Custom selection (legacy behavior)
        print("\n📊 Custom feature selection:")
        if ask_use_all_features():
            use_rsi = use_sma = use_sentiment = True
        else:
            use_rsi       = ask_use_rsi()
            use_sma       = ask_use_sma()
            use_sentiment = ask_use_sentiment()

        # Let user choose indicators
        selected_inds = prompt_category_indicator_selection()

    print_line()
    
    # Show sentiment components status individually
    print("💭 Sentiment Analysis Components:")
    for component in ALL_SENTIMENT_COMPONENTS:
        status = "✅ Enabled" if use_sentiment else "❌ Disabled"
        print(f"   {component}: {status}")
    
    print_line()
    
    # Show each technical indicator status individually
    print("📊 Technical Indicators Status:")
    for indicator in ALL_TECHNICAL_INDICATORS:
        status = "✅ Enabled" if indicator in selected_inds else "❌ Disabled"
        print(f"   {indicator}: {status}")
    
    print_line()
    print_kv("Total Indicators Active", f"{len(selected_inds)} of {len(ALL_TECHNICAL_INDICATORS)} technical")
    print_kv("Sentiment Components Active", f"{len(ALL_SENTIMENT_COMPONENTS) if use_sentiment else 0} of {len(ALL_SENTIMENT_COMPONENTS)} components")
    print_line()

    category = get_user_choice()     # canonical string
    budget   = get_user_budget()

    # Market context (for market-aware categories)
    selection = None
    try:
        selection = get_market_selection_details()
    except Exception:
        selection = None
    market_ctx = _merge_market_context(selection, category)

    include_history = _include_history_needed(selected_inds, use_rsi, use_sma)

    # progress bar across 3 stages
    pbar = tqdm(total=3, desc="⏳ Processing", unit="step")

    # 1) Fetch data
    try:
        rows: List[Dict[str, Any]] = []
        if category == "equities":
            rows = fetch_equities_data(include_history=include_history,
                                       market=market_ctx.get("market"),
                                       region=market_ctx.get("region"))
        elif category == "crypto":
            rows = fetch_crypto_data(include_history=include_history)
        elif category == "forex":
            rows = fetch_forex_data(include_history=include_history,
                                    region=market_ctx.get("region"))
        elif category == "commodities":
            rows = fetch_commodities_data(include_history=include_history,
                                          market=market_ctx.get("market"))
        elif category == "futures":
            rows = fetch_futures_data(include_history=include_history,
                                      market=market_ctx.get("market"),
                                      region=market_ctx.get("region"))
        elif category == "warrants":
            rows = fetch_warrants_data(include_history=include_history,
                                       market=market_ctx.get("market"),
                                       region=market_ctx.get("region"))
        elif category == "funds":
            rows = fetch_funds_data(include_history=include_history,
                                    market=market_ctx.get("market"))
        else:
            print("⚠️  Unknown category.")
            return
    except Exception as e:
        log.exception("data fetch failed")
        print(f"\n❌ Data fetch error: {e}")
        return
    finally:
        pbar.update(1)

    if not rows:
        print("\n❌ No market data fetched. Try a different market/region or later.")
        return

    # 2) Analyze via rules engine (deterministic strategy)
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
        return
    pbar.update(1)

    # 3) Print outputs + diagnostics
    try:
        now = datetime.now(LOCAL_TZ)
        _print_recommendations(recs, title=f"Recommendations for Today ({now.strftime('%H:%M %Z')})")
        _print_diagnostics(category)
    finally:
        pbar.update(1)
        pbar.close()

    # Persist
    log_trade(
        market_type=category,
        budget=budget,
        recommendations=recs,
        features={
            "RSI": use_rsi, "SMA": use_sma, "Sentiment": use_sentiment,
            "Indicators": selected_inds,
            "Total_Indicators_Used": len(selected_inds)
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-asset flow - MODIFIED TO USE ALL INDICATORS BY DEFAULT
# ──────────────────────────────────────────────────────────────────────────────
def run_single_asset_flow() -> None:
    """
    Analyze one specific symbol/coin/pair with ALL indicators by default.
    """
    # Collect input (symbol, asset_class, optional market/region, chosen indicators)
    params = prompt_single_asset_input()
    symbol       = params.get("symbol", "").strip()
    asset_class  = params.get("asset_class", "equity").strip().lower()
    market       = params.get("market")
    region       = params.get("region")
    timeframes   = params.get("timeframes", ["1d"])      # reserved for MTF extensions
    
    # MODIFIED: Use ALL indicators by default instead of just user selection
    user_indicators = params.get("indicators", [])
    
    # Ask user if they want all indicators or just their selection
    if user_indicators:
        print(f"\n🔍 You selected: {', '.join(user_indicators)}")
        use_all_indicators = input("🚀 Use ALL available indicators instead for better analysis? [Y/n]: ").strip().lower()
        if use_all_indicators in {"", "y", "yes"}:
            indicators = ALL_TECHNICAL_INDICATORS.copy()
            print("✅ Using ALL technical indicators for comprehensive analysis!")
        else:
            indicators = user_indicators
    else:
        indicators = ALL_TECHNICAL_INDICATORS.copy()
        print("🚀 No specific indicators selected - using ALL available indicators!")
    
    budget = params.get("budget", 1000.0)

    if not symbol:
        print("❌ No symbol provided.")
        return

    # Build market context
    market_ctx = {
        "market": market,
        "region": region,
        "category": f"single:{asset_class}",
    }

    # Enable sentiment by default for single asset analysis
    use_sentiment = params.get("use_sentiment", True)  # Default to True
    use_rsi = True  # Always enable since we're using comprehensive analysis
    use_sma = True  # Always enable since we're using comprehensive analysis

    feat = _feature_flags(
        use_rsi=use_rsi,
        use_sma=use_sma,
        use_sentiment=use_sentiment,
        selected_indicators=indicators,
    )

    # Show what's happening
    print_header(f"Single-Asset Analysis: {symbol.upper()}")
    print_kv("Asset Class", asset_class.capitalize())
    print_kv("Budget", f"${budget:.2f}")
    if market:
        print_kv("Market", market)
    if region:
        print_kv("Region", region)
    print_line()
    
    # Show sentiment components status individually  
    print("💭 Sentiment Analysis Components:")
    for component in ALL_SENTIMENT_COMPONENTS:
        status = "✅ Enabled" if use_sentiment else "❌ Disabled"
        print(f"   {component}: {status}")
    
    print_line()
    
    # Show each technical indicator status individually
    print("📊 Technical Indicators Status:")
    for indicator in ALL_TECHNICAL_INDICATORS:
        status = "✅ Enabled" if indicator in indicators else "❌ Disabled"
        print(f"   {indicator}: {status}")
    
    print_line()
    print_kv("Total Indicators Active", f"{len(indicators)} of {len(ALL_TECHNICAL_INDICATORS)} technical")
    print_kv("Sentiment Components Active", f"{len(ALL_SENTIMENT_COMPONENTS) if use_sentiment else 0} of {len(ALL_SENTIMENT_COMPONENTS)} components")
    print_line()

    # Progress tracking
    pbar = tqdm(total=3, desc="⏳ Processing", unit="step")

    # 1) Fetch single symbol
    try:
        row = fetch_single_symbol_quote(
            symbol=symbol,
            asset_class=asset_class,
            include_history=True,  # Always get history since we're using ALL indicators
            market=market,
            region=region,
        )
        if not row:
            print(f"\n❌ No data found for {symbol}")
            return
    except Exception as e:
        log.exception("single symbol fetch failed")
        print(f"\n❌ Data fetch error: {e}")
        return
    finally:
        pbar.update(1)

    # 2) Analyze single asset
    try:
        rec = analyze_single_asset(row, asset_class=asset_class, market_ctx=market_ctx,
                                   feature_flags=feat, budget=budget)
    except Exception as e:
        log.exception("single-asset engine failed")
        print(f"\n❌ Strategy error: {e}")
        return
    pbar.update(1)

    # 3) Print recommendation
    try:
        now = datetime.now(LOCAL_TZ)
        print_header(f"Single-Asset Recommendation ({now.strftime('%H:%M %Z')})")
        print_table(
            headers=["Asset", "Action", "Confidence", "Buy $", "Target $", "Stop $", "Key Reasons"],
            rows=[[
                str(rec.get("asset", symbol)),
                rec.get("action","Hold"),
                f"{float(rec.get('confidence',0.0))*100:.0f}%",
                f"{float(rec.get('price',0.0)):.2f}",
                f"{float(rec.get('sell_target',0.0)):.2f}",
                f"{float(rec.get('stop_loss',0.0)):.2f}",
                "; ".join(rec.get("key_reasons", [])[:4]),
            ]]
        )
    finally:
        pbar.update(1)
        pbar.close()

    # Persist
    log_trade(
        market_type=f"single:{asset_class}",
        budget=budget,
        recommendations=[rec],
        features={
            "RSI": use_rsi, "SMA": use_sma, "Sentiment": use_sentiment,
            "Indicators": indicators, "Timeframes": timeframes,
            "Total_Indicators_Used": len(indicators)
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # Warm up markets config (for tz/sessions/labels); non-fatal on error
    _warm_up_config()

    mode = prompt_main_mode()  # "category" or "single_asset"
    if mode == "single_asset":
        return run_single_asset_flow()
    return run_category_flow()


if __name__ == "__main__":
    main()
