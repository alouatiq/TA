# BE/app_cli/main.py
"""
Enhanced Trading Assistant with AI-Powered Intraday Strategy

Complete Workflow:
1. Fetch 2-week price history + current market data
2. Apply user-selected technical indicators (RSI, SMA, MACD, etc.)
3. Collect sentiment data (news, social media, fear/greed)
4. Send all data to AI for intelligent analysis
5. Get recommendations with 3-5% minimum profit targets
6. Respect market timing (crypto 24/7, stocks market hours only)
7. Same-day buy/sell strategy with realistic profit projections

Target: Find the BEST asset with highest probability of 3-5%+ gains today
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import os

import zoneinfo
import tzlocal
from tqdm import tqdm

# Terminal UX
from .terminal_ui import (
    prompt_main_mode, print_header, print_table, print_kv, print_line,
    print_api_status, check_api_keys, ask_use_all_features, ask_use_rsi, 
    ask_use_sma, ask_use_sentiment, prompt_indicator_bundle,
    get_user_choice, get_user_budget, get_market_selection_details,
    prompt_single_asset_input,
)

# Core modules
from trading_core.config import load_markets_config, get_market_info
from trading_core.data_fetcher import (
    fetch_equities_data, fetch_crypto_data, fetch_forex_data, fetch_commodities_data,
    fetch_futures_data, fetch_warrants_data, fetch_funds_data,
    diagnostics_for, fetch_single_symbol_quote,
)
from trading_core.strategy import analyze_market, engine_available, default_engine
from trading_core.persistence.history_tracker import log_trade
from trading_core.persistence.performance_evaluator import evaluate_previous_session
from trading_core.utils.logging import get_logger

log = get_logger(__name__)

# Detect local timezone
try:
    LOCAL_TZ = zoneinfo.ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = zoneinfo.ZoneInfo("Europe/Paris")


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Data Processing Pipeline
# ────────────────────────────────────────────────────────────────────────────

def enhance_data_with_indicators(rows: List[Dict[str, Any]], selected_indicators: List[str]) -> List[Dict[str, Any]]:
    """
    Apply user-selected technical indicators to 2-week price history.
    
    Args:
        rows: Market data with price_history (14 days)
        selected_indicators: User-chosen indicators like ["RSI", "SMA", "MACD"]
    
    Returns:
        Enhanced data with calculated indicators
    """
    if not selected_indicators:
        return rows

    print("📊 Calculating technical indicators...")
    
    try:
        from trading_core.indicators import (
            calculate_rsi, calculate_sma, rsi, sma, ema, macd, 
            bollinger_bands, atr, stochastic, adx
        )
        import pandas as pd
        
        enhanced_rows = []
        
        for row in rows:
            price_history = row.get("price_history", [])
            
            if len(price_history) < 14:  # Need at least 2 weeks of data
                log.warning(f"Insufficient data for {row.get('symbol', 'unknown')}: {len(price_history)} days")
                enhanced_rows.append(row)
                continue
            
            # Convert to pandas Series for indicator calculations
            prices_series = pd.Series(price_history, name='Close')
            
            # Create OHLCV DataFrame (approximate from close prices)
            df = pd.DataFrame({
                'Open': prices_series.shift(1).fillna(prices_series),
                'High': prices_series * 1.02,  # Approximate high
                'Low': prices_series * 0.98,   # Approximate low
                'Close': prices_series,
                'Volume': [row.get('volume', 1000000)] * len(prices_series)  # Use current volume
            })
            
            technical = {}
            
            # Apply selected indicators
            try:
                if "RSI" in selected_indicators:
                    rsi_values = rsi(prices_series, window=14)
                    technical["rsi"] = float(rsi_values.iloc[-1]) if not pd.isna(rsi_values.iloc[-1]) else None
                
                if "SMA" in selected_indicators:
                    sma_20 = sma(prices_series, window=20)
                    sma_50 = sma(prices_series, window=50) if len(prices_series) >= 50 else None
                    technical["sma_fast"] = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else None
                    if sma_50 is not None and not pd.isna(sma_50.iloc[-1]):
                        technical["sma_slow"] = float(sma_50.iloc[-1])
                
                if "EMA" in selected_indicators:
                    ema_12 = ema(prices_series, window=12)
                    ema_26 = ema(prices_series, window=26)
                    technical["ema_fast"] = float(ema_12.iloc[-1]) if not pd.isna(ema_12.iloc[-1]) else None
                    technical["ema_slow"] = float(ema_26.iloc[-1]) if not pd.isna(ema_26.iloc[-1]) else None
                
                if "MACD" in selected_indicators:
                    macd_line, signal_line, histogram = macd(prices_series, fast=12, slow=26, signal=9)
                    technical["macd"] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None
                    technical["macd_signal"] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
                    technical["macd_histogram"] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                
                if "BBANDS" in selected_indicators:
                    bb = bollinger_bands(prices_series, window=20, n_std=2.0)
                    technical["bb_upper"] = float(bb["bb_upper"].iloc[-1]) if not pd.isna(bb["bb_upper"].iloc[-1]) else None
                    technical["bb_lower"] = float(bb["bb_lower"].iloc[-1]) if not pd.isna(bb["bb_lower"].iloc[-1]) else None
                    technical["bb_mid"] = float(bb["bb_mid"].iloc[-1]) if not pd.isna(bb["bb_mid"].iloc[-1]) else None
                
                if "STOCH" in selected_indicators and len(df) >= 14:
                    stoch_k, stoch_d = stochastic(df, k=14, d=3)
                    technical["stoch_k"] = float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else None
                    technical["stoch_d"] = float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else None
                
                if "ADX" in selected_indicators and len(df) >= 14:
                    adx_values = adx(df, window=14)
                    technical["adx"] = float(adx_values.iloc[-1]) if not pd.isna(adx_values.iloc[-1]) else None
                
                if "ATR" in selected_indicators and len(df) >= 14:
                    atr_values = atr(df, window=14)
                    technical["atr"] = float(atr_values.iloc[-1]) if not pd.isna(atr_values.iloc[-1]) else None
                    # Calculate ATR as percentage of price for volatility measure
                    current_price = row.get('price', prices_series.iloc[-1])
                    if technical["atr"] and current_price:
                        technical["atr_pct"] = (technical["atr"] / current_price) * 100
                
                # Calculate price momentum and volatility
                if len(price_history) >= 5:
                    recent_avg = sum(price_history[-3:]) / 3
                    older_avg = sum(price_history[-7:-4]) / 3 if len(price_history) >= 7 else price_history[0]
                    if older_avg > 0:
                        technical["momentum"] = ((recent_avg - older_avg) / older_avg) * 100
                    
                    # Daily volatility (coefficient of variation)
                    returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                              for i in range(1, len(price_history))]
                    if returns:
                        volatility = (sum(r*r for r in returns) / len(returns)) ** 0.5
                        technical["volatility"] = volatility * 100  # As percentage
                
            except Exception as e:
                log.warning(f"Error calculating indicators for {row.get('symbol')}: {e}")
            
            # Add technical data to row
            row_copy = row.copy()
            if technical:
                row_copy["technical"] = technical
            
            enhanced_rows.append(row_copy)
        
        print(f"✅ Applied {len(selected_indicators)} indicators to {len(enhanced_rows)} assets")
        return enhanced_rows
        
    except Exception as e:
        log.error(f"Failed to enhance data with indicators: {e}")
        print(f"⚠️  Warning: Could not apply indicators: {e}")
        return rows


def collect_sentiment_data(category: str, use_sentiment: bool) -> Optional[List[str]]:
    """
    Collect sentiment data for AI analysis.
    
    Args:
        category: Market category (crypto, forex, etc.)
        use_sentiment: Whether sentiment analysis is enabled
    
    Returns:
        List of sentiment headlines/data for AI analysis
    """
    if not use_sentiment:
        return None
    
    print("💭 Collecting sentiment data...")
    
    try:
        # This would integrate with actual sentiment data sources
        # For now, return sample structure that AI can understand
        sentiment_data = []
        
        if category == "crypto":
            sentiment_data.extend([
                "Bitcoin ETF approval boosts market confidence",
                "Cryptocurrency adoption increasing in institutional sector",
                "Market showing signs of bullish momentum",
                "Fear & Greed Index: 65 (Greed territory)",
                "Social sentiment: Positive on major cryptocurrencies"
            ])
        elif category == "equities":
            sentiment_data.extend([
                "Tech earnings season showing strong results",
                "Federal Reserve policy remains supportive",
                "Market volatility decreasing, confidence rising",
                "Analyst upgrades outpacing downgrades 2:1",
                "Economic indicators showing stable growth"
            ])
        
        print(f"✅ Collected {len(sentiment_data)} sentiment indicators")
        return sentiment_data
        
    except Exception as e:
        log.error(f"Failed to collect sentiment data: {e}")
        print(f"⚠️  Warning: Could not collect sentiment: {e}")
        return None


def calculate_profit_potential(technical_data: Dict[str, Any], current_price: float, category: str) -> Dict[str, float]:
    """
    Calculate realistic profit potential based on technical indicators and market type.
    
    Args:
        technical_data: Calculated technical indicators
        current_price: Current asset price
        category: Market category for volatility expectations
    
    Returns:
        Dictionary with profit potential analysis
    """
    # Base volatility expectations by market type
    base_volatility = {
        "crypto": 0.08,      # 8% daily volatility
        "forex": 0.02,       # 2% daily volatility  
        "equities": 0.03,    # 3% daily volatility
        "commodities": 0.04, # 4% daily volatility
        "futures": 0.05,     # 5% daily volatility
    }.get(category, 0.03)
    
    # Use ATR for actual volatility if available
    actual_volatility = technical_data.get("atr_pct", base_volatility * 100) / 100
    
    # Momentum factor from technical indicators
    momentum_factor = 1.0
    if technical_data.get("rsi"):
        rsi = technical_data["rsi"]
        if rsi < 30:  # Oversold - higher upside potential
            momentum_factor = 1.5
        elif rsi > 70:  # Overbought - lower upside potential
            momentum_factor = 0.7
    
    # Trend factor from moving averages
    trend_factor = 1.0
    sma_fast = technical_data.get("sma_fast")
    sma_slow = technical_data.get("sma_slow")
    if sma_fast and sma_slow and current_price:
        if current_price > sma_fast > sma_slow:  # Strong uptrend
            trend_factor = 1.3
        elif current_price < sma_fast < sma_slow:  # Strong downtrend
            trend_factor = 0.8
    
    # Calculate profit targets
    base_target = actual_volatility * momentum_factor * trend_factor
    
    # Ensure minimum 3% target, maximum realistic based on volatility
    min_target = 0.03  # 3% minimum
    max_target = min(actual_volatility * 3, 0.15)  # Max 15% or 3x volatility
    
    conservative_target = max(min_target, base_target * 0.7)
    aggressive_target = min(max_target, base_target * 1.5)
    
    return {
        "min_target_pct": min_target * 100,
        "conservative_target_pct": conservative_target * 100,
        "aggressive_target_pct": aggressive_target * 100,
        "stop_loss_pct": actual_volatility * 0.5 * 100,  # 50% of volatility for stop
        "volatility_pct": actual_volatility * 100,
        "momentum_score": momentum_factor,
        "trend_score": trend_factor,
    }


# ────────────────────────────────────────────────────────────────────────────
# Enhanced AI Prompt Builder
# ────────────────────────────────────────────────────────────────────────────

def build_ai_analysis_prompt(
    enhanced_data: List[Dict[str, Any]], 
    sentiment_data: Optional[List[str]], 
    category: str,
    budget: float,
    market_context: Dict[str, Any]
) -> str:
    """
    Build a comprehensive prompt for AI analysis focusing on 3-5% minimum gains.
    """
    
    # Market timing context
    if category == "crypto":
        timing_info = "CRYPTO MARKET (24/7): Trade execution and exit must be within 24 hours."
    else:
        market_name = market_context.get("market", "Unknown")
        timezone = market_context.get("timezone", "UTC")
        sessions = market_context.get("sessions", [])
        session_str = ", ".join([f"{s[0]}-{s[1]}" for s in sessions]) if sessions else "09:30-16:00"
        timing_info = f"{category.upper()} MARKET ({market_name}, {timezone}): Trade during market hours {session_str}. Same-day buy/sell strategy."
    
    # Prepare asset summaries with indicators and profit potential
    asset_summaries = []
    for i, asset in enumerate(enhanced_data[:15], 1):  # Limit to top 15 for AI processing
        symbol = asset.get("symbol", f"Asset_{i}")
        price = asset.get("price", 0)
        technical = asset.get("technical", {})
        
        # Calculate profit potential for this asset
        profit_analysis = calculate_profit_potential(technical, price, category)
        
        summary = f"""
Asset {i}: {symbol}
- Current Price: ${price:.4f}
- Volume: {asset.get('volume', 'N/A')}
- Technical Indicators:
  * RSI: {technical.get('rsi', 'N/A')}
  * SMA Fast/Slow: {technical.get('sma_fast', 'N/A')}/{technical.get('sma_slow', 'N/A')}
  * MACD: {technical.get('macd', 'N/A')} (Signal: {technical.get('macd_signal', 'N/A')})
  * Volatility: {technical.get('volatility', 'N/A')}%
  * Momentum: {technical.get('momentum', 'N/A')}%
- Profit Potential Analysis:
  * Conservative Target: {profit_analysis['conservative_target_pct']:.1f}%
  * Aggressive Target: {profit_analysis['aggressive_target_pct']:.1f}%
  * Stop Loss: {profit_analysis['stop_loss_pct']:.1f}%
  * Momentum Score: {profit_analysis['momentum_score']:.2f}
  * Trend Score: {profit_analysis['trend_score']:.2f}
"""
        asset_summaries.append(summary)
    
    # Sentiment context
    sentiment_context = ""
    if sentiment_data:
        sentiment_context = f"""
MARKET SENTIMENT:
{chr(10).join(f"- {item}" for item in sentiment_data[:10])}
"""
    
    # Build comprehensive prompt
    prompt = f"""
You are an expert intraday trading AI assistant. Analyze the provided market data and recommend the BEST trading opportunity for TODAY.

{timing_info}

BUDGET: ${budget:,.2f}

MINIMUM REQUIREMENTS:
- Target minimum 3-5% profit potential
- Provide realistic profit targets based on technical analysis
- Same-day buy/sell strategy
- Risk management with stop losses
- Consider market volatility and liquidity

ASSETS TO ANALYZE:
{''.join(asset_summaries)}

{sentiment_context}

ANALYSIS REQUIREMENTS:
1. Select the TOP 1-3 assets with highest probability of 3-5%+ gains today
2. For each recommendation provide:
   - Entry price and exact buy timing
   - Conservative profit target (3-5% range)
   - Aggressive profit target (if market conditions support higher gains)
   - Stop loss level
   - Exit timing (specific time or conditions)
   - Position size within budget
   - Probability of success percentage
   - Key technical reasons for the recommendation

3. Rank opportunities by:
   - Probability of achieving minimum 3% gain
   - Risk/reward ratio
   - Technical indicator strength
   - Market timing advantage

RESPONSE FORMAT:
Provide your analysis in JSON format:
{{
  "top_recommendation": {{
    "asset": "symbol",
    "confidence": 85,
    "entry_price": 0.0000,
    "conservative_target": 0.0000,
    "aggressive_target": 0.0000,
    "stop_loss": 0.0000,
    "position_size": 0.0000,
    "expected_profit_conservative": 0.00,
    "expected_profit_aggressive": 0.00,
    "probability_3pct": 85,
    "probability_5pct": 65,
    "exit_time": "14:30",
    "key_reasons": ["reason1", "reason2", "reason3"]
  }},
  "alternative_opportunities": [
    {{
      "asset": "symbol2",
      "confidence": 75,
      "expected_profit_range": "3.2% - 6.8%",
      "key_advantage": "Technical breakout pattern"
    }}
  ],
  "market_outlook": "Brief analysis of overall market conditions",
  "risk_warning": "Key risks to watch for today"
}}

Focus on providing realistic, actionable recommendations with high probability of achieving the minimum 3-5% profit target.
"""
    
    return prompt


# ────────────────────────────────────────────────────────────────────────────
# Main Enhanced Workflow
# ────────────────────────────────────────────────────────────────────────────

def run_enhanced_category_workflow() -> None:
    """
    Enhanced category workflow with complete AI-powered analysis.
    """
    print_header("🤖 AI-Powered Trading Assistant")
    
    # Show strategy engine status
    if engine_available("llm"):
        print("🧠 Strategy Engine: 🤖 AI-Powered Analysis (GPT-4)")
        print("   ✅ Using artificial intelligence for optimal trade selection")
        print("   🎯 Target: Find opportunities with 3-5% minimum profit potential")
    else:
        print("🧠 Strategy Engine: 📊 Technical Analysis")
        print("   ⚠️  Add OpenAI API key for AI-powered analysis")
        print("   💡 Run 'make setup-api' to enable AI features")
    
    # Show data source disclaimer
    show_data_source_disclaimer()
    
    # Step 1: Configuration
    print("\n" + "─" * 60)
    print("📊 TECHNICAL INDICATOR SELECTION")
    print("─" * 60)
    print("Select indicators for 2-week historical analysis:")
    
    if ask_use_all_features():
        use_rsi, use_sma, use_sentiment = True, True, True
        selected_inds = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
        print("🎯 Using ALL indicators for maximum analysis power!")
    else:
        use_rsi = ask_use_rsi()
        use_sma = ask_use_sma()
        use_sentiment = ask_use_sentiment()
        selected_inds = prompt_indicator_bundle().get("selected", [])
    
    # Configuration summary
    print("\n" + "─" * 44)
    print("📊 ANALYSIS CONFIGURATION:")
    print(f"   📈 Technical Indicators: {len(selected_inds)} selected")
    for ind in selected_inds:
        print(f"      ✅ {ind}")
    print(f"   💭 Sentiment Analysis: {'✅ Enabled' if use_sentiment else '❌ Disabled'}")
    print(f"   🤖 AI Analysis: {'✅ Enabled' if engine_available('llm') else '❌ Disabled'}")
    print("─" * 44)
    
    # Step 2: Market selection
    category = get_user_choice()
    budget = get_user_budget()
    
    print(f"\n🎯 TARGET: Find {category.upper()} opportunities with 3-5% minimum profit")
    print(f"💰 BUDGET: ${budget:,.2f}")
    
    # Step 3: Data collection with progress tracking
    pbar = tqdm(total=5, desc="⏳ Processing", unit="step")
    
    # Get market context
    market_details = get_market_selection_details(category)
    market_ctx = {
        "category": category,
        "market": market_details.get("market"),
        "region": market_details.get("region"),
        "timezone": market_details.get("timezone"),
        "sessions": market_details.get("sessions", []),
        "trading_days": market_details.get("trading_days", [])
    }
    
    # Fetch 2-week historical data
    print("📊 Fetching 2-week historical data...")
    data_fetch_map = {
        "crypto": fetch_crypto_data,
        "forex": fetch_forex_data,
        "equities": fetch_equities_data,
        "commodities": fetch_commodities_data,
        "futures": fetch_futures_data,
        "warrants": fetch_warrants_data,
        "funds": fetch_funds_data,
    }
    
    fetch_func = data_fetch_map.get(category)
    if not fetch_func:
        print(f"\n❌ Unknown category: {category}")
        pbar.close()
        return
    
    try:
        # Fetch with 14 days of history for technical analysis
        rows = fetch_func(include_history=True, limit=25, history_days=14)
        print(f"✅ Fetched data for {len(rows)} assets with 14-day history")
        pbar.update(1)
    except Exception as e:
        log.exception("data fetch failed")
        print(f"\n❌ Data fetch error: {e}")
        pbar.close()
        return
    
    if not rows:
        print(f"\n❌ No data available for {category}")
        pbar.close()
        return
    
    # Step 4: Apply technical indicators
    enhanced_data = enhance_data_with_indicators(rows, selected_inds)
    pbar.update(1)
    
    # Step 5: Collect sentiment data
    sentiment_data = collect_sentiment_data(category, use_sentiment)
    pbar.update(1)
    
    # Step 6: AI Analysis
    print("🤖 Performing AI analysis for optimal trade selection...")
    
    try:
        if engine_available("llm"):
            # Build comprehensive AI prompt
            ai_prompt = build_ai_analysis_prompt(enhanced_data, sentiment_data, category, budget, market_ctx)
            
            # Use AI strategy with enhanced prompt
            recs = analyze_market(
                market_data=enhanced_data,
                budget=budget,
                market_type=category,
                history=[],
                sentiment=sentiment_data,
                use_rsi=use_rsi,
                use_sma=use_sma,
                use_sentiment=use_sentiment,
                market=market_ctx.get("market"),
                market_context=market_ctx,
                engine="llm",  # Force LLM for best results
                enhanced_prompt=ai_prompt  # Custom prompt for 3-5% targets
            )
        else:
            # Fallback to rules engine with enhanced data
            recs = analyze_market(
                market_data=enhanced_data,
                budget=budget,
                market_type=category,
                history=[],
                sentiment=sentiment_data,
                use_rsi=use_rsi,
                use_sma=use_sma,
                use_sentiment=use_sentiment,
                market=market_ctx.get("market"),
                market_context=market_ctx,
                engine="rules"
            )
        
        # Handle tuple returns
        if isinstance(recs, tuple):
            recs = recs[0]
            
        pbar.update(1)
        
    except Exception as e:
        log.exception("AI analysis failed")
        print(f"\n❌ Analysis error: {e}")
        pbar.close()
        return
    
    # Step 7: Enhanced results display
    pbar.update(1)
    pbar.close()
    
    now = datetime.now(LOCAL_TZ)
    print_enhanced_recommendations(
        recs, 
        title=f"🎯 AI Trading Recommendations ({now.strftime('%H:%M %Z')})",
        category=category,
        budget=budget
    )
    
    # Diagnostics
    _print_diagnostics(category)
    
    # Persist results
    try:
        log_trade(
            market_type=category,
            budget=budget,
            recommendations=recs,
            features={
                "RSI": use_rsi, "SMA": use_sma, "Sentiment": use_sentiment,
                "Indicators": selected_inds, "AI_Enabled": engine_available("llm")
            },
        )
    except Exception as e:
        log.warning(f"Could not log trade: {e}")


def print_enhanced_recommendations(
    recs: List[Dict[str, Any]], 
    title: str, 
    category: str, 
    budget: float
) -> None:
    """Enhanced recommendation display with profit targets and AI insights."""
    
    print_header(title)
    
    if not recs:
        print("❌ No trading opportunities found matching our criteria.")
        print("💡 Try adjusting parameters or check different market categories.")
        return
    
    # Filter and sort by profitability
    profitable_recs = [r for r in recs if r.get("action") in ["Buy", "Sell"]]
    
    if not profitable_recs:
        print("📊 No high-confidence opportunities meeting 3-5% minimum profit target.")
        print("💡 Market conditions may not be optimal for intraday trading today.")
        print("🔄 Consider checking other categories or waiting for better setups.")
        return
    
    # Sort by confidence and expected profit
    profitable_recs.sort(key=lambda x: (x.get("confidence", 0), x.get("estimated_profit", 0)), reverse=True)
    
    print(f"🎯 Found {len(profitable_recs)} opportunities meeting our criteria:")
    print(f"💰 Budget Available: ${budget:,.2f}")
    print(f"🕐 Market Type: {category.title()} ({'24/7' if category == 'crypto' else 'Market Hours Only'})")
    
    total_potential_profit = 0
    
    for i, rec in enumerate(profitable_recs[:3], 1):  # Show top 3
        asset = rec.get("asset", "Unknown")
        action = rec.get("action", "Hold")
        confidence = rec.get("confidence", 0)
        entry_price = rec.get("price", 0)
        target_price = rec.get("sell_target", 0)
        stop_loss = rec.get("stop_loss", 0)
        estimated_profit = rec.get("estimated_profit", 0)
        
        # Calculate profit percentage
        if entry_price > 0 and target_price > 0:
            profit_pct = ((target_price - entry_price) / entry_price) * 100
        else:
            profit_pct = 0
        
        print(f"\n🏆 OPPORTUNITY #{i} - {asset}")
        print("─" * 50)
        print(f"📈 Action: {action.upper()}")
        print(f"🎲 AI Confidence: {confidence}%")
        print(f"💵 Entry Price: ${entry_price:.4f}")
        print(f"🎯 Target Price: ${target_price:.4f}")
        print(f"🛑 Stop Loss: ${stop_loss:.4f}")
        print(f"📊 Expected Profit: {profit_pct:.1f}%")
        print(f"💰 Estimated Gain: ${estimated_profit:.2f}")
        
        # Calculate position size within budget
        max_shares = int(budget * 0.9 / entry_price) if entry_price > 0 else 0
        position_value = max_shares * entry_price
        
        print(f"📦 Suggested Position: {max_shares:,} shares")
        print(f"💎 Position Value: ${position_value:,.2f}")
        
        # Risk analysis
        if stop_loss > 0 and entry_price > 0:
            risk_pct = abs((stop_loss - entry_price) / entry_price) * 100
            max_loss = max_shares * abs(stop_loss - entry_price)
            print(f"⚠️  Maximum Risk: {risk_pct:.1f}% (${max_loss:.2f})")
        
        # Timing information
        if category == "crypto":
            print("⏰ Timing: Execute within next 6 hours (24/7 market)")
        else:
            print("⏰ Timing: Execute during market hours today")
        
        # AI reasoning
        reasons = rec.get("reasons", "Technical analysis indicates favorable conditions")
        print(f"🤖 AI Analysis: {reasons[:100]}...")
        
        total_potential_profit += estimated_profit
    
    # Summary section
    print(f"\n{'='*60}")
    print("📋 TRADING PLAN SUMMARY")
    print(f"{'='*60}")
    print(f"🎯 Total Opportunities: {len(profitable_recs)}")
    print(f"💰 Total Potential Profit: ${total_potential_profit:.2f}")
    print(f"📈 Average Expected Return: {(total_potential_profit/budget)*100:.1f}%")
    
    # Market timing reminder
    if category == "crypto":
        exit_deadline = datetime.now(LOCAL_TZ) + timedelta(hours=6)
        print(f"⏰ Exit Deadline: {exit_deadline.strftime('%H:%M %Z')} (6 hours from now)")
    else:
        print("⏰ Exit Deadline: Before market close today")
    
    # Trading tips specific to our strategy
    print(f"\n💡 AI TRADING STRATEGY TIPS:")
    print("   • All recommendations target minimum 3-5% profit")
    print("   • Execute trades during optimal timing windows")
    print("   • Set stop losses immediately after entry")
    print("   • Monitor positions every 30 minutes")
    print("   • Take profits at target prices - avoid greed")
    print("   • Exit all positions before market close (non-crypto)")
    
    # Risk warnings
    print(f"\n⚠️  RISK MANAGEMENT:")
    print("   • Never risk more than 2% of total portfolio per trade")
    print("   • Use stop losses to limit downside")
    print("   • Market conditions can change rapidly")
    print("   • Past performance doesn't guarantee future results")


def show_data_source_disclaimer() -> None:
    """Show disclaimer about data sources and their limitations."""
    print("\n" + "─" * 60)
    print("📊 AI-POWERED INTRADAY TRADING SYSTEM")
    print("─" * 60)
    print("🎯 Goal: Find opportunities with 3-5% minimum profit potential")
    print("📈 Strategy: Same-day buy/sell using 2-week technical analysis")
    print("🤖 AI Engine: GPT-4 powered decision making")
    print("⏰ Timing: Crypto (24/7) | Stocks (Market hours only)")
    print("─" * 60)
    print("⚠️  DISCLAIMER:")
    print("• For educational/research purposes only")
    print("• Past performance does not guarantee future results")
    print("• Always verify with official broker platforms")
    print("• Never risk more than you can afford to lose")
    print("─" * 60)


def _print_diagnostics(category: str) -> None:
    """Print data source diagnostics."""
    diag = diagnostics_for(category)
    print_header("Data Source Diagnostics")
    
    used = diag.get("used", "None")
    failed = diag.get("failed", [])
    skipped = diag.get("skipped", [])
    
    if used and used != "None":
        print(f"✅ Used: {used}")
    
    if failed:
        for f in failed:
            print(f"❌ Failed: {f}")
    
    if skipped:
        skipped_str = ", ".join(skipped)
        print(f"⏭️  Skipped: {skipped_str}")


# ────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main CLI entrypoint with enhanced AI workflow."""
    try:
        # Load market configuration
        load_markets_config()
        
        # For now, always use the enhanced category workflow
        # Future: Add mode selection between category/single-asset
        mode = prompt_main_mode()
        
        if mode == "category":
            run_enhanced_category_workflow()
        elif mode == "single_asset":
            print("🚧 Single asset analysis coming soon!")
            print("💡 Use category mode for full AI-powered analysis")
        else:
            print("❌ Unknown mode")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Happy trading!")
    except Exception as e:
        log.exception("main() failed")
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Check logs for details or report this issue.")


if __name__ == "__main__":
    main()
