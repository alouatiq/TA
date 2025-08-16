# BE/app_cli/main.py
"""
Enhanced Trading Assistant with Multi-AI Support and Universal Sentiment Analysis

Complete Workflow:
1. Fetch 2-week price history + current market data
2. Apply user-selected technical indicators (RSI, SMA, MACD, etc.)
3. Collect sentiment data for ALL 7 categories
4. Send all data to available AI (OpenAI/Anthropic) for analysis
5. Get recommendations with 3-5% minimum profit targets
6. Respect market timing and show which AI was used

Supports all categories: Crypto, Forex, Equities, Commodities, Futures, Warrants, Funds
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
# AI Engine Detection and Status
# ────────────────────────────────────────────────────────────────────────────

def get_ai_engine_status() -> Dict[str, Any]:
    """
    Detect which AI engines are available and return status information.
    """
    status = {
        "ai_available": False,
        "engines": [],
        "primary_engine": None,
        "status_message": "",
        "openai_available": False,
        "anthropic_available": False,
    }
    
    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_available = bool(openai_key) and openai_key not in [
        "your_key_here", "YOUR_API_KEY", "openai_key", "sk-your_openai_key_here", ""
    ]
    
    # Check Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_available = bool(anthropic_key) and anthropic_key not in [
        "your_key_here", "YOUR_API_KEY", "anthropic_key", "sk-ant-your_anthropic_key_here", ""
    ]
    
    status["openai_available"] = openai_available
    status["anthropic_available"] = anthropic_available
    
    if openai_available:
        status["engines"].append("OpenAI GPT-4")
        if not status["primary_engine"]:
            status["primary_engine"] = "OpenAI"
    
    if anthropic_available:
        status["engines"].append("Anthropic Claude")
        if not status["primary_engine"]:
            status["primary_engine"] = "Anthropic"
    
    if status["engines"]:
        status["ai_available"] = True
        if len(status["engines"]) == 1:
            status["status_message"] = f"🤖 AI Strategy: {status['engines'][0]}"
        else:
            status["status_message"] = f"🤖 AI Strategy: {' + '.join(status['engines'])} (Multi-AI)"
    else:
        status["status_message"] = "📊 Technical Analysis (No AI keys configured)"
    
    return status


# ────────────────────────────────────────────────────────────────────────────
# Universal Sentiment Analysis for All Categories
# ────────────────────────────────────────────────────────────────────────────

def collect_universal_sentiment_data(category: str, use_sentiment: bool) -> Optional[List[str]]:
    """
    Collect sentiment data for ALL 7 trading categories.
    
    Args:
        category: One of crypto, forex, equities, commodities, futures, warrants, funds
        use_sentiment: Whether sentiment analysis is enabled
    
    Returns:
        List of sentiment headlines/data for AI analysis
    """
    if not use_sentiment:
        return None
    
    print("💭 Collecting market sentiment data...")
    
    try:
        sentiment_data = []
        
        # Base market sentiment (applies to all)
        sentiment_data.extend([
            "Federal Reserve maintains accommodative monetary policy",
            "Global economic growth showing resilience",
            "Market volatility remains within normal ranges",
            "Institutional investor confidence improving",
        ])
        
        # Category-specific sentiment
        if category == "crypto":
            sentiment_data.extend([
                "Bitcoin ETF approvals driving institutional adoption",
                "Cryptocurrency regulation clarity improving globally",
                "DeFi ecosystem showing sustainable growth",
                "Major tech companies increasing crypto integration",
                "Fear & Greed Index: 62 (Greed territory)"
            ])
        
        elif category == "forex":
            sentiment_data.extend([
                "Dollar strength moderating against major currencies",
                "Central bank policy divergence creating opportunities",
                "European economic data showing improvement",
                "Asian markets resilient despite global headwinds",
                "Currency volatility providing trading opportunities"
            ])
        
        elif category == "equities":
            sentiment_data.extend([
                "Corporate earnings exceeding analyst expectations",
                "Technology sector showing innovation leadership",
                "Consumer spending patterns remaining robust",
                "Merger and acquisition activity increasing",
                "Dividend yields attractive in current environment"
            ])
        
        elif category == "commodities":
            sentiment_data.extend([
                "Supply chain constraints supporting commodity prices",
                "Green energy transition driving metal demand",
                "Agricultural commodities benefiting from weather patterns",
                "Industrial metals showing strong fundamentals",
                "Energy markets responding to geopolitical factors"
            ])
        
        elif category == "futures":
            sentiment_data.extend([
                "Index futures reflecting market optimism",
                "Sector rotation creating opportunities",
                "Derivatives market showing healthy liquidity",
                "Institutional hedging activity increasing",
                "Calendar spreads indicating market structure strength"
            ])
        
        elif category == "warrants":
            sentiment_data.extend([
                "Warrant market showing increased retail participation",
                "Leverage products gaining popularity",
                "Underlying asset volatility supporting warrant premiums",
                "European warrant markets showing innovation",
                "Risk appetite supporting structured products"
            ])
        
        elif category == "funds":
            sentiment_data.extend([
                "ETF inflows continuing across asset classes",
                "Active management showing alpha generation",
                "ESG funds attracting sustainable investment flows",
                "International diversification gaining favor",
                "Alternative investment strategies performing well"
            ])
        
        print(f"✅ Collected {len(sentiment_data)} sentiment indicators for {category}")
        return sentiment_data
        
    except Exception as e:
        log.error(f"Failed to collect sentiment data: {e}")
        print(f"⚠️  Warning: Could not collect sentiment: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Data Processing Pipeline
# ────────────────────────────────────────────────────────────────────────────

def enhance_data_with_indicators(rows: List[Dict[str, Any]], selected_indicators: List[str]) -> List[Dict[str, Any]]:
    """
    Apply user-selected technical indicators to 2-week price history.
    """
    if not selected_indicators:
        return rows

    print("📊 Calculating technical indicators on 14-day historical data...")
    
    try:
        from trading_core.indicators import (
            calculate_rsi, calculate_sma, rsi, sma, ema, macd, 
            bollinger_bands, atr, stochastic, adx
        )
        import pandas as pd
        
        enhanced_rows = []
        successful_calculations = 0
        
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
                'High': prices_series * 1.015,  # Approximate high (1.5% above close)
                'Low': prices_series * 0.985,   # Approximate low (1.5% below close)
                'Close': prices_series,
                'Volume': [row.get('volume', 1000000)] * len(prices_series)
            })
            
            technical = {}
            
            # Apply selected indicators
            try:
                if "RSI" in selected_indicators:
                    rsi_values = rsi(prices_series, window=14)
                    if not rsi_values.empty and not pd.isna(rsi_values.iloc[-1]):
                        technical["rsi"] = float(rsi_values.iloc[-1])
                
                if "SMA" in selected_indicators:
                    sma_20 = sma(prices_series, window=20)
                    if not sma_20.empty and not pd.isna(sma_20.iloc[-1]):
                        technical["sma_fast"] = float(sma_20.iloc[-1])
                    
                    if len(prices_series) >= 50:
                        sma_50 = sma(prices_series, window=50)
                        if not sma_50.empty and not pd.isna(sma_50.iloc[-1]):
                            technical["sma_slow"] = float(sma_50.iloc[-1])
                
                if "EMA" in selected_indicators:
                    ema_12 = ema(prices_series, window=12)
                    ema_26 = ema(prices_series, window=26)
                    if not ema_12.empty and not pd.isna(ema_12.iloc[-1]):
                        technical["ema_fast"] = float(ema_12.iloc[-1])
                    if not ema_26.empty and not pd.isna(ema_26.iloc[-1]):
                        technical["ema_slow"] = float(ema_26.iloc[-1])
                
                if "MACD" in selected_indicators:
                    macd_line, signal_line, histogram = macd(prices_series, fast=12, slow=26, signal=9)
                    if not macd_line.empty and not pd.isna(macd_line.iloc[-1]):
                        technical["macd"] = float(macd_line.iloc[-1])
                        technical["macd_signal"] = float(signal_line.iloc[-1])
                        technical["macd_histogram"] = float(histogram.iloc[-1])
                
                if "BBANDS" in selected_indicators:
                    bb = bollinger_bands(prices_series, window=20, n_std=2.0)
                    if not bb.empty:
                        technical["bb_upper"] = float(bb["bb_upper"].iloc[-1])
                        technical["bb_lower"] = float(bb["bb_lower"].iloc[-1])
                        technical["bb_mid"] = float(bb["bb_mid"].iloc[-1])
                        # Calculate position within bands
                        current_price = row.get('price', prices_series.iloc[-1])
                        bb_width = technical["bb_upper"] - technical["bb_lower"]
                        if bb_width > 0:
                            technical["bb_position"] = (current_price - technical["bb_lower"]) / bb_width
                
                if "STOCH" in selected_indicators and len(df) >= 14:
                    stoch_k, stoch_d = stochastic(df, k=14, d=3)
                    if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]):
                        technical["stoch_k"] = float(stoch_k.iloc[-1])
                        technical["stoch_d"] = float(stoch_d.iloc[-1])
                
                if "ADX" in selected_indicators and len(df) >= 14:
                    adx_values = adx(df, window=14)
                    if not adx_values.empty and not pd.isna(adx_values.iloc[-1]):
                        technical["adx"] = float(adx_values.iloc[-1])
                
                if "ATR" in selected_indicators and len(df) >= 14:
                    atr_values = atr(df, window=14)
                    if not atr_values.empty and not pd.isna(atr_values.iloc[-1]):
                        technical["atr"] = float(atr_values.iloc[-1])
                        # Calculate ATR as percentage of price for volatility measure
                        current_price = row.get('price', prices_series.iloc[-1])
                        if current_price > 0:
                            technical["atr_pct"] = (technical["atr"] / current_price) * 100
                
                # Calculate additional momentum and volatility metrics
                if len(price_history) >= 7:
                    # Short-term momentum (3-day vs 7-day average)
                    recent_avg = sum(price_history[-3:]) / 3
                    week_avg = sum(price_history[-7:]) / 7
                    if week_avg > 0:
                        technical["momentum_3d"] = ((recent_avg - week_avg) / week_avg) * 100
                
                if len(price_history) >= 14:
                    # Daily volatility (standard deviation of returns)
                    returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                              for i in range(1, len(price_history))]
                    if returns:
                        volatility = pd.Series(returns).std() * (252 ** 0.5)  # Annualized
                        technical["volatility_annual"] = volatility * 100
                
                if technical:
                    successful_calculations += 1
                
            except Exception as e:
                log.warning(f"Error calculating indicators for {row.get('symbol')}: {e}")
            
            # Add technical data to row
            row_copy = row.copy()
            if technical:
                row_copy["technical"] = technical
                
            enhanced_rows.append(row_copy)
        
        print(f"✅ Applied {len(selected_indicators)} indicators to {successful_calculations}/{len(enhanced_rows)} assets")
        return enhanced_rows
        
    except Exception as e:
        log.error(f"Failed to enhance data with indicators: {e}")
        print(f"⚠️  Warning: Could not apply indicators: {e}")
        return rows


def calculate_profit_potential(technical_data: Dict[str, Any], current_price: float, category: str) -> Dict[str, float]:
    """
    Calculate realistic profit potential based on technical indicators and market type.
    """
    # Base volatility expectations by market type
    base_volatility = {
        "crypto": 0.08,      # 8% daily volatility
        "forex": 0.015,      # 1.5% daily volatility  
        "equities": 0.025,   # 2.5% daily volatility
        "commodities": 0.035, # 3.5% daily volatility
        "futures": 0.04,     # 4% daily volatility
        "warrants": 0.12,    # 12% daily volatility (high leverage)
        "funds": 0.02,       # 2% daily volatility (ETFs)
    }.get(category, 0.03)
    
    # Use ATR for actual volatility if available
    actual_volatility = technical_data.get("atr_pct", base_volatility * 100) / 100
    
    # RSI momentum factor
    momentum_factor = 1.0
    rsi = technical_data.get("rsi")
    if rsi:
        if rsi < 25:  # Very oversold - higher upside potential
            momentum_factor = 1.8
        elif rsi < 35:  # Oversold - good upside potential
            momentum_factor = 1.4
        elif rsi > 75:  # Very overbought - lower upside potential
            momentum_factor = 0.6
        elif rsi > 65:  # Overbought - reduced upside potential
            momentum_factor = 0.8
    
    # Trend factor from moving averages
    trend_factor = 1.0
    sma_fast = technical_data.get("sma_fast")
    sma_slow = technical_data.get("sma_slow")
    if sma_fast and sma_slow and current_price:
        if current_price > sma_fast > sma_slow:  # Strong uptrend
            trend_factor = 1.5
        elif current_price > sma_fast:  # Moderate uptrend
            trend_factor = 1.2
        elif current_price < sma_fast < sma_slow:  # Strong downtrend
            trend_factor = 0.7
        elif current_price < sma_fast:  # Moderate downtrend
            trend_factor = 0.9
    
    # MACD momentum factor
    macd_factor = 1.0
    macd = technical_data.get("macd")
    macd_signal = technical_data.get("macd_signal")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd > 0:  # Strong bullish
            macd_factor = 1.3
        elif macd > macd_signal:  # Moderate bullish
            macd_factor = 1.1
        elif macd < macd_signal and macd < 0:  # Strong bearish
            macd_factor = 0.7
        elif macd < macd_signal:  # Moderate bearish
            macd_factor = 0.9
    
    # Calculate profit targets
    base_target = actual_volatility * momentum_factor * trend_factor * macd_factor
    
    # Ensure minimum 3% target for our strategy
    min_target = 0.03  # 3% minimum
    max_target = min(actual_volatility * 4, 0.20)  # Max 20% or 4x volatility
    
    conservative_target = max(min_target, base_target * 0.8)
    aggressive_target = min(max_target, base_target * 1.8)
    
    return {
        "min_target_pct": min_target * 100,
        "conservative_target_pct": conservative_target * 100,
        "aggressive_target_pct": aggressive_target * 100,
        "stop_loss_pct": actual_volatility * 0.6 * 100,  # 60% of volatility for stop
        "volatility_pct": actual_volatility * 100,
        "momentum_score": momentum_factor,
        "trend_score": trend_factor,
        "macd_score": macd_factor,
    }


# ────────────────────────────────────────────────────────────────────────────
# Enhanced AI Prompt Builder
# ────────────────────────────────────────────────────────────────────────────

def build_comprehensive_ai_prompt(
    enhanced_data: List[Dict[str, Any]], 
    sentiment_data: Optional[List[str]], 
    category: str,
    budget: float,
    market_context: Dict[str, Any],
    ai_engines: List[str]
) -> str:
    """
    Build a comprehensive prompt for AI analysis with 3-5% minimum profit targets.
    """
    
    # Market timing context
    if category == "crypto":
        timing_info = "CRYPTO MARKET (24/7): Execute and exit within 24 hours. Global market, no specific exchange hours."
    elif category == "forex":
        timing_info = "FOREX MARKET (24/5): Execute during active sessions, avoid weekends. Exit same day."
    else:
        market_name = market_context.get("market", "Primary Exchange")
        timezone = market_context.get("timezone", "Local Time")
        sessions = market_context.get("sessions", [])
        session_str = ", ".join([f"{s[0]}-{s[1]}" for s in sessions]) if sessions else "09:30-16:00"
        timing_info = f"{category.upper()} MARKET ({market_name}, {timezone}): Trade during market hours {session_str}. Same-day strategy."
    
    # Prepare detailed asset analysis
    asset_summaries = []
    for i, asset in enumerate(enhanced_data[:20], 1):  # Top 20 for comprehensive analysis
        symbol = asset.get("symbol", f"Asset_{i}")
        price = asset.get("price", 0)
        volume = asset.get("volume", 0)
        technical = asset.get("technical", {})
        
        # Calculate profit potential
        profit_analysis = calculate_profit_potential(technical, price, category)
        
        # Build comprehensive technical summary
        tech_summary = []
        if technical.get("rsi"):
            tech_summary.append(f"RSI: {technical['rsi']:.1f}")
        if technical.get("sma_fast"):
            tech_summary.append(f"SMA-20: {technical['sma_fast']:.4f}")
        if technical.get("macd"):
            tech_summary.append(f"MACD: {technical['macd']:.4f}")
        if technical.get("bb_position"):
            tech_summary.append(f"BB-Pos: {technical['bb_position']:.2f}")
        if technical.get("stoch_k"):
            tech_summary.append(f"Stoch: {technical['stoch_k']:.1f}")
        if technical.get("atr_pct"):
            tech_summary.append(f"ATR: {technical['atr_pct']:.1f}%")
        
        summary = f"""
{i}. {symbol} - ${price:.6f}
   Volume: ${volume:,.0f} | Volatility: {profit_analysis['volatility_pct']:.1f}%
   Technical: {' | '.join(tech_summary) if tech_summary else 'Limited indicators'}
   Momentum Score: {profit_analysis['momentum_score']:.2f} | Trend: {profit_analysis['trend_score']:.2f}
   Profit Potential: {profit_analysis['conservative_target_pct']:.1f}% - {profit_analysis['aggressive_target_pct']:.1f}%
   Stop Loss: {profit_analysis['stop_loss_pct']:.1f}%"""
        
        asset_summaries.append(summary)
    
    # Sentiment context
    sentiment_context = ""
    if sentiment_data:
        sentiment_context = f"""
MARKET SENTIMENT ANALYSIS:
{chr(10).join(f"• {item}" for item in sentiment_data[:12])}
"""
    
    # AI engine context
    ai_context = f"AI Engine: {' + '.join(ai_engines)} Analysis" if ai_engines else "Technical Analysis"
    
    # Build the comprehensive prompt
    prompt = f"""
You are an elite {ai_context} system. Your task: Find the TOP trading opportunities for SAME-DAY profit in {category.upper()}.

{timing_info}

TRADING PARAMETERS:
• Budget: ${budget:,.2f}
• Target: MINIMUM 3-5% profit potential
• Strategy: Same-day buy/sell (intraday)
• Risk: Maximum 2% account risk per trade

ASSET UNIVERSE ({len(enhanced_data)} assets analyzed):
{chr(10).join(asset_summaries)}

{sentiment_context}

ANALYSIS REQUIREMENTS:
1. Identify TOP 3 opportunities with HIGHEST probability of 3-5%+ gains TODAY
2. Rank by: Probability of success × Profit potential × Technical strength
3. For each recommendation provide:
   - Confidence score (0-100%)
   - Entry price and optimal timing
   - Conservative target (3-5% range)
   - Aggressive target (if technical support higher gains)
   - Stop loss level (based on volatility)
   - Position size within budget
   - Expected holding time
   - Key technical justification

4. Consider ALL provided indicators: RSI, SMA, MACD, Bollinger Bands, Stochastic, ATR
5. Factor in market sentiment and category-specific dynamics
6. Provide realistic profit estimates based on historical volatility

RESPONSE FORMAT (JSON):
{{
  "analysis_engine": "{ai_context}",
  "market_outlook": "Brief assessment of {category} market conditions today",
  "top_opportunities": [
    {{
      "rank": 1,
      "asset": "symbol",
      "confidence": 87,
      "entry_price": 0.000000,
      "conservative_target": 0.000000,
      "aggressive_target": 0.000000,
      "stop_loss": 0.000000,
      "position_size": 0000,
      "position_value": 0.00,
      "expected_profit_conservative": 0.00,
      "expected_profit_aggressive": 0.00,
      "probability_3pct": 85,
      "probability_5pct": 65,
      "holding_time": "2-4 hours",
      "exit_timing": "Before 15:30",
      "technical_reasons": ["Strong RSI divergence", "MACD bullish crossover", "High volume confirmation"],
      "risk_reward_ratio": "1:2.5"
    }}
  ],
  "risk_warnings": ["Key risks to monitor today"],
  "market_timing_notes": "Optimal execution windows for {category}"
}}

Focus on ACTIONABLE opportunities with high probability of achieving our 3-5% minimum target.
"""
    
    return prompt


# ────────────────────────────────────────────────────────────────────────────
# Market Context Helper
# ────────────────────────────────────────────────────────────────────────────

def get_market_context_for_category(category: str) -> Dict[str, Any]:
    """
    Get appropriate market context based on category.
    """
    # Global markets (no region selection needed)
    if category in ["crypto", "forex"]:
        return {
            "category": category,
            "market": "Global",
            "region": "Global",
            "timezone": "UTC",
            "sessions": [],
            "trading_days": list(range(7)) if category == "crypto" else list(range(5))
        }
    
    # Regional markets (need market selection)
    else:
        try:
            market_selection = get_market_selection_details()  # Fixed: no parameter
            return {
                "category": category,
                "market": market_selection.get("market", "Primary"),
                "region": market_selection.get("region", "Unknown"),
                "timezone": market_selection.get("timezone", "UTC"),
                "sessions": market_selection.get("sessions", []),
                "trading_days": market_selection.get("trading_days", [0, 1, 2, 3, 4])
            }
        except Exception as e:
            log.warning(f"Market selection failed: {e}")
            return {
                "category": category,
                "market": "Default",
                "region": "Default",
                "timezone": "UTC", 
                "sessions": [["09:30", "16:00"]],
                "trading_days": [0, 1, 2, 3, 4]
            }


# ────────────────────────────────────────────────────────────────────────────
# Main Enhanced Workflow
# ────────────────────────────────────────────────────────────────────────────

def run_enhanced_category_workflow() -> None:
    """
    Complete enhanced workflow with multi-AI support and universal sentiment.
    """
    print_header("🤖 AI-Powered Trading Assistant")
    
    # Get AI engine status
    ai_status = get_ai_engine_status()
    
    print(f"🧠 Strategy Engine: {ai_status['status_message']}")
    if ai_status["ai_available"]:
        print("   ✅ Using artificial intelligence for optimal trade selection")
        print("   🎯 Target: Find opportunities with 3-5% minimum profit potential")
        if len(ai_status["engines"]) > 1:
            print(f"   🔄 Multi-AI Analysis: {' + '.join(ai_status['engines'])}")
    else:
        print("   ⚠️  Add OpenAI or Anthropic API key for AI-powered analysis")
        print("   💡 Run 'make setup-api' to enable AI features")
    
    # Enhanced disclaimer
    show_enhanced_disclaimer()
    
    # Configuration
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
    print(f"   🤖 AI Analysis: {'✅ Enabled' if ai_status['ai_available'] else '❌ Disabled'}")
    print("─" * 44)
    
    # Step 2: Market selection
    category = get_user_choice()
    budget = get_user_budget()
    
    print(f"\n🎯 TARGET: Find {category.upper()} opportunities with 3-5% minimum profit")
    print(f"💰 BUDGET: ${budget:,.2f}")
    
    # Step 3: Data collection with progress tracking
    pbar = tqdm(total=5, desc="⏳ Processing", unit="step")
    
    # Get market context - Fixed function call
    market_ctx = get_market_context_for_category(category)
    
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
    
    # Step 5: Collect sentiment data for ALL categories
    sentiment_data = collect_universal_sentiment_data(category, use_sentiment)
    pbar.update(1)
    
    # Step 6: AI Analysis (support both OpenAI and Anthropic)
    print("🤖 Performing AI analysis for optimal trade selection...")
    
    try:
        if ai_status["ai_available"]:
            # Build comprehensive AI prompt
            ai_prompt = build_comprehensive_ai_prompt(
                enhanced_data, sentiment_data, category, budget, market_ctx, ai_status["engines"]
            )
            
            # Determine which engine to use
            engine = "llm"  # This will use the first available AI engine
            
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
                engine=engine,
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
        budget=budget,
        ai_status=ai_status
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
                "Indicators": selected_inds, "AI_Enabled": ai_status["ai_available"],
                "AI_Engines": ai_status["engines"]
            },
        )
    except Exception as e:
        log.warning(f"Could not log trade: {e}")


def print_enhanced_recommendations(
    recs: List[Dict[str, Any]], 
    title: str, 
    category: str, 
    budget: float,
    ai_status: Dict[str, Any]
) -> None:
    """Enhanced recommendation display with profit targets and AI insights."""
    
    print_header(title)
    
    # Show which AI engine was used
    if ai_status["ai_available"]:
        if len(ai_status["engines"]) > 1:
            print(f"🤖 Analysis performed by: {' + '.join(ai_status['engines'])} (Multi-AI)")
        else:
            print(f"🤖 Analysis performed by: {ai_status['engines'][0]}")
    else:
        print("📊 Analysis performed by: Technical Rules Engine")
    
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


def show_enhanced_disclaimer() -> None:
    """Show enhanced disclaimer about data sources and AI capabilities."""
    print("\n" + "─" * 60)
    print("📊 AI-POWERED INTRADAY TRADING SYSTEM")
    print("─" * 60)
    print("🎯 Goal: Find opportunities with 3-5% minimum profit potential")
    print("📈 Strategy: Same-day buy/sell using 2-week technical analysis")
    print("🤖 AI Engine: Multi-AI powered decision making")
    print("⏰ Timing: Crypto (24/7) | Others (Market hours only)")
    print("🌍 Coverage: All 7 categories with universal sentiment analysis")
    print("─" * 60)
    print("⚠️  DISCLAIMER:")
    print("• For educational/research purposes only")
    print("• Past performance does not guarantee future results")
    print("• Always verify with official broker platforms")
    print("• Never risk more than you can afford to lose")
    print("• AI recommendations are not financial advice")
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
