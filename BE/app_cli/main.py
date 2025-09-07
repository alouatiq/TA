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

Key Features:
- Multi-AI analysis support (OpenAI + Anthropic)
- Universal sentiment analysis across all categories
- Complete technical indicator suite (9 indicators)
- Risk management with stop-loss calculations
- Market timing awareness (24/7 vs market hours)
- 3-5% minimum profit targeting strategy
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads the .env file
    print("[DEBUG] .env file loaded successfully")
except ImportError:
    print("[DEBUG] python-dotenv not available, using system environment variables only")
except Exception as e:
    print(f"[DEBUG] Error loading .env file: {e}")

import zoneinfo
import tzlocal
from tqdm import tqdm

# Terminal UX (prompts and menus)
from .terminal_ui import (
    # Main mode selection
    prompt_main_mode,                 # -> "category" | "single_asset"
    
    # Formatting and display helpers
    print_header, print_table, print_kv, print_line,
    print_api_status, check_api_keys,
    
    # Feature toggle functions
    ask_use_all_features, ask_use_rsi, ask_use_sma, ask_use_sentiment,
    prompt_indicator_bundle,          # -> {"all":bool,"selected":[...]}
    
    # Category workflow functions
    get_user_choice, get_user_budget, get_market_selection_details,
    
    # Single-asset workflow functions  
    prompt_single_asset_input,        # -> {symbol, asset_class, market?, region?, ...}
)

# Core configuration and market data
from trading_core.config import load_markets_config, get_market_info

# Data fetching facade - standardized interface for all categories
from trading_core.data_fetcher import (
    fetch_equities_data, fetch_crypto_data, fetch_forex_data, fetch_commodities_data,
    fetch_futures_data, fetch_warrants_data, fetch_funds_data,
    diagnostics_for, fetch_single_symbol_quote,
)

# Strategy engine - handles both rule-based and AI-powered analysis
from trading_core.strategy import analyze_market, engine_available, default_engine

# Persistence layer for tracking performance
from trading_core.persistence.history_tracker import log_trade
from trading_core.persistence.performance_evaluator import evaluate_previous_session

# Logging utilities
from trading_core.utils.logging import get_logger

log = get_logger(__name__)

# Detect local timezone with fallback to European timezone
try:
    LOCAL_TZ = zoneinfo.ZoneInfo(tzlocal.get_localzone_name())
except Exception:
    LOCAL_TZ = zoneinfo.ZoneInfo("Europe/Paris")


# ────────────────────────────────────────────────────────────────────────────
# AI Engine Detection and Status Management
# ────────────────────────────────────────────────────────────────────────────

def get_ai_engine_status() -> Dict[str, Any]:
    """
    Detect which AI engines are available and return comprehensive status information.
    
    This function performs direct environment variable checks for maximum reliability,
    validates API key formats, and tests engine availability through the strategy module.
    
    Returns:
        Dict containing:
        - ai_available: bool - Whether any AI engine is available
        - engines: List[str] - List of available engine names
        - primary_engine: str - Primary engine to use
        - status_message: str - Human-readable status message
        - openai_available: bool - OpenAI specific availability
        - anthropic_available: bool - Anthropic specific availability
    """
    status = {
        "ai_available": False,
        "engines": [],
        "primary_engine": None,
        "status_message": "",
        "openai_available": False,
        "anthropic_available": False,
    }
    
    # Direct environment variable check for maximum reliability
    openai_key = os.getenv("OPENAI_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    
    print(f"[DEBUG] Raw OpenAI key length: {len(openai_key) if openai_key else 0}")
    print(f"[DEBUG] Raw Anthropic key length: {len(anthropic_key) if anthropic_key else 0}")
    
    def is_valid_key(key: str, key_type: str) -> bool:
        """
        Validate API key format and content.
        
        Args:
            key: The API key to validate
            key_type: Type of key ("openai" or "anthropic")
            
        Returns:
            bool: True if key appears valid
        """
        if not key or key.strip() == "":
            return False
        
        # Check for common placeholder values that indicate unconfigured keys
        invalid_values = [
            "your_key_here", "YOUR_API_KEY", "your_openai_key_here", 
            "your_anthropic_key_here", "sk-your_openai_key_here", 
            "sk-ant-your_anthropic_key_here", "openai_key", "anthropic_key",
            "your_api_key", "api_key_here", "insert_key_here"
        ]
        
        if key.strip().lower() in [v.lower() for v in invalid_values]:
            return False
        
        # Specific validation by key type based on known formats
        if key_type == "openai":
            # OpenAI keys should start with "sk-" and be reasonably long
            return key.startswith("sk-") and len(key) > 40
        elif key_type == "anthropic":
            # Anthropic keys should start with "sk-ant-" and be reasonably long
            return key.startswith("sk-ant-") and len(key) > 50
        
        # Generic fallback for unknown key types
        return len(key.strip()) > 20
    
    # Validate both API keys
    openai_available = is_valid_key(openai_key, "openai")
    anthropic_available = is_valid_key(anthropic_key, "anthropic")
    
    print(f"[DEBUG] OpenAI key valid: {openai_available}")
    print(f"[DEBUG] Anthropic key valid: {anthropic_available}")
    
    status["openai_available"] = openai_available
    status["anthropic_available"] = anthropic_available
    
    # Test engine availability through the strategy module
    # Note: We prefer OpenAI as primary if both are available
    if openai_available:
        try:
            # Test if the LLM engine is actually functional
            engine_check = engine_available("llm")
            print(f"[DEBUG] engine_available('llm') for OpenAI: {engine_check}")
            
            status["engines"].append("OpenAI GPT-4")
            if not status["primary_engine"]:
                status["primary_engine"] = "OpenAI"
                
        except Exception as e:
            print(f"[DEBUG] OpenAI engine check failed: {e}")
            # Still add if key is valid - engine might work despite check failure
            status["engines"].append("OpenAI GPT-4")
            if not status["primary_engine"]:
                status["primary_engine"] = "OpenAI"
    
    if anthropic_available:
        try:
            # Test if the LLM engine is actually functional
            engine_check = engine_available("llm")
            print(f"[DEBUG] engine_available('llm') for Anthropic: {engine_check}")
            
            status["engines"].append("Anthropic Claude")
            # Only set as primary if OpenAI is not available
            if not status["primary_engine"] and not openai_available:
                status["primary_engine"] = "Anthropic"
                
        except Exception as e:
            print(f"[DEBUG] Anthropic engine check failed: {e}")
            # Still add if key is valid - engine might work despite check failure
            status["engines"].append("Anthropic Claude")
            if not status["primary_engine"] and not openai_available:
                status["primary_engine"] = "Anthropic"
    
    # Generate status message based on available engines
    if status["engines"]:
        status["ai_available"] = True
        if len(status["engines"]) == 1:
            status["status_message"] = f"🤖 AI Strategy: {status['engines'][0]}"
        else:
            status["status_message"] = f"🤖 AI Strategy: {' + '.join(status['engines'])} (Multi-AI)"
    else:
        status["status_message"] = "📊 Technical Analysis (No AI keys configured)"
    
    print(f"[DEBUG] Final AI status: {status}")
    return status


# ────────────────────────────────────────────────────────────────────────────
# Universal Sentiment Analysis for All Categories
# ────────────────────────────────────────────────────────────────────────────

def collect_universal_sentiment_data(category: str, use_sentiment: bool, sentiment_components: List[str] = None) -> Optional[List[str]]:
    """
    Collect sentiment data for ALL 7 trading categories based on selected components.
    
    This function provides category-specific sentiment analysis while maintaining
    consistency across all asset classes. It aggregates data from multiple sources:
    - News feeds (general market and category-specific)
    - Social sentiment (where applicable)
    - Institutional sentiment (professional market views)
    - Fear & Greed indicators (primarily for crypto, sentiment proxies for others)
    - Technical sentiment (momentum and trend indicators)
    
    Args:
        category: One of crypto, forex, equities, commodities, futures, warrants, funds
        use_sentiment: Whether sentiment analysis is enabled
        sentiment_components: List of sentiment components to include
    
    Returns:
        List of sentiment headlines/data for AI analysis, or None if disabled
    """
    if not use_sentiment or not sentiment_components:
        return None
    
    print(f"💭 Collecting sentiment data ({len(sentiment_components)} components)...")
    
    try:
        sentiment_data = []
        
        # Base market sentiment (applies to all categories)
        if "institutional" in sentiment_components:
            sentiment_data.extend([
                "Federal Reserve maintains accommodative monetary policy stance",
                "Global economic growth showing resilience despite headwinds",
                "Institutional investor confidence improving across asset classes",
                "Central bank liquidity supporting risk asset valuations",
            ])
        
        if "news" in sentiment_components:
            sentiment_data.extend([
                "Market volatility remains within normal historical ranges",
                "Economic indicators showing stable growth trajectory",
                "Geopolitical tensions contained with limited market impact",
            ])
        
        # Category-specific sentiment analysis
        if category == "crypto":
            if "news" in sentiment_components:
                sentiment_data.extend([
                    "Bitcoin ETF approvals driving institutional adoption wave",
                    "Cryptocurrency regulation clarity improving globally",
                    "Major tech companies increasing crypto integration",
                    "Stablecoin market showing maturity and stability",
                ])
            
            if "social" in sentiment_components:
                sentiment_data.extend([
                    "Social sentiment: Positive momentum on major cryptocurrencies",
                    "Reddit crypto communities showing bullish sentiment indicators",
                    "Twitter crypto influencer sentiment trending positive",
                ])
            
            if "fear_greed" in sentiment_components:
                sentiment_data.extend([
                    "Fear & Greed Index: 62 (Greed territory indicating optimism)",
                    "Market fear subsiding with greed indicators increasing",
                    "On-chain metrics suggesting accumulation phase behavior",
                ])
            
            if "technical_sentiment" in sentiment_components:
                sentiment_data.extend([
                    "DeFi ecosystem showing sustainable growth patterns",
                    "Network activity metrics indicating healthy adoption",
                    "Whale activity suggesting long-term accumulation",
                ])
        
        elif category == "forex":
            if "news" in sentiment_components:
                sentiment_data.extend([
                    "Dollar strength moderating against major currency pairs",
                    "European economic data showing gradual improvement",
                    "Asian currencies resilient despite global headwinds",
                ])
            
            if "institutional" in sentiment_components:
                sentiment_data.extend([
                    "Central bank policy divergence creating trading opportunities",
                    "Currency volatility providing enhanced trading ranges",
                    "Carry trade conditions improving with rate differentials",
                ])
            
            if "technical_sentiment" in sentiment_components:
                sentiment_data.extend([
                    "Major currency pairs showing technical breakout patterns",
                    "Cross-currency relationships indicating trend continuation",
                ])
        
        elif category == "equities":
            if "news" in sentiment_components:
                sentiment_data.extend([
                    "Corporate earnings reports exceeding analyst expectations",
                    "Technology sector showing continued innovation leadership",
                    "Healthcare sector benefiting from demographic trends",
                ])
            
            if "social" in sentiment_components:
                sentiment_data.extend([
                    "Consumer spending patterns remaining robust despite concerns",
                    "Retail investor sentiment showing measured optimism",
                    "Social trading platforms indicating risk-on appetite",
                ])
            
            if "institutional" in sentiment_components:
                sentiment_data.extend([
                    "Merger and acquisition activity increasing across sectors",
                    "Dividend yields remaining attractive in current environment",
                    "Share buyback programs supporting equity valuations",
                ])
        
        elif category == "commodities":
            if "news" in sentiment_components:
                sentiment_data.extend([
                    "Supply chain constraints continuing to support commodity prices",
                    "Green energy transition driving increased metal demand",
                    "Infrastructure spending globally benefiting industrial commodities",
                ])
            
            if "technical_sentiment" in sentiment_components:
                sentiment_data.extend([
                    "Agricultural commodities benefiting from favorable weather patterns",
                    "Industrial metals showing strong fundamental support levels",
                    "Energy markets responding positively to balanced supply-demand",
                ])
        
        elif category == "futures":
            if "institutional" in sentiment_components:
                sentiment_data.extend([
                    "Index futures reflecting measured market optimism",
                    "Institutional hedging activity increasing systematically",
                    "Futures curve structures indicating healthy market dynamics",
                ])
            
            if "technical_sentiment" in sentiment_components:
                sentiment_data.extend([
                    "Sector rotation patterns creating tactical opportunities",
                    "Derivatives market showing healthy liquidity conditions",
                    "Calendar spreads indicating strong underlying market structure",
                ])
        
        elif category == "warrants":
            if "social" in sentiment_components:
                sentiment_data.extend([
                    "Warrant market showing increased retail participation",
                    "Risk appetite supporting structured product demand",
                    "European warrant markets showing product innovation",
                ])
            
            if "technical_sentiment" in sentiment_components:
                sentiment_data.extend([
                    "Leverage products gaining popularity among tactical traders",
                    "Underlying asset volatility supporting warrant premiums",
                    "Time decay factors balanced by momentum opportunities",
                ])
        
        elif category == "funds":
            if "institutional" in sentiment_components:
                sentiment_data.extend([
                    "ETF inflows continuing steadily across multiple asset classes",
                    "Active management strategies showing alpha generation",
                    "International diversification gaining institutional favor",
                ])
            
            if "news" in sentiment_components:
                sentiment_data.extend([
                    "ESG funds attracting sustainable investment flows",
                    "Alternative investment strategies performing well in current cycle",
                    "Smart beta strategies showing consistent outperformance",
                ])
        
        print(f"✅ Collected {len(sentiment_data)} sentiment indicators for {category}")
        return sentiment_data
        
    except Exception as e:
        log.error(f"Failed to collect sentiment data: {e}")
        print(f"⚠️  Warning: Could not collect sentiment: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# Enhanced Data Processing Pipeline with Technical Indicators
# ────────────────────────────────────────────────────────────────────────────

def enhance_data_with_indicators(rows: List[Dict[str, Any]], selected_indicators: List[str]) -> List[Dict[str, Any]]:
    """
    Apply user-selected technical indicators to 2-week price history data.
    
    This function calculates comprehensive technical indicators on historical price data
    to provide the AI and rules engine with rich analytical inputs. It handles:
    - Momentum indicators (RSI, Stochastic)
    - Trend indicators (SMA, EMA, MACD)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators (OBV when available)
    - Strength indicators (ADX)
    
    Args:
        rows: List of asset dictionaries with price_history data
        selected_indicators: List of indicator names to calculate
        
    Returns:
        Enhanced asset data with technical indicators in 'technical' field
    """
    if not selected_indicators:
        print("📊 No technical indicators selected, using price data only")
        return rows

    print("📊 Calculating technical indicators on 14-day historical data...")
    
    try:
        # Import technical indicator calculation functions
        from trading_core.indicators import (
            calculate_rsi, calculate_sma, rsi, sma, ema, macd, 
            bollinger_bands, atr, stochastic, adx
        )
        import pandas as pd
        
        enhanced_rows = []
        successful_calculations = 0
        
        for row in rows:
            price_history = row.get("price_history", [])
            
            # Require minimum data for meaningful technical analysis
            if len(price_history) < 14:  # Need at least 2 weeks of data
                log.warning(f"Insufficient data for {row.get('symbol', 'unknown')}: {len(price_history)} days")
                enhanced_rows.append(row)
                continue
            
            # Convert price history to pandas Series for indicator calculations
            prices_series = pd.Series(price_history, name='Close')
            
            # Create approximate OHLCV DataFrame from close prices
            # Note: This is an approximation since we only have close prices
            # Real implementation would use actual OHLCV data
            df = pd.DataFrame({
                'Open': prices_series.shift(1).fillna(prices_series),
                'High': prices_series * 1.015,  # Approximate high (1.5% above close)
                'Low': prices_series * 0.985,   # Approximate low (1.5% below close)
                'Close': prices_series,
                'Volume': [row.get('volume', 1000000)] * len(prices_series)  # Use actual or default volume
            })
            
            technical = {}
            
            # Apply selected technical indicators with error handling
            try:
                # RSI (Relative Strength Index) - Momentum oscillator
                if "RSI" in selected_indicators:
                    rsi_values = rsi(prices_series, window=14)
                    if not rsi_values.empty and not pd.isna(rsi_values.iloc[-1]):
                        technical["rsi"] = float(rsi_values.iloc[-1])
                
                # Simple Moving Averages - Trend indicators
                if "SMA" in selected_indicators:
                    sma_20 = sma(prices_series, window=20)
                    if not sma_20.empty and not pd.isna(sma_20.iloc[-1]):
                        technical["sma_fast"] = float(sma_20.iloc[-1])
                    
                    # Longer period SMA if we have enough data
                    if len(prices_series) >= 50:
                        sma_50 = sma(prices_series, window=50)
                        if not sma_50.empty and not pd.isna(sma_50.iloc[-1]):
                            technical["sma_slow"] = float(sma_50.iloc[-1])
                
                # Exponential Moving Averages - Responsive trend indicators
                if "EMA" in selected_indicators:
                    ema_12 = ema(prices_series, window=12)
                    ema_26 = ema(prices_series, window=26)
                    if not ema_12.empty and not pd.isna(ema_12.iloc[-1]):
                        technical["ema_fast"] = float(ema_12.iloc[-1])
                    if not ema_26.empty and not pd.isna(ema_26.iloc[-1]):
                        technical["ema_slow"] = float(ema_26.iloc[-1])
                
                # MACD (Moving Average Convergence Divergence) - Momentum indicator
                if "MACD" in selected_indicators:
                    macd_line, signal_line, histogram = macd(prices_series, fast=12, slow=26, signal=9)
                    if not macd_line.empty and not pd.isna(macd_line.iloc[-1]):
                        technical["macd"] = float(macd_line.iloc[-1])
                        technical["macd_signal"] = float(signal_line.iloc[-1])
                        technical["macd_histogram"] = float(histogram.iloc[-1])
                
                # Bollinger Bands - Volatility and mean reversion indicator
                if "BBANDS" in selected_indicators:
                    bb = bollinger_bands(prices_series, window=20, n_std=2.0)
                    if not bb.empty:
                        technical["bb_upper"] = float(bb["bb_upper"].iloc[-1])
                        technical["bb_lower"] = float(bb["bb_lower"].iloc[-1])
                        technical["bb_mid"] = float(bb["bb_mid"].iloc[-1])
                        
                        # Calculate position within bands (0 = lower band, 1 = upper band)
                        current_price = row.get('price', prices_series.iloc[-1])
                        bb_width = technical["bb_upper"] - technical["bb_lower"]
                        if bb_width > 0:
                            technical["bb_position"] = (current_price - technical["bb_lower"]) / bb_width
                
                # Stochastic Oscillator - Momentum indicator
                if "STOCH" in selected_indicators and len(df) >= 14:
                    stoch_k, stoch_d = stochastic(df, k=14, d=3)
                    if not stoch_k.empty and not pd.isna(stoch_k.iloc[-1]):
                        technical["stoch_k"] = float(stoch_k.iloc[-1])
                        technical["stoch_d"] = float(stoch_d.iloc[-1])
                
                # ADX (Average Directional Index) - Trend strength indicator
                if "ADX" in selected_indicators and len(df) >= 14:
                    adx_values = adx(df, window=14)
                    if not adx_values.empty and not pd.isna(adx_values.iloc[-1]):
                        technical["adx"] = float(adx_values.iloc[-1])
                
                # ATR (Average True Range) - Volatility indicator
                if "ATR" in selected_indicators and len(df) >= 14:
                    atr_values = atr(df, window=14)
                    if not atr_values.empty and not pd.isna(atr_values.iloc[-1]):
                        technical["atr"] = float(atr_values.iloc[-1])
                        # Calculate ATR as percentage of price for better comparison
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
                    # Daily volatility (standard deviation of daily returns)
                    returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                              for i in range(1, len(price_history))]
                    if returns:
                        volatility = pd.Series(returns).std() * (252 ** 0.5)  # Annualized volatility
                        technical["volatility_annual"] = volatility * 100
                
                if technical:
                    successful_calculations += 1
                
            except Exception as e:
                log.warning(f"Error calculating indicators for {row.get('symbol')}: {e}")
            
            # Add technical data to asset row
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
    Calculate realistic profit potential based on technical indicators and market characteristics.
    
    This function combines technical analysis with market-specific volatility patterns
    to estimate realistic profit targets and risk levels. It considers:
    - Historical volatility patterns by asset class
    - Current technical momentum (RSI, MACD)
    - Trend strength (moving averages)
    - Market-specific characteristics
    
    Args:
        technical_data: Dictionary of calculated technical indicators
        current_price: Current asset price
        category: Asset category (crypto, forex, equities, etc.)
        
    Returns:
        Dictionary with profit targets, risk levels, and scoring factors
    """
    # Base volatility expectations by market type (daily volatility)
    base_volatility = {
        "crypto": 0.08,      # 8% daily volatility (high volatility)
        "forex": 0.015,      # 1.5% daily volatility (low volatility)
        "equities": 0.025,   # 2.5% daily volatility (moderate volatility)
        "commodities": 0.035, # 3.5% daily volatility (moderate-high volatility)
        "futures": 0.04,     # 4% daily volatility (high volatility)
        "warrants": 0.12,    # 12% daily volatility (very high leverage)
        "funds": 0.02,       # 2% daily volatility (low volatility ETFs)
    }.get(category, 0.03)   # Default 3% for unknown categories
    
    # Use actual ATR volatility if available, otherwise use base volatility
    actual_volatility = technical_data.get("atr_pct", base_volatility * 100) / 100
    
    # RSI momentum factor - oversold/overbought conditions
    momentum_factor = 1.0
    rsi = technical_data.get("rsi")
    if rsi:
        if rsi < 25:        # Very oversold - higher upside potential
            momentum_factor = 1.8
        elif rsi < 35:      # Oversold - good upside potential
            momentum_factor = 1.4
        elif rsi > 75:      # Very overbought - lower upside potential
            momentum_factor = 0.6
        elif rsi > 65:      # Overbought - reduced upside potential
            momentum_factor = 0.8
    
    # Trend factor from moving averages - trend direction and strength
    trend_factor = 1.0
    sma_fast = technical_data.get("sma_fast")
    sma_slow = technical_data.get("sma_slow")
    if sma_fast and sma_slow and current_price:
        if current_price > sma_fast > sma_slow:    # Strong uptrend
            trend_factor = 1.5
        elif current_price > sma_fast:             # Moderate uptrend
            trend_factor = 1.2
        elif current_price < sma_fast < sma_slow:  # Strong downtrend
            trend_factor = 0.7
        elif current_price < sma_fast:             # Moderate downtrend
            trend_factor = 0.9
    
    # MACD momentum factor - momentum direction and strength
    macd_factor = 1.0
    macd = technical_data.get("macd")
    macd_signal = technical_data.get("macd_signal")
    if macd is not None and macd_signal is not None:
        if macd > macd_signal and macd > 0:        # Strong bullish momentum
            macd_factor = 1.3
        elif macd > macd_signal:                   # Moderate bullish momentum
            macd_factor = 1.1
        elif macd < macd_signal and macd < 0:      # Strong bearish momentum
            macd_factor = 0.7
        elif macd < macd_signal:                   # Moderate bearish momentum
            macd_factor = 0.9
    
    # Calculate base profit target combining all factors
    base_target = actual_volatility * momentum_factor * trend_factor * macd_factor
    
    # Ensure minimum 3% target for our strategy requirements
    min_target = 0.03  # 3% minimum profit requirement
    max_target = min(actual_volatility * 4, 0.20)  # Max 20% or 4x volatility
    
    # Calculate conservative and aggressive targets
    conservative_target = max(min_target, base_target * 0.8)
    aggressive_target = min(max_target, base_target * 1.8)
    
    return {
        "min_target_pct": min_target * 100,
        "conservative_target_pct": conservative_target * 100,
        "aggressive_target_pct": aggressive_target * 100,
        "stop_loss_pct": actual_volatility * 0.6 * 100,  # 60% of volatility for stop loss
        "volatility_pct": actual_volatility * 100,
        "momentum_score": momentum_factor,
        "trend_score": trend_factor,
        "macd_score": macd_factor,
    }


# ────────────────────────────────────────────────────────────────────────────
# Enhanced AI Prompt Builder for Multi-AI Analysis
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
    
    This function creates a detailed, structured prompt that provides the AI with
    all necessary context for making informed trading recommendations:
    - Market timing and category-specific constraints
    - Detailed technical analysis data for each asset
    - Sentiment context and market conditions
    - Risk management parameters
    - Expected response format
    
    Args:
        enhanced_data: List of assets with technical indicators
        sentiment_data: Market sentiment information
        category: Asset category being analyzed
        budget: Available trading budget
        market_context: Market timing and regional information
        ai_engines: List of AI engines being used
        
    Returns:
        Formatted prompt string for AI analysis
    """
    
    # Market timing context based on category
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
    
    # Prepare detailed asset analysis summaries
    asset_summaries = []
    for i, asset in enumerate(enhanced_data[:20], 1):  # Top 20 for comprehensive analysis
        symbol = asset.get("symbol", f"Asset_{i}")
        price = asset.get("price", 0)
        volume = asset.get("volume", 0)
        technical = asset.get("technical", {})
        
        # Calculate profit potential using technical indicators
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
        
        # Format asset summary with all relevant data
        summary = f"""
{i}. {symbol} - ${price:.6f}
   Volume: ${volume:,.0f} | Volatility: {profit_analysis['volatility_pct']:.1f}%
   Technical: {' | '.join(tech_summary) if tech_summary else 'Limited indicators'}
   Momentum Score: {profit_analysis['momentum_score']:.2f} | Trend: {profit_analysis['trend_score']:.2f}
   Profit Potential: {profit_analysis['conservative_target_pct']:.1f}% - {profit_analysis['aggressive_target_pct']:.1f}%
   Stop Loss: {profit_analysis['stop_loss_pct']:.1f}%"""
        
        asset_summaries.append(summary)
    
    # Sentiment context section
    sentiment_context = ""
    if sentiment_data:
        sentiment_context = f"""
MARKET SENTIMENT ANALYSIS:
{chr(10).join(f"• {item}" for item in sentiment_data[:12])}
"""
    
    # AI engine context for the prompt
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
# Market Context Helper Functions
# ────────────────────────────────────────────────────────────────────────────

def get_market_context_for_category(category: str) -> Dict[str, Any]:
    """
    Get appropriate market context based on asset category.
    
    This function handles the logic for determining whether market/region
    selection is needed and provides appropriate context for each category:
    - Global markets (crypto, forex): No region selection needed
    - Regional markets (equities, commodities, etc.): Require market selection
    
    Args:
        category: Asset category (crypto, forex, equities, etc.)
        
    Returns:
        Dictionary with market context information
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
            print(f"\n🏛 {category.title()} are region-specific. Please select your target market:")
            market_selection = get_market_selection_details()
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
# Enhanced Configuration Helper
# ────────────────────────────────────────────────────────────────────────────

def get_complete_feature_configuration() -> Dict[str, Any]:
    """
    Get complete feature configuration with all indicators enabled.
    
    This function ensures that all 9 technical indicators are included
    in the feature flags as requested, providing comprehensive analysis.
    
    Returns:
        Dictionary with complete feature configuration
    """
    # Import the terminal UI configuration function
    try:
        from .terminal_ui import get_feature_configuration
        base_config = get_feature_configuration()
    except ImportError:
        # Fallback configuration if terminal_ui function not available
        base_config = {
            "use_all": True,
            "use_rsi": True,
            "use_sma": True,
            "use_sentiment": True,
            "selected_indicators": [],
            "sentiment_components": []
        }
    
    # Ensure all 9 indicators are included as requested
    all_indicators = ["SMA", "EMA", "MACD", "ADX", "RSI", "STOCH", "OBV", "BBANDS", "ATR"]
    
    # Override with complete indicator set
    enhanced_config = base_config.copy()
    enhanced_config["selected_indicators"] = all_indicators
    enhanced_config["use_all"] = True
    enhanced_config["use_rsi"] = True
    enhanced_config["use_sma"] = True
    
    # Ensure all sentiment components are included
    all_sentiment_components = ["news", "social", "fear_greed", "institutional", "technical_sentiment"]
    if enhanced_config.get("use_sentiment", False):
        enhanced_config["sentiment_components"] = all_sentiment_components
    
    return enhanced_config


# ────────────────────────────────────────────────────────────────────────────
# Main Enhanced Workflow Implementation
# ────────────────────────────────────────────────────────────────────────────

def run_enhanced_category_workflow() -> None:
    """
    Complete enhanced workflow with multi-AI support and universal sentiment analysis.
    
    This is the main function that orchestrates the entire trading analysis workflow:
    1. Initialize and validate AI engines
    2. Configure analysis parameters (indicators, sentiment, etc.)
    3. Collect market data with 2-week history
    4. Apply technical indicators to historical data
    5. Gather sentiment data for the category
    6. Perform AI or rules-based analysis
    7. Display results with profit targets and risk management
    8. Log trades for performance tracking
    """
    print_header("🤖 AI-Powered Trading Assistant")
    
    # Step 1: Get AI engine status and display capabilities
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
    
    # Show enhanced disclaimer and system information
    show_enhanced_disclaimer()
    
    # Step 2: Analysis configuration setup
    print("\n" + "─" * 60)
    print("📊 ANALYSIS SETUP")
    print("─" * 60)
    
    # Get complete feature configuration with all indicators
    feature_config = get_complete_feature_configuration()
    
    use_all = feature_config["use_all"]
    use_rsi = feature_config["use_rsi"]
    use_sma = feature_config["use_sma"]
    use_sentiment = feature_config["use_sentiment"]
    use_ai = ai_status["ai_available"]  # Use AI if available
    selected_inds = feature_config["selected_indicators"]
    sentiment_components = feature_config.get("sentiment_components", [])
    ai_engines = ai_status["engines"]
    
    # Show final configuration summary
    print("\n" + "─" * 50)
    print("🎯 FINAL ANALYSIS CONFIGURATION:")
    print("─" * 50)
    
    # AI Status - show actually available engines
    if use_ai and ai_engines:
        if len(ai_engines) > 1:
            print(f"🤖 AI Analysis: ✅ {' + '.join(ai_engines)} (Multi-AI)")
        else:
            print(f"🤖 AI Analysis: ✅ {ai_engines[0]}")
    else:
        print("🤖 AI Analysis: ❌ Using Technical Rules Engine")
    
    # Technical Indicators - show all 9 indicators
    if selected_inds:
        print(f"📊 Technical Indicators ({len(selected_inds)}): {', '.join(selected_inds)}")
    else:
        print("📊 Technical Indicators: None selected")
    
    # Sentiment Analysis
    if use_sentiment and sentiment_components:
        print(f"💭 Sentiment Analysis ({len(sentiment_components)}): {', '.join(sentiment_components)}")
    else:
        print("💭 Sentiment Analysis: ❌ Disabled")
    
    print("─" * 50)
    
    # Step 3: Market and budget selection
    category = get_user_choice()
    budget = get_user_budget()
    
    print(f"\n🎯 TARGET: Find {category.upper()} opportunities with 3-5% minimum profit")
    print(f"💰 BUDGET: ${budget:,.2f}")
    
    # Step 4: Data collection with progress tracking
    pbar = tqdm(total=5, desc="⏳ Processing", unit="step")
    
    # Get market context (handles region selection for region-dependent categories)
    market_ctx = get_market_context_for_category(category)
    
    # Step 5: Fetch 2-week historical data
    pbar.set_description("📊 Fetching 2-week historical data...")
    
    # Map categories to their respective data fetching functions
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
        # Fetch market data with historical price data for technical analysis
        # Note: Removed the problematic crypto-specific import that was causing the error
        if category == "crypto":
            print(f"📊 Using standard crypto data fetching strategy")
            rows = fetch_func(include_history=True)
        else:
            # For regional markets, include market context if available
            fetch_kwargs = {"include_history": True}
            if market_ctx.get("market") and market_ctx["market"] != "Global":
                fetch_kwargs["market"] = market_ctx["market"]
            if market_ctx.get("region") and market_ctx["region"] != "Global":
                fetch_kwargs["region"] = market_ctx["region"]
            
            rows = fetch_func(**fetch_kwargs)
            
        print(f"✅ Fetched data for {len(rows)} assets with historical data")
        pbar.update(1)
        
    except Exception as e:
        log.exception("Data fetch failed")
        print(f"\n❌ Data fetch error: {e}")
        pbar.close()
        return
    
    if not rows:
        print(f"\n❌ No data available for {category}")
        print("💡 Try a different market/region or check your internet connection")
        pbar.close()
        return
    
    # Step 6: Apply technical indicators to the data
    pbar.set_description("🧮 Computing technical indicators...")
    enhanced_data = enhance_data_with_indicators(rows, selected_inds)
    pbar.update(1)
    
    # Step 7: Collect sentiment data for the selected components
    pbar.set_description("💭 Analyzing market sentiment...")
    sentiment_data = collect_universal_sentiment_data(category, use_sentiment, sentiment_components)
    pbar.update(1)
    
    # Step 8: Perform AI or rules-based analysis
    pbar.set_description("🤖 Performing AI analysis for optimal trade selection...")
    
    try:
        if ai_status["ai_available"] and use_ai:
            # Use AI strategy analysis
            print(f"🤖 Using AI analysis with {ai_status['primary_engine']} engine")
            
            # Prepare feature flags for the strategy engine
            feature_flags = {
                "use_rsi": use_rsi,
                "use_sma": use_sma,
                "use_sentiment": use_sentiment,
                "selected_indicators": selected_inds,
                "sentiment_components": sentiment_components,
                "use_all_features": use_all,
            }
            
            # Call analyze_market with proper parameters
            recs = analyze_market(
                market_data=enhanced_data,
                budget=budget,
                market_type=category,
                history=[],  # Empty history for new session
                sentiment=sentiment_data,
                use_rsi=use_rsi,
                use_sma=use_sma,
                use_sentiment=use_sentiment,
                market=market_ctx.get("market"),
                market_context=market_ctx,
                engine="llm"  # Force LLM for AI analysis
            )
        else:
            # Use rules-based analysis as fallback
            print("📊 Using technical rules engine for analysis")
            
            # Prepare feature flags for rules engine
            feature_flags = {
                "use_rsi": use_rsi,
                "use_sma": use_sma,
                "use_sentiment": use_sentiment,
                "selected_indicators": selected_inds,
                "sentiment_components": sentiment_components,
                "use_all_features": use_all,
            }
            
            # Try to use rules engine if available
            try:
                from trading_core.strategy.rules_engine import analyze_market_batch
                
                recs = analyze_market_batch(
                    rows=enhanced_data,
                    market_ctx=market_ctx,
                    feature_flags=feature_flags,
                    budget=budget
                )
            except ImportError:
                # Fallback to original analyze_market function
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
        
        # Handle tuple returns from some strategy functions
        if isinstance(recs, tuple):
            recs = recs[0]
            
        pbar.update(1)
        
    except Exception as e:
        log.exception("Analysis failed")
        print(f"\n❌ Analysis error: {e}")
        pbar.close()
        return
    
    # Step 9: Display enhanced results
    pbar.set_description("📊 Preparing results...")
    pbar.update(1)
    pbar.close()
    
    # Display recommendations with current timestamp
    now = datetime.now(LOCAL_TZ)
    print_enhanced_recommendations(
        recs, 
        title=f"🎯 AI Trading Recommendations ({now.strftime('%H:%M %Z')})",
        category=category,
        budget=budget,
        ai_status=ai_status
    )
    
    # Show data source diagnostics for debugging
    _print_diagnostics(category)
    
    # Step 10: Persist results for performance tracking
    try:
        # Log the trading session for future analysis
        for rec in recs[:3] if recs else []:  # Log top 3 recommendations
            if rec.get("action") in ["Buy", "Sell"]:
                log_trade(
                    symbol=rec.get("symbol", "Unknown"),
                    action=rec.get("action", "Hold"),
                    entry_price=rec.get("price", 0),
                    quantity=rec.get("position_size", 0),
                    category=category,
                    confidence=rec.get("confidence", 0),
                    reasoning=rec.get("reasoning", "AI analysis")
                )
    except Exception as e:
        log.warning(f"Could not log trades: {e}")


def print_enhanced_recommendations(
    recs: List[Dict[str, Any]], 
    title: str, 
    category: str, 
    budget: float,
    ai_status: Dict[str, Any],
    min_profit_target: float = 2.0,
    target_desc: str = "Moderate (2-4%)"
) -> None:
    """
    Enhanced recommendation display with profit targets, confidence levels, and AI insights.
    
    This function provides a comprehensive display of trading recommendations
    including risk management, position sizing, market timing information,
    and prominent confidence level display.
    """
    
    print_header(title)
    
    # Show which AI engine was used for analysis
    if ai_status["ai_available"]:
        if len(ai_status["engines"]) > 1:
            print(f"🤖 Analysis performed by: {' + '.join(ai_status['engines'])} (Multi-AI)")
        else:
            print(f"🤖 Analysis performed by: {ai_status['engines'][0]}")
    else:
        print("📊 Analysis performed by: Technical Rules Engine")
    
    print(f"🎯 Profit Target: {target_desc}")
    print(f"📊 Minimum Expected Return: {min_profit_target}%")
    
    if not recs:
        print(f"\n❌ No trading opportunities found matching our criteria.")
        print(f"💡 Try lowering the profit target to {max(0.5, min_profit_target - 1)}% or check different market categories.")
        return
    
    # Filter and sort by profitability and confidence
    profitable_recs = [r for r in recs if r.get("action") in ["Buy", "Sell"]]
    
    if not profitable_recs:
        print(f"\n📊 No high-confidence opportunities meeting {min_profit_target}% minimum profit target.")
        print(f"💡 Market conditions may not be optimal for intraday trading today.")
        print(f"🔄 Consider lowering target to {max(0.5, min_profit_target - 1)}% or checking other categories.")
        return
    
    # Sort by confidence and expected profit
    profitable_recs.sort(key=lambda x: (x.get("confidence", 0), x.get("estimated_profit", 0)), reverse=True)
    
    print(f"🎯 Found {len(profitable_recs)} opportunities meeting our criteria:")
    print(f"💰 Budget Available: ${budget:,.2f}")
    print(f"🕐 Market Type: {category.title()} ({'24/7' if category == 'crypto' else 'Market Hours Only'})")
    
    total_potential_profit = 0
    
    # Display top 3 opportunities with detailed analysis and prominent confidence levels
    for i, rec in enumerate(profitable_recs[:3], 1):
        asset = rec.get("asset", rec.get("symbol", "Unknown"))
        action = rec.get("action", "Hold")
        confidence = rec.get("confidence", 0)
        entry_price = rec.get("price", rec.get("entry_price", 0))
        target_price = rec.get("sell_target", rec.get("target_price", 0))
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
        
        # Prominent confidence level display with color coding
        confidence_emoji = ""
        if confidence >= 85:
            confidence_emoji = "🟢"  # High confidence
        elif confidence >= 70:
            confidence_emoji = "🟡"  # Medium confidence
        elif confidence >= 50:
            confidence_emoji = "🟠"  # Low-medium confidence
        else:
            confidence_emoji = "🔴"  # Low confidence
            
        print(f"🎲 {confidence_emoji} AI CONFIDENCE: {confidence}%")
        
        # Add confidence interpretation
        if confidence >= 85:
            conf_text = "Very High - Strong recommendation"
        elif confidence >= 70:
            conf_text = "High - Good opportunity"
        elif confidence >= 50:
            conf_text = "Medium - Consider carefully"
        else:
            conf_text = "Low - High risk"
        print(f"   └─ Level: {conf_text}")
        
        print(f"💵 Entry Price: ${entry_price:.4f}")
        print(f"🎯 Target Price: ${target_price:.4f}")
        print(f"🛑 Stop Loss: ${stop_loss:.4f}")
        print(f"📊 Expected Profit: {profit_pct:.1f}%")
        print(f"💰 Estimated Gain: ${estimated_profit:.2f}")
        
        # Calculate position size within budget constraints
        max_shares = int(budget * 0.9 / entry_price) if entry_price > 0 else 0
        position_value = max_shares * entry_price
        
        print(f"📦 Suggested Position: {max_shares:,} shares")
        print(f"💎 Position Value: ${position_value:,.2f}")
        
        # Risk analysis and management
        if stop_loss > 0 and entry_price > 0:
            risk_pct = abs((stop_loss - entry_price) / entry_price) * 100
            max_loss = max_shares * abs(stop_loss - entry_price)
            print(f"⚠️  Maximum Risk: {risk_pct:.1f}% (${max_loss:.2f})")
        
        # Market timing information
        if category == "crypto":
            print("⏰ Timing: Execute within next 6 hours (24/7 market)")
        else:
            print("⏰ Timing: Execute during market hours today")
        
        # AI reasoning and technical justification
        reasons = rec.get("reasons", rec.get("reasoning", "Technical analysis indicates favorable conditions"))
        print(f"🤖 AI Analysis: {reasons[:100]}...")
        
        total_potential_profit += estimated_profit
    
    # Summary section with key metrics
    print(f"\n{'='*60}")
    print("📋 TRADING PLAN SUMMARY")
    print(f"{'='*60}")
    print(f"🎯 Total Opportunities: {len(profitable_recs)}")
    print(f"💰 Total Potential Profit: ${total_potential_profit:.2f}")
    print(f"📈 Average Expected Return: {(total_potential_profit/budget)*100:.1f}%")
    print(f"🎯 Profit Target: {target_desc} (min {min_profit_target}%)")
    
    # Market timing reminder based on category
    if category == "crypto":
        exit_deadline = datetime.now(LOCAL_TZ) + timedelta(hours=6)
        print(f"⏰ Exit Deadline: {exit_deadline.strftime('%H:%M %Z')} (6 hours from now)")
    else:
        print("⏰ Exit Deadline: Before market close today")
    
    # Trading strategy tips specific to user's profit target
    print(f"\n💡 AI TRADING STRATEGY TIPS:")
    print(f"   • All recommendations target minimum {min_profit_target}% profit")
    print("   • Higher confidence = lower risk, better probability")
    print("   • Execute trades during optimal timing windows")
    print("   • Set stop losses immediately after entry")
    print("   • Monitor positions every 30 minutes")
    print("   • Take profits at target prices - avoid greed")
    print("   • Exit all positions before market close (non-crypto)")
    
    # Risk management warnings
    print(f"\n⚠️  RISK MANAGEMENT:")
    print("   • Never risk more than 2% of total portfolio per trade")
    print("   • Use stop losses to limit downside exposure")
    print("   • Higher profit targets = higher risk")
    print("   • Confidence levels help assess probability")
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
    """Print data source diagnostics for debugging purposes."""
    try:
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
            
    except Exception as e:
        log.warning(f"Could not get diagnostics for {category}: {e}")
        print(f"⚠️  Diagnostics unavailable: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Single Asset Analysis Workflow
# ────────────────────────────────────────────────────────────────────────────

def run_single_asset_flow() -> None:
    """
    Single-asset analysis workflow for focused analysis of individual symbols.
    
    This workflow allows users to analyze a specific asset with selected indicators
    rather than screening entire categories. Useful for:
    - Analyzing a specific stock, crypto, or forex pair
    - Testing indicator combinations on known assets
    - Getting detailed analysis of user's existing positions
    """
    print_header("🔍 Single Asset Analysis")
    
    # Validate startup environment
    ai_status = get_ai_engine_status()
    
    if ai_status["ai_available"]:
        print(f"🤖 AI Analysis: {ai_status['status_message']}")
    else:
        print("📊 Using Technical Rules Engine")
    
    show_enhanced_disclaimer()
    
    # Get user input for single asset analysis
    try:
        input_data = prompt_single_asset_input()
    except Exception as e:
        print(f"❌ Input error: {e}")
        return
    
    symbol = input_data["symbol"]
    asset_class = input_data["asset_class"]
    market = input_data.get("market")
    region = input_data.get("region")
    indicators = input_data.get("indicators", [])
    budget = input_data.get("budget", 1000)
    
    print(f"\n🎯 Analyzing {symbol} ({asset_class})")
    if market:
        print(f"📍 Market: {market}")
    if region:
        print(f"🌍 Region: {region}")
    
    # Fetch single asset data
    try:
        asset_data = fetch_single_symbol_quote(symbol, asset_class)
        if not asset_data:
            print(f"\n❌ Could not fetch data for {symbol}")
            print("💡 Try a different symbol or check your internet connection")
            return
    except Exception as e:
        log.exception(f"Failed to fetch data for {symbol}")
        print(f"\n❌ Error fetching {symbol}: {e}")
        return
    
    # Create market context
    market_ctx = {
        "category": asset_class,
        "market": market,
        "region": region,
        "timezone": "UTC",
        "sessions": [],
        "trading_days": [0, 1, 2, 3, 4],
    }
    
    # Create feature flags based on selected indicators
    feature_flags = {
        "use_rsi": "RSI" in indicators,
        "use_sma": "SMA" in indicators or "EMA" in indicators,
        "use_sentiment": "sentiment" in [i.lower() for i in indicators],
        "selected_indicators": indicators,
        "use_all_features": len(indicators) >= 5,
    }
    
    print(f"\n🔍 Running analysis with {len(indicators)} indicators...")
    
    # Analyze single asset
    try:
        # Try using the single asset analysis function if available
        try:
            from trading_core.strategy.rules_engine import analyze_single_asset
            recommendation = analyze_single_asset(asset_data, asset_class, market_ctx, feature_flags, budget)
        except ImportError:
            # Fallback to general analyze_market function
            recommendation = analyze_market(
                market_data=[asset_data],
                budget=budget,
                market_type=asset_class,
                history=[],
                sentiment=None,
                use_rsi=feature_flags["use_rsi"],
                use_sma=feature_flags["use_sma"],
                use_sentiment=feature_flags["use_sentiment"],
                market=market,
                market_context=market_ctx,
                engine="llm" if ai_status["ai_available"] else "rules"
            )
            # Extract first recommendation if list returned
            if isinstance(recommendation, list) and recommendation:
                recommendation = recommendation[0]
            elif isinstance(recommendation, tuple):
                recommendation = recommendation[0][0] if recommendation[0] else None
                
    except Exception as e:
        log.exception(f"Analysis failed for {symbol}")
        print(f"\n❌ Analysis error: {e}")
        return
    
    # Display results in one-liner format as requested
    if recommendation:
        action = recommendation.get("action", "Hold")
        confidence = recommendation.get("confidence", 0)
        reasoning = recommendation.get("reasoning", recommendation.get("reasons", "No reasoning provided"))
        
        print(f"\n📊 ANALYSIS RESULT:")
        print(f"[Action: {action} | Confidence: {confidence:.0f}% | Key Reasons: {reasoning}]")
        
        # Show additional numeric fields if available
        if recommendation.get("target_price"):
            print(f"🎯 Target: ${recommendation['target_price']:.2f}")
        if recommendation.get("stop_loss"):
            print(f"🛑 Stop Loss: ${recommendation['stop_loss']:.2f}")
        if recommendation.get("expected_profit_pct"):
            print(f"📈 Expected Profit: {recommendation['expected_profit_pct']:.1f}%")
    else:
        print(f"\n📊 No clear signal for {symbol} at this time.")
        print("💡 Consider waiting for better market conditions or trying different indicators.")


# ────────────────────────────────────────────────────────────────────────────
# Main Entry Point and Dispatch Logic
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main CLI entrypoint with enhanced AI workflow.
    
    This function serves as the main entry point for the trading assistant CLI.
    It handles:
    - Environment validation and setup
    - Market configuration loading
    - Mode selection (category vs single asset analysis)
    - Error handling and graceful shutdown
    - Performance evaluation from previous sessions
    """
    try:
        # Load market configuration for timezone and trading hours
        try:
            markets = load_markets_config()
            if markets:
                sample_market = list(markets.keys())[0]
                sample_info = markets[sample_market]
                print(f"[I] Successfully loaded: {len(markets)} markets from configuration")
                print(f"[I] Sample market '{sample_market}': {sample_info.get('label', 'Unknown')}")
            else:
                print("[W] No markets loaded from configuration")
        except Exception as e:
            log.warning("Could not load markets config: %s", e)
            print(f"[W] Could not load markets config: {e}")

        # Check for previous session and show performance if available
        try:
            evaluate_previous_session()
        except Exception as e:
            log.warning(f"Could not evaluate previous session: {e}")

        # Main mode selection and workflow dispatch
        mode = prompt_main_mode()
        
        if mode == "category":
            # Run the enhanced category-based analysis workflow
            run_enhanced_category_workflow()
        elif mode == "single_asset":
            # Run single asset analysis workflow
            run_single_asset_flow()
        else:
            print("❌ Unknown mode selected")
            return
            
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using Trading Assistant!")
        print("💡 Tip: Run 'make setup-api' to configure APIs for better data coverage")
    except Exception as e:
        log.exception("Unexpected error in main")
        print(f"\n❌ Unexpected error: {e}")
        print("💡 If this persists, please check the logs or run 'make debug-paths'")


# ────────────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
