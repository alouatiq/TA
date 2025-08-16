# BE/trading_core/strategy/rules_engine.py
"""
Deterministic rules engine: multi-signal → recommendation.

This module provides both the internal recommendation functions and the 
API functions expected by main.py:
- analyze_market_batch: For category flow (multiple assets)
- analyze_single_asset: For single asset analysis

The internal functions are:
- recommend_for_assets: Core logic for multiple assets
- recommend_one: Core logic for single asset

Design:
• Regime detection → adjust weights across buckets.
• Each bucket returns (score: 0..100, top_notes: list of strings).
• Weighted blend → confidence 0..100; thresholds map to Buy/Sell/Hold.
• Risk: propose stop/target using ATR% if available, else sensible defaults.
• Graceful degradation: missing signals simply reduce that bucket's weight.

No hard-coded tickers, regions, or markets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

Number = float
Row = Dict[str, Any]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _safe_get(d: Optional[dict], key: str, default=None):
    """Safely get a value from a dict that might be None."""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _nz(v, default=0.0) -> float:
    """Convert to float, handling None and NaN safely."""
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):  # NaN-safe
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp01(x: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, float(x)))


def _pct(x: float) -> int:
    """Convert 0-1 score to 0-100 percentage."""
    return int(round(100.0 * _clamp01(x)))


def _sign(x: float) -> int:
    """Return sign of number: 1, 0, or -1."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


# ────────────────────────────────────────────────────────────────────────────
# 1) Regime detection (simple but effective)
# ────────────────────────────────────────────────────────────────────────────

def _detect_regime(asset: Row, hint: Optional[str] = None) -> str:
    """
    Quick regime classification: bull/bear/range/volatile.
    Uses price history, volume, and basic momentum if available.
    """
    if hint and hint in ["bull", "bear", "range", "volatile"]:
        return hint

    # Try to determine from available data
    price_history = asset.get("price_history", [])
    if len(price_history) < 5:
        return "range"  # Default when insufficient data

    # Simple trend detection: compare recent vs older prices
    recent = sum(price_history[-3:]) / 3 if len(price_history) >= 3 else price_history[-1]
    older = sum(price_history[:3]) / 3 if len(price_history) >= 3 else price_history[0]
    
    trend = (recent - older) / older if older > 0 else 0
    
    # Volatility check: coefficient of variation
    if len(price_history) >= 5:
        avg_price = sum(price_history) / len(price_history)
        variance = sum((p - avg_price) ** 2 for p in price_history) / len(price_history)
        vol = (variance ** 0.5) / avg_price if avg_price > 0 else 0
        
        if vol > 0.05:  # 5% daily volatility threshold
            return "volatile"

    # Trend classification
    if trend > 0.02:  # 2% uptrend
        return "bull"
    elif trend < -0.02:  # 2% downtrend
        return "bear"
    else:
        return "range"


# ────────────────────────────────────────────────────────────────────────────
# 2) Scoring functions (each returns score: 0..1, notes: list)
# ────────────────────────────────────────────────────────────────────────────

def _score_technical(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes) for technical analysis."""
    notes: List[str] = []
    t = asset.get("technical") or {}
    if not t:
        # Try to calculate basic indicators from price history
        price_history = asset.get("price_history", [])
        if len(price_history) < 5:
            return 0.5, ["Insufficient price data"]
        
        # Basic momentum analysis
        recent_prices = price_history[-5:]
        older_prices = price_history[:5] if len(price_history) >= 10 else price_history[:-5]
        
        if older_prices:
            recent_avg = sum(recent_prices) / len(recent_prices)
            older_avg = sum(older_prices) / len(older_prices)
            momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
            
            if momentum > 0.01:  # 1% positive momentum
                score = 0.65
                notes.append("Positive momentum")
            elif momentum < -0.01:  # 1% negative momentum
                score = 0.35
                notes.append("Negative momentum")
            else:
                score = 0.5
                notes.append("Neutral momentum")
        else:
            score = 0.5
            notes.append("Insufficient data for momentum")
        
        return _clamp01(score), notes

    price = _nz(asset.get("price"))
    sma_f = _nz(_safe_get(t, "sma_fast"))
    sma_s = _nz(_safe_get(t, "sma_slow"))
    ema_f = _nz(_safe_get(t, "ema_fast"))
    ema_s = _nz(_safe_get(t, "ema_slow"))
    macd = _nz(_safe_get(t, "macd"))
    macdsig = _nz(_safe_get(t, "macd_signal"))
    adx = _nz(_safe_get(t, "adx"))
    rsi = _nz(_safe_get(t, "rsi"))
    k = _nz(_safe_get(t, "stoch_k"))
    d = _nz(_safe_get(t, "stoch_d"))
    obv_sl = _nz(_safe_get(t, "obv_slope"))
    bb_pos = _nz(_safe_get(t, "bb_pos"))
    atr_pct = _nz(_safe_get(t, "atr_pct"))
    vol_spk = _nz(_safe_get(t, "volume_spike"))
    drp = _nz(asset.get("day_range_pct"))

    score = 0.0
    weight_sum = 0.0

    # Trend (SMA/EMA alignment) - More sensitive
    if sma_f and sma_s and price:
        w = 0.20  # Increased weight
        if price > sma_f > sma_s:
            trend_score = 0.75  # Strong uptrend
            notes.append("Strong uptrend")
        elif price > sma_f:
            trend_score = 0.65  # Moderate uptrend
            notes.append("Moderate uptrend")
        elif price < sma_f < sma_s:
            trend_score = 0.25  # Strong downtrend
            notes.append("Strong downtrend")
        elif price < sma_f:
            trend_score = 0.35  # Moderate downtrend
            notes.append("Moderate downtrend")
        else:
            trend_score = 0.5  # Neutral
            notes.append("Sideways trend")
        score += w * trend_score
        weight_sum += w

    # MACD - More sensitive
    if macd is not None and macdsig is not None:
        w = 0.15
        if macd > macdsig and macd > 0:
            macd_score = 0.75  # Strong bullish
            notes.append("MACD bullish")
        elif macd > macdsig:
            macd_score = 0.65  # Moderate bullish
            notes.append("MACD turning bullish")
        elif macd < macdsig and macd < 0:
            macd_score = 0.25  # Strong bearish
            notes.append("MACD bearish")
        elif macd < macdsig:
            macd_score = 0.35  # Moderate bearish
            notes.append("MACD turning bearish")
        else:
            macd_score = 0.5
            notes.append("MACD neutral")
        score += w * macd_score
        weight_sum += w

    # RSI (momentum) - More aggressive thresholds
    if rsi:
        w = 0.15  # Increased weight
        if rsi > 75:
            rscore = 0.20  # Very overbought
            notes.append("RSI very overbought")
        elif rsi > 65:
            rscore = 0.35  # Overbought
            notes.append("RSI overbought")
        elif rsi < 25:
            rscore = 0.80  # Very oversold (buy opportunity)
            notes.append("RSI very oversold")
        elif rsi < 35:
            rscore = 0.70  # Oversold
            notes.append("RSI oversold")
        elif 45 <= rsi <= 55:
            rscore = 0.55  # Neutral-bullish
            notes.append("RSI neutral")
        elif rsi > 55:
            rscore = 0.65  # Moderate bullish
            notes.append("RSI bullish")
        else:
            rscore = 0.45  # Moderate bearish
            notes.append("RSI bearish")
        score += w * rscore
        weight_sum += w

    # Stochastic Oscillator - Added
    if k and d:
        w = 0.10
        if k > 80 and d > 80:
            stoch_score = 0.25  # Overbought
            notes.append("Stoch overbought")
        elif k < 20 and d < 20:
            stoch_score = 0.75  # Oversold
            notes.append("Stoch oversold")
        elif k > d:
            stoch_score = 0.65  # Bullish crossover
            notes.append("Stoch bullish")
        else:
            stoch_score = 0.35  # Bearish crossover
            notes.append("Stoch bearish")
        score += w * stoch_score
        weight_sum += w

    # Volume analysis - More sensitive
    if vol_spk:
        w = 0.10  # Increased weight
        if vol_spk > 2.0:
            vscore = 0.75  # Very high volume confirms moves
            notes.append("Very high volume")
        elif vol_spk > 1.3:
            vscore = 0.65  # High volume
            notes.append("High volume")
        elif vol_spk < 0.7:
            vscore = 0.35  # Low volume = weak signals
            notes.append("Low volume")
        else:
            vscore = 0.55  # Normal volume
            notes.append("Normal volume")
        score += w * vscore
        weight_sum += w

    # Price momentum (day range) - More sensitive
    if drp:
        w = 0.10
        if drp > 3:
            pscore = 0.75  # Strong daily move up
            notes.append("Strong bullish move")
        elif drp > 1:
            pscore = 0.65  # Moderate move up
            notes.append("Moderate bullish move")
        elif drp < -3:
            pscore = 0.25  # Strong decline
            notes.append("Strong bearish move")
        elif drp < -1:
            pscore = 0.35  # Moderate decline
            notes.append("Moderate bearish move")
        else:
            pscore = 0.50
            notes.append("Consolidation")
        score += w * pscore
        weight_sum += w

    # ADX for trend strength
    if adx:
        w = 0.05
        if adx > 25:
            notes.append("Strong trend")
            # Don't change score, just confirm trend strength
        elif adx < 20:
            notes.append("Weak trend")
            # Reduce confidence slightly
            score *= 0.95
        weight_sum += w

    # Bollinger Bands position
    if bb_pos is not None:
        w = 0.05
        if bb_pos > 0.8:
            notes.append("Near upper BB")
        elif bb_pos < 0.2:
            notes.append("Near lower BB")
        weight_sum += w

    # Normalize score
    if weight_sum > 0:
        score = score / weight_sum
    else:
        score = 0.5  # Neutral if no indicators

    return _clamp01(score), notes[:4]  # Limit notes


def _score_fundamentals(asset: Row, category: str = "") -> Tuple[float, List[str]]:
    """Return (0..1, notes) for fundamental analysis."""
    notes: List[str] = []
    f = asset.get("fundamentals") or {}
    if not f:
        return 0.5, ["No fundamental data"]

    score = 0.5  # Start neutral
    
    # For crypto, fundamentals are different
    if category in ["crypto", "cryptocurrency"]:
        market_cap = _nz(_safe_get(f, "market_cap"))
        volume_24h = _nz(_safe_get(f, "volume_24h"))
        
        if market_cap > 10_000_000_000:  # $10B+ = established
            score += 0.1
            notes.append("Large market cap")
        elif market_cap < 1_000_000_000:  # <$1B = risky
            score -= 0.05
            notes.append("Small market cap")
            
        if volume_24h and market_cap:
            volume_ratio = volume_24h / market_cap
            if volume_ratio > 0.1:  # High turnover
                score += 0.05
                notes.append("High liquidity")
    
    # For stocks
    elif category in ["equities", "stocks"]:
        pe_ratio = _nz(_safe_get(f, "pe_ratio"))
        revenue_growth = _nz(_safe_get(f, "revenue_growth"))
        
        if pe_ratio and 10 <= pe_ratio <= 25:  # Reasonable valuation
            score += 0.1
            notes.append("Fair valuation")
        elif pe_ratio and pe_ratio > 50:  # Overvalued
            score -= 0.1
            notes.append("High valuation")
            
        if revenue_growth and revenue_growth > 0.15:  # 15%+ growth
            score += 0.15
            notes.append("Strong growth")
        elif revenue_growth and revenue_growth < -0.05:  # Declining
            score -= 0.1
            notes.append("Declining revenue")

    return _clamp01(score), notes[:2]


def _score_sentiment(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes) for sentiment analysis."""
    notes: List[str] = []
    s = asset.get("sentiment") or {}
    if not s:
        return 0.5, ["No sentiment data"]

    news_score = _nz(_safe_get(s, "news_score"))
    social_score = _nz(_safe_get(s, "social_score"))
    fear_greed = _nz(_safe_get(s, "fear_greed"))

    scores = []
    
    if news_score is not None:
        # Convert from -1..1 to 0..1
        news_norm = (news_score + 1) / 2
        scores.append(news_norm)
        if news_score > 0.3:
            notes.append("Very positive news")
        elif news_score > 0.1:
            notes.append("Positive news")
        elif news_score < -0.3:
            notes.append("Very negative news")
        elif news_score < -0.1:
            notes.append("Negative news")

    if social_score is not None:
        # Convert from -1..1 to 0..1
        social_norm = (social_score + 1) / 2
        scores.append(social_norm)
        if social_score > 0.2:
            notes.append("Positive social sentiment")
        elif social_score < -0.2:
            notes.append("Negative social sentiment")

    if fear_greed:
        # Assume fear_greed is 0..100, convert to 0..1
        fg_norm = fear_greed / 100.0
        scores.append(fg_norm)
        if fear_greed > 75:
            notes.append("Extreme greed (caution)")
        elif fear_greed > 60:
            notes.append("Market greed")
        elif fear_greed < 25:
            notes.append("Extreme fear (opportunity)")
        elif fear_greed < 40:
            notes.append("Market fear")

    if scores:
        final_score = sum(scores) / len(scores)
    else:
        final_score = 0.5

    return _clamp01(final_score), notes[:2]


def _score_microstructure(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes) for microstructure analysis."""
    notes: List[str] = []
    m = asset.get("microstructure") or {}
    if not m:
        return 0.5, ["No microstructure data"]

    order_imbalance = _nz(_safe_get(m, "order_imbalance"))
    liq_depth = _nz(_safe_get(m, "liq_depth_score"))
    whale_score = _nz(_safe_get(m, "whale_score"))

    score = 0.5  # Start neutral

    if order_imbalance:
        if order_imbalance > 0.3:
            score += 0.15
            notes.append("Strong buy pressure")
        elif order_imbalance > 0.1:
            score += 0.08
            notes.append("Buy pressure")
        elif order_imbalance < -0.3:
            score -= 0.15
            notes.append("Strong sell pressure")
        elif order_imbalance < -0.1:
            score -= 0.08
            notes.append("Sell pressure")

    if liq_depth and liq_depth > 0.7:
        score += 0.05
        notes.append("Good liquidity")
    elif liq_depth and liq_depth < 0.3:
        score -= 0.05
        notes.append("Poor liquidity")

    if whale_score and whale_score > 0.5:
        score += 0.1
        notes.append("Whale accumulation")
    elif whale_score and whale_score < -0.5:
        score -= 0.1
        notes.append("Whale distribution")

    return _clamp01(score), notes[:2]


def _score_risk_context(asset: Row, category: str = "") -> Tuple[float, List[str]]:
    """Return (0..1, notes) for risk/liquidity context."""
    notes: List[str] = []
    vol = _nz(asset.get("volume"))
    price = _nz(asset.get("price"))
    
    score = 0.5  # Start neutral

    # Volume/liquidity checks - category specific
    if category in ["crypto", "cryptocurrency"]:
        if vol >= 10_000_000:  # $10M+ daily volume
            score += 0.1
            notes.append("High liquidity")
        elif vol >= 1_000_000:  # $1M+ daily volume
            score += 0.05
            notes.append("OK liquidity")
        elif vol <= 100_000:  # <$100K daily volume
            score -= 0.15
            notes.append("Low liquidity risk")
    elif category in ["equities", "stocks"]:
        if vol >= 1_000_000:
            score += 0.1
            notes.append("High liquidity")
        elif vol >= 500_000:
            score += 0.05
            notes.append("OK liquidity")
        elif vol <= 100_000:
            score -= 0.1
            notes.append("Low liquidity")
    
    # Price level checks
    if price:
        if category in ["crypto", "cryptocurrency"]:
            # For crypto, very low prices can be normal
            if price < 0.01:
                score -= 0.05
                notes.append("Very low price")
        else:
            # For stocks, avoid penny stocks
            if price < 1.0:
                score -= 0.15
                notes.append("Penny stock risk")
            elif price < 5.0:
                score -= 0.05
                notes.append("Low price risk")
            elif price > 100.0:
                score += 0.05
                notes.append("Established price level")

    return _clamp01(score), notes[:2]


# ────────────────────────────────────────────────────────────────────────────
# 3) Weighting by regime - More aggressive
# ────────────────────────────────────────────────────────────────────────────

_REGIME_WEIGHTS = {
    # tech, fund, sent, micro, risk
    "bull":     (0.45, 0.15, 0.25, 0.10, 0.05),  # More weight on technical and sentiment
    "bear":     (0.40, 0.20, 0.25, 0.10, 0.05),  # Balanced approach in bear market
    "range":    (0.35, 0.20, 0.30, 0.10, 0.05),  # More sentiment weight in ranging market
    "volatile": (0.35, 0.15, 0.25, 0.15, 0.10),  # More microstructure weight in volatile market
}

def _blend_scores(regime: str, tech: float, fund: float, sent: float, micro: float, riskc: float) -> float:
    """Blend individual scores using regime-specific weights."""
    w = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["range"])
    return _clamp01(tech*w[0] + fund*w[1] + sent*w[2] + micro*w[3] + riskc*w[4])


def _action_from_conf(conf: float) -> str:
    """Convert confidence score to action - MORE AGGRESSIVE THRESHOLDS."""
    # Much more sensitive thresholds to match your old system
    if conf >= 0.55:        # Lowered from 0.62 to 0.55 (55% confidence for BUY)
        return "Buy"
    if conf <= 0.45:        # Raised from 0.38 to 0.45 (45% confidence for SELL)
        return "Sell"
    return "Hold"           # Only 45-55% range is Hold (much smaller neutral zone)


# ────────────────────────────────────────────────────────────────────────────
# 4) Risk: stop & target
# ────────────────────────────────────────────────────────────────────────────

def _propose_stop_target(price: float, asset: Row, action: str) -> Tuple[Optional[float], Optional[float]]:
    """Propose stop loss and target based on ATR or default percentages."""
    if action == "Hold":
        return None, None

    t = asset.get("technical") or {}
    atr_pct = _nz(_safe_get(t, "atr_pct"))
    
    # Use ATR bands if available; else smaller default stops for more sensitivity
    if atr_pct > 0:
        stop_dist = max(1.0 * atr_pct / 100.0, 0.008)  # Minimum 0.8% stop
    else:
        stop_dist = 0.02  # 2% default stop (smaller than before)
    
    rr = 2.0  # 2:1 Risk-reward ratio (increased from 1.5)

    if action == "Buy":
        stop = price * (1.0 - stop_dist)
        tgt = price * (1.0 + rr * stop_dist)
    elif action == "Sell":
        stop = price * (1.0 + stop_dist)
        tgt = price * (1.0 - rr * stop_dist)
    else:
        return None, None
    
    return round(stop, 6), round(tgt, 6)


# ────────────────────────────────────────────────────────────────────────────
# Core recommendation functions
# ────────────────────────────────────────────────────────────────────────────

def recommend_for_assets(
    assets: List[Row],
    *,
    category: str,
    budget: float = 0.0,
    risk_per_trade: float = 0.01,
    regime_hint: Optional[str] = None
) -> List[Row]:
    """
    Produce ranked recommendations across multiple assets.
    
    This is "deterministic logic" with more aggressive thresholds to find opportunities.
    The CLI can still enforce budget/position sizing afterwards.
    """
    out: List[Row] = []
    cat = (category or "").lower()

    for a in assets:
        price = _nz(a.get("price"))
        if price <= 0:
            continue

        regime = _detect_regime(a, regime_hint)

        tech_s, tech_notes = _score_technical(a)
        fund_s, fund_notes = _score_fundamentals(a, category=cat)
        sent_s, sent_notes = _score_sentiment(a)
        micr_s, micr_notes = _score_microstructure(a)
        risk_s, risk_notes = _score_risk_context(a, category=cat)

        conf = _blend_scores(regime, tech_s, fund_s, sent_s, micr_s, risk_s)
        action = _action_from_conf(conf)

        stop, tgt = _propose_stop_target(price, a, action)

        # Compose human "Key Reasons": pick 3—5 strongest signals
        reason_bits: List[str] = []
        # Pick from technical first (usually most influential)
        reason_bits.extend(tech_notes[:2])
        # Then fundamentals/sentiment/microstructure/risk (1 each if present)
        if fund_notes:
            reason_bits.append(fund_notes[0])
        if sent_notes:
            reason_bits.append(sent_notes[0])
        if micr_notes:
            reason_bits.append(micr_notes[0])
        if risk_notes:
            reason_bits.append(risk_notes[0])

        reasons = "; ".join([r for r in reason_bits if r])[:400]

        rec = {
            "asset": a.get("asset") or a.get("symbol"),
            "symbol": a.get("symbol") or a.get("asset"),
            "price": price,
            "quantity": 0,  # CLI enforces budget
            "sell_target": tgt if action != "Hold" else 0.0,
            "stop_loss": stop if action != "Hold" else 0.0,
            "estimated_profit": 0.0,
            "action": action,
            "confidence": _pct(conf),
            "reasons": f"[Regime: {regime}] " + reasons if reasons else f"[Regime: {regime}]",
        }
        out.append(rec)

    # Rank by confidence (Buy first, then Hold, then Sell)
    def _rank_key(r: Row):
        pri = {"Buy": 0, "Hold": 1, "Sell": 2}.get(r["action"], 3)
        return (pri, -r["confidence"])

    out.sort(key=_rank_key)
    return out


def recommend_one(asset: Row, *, category: str, **kwargs) -> Row:
    """
    Convenience for single-asset analysis path.
    Returns the first (and only) recommendation row.
    """
    rows = recommend_for_assets([asset], category=category, **kwargs)
    return rows[0] if rows else {}


# ────────────────────────────────────────────────────────────────────────────
# API functions expected by main.py
# ────────────────────────────────────────────────────────────────────────────

def analyze_market_batch(
    rows: List[Row],
    *,
    market_ctx: Dict[str, Any],
    feature_flags: Dict[str, Any],
    budget: float
) -> List[Row]:
    """
    Main API function for category flow analysis.
    
    Args:
        rows: List of asset dictionaries from data fetchers
        market_ctx: Market context (region, timezone, sessions, etc.)
        feature_flags: Feature toggles (use_rsi, use_sma, use_sentiment, etc.)
        budget: Available budget for trading
    
    Returns:
        List of recommendation dictionaries
    """
    # Extract category from market context or default to equities
    category = market_ctx.get("category", "equities")
    
    # Determine regime hint from market context if available
    regime_hint = market_ctx.get("regime_hint")
    
    # Calculate risk per trade (could be configurable)
    risk_per_trade = 0.015  # 1.5% of portfolio per trade (more aggressive)
    
    return recommend_for_assets(
        rows,
        category=category,
        budget=budget,
        risk_per_trade=risk_per_trade,
        regime_hint=regime_hint
    )


def analyze_single_asset(
    row: Row,
    *,
    asset_class: str,
    market_ctx: Dict[str, Any],
    feature_flags: Dict[str, Any],
    budget: float
) -> Row:
    """
    Main API function for single asset analysis.
    
    Args:
        row: Single asset dictionary from data fetcher
        asset_class: Type of asset ("equities", "crypto", "forex", etc.)
        market_ctx: Market context (region, timezone, sessions, etc.)
        feature_flags: Feature toggles (use_rsi, use_sma, use_sentiment, etc.)
        budget: Available budget for trading
    
    Returns:
        Single recommendation dictionary
    """
    # Determine regime hint from market context if available
    regime_hint = market_ctx.get("regime_hint")
    
    # Calculate risk per trade
    risk_per_trade = 0.015  # 1.5% of portfolio per trade
    
    return recommend_one(
        row,
        category=asset_class,
        budget=budget,
        risk_per_trade=risk_per_trade,
        regime_hint=regime_hint
    )
