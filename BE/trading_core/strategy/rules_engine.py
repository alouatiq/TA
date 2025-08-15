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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Regime detection (quick, local)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_regime(asset: Row, regime_hint: Optional[str]) -> str:
    """
    Heuristic regime from technicals:
      - bull: price above SMA slow & MACD>0 & ADX>=20
      - bear: price below SMA slow & MACD<0 & ADX>=20
      - range: ADX<20 or conflicting signals
      - volatile: ATR%>3.5 or day_range_pct>3
    `regime_hint` (e.g., 'bull','bear','range','volatile') can override.
    """
    if regime_hint:
        rh = regime_hint.strip().lower()
        if rh in {"bull", "bear", "range", "volatile"}:
            return rh

    price = _nz(asset.get("price"), 0.0)
    tech = asset.get("technical") or {}
    sma_slow = _nz(_safe_get(tech, "sma_slow"), 0.0)
    macd = _nz(_safe_get(tech, "macd"), 0.0)
    adx = _nz(_safe_get(tech, "adx"), 0.0)
    atr_pct = _nz(_safe_get(tech, "atr_pct"), 0.0)
    drp = _nz(asset.get("day_range_pct"), 0.0)

    # Volatility first
    if atr_pct >= 3.5 or drp >= 3.0:
        return "volatile"

    if price and sma_slow:
        above = price > sma_slow
        if adx >= 20:
            if above and macd > 0:
                return "bull"
            if (not above) and macd < 0:
                return "bear"

    return "range"


# ──────────────────────────────────────────────────────────────────────────────
# 2) Bucket scoring
# ──────────────────────────────────────────────────────────────────────────────

def _score_technical(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes) for technical analysis."""
    notes: List[str] = []
    t = asset.get("technical") or {}
    if not t:
        return 0.5, ["No technical signals available."]  # Neutral when no data

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

    # Trend (SMA/EMA alignment)
    if sma_f and sma_s:
        w = 0.15
        trend_up = sma_f > sma_s
        score += w * (1.0 if trend_up else 0.0)
        weight_sum += w
        notes.append(f"SMA trend {'up' if trend_up else 'down'}")

    # MACD
    if macd or macdsig:
        w = 0.15
        bull = macd > macdsig and macd > 0
        score += w * (1.0 if bull else 0.0)
        weight_sum += w
        notes.append(f"MACD {'bullish' if bull else 'bearish'}")

    # RSI (momentum)
    if rsi:
        w = 0.10
        if rsi > 70:
            rscore = 0.2  # Overbought
            notes.append("RSI overbought")
        elif rsi < 30:
            rscore = 0.8  # Oversold (potential buy)
            notes.append("RSI oversold")
        elif 40 <= rsi <= 60:
            rscore = 0.6  # Neutral-bullish
            notes.append("RSI neutral")
        else:
            rscore = 0.5  # Moderate
            notes.append("RSI moderate")
        score += w * rscore
        weight_sum += w

    # Volume analysis
    if vol_spk:
        w = 0.05
        if vol_spk > 1.5:
            vscore = 0.8  # High volume confirms moves
            notes.append("High volume")
        elif vol_spk < 0.5:
            vscore = 0.3  # Low volume = weak signals
            notes.append("Low volume")
        else:
            vscore = 0.5
            notes.append("Normal volume")
        score += w * vscore
        weight_sum += w

    # Simple price momentum (fallback)
    if drp:
        w = 0.05
        if drp > 2:
            pscore = 0.7  # Strong daily move
            notes.append("Strong price move")
        elif drp < -2:
            pscore = 0.3  # Strong decline
            notes.append("Strong decline")
        else:
            pscore = 0.5
            notes.append("Moderate price action")
        score += w * pscore
        weight_sum += w

    # Normalize score
    if weight_sum > 0:
        score = score / weight_sum
    else:
        score = 0.5  # Neutral if no indicators

    return _clamp01(score), notes[:3]  # Limit notes


def _score_fundamentals(asset: Row, category: str = "") -> Tuple[float, List[str]]:
    """Return (0..1, notes) for fundamental analysis."""
    notes: List[str] = []
    f = asset.get("fundamentals") or {}
    if not f:
        return 0.5, ["No fundamental data available."]

    # Simplified fundamental scoring
    score = 0.5  # Start neutral
    
    # For equities
    if category in ["equities", "stocks", "equity"]:
        pe = _nz(_safe_get(f, "pe"))
        revenue_yoy = _nz(_safe_get(f, "revenue_yoy"))
        eps_yoy = _nz(_safe_get(f, "eps_yoy"))
        
        if pe:
            if 10 <= pe <= 25:
                score += 0.1
                notes.append("Reasonable P/E")
            elif pe > 30:
                score -= 0.1
                notes.append("High P/E")
        
        if revenue_yoy:
            if revenue_yoy > 0.10:  # 10% growth
                score += 0.1
                notes.append("Strong revenue growth")
            elif revenue_yoy < -0.05:  # -5% decline
                score -= 0.1
                notes.append("Revenue declining")

    # For crypto
    elif category in ["crypto", "cryptocurrencies"]:
        dev_activity = _nz(_safe_get(f, "dev_activity"))
        tx_growth = _nz(_safe_get(f, "tx_growth_yoy"))
        
        if dev_activity > 0.7:
            score += 0.1
            notes.append("High dev activity")
        elif dev_activity < 0.3:
            score -= 0.1
            notes.append("Low dev activity")
            
        if tx_growth > 0.20:  # 20% transaction growth
            score += 0.1
            notes.append("Strong network growth")

    return _clamp01(score), notes[:2]


def _score_sentiment(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes) for sentiment analysis."""
    notes: List[str] = []
    s = asset.get("sentiment") or {}
    if not s:
        return 0.5, ["No sentiment data available."]

    news_score = _nz(_safe_get(s, "news_score"))
    social_score = _nz(_safe_get(s, "social_score"))
    fear_greed = _nz(_safe_get(s, "fear_greed"))

    scores = []
    
    if news_score:
        # Convert from -1..1 to 0..1
        news_norm = (news_score + 1) / 2
        scores.append(news_norm)
        if news_score > 0.2:
            notes.append("Positive news sentiment")
        elif news_score < -0.2:
            notes.append("Negative news sentiment")

    if social_score:
        # Convert from -1..1 to 0..1
        social_norm = (social_score + 1) / 2
        scores.append(social_norm)
        if social_score > 0.2:
            notes.append("Positive social sentiment")

    if fear_greed:
        # Assume fear_greed is 0..100, convert to 0..1
        fg_norm = fear_greed / 100.0
        scores.append(fg_norm)
        if fear_greed > 70:
            notes.append("Market greed (caution)")
        elif fear_greed < 30:
            notes.append("Market fear (opportunity)")

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
        return 0.5, ["No microstructure data available."]

    order_imbalance = _nz(_safe_get(m, "order_imbalance"))
    liq_depth = _nz(_safe_get(m, "liq_depth_score"))
    whale_score = _nz(_safe_get(m, "whale_score"))

    score = 0.5  # Start neutral

    if order_imbalance:
        if order_imbalance > 0.2:
            score += 0.1
            notes.append("Buy pressure")
        elif order_imbalance < -0.2:
            score -= 0.1
            notes.append("Sell pressure")

    if liq_depth and liq_depth > 0.7:
        score += 0.05
        notes.append("Good liquidity")

    return _clamp01(score), notes[:2]


def _score_risk_context(asset: Row, category: str = "") -> Tuple[float, List[str]]:
    """Return (0..1, notes) for risk/liquidity context."""
    notes: List[str] = []
    vol = _nz(asset.get("volume"))
    price = _nz(asset.get("price"))
    
    score = 0.5  # Start neutral

    # Basic volume/liquidity check
    if category in ["equities", "stocks"]:
        if vol >= 1_000_000:
            score += 0.1
            notes.append("High liquidity")
        elif vol >= 500_000:
            score += 0.05
            notes.append("OK liquidity")
        elif vol <= 100_000:
            score -= 0.1
            notes.append("Low liquidity")
    
    # Price level check (avoid penny stocks)
    if price:
        if price < 1.0:
            score -= 0.1
            notes.append("Low price risk")
        elif price > 100.0:
            score += 0.05
            notes.append("Established price level")

    return _clamp01(score), notes[:2]


# ──────────────────────────────────────────────────────────────────────────────
# 3) Weighting by regime
# ──────────────────────────────────────────────────────────────────────────────

_REGIME_WEIGHTS = {
    # tech, fund, sent, micro, risk
    "bull":     (0.40, 0.20, 0.20, 0.10, 0.10),
    "bear":     (0.35, 0.25, 0.20, 0.10, 0.10),
    "range":    (0.30, 0.25, 0.25, 0.10, 0.10),
    "volatile": (0.30, 0.15, 0.20, 0.15, 0.20),
}

def _blend_scores(regime: str, tech: float, fund: float, sent: float, micro: float, riskc: float) -> float:
    """Blend individual scores using regime-specific weights."""
    w = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["range"])
    return _clamp01(tech*w[0] + fund*w[1] + sent*w[2] + micro*w[3] + riskc*w[4])


def _action_from_conf(conf: float) -> str:
    """Convert confidence score to action."""
    if conf >= 0.62:
        return "Buy"
    if conf <= 0.38:
        return "Sell"
    return "Hold"


# ──────────────────────────────────────────────────────────────────────────────
# 4) Risk: stop & target
# ──────────────────────────────────────────────────────────────────────────────

def _propose_stop_target(price: float, asset: Row, action: str) -> Tuple[Optional[float], Optional[float]]:
    """Propose stop loss and target based on ATR or default percentages."""
    t = asset.get("technical") or {}
    atr_pct = _nz(_safe_get(t, "atr_pct"))
    
    # Use ATR bands if available; else 3% stop, 1.5R target
    if atr_pct > 0:
        stop_dist = max(1.2 * atr_pct / 100.0, 0.01)  # avoid ultra-tight stops
    else:
        stop_dist = 0.03  # 3% default
    
    rr = 1.5  # Risk-reward ratio

    if action == "Buy":
        stop = price * (1.0 - stop_dist)
        tgt = price * (1.0 + rr * stop_dist)
    elif action == "Sell":
        stop = price * (1.0 + stop_dist)
        tgt = price * (1.0 - rr * stop_dist)
    else:
        return None, None
    
    return round(stop, 4), round(tgt, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Core recommendation functions
# ──────────────────────────────────────────────────────────────────────────────

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
    
    This is "deterministic logic" parallel to your LLM strategy.
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

        # Compose human "Key Reasons": pick 3–5 strongest signals
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


# ──────────────────────────────────────────────────────────────────────────────
# API functions expected by main.py
# ──────────────────────────────────────────────────────────────────────────────

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
    risk_per_trade = 0.02  # 2% of portfolio per trade
    
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
    risk_per_trade = 0.02  # 2% of portfolio per trade
    
    return recommend_one(
        row,
        category=asset_class,
        budget=budget,
        risk_per_trade=risk_per_trade,
        regime_hint=regime_hint
    )
