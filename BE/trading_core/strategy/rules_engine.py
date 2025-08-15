"""
Deterministic rules engine: multi-signal → recommendation.

Inputs
------
`recommend_for_assets(assets, category, budget, risk_per_trade, regime_hint)`
Where each `asset` is a dict you already use throughout the app, e.g.:

{
  "asset": "AAPL", "symbol": "AAPL", "price": 183.12, "volume": 54234123,
  "day_range_pct": 1.8,
  "price_history": [...],                 # optional; used upstream for RSI/SMA/etc.
  "technical": {                          # optional; attach what you have
      "sma_fast": 20d SMA value,
      "sma_slow": 50d SMA value,
      "ema_fast": 12d EMA,
      "ema_slow": 26d EMA,
      "macd": float,
      "macd_signal": float,
      "adx": float,
      "rsi": float,
      "stoch_k": float, "stoch_d": float,
      "obv_slope": float,                 # normalized slope (-1..+1) if you compute it
      "bb_pos": float,                    # position in Bollinger band [0..1]
      "atr_pct": float,                   # ATR/Price * 100
      "volume_spike": float               # today_vol / 20d_avg_vol
  },
  "fundamentals": {                       # optional
      "pe": float, "pe_sector": float,
      "revenue_yoy": float, "eps_yoy": float,
      "debt_to_equity": float, "gross_margin": float,
      # crypto-style:
      "dev_activity": float,              # normalized 0..1
      "tx_growth_yoy": float,             # %
      "active_addr_growth_yoy": float,    # %
      "token_inflation_yoy": float        # %
  },
  "sentiment": {                          # optional
      "news_score": float,                # -1..+1
      "social_score": float,              # -1..+1
      "fear_greed": float                 # 0..100
  },
  "microstructure": {                     # optional
      "order_imbalance": float,           # -1..+1
      "liq_depth_score": float,           # 0..1
      "whale_score": float                # -1..1
  }
}

Outputs
-------
List of recommendation rows the CLI can print directly:

{
  "asset": "AAPL",
  "symbol": "AAPL",
  "price": 183.12,
  "quantity": 0,                          # CLI will resize to budget anyway
  "sell_target": 188.5,
  "stop_loss": 177.6,
  "estimated_profit": 0.0,                # placeholder (CLI handles totals)
  "action": "Buy" | "Sell" | "Hold",
  "confidence": 74,                       # 0..100
  "reasons": "Trend up (SMA/EMA, MACD>0), momentum mid-high (RSI 58), ..."
}

Design
------
• Regime detection → adjust weights across buckets.
• Each bucket returns (score: 0..100, top_notes: list of strings).
• Weighted blend → confidence 0..100; thresholds map to Buy/Sell/Hold.
• Risk: propose stop/target using ATR% if available, else sensible defaults.
• Graceful degradation: missing signals simply reduce that bucket’s weight.

No hard-coded tickers, regions, or markets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

Number = float
Row = Dict[str, Any]


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _safe_get(d: Optional[dict], key: str, default=None):
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _nz(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):  # NaN-safe
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _pct(x: float) -> int:
    return int(round(100.0 * _clamp01(x)))


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


# ────────────────────────────────────────────────────────────
# 1) Regime detection (quick, local)
# ────────────────────────────────────────────────────────────

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
        if rh in {"bull","bear","range","volatile"}:
            return rh

    price = _nz(asset.get("price"), 0.0)
    tech  = asset.get("technical") or {}
    sma_slow = _nz(_safe_get(tech, "sma_slow"), 0.0)
    macd     = _nz(_safe_get(tech, "macd"), 0.0)
    adx      = _nz(_safe_get(tech, "adx"), 0.0)
    atr_pct  = _nz(_safe_get(tech, "atr_pct"), 0.0)
    drp      = _nz(asset.get("day_range_pct"), 0.0)

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


# ────────────────────────────────────────────────────────────
# 2) Bucket scoring
# ────────────────────────────────────────────────────────────

def _score_technical(asset: Row) -> Tuple[float, List[str]]:
    """Return (0..1, notes)."""
    notes: List[str] = []
    t = asset.get("technical") or {}
    if not t:
        return 0.0, ["No technical signals."]

    price    = _nz(asset.get("price"))
    sma_f    = _nz(_safe_get(t, "sma_fast"))
    sma_s    = _nz(_safe_get(t, "sma_slow"))
    ema_f    = _nz(_safe_get(t, "ema_fast"))
    ema_s    = _nz(_safe_get(t, "ema_slow"))
    macd     = _nz(_safe_get(t, "macd"))
    macdsig  = _nz(_safe_get(t, "macd_signal"))
    adx      = _nz(_safe_get(t, "adx"))
    rsi      = _nz(_safe_get(t, "rsi"))
    k        = _nz(_safe_get(t, "stoch_k"))
    d        = _nz(_safe_get(t, "stoch_d"))
    obv_sl   = _nz(_safe_get(t, "obv_slope"))
    bb_pos   = _nz(_safe_get(t, "bb_pos"))
    atr_pct  = _nz(_safe_get(t, "atr_pct"))
    vol_spk  = _nz(_safe_get(t, "volume_spike"))
    drp      = _nz(asset.get("day_range_pct"))

    score = 0.0
    weight_sum = 0.0

    # Trend (SMA/EMA alignment)
    if sma_f and sma_s:
        w = 0.15
        trend_up = sma_f > sma_s
        score += w * (1.0 if trend_up else 0.0)
        weight_sum += w
        notes.append(f"SMA trend {'up' if trend_up else 'down'} (fast vs slow).")
    if ema_f and ema_s:
        w = 0.10
        ema_up = ema_f > ema_s
        score += w * (1.0 if ema_up else 0.0)
        weight_sum += w
        notes.append(f"EMA trend {'up' if ema_up else 'down'} (fast vs slow).")

    # MACD
    if macd or macdsig:
        w = 0.15
        bull = macd > macdsig and macd > 0
        score += w * (1.0 if bull else 0.0)
        weight_sum += w
        notes.append(f"MACD {'bullish' if bull else 'bearish/flat'}.")

    # ADX (trend strength)
    if adx:
        w = 0.05
        strong = adx >= 20
        score += w * (1.0 if strong else 0.5 if adx >= 15 else 0.0)
        weight_sum += w
        notes.append(f"ADX {adx:.0f} ({'strong' if strong else 'weak'} trend).")

    # Momentum (RSI/Stoch)
    if rsi:
        w = 0.15
        if 50 <= rsi <= 70:
            val = 0.8
        elif 40 <= rsi < 50:
            val = 0.5
        elif 30 <= rsi < 40:
            val = 0.3
        elif rsi > 70:
            val = 0.4  # overbought risk
        else:
            val = 0.2  # weak/oversold (could be value, but penalize until reversal confirmed)
        score += w * val
        weight_sum += w
        notes.append(f"RSI {rsi:.0f}.")
    if k and d:
        w = 0.05
        cross_up = k > d and k < 80
        score += w * (0.7 if cross_up else 0.3)
        weight_sum += w
        notes.append(f"Stoch {'bullish' if cross_up else 'neutral/bear'} (K/D).")

    # Volume (OBV slope + spikes)
    if obv_sl:
        w = 0.05
        score += w * _clamp01((obv_sl + 1.0) / 2.0)  # map -1..1 → 0..1
        weight_sum += w
        notes.append("OBV slope supportive." if obv_sl > 0 else "OBV slope weak.")
    if vol_spk:
        w = 0.05
        val = 1.0 if vol_spk >= 1.5 else (0.5 if vol_spk >= 1.1 else 0.2)
        score += w * val
        weight_sum += w
        notes.append(f"Volume {'spike' if vol_spk >= 1.5 else 'normal'} ({vol_spk:.2f}x avg).")

    # Volatility (Bollinger position + ATR%)
    if bb_pos:
        w = 0.05
        # Middle band ≈ neutrally safe; near lower band can be value; near upper band risk
        val = 0.7 if 0.35 <= bb_pos <= 0.65 else (0.5 if 0.2 <= bb_pos <= 0.8 else 0.3)
        score += w * val
        weight_sum += w
        notes.append(f"BBand position {bb_pos:.2f}.")
    if atr_pct or drp:
        w = 0.10
        vol = atr_pct if atr_pct else drp
        # too high volatility penalizes; moderate gets 0.7; very low 0.5
        if vol <= 1.5:
            val = 0.7
        elif vol <= 3.0:
            val = 0.55
        else:
            val = 0.25
        score += w * val
        weight_sum += w
        notes.append(f"Volatility {vol:.1f}% ATR/Range.")

    if weight_sum == 0:
        return 0.0, ["No technical inputs."]
    return _clamp01(score / weight_sum), notes


def _score_fundamentals(asset: Row, *, category: str) -> Tuple[float, List[str]]:
    f = asset.get("fundamentals") or {}
    notes: List[str] = []
    if not f:
        return 0.5, ["No fundamentals — neutral."]  # degrade gracefully

    score = 0.0
    wsum  = 0.0

    if category == "equities" or category == "stocks":
        pe        = _nz(_safe_get(f, "pe"))
        pe_sec    = _nz(_safe_get(f, "pe_sector"))
        rev_yoy   = _nz(_safe_get(f, "revenue_yoy"))
        eps_yoy   = _nz(_safe_get(f, "eps_yoy"))
        dte       = _nz(_safe_get(f, "debt_to_equity"))
        margin    = _nz(_safe_get(f, "gross_margin"))

        # Valuation relative to sector
        if pe and pe_sec:
            w = 0.2
            val = 0.7 if pe <= 0.9*pe_sec else (0.5 if pe <= 1.1*pe_sec else 0.3)
            score += w * val; wsum += w
            notes.append(f"P/E vs sector: {pe:.1f} vs {pe_sec:.1f}.")

        # Growth
        if rev_yoy or eps_yoy:
            w = 0.25
            g = 0.5
            if rev_yoy > 5 or eps_yoy > 5:
                g = 0.7
            if rev_yoy > 15 or eps_yoy > 15:
                g = 0.85
            if rev_yoy > 30 or eps_yoy > 30:
                g = 1.0
            score += w * g; wsum += w
            notes.append(f"Growth YoY (rev {rev_yoy:.1f}%, eps {eps_yoy:.1f}%).")

        # Balance sheet quality
        if dte:
            w = 0.15
            val = 0.75 if dte < 0.8 else (0.55 if dte < 1.5 else 0.35)
            score += w * val; wsum += w
            notes.append(f"Debt/Equity {dte:.2f}.")

        if margin:
            w = 0.15
            val = 0.75 if margin >= 40 else (0.6 if margin >= 20 else 0.45)
            score += w * val; wsum += w
            notes.append(f"Gross margin {margin:.1f}%.")

    else:
        # Crypto-style fundamentals / on-chain proxies
        dev_act  = _nz(_safe_get(f, "dev_activity"))
        tx_g     = _nz(_safe_get(f, "tx_growth_yoy"))
        addr_g   = _nz(_safe_get(f, "active_addr_growth_yoy"))
        infl     = _nz(_safe_get(f, "token_inflation_yoy"))

        if dev_act:
            w = 0.25
            score += w * _clamp01(dev_act)  # already 0..1
            wsum  += w
            notes.append(f"Dev activity {dev_act:.2f}.")

        if tx_g or addr_g:
            w = 0.25
            g = 0.5
            if tx_g > 5 or addr_g > 5:
                g = 0.7
            if tx_g > 15 or addr_g > 15:
                g = 0.85
            if tx_g > 30 or addr_g > 30:
                g = 1.0
            score += w * g; wsum += w
            notes.append(f"On-chain growth (tx {tx_g:.1f}%, addr {addr_g:.1f}%).")

        if infl:
            w = 0.2
            # lower inflation is better
            val = 0.8 if infl <= 2 else (0.6 if infl <= 6 else 0.4)
            score += w * val; wsum += w
            notes.append(f"Token inflation {infl:.1f}%.")

    if wsum == 0:
        return 0.5, ["Sparse fundamentals — neutral."]
    return _clamp01(score / wsum), notes


def _score_sentiment(asset: Row) -> Tuple[float, List[str]]:
    s = asset.get("sentiment") or {}
    news   = _nz(_safe_get(s, "news_score"))
    social = _nz(_safe_get(s, "social_score"))
    fg     = _nz(_safe_get(s, "fear_greed"))
    notes: List[str] = []
    has_any = False

    score = 0.0
    wsum  = 0.0

    if news:
        has_any = True
        w = 0.45
        score += w * _clamp01((news + 1.0) / 2.0)  # -1..1 → 0..1
        wsum  += w
        notes.append(f"News sentiment {news:+.2f}.")
    if social:
        has_any = True
        w = 0.35
        score += w * _clamp01((social + 1.0) / 2.0)
        wsum  += w
        notes.append(f"Social sentiment {social:+.2f}.")
    if fg:
        has_any = True
        w = 0.20
        val = 0.5
        if fg >= 60:
            val = 0.7
        if fg >= 75:
            val = 0.85
        if fg <= 25:
            val = 0.3  # fear may be contrarian, but keep conservative
        score += w * val; wsum += w
        notes.append(f"Fear & Greed {fg:.0f}.")

    if not has_any:
        return 0.5, ["No sentiment data — neutral."]
    return _clamp01(score / wsum), notes


def _score_microstructure(asset: Row) -> Tuple[float, List[str]]:
    m = asset.get("microstructure") or {}
    imbal = _nz(_safe_get(m, "order_imbalance"))
    liq   = _nz(_safe_get(m, "liq_depth_score"))
    whale = _nz(_safe_get(m, "whale_score"))
    notes: List[str] = []
    has_any = False

    score = 0.0
    wsum  = 0.0

    if imbal:
        has_any = True
        w = 0.5
        score += w * _clamp01((imbal + 1.0) / 2.0)   # -1..1 → 0..1
        wsum  += w
        notes.append(f"Order imbalance {imbal:+.2f}.")
    if liq:
        has_any = True
        w = 0.3
        score += w * _clamp01(liq)                   # 0..1
        wsum  += w
        notes.append(f"Liquidity depth {liq:.2f}.")
    if whale:
        has_any = True
        w = 0.2
        score += w * _clamp01((whale + 1.0) / 2.0)   # -1..1 → 0..1
        wsum  += w
        notes.append(f"Whale activity {whale:+.2f}.")

    if not has_any:
        return 0.5, ["No microstructure data — neutral."]
    return _clamp01(score / wsum), notes


def _score_risk_context(asset: Row, *, category: str) -> Tuple[float, List[str]]:
    """
    Risk context is not “performance” but tradability:
      • Lower ATR% and reasonable liquidity → higher score
      • Extremely high volatility or very low volume → lower score
    """
    notes: List[str] = []
    t = asset.get("technical") or {}
    atr_pct = _nz(_safe_get(t, "atr_pct"))
    vol     = _nz(asset.get("volume"))

    score = 0.5
    wsum  = 1.0

    # Volatility
    if atr_pct:
        if atr_pct <= 1.0:
            score += 0.2
            notes.append("Low volatility (ATR%).")
        elif atr_pct <= 3.0:
            score += 0.05
            notes.append("Moderate volatility.")
        else:
            score -= 0.15
            notes.append("High volatility — position carefully.")

    # Liquidity proxy
    if vol:
        # Normalize loosely by category
        if category in {"equities","stocks","funds","warrants"}:
            if vol >= 5_000_000:
                score += 0.1; notes.append("High liquidity.")
            elif vol >= 500_000:
                score += 0.03; notes.append("OK liquidity.")
            else:
                score -= 0.10; notes.append("Thin liquidity.")
        else:
            # crypto/forex/futures/commodities: volume scale varies wildly; keep mild
            if vol <= 0:
                score -= 0.05; notes.append("Unknown liquidity.")
            else:
                score += 0.02; notes.append("Liquidity acceptable.")

    return _clamp01(score), notes


# ────────────────────────────────────────────────────────────
# 3) Weighting by regime
# ────────────────────────────────────────────────────────────

_REGIME_WEIGHTS = {
    # tech, fund, sent, micro, risk
    "bull":     (0.40, 0.20, 0.20, 0.10, 0.10),
    "bear":     (0.35, 0.25, 0.20, 0.10, 0.10),
    "range":    (0.30, 0.25, 0.25, 0.10, 0.10),
    "volatile": (0.30, 0.15, 0.20, 0.15, 0.20),
}

def _blend_scores(regime: str, tech: float, fund: float, sent: float, micro: float, riskc: float) -> float:
    w = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS["range"])
    return _clamp01( tech*w[0] + fund*w[1] + sent*w[2] + micro*w[3] + riskc*w[4] )


def _action_from_conf(conf: float) -> str:
    if conf >= 0.62:
        return "Buy"
    if conf <= 0.38:
        return "Sell"
    return "Hold"


# ────────────────────────────────────────────────────────────
# 4) Risk: stop & target
# ────────────────────────────────────────────────────────────

def _propose_stop_target(price: float, asset: Row, action: str) -> Tuple[Optional[float], Optional[float]]:
    t = asset.get("technical") or {}
    atr_pct = _nz(_safe_get(t, "atr_pct"))
    # Use ATR bands if available; else 3% stop, 1.5R target
    if atr_pct > 0:
        stop_dist = max(1.2 * atr_pct / 100.0, 0.01)  # avoid ultra-tight stops
    else:
        stop_dist = 0.03
    rr = 1.5

    if action == "Buy":
        stop = price * (1.0 - stop_dist)
        tgt  = price * (1.0 + rr*stop_dist)
    elif action == "Sell":
        stop = price * (1.0 + stop_dist)
        tgt  = price * (1.0 - rr*stop_dist)
    else:
        return None, None
    return round(stop, 4), round(tgt, 4)


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def recommend_for_assets(
    assets: List[Row],
    *,
    category: str,
    budget: float = 0.0,
    risk_per_trade: float = 0.01,            # fraction of equity to risk per trade (for info only)
    regime_hint: Optional[str] = None
) -> List[Row]:
    """
    Produce ranked recommendations across multiple assets (or a single asset).
    This is “deterministic logic” parallel to your LLM strategy.

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

        # Compose human “Key Reasons”: pick 3–5 strongest signals
        reason_bits: List[str] = []
        # Pick from technical first (usually most influential)
        reason_bits.extend(tech_notes[:2])
        # Then fundamentals/sentiment/microstructure/risk (1 each if present)
        if fund_notes: reason_bits.append(fund_notes[0])
        if sent_notes: reason_bits.append(sent_notes[0])
        if micr_notes: reason_bits.append(micr_notes[0])
        if risk_notes: reason_bits.append(risk_notes[0])

        reasons = "; ".join([r for r in reason_bits if r])[:400]

        rec = {
            "asset": a.get("asset") or a.get("symbol"),
            "symbol": a.get("symbol") or a.get("asset"),
            "price": price,
            "quantity": 0,                      # CLI enforces budget
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
