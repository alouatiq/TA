# BE/trading_core/scoring/regime.py
"""
Market regime detection (Bull / Bear / Range / Volatile), multi-timeframe.

Inputs
------
- Close price history (list/array/Series) at daily granularity (>= 60 bars recommended).
- Optional helper metrics you might already have (ATR, BB width), but this module
  can compute minimal features itself.

Outputs
-------
detect_regime(close, **kwargs) -> dict:
{
  "regime": "bull" | "bear" | "range" | "volatile",
  "confidence": float in [0,1],
  "features": {
      "mt_trend": float,         # multi-timeframe trend score [-1..+1]
      "volatility": float,       # normalized recent vol  [0..1]
      "rangeiness": float,       # BB% / ATR mean reversion hint [0..1]
      "momentum": float,         # RSI(14) z-scored-ish      [-1..+1]
  },
  "thresholds": { ... }          # the thresholds that were used
}

Design
------
1) Compute short/medium/long EMAs to gauge directional slope across timeframes.
2) Compute RSI(14) for momentum context.
3) Compute ATR(14) and Bollinger Band width to separate “calm” vs “volatile/ranging”.
4) Combine into coarse scores; apply simple, explainable thresholds.

This module is *deterministic* and has zero market hard-coding.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _bb_width(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    width = (upper - lower) / ma.replace(0, np.nan)
    return width.replace([np.inf, -np.inf], np.nan).fillna(method="bfill").fillna(0.0)


def _normalize_01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return float(max(0.0, min(1.0, v)))


def _normalize_signed(x: float, lo: float, hi: float) -> float:
    """Map [lo..hi] roughly to [-1..+1]."""
    z = _normalize_01(x, lo, hi)
    return 2.0 * z - 1.0


def _multi_tf_trend(close: pd.Series) -> float:
    """
    Simple multi-timeframe trend: weigh EMA slopes at 10/20/50 days.
    Output in [-1..+1].
    """
    if len(close) < 60:
        # Not enough for stable 50 EMA slope; degrade gracefully
        spans = [10, 20]
        weights = [0.6, 0.4]
    else:
        spans = [10, 20, 50]
        weights = [0.5, 0.3, 0.2]

    slopes = []
    for s in spans:
        ema_s = _ema(close, s)
        # slope as % of price over last s bars
        # use small window to estimate recent slope direction
        win = max(3, int(s / 4))
        d = ema_s.diff(win)
        denom = (close.shift(win)).replace(0, np.nan).abs()
        slope = (d / denom).iloc[-1] if len(d) else 0.0
        slopes.append(float(slope if np.isfinite(slope) else 0.0))

    # robust scale to [-1..1] using soft bounds
    # positive ~ uptrend; negative ~ downtrend
    soft_lo, soft_hi = -0.15, 0.15
    norm = [_normalize_signed(s, soft_lo, soft_hi) for s in slopes]
    return float(np.average(norm, weights=weights))


def detect_regime(
    close: Sequence[float],
    *,
    high: Optional[Sequence[float]] = None,
    low: Optional[Sequence[float]] = None,
    # thresholds (tune if needed)
    trend_bull: float = 0.25,
    trend_bear: float = -0.25,
    vol_volatile: float = 0.65,     # 0..1 scale
    rangeiness_hi: float = 0.55,    # 0..1 scale
) -> Dict[str, object]:
    """
    Determine market regime using multi-timeframe trend + volatility + rangeiness + momentum.

    Heuristics:
      • Bull if trend >> 0 and momentum supportive, unless volatility extremely high.
      • Bear if trend << 0 and momentum weak, unless volatility collapses and BB narrows.
      • Range if trend ~ 0 and BB width/ATR suggest mean-reverting conditions.
      • Volatile if normalized vol is high regardless of direction.

    Returns regime + confidence + feature breakdown.
    """
    s = pd.Series(close, dtype="float64").dropna()
    if s.size < 30:
        return {
            "regime": "range",
            "confidence": 0.3,
            "features": {"mt_trend": 0.0, "volatility": 0.0, "rangeiness": 0.0, "momentum": 0.0},
            "thresholds": {
                "trend_bull": trend_bull, "trend_bear": trend_bear,
                "vol_volatile": vol_volatile, "rangeiness_hi": rangeiness_hi
            },
        }

    # compute minimal OHLC if not provided
    if high is None or low is None:
        # approximate highs/lows using close ± rolling std
        std20 = s.rolling(20, min_periods=5).std().fillna(method="bfill").fillna(0.0)
        h = (s + std20).rename("high")
        l = (s - std20).rename("low")
    else:
        h = pd.Series(high, dtype="float64").reindex_like(s).fillna(method="bfill").fillna(method="ffill")
        l = pd.Series(low, dtype="float64").reindex_like(s).fillna(method="bfill").fillna(method="ffill")

    # features
    mt_trend = _multi_tf_trend(s)                                     # [-1..+1]
    rsi = _rsi(s).iloc[-1]                                            # 0..100
    momentum = _normalize_signed(rsi, 30, 70)                         # ~[-1..+1]
    atr = _atr(h, l, s).iloc[-1]                                      # absolute
    bbw = _bb_width(s).iloc[-1]                                       # relative (% of MA)

    # normalize volatility via ATR as fraction of price (last close)
    last = float(s.iloc[-1])
    atr_norm = _normalize_01(atr / (last + 1e-12), 0.005, 0.06)       # 0.5%..6% of price soft bounds
    # rangeiness: prefer BB width capturing choppy/sideways; combine with low trend magnitude
    trend_mag = abs(mt_trend)
    bbw_norm = _normalize_01(float(bbw), 0.02, 0.18)                  # 2%..18% soft bounds
    rangeiness = float(bbw_norm * (1.0 - min(1.0, trend_mag)))        # 0..1

    # rules
    vol_is_high = atr_norm >= vol_volatile
    bullish = (mt_trend >= trend_bull) and (momentum > -0.1)
    bearish = (mt_trend <= trend_bear) and (momentum < 0.1)
    ranging = (trend_mag < 0.2) and (rangeiness >= rangeiness_hi)

    # choose regime
    if vol_is_high:
        regime = "volatile"
        base_conf = atr_norm
    elif bullish and not ranging:
        regime = "bull"
        base_conf = 0.5 * _normalize_01(mt_trend, 0.0, 1.0) + 0.5 * _normalize_01(momentum, 0.0, 1.0)
    elif bearish and not ranging:
        regime = "bear"
        base_conf = 0.5 * _normalize_01(-mt_trend, 0.0, 1.0) + 0.5 * _normalize_01(-momentum, 0.0, 1.0)
    elif ranging:
        regime = "range"
        base_conf = rangeiness
    else:
        # ambiguous → pick by trend sign with low confidence
        regime = "bull" if mt_trend >= 0 else "bear"
        base_conf = 0.35 + 0.15 * (abs(mt_trend))

    # clamp
    confidence = float(max(0.0, min(1.0, base_conf)))

    return {
        "regime": regime,
        "confidence": confidence,
        "features": {
            "mt_trend": float(mt_trend),
            "volatility": float(atr_norm),
            "rangeiness": float(rangeiness),
            "momentum": float(momentum),
        },
        "thresholds": {
            "trend_bull": float(trend_bull),
            "trend_bear": float(trend_bear),
            "vol_volatile": float(vol_volatile),
            "rangeiness_hi": float(rangeiness_hi),
        },
    }
