from __future__ import annotations

from typing import Dict, Tuple


# Default weights per regime (sum need not be 1; they’re normalized at runtime)
# You can tweak these centrally without touching the rules elsewhere.
_DEFAULT_WEIGHTS = {
    "bull": {
        "trend": 1.0,          # SMA/EMA alignment, MACD > 0
        "momentum": 0.8,       # RSI > 50, Stoch rising
        "volume": 0.6,         # OBV up, volume expansion on up days
        "volatility": 0.2,     # narrower bands preferred than spike-y vol
        "fundamental": 0.7,    # EPS/revenue growth, healthy margins
        "sentiment": 0.6,      # positive news/social
        "microstructure": 0.4, # bids > asks, depth improving
        "risk": 0.5,           # tight stop, good R:R
    },
    "bear": {
        "trend": 1.0,
        "momentum": 0.8,
        "volume": 0.7,         # distribution days matter more
        "volatility": 0.4,
        "fundamental": 0.6,    # deteriorating fundamentals accelerate downside
        "sentiment": 0.6,
        "microstructure": 0.5,
        "risk": 0.6,
    },
    "range": {
        "trend": 0.5,          # mean-reversion bias
        "momentum": 0.6,
        "volume": 0.5,
        "volatility": 0.5,
        "fundamental": 0.5,
        "sentiment": 0.4,
        "microstructure": 0.5,
        "risk": 0.7,           # risk discipline matters
    },
    "volatile": {
        "trend": 0.7,
        "momentum": 0.8,
        "volume": 0.8,
        "volatility": 0.9,     # reward/penalize based on controllable risk
        "fundamental": 0.5,
        "sentiment": 0.7,
        "microstructure": 0.7,
        "risk": 1.0,
    },
}


def default_weights_for_regime(regime: str) -> Dict[str, float]:
    return dict(_DEFAULT_WEIGHTS.get(regime, _DEFAULT_WEIGHTS["range"]))


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(v)) for v in d.values())
    if s <= 1e-9:
        return {k: 0.0 for k in d}
    return {k: max(0.0, float(v)) / s for k, v in d.items()}


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def combine_signals(
    *,
    regime: str,
    # each block expected in [0,1], where 0=strongly bearish, 0.5=neutral, 1=strongly bullish
    trend: float,
    momentum: float,
    volume: float,
    volatility: float,
    fundamental: float,
    sentiment: float,
    microstructure: float,
    risk: float,
) -> Dict[str, object]:
    """
    Fuse normalized block scores into a decision.

    Returns:
      {
        "action": "Buy"|"Sell"|"Hold",
        "confidence": 0-100,
        "score_components": { block: (score, weight, contribution) ... },
        "buy_score": float,
        "sell_score": float
      }

    Interpretation:
      - buy_score ~ weighted average of block scores
      - sell_score ~ weighted average of (1 - block score)
      - confidence = abs(buy_score - sell_score)
    """
    weights = _normalize(default_weights_for_regime(regime))

    blocks = {
        "trend": _clip01(trend),
        "momentum": _clip01(momentum),
        "volume": _clip01(volume),
        "volatility": _clip01(volatility),
        "fundamental": _clip01(fundamental),
        "sentiment": _clip01(sentiment),
        "microstructure": _clip01(microstructure),
        "risk": _clip01(risk),
    }

    # weighted means
    buy_score = 0.0
    sell_score = 0.0
    comps: Dict[str, Tuple[float, float, float]] = {}
    for k, v in blocks.items():
        w = weights.get(k, 0.0)
        buy_score += w * v
        sell_score += w * (1.0 - v)
        comps[k] = (v, w, w * (v - 0.5) * 2.0)  # contribution around neutral

    # pick action
    if buy_score > sell_score + 0.05:
        action = "Buy"
    elif sell_score > buy_score + 0.05:
        action = "Sell"
    else:
        action = "Hold"

    confidence = int(round(abs(buy_score - sell_score) * 100))

    return {
        "action": action,
        "confidence": confidence,
        "score_components": comps,
        "buy_score": round(buy_score, 4),
        "sell_score": round(sell_score, 4),
        "weights_used": weights,
    }
