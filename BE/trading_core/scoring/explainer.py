from __future__ import annotations

from typing import Dict, List, Tuple


# human-friendly labels per block
_LABELS = {
    "trend": "Trend (SMA/EMA, MACD)",
    "momentum": "Momentum (RSI/Stoch)",
    "volume": "Volume/OBV",
    "volatility": "Volatility (BB/ATR)",
    "fundamental": "Fundamentals",
    "sentiment": "Sentiment (news/social)",
    "microstructure": "Order book/liquidity",
    "risk": "Risk/Stops/R:R",
}


def _top_contributors(score_components: Dict[str, Tuple[float, float, float]], k: int = 4) -> List[Tuple[str, float]]:
    # score_components: { block: (score, weight, contribution) }
    # rank by magnitude of contribution
    ranked = sorted(score_components.items(), key=lambda kv: abs(kv[1][2]), reverse=True)
    out: List[Tuple[str, float]] = []
    for name, (_, _, contrib) in ranked[:k]:
        out.append((name, contrib))
    return out


def explain_recommendation(
    *,
    action: str,
    confidence: int,
    score_components: Dict[str, tuple],
) -> Dict[str, object]:
    """
    Turn fused scores into concise key reasons.
    Returns { "line": "...", "reasons": [ ... ] }
    """
    tops = _top_contributors(score_components, k=4)
    bullets: List[str] = []
    for name, contrib in tops:
        direction = "supportive" if contrib > 0 else "opposing"
        tag = _LABELS.get(name, name)
        strength = f"{abs(contrib):.2f}"
        bullets.append(f"{tag}: {direction} (impact {strength})")

    line = f"[Action: {action} | Confidence: {confidence}% | Key Reasons: " + "; ".join(bullets) + "]"
    return {"line": line, "reasons": bullets}
