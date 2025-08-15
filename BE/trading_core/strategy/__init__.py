# BE/trading_core/strategy/__init__.py
"""
Strategy engines (facade)

This package exposes a single entrypoint `analyze_market(...)` that can route to:
  • Deterministic rules engine  → trading_core.strategy.rules_engine
  • LLM-based strategy engine   → trading_core.strategy.strategy_llm

Selection logic:
  • Default = "rules"
  • If OPENAI_API_KEY (or compatible key) is present, we *can* use "llm"
  • Env overrides:
        USE_LLM=true|1|yes    → default to LLM
        FORCE_RULES=true|1    → force rules (ignore LLM)
        FORCE_LLM=true|1      → force LLM  (errors if no key)

Both engines should accept the same kwargs:
    market_data: List[dict]
    budget: float
    market_type: str           # 'crypto','equities','forex','commodities','futures','warrants','funds'
    history: Any               # past trades/sessions (optional)
    sentiment: Optional[List[str]]
    use_rsi: bool
    use_sma: bool
    use_sentiment: bool
    market: Optional[str]
    market_context: Dict[str, Any]

Return shape:
    List[dict]  OR  (List[dict], Dict[str, Any])
The facade just forwards whatever the engine returns (to keep compatibility).

Usage:
    from trading_core.strategy import analyze_market
    recs = analyze_market(..., engine="auto")   # or "rules"/"llm"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, Literal
import os

# Soft imports (engines may not both be available during early dev)
try:
    from . import rules_engine as _rules
except Exception:  # pragma: no cover
    _rules = None  # type: ignore

try:
    from . import strategy_llm as _llm
except Exception:  # pragma: no cover
    _llm = None  # type: ignore

EngineName = Literal["rules", "llm", "auto"]
StrategyOutput = Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]


# ────────────────────────────────────────────────────────────
# Capability detection
# ────────────────────────────────────────────────────────────
def _has_llm_key() -> bool:
    """
    Detect if an LLM backend is plausibly usable.
    We keep the check broad so it works across providers/wrappers.
    """
    candidates = [
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",  # some setups use base + key
        "ANTHROPIC_API_KEY",
        "AZURE_OPENAI_API_KEY",
    ]
    return any(os.getenv(k) for k in candidates)


def engine_available(name: str) -> bool:
    """Best-effort probe."""
    n = (name or "").lower()
    if n == "rules":
        return _rules is not None
    if n == "llm":
        return _llm is not None and _has_llm_key()
    return False


def _env_truthy(var: str) -> bool:
    v = (os.getenv(var) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def default_engine() -> EngineName:
    """
    Decide a sensible default:
      - FORCE_RULES → "rules"
      - FORCE_LLM   → "llm" (only if available)
      - USE_LLM     → "llm" (only if available)
      - otherwise   → "rules"
    """
    if _env_truthy("FORCE_RULES"):
        return "rules"
    if _env_truthy("FORCE_LLM"):
        return "llm" if engine_available("llm") else "rules"

    # soft preference
    if _env_truthy("USE_LLM") and engine_available("llm"):
        return "llm"

    # if no hint but LLM is fully available, still default to rules for determinism
    return "rules"


# ────────────────────────────────────────────────────────────
# Facade API
# ────────────────────────────────────────────────────────────
def analyze_market(
    *,
    market_data: List[Dict[str, Any]],
    budget: float,
    market_type: str,
    history: Optional[Any] = None,
    sentiment: Optional[List[str]] = None,
    use_rsi: bool = False,
    use_sma: bool = False,
    use_sentiment: bool = False,
    market: Optional[str] = None,
    market_context: Optional[Dict[str, Any]] = None,
    engine: EngineName = "auto",
    **kwargs: Any,
) -> StrategyOutput:
    """
    Route the request to either rules or LLM strategy based on `engine`.

    engine:
      - "rules" → always use deterministic rules engine
      - "llm"   → always use LLM engine (raises if unavailable)
      - "auto"  → pick `default_engine()` at runtime
    """
    chosen: EngineName
    if engine == "auto":
        chosen = default_engine()
    else:
        chosen = engine

    if chosen == "llm":
        if not engine_available("llm"):
            raise RuntimeError("LLM engine requested but not available (missing key or module).")
        if _llm is None:
            raise RuntimeError("LLM engine module not importable.")
        return _llm.analyze_market(
            market_data=market_data,
            budget=budget,
            market_type=market_type,
            history=history,
            sentiment=sentiment,
            use_rsi=use_rsi,
            use_sma=use_sma,
            use_sentiment=use_sentiment,
            market=market,
            market_context=market_context or {},
            **kwargs,
        )

    # Fallback: rules
    if not engine_available("rules"):
        raise RuntimeError("Rules engine module not importable.")
    return _rules.analyze_market(
        market_data=market_data,
        budget=budget,
        market_type=market_type,
        history=history,
        sentiment=sentiment,
        use_rsi=use_rsi,
        use_sma=use_sma,
        use_sentiment=use_sentiment,
        market=market,
        market_context=market_context or {},
        **kwargs,
    )


__all__ = [
    "analyze_market",
    "engine_available",
    "default_engine",
]
