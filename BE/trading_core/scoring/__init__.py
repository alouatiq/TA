"""
Signal fusion → recommendation

Exports a small, stable surface the rest of the app can call:

- detect_regime(...)              -> str ("bull"|"bear"|"range"|"volatile")
- suggest_stops_and_size(...)     -> dict with stop_loss, take_profit, position sizing
- combine_signals(...)            -> dict with action/confidence and scored components
- explain_recommendation(...)     -> human-readable reasons list / string
"""

from .regime import detect_regime, regime_features
from .risk import suggest_stops_and_size, volatility_target_position, portfolio_correlation_matrix
from .weights import combine_signals, default_weights_for_regime
from .explainer import explain_recommendation

__all__ = [
    "detect_regime",
    "regime_features",
    "suggest_stops_and_size",
    "volatility_target_position",
    "portfolio_correlation_matrix",
    "combine_signals",
    "default_weights_for_regime",
    "explain_recommendation",
]
