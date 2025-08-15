"""
trading_core
────────────
Core backend package for the Trading Assistant.

This namespace exposes stable helpers used across CLI and (future) web API:
• config: market calendars, regions, tz helpers, API keys
• data_fetcher: category fetchers (equities, crypto, forex, etc.)
• indicators: calculation entrypoints (technical, fundamentals, sentiment, microstructure)
• scoring: regime detection, risk, weights, explanation
• strategy: rules engine + LLM wrapper
• persistence: history tracking, performance evaluation
• utils: logging, IO, timezones

Import conveniences:
    from trading_core.config import (
        get_market_info, sessions_today, is_market_open,
        get_region_for_market, load_api_keys
    )
"""

from __future__ import annotations

# Re-export the most commonly used helpers so callers can `from trading_core import X`
from .config import (
    get_market_info,
    sessions_today,
    is_market_open,
    get_region_for_market,
    load_api_keys,
)

__all__ = [
    "get_market_info",
    "sessions_today",
    "is_market_open",
    "get_region_for_market",
    "load_api_keys",
]
