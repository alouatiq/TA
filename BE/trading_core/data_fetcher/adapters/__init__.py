# BE/trading_core/data_fetcher/__init__.py
"""
Unified data fetcher API.

This package exposes a stable facade over individual category fetchers
(equities, crypto, forex, commodities, futures, warrants, funds) and
their provider adapters (Yahoo, Stooq, CoinGecko, TwelveData, etc.).

Goals
-----
- Keep import sites simple: `from trading_core.data_fetcher import fetch_equities_data`
- Centralize backend selection hints (e.g., prefer TwelveData if API key present)
- Provide light, optional diagnostics accessors so the CLI can print
  "used/failed/skipped sources" without coupling to module internals.

Notes
-----
Each category module is responsible for:
  • Discovery (universe for a market/region)
  • Quotes + optional price history (OHLCV)
  • Setting its own diagnostics:
       LAST_*_SOURCE, FAILED_*_SOURCES, SKIPPED_*_SOURCES

This __init__ only aggregates and re-exports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

# Re-export public fetch functions (category modules must define these)
from .equities import fetch_equities_data  # noqa: F401
from .crypto import fetch_crypto_data      # noqa: F401
from .forex import fetch_forex_data        # noqa: F401
from .commodities import fetch_commodities_data  # noqa: F401
from .futures import fetch_futures_data    # noqa: F401
from .warrants import fetch_warrants_data  # noqa: F401
from .funds import fetch_funds_data        # noqa: F401

# Type aliases used across callers
AssetRow = Dict[str, Any]
AssetRows = List[AssetRow]


# ────────────────────────────────────────────────────────────
# Backend preference helpers
# ────────────────────────────────────────────────────────────
def prefer_twelvedata() -> bool:
    """
    Return True if TwelveData should be preferred for quotes/history.

    Category modules may call this to decide their primary path.
    We don't enforce – it's only a hint – so modules remain free to
    pick what’s best per symbol/region.
    """
    key = os.getenv("TWELVEDATA_API_KEY", "") or os.getenv("TWELVE_DATA_API_KEY", "")
    # Allow explicit override for testing:
    force = os.getenv("FORCE_TWELVEDATA", "").strip().lower() in {"1", "true", "yes"}
    if force:
        return True
    return bool(key)


def available_quote_backends() -> List[str]:
    """
    Introspect which HTTP providers are realistically usable in this process.
    (Purely advisory; fetchers decide their own fallback order.)
    """
    backends = ["yahoo", "stooq", "coingecko", "coincap", "coinpaprika"]
    if os.getenv("TWELVEDATA_API_KEY") or os.getenv("TWELVE_DATA_API_KEY"):
        backends.insert(0, "twelvedata")
    if os.getenv("CRYPTOCOMPARE_API_KEY"):
        backends.append("cryptocompare")
    return backends


# ────────────────────────────────────────────────────────────
# Diagnostics aggregation (optional)
# ────────────────────────────────────────────────────────────
def diagnostics_for(category: str) -> Dict[str, Any]:
    """
    Return a {used, failed, skipped} dict for a given category, when available.

    Examples:
        diagnostics_for("equities")
        diagnostics_for("crypto")

    If a category doesn't expose diagnostics, returns empty defaults.
    """
    cat = (category or "").strip().lower()
    mod = None
    try:
        if cat == "equities":
            from . import equities as mod  # type: ignore
        elif cat == "crypto":
            from . import crypto as mod  # type: ignore
        elif cat == "forex":
            from . import forex as mod  # type: ignore
        elif cat == "commodities":
            from . import commodities as mod  # type: ignore
        elif cat == "futures":
            from . import futures as mod  # type: ignore
        elif cat == "warrants":
            from . import warrants as mod  # type: ignore
        elif cat == "funds":
            from . import funds as mod  # type: ignore
    except Exception:
        mod = None

    if not mod:
        return {"used": None, "failed": [], "skipped": []}

    def _get(name: str, default):
        try:
            return getattr(mod, name)
        except Exception:
            return default

    used = _get(f"LAST_{cat.upper()}_SOURCE", None)
    failed = _get(f"FAILED_{cat.upper()}_SOURCES", [])
    skipped = _get(f"SKIPPED_{cat.upper()}_SOURCES", [])
    return {"used": used, "failed": failed, "skipped": skipped}


# ────────────────────────────────────────────────────────────
# Convenience: single-symbol quote fetch
# (optional utility used by single-asset analysis flow)
# ────────────────────────────────────────────────────────────
def fetch_single_symbol_quote(symbol: str, *, asset_class: str = "equity",
                              include_history: bool = True,
                              market: Optional[str] = None) -> Optional[AssetRow]:
    """
    Minimal convenience wrapper used by the single-asset analysis path in the CLI.

    It dispatches to the closest category module:
      - asset_class in {"equity","stock"}   -> equities.fetch_equities_data(symbols=[...])
      - asset_class in {"crypto","coin"}    -> crypto.fetch_crypto_data(symbols=[...])
      - asset_class in {"forex","fx"}       -> forex.fetch_forex_data(pairs=[...])
      - otherwise                           -> tries equities as a best-effort default

    Returns a single AssetRow or None.
    """
    ac = (asset_class or "").strip().lower()
    try:
        if ac in {"equity", "stock"}:
            from .equities import fetch_equities_data as _fe
            rows = _fe(include_history=include_history, market=market, symbols=[symbol])
        elif ac in {"crypto", "coin"}:
            from .crypto import fetch_crypto_data as _fc
            rows = _fc(include_history=include_history, symbols=[symbol])
        elif ac in {"forex", "fx"}:
            from .forex import fetch_forex_data as _ffx
            # Accept both "EURUSD" and "EUR/USD"
            norm = symbol.replace("/", "").upper()
            rows = _ffx(include_history=include_history, pairs=[norm])
        else:
            from .equities import fetch_equities_data as _fe
            rows = _fe(include_history=include_history, market=market, symbols=[symbol])
    except Exception:
        rows = []

    return rows[0] if rows else None


__all__ = [
    # category fetchers
    "fetch_equities_data",
    "fetch_crypto_data",
    "fetch_forex_data",
    "fetch_commodities_data",
    "fetch_futures_data",
    "fetch_warrants_data",
    "fetch_funds_data",
    # helpers
    "prefer_twelvedata",
    "available_quote_backends",
    "diagnostics_for",
    "fetch_single_symbol_quote",
    # common types
    "AssetRow",
    "AssetRows",
]
