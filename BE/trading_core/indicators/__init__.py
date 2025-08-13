"""
trading_core.indicators
=======================

Unified indicator API for the Trading Assistant.

This package exposes a clean, stable interface that higher layers (CLI, strategy
engines, scoring) can call without worrying about *where* each indicator is
implemented. Internally, it groups indicators into four domains:

1. Technical Indicators — trend (SMA, EMA, MACD, ADX), momentum (RSI, Stochastic),
   volume (OBV, volume spikes), volatility (Bollinger Bands, ATR).
2. Fundamentals — stocks (earnings, revenue growth, P/E, debt/leverage),
   crypto (developer activity, network usage, tokenomics, on-chain proxies).
3. Sentiment — normalized news/social scores, Fear & Greed index.
4. Market Microstructure — order book imbalance, depth, whale flow (when data is available).

Design goals
------------
- Stable, minimal imports for the public API (lazy import heavy modules).
- Clear typing & dataclass-friendly outputs.
- Backward-compatible helpers for your current code (`calculate_rsi`, `calculate_sma`).
- Multi-timeframe-ready: helpers accept OHLCV frames with datetime index.

Public API (re-exported)
------------------------
- Technical:
    sma, ema, rsi, macd, adx, stochastic, obv, bollinger_bands, atr, detect_volume_spike
- Fundamentals:
    stock_fundamentals, crypto_fundamentals
- Sentiment:
    news_sentiment_score, social_sentiment_score, fear_greed_index
- Microstructure:
    orderbook_imbalance, liquidity_depth, whale_activity

Utility:
- ensure_ohlcv(df), resample_ohlcv(df, rule)
- compute_indicator_bundle(…) → dict       # one-shot technical bundle
- compute_full_signal_set(…) → dict        # technical + sentiment + fundamentals + microstructure
- calculate_rsi / calculate_sma             # Back-compat shims for existing code

Notes
-----
Actual implementations live in sibling modules:
    price_history.py, technical.py, fundamentals.py, sentiment.py, microstructure.py
This __init__ wires them together and offers safe fallbacks where appropriate.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

# We *lazily import* heavy modules inside the functions to keep CLI startup fast and
# to make partial deployments (without every dependency) degrade gracefully.

__all__ = [
    # Technical re-exports
    "sma", "ema", "rsi", "macd", "adx", "stochastic",
    "obv", "bollinger_bands", "atr", "detect_volume_spike",
    # Fundamentals re-exports
    "stock_fundamentals", "crypto_fundamentals",
    # Sentiment re-exports
    "news_sentiment_score", "social_sentiment_score", "fear_greed_index",
    # Microstructure re-exports
    "orderbook_imbalance", "liquidity_depth", "whale_activity",
    # Utilities
    "ensure_ohlcv", "resample_ohlcv",
    "compute_indicator_bundle", "compute_full_signal_set",
    # Compatibility shims
    "calculate_rsi", "calculate_sma",
]

# -----------------------------------------------------------------------------
# Internal: Lazy import helpers
# -----------------------------------------------------------------------------

def _technical():
    from . import technical as _t
    return _t

def _fundamentals():
    from . import fundamentals as _f
    return _f

def _sentiment():
    from . import sentiment as _s
    return _s

def _microstructure():
    from . import microstructure as _m
    return _m

def _ph():
    from . import price_history as _phm
    return _phm


# -----------------------------------------------------------------------------
# Public Technical: thin re-exports
# -----------------------------------------------------------------------------

def sma(series_or_df, window: int = 14, price_col: str = "Close"):
    """Simple Moving Average."""
    return _technical().sma(series_or_df, window=window, price_col=price_col)

def ema(series_or_df, window: int = 14, price_col: str = "Close"):
    """Exponential Moving Average."""
    return _technical().ema(series_or_df, window=window, price_col=price_col)

def rsi(series_or_df, window: int = 14, price_col: str = "Close"):
    """Relative Strength Index."""
    return _technical().rsi(series_or_df, window=window, price_col=price_col)

def macd(series_or_df, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = "Close"):
    """MACD line, signal, histogram."""
    return _technical().macd(series_or_df, fast=fast, slow=slow, signal=signal, price_col=price_col)

def adx(df, window: int = 14):
    """Average Directional Index (expects OHLC)."""
    return _technical().adx(df, window=window)

def stochastic(df, k: int = 14, d: int = 3, smooth_k: int = 3):
    """Stochastic Oscillator %K/%D (expects OHLC)."""
    return _technical().stochastic(df, k=k, d=d, smooth_k=smooth_k)

def obv(df, price_col: str = "Close", volume_col: str = "Volume"):
    """On-Balance Volume (expects price + volume)."""
    return _technical().obv(df, price_col=price_col, volume_col=volume_col)

def bollinger_bands(series_or_df, window: int = 20, n_std: float = 2.0, price_col: str = "Close"):
    """Bollinger Bands (middle, upper, lower)."""
    return _technical().bollinger_bands(series_or_df, window=window, n_std=n_std, price_col=price_col)

def atr(df, window: int = 14):
    """Average True Range (expects OHLC)."""
    return _technical().atr(df, window=window)

def detect_volume_spike(df, window: int = 20, volume_col: str = "Volume", threshold_sigma: float = 2.0):
    """Flag bars where volume exceeds mean + N*std over lookback window."""
    return _technical().detect_volume_spike(df, window=window, volume_col=volume_col, threshold_sigma=threshold_sigma)


# -----------------------------------------------------------------------------
# Public Fundamentals
# -----------------------------------------------------------------------------

def stock_fundamentals(symbol: str, *, provider_hint: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a dictionary of normalized stock fundamentals:
        {
          'pe': float|None, 'ps': float|None, 'pb': float|None,
          'eps_growth_yoy': float|None, 'revenue_growth_yoy': float|None,
          'debt_to_equity': float|None, 'dividend_yield': float|None,
          'currency': 'USD'|'EUR'|...,
          'source': 'yahoo'|'twelvedata'|...
        }
    """
    return _fundamentals().stock_fundamentals(symbol, provider_hint=provider_hint)

def crypto_fundamentals(asset_id_or_symbol: str) -> Dict[str, Any]:
    """
    Return a dictionary of normalized crypto fundamentals/on-chain proxies:
        {
          'dev_activity': float|None, 'tx_activity': float|None,
          'real_volume_score': float|None, 'supply_inflation': float|None,
          'tokenomics_flag': str|None,
          'source': 'coingecko'|'cryptocompare'|...
        }
    """
    return _fundamentals().crypto_fundamentals(asset_id_or_symbol)


# -----------------------------------------------------------------------------
# Public Sentiment
# -----------------------------------------------------------------------------

def news_sentiment_score(headlines: Iterable[str]) -> Dict[str, Any]:
    """
    Score news headlines into a compact dict:
        {'score': -1..+1, 'confidence': 0..1, 'n': int, 'method': 'vader'|'llm'|...}
    """
    return _sentiment().news_sentiment_score(headlines)

def social_sentiment_score(messages: Iterable[str]) -> Dict[str, Any]:
    """
    Score social messages (tweets, reddit) into a compact dict:
        {'score': -1..+1, 'confidence': 0..1, 'n': int, 'method': 'vader'|'llm'|...}
    """
    return _sentiment().social_sentiment_score(messages)

def fear_greed_index(asset_class: str = "crypto") -> Dict[str, Any]:
    """
    Fetch a normalized Fear & Greed index for the asset class (if available):
        {'score': 0..100, 'label': 'Extreme Fear'|'Fear'|'Neutral'|'Greed'|'Extreme Greed', 'source': 'altfg'|...}
    """
    return _sentiment().fear_greed_index(asset_class=asset_class)


# -----------------------------------------------------------------------------
# Public Microstructure
# -----------------------------------------------------------------------------

def orderbook_imbalance(orderbook_levels: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Compute order book imbalance given aggregated top-of-book (or depth) snapshot:
        input example: {'bids': [(price, qty), ...], 'asks': [(price, qty), ...]}
        returns: {'imbalance': -1..+1, 'bid_vol': float, 'ask_vol': float}
    """
    return _microstructure().orderbook_imbalance(orderbook_levels)

def liquidity_depth(orderbook_levels: Mapping[str, Any], pct_away: float = 0.5) -> Dict[str, Any]:
    """
    Estimate liquidity depth within a band around mid-price (e.g., ±0.5%):
        returns: {'depth_buy': float, 'depth_sell': float, 'mid': float}
    """
    return _microstructure().liquidity_depth(orderbook_levels, pct_away=pct_away)

def whale_activity(trades: Iterable[Mapping[str, Any]], notional_threshold: float) -> Dict[str, Any]:
    """
    Scan recent prints for whale-sized trades:
        returns: {'count': int, 'net_flow': float, 'buy_count': int, 'sell_count': int}
    """
    return _microstructure().whale_activity(trades, notional_threshold)


# -----------------------------------------------------------------------------
# Utilities & Bundles
# -----------------------------------------------------------------------------

def ensure_ohlcv(df):
    """
    Validate an OHLCV DataFrame. Ensures at least ['Open','High','Low','Close'] exist.
    Delegates to price_history.ensure_ohlcv for consistent behavior.
    """
    return _ph().ensure_ohlcv(df)

def resample_ohlcv(df, rule: str = "1D", how: str = "ohlc"):
    """
    Resample an OHLCV DataFrame to a different timeframe (e.g., '1H', '4H', '1D').
    Behavior is centralized in price_history to keep semantics identical.
    """
    return _ph().resample_ohlcv(df, rule=rule, how=how)


def compute_indicator_bundle(
    df,
    *,
    price_col: str = "Close",
    volume_col: str = "Volume",
    trend_windows = (20, 50, 200),
    rsi_window: int = 14,
    bb_window: int = 20,
    bb_n_std: float = 2.0,
    atr_window: int = 14,
) -> Dict[str, Any]:
    """
    Convenience function that computes a compact set of core technical indicators
    suitable for regime detection & scoring.

    Returns e.g.:
    {
      'sma_20': Series, 'sma_50': Series, 'sma_200': Series,
      'ema_20': Series,
      'rsi_14': Series,
      'macd': DataFrame[macd, signal, hist],
      'adx_14': Series,
      'stoch': DataFrame[k, d],
      'obv': Series,
      'bbands': DataFrame[middle, upper, lower],
      'atr_14': Series,
      'volume_spike': Series[bool]
    }
    """
    t = _technical()
    df2 = ensure_ohlcv(df)

    out: Dict[str, Any] = {}

    # Trend — SMA/EMA over common windows
    for w in trend_windows:
        out[f"sma_{w}"] = t.sma(df2, window=w, price_col=price_col)
    out["ema_20"] = t.ema(df2, window=20, price_col=price_col)

    # Momentum
    out[f"rsi_{rsi_window}"] = t.rsi(df2, window=rsi_window, price_col=price_col)
    out["macd"] = t.macd(df2, fast=12, slow=26, signal=9, price_col=price_col)
    out["adx_14"] = t.adx(df2, window=14)
    out["stoch"] = t.stochastic(df2, k=14, d=3, smooth_k=3)

    # Volume
    out["obv"] = t.obv(df2, price_col=price_col, volume_col=volume_col)
    out["volume_spike"] = t.detect_volume_spike(df2, window=20, volume_col=volume_col, threshold_sigma=2.0)

    # Volatility
    out["bbands"] = t.bollinger_bands(df2, window=bb_window, n_std=bb_n_std, price_col=price_col)
    out["atr_14"] = t.atr(df2, window=atr_window)

    return out


def compute_full_signal_set(
    *,
    df=None,
    symbol: Optional[str] = None,
    asset_class: str = "equity",  # 'equity' | 'crypto' | 'forex' | ...
    headlines: Optional[Iterable[str]] = None,
    social_msgs: Optional[Iterable[str]] = None,
    orderbook: Optional[Mapping[str, Any]] = None,
    trades: Optional[Iterable[Mapping[str, Any]]] = None,
    whale_notional_threshold: float = 250_000.0,
) -> Dict[str, Any]:
    """
    One-call signal aggregator:
      - Technical bundle from OHLCV (if df provided)
      - Fundamentals (if symbol provided)
      - Sentiment (if headlines/social provided + F&G for crypto)
      - Microstructure (if orderbook/trades provided)

    Returns a dictionary with sub-dicts:
      {
        'technical': {...}, 'fundamentals': {...},
        'sentiment': {...}, 'micro': {...},
        'meta': {'asset_class': str, 'symbol': str|None}
      }
    """
    out: Dict[str, Any] = {"technical": {}, "fundamentals": {}, "sentiment": {}, "micro": {}, "meta": {"asset_class": asset_class, "symbol": symbol}}

    # Technical
    if df is not None:
        out["technical"] = compute_indicator_bundle(df)

    # Fundamentals
    if symbol:
        if asset_class == "equity":
            out["fundamentals"] = stock_fundamentals(symbol)
        elif asset_class == "crypto":
            out["fundamentals"] = crypto_fundamentals(symbol)
        else:
            out["fundamentals"] = {}

    # Sentiment
    s_mod = _sentiment()
    sent: Dict[str, Any] = {}
    if headlines:
        sent["news"] = s_mod.news_sentiment_score(headlines)
    if social_msgs:
        sent["social"] = s_mod.social_sentiment_score(social_msgs)
    if asset_class == "crypto":
        sent["fear_greed"] = s_mod.fear_greed_index(asset_class="crypto")
    out["sentiment"] = sent

    # Microstructure
    m_mod = _microstructure()
    micro: Dict[str, Any] = {}
    if orderbook:
        micro["book"] = m_mod.orderbook_imbalance(orderbook)
        micro["depth"] = m_mod.liquidity_depth(orderbook, pct_away=0.5)
    if trades:
        micro["whales"] = m_mod.whale_activity(trades, notional_threshold=whale_notional_threshold)
    out["micro"] = micro

    return out


# -----------------------------------------------------------------------------
# Backward-compatibility shims (existing code used calculate_rsi / calculate_sma)
# -----------------------------------------------------------------------------

def calculate_rsi(price_history) -> Optional[float]:
    """
    Compatibility: accept a list/array of closes and compute the *latest* RSI.
    This mirrors the existing code path in your project.
    """
    t = _technical()
    try:
        import pandas as pd
        ser = pd.Series(price_history)
        r = t.rsi(ser, window=14)
        # If a Series is returned, take the last value.
        return float(r.iloc[-1]) if hasattr(r, "iloc") else float(r)
    except Exception:
        return None

def calculate_sma(price_history) -> Optional[float]:
    """
    Compatibility: accept a list/array of closes and compute the *latest* SMA(14).
    """
    t = _technical()
    try:
        import pandas as pd
        ser = pd.Series(price_history)
        r = t.sma(ser, window=14)
        return float(r.iloc[-1]) if hasattr(r, "iloc") else float(r)
    except Exception:
        return None
