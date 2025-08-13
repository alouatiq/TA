"""
price_history.py
────────────────
Core OHLCV utilities used across indicators and strategies.

Goals
• Provide a single function to fetch normalized OHLCV for a symbol/asset type.
• Prefer TwelveData when TWELVEDATA_API_KEY is present; fall back to Yahoo (yfinance).
• For crypto daily history, optionally fall back to CoinGecko (best‑effort map).
• Offer resampling to multi-timeframes with proper OHLC aggregation.
• Provide helpers to align multiple timeframes and validate minimum history length.

Design
• get_price_history(...) → pandas.DataFrame with tz‑aware DatetimeIndex (UTC),
  columns: ["open","high","low","close","volume"].
• resample_ohlcv(df, timeframe) → same schema, sampled to '1m','5m','15m','1h','4h','1d','1w'.
• align_timeframes({"1h": df1h, "1d": df1d, ...}) → reindexed dict with common intersection.
• ensure_min_points(df, min_points) → returns df or None if insufficient data.

Notes
• This module is backend‑only. It can be reused by CLI and (later) the web API.
• We keep light provider logic here for convenience; in the final architecture,
  the dedicated adapters in trading_core/data_fetcher/adapters/ should back these calls.
• We do NOT raise on partial failures; callers can decide to skip indicators if data is short.

Dependencies
• pandas, numpy, requests, yfinance
"""

from __future__ import annotations

import os
import math
import time
import logging
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Public timeframe type
Timeframe = Literal["1m","5m","15m","30m","1h","4h","1d","1w"]

# ────────────────────────────────────────────────────────────
# Config & logger
# ────────────────────────────────────────────────────────────

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# TwelveData
_TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY", "").strip()
_TWELVEDATA_URL = "https://api.twelvedata.com/time_series"

# CoinGecko (crypto fallback for daily)
_COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# Minimal symbol→coingecko id map (extend as needed)
_CG_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "BNB": "binancecoin",
    "DOT": "polkadot",
    "MATIC": "matic-network",
    "LTC": "litecoin",
}

# yfinance interval map
_YF_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "4h": "60m",   # we’ll resample 1h → 4h locally
    "1d": "1d",
    "1w": "1wk",
}

# Reasonable lookbacks per timeframe (number of raw rows we aim to request)
_DEFAULT_LIMIT = {
    "1m": 1500,    # ~1–2 days (market dependent)
    "5m": 2000,    # ~1–2 weeks
    "15m": 2000,   # ~1–2 months
    "30m": 2000,   # ~3 months
    "1h": 2000,    # ~3–6 months
    "4h": 2000,    # resampled from 1h
    "1d": 1200,    # ~3–4 years
    "1w": 520,     # ~10 years
}

# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def get_price_history(
    symbol: str,
    asset_type: Literal["stock","crypto","forex","fund","warrant","future","commodity"] = "stock",
    *,
    timeframe: Timeframe = "1d",
    limit: Optional[int] = None,
    market: Optional[str] = None,
    currency: str = "USD",
    prefer: Literal["twelvedata","yahoo","auto"] = "auto",
) -> Optional[pd.DataFrame]:
    """
    Fetch normalized OHLCV for a symbol and asset type.

    Parameters
    ----------
    symbol : str
        Provider-friendly symbol (Yahoo format works well for stocks/funds/warrants/futures).
        For FX (TwelveData): "EUR/USD"; for yfinance FX: "EURUSD=X".
        For crypto: "BTC-USD" for yfinance; "BTC" (CoinGecko map) for CG fallback.
    asset_type : {'stock','crypto','forex','fund','warrant','future','commodity'}
        Used to pick the most appropriate provider and symbol normalization.
    timeframe : Timeframe
        One of '1m','5m','15m','30m','1h','4h','1d','1w'.
    limit : int | None
        Approximate number of points desired. If None, use sensible defaults.
    market : str | None
        Optional market/exchange hint (e.g., 'LSE','XETRA'). Currently advisory.
    currency : str
        Pricing currency for providers that require it (TD & CG). Default 'USD'.
    prefer : {'twelvedata','yahoo','auto'}
        Data source preference. 'auto' picks TwelveData when key exists, otherwise Yahoo.

    Returns
    -------
    pandas.DataFrame | None
        tz-aware (UTC) OHLCV DataFrame, or None if all providers fail.

    Notes
    -----
    • We request the smallest feasible interval upstream (e.g., 1h for 4h) and resample locally.
    • Volume units differ by asset type (shares, coins, contracts); we pass through raw volume.
    """
    tf = timeframe
    lim = limit or _DEFAULT_LIMIT.get(tf, 1000)

    # Decide source
    use_td = (_TWELVEDATA_KEY != "") if prefer == "auto" else (prefer == "twelvedata")

    # 1) Try TwelveData when available
    if use_td:
        try:
            df = _fetch_twelvedata(symbol, asset_type, tf, lim, currency=currency, market=market)
            if df is not None and not df.empty:
                if tf == "4h":  # resample 1h → 4h
                    df = resample_ohlcv(df, "4h")
                return df
        except Exception as e:
            LOG.warning("TwelveData fetch failed for %s (%s): %s", symbol, asset_type, str(e))

    # 2) Try Yahoo Finance
    try:
        df = _fetch_yahoo(symbol, asset_type, tf, lim)
        if df is not None and not df.empty:
            if tf == "4h" and "1h" in _YF_INTERVAL:  # if we pulled 60m for 1h, resample to 4h
                if df.index.freqstr != "H":
                    # even if not an hourly freq, we can still resample on the datetime index
                    pass
                df = resample_ohlcv(df, "4h")
            return df
    except Exception as e:
        LOG.warning("Yahoo fetch failed for %s (%s): %s", symbol, asset_type, str(e))

    # 3) Crypto-only fallback: CoinGecko (daily only)
    if asset_type == "crypto" and tf in ("1d","1w"):
        try:
            df = _fetch_coingecko_daily(symbol, currency=currency, bars=lim)
            if df is not None and not df.empty:
                if tf == "1w":
                    df = resample_ohlcv(df, "1w")
                return df
        except Exception as e:
            LOG.warning("CoinGecko fallback failed for %s: %s", symbol, str(e))

    return None


def resample_ohlcv(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to another timeframe using proper OHLC rules.
    Index must be tz-aware or naive timestamps (interpreted as UTC).

    Rules:
    • open  = first
    • high  = max
    • low   = min
    • close = last
    • volume = sum
    """
    if df is None or df.empty:
        return df

    # Ensure UTC tz
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")

    rule = _to_pandas_rule(timeframe)
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        df
        .resample(rule, label="right", closed="right")
        .agg(ohlc)
        .dropna(how="any")
    )
    return out


def align_timeframes(dfs: Dict[Timeframe, pd.DataFrame]) -> Dict[Timeframe, pd.DataFrame]:
    """
    Align multiple timeframe DataFrames to the common intersection of timestamps.
    Returns a dict with reindexed copies; DataFrames may become empty if no overlap exists.
    """
    if not dfs:
        return dfs

    # Ensure tz and drop empties
    norm: Dict[Timeframe, pd.DataFrame] = {}
    for k, df in dfs.items():
        if df is None or df.empty:
            continue
        if df.index.tz is None:
            df = df.tz_localize("UTC")
        else:
            df = df.tz_convert("UTC")
        norm[k] = df

    if not norm:
        return {}

    # Intersection of all indices
    idx = None
    for df in norm.values():
        idx = df.index if idx is None else idx.intersection(df.index)

    if idx is None or len(idx) == 0:
        return {k: df.iloc[0:0] for k, df in norm.items()}  # empty copies

    aligned = {k: df.reindex(idx).dropna(how="any") for k, df in norm.items()}
    return aligned


def ensure_min_points(df: Optional[pd.DataFrame], min_points: int) -> Optional[pd.DataFrame]:
    """
    Return df if it has at least `min_points` rows, otherwise None.
    """
    if df is None or df.empty:
        return None
    return df if len(df) >= max(1, min_points) else None


# ────────────────────────────────────────────────────────────
# Provider helpers
# ────────────────────────────────────────────────────────────

def _fetch_twelvedata(
    symbol: str,
    asset_type: str,
    timeframe: Timeframe,
    limit: int,
    *,
    currency: str = "USD",
    market: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    TwelveData normalized fetch.
    Symbol format:
      • Stocks/Funds/Warrants/Futures: Yahoo-like symbols often work. No exchange param by default.
      • Forex: 'EUR/USD', 'USD/JPY', ...
      • Crypto: 'BTC/USD', 'ETH/USD'. (You can pass pairs like 'BTC/USDT', TwelveData supports many.)
    """
    if not _TWELVEDATA_KEY:
        return None

    td_interval = _to_twelvedata_interval(timeframe)
    # TwelveData has a max outputsize per request. We'll just request `limit` and let TD clamp it.
    params = {
        "symbol": _normalize_td_symbol(symbol, asset_type, currency),
        "interval": td_interval,
        "outputsize": str(limit),
        "timezone": "UTC",
        "format": "JSON",
        "apikey": _TWELVEDATA_KEY,
    }
    r = requests.get(_TWELVEDATA_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Handle errors gracefully
    if isinstance(data, dict) and "status" in data and data.get("status") == "error":
        # Sometimes TD uses {"code":..., "message":...}
        raise RuntimeError(f"TwelveData error: {data.get('message') or data}")

    # Time series is under "values"
    values = data.get("values") if isinstance(data, dict) else None
    if not values:
        return None

    df = pd.DataFrame(values)
    # TD columns are strings; ensure required columns exist
    for c in ("open","high","low","close","volume","datetime"):
        if c not in df.columns:
            return None

    # Convert types
    _num = lambda x: pd.to_numeric(x, errors="coerce")
    df["open"] = _num(df["open"])
    df["high"] = _num(df["high"])
    df["low"] = _num(df["low"])
    df["close"] = _num(df["close"])
    df["volume"] = _num(df["volume"])
    # TD returns reverse-chronological; sort ascending
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")
    df = df[["open","high","low","close","volume"]].dropna(how="any")
    return df


def _fetch_yahoo(
    symbol: str,
    asset_type: str,
    timeframe: Timeframe,
    limit: int,
) -> Optional[pd.DataFrame]:
    """
    yfinance normalized fetch.
    For 4h we fetch 60m and resample locally.
    For FX on yfinance use 'EURUSD=X' format (callers can pass this directly).
    For crypto on yfinance use 'BTC-USD' format (callers can pass this directly).
    """
    yf_interval = _YF_INTERVAL[timeframe]
    fetch_symbol = symbol

    # If the user gave an FX pair in TD-style "EUR/USD", map to yfinance "EURUSD=X"
    if asset_type == "forex" and "/" in symbol and "=X" not in symbol:
        p = symbol.replace("/", "")
        fetch_symbol = f"{p}=X"

    # If crypto passed as plain ticker (e.g., 'BTC'), try 'BTC-USD'
    if asset_type == "crypto" and "-" not in symbol and "=X" not in symbol and "/" not in symbol:
        fetch_symbol = f"{symbol.upper()}-USD"

    # yfinance: period must fit the interval; we’ll request max period, then tail(limit)
    # Map timeframe to period
    period = _yf_period_for_interval(yf_interval, limit)

    tkr = yf.Ticker(fetch_symbol)
    hist = tkr.history(period=period, interval=yf_interval, auto_adjust=False)
    if hist is None or hist.empty:
        return None

    # Normalize columns
    out = pd.DataFrame({
        "open": hist["Open"].astype(float),
        "high": hist["High"].astype(float),
        "low": hist["Low"].astype(float),
        "close": hist["Close"].astype(float),
        "volume": hist.get("Volume", pd.Series(index=hist.index, dtype=float)).fillna(0).astype(float),
    })
    # Ensure tz-aware UTC
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")

    out = out.dropna(how="any").sort_index()
    if len(out) > limit:
        out = out.tail(limit)
    # For '4h', we fetched 60m; we’ll resample outside if requested by caller
    return out


def _fetch_coingecko_daily(symbol: str, currency: str = "USD", bars: int = 365) -> Optional[pd.DataFrame]:
    """
    CoinGecko simple daily OHLCV fallback for crypto. Requires a CoinGecko ID.
    We use a small static map for popular assets; otherwise, try to search quickly.
    """
    cg_id = _CG_MAP.get(symbol.upper())
    if not cg_id:
        # quick search (best-effort)
        try:
            s = requests.get(f"{_COINGECKO_BASE}/search", params={"query": symbol}, timeout=12).json()
            coins = s.get("coins", [])
            if coins:
                cg_id = coins[0]["id"]
        except Exception:
            pass
    if not cg_id:
        return None

    # days param is rounded up; for bars daily, request somewhat larger window
    days = max(30, int(bars * 1.2))
    mc = requests.get(
        f"{_COINGECKO_BASE}/coins/{cg_id}/market_chart",
        params={"vs_currency": currency.lower(), "days": days, "interval": "daily"},
        timeout=20,
    ).json()

    prices = mc.get("prices") or []
    volumes = mc.get("total_volumes") or []

    if not prices:
        return None

    # Build OHLC from daily closes (CG daily OHLC endpoint is paid; we approximate using ranges if available)
    # coingecko market_chart also has "market_caps" and sometimes "high_24h"/"low_24h" not per day; so we approximate
    # by using close-only series and deriving H/L with small padding — acceptable for indicator fallback.
    ts = [pd.to_datetime(p[0], unit="ms", utc=True) for p in prices]
    close = pd.Series([float(p[1]) for p in prices], index=ts, name="close")

    vol_map = {pd.to_datetime(v[0], unit="ms", utc=True): float(v[1]) for v in volumes}
    volume = close.index.to_series().map(vol_map).fillna(0.0)

    # Approximate OHLC as: open=prev_close, high=max(prev_close, close), low=min(prev_close, close)
    open_ = close.shift(1).fillna(close)
    high_ = np.maximum(open_.values, close.values)
    low_  = np.minimum(open_.values, close.values)

    df = pd.DataFrame(
        {"open": open_.values, "high": high_, "low": low_, "close": close.values, "volume": volume.values},
        index=close.index,
    ).sort_index()

    if len(df) > bars:
        df = df.tail(bars)
    return df.dropna(how="any")


# ────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────

def _to_pandas_rule(tf: Timeframe) -> str:
    if tf == "1m": return "1min"
    if tf == "5m": return "5min"
    if tf == "15m": return "15min"
    if tf == "30m": return "30min"
    if tf == "1h": return "1H"
    if tf == "4h": return "4H"
    if tf == "1d": return "1D"
    if tf == "1w": return "1W"
    raise ValueError(f"Unsupported timeframe: {tf}")


def _to_twelvedata_interval(tf: Timeframe) -> str:
    # TwelveData interval strings largely match ours, except '4h' is '4h'
    return tf


def _normalize_td_symbol(symbol: str, asset_type: str, currency: str) -> str:
    """
    Convert common forms to TwelveData expected forms where needed.
    • FX: 'EURUSD=X' → 'EUR/USD'
    • Crypto: 'BTC-USD' or 'BTCUSD' → 'BTC/USD'
    Otherwise return as-is.
    """
    s = symbol.strip().upper()
    if asset_type == "forex":
        if s.endswith("=X") and len(s) >= 7:
            return f"{s[:3]}/{s[3:6]}"
        if "/" not in s and len(s) >= 6:
            return f"{s[:3]}/{s[3:6]}"
        return s
    if asset_type == "crypto":
        if "-" in s:
            base, quote = s.split("-", 1)
            return f"{base}/{quote}"
        if s.endswith("USD"):
            return f"{s[:-3]}/USD"
        if "/" not in s:
            return f"{s}/{currency.upper()}"
        return s
    return symbol


def _yf_period_for_interval(interval: str, limit: int) -> str:
    """
    Choose a yfinance period that likely contains `limit` bars for the chosen interval.
    yfinance constraints (approx):
      - 1m: last 7 days
      - 2m/5m/15m/30m/60m: last 60 days
      - 1d:  max
      - 1wk: max
    We'll pick conservatively.
    """
    if interval == "1m":
        return "7d"
    if interval in ("2m","5m","15m","30m","60m"):
        return "60d"
    if interval in ("1d","1wk","1mo","3mo"):
        return "max"
    # Fallback
    return "max"
