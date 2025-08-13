"""
price_history.py
────────────────
Helpers for fetching, normalizing, and resampling OHLCV price history used by
indicator modules. This module stays *provider‑agnostic* by allowing you to pass
in a `fetch_fn` (adapter) that returns a pandas DataFrame of OHLCV. If you don’t
provide one, a safe yfinance-based fallback is used for symbols that yfinance
can resolve (equities/ETFs, many crypto tickers via Yahoo’s synthetic pairs).

Design goals
------------
1) Single, consistent OHLCV shape:
      index: tz-aware datetime (UTC)
      columns: ["open","high","low","close","volume"]  (lower‑case, float volume)
2) Minimal assumptions: we handle a variety of provider shapes/column names.
3) Durable resampling: intraday↔daily with gaps handled and strict sorting.
4) Multi-timeframe prep: helper to request ≥N bars (for indicators like RSI-14).

Typical usage
-------------
from trading_core.indicators.price_history import (
    get_price_history,
    resample_ohlcv,
)

# Fetch 90 daily bars for an equity via default yfinance adapter:
df = get_price_history("AAPL", timeframe="1d", bars=90)

# Resample those 1m bars to 15m:
df_15m = resample_ohlcv(df_1m, rule="15min")

# Use a custom adapter (e.g., TwelveData) that matches the signature below:
def twelvedata_fetcher(symbol: str, interval: str, start: Optional[datetime], end: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
    ...
df = get_price_history("EURUSD", timeframe="1h", bars=300, fetch_fn=twelvedata_fetcher)

Notes on intervals
------------------
Input `timeframe` uses compact codes:
  - minutes: "1m","5m","15m","30m"
  - hours:   "1h","4h"
  - days:    "1d","1wk","1mo" (weekly/monthly are respected if adapter supports)

For pandas resample rules, we expose a separate `rule` parameter (e.g., "15min",
"4H", "1D") in `resample_ohlcv`. Don’t mix those two; `timeframe` targets your
data provider, `rule` targets pandas.

This file includes a pragmatic yfinance fallback for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # optional; only used by fallback adapter
except Exception:  # pragma: no cover
    yf = None  # type: ignore


# ────────────────────────────────────────────────────────────
# Public types
# ────────────────────────────────────────────────────────────

# Signature for provider adapters. MUST return a DataFrame indexed by datetime
# with OHLCV columns (any case). We will normalize names and TZ.
FetchFn = Callable[
    [str, str, Optional[datetime], Optional[datetime], Optional[int]],
    pd.DataFrame
]


# ────────────────────────────────────────────────────────────
# Timeframe helpers
# ────────────────────────────────────────────────────────────

_VALID_TIMEFRAMES = {
    "1m": ("1m", timedelta(minutes=1)),
    "5m": ("5m", timedelta(minutes=5)),
    "15m": ("15m", timedelta(minutes=15)),
    "30m": ("30m", timedelta(minutes=30)),
    "1h": ("60m", timedelta(hours=1)),       # yfinance uses "60m"
    "4h": ("240m", timedelta(hours=4)),      # yfinance uses minutes for intraday
    "1d": ("1d", timedelta(days=1)),
    "1wk": ("1wk", timedelta(weeks=1)),
    "1mo": ("1mo", None),  # month length varies; handle via bars logic
}


def validate_timeframe(tf: str) -> None:
    if tf not in _VALID_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe '{tf}'. Allowed: {sorted(_VALID_TIMEFRAMES.keys())}")


def timeframe_to_provider_interval(tf: str) -> str:
    """Map our compact timeframe to a provider interval string (yfinance-friendly)."""
    validate_timeframe(tf)
    return _VALID_TIMEFRAMES[tf][0]


def timeframe_to_timedelta(tf: str) -> Optional[timedelta]:
    """Return approximate timedelta for bar distance (None for variable months)."""
    validate_timeframe(tf)
    return _VALID_TIMEFRAMES[tf][1]


# ────────────────────────────────────────────────────────────
# Normalization helpers
# ────────────────────────────────────────────────────────────

_COL_ALIASES = {
    "open": {"open", "o", "Open", "OPEN"},
    "high": {"high", "h", "High", "HIGH"},
    "low": {"low", "l", "Low", "LOW"},
    "close": {"close", "c", "Close", "CLOSE", "adjclose", "Adj Close", "AdjClose"},
    "volume": {"volume", "v", "Volume", "VOL"},
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to ['open','high','low','close','volume'] lower-case order."""
    col_map: Dict[str, str] = {}
    for std_name, aliases in _COL_ALIASES.items():
        for c in df.columns:
            if c in col_map.values():
                continue
            if c in aliases:
                col_map[c] = std_name
    # If Yahoo returns multi-index columns like ('Close','AAPL'), flatten:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["|".join([str(x) for x in tup]).strip() for tup in df.columns.values]

    # Try again after flatten
    if not col_map:
        for std_name, aliases in _COL_ALIASES.items():
            for c in df.columns:
                base = c.split("|")[0]
                if base in aliases:
                    col_map[c] = std_name

    out = df.rename(columns=col_map).copy()
    # Ensure all required columns exist
    for need in ["open", "high", "low", "close", "volume"]:
        if need not in out.columns:
            if need == "volume":
                out["volume"] = np.nan
            else:
                raise ValueError(f"Missing required column '{need}' after normalization.")
    # Keep only in correct order
    out = out[["open", "high", "low", "close", "volume"]]
    return out


def _ensure_datetime_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is tz-aware UTC, sorted, unique."""
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try common columns
        for cand in ["date", "time", "timestamp", "Datetime", "Date"]:
            if cand in df.columns:
                df = df.set_index(pd.to_datetime(df[cand], utc=True, errors="coerce"))
                break
        else:
            # Last resort: try to parse the current index
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        else:
            df.index = df.index.tz_convert(timezone.utc)

    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all normalizations: index→UTC, columns→standard, numeric dtypes."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = _ensure_datetime_index_utc(df)
    df = _normalize_columns(df)
    # cast numerics
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# ────────────────────────────────────────────────────────────
# Resampling
# ────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample normalized OHLCV DataFrame to a new timeframe using pandas rule strings.
    Common rules: '5min', '15min', '30min', '1H', '4H', '1D', '1W'.

    OHLC aggregation:
        open:  first
        high:  max
        low:   min
        close: last
        volume: sum
    """
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resample_ohlcv: DataFrame must have a DatetimeIndex.")

    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    l = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum(min_count=1)

    out = pd.concat([o, h, l, c, v], axis=1)
    out.columns = ["open", "high", "low", "close", "volume"]
    out = out.dropna(subset=["open", "high", "low", "close"], how="any")
    return out


# ────────────────────────────────────────────────────────────
# Bars / lookback helpers
# ────────────────────────────────────────────────────────────

def estimate_start_for_bars(
    end: Optional[datetime],
    timeframe: str,
    bars: int
) -> Optional[datetime]:
    """
    Roughly estimate a start datetime to cover `bars` bars of `timeframe` until `end`.
    For variable-length months ('1mo') we simply subtract 31*bars days as an upper bound.
    """
    validate_timeframe(timeframe)
    if end is None:
        end = datetime.now(timezone.utc)

    delta = timeframe_to_timedelta(timeframe)
    if delta is None:
        # monthly… be generous
        start = end - timedelta(days=31 * bars)
    else:
        start = end - (delta * (bars + 5))  # +5 slack for market closures / missing bars
    return start


# ────────────────────────────────────────────────────────────
# Default (optional) yfinance adapter
# ────────────────────────────────────────────────────────────

def _yfinance_fetcher(
    symbol: str,
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
    limit: Optional[int],
) -> pd.DataFrame:
    """
    Minimal yfinance-based adapter to satisfy FetchFn signature.
    Only used if no custom `fetch_fn` is provided.

    Notes:
      • yfinance `interval` strings include: 1m,5m,15m,30m,60m,90m,1h,1d,1wk,1mo.
      • For intraday, Yahoo limits ~30–60 days of history depending on interval.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed; please provide a custom fetch_fn.")

    # yfinance doesn’t accept '1h' historically on all paths; map to '60m'
    yf_interval = interval.replace("1h", "60m")
    if yf_interval == "240m":  # our mapping for 4h
        yf_interval = "240m"

    # yfinance uses NAIVE datetimes in local tz; we pass naive UTC times
    s = None if start is None else start.replace(tzinfo=None)
    e = None if end is None else end.replace(tzinfo=None)

    t = yf.Ticker(symbol)
    df = t.history(interval=yf_interval, start=s, end=e, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # yfinance returns columns like Open/High/Low/Close/Volume, index tz-naive local
    # Attach UTC tz
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    return df


# ────────────────────────────────────────────────────────────
# Main fetch entrypoint
# ────────────────────────────────────────────────────────────

def get_price_history(
    symbol: str,
    timeframe: str = "1d",
    *,
    bars: int = 200,
    end: Optional[datetime] = None,
    fetch_fn: Optional[FetchFn] = None,
) -> pd.DataFrame:
    """
    Fetch ≥ `bars` OHLCV rows for `symbol` at `timeframe`, normalize to UTC OHLCV.

    Parameters
    ----------
    symbol : str
        Provider-specific ticker (e.g., "AAPL", "BTC-USD", "EURUSD=X").
    timeframe : str
        One of {"1m","5m","15m","30m","1h","4h","1d","1wk","1mo"}.
    bars : int
        Approximate number of rows requested. We may fetch a little extra to
        compensate for market holidays, illiquid symbols, etc.
    end : datetime | None
        End timestamp (defaults to now, UTC).
    fetch_fn : FetchFn | None
        Custom provider adapter. If None, uses the yfinance fallback.

    Returns
    -------
    pd.DataFrame
        Normalized OHLCV with a UTC DatetimeIndex and columns:
        ["open","high","low","close","volume"].
    """
    validate_timeframe(timeframe)
    if end is None:
        end = datetime.now(timezone.utc)

    start = estimate_start_for_bars(end, timeframe, bars)
    interval = timeframe_to_provider_interval(timeframe)

    adapter = fetch_fn or _yfinance_fetcher
    raw = adapter(symbol, interval, start, end, bars)

    df = normalize_ohlcv(raw)
    if df.empty:
        return df

    # Ensure we have at least `bars` rows. Some providers ignore 'limit'/'bars'.
    if len(df) < bars and start is not None:
        # Try to push the start further back by 2x and refetch (best effort).
        start2 = start - (start - start.replace(year=max(1971, start.year - 5))) / 2  # crude widen window
        raw2 = adapter(symbol, interval, start2, end, bars * 2)
        df2 = normalize_ohlcv(raw2)
        if len(df2) > len(df):
            df = df2

    # Keep most recent `bars` rows
    if len(df) > bars:
        df = df.iloc[-bars:]

    return df


# ────────────────────────────────────────────────────────────
# Multi-timeframe convenience
# ────────────────────────────────────────────────────────────

@dataclass
class MTFWindow:
    """
    Container for multi-timeframe history for the same symbol.
    Example:
        mtf = get_mtf_price_history("AAPL", base_timeframe="1m",
                                    resample_rules=["5min","15min","1H","1D"],
                                    base_bars=2000)
        df_1m = mtf.base
        df_15m = mtf.resampled["15min"]
    """
    symbol: str
    base_timeframe: str
    base: pd.DataFrame
    resampled: Dict[str, pd.DataFrame]


def get_mtf_price_history(
    symbol: str,
    *,
    base_timeframe: str = "1m",
    base_bars: int = 2000,
    resample_rules: Optional[List[str]] = None,
    fetch_fn: Optional[FetchFn] = None,
) -> MTFWindow:
    """
    Fetch a high-resolution base OHLCV and resample to coarser frames.

    Parameters
    ----------
    symbol : str
        Ticker (provider-specific).
    base_timeframe : str
        The provider timeframe for the base fetch (e.g., '1m').
    base_bars : int
        How many base bars to retrieve (more → better resampling accuracy).
    resample_rules : list[str] | None
        Pandas resample rules to produce coarser frames (e.g., ['5min','15min','1H','1D']).
    fetch_fn : FetchFn | None
        Provider adapter.

    Returns
    -------
    MTFWindow
    """
    base = get_price_history(symbol, timeframe=base_timeframe, bars=base_bars, fetch_fn=fetch_fn)
    out: Dict[str, pd.DataFrame] = {}
    if resample_rules:
        for rule in resample_rules:
            out[rule] = resample_ohlcv(base, rule)
    return MTFWindow(symbol=symbol, base_timeframe=base_timeframe, base=base, resampled=out)


# ────────────────────────────────────────────────────────────
# Alignment helpers (portfolio or paired signals)
# ────────────────────────────────────────────────────────────

def align_histories(histories: Dict[str, pd.DataFrame], how: str = "inner") -> pd.DataFrame:
    """
    Align multiple normalized OHLCV DataFrames on a single DatetimeIndex by the chosen join.
    We suffix columns with the symbol to avoid collisions.

    Parameters
    ----------
    histories : dict[symbol -> df]
        Each df must already be normalized (UTC index, ohlcv columns).
    how : str
        Join strategy ('inner','outer','left','right').

    Returns
    -------
    pd.DataFrame
        Multi-asset panel with columns like: 'AAPL_close', 'BTC-USD_volume', ...
    """
    frames: List[pd.DataFrame] = []
    for sym, df in histories.items():
        if df is None or df.empty:
            continue
        renamed = df.add_suffix(f"_{sym}")
        frames.append(renamed)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1, join=how).sort_index()
