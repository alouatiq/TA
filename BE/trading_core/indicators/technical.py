"""
technical.py
────────────
Technical indicators used by the Trading Assistant.

Covers:
• Trend:      SMA, EMA, MACD (line/signal/hist), ADX (+DI, -DI)
• Momentum:   RSI, Stochastic Oscillator (%K, %D)
• Volume:     OBV, volume spikes (z-score method)
• Volatility: Bollinger Bands (mid/upper/lower, width), ATR

Also includes:
• Multi-timeframe computation helpers
• Safe, dependency-light implementations (NumPy/Pandas only)
• NaN-safe returns (avoids crashing when history is short)

Input conventions
────────────────
Most functions accept a pandas Series/DataFrame with a DateTime index.
- Close prices: series or df["close"]
- High/Low/Close/Volume: columns "high","low","close","volume"
Return values align with the input index and preserve types.

Back-compat shims (used by the legacy CLI path):
- simple SMA/RSI wrappers that accept a list[float] price history.

NOTE:
These implementations are intentionally explicit (not TA-Lib) to keep
the project portable and to avoid native deps.

Authoring guidelines
────────────────────
- Keep each indicator pure (no I/O)
- Avoid altering inputs; always return new series/frames
- Provide clear docstrings & type hints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────

def _to_series(x: Iterable[float], name: str = "value") -> pd.Series:
    """
    Convert an iterable of floats into a pandas Series with a simple RangeIndex.
    """
    return pd.Series(list(x), name=name, dtype=float)


def _safe_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential moving average with sensible defaults.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: (x - mean) / std with min_periods to avoid NaN-only stretches.
    """
    m = series.rolling(window=window, min_periods=max(3, window // 2)).mean()
    s = series.rolling(window=window, min_periods=max(3, window // 2)).std(ddof=0)
    return (series - m) / s.replace(0, np.nan)


# ───────────────────────────────────────────────────────────
# Trend
# ───────────────────────────────────────────────────────────

def sma(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Simple Moving Average.
    """
    return close.rolling(window=window, min_periods=max(2, window // 2)).mean()


def ema(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Exponential Moving Average.
    """
    return _safe_ema(close, span=window)


def macd(close: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD line/signal/histogram.
    Returns DataFrame with columns: ['macd', 'signal', 'hist']
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    line = fast_ema - slow_ema
    sig = _safe_ema(line, span=signal)
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "hist": hist})


def adx(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (+DI, -DI, ADX).
    Returns DataFrame with ['plus_di', 'minus_di', 'adx'].
    """
    # True Range components
    up_move = high.diff()
    down_move = low.diff().abs().multiply(-1.0)  # negative raw
    plus_dm = np.where((up_move > 0) & (up_move > -down_move), up_move, 0.0)
    minus_dm = np.where((-down_move > 0) & (-down_move > up_move), -down_move, 0.0)

    # True Range
    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Smoothed (Wilder)
    atr_sm = tr.rolling(window=window, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).rolling(window=window, min_periods=window).mean() / atr_sm
    minus_di = 100 * pd.Series(minus_dm, index=close.index).rolling(window=window, min_periods=window).mean() / atr_sm

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_val = dx.rolling(window=window, min_periods=window).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_val})


# ───────────────────────────────────────────────────────────
# Momentum
# ───────────────────────────────────────────────────────────

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder's).
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def stochastic_oscillator(high: pd.Series,
                          low: pd.Series,
                          close: pd.Series,
                          k_window: int = 14,
                          d_window: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator %K and %D.
    """
    lowest_low = low.rolling(window=k_window, min_periods=max(2, k_window // 2)).min()
    highest_high = high.rolling(window=k_window, min_periods=max(2, k_window // 2)).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_window, min_periods=1).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


# ───────────────────────────────────────────────────────────
# Volume
# ───────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume.
    """
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def volume_spikes(volume: pd.Series, window: int = 20, z: float = 2.5) -> pd.Series:
    """
    Boolean series: True when rolling z-score of volume exceeds `z`.
    """
    zscores = _rolling_zscore(volume.fillna(0.0), window=window)
    return (zscores >= z)


# ───────────────────────────────────────────────────────────
# Volatility
# ───────────────────────────────────────────────────────────

def bollinger_bands(close: pd.Series,
                    window: int = 20,
                    n_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands (mid/upper/lower) + width.
    """
    mid = close.rolling(window=window, min_periods=max(3, window // 2)).mean()
    std = close.rolling(window=window, min_periods=max(3, window // 2)).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14) -> pd.Series:
    """
    Average True Range.
    """
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


# ───────────────────────────────────────────────────────────
# Bundles / Multi-timeframe
# ───────────────────────────────────────────────────────────

@dataclass
class TechConfig:
    """
    Configuration for computing a standard technical bundle.
    """
    rsi_window: int = 14
    sma_windows: Tuple[int, int] = (20, 50)    # short, long
    ema_windows: Tuple[int, int] = (12, 26)    # fast, slow (also for MACD defaults)
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    adx_window: int = 14
    bb_window: int = 20
    bb_nstd: float = 2.0
    atr_window: int = 14
    vol_spike_window: int = 20
    vol_spike_z: float = 2.5


def compute_bundle(ohlcv: pd.DataFrame,
                   cfg: Optional[TechConfig] = None) -> pd.DataFrame:
    """
    Compute a standard technical feature bundle on OHLCV data.

    Parameters
    ----------
    ohlcv : DataFrame with columns ['open','high','low','close','volume']
    cfg   : TechConfig with indicator parameters

    Returns
    -------
    DataFrame with appended columns for indicators.
    """
    if cfg is None:
        cfg = TechConfig()

    df = ohlcv.copy()

    # Trend
    df[f"sma_{cfg.sma_windows[0]}"] = sma(df["close"], cfg.sma_windows[0])
    df[f"sma_{cfg.sma_windows[1]}"] = sma(df["close"], cfg.sma_windows[1])
    df[f"ema_{cfg.ema_windows[0]}"] = ema(df["close"], cfg.ema_windows[0])
    df[f"ema_{cfg.ema_windows[1]}"] = ema(df["close"], cfg.ema_windows[1])

    macd_df = macd(df["close"], cfg.ema_windows[0], cfg.ema_windows[1], cfg.macd_signal)
    df = df.join(macd_df)

    adx_df = adx(df["high"], df["low"], df["close"], cfg.adx_window)
    df = df.join(adx_df)

    # Momentum
    df[f"rsi_{cfg.rsi_window}"] = rsi(df["close"], cfg.rsi_window)
    stoch_df = stochastic_oscillator(df["high"], df["low"], df["close"], cfg.stoch_k, cfg.stoch_d)
    df = df.join(stoch_df)

    # Volume
    df["obv"] = obv(df["close"], df["volume"])
    df["vol_spike"] = volume_spikes(df["volume"], cfg.vol_spike_window, cfg.vol_spike_z)

    # Volatility
    bb = bollinger_bands(df["close"], cfg.bb_window, cfg.bb_nstd)
    df = df.join(bb)
    df[f"atr_{cfg.atr_window}"] = atr(df["high"], df["low"], df["close"], cfg.atr_window)

    return df


def resample_ohlcv(ohlcv: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample OHLCV to a different timeframe (e.g., '1H', '4H', '1D').
    Assumes DateTimeIndex. Aggregates as common in TA practice.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return ohlcv.resample(rule).apply(agg).dropna(how="any")


def compute_multi_timeframe(ohlcv: pd.DataFrame,
                            timeframes: Tuple[str, ...] = ("1H", "4H", "1D"),
                            cfg: Optional[TechConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    Compute the technical bundle across multiple timeframes.

    Returns dict timeframe -> feature DataFrame.
    """
    out: Dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        try:
            ohlcv_tf = resample_ohlcv(ohlcv, tf)
            out[tf] = compute_bundle(ohlcv_tf, cfg)
        except Exception:
            # Keep going; a single bad resample shouldn't kill the pipeline
            continue
    return out


# ───────────────────────────────────────────────────────────
# Backward-compat lightweight shims (legacy CLI expects these)
# ───────────────────────────────────────────────────────────

def calculate_sma(price_history: List[float], window: int = 14) -> Optional[float]:
    """
    Legacy wrapper: returns the LAST SMA value for quick RSI/SMA display.
    """
    if not price_history or len(price_history) < max(2, window // 2):
        return None
    ser = _to_series(price_history, "close")
    return float(ser.rolling(window=window, min_periods=max(2, window // 2)).mean().iloc[-1])


def calculate_rsi(price_history: List[float], window: int = 14) -> Optional[float]:
    """
    Legacy wrapper: returns the LAST RSI value for quick RSI-only paths.
    """
    if not price_history or len(price_history) < window + 1:
        return None
    ser = _to_series(price_history, "close")
    val = rsi(ser, window=window).iloc[-1]
    return None if pd.isna(val) else float(val)
