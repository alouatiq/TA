"""
trading_core.indicators.technical
================================

Core technical analysis indicators implemented with pandas.

This module provides standard TA indicators with sensible defaults:
- Trend: SMA, EMA, MACD, ADX
- Momentum: RSI, Stochastic Oscillator
- Volume: OBV, volume spike detection
- Volatility: Bollinger Bands, ATR

All indicators accept either Series or DataFrame input and return pandas objects.
For multi-column DataFrames, specify price_col (default "Close").

Example usage:
    import pandas as pd
    from trading_core.indicators.technical import rsi, sma, macd
    
    # From Series
    closes = pd.Series([100, 102, 101, 105, 103, ...])
    rsi_values = rsi(closes, window=14)
    
    # From DataFrame
    df = pd.DataFrame({'Open': [...], 'High': [...], 'Low': [...], 'Close': [...], 'Volume': [...]})
    sma_20 = sma(df, window=20, price_col="Close")
    macd_line, signal_line, histogram = macd(df)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _extract_price_series(data: Union[pd.Series, pd.DataFrame], price_col: str = "Close") -> pd.Series:
    """
    Extract price series from input data.
    
    Args:
        data: pandas Series or DataFrame
        price_col: column name if DataFrame (default "Close")
    
    Returns:
        pandas Series with price data
    """
    if isinstance(data, pd.Series):
        return data
    elif isinstance(data, pd.DataFrame):
        if price_col not in data.columns:
            # Try common variations
            cols_lower = {col.lower(): col for col in data.columns}
            price_col_lower = price_col.lower()
            if price_col_lower in cols_lower:
                price_col = cols_lower[price_col_lower]
            else:
                raise ValueError(f"Column '{price_col}' not found in DataFrame. Available columns: {list(data.columns)}")
        return data[price_col]
    else:
        raise TypeError(f"Expected pandas Series or DataFrame, got {type(data)}")


def _to_series(price_history: List[float], name: str = "price") -> pd.Series:
    """Convert list of prices to pandas Series for compatibility functions."""
    return pd.Series(price_history, name=name)


# ──────────────────────────────────────────────────────────────────────────────
# Trend Indicators
# ──────────────────────────────────────────────────────────────────────────────

def sma(data: Union[pd.Series, pd.DataFrame], window: int = 14, price_col: str = "Close") -> pd.Series:
    """
    Simple Moving Average.
    
    Args:
        data: Price data (Series) or OHLCV (DataFrame)
        window: Period for moving average
        price_col: Column name if DataFrame input
    
    Returns:
        pandas Series with SMA values
    """
    series = _extract_price_series(data, price_col)
    min_periods = max(1, window // 2)  # Allow partial calculation with at least half the window
    return series.rolling(window=window, min_periods=min_periods).mean()


def ema(data: Union[pd.Series, pd.DataFrame], window: int = 14, price_col: str = "Close") -> pd.Series:
    """
    Exponential Moving Average.
    
    Args:
        data: Price data (Series) or OHLCV (DataFrame) 
        window: Period for EMA
        price_col: Column name if DataFrame input
    
    Returns:
        pandas Series with EMA values
    """
    series = _extract_price_series(data, price_col)
    return series.ewm(span=window, adjust=False).mean()


def macd(data: Union[pd.Series, pd.DataFrame], 
         fast: int = 12, 
         slow: int = 26, 
         signal: int = 9, 
         price_col: str = "Close") -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence).
    
    Args:
        data: Price data (Series) or OHLCV (DataFrame)
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line EMA period
        price_col: Column name if DataFrame input
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    series = _extract_price_series(data, price_col)
    
    ema_fast = ema(series, window=fast)
    ema_slow = ema(series, window=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, window=signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX) - measures trend strength.
    
    Args:
        df: DataFrame with OHLC columns
        window: Period for ADX calculation
    
    Returns:
        pandas Series with ADX values
    """
    # Extract OHLC data
    high = _extract_price_series(df, "High")
    low = _extract_price_series(df, "Low") 
    close = _extract_price_series(df, "Close")
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    dm_plus = pd.Series(0.0, index=high.index)
    dm_minus = pd.Series(0.0, index=high.index)
    
    high_diff = high.diff()
    low_diff = -low.diff()
    
    # Calculate +DM and -DM
    dm_plus = pd.Series(np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0), index=high.index)
    dm_minus = pd.Series(np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0), index=high.index)
    
    # Smooth TR, +DM, and -DM with modified moving average
    min_periods = max(1, window // 2)
    atr = tr.rolling(window=window, min_periods=min_periods).mean()
    adm_plus = dm_plus.rolling(window=window, min_periods=min_periods).mean()
    adm_minus = dm_minus.rolling(window=window, min_periods=min_periods).mean()
    
    # Calculate +DI and -DI
    di_plus = 100 * (adm_plus / atr.replace(0, np.nan))
    di_minus = 100 * (adm_minus / atr.replace(0, np.nan))
    
    # Calculate DX and ADX
    dx = 100 * ((di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan))
    adx_values = dx.rolling(window=window, min_periods=min_periods).mean()
    
    return adx_values


# ──────────────────────────────────────────────────────────────────────────────
# Momentum Indicators  
# ──────────────────────────────────────────────────────────────────────────────

def rsi(data: Union[pd.Series, pd.DataFrame], window: int = 14, price_col: str = "Close") -> pd.Series:
    """
    Relative Strength Index (RSI).
    
    Args:
        data: Price data (Series) or OHLCV (DataFrame)
        window: Period for RSI calculation (default 14)
        price_col: Column name if DataFrame input
    
    Returns:
        pandas Series with RSI values (0-100)
    """
    series = _extract_price_series(data, price_col)
    
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Calculate average gains and losses using simple moving average
    min_periods = max(1, window // 2)
    avg_gains = gains.rolling(window=window, min_periods=min_periods).mean()
    avg_losses = losses.rolling(window=window, min_periods=min_periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses.replace(0, np.nan)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def stochastic(df: pd.DataFrame, 
               k: int = 14, 
               d: int = 3, 
               smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator (%K and %D).
    
    Args:
        df: DataFrame with OHLC data
        k: Period for %K calculation
        d: Period for %D (moving average of %K)
        smooth_k: Period for smoothing %K
    
    Returns:
        Tuple of (%K, %D) series
    """
    high = _extract_price_series(df, "High")
    low = _extract_price_series(df, "Low")
    close = _extract_price_series(df, "Close")
    
    # Calculate %K
    min_periods = max(1, k // 2)
    lowest_low = low.rolling(window=k, min_periods=min_periods).min()
    highest_high = high.rolling(window=k, min_periods=min_periods).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan))
    
    # Smooth %K if requested
    if smooth_k > 1:
        k_percent = k_percent.rolling(window=smooth_k, min_periods=max(1, smooth_k // 2)).mean()
    
    # Calculate %D (moving average of %K)
    d_percent = k_percent.rolling(window=d, min_periods=max(1, d // 2)).mean()
    
    return k_percent, d_percent


# ──────────────────────────────────────────────────────────────────────────────
# Volume Indicators
# ──────────────────────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame, price_col: str = "Close", volume_col: str = "Volume") -> pd.Series:
    """
    On-Balance Volume (OBV).
    
    Args:
        df: DataFrame with price and volume data
        price_col: Price column name
        volume_col: Volume column name
    
    Returns:
        pandas Series with OBV values
    """
    close = _extract_price_series(df, price_col)
    volume = _extract_price_series(df, volume_col)
    
    # Calculate price changes
    price_change = close.diff()
    
    # Create volume direction multiplier
    volume_direction = pd.Series(0, index=close.index)
    volume_direction[price_change > 0] = 1    # Up days
    volume_direction[price_change < 0] = -1   # Down days
    volume_direction[price_change == 0] = 0   # Unchanged days
    
    # Calculate OBV
    obv_values = (volume * volume_direction).cumsum()
    
    return obv_values


def detect_volume_spike(volume: pd.Series, 
                       window: int = 20, 
                       threshold: float = 2.0) -> pd.Series:
    """
    Detect volume spikes using Z-score method.
    
    Args:
        volume: Volume data series
        window: Period for calculating average volume
        threshold: Z-score threshold for spike detection
    
    Returns:
        pandas Series with boolean spike indicators
    """
    min_periods = max(1, window // 2)
    vol_mean = volume.rolling(window=window, min_periods=min_periods).mean()
    vol_std = volume.rolling(window=window, min_periods=min_periods).std()
    
    z_score = (volume - vol_mean) / vol_std.replace(0, np.nan)
    spikes = z_score > threshold
    
    return spikes


# ──────────────────────────────────────────────────────────────────────────────
# Volatility Indicators
# ──────────────────────────────────────────────────────────────────────────────

def bollinger_bands(data: Union[pd.Series, pd.DataFrame], 
                   window: int = 20, 
                   n_std: float = 2.0, 
                   price_col: str = "Close") -> pd.DataFrame:
    """
    Bollinger Bands.
    
    Args:
        data: Price data (Series) or OHLCV (DataFrame)
        window: Period for moving average and standard deviation
        n_std: Number of standard deviations for bands
        price_col: Column name if DataFrame input
    
    Returns:
        DataFrame with columns: bb_mid, bb_upper, bb_lower, bb_width
    """
    close = _extract_price_series(data, price_col)
    
    min_periods = max(3, window // 2)
    mid = close.rolling(window=window, min_periods=min_periods).mean()
    std = close.rolling(window=window, min_periods=min_periods).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    
    return pd.DataFrame({
        "bb_mid": mid, 
        "bb_upper": upper, 
        "bb_lower": lower, 
        "bb_width": width
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - measures volatility.
    
    Args:
        df: DataFrame with OHLC data
        window: Period for ATR calculation
    
    Returns:
        pandas Series with ATR values
    """
    high = _extract_price_series(df, "High")
    low = _extract_price_series(df, "Low")
    close = _extract_price_series(df, "Close")
    
    # Calculate True Range components
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    min_periods = max(1, window // 2)
    atr_values = tr.rolling(window=window, min_periods=min_periods).mean()
    
    return atr_values


# ──────────────────────────────────────────────────────────────────────────────
# Bundle / Multi-timeframe computation
# ──────────────────────────────────────────────────────────────────────────────

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


def compute_bundle(ohlcv: pd.DataFrame, cfg: Optional[TechConfig] = None) -> pd.DataFrame:
    """
    Compute a standard technical feature bundle on OHLCV data.

    Args:
        ohlcv: DataFrame with columns ['Open','High','Low','Close','Volume']
        cfg: TechConfig with indicator parameters

    Returns:
        DataFrame with appended columns for indicators
    """
    if cfg is None:
        cfg = TechConfig()

    result = ohlcv.copy()

    try:
        # RSI
        result["rsi"] = rsi(ohlcv, window=cfg.rsi_window)
        
        # SMAs
        result["sma_short"] = sma(ohlcv, window=cfg.sma_windows[0])
        result["sma_long"] = sma(ohlcv, window=cfg.sma_windows[1])
        
        # EMAs  
        result["ema_fast"] = ema(ohlcv, window=cfg.ema_windows[0])
        result["ema_slow"] = ema(ohlcv, window=cfg.ema_windows[1])
        
        # MACD
        macd_line, signal_line, histogram = macd(
            ohlcv, 
            fast=cfg.ema_windows[0], 
            slow=cfg.ema_windows[1], 
            signal=cfg.macd_signal
        )
        result["macd_line"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram
        
        # ADX
        result["adx"] = adx(ohlcv, window=cfg.adx_window)
        
        # Stochastic
        stoch_k, stoch_d = stochastic(ohlcv, k=cfg.stoch_k, d=cfg.stoch_d)
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d
        
        # OBV
        if "Volume" in ohlcv.columns:
            result["obv"] = obv(ohlcv)
            result["volume_spike"] = detect_volume_spike(
                ohlcv["Volume"], 
                window=cfg.vol_spike_window, 
                threshold=cfg.vol_spike_z
            )
        
        # Bollinger Bands
        bb_df = bollinger_bands(ohlcv, window=cfg.bb_window, n_std=cfg.bb_nstd)
        for col in bb_df.columns:
            result[col] = bb_df[col]
        
        # ATR
        result["atr"] = atr(ohlcv, window=cfg.atr_window)
        
    except Exception as e:
        # Log warning but don't crash - partial results are better than none
        print(f"Warning: Some technical indicators failed to compute: {e}")
    
    return result


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame has standard OHLCV column names with proper case.
    
    Args:
        df: DataFrame with price/volume data
        
    Returns:
        DataFrame with standardized column names
    """
    # Create mapping of common variations to standard names
    column_mapping = {}
    standard_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        if col_lower in ['open', 'o']:
            column_mapping[col] = 'Open'
        elif col_lower in ['high', 'h']:
            column_mapping[col] = 'High'
        elif col_lower in ['low', 'l']:
            column_mapping[col] = 'Low'
        elif col_lower in ['close', 'c', 'price']:
            column_mapping[col] = 'Close'
        elif col_lower in ['volume', 'vol', 'v']:
            column_mapping[col] = 'Volume'
    
    # Rename columns
    result_df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in result_df.columns:
            # If Close exists but not OHLC, use Close for missing values
            if 'Close' in result_df.columns:
                if col in ['Open', 'High', 'Low']:
                    result_df[col] = result_df['Close']
            else:
                raise ValueError(f"Required column '{col}' not found and cannot be inferred")
    
    return result_df


def resample_ohlcv(ohlcv: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.
    
    Args:
        ohlcv: DataFrame with datetime index and OHLCV columns
        rule: Pandas resample rule (e.g., '1H', '4H', '1D')
    
    Returns:
        Resampled DataFrame
    """
    agg_dict = {
        "Open": "first",
        "High": "max", 
        "Low": "min",
        "Close": "last"
    }
    
    if "Volume" in ohlcv.columns:
        agg_dict["Volume"] = "sum"
    
    return ohlcv.resample(rule).agg(agg_dict).dropna(how="any")


def compute_multi_timeframe(ohlcv: pd.DataFrame,
                           timeframes: Tuple[str, ...] = ("1H", "4H", "1D"),
                           cfg: Optional[TechConfig] = None) -> dict:
    """
    Compute the technical bundle across multiple timeframes.

    Args:
        ohlcv: DataFrame with datetime index and OHLCV columns
        timeframes: Tuple of timeframe strings  
        cfg: TechConfig with indicator parameters

    Returns:
        Dictionary mapping timeframe -> feature DataFrame
    """
    out = {}
    for tf in timeframes:
        try:
            ohlcv_tf = resample_ohlcv(ohlcv, tf)
            out[tf] = compute_bundle(ohlcv_tf, cfg)
        except Exception:
            # Keep going; a single bad resample shouldn't kill the pipeline
            continue
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatibility lightweight shims (legacy CLI expects these)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_sma(price_history: List[float], window: int = 14) -> Optional[float]:
    """
    Legacy wrapper: returns the LAST SMA value for quick RSI/SMA display.
    
    Args:
        price_history: List of price values
        window: Period for SMA calculation
        
    Returns:
        Latest SMA value or None if insufficient data
    """
    if not price_history or len(price_history) < max(2, window // 2):
        return None
    
    ser = _to_series(price_history, "close")
    sma_values = ser.rolling(window=window, min_periods=max(2, window // 2)).mean()
    
    return float(sma_values.iloc[-1]) if not pd.isna(sma_values.iloc[-1]) else None


def calculate_rsi(price_history: List[float], window: int = 14) -> Optional[float]:
    """
    Legacy wrapper: returns the LAST RSI value for quick RSI-only paths.
    
    Note: This function signature is kept for backward compatibility.
    The main.py code calls this with a 'window' parameter.
    
    Args:
        price_history: List of price values  
        window: Period for RSI calculation (default 14)
        
    Returns:
        Latest RSI value or None if insufficient data
    """
    if not price_history or len(price_history) < window + 1:
        return None
        
    ser = _to_series(price_history, "close")
    rsi_values = rsi(ser, window=window)
    
    last_rsi = rsi_values.iloc[-1]
    return None if pd.isna(last_rsi) else float(last_rsi)
