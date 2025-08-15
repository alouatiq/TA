from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd


def _to_series(prices) -> pd.Series:
    if isinstance(prices, pd.Series):
        return prices.dropna()
    return pd.Series(prices).dropna()


def atr_stop(
    close_prices,
    *,
    atr: Optional[pd.Series] = None,
    atr_mult: float = 2.0,
    min_stop_pct: float = 0.01,
) -> float:
    """
    ATR-based stop distance as a fraction of current price (e.g., 0.02 = 2%).
    If `atr` is not provided, approximate from abs returns.
    """
    c = _to_series(close_prices)
    if atr is None:
        ar = c.pct_change().abs().rolling(14).mean() * c.shift(1)
        atr = (ar / c.replace(0, np.nan)).fillna(method="ffill")
    last_atr_pct = float((atr / c.replace(0, np.nan)).ffill().iloc[-1])
    stop_pct = max(min_stop_pct, atr_mult * last_atr_pct)
    return stop_pct


def swing_stop(
    close_prices,
    *,
    lookback: int = 10,
    buffer_pct: float = 0.003,
) -> float:
    """
    Distance to recent swing low (longs) as a fraction of current price.
    """
    c = _to_series(close_prices)
    if len(c) < lookback + 3:
        return 0.02  # fallback 2%
    swing_low = float(c.iloc[-lookback:].min())
    current = float(c.iloc[-1])
    dist = (current - swing_low) / max(1e-9, current)
    return max(buffer_pct, dist)


def volatility_target_position(
    close_prices,
    *,
    account_equity: float,
    target_vol_annual: float = 0.15,
    max_position_pct: float = 0.25,
) -> float:
    """
    Volatility targeting position size in notional currency.

    - Estimate realized annualized volatility from daily returns
    - Position so that contribution ≈ target_vol_annual of account
    """
    c = _to_series(close_prices)
    if len(c) < 30 or account_equity <= 0:
        return 0.0

    ret = c.pct_change().dropna()
    # daily vol → annualized
    vol_ann = float(ret.std() * np.sqrt(252))
    if vol_ann <= 1e-9:
        return 0.0

    # risk parity-ish: weight ~ target / vol
    weight = min(max_position_pct, target_vol_annual / vol_ann)
    return weight * account_equity


def suggest_stops_and_size(
    close_prices,
    *,
    side: str = "long",
    account_equity: float = 10000.0,
    risk_per_trade_pct: float = 0.01,
    use_atr_stop: bool = True,
    atr_mult: float = 2.0,
    lookback_swing: int = 10,
    target_vol_annual: float = 0.15,
    max_position_pct: float = 0.25,
) -> Dict[str, float]:
    """
    Combine stop logic and position sizing:
      • Stop distance = max(ATR stop, swing stop)
      • Position size = min( fixed risk-per-trade sizing, volatility targeting )

    Returns:
      {
        "stop_loss_pct": ...,
        "take_profit_pct": ...,   # 2R default
        "position_notional": ...,
        "max_position_notional": ...,
      }
    """
    c = _to_series(close_prices)
    if c.empty:
        return {
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
            "position_notional": 0.0,
            "max_position_notional": 0.0,
        }

    stop_atr = atr_stop(c, atr_mult=atr_mult) if use_atr_stop else 0.0
    stop_swing = swing_stop(c, lookback=lookback_swing)
    stop_pct = max(stop_atr, stop_swing)

    # 2R default TP
    take_profit_pct = 2.0 * stop_pct

    # fixed risk per trade sizing
    risk_cash = risk_per_trade_pct * account_equity
    # position such that (price * qty * stop_pct) ≈ risk_cash → notional ≈ risk_cash / stop_pct
    if stop_pct <= 1e-9:
        pos_risk_based = 0.0
    else:
        pos_risk_based = risk_cash / stop_pct

    # volatility targeting cap
    pos_vol_target = volatility_target_position(c, account_equity=account_equity,
                                                target_vol_annual=target_vol_annual,
                                                max_position_pct=max_position_pct)

    position_notional = float(min(pos_risk_based, pos_vol_target if pos_vol_target > 0 else pos_risk_based))
    max_pos_notional = float(max_position_pct * account_equity)

    return {
        "stop_loss_pct": float(stop_pct),
        "take_profit_pct": float(take_profit_pct),
        "position_notional": position_notional,
        "max_position_notional": max_pos_notional,
    }


def portfolio_correlation_matrix(price_dict: Dict[str, Sequence[float]]) -> pd.DataFrame:
    """
    Build a correlation matrix from multiple assets' close histories.
    price_dict: { symbol: [closes...] }
    """
    frames = []
    keys = []
    for k, v in price_dict.items():
        s = _to_series(v)
        if len(s) >= 20:
            frames.append(s.pct_change().dropna().rename(k))
            keys.append(k)
    if not frames:
        return pd.DataFrame()
    mat = pd.concat(frames, axis=1).corr()
    return mat.loc[keys, keys]
