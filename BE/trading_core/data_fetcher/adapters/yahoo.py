# BE/trading_core/data_fetcher/adapters/yahoo.py
"""
Yahoo adapter
─────────────
Lightweight helpers around Yahoo Finance:
• Primary: yfinance.Ticker.history for OHLCV history
• Fallback: quote JSON API for last price/volume/day range
• Utilities to fetch many symbols efficiently and normalize outputs

All functions are side‑effect free and never hard‑code symbols.
They return Python dicts/Series you can pass into indicators/price_history.

Dependencies: yfinance, requests, pandas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import time
import math
import json
import logging
import requests
import pandas as pd
import yfinance as yf

# Basic defaults – keep them conservative; callers can override per call.
_DEFAULT_PERIOD = "30d"
_DEFAULT_INTERVAL = "1d"
_REQUEST_TIMEOUT = 15
_UA = {"User-Agent": "Mozilla/5.0 (TradingAssistant/1.0)"}

YF_QUOTE_JSON = "https://query1.finance.yahoo.com/v7/finance/quote"

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Models
# ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QuoteLite:
    symbol: str
    price: Optional[float]
    volume: Optional[float]
    day_range_pct: Optional[float]  # (high - low) / price * 100 if available


# ────────────────────────────────────────────────────────────
# Helpers: yfinance history
# ────────────────────────────────────────────────────────────

def fetch_history(
    symbol: str,
    *,
    period: str = _DEFAULT_PERIOD,
    interval: str = _DEFAULT_INTERVAL,
    auto_adjust: bool = True,
    prepost: bool = False,
) -> pd.DataFrame:
    """
    Get OHLCV history via yfinance for one symbol.
    Returns empty DataFrame on failure (never raises outward).
    """
    try:
        df = yf.Ticker(symbol).history(
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            prepost=prepost,
        )
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        # normalize column names
        cols = {c: c.capitalize() for c in df.columns}
        df = df.rename(columns=cols)
        # ensure required columns exist
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = pd.NA
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.debug("yfinance history failed for %s: %s", symbol, e)
        return pd.DataFrame()


# ────────────────────────────────────────────────────────────
# Helpers: Yahoo quote JSON (fallback or fast multi‑symbol)
# ────────────────────────────────────────────────────────────

def fetch_quote_json(symbols: Iterable[str]) -> Dict[str, QuoteLite]:
    """
    Call Yahoo's quote JSON endpoint for one or many symbols.
    Returns mapping symbol -> QuoteLite. Missing symbols are omitted.

    NOTE: This is best‑effort; Yahoo can rate‑limit. Keep batches small (<= 50).
    """
    syms = [s for s in symbols if s]
    if not syms:
        return {}

    out: Dict[str, QuoteLite] = {}
    try:
        resp = requests.get(
            YF_QUOTE_JSON,
            params={"symbols": ",".join(syms)},
            headers=_UA,
            timeout=_REQUEST_TIMEOUT,
        )
        data = resp.json().get("quoteResponse", {}).get("result", [])
        for row in data:
            sym = row.get("symbol")
            price = row.get("regularMarketPrice") or row.get("postMarketPrice") or row.get("preMarketPrice")
            vol = row.get("regularMarketVolume") or row.get("averageDailyVolume3Month")
            high = row.get("regularMarketDayHigh")
            low = row.get("regularMarketDayLow")
            drp = None
            try:
                if price and high and low and float(price) != 0.0:
                    drp = (float(high) - float(low)) / float(price) * 100.0
            except Exception:
                drp = None
            if sym:
                out[sym] = QuoteLite(
                    symbol=sym,
                    price=float(price) if price is not None else None,
                    volume=float(vol) if vol is not None else None,
                    day_range_pct=float(drp) if drp is not None else None,
                )
    except Exception as e:
        logger.debug("Yahoo quote JSON failed for %s symbols: %s", len(syms), e)

    return out


# ────────────────────────────────────────────────────────────
# High‑level convenience
# ────────────────────────────────────────────────────────────

def fetch_quote_and_history(
    symbol: str,
    *,
    need_history: bool = True,
    history_period: str = _DEFAULT_PERIOD,
    history_interval: str = _DEFAULT_INTERVAL,
) -> Tuple[Optional[QuoteLite], pd.DataFrame]:
    """
    Try full history first (contains last Close & Volume); if empty, fall back to quote JSON.
    Returns (QuoteLite or None, history_df possibly empty).
    """
    hist = fetch_history(
        symbol,
        period=history_period,
        interval=history_interval,
    )
    if not hist.empty:
        last = hist.iloc[-1]
        price = float(last["Close"]) if not pd.isna(last["Close"]) else None
        vol = float(last["Volume"]) if not pd.isna(last["Volume"]) else None
        day_range_pct = None
        try:
            if price and not pd.isna(last["High"]) and not pd.isna(last["Low"]) and price != 0.0:
                day_range_pct = (float(last["High"]) - float(last["Low"])) / price * 100.0
        except Exception:
            day_range_pct = None
        return QuoteLite(symbol=symbol, price=price, volume=vol, day_range_pct=day_range_pct), hist

    # History missing → quote JSON fallback
    quotes = fetch_quote_json([symbol])
    q = quotes.get(symbol)
    if q is None:
        return None, pd.DataFrame()
    return q, pd.DataFrame()  # no history available


def batch_fetch_quotes(
    symbols: Iterable[str],
    *,
    batch_size: int = 50,
    sleep_secs: float = 0.0,
) -> Dict[str, QuoteLite]:
    """
    Fetch quotes for many symbols via the JSON endpoint in small batches.
    Returns mapping for symbols successfully resolved.
    """
    out: Dict[str, QuoteLite] = {}
    buf: List[str] = []
    for s in symbols:
        if not s:
            continue
        buf.append(s)
        if len(buf) >= batch_size:
            out.update(fetch_quote_json(buf))
            buf = []
            if sleep_secs:
                time.sleep(sleep_secs)
    if buf:
        out.update(fetch_quote_json(buf))
    return out
