# BE/trading_core/data_fetcher/adapters/stooq.py
"""
Stooq adapter
─────────────
Lightweight CSV-based fallback for quotes and daily history.

Endpoints used (documented by Stooq's CSV API):
- Latest quotes:   https://stooq.com/q/l/?s=<sym[,sym2,...]>&f=sd2t2ohlcv&e=csv
- Daily history:   https://stooq.com/q/d/l/?s=<sym>&i=d

Notes
-----
• Symbols are Stooq-formatted (e.g., AAPL.US, RY.TO, 7203.T).
• Helper `yahoo_to_stooq()` maps Yahoo-style symbols to Stooq where possible.
• This adapter keeps zero external state; safe to use as a last-resort fallback.
• Intended for equities / funds / some indices. If a symbol isn't supported by Stooq,
  the functions return None / empty results gracefully.
"""

from __future__ import annotations

import csv
import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import requests

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────

STOOQ_QUOTE_URL = "https://stooq.com/q/l/"
STOOQ_HISTORY_URL = "https://stooq.com/q/d/l/"
REQUEST_TIMEOUT = float(os.getenv("STOOQ_TIMEOUT", "12"))

# User-Agent helps avoid occasional anti-bot trips.
UA = {"User-Agent": os.getenv("HTTP_UA", "Mozilla/5.0 (X11; Linux x86_64) stooq-adapter/1.0")}

# Yahoo → Stooq suffix mapping (best-effort, expandable)
_YF_TO_STOOQ_SUFFIX = {
    "": ".US",        # US
    ".TO": ".TO",     # Toronto
    ".SA": ".SA",     # Brazil B3
    ".MX": ".MX",     # Mexico
    ".BA": ".AR",     # Buenos Aires
    ".SN": ".SN",     # Santiago
    ".CO": ".CO",     # Copenhagen or Colombia? (Stooq uses .CO for Denmark)
    ".LM": ".PE",     # Peru approx
    ".L": ".L",       # London
    ".DE": ".DE",     # XETRA
    ".SW": ".SW",     # SIX
    ".MC": ".MC",     # BME
    ".PA": ".PA",     # Euronext Paris
    ".AS": ".AS",     # Euronext Amsterdam
    ".BR": ".BR",     # Euronext Brussels
    ".LS": ".LS",     # Euronext Lisbon
    ".IR": ".IR",     # Euronext Dublin (approx)
    ".ST": ".ST",     # Stockholm
    ".HE": ".HE",     # Helsinki
    ".OL": ".OL",     # Oslo
    ".WA": ".WA",     # Warsaw
    ".VI": ".VI",     # Vienna
    ".RO": ".RO",     # Bucharest
    ".AT": ".AT",     # Athens (approx mapping)
    ".PR": ".PR",     # Prague
    ".BU": ".BU",     # Budapest
    ".MI": ".MI",     # Borsa Italiana
    ".SR": ".SR",     # Saudi (some support)
    ".JO": ".JO",     # JSE
    ".NS": ".NS",     # NSE India
    ".BO": ".BO",     # BSE India
    ".AX": ".AX",     # ASX
    ".SI": ".SG",     # SGX (Stooq often uses .SG)
    ".KS": ".KS",     # KOSPI
    ".TW": ".TW",     # TWSE
    ".HK": ".HK",     # HKEX
    ".SS": ".SS",     # SSE
    ".SZ": ".SZ",     # SZSE
}

# ────────────────────────────────────────────────────────────
# Dataclasses
# ────────────────────────────────────────────────────────────

@dataclass
class Quote:
    symbol: str
    price: Optional[float]
    high: Optional[float]
    low: Optional[float]
    volume: Optional[int]
    day_range_pct: Optional[float]

@dataclass
class Candle:
    date: datetime
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[int]

# ────────────────────────────────────────────────────────────
# Symbol helpers
# ────────────────────────────────────────────────────────────

def yahoo_to_stooq(yahoo_symbol: str) -> Optional[str]:
    """
    Convert a Yahoo-style symbol into a Stooq-style symbol.
    Examples:
      AAPL          -> AAPL.US
      RY.TO         -> RY.TO
      7203.T        -> 7203.T
      D05.SI        -> D05.SG
    If we can't infer a mapping, return None.
    """
    if not yahoo_symbol or not isinstance(yahoo_symbol, str):
        return None
    s = yahoo_symbol.upper().strip()

    if "." not in s:
        # Assume US
        return f"{s}.US"

    base, suf = s.split(".", 1)
    suf = "." + suf
    mapped = _YF_TO_STOOQ_SUFFIX.get(suf)
    return f"{base}{mapped}" if mapped else f"{base}{suf}"  # last resort: reuse same suffix


# ────────────────────────────────────────────────────────────
# CSV → dict parsing
# ────────────────────────────────────────────────────────────

def _csv_rows(text: str) -> List[Dict[str, str]]:
    if not text:
        return []
    buf = io.StringIO(text)
    reader = csv.DictReader(buf)
    return [row for row in reader]

def _safe_float(v: Optional[str]) -> Optional[float]:
    try:
        if v is None or v == "" or v == "N/A":
            return None
        return float(v)
    except Exception:
        return None

def _safe_int(v: Optional[str]) -> Optional[int]:
    try:
        if v is None or v == "" or v == "N/A":
            return None
        return int(float(v))
    except Exception:
        return None

# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def fetch_quote(stooq_symbol: str) -> Optional[Quote]:
    """
    Get the latest quote for a single Stooq symbol.
    Returns None if unavailable.
    """
    try:
        params = {
            "s": stooq_symbol.lower(),
            "f": "sd2t2ohlcv",  # symbol,date,time,open,high,low,close,volume
            "h": "",
            "e": "csv",
        }
        resp = requests.get(STOOQ_QUOTE_URL, params=params, timeout=REQUEST_TIMEOUT, headers=UA)
        resp.raise_for_status()
        rows = _csv_rows(resp.text)
        if not rows:
            return None
        r = rows[0]
        close = _safe_float(r.get("close"))
        high  = _safe_float(r.get("high"))
        low   = _safe_float(r.get("low"))
        vol   = _safe_int(r.get("volume"))

        day_range_pct = None
        if close and high and low and close != 0:
            try:
                day_range_pct = round(((high - low) / close) * 100, 2)
            except Exception:
                day_range_pct = None

        return Quote(
            symbol=stooq_symbol.upper(),
            price=close,
            high=high,
            low=low,
            volume=vol,
            day_range_pct=day_range_pct,
        )
    except Exception:
        return None


def fetch_quotes_batch(stooq_symbols: Iterable[str]) -> Dict[str, Quote]:
    """
    Batch version of fetch_quote. Stooq supports comma-separated symbols.
    Returns a dict {symbol: Quote} for the ones that succeeded.
    """
    syms = [s.lower() for s in stooq_symbols if s]
    if not syms:
        return {}
    # Stooq accepts quite a few; keep chunks modest (<=100) to be safe
    out: Dict[str, Quote] = {}
    step = 80
    for i in range(0, len(syms), step):
        chunk = syms[i : i + step]
        try:
            params = {
                "s": ",".join(chunk),
                "f": "sd2t2ohlcv",
                "h": "",
                "e": "csv",
            }
            resp = requests.get(STOOQ_QUOTE_URL, params=params, timeout=REQUEST_TIMEOUT, headers=UA)
            resp.raise_for_status()
            for r in _csv_rows(resp.text):
                sym = (r.get("Symbol") or r.get("symbol") or "").upper()
                if not sym:
                    continue
                close = _safe_float(r.get("close"))
                high  = _safe_float(r.get("high"))
                low   = _safe_float(r.get("low"))
                vol   = _safe_int(r.get("volume"))
                day_range_pct = None
                if close and high and low and close != 0:
                    try:
                        day_range_pct = round(((high - low) / close) * 100, 2)
                    except Exception:
                        day_range_pct = None

                out[sym] = Quote(
                    symbol=sym,
                    price=close,
                    high=high,
                    low=low,
                    volume=vol,
                    day_range_pct=day_range_pct,
                )
        except Exception:
            # Skip failing chunk, continue
            continue
    return out


def fetch_history_daily(stooq_symbol: str, limit_days: int = 60) -> List[Candle]:
    """
    Fetch **daily** OHLCV candles for a Stooq symbol (most recent first after slicing).
    Returns up to `limit_days` items (if available). Oldest→Newest order in the result.
    """
    try:
        params = {
            "s": stooq_symbol.lower(),
            "i": "d",      # daily
        }
        resp = requests.get(STOOQ_HISTORY_URL, params=params, timeout=REQUEST_TIMEOUT, headers=UA)
        resp.raise_for_status()
        rows = _csv_rows(resp.text)
        # Rows usually contain: Date,Open,High,Low,Close,Volume
        candles: List[Candle] = []
        for r in rows:
            try:
                d = r.get("Date") or r.get("date")
                dt = datetime.strptime(d, "%Y-%m-%d")
            except Exception:
                # Sometimes header row or malformed row
                continue
            candles.append(
                Candle(
                    date=dt,
                    open=_safe_float(r.get("Open") or r.get("open")),
                    high=_safe_float(r.get("High") or r.get("high")),
                    low=_safe_float(r.get("Low") or r.get("low")),
                    close=_safe_float(r.get("Close") or r.get("close")),
                    volume=_safe_int(r.get("Volume") or r.get("volume")),
                )
            )
        # keep only the last `limit_days`, in chronological order
        if limit_days and len(candles) > limit_days:
            candles = candles[-limit_days:]
        return candles
    except Exception:
        return []


# ────────────────────────────────────────────────────────────
# Convenience: Yahoo symbol in → Stooq quote/history out
# ────────────────────────────────────────────────────────────

def quote_from_yahoo_symbol(yahoo_symbol: str) -> Optional[Quote]:
    stq = yahoo_to_stooq(yahoo_symbol)
    if not stq:
        return None
    return fetch_quote(stq)

def history_from_yahoo_symbol(yahoo_symbol: str, limit_days: int = 60) -> List[Candle]:
    stq = yahoo_to_stooq(yahoo_symbol)
    if not stq:
        return []
    return fetch_history_daily(stq, limit_days=limit_days)
