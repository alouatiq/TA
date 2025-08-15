"""
twelvedata.py — lightweight adapter for Twelve Data REST API
─────────────────────────────────────────────────────────────
Supports: equities, forex, crypto

Env:
  TWELVEDATA_API_KEY=...   # required

Main helpers:
  - get_quote(symbol, asset_class, exchange=None)
  - get_history(symbol, asset_class, interval="1day", outputsize=100, exchange=None)
  - search_symbols(query, asset_class=None, exchange=None, limit=20)

Notes:
  • symbol normalization is handled (e.g., "EURUSD" → "EUR/USD", "BTCUSDT" → "BTC/USDT")
  • pass `exchange` when you know the local market (e.g., "LSE", "XETRA"). If omitted,
    Twelve Data will resolve the primary listing.
  • intervals follow Twelve Data (1min, 5min, 15min, 1h, 1day, 1week, 1month).
  • returns normalized dicts (floats/ints) and raises RuntimeError on missing API key.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests


# ────────────────────────────────────────────────────────────
#  Config
# ────────────────────────────────────────────────────────────

API_BASE = "https://api.twelvedata.com"
DEFAULT_TIMEOUT = 12  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 0.8


def _get_api_key() -> str:
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY is not set in the environment.")
    return key


# ────────────────────────────────────────────────────────────
#  HTTP core with simple backoff (429/5xx)
# ────────────────────────────────────────────────────────────

def _request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform a GET with API key + retries; return parsed JSON or raise."""
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    params = dict(params or {})
    params["apikey"] = _get_api_key()

    last_err: Optional[str] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            # handle rate-limit politely
            if resp.status_code == 429:
                last_err = "rate_limited"
                sleep_s = BACKOFF_BASE * attempt
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            data = resp.json()

            # Twelve Data returns {"code":"400","message":"..."} for API errors
            if isinstance(data, dict) and "code" in data and "message" in data:
                code = str(data.get("code"))
                msg = str(data.get("message"))
                # retry only for transient messages
                if code in ("400", "401", "404"):
                    # usually not transient; break
                    raise RuntimeError(f"TwelveData error {code}: {msg}")
                last_err = msg
                time.sleep(BACKOFF_BASE * attempt)
                continue

            return data

        except requests.RequestException as e:
            last_err = str(e)
            # transient network? backoff & retry
            time.sleep(BACKOFF_BASE * attempt)

    raise RuntimeError(f"TwelveData request failed after retries: {last_err or 'unknown'}")


# ────────────────────────────────────────────────────────────
#  Symbol normalization helpers
# ────────────────────────────────────────────────────────────

def _norm_fx_symbol(sym: str) -> str:
    """EURUSD → EUR/USD; USDJPY → USD/JPY; EUR/USD stays same."""
    s = sym.replace(" ", "").upper()
    if "/" in s:
        return s
    if len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    # tolerate longer like "EURUSD=X" (Yahoo style)
    s = s.replace("=X", "")
    if len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    return s


def _norm_crypto_symbol(sym: str) -> str:
    """
    BTCUSD → BTC/USD; BTCUSDT → BTC/USDT; ETH/BTC stays same.
    Accepts common CEX forms like 'BTC-USD' or 'BTCUSD'.
    """
    s = sym.replace(" ", "").replace("-", "/").upper()
    if "/" in s:
        return s
    # try split into base/quote by common quote assets
    # prefer the longest quote tickers first
    common_quotes = ("USDT", "USDC", "BUSD", "EUR", "USD", "BTC", "ETH", "GBP", "AUD", "JPY")
    for q in common_quotes:
        if s.endswith(q) and len(s) > len(q):
            return f"{s[:-len(q)]}/{q}"
    return s


def _norm_equity_symbol(sym: str) -> str:
    """Leave as-is (AAPL, RY.TO, BN.PA). Twelve Data accepts many local formats via `exchange` param."""
    return sym.strip()


def _normalize_symbol(sym: str, asset_class: str) -> str:
    ac = asset_class.lower()
    if ac in ("forex", "fx", "currencies", "currency"):
        return _norm_fx_symbol(sym)
    if ac in ("crypto", "cryptocurrency", "digital"):
        return _norm_crypto_symbol(sym)
    return _norm_equity_symbol(sym)


# ────────────────────────────────────────────────────────────
#  Public: search
# ────────────────────────────────────────────────────────────

def search_symbols(query: str,
                   asset_class: Optional[str] = None,
                   exchange: Optional[str] = None,
                   limit: int = 20) -> List[Dict[str, Any]]:
    """
    Use Twelve Data symbol search.
    Returns a list of dicts: {symbol, instrument_name, exchange, country, currency, type}
    """
    params: Dict[str, Any] = {"symbol": query, "outputsize": max(1, min(limit, 100))}
    if exchange:
        params["exchange"] = exchange

    # Twelve Data's /symbol_search doesn't filter by class perfectly; we post-filter.
    data = _request("symbol_search", params)
    items = data.get("data", []) if isinstance(data, dict) else []

    def _class_ok(t: str) -> bool:
        if not asset_class:
            return True
        ac = asset_class.lower()
        t = (t or "").lower()
        if ac in ("equities", "equity", "stock", "stocks"):
            return "stock" in t or t == "etf" or "adr" in t
        if ac in ("forex", "fx", "currency", "currencies"):
            return "forex" in t or "currency" in t or "fx" in t
        if ac in ("crypto", "cryptocurrency", "digital"):
            return "crypto" in t or "cryptocurrency" in t or "digital" in t
        return True

    out: List[Dict[str, Any]] = []
    for it in items:
        t = {
            "symbol": it.get("symbol"),
            "instrument_name": it.get("instrument_name"),
            "exchange": it.get("exchange"),
            "country": it.get("country"),
            "currency": it.get("currency"),
            "type": it.get("type"),
        }
        if t["symbol"] and _class_ok(str(t["type"] or "")):
            out.append(t)

    return out


# ────────────────────────────────────────────────────────────
#  Public: quotes
# ────────────────────────────────────────────────────────────

def get_quote(symbol: str,
              asset_class: str,
              exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Return a normalized quote dict:
      { "symbol", "price", "open", "high", "low", "volume", "currency", "exchange", "datetime" }
    or None if not found.
    """
    sym = _normalize_symbol(symbol, asset_class)
    params: Dict[str, Any] = {"symbol": sym}
    if exchange:
        params["exchange"] = exchange

    data = _request("quote", params)

    # Batch and single look similar; unify to dict
    if isinstance(data, dict) and "symbol" in data:
        payloads = [data]
    elif isinstance(data, dict) and "data" in data:
        payloads = data.get("data", [])
    else:
        payloads = []

    if not payloads:
        return None

    q = payloads[0]
    def _f(x):  # safe float
        try:
            return float(x)
        except Exception:
            return None

    return {
        "symbol": q.get("symbol") or sym,
        "price": _f(q.get("price")),
        "open": _f(q.get("open")),
        "high": _f(q.get("high")),
        "low": _f(q.get("low")),
        "volume": int(float(q.get("volume"))) if q.get("volume") not in (None, "", "NaN") else None,
        "currency": q.get("currency"),
        "exchange": q.get("exchange") or exchange,
        "datetime": q.get("datetime") or q.get("timestamp"),
    }


# ────────────────────────────────────────────────────────────
#  Public: history (time_series)
# ────────────────────────────────────────────────────────────

_VALID_INTERVALS = {
    # intraday
    "1min", "5min", "15min", "30min", "45min",
    # hourly-ish
    "1h", "2h", "4h", "8h",
    # higher TF
    "1day", "1week", "1month"
}


def get_history(symbol: str,
                asset_class: str,
                interval: str = "1day",
                outputsize: int = 120,
                exchange: Optional[str] = None) -> Dict[str, Any]:
    """
    Return normalized OHLCV time-series:
      {
        "symbol": "...",
        "interval": "1day",
        "currency": "...",
        "data": [ { "datetime": "...", "open":.., "high":.., "low":.., "close":.., "volume":.. }, ...]
      }

    outputsize up to ~5000 for paid; we'll cap reasonably here.
    """
    if interval not in _VALID_INTERVALS:
        raise ValueError(f"Unsupported interval '{interval}'. Supported: {sorted(_VALID_INTERVALS)}")

    sym = _normalize_symbol(symbol, asset_class)
    params: Dict[str, Any] = {
        "symbol": sym,
        "interval": interval,
        "outputsize": max(10, min(int(outputsize), 5000)),
        "order": "ASC",
    }
    if exchange:
        params["exchange"] = exchange

    data = _request("time_series", params)

    # Twelve Data returns { meta: {...}, values: [...] } or {status:'error', message:'...'}
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    values = data.get("values", []) if isinstance(data, dict) else []

    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    out_rows: List[Dict[str, Any]] = []
    for v in values:
        out_rows.append({
            "datetime": v.get("datetime"),
            "open": _f(v.get("open")),
            "high": _f(v.get("high")),
            "low": _f(v.get("low")),
            "close": _f(v.get("close")),
            "volume": int(float(v.get("volume"))) if v.get("volume") not in (None, "", "NaN") else None,
        })

    return {
        "symbol": meta.get("symbol") or sym,
        "interval": meta.get("interval") or interval,
        "currency": meta.get("currency"),
        "exchange": meta.get("exchange") or exchange,
        "data": out_rows,
    }


# ────────────────────────────────────────────────────────────
#  Convenience: batch quotes (best effort)
# ────────────────────────────────────────────────────────────

def get_quotes_batch(pairs: List[Tuple[str, str]],
                     exchange: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Batch fetch quotes for multiple (symbol, asset_class) pairs.

    Args:
      pairs: list of (symbol, asset_class), e.g. [("AAPL","equities"),("EURUSD","forex"),("BTCUSDT","crypto")]
    Returns:
      list of normalized quote dicts; silently skips failures.
    """
    if not pairs:
        return []

    # Normalize all symbols first
    norm_syms: List[str] = []
    for sym, ac in pairs:
        norm_syms.append(_normalize_symbol(sym, ac))

    # Twelve Data supports comma-separated symbols on /quote in many cases
    # but mixed classes can be flaky; to be safe we group by apparent style:
    #   - contains '/' → pass as-is
    #   - otherwise pass as-is; TD resolves locally
    joined = ",".join(norm_syms)
    params: Dict[str, Any] = {"symbol": joined}
    if exchange:
        params["exchange"] = exchange

    try:
        data = _request("quote", params)
    except Exception:
        # fallback: sequential one-by-one
        out: List[Dict[str, Any]] = []
        for (sym, ac) in pairs:
            q = None
            try:
                q = get_quote(sym, ac, exchange=exchange)
            except Exception:
                q = None
            if q:
                out.append(q)
        return out

    out: List[Dict[str, Any]] = []
    items = []
    if isinstance(data, dict) and "data" in data:
        items = data.get("data", [])
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "symbol" in data:
        items = [data]

    for q in items:
        def _f(x):
            try:
                return float(x)
            except Exception:
                return None
        out.append({
            "symbol": q.get("symbol"),
            "price": _f(q.get("price")),
            "open": _f(q.get("open")),
            "high": _f(q.get("high")),
            "low": _f(q.get("low")),
            "volume": int(float(q.get("volume"))) if q.get("volume") not in (None, "", "NaN") else None,
            "currency": q.get("currency"),
            "exchange": q.get("exchange") or exchange,
            "datetime": q.get("datetime") or q.get("timestamp"),
        })

    return out
