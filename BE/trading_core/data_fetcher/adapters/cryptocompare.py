"""
trading_core.data_fetcher.adapters.cryptocompare
────────────────────────────────────────────────
Thin wrapper around the CryptoCompare REST API.

Features
• Reads API key from env: CRYPTOCOMPARE_API_KEY (or via constructor)
• Safe, minimal surface:
    - prices:   /data/pricemultifull
    - hist_hour / hist_day: /data/v2/histohour, /data/v2/histoday
• Retries with exponential backoff for transient errors (HTTP 5xx / 429)
• No hard‑coded coin lists — callers pass any symbols (fsyms/tsyms)
• Normalized outputs (floats, ints), graceful None on missing fields

Notes
• Docs: https://min-api.cryptocompare.com/documentation
• Rate limits vary; we do small, respectful retry/backoff.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import requests

LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15
_BASE_URL = "https://min-api.cryptocompare.com"


class CryptoCompareError(RuntimeError):
    """Raised for non-retryable API issues or when retries are exhausted."""


def _mk_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Apikey {api_key}"
    return headers


def _request_json(
    path: str,
    *,
    params: Dict[str, Any],
    api_key: Optional[str],
    timeout: int = _DEFAULT_TIMEOUT,
    max_retries: int = 3,
    backoff_base: float = 0.8,
) -> Dict[str, Any]:
    """
    Make a GET request with simple retry/backoff for 5xx and 429.
    Raises CryptoCompareError on final failure or 4xx (except 429).
    """
    url = _BASE_URL + path
    headers = _mk_headers(api_key)

    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        except requests.RequestException as e:
            # Network-level failures: retry
            if attempt < max_retries:
                sleep_s = backoff_base * (2 ** attempt)
                LOG.warning("CryptoCompare network error (%s). Retrying in %.2fs...", e, sleep_s)
                time.sleep(sleep_s)
                continue
            raise CryptoCompareError(f"Network error: {e}") from e

        # Too many requests: retry with backoff
        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt < max_retries:
                sleep_s = backoff_base * (2 ** attempt)
                LOG.warning("CryptoCompare HTTP %s. Retrying in %.2fs...", resp.status_code, sleep_s)
                time.sleep(sleep_s)
                continue
            raise CryptoCompareError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        # Other 4xx → do not retry
        if 400 <= resp.status_code < 500:
            raise CryptoCompareError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise CryptoCompareError(f"Invalid JSON: {e}") from e

        # Some v2 endpoints embed status
        # e.g. {"Response":"Success","Data":{...}} or {"Response":"Error","Message":"..."}
        if isinstance(data, dict) and data.get("Response") == "Error":
            msg = data.get("Message", "Unknown API error")
            # Can retry some server-ish error messages if allowed
            if attempt < max_retries and resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = backoff_base * (2 ** attempt)
                LOG.warning("CryptoCompare API error '%s'. Retrying in %.2fs...", msg, sleep_s)
                time.sleep(sleep_s)
                continue
            raise CryptoCompareError(f"API error: {msg}")

        return data

    # Should not reach here
    raise CryptoCompareError("Retries exhausted")


# ────────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────────

def get_price_multi(
    fsyms: List[str],
    tsyms: List[str] = ["USD"],
    *,
    api_key: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    /data/pricemultifull
    Return raw+display fields for multiple from‑symbol to multiple to‑symbol pairs.

    Output (simplified):
    {
      "<FROM>": {
        "<TO>": {
          "PRICE": float,
          "VOLUME24HOUR": float,
          "CHANGEPCT24HOUR": float,
          "HIGH24HOUR": float,
          "LOW24HOUR": float,
          ...
        },
        ...
      },
      ...
    }
    """
    key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    params = {"fsyms": ",".join(sorted(set([s.upper() for s in fsyms]))),
              "tsyms": ",".join(sorted(set([s.upper() for s in tsyms])))}

    data = _request_json("/data/pricemultifull", params=params, api_key=key, timeout=timeout)

    # CryptoCompare returns {"RAW": {...}, "DISPLAY": {...}}
    raw = data.get("RAW", {}) if isinstance(data, dict) else {}
    # ensure floats where possible
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fsym, by_to in raw.items():
        out[fsym] = {}
        for tsym, payload in (by_to or {}).items():
            # pick a few key fields + carry the whole dict as needed by callers
            norm: Dict[str, float] = {}
            for k, v in (payload or {}).items():
                # Keep numeric values only
                if isinstance(v, (int, float)):
                    norm[k] = float(v)
            out[fsym][tsym] = norm
    return out


def get_hist_hour(
    fsym: str,
    tsym: str = "USD",
    *,
    limit: int = 200,
    aggregate: int = 1,
    api_key: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> List[Dict[str, float]]:
    """
    /data/v2/histohour
    Returns up to 2000 hourly OHLCV bars (limit<=2000). Default 200.

    Output (list of dicts):
    [{"time": 1710000000, "open":..., "high":..., "low":..., "close":..., "volumefrom":..., "volumeto":...}, ...]
    """
    key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    params = {
        "fsym": fsym.upper(),
        "tsym": tsym.upper(),
        "limit": int(limit),
        "aggregate": int(aggregate),
    }
    data = _request_json("/data/v2/histohour", params=params, api_key=key, timeout=timeout)
    d = (data or {}).get("Data", {})
    points = d.get("Data", []) if isinstance(d, dict) else []
    out: List[Dict[str, float]] = []
    for p in points:
        try:
            out.append({
                "time": float(p.get("time", 0.0)),
                "open": float(p.get("open", 0.0)),
                "high": float(p.get("high", 0.0)),
                "low":  float(p.get("low", 0.0)),
                "close": float(p.get("close", 0.0)),
                "volumefrom": float(p.get("volumefrom", 0.0)),
                "volumeto": float(p.get("volumeto", 0.0)),
            })
        except Exception:
            # skip malformed rows
            continue
    return out


def get_hist_day(
    fsym: str,
    tsym: str = "USD",
    *,
    limit: int = 200,
    aggregate: int = 1,
    api_key: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> List[Dict[str, float]]:
    """
    /data/v2/histoday
    Returns up to 2000 daily OHLCV bars (limit<=2000). Default 200.

    Output format mirrors get_hist_hour().
    """
    key = api_key or os.getenv("CRYPTOCOMPARE_API_KEY")
    params = {
        "fsym": fsym.upper(),
        "tsym": tsym.upper(),
        "limit": int(limit),
        "aggregate": int(aggregate),
    }
    data = _request_json("/data/v2/histoday", params=params, api_key=key, timeout=timeout)
    d = (data or {}).get("Data", {})
    points = d.get("Data", []) if isinstance(d, dict) else []
    out: List[Dict[str, float]] = []
    for p in points:
        try:
            out.append({
                "time": float(p.get("time", 0.0)),
                "open": float(p.get("open", 0.0)),
                "high": float(p.get("high", 0.0)),
                "low":  float(p.get("low", 0.0)),
                "close": float(p.get("close", 0.0)),
                "volumefrom": float(p.get("volumefrom", 0.0)),
                "volumeto": float(p.get("volumeto", 0.0)),
            })
        except Exception:
            continue
    return out


# Convenience: build a compact price row (for your fetchers) --------------------

def to_row_from_price(
    raw_multi: Dict[str, Dict[str, Dict[str, float]]],
    fsym: str,
    tsym: str = "USD",
) -> Optional[Dict[str, Any]]:
    """
    Convert a pricemultifull result into a compact dict used elsewhere:
    {"asset": "btc", "symbol": "BTC", "price": 68000.0, "volume": 1.2e9, "day_range_pct": 3.1}

    Returns None if missing.
    """
    fsym_u, tsym_u = fsym.upper(), tsym.upper()
    try:
        payload = raw_multi[fsym_u][tsym_u]
    except Exception:
        return None

    price = payload.get("PRICE")
    vol24 = payload.get("TOTALVOLUME24H", payload.get("VOLUME24HOUR", 0.0))
    chg24 = payload.get("CHANGEPCT24HOUR", 0.0)
    high24 = payload.get("HIGH24HOUR")
    low24  = payload.get("LOW24HOUR")

    # Prefer CHANGE%24H when present; else derive from high/low (less ideal)
    if chg24 is None and price and high24 and low24 and price != 0:
        try:
            chg24 = ((high24 - low24) / price) * 100.0
        except Exception:
            chg24 = None

    return {
        "asset": fsym.lower(),
        "symbol": fsym_u,
        "price": float(price) if price is not None else None,
        "volume": float(vol24 or 0.0),
        "day_range_pct": float(chg24 or 0.0),
    }
