"""
fundamentals.py
────────────────
Fundamental signals for stocks and crypto with graceful fallbacks.

Goals
=====
• Provide a small, dependency-light layer that can:
  - Pull *practical* stock fundamentals from yfinance (P/E, debt, margins, growth).
  - Pull *proxy* crypto fundamentals from CoinGecko (dev & network activity; tokenomics).
• Normalize heterogeneous fields to comparable 0..1 scores.
• Be resilient: return partial metrics when some sources are unavailable.
• Keep interfaces stable for both "multi-asset screen" and "single-asset deep dive".

Key exports
===========
- get_stock_fundamentals(symbol: str) -> Dict[str, Any]
- get_crypto_fundamentals(asset_id_or_symbol: str) -> Dict[str, Any]
- score_stock_fundamentals(metrics: Dict[str, Any]) -> float   # 0..1 composite
- score_crypto_fundamentals(metrics: Dict[str, Any]) -> float  # 0..1 composite
- analyze_stock_fundamentals(symbols: list[str]) -> List[Dict[str, Any]]
- analyze_crypto_fundamentals(ids_or_symbols: list[str]) -> List[Dict[str, Any]]

Notes
=====
• yfinance ‘info’ fields vary by version/provider; we defensively try multiple sources.
• CoinGecko ‘/coins/{id}’ is preferred. If the caller passed a ticker like "btc" or "BTC",
  we attempt a small map to common IDs (e.g., "bitcoin"). You can extend COINGECKO_MAP.
• All numeric outputs are cast to native Python floats/ints to be JSON-safe.

This module is intentionally self-contained so it can be imported by:
  - CLI flow (multi-asset recommendations)
  - Future web API (single-asset “analyze now” tool)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math
import time
import os

import requests
import yfinance as yf

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = 15
USER_AGENT = {"User-Agent": "TA/1.0 (fundamentals module)"}

# Optional: If you later add keyed providers (e.g., TwelveData fundamentals),
# read keys here to avoid hard-coding across modules.
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")

# Minimal symbol→CoinGecko id hints. Extend as needed.
COINGECKO_MAP = {
    "btc": "bitcoin",
    "xbt": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "ada": "cardano",
    "bnb": "binancecoin",
    "xrp": "ripple",
    "doge": "dogecoin",
    "dot": "polkadot",
    "matic": "matic-network",
}


# ────────────────────────────────────────────────────────────
# Data classes
# ────────────────────────────────────────────────────────────

@dataclass
class StockFundamentals:
    symbol: str
    pe: Optional[float] = None               # trailing P/E
    ps: Optional[float] = None               # price/sales (ttm)
    debt_to_equity: Optional[float] = None   # total debt / total equity
    gross_margin: Optional[float] = None     # %
    operating_margin: Optional[float] = None # %
    net_margin: Optional[float] = None       # %
    revenue_growth_yoy: Optional[float] = None  # %
    earnings_growth_yoy: Optional[float] = None # %
    market_cap: Optional[float] = None       # USD
    # Scaled 0..1 scores (computed later)
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None  # free-form diagnostic fields


@dataclass
class CryptoFundamentals:
    asset: str                                # input id or symbol
    coingecko_id: Optional[str] = None
    dev_stars: Optional[int] = None
    dev_forks: Optional[int] = None
    dev_commits_4w: Optional[int] = None
    dev_contributors_4w: Optional[int] = None
    onchain_tx_volume_24h: Optional[float] = None
    community_score: Optional[float] = None
    public_interest_score: Optional[float] = None
    market_cap: Optional[float] = None
    circulating_supply: Optional[float] = None
    # Scaled 0..1 score
    score: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b in (None, 0, 0.0):
            return None
        return float(a) / float(b)
    except Exception:
        return None


def _clip01(x: Optional[float]) -> Optional[float]:
    if x is None or math.isnan(x):
        return None
    return max(0.0, min(1.0, float(x)))


def _z_to_01(z: Optional[float], lo: float = -2.0, hi: float = 2.0) -> Optional[float]:
    """Map a z-like value into 0..1 with soft clipping."""
    if z is None:
        return None
    return _clip01((float(z) - lo) / (hi - lo))


def _normalize_growth_pct(pct: Optional[float], clamp: Tuple[float, float] = (-50.0, 100.0)) -> Optional[float]:
    """Turn -50%..+100% into 0..1; outliers clipped."""
    if pct is None:
        return None
    p = max(clamp[0], min(clamp[1], float(pct)))
    return _clip01((p - clamp[0]) / (clamp[1] - clamp[0]))


def _normalize_ratio_inverse_better(x: Optional[float], clamp: Tuple[float, float] = (0.0, 60.0)) -> Optional[float]:
    """
    For ratios where LOWER is better (e.g., P/E, P/S, Debt/Equity).
    Map clamp range to 1..0 (inverted), then clip to 0..1.
    """
    if x is None:
        return None
    lo, hi = clamp
    v = max(lo, min(hi, float(x)))
    # invert: lower -> closer to 1.0
    return _clip01(1.0 - (v - lo) / (hi - lo))


def _normalize_margin_pct(margin: Optional[float], clamp: Tuple[float, float] = (0.0, 50.0)) -> Optional[float]:
    """Map 0..50% to 0..1 (higher better)."""
    if margin is None:
        return None
    lo, hi = clamp
    v = max(lo, min(hi, float(margin)))
    return _clip01((v - lo) / (hi - lo))


def _try_get(info: Dict[str, Any], *keys, default=None):
    """Multi-key getter for yfinance info dict variant fields."""
    for k in keys:
        if k in info and info[k] not in (None, "", "None"):
            return info[k]
    return default


# ────────────────────────────────────────────────────────────
# STOCK FUNDAMENTALS
# ────────────────────────────────────────────────────────────

def get_stock_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Pulls practical stock fundamentals from yfinance, with multiple fallbacks:
    - Try .get_info() / .info fields (varies by yfinance version/provider)
    - Try .fast_info when possible (market cap, etc.)
    - Growth estimates are noisy; we compute normalized scores when available.

    Returns a dict ready for scoring (numbers in basic Python types).
    """
    tkr = yf.Ticker(symbol)

    # Gather info dictionaries defensively
    info: Dict[str, Any] = {}
    try:
        # Newer yfinance:
        info = tkr.get_info() or {}
    except Exception:
        try:
            info = tkr.info or {}
        except Exception:
            info = {}

    fast = {}
    try:
        fast = tkr.fast_info or {}
    except Exception:
        fast = {}

    # Basic fields
    pe = _try_get(info, "trailingPE", "trailing_pe", default=None)
    ps = _try_get(info, "priceToSalesTrailing12Months", "price_to_sales_trailing_12_months", default=None)

    # Debt/Equity often absent; try several hints
    debt_to_equity = _try_get(info, "debtToEquity", "totalDebt/totalEquity", "debt_to_equity", default=None)

    # Margins
    gross_margin      = _try_get(info, "grossMargins", "gross_margins", default=None)
    operating_margin  = _try_get(info, "operatingMargins", "operating_margins", default=None)
    net_margin        = _try_get(info, "profitMargins", "profit_margins", default=None)

    # Growth – try YoY approximations if exposed
    revenue_growth_yoy  = _try_get(info, "revenueGrowth", "revenue_growth", default=None)
    earnings_growth_yoy = _try_get(info, "earningsGrowth", "earnings_growth", default=None)

    # Market cap best-effort
    market_cap = _try_get(info, "marketCap", "market_cap", default=None)
    if market_cap is None and "market_cap" in fast:
        market_cap = fast.get("market_cap")

    # Coerce numeric-like strings; ignore if still not numeric
    def _num_or_none(x):
        try:
            return float(x)
        except Exception:
            return None

    result = StockFundamentals(
        symbol=symbol,
        pe=_num_or_none(pe),
        ps=_num_or_none(ps),
        debt_to_equity=_num_or_none(debt_to_equity),
        gross_margin=_num_or_none(gross_margin if gross_margin is None or gross_margin < 1.1 else gross_margin*100),
        operating_margin=_num_or_none(operating_margin if operating_margin is None or operating_margin < 1.1 else operating_margin*100),
        net_margin=_num_or_none(net_margin if net_margin is None or net_margin < 1.1 else net_margin*100),
        revenue_growth_yoy=_num_or_none(revenue_growth_yoy if revenue_growth_yoy is None or revenue_growth_yoy < 1.1 else revenue_growth_yoy*100),
        earnings_growth_yoy=_num_or_none(earnings_growth_yoy if earnings_growth_yoy is None or earnings_growth_yoy < 1.1 else earnings_growth_yoy*100),
        market_cap=_num_or_none(market_cap),
        details={"raw_info_keys": list(info.keys())[:20]},  # small diagnostic slice
    )

    # Compute composite score now
    result.score = score_stock_fundamentals(vars(result))
    return vars(result)


def score_stock_fundamentals(m: Dict[str, Any]) -> float:
    """
    Composite 0..1 score using soft heuristics:
    We prefer lower P/E & P/S, lower debt, higher margins, and positive growth.

    Weights (sum to 1.0):
      P/E 0.15, P/S 0.10, D/E 0.10,
      Gross 0.10, Oper 0.10, Net 0.10,
      RevG 0.15, EPSG 0.20
    """
    w = {
        "pe": 0.15, "ps": 0.10, "debt_to_equity": 0.10,
        "gross_margin": 0.10, "operating_margin": 0.10, "net_margin": 0.10,
        "revenue_growth_yoy": 0.15, "earnings_growth_yoy": 0.20,
    }

    s_pe = _normalize_ratio_inverse_better(m.get("pe"), (0, 60))
    s_ps = _normalize_ratio_inverse_better(m.get("ps"), (0, 20))
    s_de = _normalize_ratio_inverse_better(m.get("debt_to_equity"), (0, 200))

    s_gm = _normalize_margin_pct(m.get("gross_margin"), (0, 60))
    s_om = _normalize_margin_pct(m.get("operating_margin"), (0, 40))
    s_nm = _normalize_margin_pct(m.get("net_margin"), (0, 35))

    s_rg = _normalize_growth_pct(m.get("revenue_growth_yoy"), (-50, 100))
    s_eg = _normalize_growth_pct(m.get("earnings_growth_yoy"), (-50, 100))

    parts = {
        "pe": s_pe, "ps": s_ps, "debt_to_equity": s_de,
        "gross_margin": s_gm, "operating_margin": s_om, "net_margin": s_nm,
        "revenue_growth_yoy": s_rg, "earnings_growth_yoy": s_eg,
    }

    score = 0.0
    total_w = 0.0
    for k, weight in w.items():
        v = parts.get(k)
        if v is not None:
            score += weight * v
            total_w += weight

    return float(score / total_w) if total_w > 0 else 0.0


# ────────────────────────────────────────────────────────────
# CRYPTO FUNDAMENTALS (CoinGecko proxies)
# ────────────────────────────────────────────────────────────

def _resolve_coingecko_id(asset_id_or_symbol: str) -> str:
    a = asset_id_or_symbol.strip().lower()
    if a in COINGECKO_MAP:
        return COINGECKO_MAP[a]
    # If already a proper id (e.g., "bitcoin"), trust it
    return a


def _coingecko_get_coin(coin_id: str) -> Optional[Dict[str, Any]]:
    """
    Read a single coin document from CoinGecko with reduced payload.
    Returns None on failure.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "true",
        "developer_data": "true",
        "sparkline": "false",
    }
    try:
        r = requests.get(url, params=params, headers=USER_AGENT, timeout=REQUEST_TIMEOUT)
        if r.status_code == 429:
            # simple backoff if rate-limited
            time.sleep(1.0)
            r = requests.get(url, params=params, headers=USER_AGENT, timeout=REQUEST_TIMEOUT)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def get_crypto_fundamentals(asset_id_or_symbol: str) -> Dict[str, Any]:
    """
    Pulls *proxy* crypto fundamentals via CoinGecko:
      • developer_data (stars, forks, commits/contributors 4w)
      • market_data (market cap, circulating supply, total_volume)
      • community/public_interest scores (if present)
    Returns numbers in basic Python types, plus a composite 0..1 score.
    """
    cg_id = _resolve_coingecko_id(asset_id_or_symbol)
    doc = _coingecko_get_coin(cg_id)

    # Defaults
    res = CryptoFundamentals(asset=asset_id_or_symbol, coingecko_id=cg_id)

    if not doc:
        # No data – keep None metrics and return 0 score
        res.score = 0.0
        res.details = {"error": "coingecko_unavailable"}
        return vars(res)

    dd = doc.get("developer_data", {}) or {}
    md = doc.get("market_data", {}) or {}
    cd = doc.get("community_data", {}) or {}
    pi = doc.get("public_interest_stats", {}) or {}

    # Developer
    res.dev_stars = _safe_to_int(dd.get("stars"))
    res.dev_forks = _safe_to_int(dd.get("forks"))
    res.dev_commits_4w = _safe_to_int(dd.get("commit_count_4_weeks"))
    res.dev_contributors_4w = _safe_to_int(dd.get("pull_request_contributors"))

    # Market/Tokenomics
    market_cap = _get_nested(md, "market_cap", "usd")
    total_volume = _get_nested(md, "total_volume", "usd")
    circ_supply = md.get("circulating_supply")

    res.market_cap = _safe_to_float(market_cap)
    res.onchain_tx_volume_24h = _safe_to_float(total_volume)
    res.circulating_supply = _safe_to_float(circ_supply)

    # Community / public interest
    # (CoinGecko sometimes has sentiment_votes_up_percentage or other hints)
    # We fold them gently into a 0..1-like representation if present.
    comm_score = _safe_to_float(cd.get("facebook_likes")) or 0.0
    # Normalize community proxies with a loose saturating transform
    res.community_score = _saturating_norm(comm_score, k=20000.0)  # 20k likes ~ 0.63 → soft

    pi_score = _safe_to_float(pi.get("alexa_rank"))
    # Lower alexa rank is better → invert (roughly)
    res.public_interest_score = None
    if pi_score is not None and pi_score > 0:
        res.public_interest_score = _clip01(1.0 - math.log10(pi_score) / 7.0)  # 10^0..10^7

    # Composite score
    res.score = score_crypto_fundamentals(vars(res))
    res.details = {
        "has_developer_data": bool(dd),
        "has_market_data": bool(md),
        "has_community_data": bool(cd),
        "has_public_interest": bool(pi),
    }
    return vars(res)


def _safe_to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_to_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        try:
            # sometimes floats
            return int(float(x))
        except Exception:
            return None


def _get_nested(d: Dict[str, Any], *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
        if cur is None:
            return None
    return cur


def _saturating_norm(value: float, k: float) -> float:
    """
    Map a non-negative value to 0..1 with a simple saturation curve:
        f(x) = x / (x + k)
    """
    v = max(0.0, float(value))
    return v / (v + float(k))


def score_crypto_fundamentals(m: Dict[str, Any]) -> float:
    """
    Composite 0..1 score using soft heuristics:
      • Developer intensity (commits, contributors) → momentum of building
      • Community/public interest → traction
      • On-chain/volume → network/economic activity
      • High market cap is not inherently “better”, but we softly avoid tiny caps

    Weights (sum to 1.0):
      Dev (commits+contributors mix)   0.35
      Activity (tx/volume proxy)       0.25
      Interest (community+public)      0.25
      Scale (market cap saturating)    0.15
    """
    # Dev sub-score
    commits = m.get("dev_commits_4w")
    contrib = m.get("dev_contributors_4w")
    s_commits = _clip01(_saturating_norm(float(commits or 0), k=100.0))          # 100 commits ~0.5
    s_contrib = _clip01(_saturating_norm(float(contrib or 0), k=15.0))           # 15 contributors ~0.5
    s_dev = 0.6 * s_commits + 0.4 * s_contrib

    # Activity (use onchain_tx_volume_24h as proxy; saturate)
    act = m.get("onchain_tx_volume_24h") or 0.0
    s_act = _clip01(_saturating_norm(float(act), k=5e8))  # $500m volume ~0.5 (soft guess)

    # Interest (community + public interest)
    s_comm = m.get("community_score")
    s_pi   = m.get("public_interest_score")
    # If both missing, 0; else average available
    interest_vals = [v for v in [s_comm, s_pi] if v is not None]
    s_int = sum(interest_vals) / len(interest_vals) if interest_vals else 0.0

    # Scale bias (avoid micro-caps): saturate market cap
    mc = m.get("market_cap") or 0.0
    s_scale = _clip01(_saturating_norm(float(mc), k=5e9))  # $5B cap ~0.5 (soft guess)

    score = (
        0.35 * s_dev +
        0.25 * s_act +
        0.25 * s_int +
        0.15 * s_scale
    )
    return float(_clip01(score))


# ────────────────────────────────────────────────────────────
# Batch helpers
# ────────────────────────────────────────────────────────────

def analyze_stock_fundamentals(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Multi-asset convenience: returns list of dicts with computed scores.
    Silently skips symbols that repeatedly fail.
    """
    out: List[Dict[str, Any]] = []
    for s in symbols:
        try:
            out.append(get_stock_fundamentals(s))
        except Exception:
            continue
    return out


def analyze_crypto_fundamentals(ids_or_symbols: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for a in ids_or_symbols:
        try:
            out.append(get_crypto_fundamentals(a))
        except Exception:
            continue
    return out
