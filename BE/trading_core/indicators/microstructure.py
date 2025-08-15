"""
microstructure.py
─────────────────
Lightweight market microstructure features used by the scoring layer.

Scope (v1):
- Crypto (public endpoints; no API keys required)
  • Order book snapshot (Binance / Coinbase)
  • Order book imbalance (−1 … +1)
  • Liquidity depth around mid (±X%)
  • Whale trade detection (large notional prints over last N trades)

- Equities (placeholders)
  • True L2 data typically requires paid feeds (NASDAQ TotalView, ARCABook, etc.)
  • We expose typed stubs that return None with a clear diagnostic.

Notes
-----
• All functions are best‑effort, fail‑closed (return None) and include a small
  diagnostics dict to help the caller decide whether to use these signals.
• Price/size math is done in float for simplicity; callers should treat outputs
  as *features*, not precise risk metrics.

Examples (crypto)
-----------------
from trading_core.indicators.microstructure import (
    fetch_crypto_orderbook, compute_orderbook_imbalance, compute_liquidity_depth,
    detect_whale_trades, analyze_microstructure
)

book, dx = fetch_crypto_orderbook("BTC-USD", exchange="binance", limit=100)
imb = compute_orderbook_imbalance(book["bids"], book["asks"])
liq = compute_liquidity_depth(book["bids"], book["asks"], book["mid"], pct_window=0.01)
whales, wdx = detect_whale_trades("BTC-USD", exchange="binance", min_usd=500_000)

result = analyze_microstructure(
    symbol="BTC-USD",
    category="crypto",
    exchange="binance",
    limit=200,
    depth_window_pct=0.01,
    whale_usd=500_000,
)

"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple, Any

import requests

# ────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 10
BINANCE_API = "https://api.binance.com"
COINBASE_API = "https://api.exchange.coinbase.com"  # (a.k.a. Coinbase Advanced Trade / Exchange)
DEFAULT_CRYPTO_EXCHANGE = "binance"  # "binance" | "coinbase"

# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────
def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _dedup_sorted_book(levels: List[Tuple[float, float]], reverse: bool) -> List[Tuple[float, float]]:
    """
    Deduplicate price levels and ensure sorted.
    reverse=True → bids (high → low)
    reverse=False → asks (low → high)
    """
    if not levels:
        return []
    # aggregate by price
    agg: Dict[float, float] = {}
    for p, q in levels:
        if p is None or q is None:
            continue
        agg[p] = agg.get(p, 0.0) + q
    out = sorted(((p, q) for p, q in agg.items() if q > 0), key=lambda t: t[0], reverse=reverse)
    return out


def _mid_from_top(bids: List[Tuple[float, float]], asks: List[Tuple[float, float]]) -> Optional[float]:
    try:
        if not bids or not asks:
            return None
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        if best_bid is None or best_ask is None:
            return None
        return (best_bid + best_ask) / 2.0
    except Exception:
        return None


def _normalize_symbol(symbol: str, exchange: str) -> str:
    """
    Normalize user-friendly symbols into exchange‑specific product ids.

    Inputs we try to accept:
      "BTCUSDT", "BTC/USDT", "BTC-USD", "btc-usdt", "ETHUSD", "eth-usdt", "ETH-USD"

    • Binance  : "BTCUSDT" | "ETHUSDT" (no delimiter, USDT default)
    • Coinbase : "BTC-USD" | "ETH-USD" (dash, quote = USD/USDT/USDC common)
    """
    s = str(symbol).upper().strip().replace(" ", "")
    if exchange == "binance":
        # Already raw?
        if s.endswith("USDT") or s.endswith("BUSD") or s.endswith("USDC") or s.endswith("USD"):
            return s.replace("-", "").replace("/", "")
        # "BTC-USD" or "BTC/USD"
        if "-" in s or "/" in s:
            base, quote = s.replace("/", "-").split("-", 1)
            return f"{base}{quote}"
        # Just base → assume USDT
        if s.isalpha():
            return f"{s}USDT"
        return s
    else:
        # coinbase symbols are dash separated (BTC-USD)
        if "-" in s:
            return s
        if "/" in s:
            return s.replace("/", "-")
        # Heuristics: map *USDT* or *USD* to dash form
        for q in ("USDT", "USDC", "USD", "EUR", "GBP"):
            if s.endswith(q):
                return f"{s[:-len(q)]}-{q}"
        # just base → default USD
        if s.isalpha():
            return f"{s}-USD"
        return s


# ────────────────────────────────────────────────────────────
# Crypto – Order book (Binance / Coinbase)
# ────────────────────────────────────────────────────────────
def fetch_crypto_orderbook(
    symbol: str,
    *,
    exchange: str = DEFAULT_CRYPTO_EXCHANGE,
    limit: int = 100,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Get a *snapshot* of the order book for a crypto symbol from a public endpoint.

    Returns:
        (book | None, diagnostics)
    where book = {
        "bids": [(price, size), ...],         # sorted desc
        "asks": [(price, size), ...],         # sorted asc
        "mid": float | None,
        "ts": unix_ms
    }
    """
    dx: Dict[str, Any] = {"exchange": exchange, "symbol_in": symbol, "limit": limit, "ok": False, "error": None}
    exchange = (exchange or DEFAULT_CRYPTO_EXCHANGE).lower().strip()
    try:
        if exchange == "binance":
            sym = _normalize_symbol(symbol, "binance")
            # depth endpoint (limit 5..5000; typical: 100/500)
            resp = requests.get(
                f"{BINANCE_API}/api/v3/depth",
                params={"symbol": sym, "limit": min(max(int(limit), 5), 5000)},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            j = resp.json()
            bids = [( _safe_float(p), _safe_float(q) ) for p, q in j.get("bids", [])]
            asks = [( _safe_float(p), _safe_float(q) ) for p, q in j.get("asks", [])]
            bids = _dedup_sorted_book(bids, reverse=True)
            asks = _dedup_sorted_book(asks, reverse=False)
            mid = _mid_from_top(bids, asks)
            book = {"bids": bids, "asks": asks, "mid": mid, "ts": int(time.time() * 1000)}
            dx["ok"] = True
            dx["normalized_symbol"] = sym
            return book, dx

        elif exchange == "coinbase":
            prod = _normalize_symbol(symbol, "coinbase")
            # level=2 returns aggregated levels (by price) – good for imbalance/depth
            resp = requests.get(
                f"{COINBASE_API}/products/{prod}/book",
                params={"level": 2},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            j = resp.json()
            bids = [( _safe_float(p), _safe_float(q) ) for p, q, _ in j.get("bids", [])]
            asks = [( _safe_float(p), _safe_float(q) ) for p, q, _ in j.get("asks", [])]
            bids = _dedup_sorted_book(bids, reverse=True)
            asks = _dedup_sorted_book(asks, reverse=False)
            mid = _mid_from_top(bids, asks)
            book = {"bids": bids, "asks": asks, "mid": mid, "ts": int(time.time() * 1000)}
            dx["ok"] = True
            dx["normalized_symbol"] = prod
            return book, dx

        else:
            dx["error"] = f"Unsupported exchange '{exchange}'. Use 'binance' or 'coinbase'."
            return None, dx

    except Exception as e:
        dx["error"] = str(e)
        return None, dx


def compute_orderbook_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    *,
    top_n: int = 20,
) -> Optional[float]:
    """
    Compute a bounded imbalance signal in [−1, +1] using the top N levels.

    I = (sum_bid_sizes - sum_ask_sizes) / (sum_bid_sizes + sum_ask_sizes)

    • +1  → all size on bids (strong buy pressure)
    • −1  → all size on asks (strong sell pressure)
    •  0  → balanced
    """
    try:
        if not bids or not asks:
            return None
        b = sum(q for _, q in bids[:top_n] if q is not None and q > 0)
        a = sum(q for _, q in asks[:top_n] if q is not None and q > 0)
        denom = (a + b)
        if denom <= 0:
            return None
        return (b - a) / denom
    except Exception:
        return None


def compute_liquidity_depth(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    mid: Optional[float],
    *,
    pct_window: float = 0.01,
) -> Optional[Dict[str, float]]:
    """
    Aggregate notional depth within ±pct_window around MID.

    Returns:
       {
         "buy_depth_notional": USD within [mid*(1-p), mid],
         "sell_depth_notional": USD within [mid, mid*(1+p)]
       }

    If `mid` is None, returns None.
    """
    if mid is None or not bids or not asks or pct_window <= 0:
        return None
    try:
        lo = mid * (1.0 - pct_window)
        hi = mid * (1.0 + pct_window)

        buy_notional = 0.0
        for p, q in bids:
            if p is None or q is None or p < lo:
                break  # bids are sorted high→low; below window → stop
            if p <= mid:
                buy_notional += p * q

        sell_notional = 0.0
        for p, q in asks:
            if p is None or q is None or p > hi:
                break  # asks are sorted low→high; above window → stop
            if p >= mid:
                sell_notional += p * q

        return {
            "buy_depth_notional": float(buy_notional),
            "sell_depth_notional": float(sell_notional),
        }
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# Crypto – Whale trades (Binance)
# ────────────────────────────────────────────────────────────
def detect_whale_trades(
    symbol: str,
    *,
    exchange: str = DEFAULT_CRYPTO_EXCHANGE,
    min_usd: float = 200_000.0,
    limit: int = 1000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Scan recent trades and return those above `min_usd` notional.

    Returns:
      (whales, diagnostics)
    where each whale is: {
        "price": float, "qty": float, "notional": float, "side": "buy"/"sell", "ts": int
      }

    • Binance: /api/v3/aggTrades (no side field → we infer by comparing p to previous)
    • Coinbase: /products/{product_id}/trades (has side)
    """
    dx: Dict[str, Any] = {"exchange": exchange, "symbol_in": symbol, "min_usd": min_usd, "ok": False, "error": None}
    exchange = (exchange or DEFAULT_CRYPTO_EXCHANGE).lower().strip()
    whales: List[Dict[str, Any]] = []

    try:
        if exchange == "binance":
            sym = _normalize_symbol(symbol, "binance")
            resp = requests.get(
                f"{BINANCE_API}/api/v3/aggTrades",
                params={"symbol": sym, "limit": min(max(int(limit), 1), 1000)},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            j = resp.json()

            # infer side: compare price to previous agg trade price (rough heuristic)
            prev_p = None
            for t in j:
                p = _safe_float(t.get("p"))
                q = _safe_float(t.get("q"))
                ts = int(t.get("T", 0))
                if p is None or q is None:
                    continue
                notional = p * q
                if notional >= min_usd:
                    side = "buy"
                    if prev_p is not None:
                        side = "buy" if p >= prev_p else "sell"
                    whales.append({"price": float(p), "qty": float(q), "notional": float(notional), "side": side, "ts": ts})
                prev_p = p

            dx["ok"] = True
            dx["normalized_symbol"] = sym
            return whales, dx

        elif exchange == "coinbase":
            prod = _normalize_symbol(symbol, "coinbase")
            resp = requests.get(
                f"{COINBASE_API}/products/{prod}/trades",
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            j = resp.json()

            for t in j:
                p = _safe_float(t.get("price"))
                q = _safe_float(t.get("size"))
                side = t.get("side", "buy")
                ts = int(time.time() * 1000)  # endpoint returns sequence/time; we stamp now
                if p is None or q is None:
                    continue
                notional = p * q
                if notional >= min_usd:
                    whales.append({"price": float(p), "qty": float(q), "notional": float(notional), "side": side, "ts": ts})

            dx["ok"] = True
            dx["normalized_symbol"] = prod
            return whales, dx

        else:
            dx["error"] = f"Unsupported exchange '{exchange}'. Use 'binance' or 'coinbase'."
            return [], dx

    except Exception as e:
        dx["error"] = str(e)
        return [], dx


# ────────────────────────────────────────────────────────────
# Equities – Placeholders (paid L2 needed for true microstructure)
# ────────────────────────────────────────────────────────────
def fetch_equity_orderbook(symbol: str, *, venue: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Placeholder for future integration (IEX Cloud TOPS/DEEP, Polygon.io, Nasdaq TotalView, etc.).

    Returns (None, diagnostics).
    """
    dx = {
        "ok": False,
        "symbol_in": symbol,
        "venue": venue,
        "error": "Level 2 equity order book requires a paid data source (not integrated in OSS build).",
    }
    return None, dx


def detect_equity_whales(symbol: str, *, min_usd: float = 500_000.0, venue: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Placeholder for future prints tape (TRF/TAQ) analysis – needs paid market data.

    Returns ([], diagnostics).
    """
    dx = {
        "ok": False,
        "symbol_in": symbol,
        "venue": venue,
        "min_usd": min_usd,
        "error": "Block/whale trade detection for equities requires paid prints/tape feed.",
    }
    return [], dx


# ────────────────────────────────────────────────────────────
# High-level wrapper
# ────────────────────────────────────────────────────────────
def analyze_microstructure(
    *,
    symbol: str,
    category: str,  # 'crypto' | 'equities'
    exchange: Optional[str] = None,          # for crypto: 'binance' | 'coinbase'
    limit: int = 200,                        # order book levels/trades window
    depth_window_pct: float = 0.01,          # ±1% around mid
    whale_usd: float = 200_000.0,            # threshold for whale prints
) -> Dict[str, Any]:
    """
    Convenience call that fetches microstructure features and returns a compact feature dict:

    {
      "ok": bool,
      "category": "crypto" | "equities",
      "exchange": "binance" | "coinbase" | None,
      "imbalance": float | None,              # −1 … +1
      "buy_depth_notional": float | None,
      "sell_depth_notional": float | None,
      "whale_count": int | None,
      "whale_notional_sum": float | None,
      "diagnostics": {...}
    }
    """
    out: Dict[str, Any] = {
        "ok": False,
        "category": category,
        "exchange": exchange,
        "imbalance": None,
        "buy_depth_notional": None,
        "sell_depth_notional": None,
        "whale_count": None,
        "whale_notional_sum": None,
        "diagnostics": {},
    }

    category = (category or "").lower().strip()

    if category == "crypto":
        ex = (exchange or DEFAULT_CRYPTO_EXCHANGE).lower()
        # 1) Order book
        book, bdx = fetch_crypto_orderbook(symbol, exchange=ex, limit=limit)
        out["diagnostics"]["orderbook"] = bdx
        if book:
            imb = compute_orderbook_imbalance(book["bids"], book["asks"], top_n=min(50, limit))
            liq = compute_liquidity_depth(book["bids"], book["asks"], book["mid"], pct_window=depth_window_pct)
            out["imbalance"] = imb
            if liq:
                out["buy_depth_notional"] = liq["buy_depth_notional"]
                out["sell_depth_notional"] = liq["sell_depth_notional"]

        # 2) Whale prints
        whales, wdx = detect_whale_trades(symbol, exchange=ex, min_usd=whale_usd, limit=limit)
        out["diagnostics"]["whales"] = wdx
        if whales:
            out["whale_count"] = len(whales)
            out["whale_notional_sum"] = float(sum(t["notional"] for t in whales))

        out["ok"] = any([out["imbalance"] is not None, out["buy_depth_notional"], out["whale_count"]])

    elif category == "equities":
        # Placeholders – nothing actionable without a paid feed, but keep structure consistent
        book, bdx = fetch_equity_orderbook(symbol)
        whales, wdx = detect_equity_whales(symbol)
        out["diagnostics"]["orderbook"] = bdx
        out["diagnostics"]["whales"] = wdx
        out["ok"] = False  # explicitly false until we integrate a provider

    else:
        out["diagnostics"]["error"] = f"Unsupported category '{category}'. Use 'crypto' or 'equities'."
        out["ok"] = False

    return out
