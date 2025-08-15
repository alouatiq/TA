# BE/trading_core/strategy/strategy_llm.py
"""
Strategy engine (LLM-capable, with robust offline fallback).

This module provides a single entrypoint:
    analyze_market(market_data, budget, market_type, history, sentiment,
                   use_rsi, use_sma, use_sentiment, market, market_context)

Behavior
--------
1) If OPENAI_API_KEY is set AND env USE_LLM=true:
      - Builds a compact JSON prompt with the computed signals and asks the LLM
        for position suggestions (Buy/Sell/Hold + sizes).
      - Falls back to rules engine if the LLM fails for any reason.

2) Otherwise (default):
      - Uses an internal deterministic fusion of indicators across multiple
        timeframes to generate a score in [-1, +1] per asset.
      - Converts the score to an action:
            score > +0.25 → BUY
            score < -0.25 → SELL (only if you hold; CLI may filter)
            else           → HOLD
      - Sizes positions via a simple risk rule (2% risk per trade by default).
      - Targets = price + 2 * ATR_proxy, Stop = price - 1 * ATR_proxy.
        ATR_proxy is derived from returns volatility if full OHLC not available.

Inputs
------
market_data: List[dict]
    Each row should at least have:
        - symbol / asset
        - price: float
        - volume: int (optional)
        - price_history: List[float] (optional but strongly recommended)
        - rsi / sma (optional – will be recomputed if history is present)
sentiment: List[str] | None
    Raw headlines (if sentiment enabled).

market_context: dict
    Should contain:
        - market: str | None  (e.g., "XETRA")
        - region: str | None
        - timezone: "Area/City" string
        - sessions: List[[HH:MM, HH:MM], ...] in market local time

Output
------
List[dict] recommendations, where each dict contains:
    asset, symbol, price, quantity, sell_target, sell_time, sell_time_tz,
    estimated_profit, diagnostics (optional)

This file is designed to run *today* even if scoring/LLM modules are not yet built.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import math
from statistics import pstdev

# Optional OpenAI usage (guarded)
try:
    import openai  # v1+ style packages are named 'openai'; if you use 'openai>=1', adapt below.
except Exception:
    openai = None  # type: ignore

# Try to use indicator helpers if present
try:
    from trading_core.indicators import technical as ti
except Exception:
    ti = None  # type: ignore

# Try to use future scoring modules if present; otherwise we fallback
try:
    from trading_core.scoring import regime as regime_mod
except Exception:
    regime_mod = None  # type: ignore

try:
    from trading_core.scoring import weights as weights_mod
except Exception:
    weights_mod = None  # type: ignore

try:
    from trading_core.scoring import risk as risk_mod
except Exception:
    risk_mod = None  # type: ignore


# ────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────

def _last_session_close_hhmm(market_context: Dict[str, Any]) -> str:
    """
    Return the last session end time "HH:MM" from market_context, or a safe default.
    """
    sessions = market_context.get("sessions") or []
    if isinstance(sessions, list) and sessions:
        try:
            end_times = [str(s[1]) for s in sessions if isinstance(s, (list, tuple)) and len(s) == 2]
            if end_times:
                return end_times[-1]
        except Exception:
            pass
    # Safe default (US-style close in local tz)
    return "16:00"


def _pct(x: float) -> float:
    try:
        return round(100.0 * x, 2)
    except Exception:
        return 0.0


def _safe_get_price(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("price") or 0.0)
    except Exception:
        return 0.0


def _std_annualized(prices: List[float]) -> float:
    """
    Quick volatility proxy from close-to-close log returns.
    Annualizes by sqrt(252). If not enough points, returns a small baseline.
    """
    if not prices or len(prices) < 5:
        return 0.10  # 10% baseline
    rets = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if p0 and p1 and p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    if not rets:
        return 0.10
    sigma = pstdev(rets)
    return float(sigma * math.sqrt(252))  # crude annualization


def _atr_proxy(prices: List[float]) -> float:
    """
    If we don't have OHLC, approximate ATR with close-to-close volatility * price / sqrt(252).
    This is a *very* rough stand-in for intraday range.
    """
    if not prices or len(prices) < 5:
        return 0.01  # 1% fallback of (relative) price scale; we’ll convert later
    ann_vol = _std_annualized(prices)  # annualized vol in ~log-return terms
    # Convert to ~daily move scale
    daily_vol = ann_vol / math.sqrt(252)
    return float(daily_vol)


def _compute_basic_indicators(prices: List[float]) -> Dict[str, Optional[float]]:
    """
    Compute a compact set of indicators on close prices only.
    Uses `trading_core.indicators.technical` if available; otherwise,
    falls back to minimal in-module computations.
    """
    out: Dict[str, Optional[float]] = {
        "sma_20": None, "ema_20": None, "rsi_14": None,
        "macd": None, "macd_signal": None, "adx_14": None,
        "stoch_k": None, "stoch_d": None,
        "bb_hi": None, "bb_lo": None, "atr_proxy": None,
        "volatility": None
    }
    if not prices or len(prices) < 15:
        # Still compute volatility proxy
        out["volatility"] = _std_annualized(prices) if prices else None
        out["atr_proxy"] = _atr_proxy(prices) if prices else None
        return out

    try:
        if ti:
            import pandas as _pd  # local to avoid hard dependency if unused
            s = _pd.Series(prices, dtype="float64")
            out["sma_20"] = float(ti.sma(s, window=20).iloc[-1]) if len(s) >= 20 else None
            out["ema_20"] = float(ti.ema(s, window=20).iloc[-1]) if len(s) >= 20 else None

            rsi = ti.rsi(s, window=14)
            out["rsi_14"] = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None

            macd = ti.macd(s)
            if macd is not None and not macd.empty and {"macd","signal"}.issubset(macd.columns):
                out["macd"] = float(macd["macd"].iloc[-1])
                out["macd_signal"] = float(macd["signal"].iloc[-1])

            # ADX/Stoch/Bbands need more than closes in a perfect world; we just approximate using closes:
            # BB: close-based std bands
            bb = ti.bbands_from_close_only(s, window=20) if hasattr(ti, "bbands_from_close_only") else None
            if bb is not None:
                out["bb_hi"] = float(bb["bb_high"].iloc[-1])
                out["bb_lo"] = float(bb["bb_low"].iloc[-1])

            out["volatility"] = _std_annualized(prices)
            out["atr_proxy"] = _atr_proxy(prices)
        else:
            # Lightweight fallbacks
            try:
                sma20 = sum(prices[-20:]) / 20.0 if len(prices) >= 20 else None
                out["sma_20"] = float(sma20) if sma20 else None
            except Exception:
                pass

            # Exponential MA (alpha=2/(n+1))
            try:
                if len(prices) >= 20:
                    alpha = 2.0 / (20.0 + 1.0)
                    ema = prices[-20]
                    for p in prices[-19:]:
                        ema = alpha * p + (1 - alpha) * ema
                    out["ema_20"] = float(ema)
            except Exception:
                pass

            # RSI(14) simple
            try:
                gains = []
                losses = []
                for i in range(len(prices) - 14, len(prices)):
                    if i <= 0:
                        continue
                    diff = prices[i] - prices[i - 1]
                    if diff >= 0:
                        gains.append(diff)
                    else:
                        losses.append(abs(diff))
                avg_gain = sum(gains) / 14.0 if gains else 0.0
                avg_loss = sum(losses) / 14.0 if losses else 0.0
                if avg_loss == 0:
                    out["rsi_14"] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    out["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
            except Exception:
                pass

            out["volatility"] = _std_annualized(prices)
            out["atr_proxy"] = _atr_proxy(prices)
    except Exception:
        # Never block; provide at least vol & ATR proxy
        out["volatility"] = _std_annualized(prices)
        out["atr_proxy"] = _atr_proxy(prices)

    return out


def _infer_regime_from_indicators(sig: Dict[str, Optional[float]]) -> str:
    """
    Very compact regime detector:
        - Volatility high  → 'volatile'
        - MACD>Signal & EMA>SMA or RSI>55 → 'bull'
        - Opposite → 'bear'
        - Otherwise → 'range'
    """
    vol = sig.get("volatility") or 0.15
    ema, sma = sig.get("ema_20"), sig.get("sma_20")
    macd, macds = sig.get("macd"), sig.get("macd_signal")
    rsi = sig.get("rsi_14")

    if vol >= 0.35:
        return "volatile"
    bull_votes = 0
    bear_votes = 0

    if ema and sma:
        if ema > sma:
            bull_votes += 1
        else:
            bear_votes += 1
    if macd is not None and macds is not None:
        if macd > macds:
            bull_votes += 1
        else:
            bear_votes += 1
    if rsi is not None:
        if rsi >= 55:
            bull_votes += 1
        elif rsi <= 45:
            bear_votes += 1

    if bull_votes > bear_votes:
        return "bull"
    if bear_votes > bull_votes:
        return "bear"
    return "range"


def _fuse_score(sig: Dict[str, Optional[float]], regime: str, sentiment_score: Optional[float]) -> float:
    """
    Combine signals to a score in [-1, +1] with regime-adaptive weights.
    Uses scoring.weights if available; otherwise a reasonable default.
    """
    if weights_mod and hasattr(weights_mod, "weighted_score"):
        try:
            return float(weights_mod.weighted_score(sig, regime, sentiment_score))
        except Exception:
            pass

    # Default deterministic weighting (simple but effective)
    w = {
        "bull":   {"trend": 0.45, "momentum": 0.25, "vol": -0.10, "sent": 0.20},
        "bear":   {"trend": -0.40, "momentum": -0.20, "vol": -0.10, "sent": 0.15},
        "range":  {"trend": 0.10, "momentum": 0.35, "vol": -0.05, "sent": 0.20},
        "volatile":{"trend": 0.10, "momentum": 0.20, "vol": -0.25, "sent": 0.15},
    }.get(regime, {"trend": 0.30, "momentum": 0.25, "vol": -0.10, "sent": 0.15})

    # Trend proxy: EMA vs SMA & MACD signal
    trend = 0.0
    ema, sma = sig.get("ema_20"), sig.get("sma_20")
    macd, macds = sig.get("macd"), sig.get("macd_signal")
    if ema and sma:
        trend += (1.0 if ema > sma else -1.0) * 0.6
    if macd is not None and macds is not None:
        trend += (1.0 if macd > macds else -1.0) * 0.4

    # Momentum proxy: RSI centered at 50 (normalize to [-1,1])
    momentum = 0.0
    rsi = sig.get("rsi_14")
    if rsi is not None:
        momentum = max(-1.0, min(1.0, (rsi - 50.0) / 25.0))  # RSI 25→-1, 75→+1

    # Volatility penalty (more vol → more negative)
    vol = sig.get("volatility") or 0.15
    vol_pen = min(1.0, max(0.0, (vol - 0.15) / 0.25))  # 15% no penalty up to 40% worst
    vol_component = -vol_pen

    # Sentiment (already normalized to [-1, 1] if our indicators.sentiment is used)
    sent_component = 0.0
    if isinstance(sentiment_score, (int, float)):
        sent_component = max(-1.0, min(1.0, float(sentiment_score)))

    score = (
        w["trend"] * trend +
        w["momentum"] * momentum +
        w["vol"] * vol_component +
        w["sent"] * sent_component
    )
    # Clamp
    return max(-1.0, min(1.0, score))


def _choose_action(score: float) -> str:
    if score > 0.25:
        return "Buy"
    if score < -0.25:
        return "Sell"
    return "Hold"


def _position_size(
    price: float,
    budget: float,
    atr_proxy: float,
    risk_per_trade: float = 0.02,
    min_qty: int = 1,
) -> Tuple[int, float, float]:
    """
    Returns (qty, stop_price, target_price).
    ATR proxy is *relative* (e.g., 0.02 ~ 2% daily move). If ATR is missing, use 2%.
    stop distance = 1 * ATR; target distance = 2 * ATR by default.
    """
    if price <= 0:
        return 0, 0.0, 0.0

    rel_atr = atr_proxy if (atr_proxy and atr_proxy > 0) else 0.02
    stop_dist = price * rel_atr * 1.0
    tgt_dist  = price * rel_atr * 2.0

    # Risk per trade budget
    risk_cap = max(0.0, float(budget) * float(risk_per_trade))
    if stop_dist <= 0:
        qty = 0
    else:
        qty = int(max(min_qty, math.floor(risk_cap / stop_dist)))

    # Don’t overshoot capital
    if qty * price > budget:
        qty = int(budget // price)

    stop_price   = max(0.01, price - stop_dist)
    target_price = price + tgt_dist
    return qty, stop_price, target_price


def _sentiment_score_from_headlines(headlines: Optional[List[str]]) -> Optional[float]:
    """
    Try to derive a [-1,1] sentiment score from headlines using indicator helpers if present.
    If unavailable, map to neutral (0.0) unless empty.
    """
    if not headlines:
        return None
    # Try indicators layer if present
    try:
        from trading_core.indicators import sentiment as senti_ind
        if hasattr(senti_ind, "score_headlines"):
            return float(senti_ind.score_headlines(headlines))
        if hasattr(senti_ind, "aggregate_sentiment"):
            agg = senti_ind.aggregate_sentiment(headlines)
            sc = agg.get("score")
            return float(sc) if sc is not None else 0.0
    except Exception:
        pass
    # Neutral-ish default
    return 0.0


# ────────────────────────────────────────────────────────────
# Optional LLM (guarded, JSON-only)
# ────────────────────────────────────────────────────────────

def _llm_enabled() -> bool:
    if openai is None:
        return False
    use = os.getenv("USE_LLM", "").strip().lower() in {"1", "true", "yes"}
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_1")
    return bool(use and key)


def _llm_analyze(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """
    Minimal LLM call that asks for JSON recommendations.
    If any error occurs, returns None so caller can fall back.
    """
    if not _llm_enabled():
        return None

    try:
        # OpenAI SDK v1+ style; adapt if you’re on a different version.
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = (
            "You are a trading assistant. Return STRICT JSON with a list of recommendations.\n"
            "Each item: {asset, symbol, action, confidence, quantity, price, sell_target, sell_time, "
            "sell_time_tz, estimated_profit, key_reasons}."
        )
        user_prompt = (
            "Analyze this market snapshot and produce actionable recommendations. "
            "Weigh technical/fundamental/sentiment/microstructure signals conservatively. "
            "Avoid overfitting; consider multiple timeframes."
        )

        # We keep the payload compact to avoid token bloat.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": f"JSON_INPUT:\n{payload}"}
        ]
        # Use responses API if available; else fall back to chat.completions
        try:
            rsp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            txt = rsp.choices[0].message.content or "{}"
        except Exception:
            # Very old SDK fallback:
            rsp = openai.ChatCompletion.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
                temperature=0.2,
            )
            txt = rsp["choices"][0]["message"]["content"]

        import json
        parsed = json.loads(txt)
        if isinstance(parsed, dict) and "recommendations" in parsed:
            recs = parsed["recommendations"]
            if isinstance(recs, list):
                return recs
        if isinstance(parsed, list):
            return parsed
        return None
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# Public Entry
# ────────────────────────────────────────────────────────────

def analyze_market(
    *,
    market_data: List[Dict[str, Any]],
    budget: float,
    market_type: str,
    history: Optional[List[Dict[str, Any]]] = None,
    sentiment: Optional[List[str]] = None,
    use_rsi: bool = False,
    use_sma: bool = False,
    use_sentiment: bool = False,
    market: Optional[str] = None,
    market_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Main entry point used by the CLI.

    Returns a list of recommendation dicts (possibly empty).
    """

    market_context = market_context or {}
    close_hhmm = _last_session_close_hhmm(market_context)
    sell_tz = market_context.get("timezone") or "UTC"

    # Compute a global sentiment score (optional)
    global_sent_score = _sentiment_score_from_headlines(sentiment) if use_sentiment else None

    # Prepare signals per-asset
    enriched: List[Dict[str, Any]] = []
    for row in market_data:
        price = _safe_get_price(row)
        if price <= 0:
            continue

        ph = row.get("price_history") or []
        sig = _compute_basic_indicators(ph)
        asset_regime = _infer_regime_from_indicators(sig) if not regime_mod else None
        # If the regime module is available, let it decide
        if regime_mod and hasattr(regime_mod, "detect_regime"):
            try:
                asset_regime = str(regime_mod.detect_regime(sig, ph))
            except Exception:
                asset_regime = _infer_regime_from_indicators(sig)

        score = _fuse_score(sig, asset_regime or "range", global_sent_score)

        enriched.append({
            "asset": row.get("asset") or row.get("symbol"),
            "symbol": row.get("symbol") or row.get("asset"),
            "price": price,
            "indicators": sig,
            "regime": asset_regime,
            "score": score,
        })

    # Optional LLM pass (guarded). We pass a compact snapshot.
    if _llm_enabled() and enriched:
        payload = {
            "market_type": market_type,
            "market": market,
            "budget": budget,
            "sell_close": close_hhmm,
            "sell_tz": sell_tz,
            "global_sentiment": global_sent_score,
            "assets": [
                {
                    "asset": e["asset"],
                    "symbol": e["symbol"],
                    "price": e["price"],
                    "regime": e["regime"],
                    "score": round(float(e["score"]), 4),
                    "indicators": e["indicators"],
                }
                for e in enriched[:40]  # cap to keep prompt small
            ],
        }
        llm_recs = _llm_analyze(payload)
        if isinstance(llm_recs, list):
            # Validate shape + coerce to our return schema; if invalid, we ignore & fall back
            cleaned = []
            for r in llm_recs:
                try:
                    asset = r.get("asset") or r.get("symbol")
                    price = float(r.get("price"))
                    qty = int(r.get("quantity"))
                    tgt = float(r.get("sell_target"))
                    conf = float(r.get("confidence", 0.0))
                    action = str(r.get("action") or "Hold")
                    estp = float(r.get("estimated_profit", 0.0))
                    cleaned.append({
                        "asset": asset,
                        "symbol": asset,
                        "price": price,
                        "quantity": qty,
                        "sell_target": tgt,
                        "sell_time": close_hhmm,
                        "sell_time_tz": sell_tz,
                        "estimated_profit": estp,
                        "action": action,
                        "confidence": conf,
                        "key_reasons": r.get("key_reasons"),
                    })
                except Exception:
                    continue
            if cleaned:
                return cleaned

    # Offline deterministic plan:
    # 1) Rank by score desc for BUY candidates
    # 2) Propose 1-3 positions (or 1 if budget is small)
    # 3) Risk-aware sizing using ATR proxy
    buys = [e for e in enriched if _choose_action(e["score"]) == "Buy"]
    buys.sort(key=lambda x: x["score"], reverse=True)

    if not buys:
        # If no Buys, surface top Hold (as a watchlist item) or empty
        holds = sorted(enriched, key=lambda x: abs(x["score"]), reverse=True)
        if holds:
            top = holds[0]
            # Return an empty position but with reasons to print
            return [{
                "asset": top["asset"],
                "symbol": top["symbol"],
                "price": top["price"],
                "quantity": 0,
                "sell_target": top["price"],
                "sell_time": close_hhmm,
                "sell_time_tz": sell_tz,
                "estimated_profit": 0.0,
                "action": "Hold",
                "confidence": _pct(abs(top["score"])),
                "key_reasons": f"Regime: {top['regime'] or 'range'}, score={round(top['score'],3)}; "
                               f"trend/momentum mixed; waiting for confirmation.",
            }]

        return []

    # Choose how many positions based on budget (simple heuristic)
    n_positions = 1 if budget < 500 else (2 if budget < 1500 else 3)
    picks = buys[:n_positions]

    recs: List[Dict[str, Any]] = []
    per_trade_budget = budget / max(1, len(picks))

    for e in picks:
        price = e["price"]
        atr_rel = e["indicators"].get("atr_proxy") or 0.02
        qty, stop_price, target_price = _position_size(
            price=price,
            budget=per_trade_budget,
            atr_proxy=atr_rel,
            risk_per_trade=0.02,  # 2% risk model
            min_qty=1,
        )
        if qty <= 0:
            continue

        est_profit = max(0.0, (target_price - price) * qty)
        recs.append({
            "asset": e["asset"],
            "symbol": e["symbol"],
            "price": price,
            "quantity": qty,
            "sell_target": round(float(target_price), 4),
            "sell_time": close_hhmm,
            "sell_time_tz": sell_tz,
            "estimated_profit": round(float(est_profit), 2),
            "action": "Buy",
            "confidence": _pct(min(1.0, max(0.0, e["score"]))),
            "key_reasons": (
                f"Regime: {e['regime'] or 'range'}; "
                f"EMA>SMA: {bool((e['indicators'].get('ema_20') or 0) > (e['indicators'].get('sma_20') or 0))}; "
                f"MACD>Signal: {bool((e['indicators'].get('macd') or -1e9) > (e['indicators'].get('macd_signal') or 1e9))}; "
                f"RSI: {round(e['indicators'].get('rsi_14') or 50.0, 1)}; "
                f"Volatility: {round((e['indicators'].get('volatility') or 0.0)*100,1)}%"
            ),
        })

    return recs
