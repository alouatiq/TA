# BE/trading_core/persistence/history_tracker.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Storage location:
#   1) PERSIST_DIR env var (absolute or relative)
#   2) default to "<this_dir>/data"
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", Path(__file__).resolve().parent / "data")).resolve()
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS_FILE = PERSIST_DIR / "trades.jsonl"   # append-only JSONL


@dataclass
class SessionRecord:
    ts_utc: str                     # ISO 8601 UTC timestamp
    market_type: str                # 'crypto','equities', etc
    budget: float
    recommendations: List[Dict[str, Any]]
    features: Dict[str, Any]        # toggles, options
    meta: Dict[str, Any]            # optional: market, region, tz, etc

    @staticmethod
    def now_utc_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def from_payload(cls,
                     *,
                     market_type: str,
                     budget: float,
                     recommendations: List[Dict[str, Any]],
                     features: Dict[str, Any],
                     meta: Optional[Dict[str, Any]] = None) -> "SessionRecord":
        return cls(
            ts_utc=cls.now_utc_iso(),
            market_type=market_type,
            budget=float(budget),
            recommendations=recommendations or [],
            features=features or {},
            meta=meta or {},
        )


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def log_trade(*,
              market_type: str,
              budget: float,
              recommendations: List[Dict[str, Any]],
              features: Dict[str, Any],
              market: Optional[str] = None,
              market_context: Optional[Dict[str, Any]] = None) -> None:
    """
    Append a completed assistant run to the session log (JSONL).
    `market` and `market_context` are accepted to keep compatibility with callers.
    """
    meta = market_context or {}
    if market and "market" not in meta:
        meta["market"] = market
    rec = SessionRecord.from_payload(
        market_type=market_type,
        budget=budget,
        recommendations=recommendations,
        features=features,
        meta=meta,
    )
    _append_jsonl(SESSIONS_FILE, asdict(rec))


def _parse_iso(dt: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(dt.replace("Z", "+00:00"))
    except Exception:
        return None


def load_history(days: int = 7) -> List[Dict[str, Any]]:
    """
    Return sessions from the last `days` days (inclusive), newest → oldest.
    Each item is the raw dict stored in JSONL.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, days))
    items: List[Dict[str, Any]] = []
    for row in _iter_jsonl(SESSIONS_FILE):
        ts = _parse_iso(row.get("ts_utc", "")) or cutoff
        if ts >= cutoff:
            items.append(row)
    items.sort(key=lambda r: r.get("ts_utc", ""), reverse=True)
    return items


def last_session() -> Optional[Dict[str, Any]]:
    """
    Return the most recent session (or None).
    """
    for row in load_history(days=3650):  # effectively full scan, newest→oldest
        return row
    return None
