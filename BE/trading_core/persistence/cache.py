# BE/trading_core/persistence/cache.py
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Disk cache directory (optional)
CACHE_DIR = Path(os.getenv("CACHE_DIR", Path(__file__).resolve().parent / "cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SimpleCache:
    """
    Tiny TTL cache with optional disk persistence per key.

    Usage:
        cache = SimpleCache(default_ttl=300)
        val = cache.get_or_set("coingecko:top", lambda: fetch(), ttl=120)

        cache.put("quote:AAPL", {"p": 190.2}, ttl=60)
        cache.get("quote:AAPL")

    Notes
    -----
    - Keys are namespaced strings. Be consistent (e.g., "provider:endpoint:paramhash").
    - Disk writes are best-effort; failures won’t raise.
    """

    def __init__(self, default_ttl: int = 300, use_disk: bool = True) -> None:
        self._mem: Dict[str, Tuple[float, Any]] = {}
        self.default_ttl = max(1, int(default_ttl))
        self.use_disk = bool(use_disk)

    # --------------- core ---------------

    def _expired(self, exp_ts: float) -> bool:
        return time.time() >= exp_ts

    def _mk_path(self, key: str) -> Path:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return CACHE_DIR / f"{h}.json"

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        # memory first
        if key in self._mem:
            exp_ts, val = self._mem[key]
            if not self._expired(exp_ts):
                return val
            # drop expired
            self._mem.pop(key, None)

        # disk fallback
        if self.use_disk:
            p = self._mk_path(key)
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        blob = json.load(f)
                    exp_ts = float(blob.get("exp", 0))
                    if now < exp_ts:
                        val = blob.get("val")
                        # refresh mem
                        self._mem[key] = (exp_ts, val)
                        return val
                    else:
                        p.unlink(missing_ok=True)
                except Exception:
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass
        return None

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl_eff = self.default_ttl if ttl is None else max(1, int(ttl))
        exp_ts = time.time() + ttl_eff
        self._mem[key] = (exp_ts, value)
        if self.use_disk:
            p = self._mk_path(key)
            try:
                with p.open("w", encoding="utf-8") as f:
                    json.dump({"exp": exp_ts, "val": value}, f)
            except Exception:
                pass

    def get_or_set(self, key: str, factory, ttl: Optional[int] = None) -> Any:
        found = self.get(key)
        if found is not None:
            return found
        val = factory()
        self.put(key, val, ttl=ttl)
        return val

    # --------------- housekeeping ---------------

    def purge_expired(self) -> int:
        """
        Remove expired items from memory and disk. Returns count purged.
        """
        now = time.time()
        purged = 0
        # mem
        for k in list(self._mem.keys()):
            exp_ts, _ = self._mem[k]
            if self._expired(exp_ts):
                self._mem.pop(k, None)
                purged += 1
        # disk
        if self.use_disk:
            for p in CACHE_DIR.glob("*.json"):
                try:
                    with p.open("r", encoding="utf-8") as f:
                        blob = json.load(f)
                    if now >= float(blob.get("exp", 0)):
                        p.unlink(missing_ok=True)
                        purged += 1
                except Exception:
                    try:
                        p.unlink(missing_ok=True)
                        purged += 1
                    except Exception:
                        pass
        return purged
