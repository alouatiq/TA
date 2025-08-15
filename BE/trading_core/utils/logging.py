# BE/trading_core/utils/logging.py
"""
Unified logger factory for the backend.
- Colorful, short console output for CLI
- Optional file handler (set TA_LOG_FILE or pass file_path)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


class _ConsoleFormatter(logging.Formatter):
    # simple, compact format
    default_fmt = "[%(levelname).1s] %(message)s"
    debug_fmt = "[%(levelname).1s] %(name)s: %(message)s"

    def __init__(self, verbose: bool = False):
        fmt = self.debug_fmt if verbose else self.default_fmt
        super().__init__(fmt)

    # add colors if TTY
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - cosmetics
        msg = super().format(record)
        if not os.isatty(1):
            return msg
        level = record.levelno
        if level >= logging.ERROR:
            return f"\033[91m{msg}\033[0m"
        if level >= logging.WARNING:
            return f"\033[93m{msg}\033[0m"
        if level >= logging.INFO:
            return f"\033[92m{msg}\033[0m"
        return f"\033[90m{msg}\033[0m"


def get_logger(name: str = "ta", *, level: int = logging.INFO, file_path: Optional[str | Path] = None) -> logging.Logger:
    """
    Create/reuse a namespaced logger with console + optional file output.
    Idempotent: calling twice returns the same configured logger.
    """
    logger = logging.getLogger(name)
    if getattr(logger, "_ta_configured", False):
        return logger

    logger.setLevel(level)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    verbose = os.getenv("TA_LOG_VERBOSE", "").lower() in {"1", "true", "yes"}
    ch.setFormatter(_ConsoleFormatter(verbose=verbose))
    logger.addHandler(ch)

    # File (opt-in)
    file_env = os.getenv("TA_LOG_FILE")
    path = file_path or file_env
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(fh)

    logger._ta_configured = True  # type: ignore[attr-defined]
    return logger
