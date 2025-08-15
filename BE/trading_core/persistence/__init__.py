# BE/trading_core/persistence/__init__.py
"""
Lightweight persistence utilities:
- history_tracker: append-only JSONL of sessions & recommendations
- performance_evaluator: simple “yesterday” score-card
- cache: tiny in-memory/disk TTL cache for quotes/headlines

These modules are intentionally dependency-light so they work in CLI
and can be reused by a future web API.
"""

from .history_tracker import log_trade, load_history, last_session  # noqa: F401
from .performance_evaluator import evaluate_previous_session  # noqa: F401
from .cache import SimpleCache  # noqa: F401

__all__ = [
    "log_trade",
    "load_history",
    "last_session",
    "evaluate_previous_session",
    "SimpleCache",
]
