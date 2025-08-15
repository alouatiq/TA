# BE/trading_core/utils/__init__.py
"""
Small cross‑cutting helpers shared across the backend.
This module re-exports the most commonly used utilities so callers can do:

    from trading_core.utils import get_logger, read_yaml, market_is_open_now

Nothing here should import heavy libs or create circular deps with
domain modules (data_fetcher, indicators, scoring...). Keep it lean.
"""

from .logging import get_logger
from .io import read_yaml, write_json, read_json, ensure_dir, atomic_write_text
from .timezones import (
    parse_hhmm,
    sessions_today,
    market_is_open_now,
    convert_hhmm_between_tz,
    localize_hhmm_in_market_tz,
)

__all__ = [
    # logging
    "get_logger",
    # io
    "read_yaml",
    "write_json",
    "read_json",
    "ensure_dir",
    "atomic_write_text",
    # time/tz
    "parse_hhmm",
    "sessions_today",
    "market_is_open_now",
    "convert_hhmm_between_tz",
    "localize_hhmm_in_market_tz",
]
