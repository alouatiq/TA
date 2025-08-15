"""
Sentiment fetchers (raw → normalized headlines).

Exports:
- fetch_equities_sentiment(market: Optional[str], limit: int) -> (headlines, diagnostics)
- fetch_crypto_sentiment(limit: int) -> (headlines, diagnostics)
- fetch_stock_sentiment(...)  # backward-compat alias
"""

from .equities import fetch_equities_sentiment as fetch_equities_sentiment
from .equities import fetch_equities_sentiment as fetch_stock_sentiment  # legacy alias

from .crypto import fetch_crypto_sentiment as fetch_crypto_sentiment

__all__ = [
    "fetch_equities_sentiment",
    "fetch_stock_sentiment",
    "fetch_crypto_sentiment",
]
