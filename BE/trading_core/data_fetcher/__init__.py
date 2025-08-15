# Add this to the END of /home/al/TA/BE/trading_core/data_fetcher/__init__.py

def diagnostics_for(category: str) -> dict:
    """
    Get diagnostic information for a specific data fetcher category.
    
    Args:
        category: One of 'crypto', 'forex', 'equities', 'commodities', 'futures', 'warrants', 'funds'
    
    Returns:
        Dict with keys: 'used', 'failed', 'skipped'
    """
    try:
        if category == "crypto":
            from .crypto import LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
            return {
                "used": LAST_CRYPTO_SOURCE,
                "failed": FAILED_CRYPTO_SOURCES,
                "skipped": SKIPPED_CRYPTO_SOURCES
            }
        elif category == "forex":
            from .forex import LAST_FOREX_SOURCE, FAILED_FOREX_SOURCES, SKIPPED_FOREX_SOURCES
            return {
                "used": LAST_FOREX_SOURCE,
                "failed": FAILED_FOREX_SOURCES,
                "skipped": SKIPPED_FOREX_SOURCES
            }
        elif category == "equities":
            from .equities import LAST_EQUITIES_SOURCE, FAILED_EQUITIES_SOURCES, SKIPPED_EQUITIES_SOURCES
            return {
                "used": LAST_EQUITIES_SOURCE,
                "failed": FAILED_EQUITIES_SOURCES,
                "skipped": SKIPPED_EQUITIES_SOURCES
            }
        elif category == "commodities":
            from .commodities import LAST_COMMODITIES_SOURCE, FAILED_COMMODITIES_SOURCES, SKIPPED_COMMODITIES_SOURCES
            return {
                "used": LAST_COMMODITIES_SOURCE,
                "failed": FAILED_COMMODITIES_SOURCES,
                "skipped": SKIPPED_COMMODITIES_SOURCES
            }
        elif category == "futures":
            from .futures import LAST_FUTURES_SOURCE, FAILED_FUTURES_SOURCES, SKIPPED_FUTURES_SOURCES
            return {
                "used": LAST_FUTURES_SOURCE,
                "failed": FAILED_FUTURES_SOURCES,
                "skipped": SKIPPED_FUTURES_SOURCES
            }
        elif category == "warrants":
            from .warrants import LAST_WARRANTS_SOURCE, FAILED_WARRANTS_SOURCES, SKIPPED_WARRANTS_SOURCES
            return {
                "used": LAST_WARRANTS_SOURCE,
                "failed": FAILED_WARRANTS_SOURCES,
                "skipped": SKIPPED_WARRANTS_SOURCES
            }
        elif category == "funds":
            from .funds import LAST_FUNDS_SOURCE, FAILED_FUNDS_SOURCES, SKIPPED_FUNDS_SOURCES
            return {
                "used": LAST_FUNDS_SOURCE,
                "failed": FAILED_FUNDS_SOURCES,
                "skipped": SKIPPED_FUNDS_SOURCES
            }
        else:
            return {"used": "Unknown", "failed": [], "skipped": []}
    except ImportError:
        return {"used": "Not available", "failed": [], "skipped": []}


# Add diagnostics_for to the __all__ list
__all__ = [
    "fetch_equities_data",
    "fetch_crypto_data",
    "fetch_forex_data",
    "fetch_commodities_data",
    "fetch_futures_data",
    "fetch_warrants_data",
    "fetch_funds_data",
    "diagnostics_for",  # Add this line
]
