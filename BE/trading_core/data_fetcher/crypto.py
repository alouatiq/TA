"""
trading_core.data_fetcher.crypto
────────────────────────────────
Cryptocurrency data fetching with fallback chain.
All data is fetched from real-time internet sources.

Primary source: CoinGecko API
Fallback sources: CryptoCompare, Binance, etc.
"""

from __future__ import annotations
import os
import time
import random
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

# Import the adapters for real-time data fetching
from .adapters.coingecko import CoinGeckoAdapter
from .adapters.cryptocompare import CryptoCompareAdapter
from .adapters.binance import BinanceAdapter
from .adapters.coinbase import CoinbaseAdapter
from .adapters.kraken import KrakenAdapter
from .adapters.kucoin import KuCoinAdapter
from .adapters.okx import OKXAdapter
from .adapters.bybit import BybitAdapter
from .adapters.gateio import GateIOAdapter
from .adapters.huobi import HuobiAdapter
from .adapters.bitstamp import BitstampAdapter
from .adapters.gemini import GeminiAdapter
from .adapters.coinmarketcap import CoinMarketCapAdapter
from .adapters.messari import MessariAdapter
from .adapters.nomics import NomicsAdapter
from .adapters.coinpaprika import CoinPaprikaAdapter
from .adapters.coinlore import CoinLoreAdapter
from .adapters.cryptorank import CryptoRankAdapter
from .adapters.coincodex import CoinCodexAdapter
from .adapters.livecoinwatch import LiveCoinWatchAdapter

# Set up logging
logger = logging.getLogger(__name__)

# Module-level diagnostics for tracking which sources are used
LAST_CRYPTO_SOURCE = "None"
FAILED_CRYPTO_SOURCES = []
SKIPPED_CRYPTO_SOURCES = []

# Cache for storing fetched data to reduce API calls
_crypto_cache = {
    "data": None,
    "timestamp": None,
    "cache_duration": 60  # Cache for 60 seconds to avoid rate limits
}

def fetch_crypto_data(
    include_history: bool = False,
    *,
    market: str | None = None,
    region: str | None = None,
    min_assets: int | None = None,
    force_seeds: bool = False,  # Deprecated parameter, will be ignored
    symbols: list[str] | None = None,
    max_universe: int | None = None
) -> list[dict]:
    """
    Fetch cryptocurrency data from real-time internet sources with intelligent fallback chain.
    
    Args:
        include_history: Whether to include price history from real sources
        market: Market filter (e.g., "spot", "futures")
        region: Region filter (not commonly used for crypto)
        min_assets: Minimum number of assets to return
        force_seeds: DEPRECATED - ignored, always fetches real data
        symbols: Specific symbols to fetch from exchanges
        max_universe: Maximum number of assets to return
        
    Returns:
        List of real-time cryptocurrency data dictionaries from internet sources
    """
    global LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
    
    # Reset diagnostics for this fetch
    FAILED_CRYPTO_SOURCES = []
    SKIPPED_CRYPTO_SOURCES = []
    
    # Check cache first to avoid excessive API calls
    if _crypto_cache["data"] is not None and _crypto_cache["timestamp"] is not None:
        elapsed = time.time() - _crypto_cache["timestamp"]
        if elapsed < _crypto_cache["cache_duration"]:
            logger.debug(f"Using cached crypto data (age: {elapsed:.1f}s)")
            LAST_CRYPTO_SOURCE = "Cache (Real Data)"
            return _filter_crypto_results(
                _crypto_cache["data"],
                symbols=symbols,
                max_universe=max_universe,
                min_assets=min_assets
            )
    
    # Define the fallback chain with all real data adapters
    # Ordered by reliability and data quality
    fallback_chain = [
        # Tier 1: Most reliable free APIs
        ("CoinGecko", CoinGeckoAdapter),
        ("CryptoCompare", CryptoCompareAdapter),
        ("CoinPaprika", CoinPaprikaAdapter),
        
        # Tier 2: Major exchanges with public APIs
        ("Binance", BinanceAdapter),
        ("Coinbase", CoinbaseAdapter),
        ("Kraken", KrakenAdapter),
        ("KuCoin", KuCoinAdapter),
        
        # Tier 3: Other exchanges
        ("OKX", OKXAdapter),
        ("Bybit", BybitAdapter),
        ("Gate.io", GateIOAdapter),
        ("Huobi", HuobiAdapter),
        ("Bitstamp", BitstampAdapter),
        ("Gemini", GeminiAdapter),
        
        # Tier 4: Data aggregators (may require API keys)
        ("CoinMarketCap", CoinMarketCapAdapter),
        ("Messari", MessariAdapter),
        ("CoinLore", CoinLoreAdapter),
        ("CryptoRank", CryptoRankAdapter),
        ("CoinCodex", CoinCodexAdapter),
        ("LiveCoinWatch", LiveCoinWatchAdapter),
        ("Nomics", NomicsAdapter),
    ]
    
    # Try each source in the fallback chain
    for source_name, adapter_class in fallback_chain:
        try:
            logger.debug(f"Attempting to fetch real-time crypto data from {source_name}")
            
            # Check if API is configured (some may need API keys)
            if not adapter_class.is_configured():
                logger.debug(f"{source_name} is not configured, skipping")
                SKIPPED_CRYPTO_SOURCES.append(source_name)
                continue
            
            # Initialize adapter for real-time data
            adapter = adapter_class()
            
            # Fetch data based on request type
            if symbols:
                # Fetch specific symbols from the exchange/API
                logger.debug(f"Fetching specific symbols: {symbols}")
                data = adapter.fetch_symbols(symbols)
            else:
                # Fetch top cryptocurrencies by market cap
                limit = max_universe or min_assets or 100
                logger.debug(f"Fetching top {limit} cryptocurrencies")
                data = adapter.fetch_top_cryptos(limit=limit)
            
            # Validate we got real data
            if not data or not isinstance(data, list):
                logger.warning(f"{source_name} returned no data or invalid format")
                FAILED_CRYPTO_SOURCES.append(source_name)
                continue
            
            # Add real-time history if requested and available
            if include_history and data:
                data = _add_real_crypto_history(data, adapter, symbols)
            
            # Validate we have enough data
            if min_assets and len(data) < min_assets:
                logger.warning(f"{source_name} returned insufficient data: {len(data)} < {min_assets}")
                FAILED_CRYPTO_SOURCES.append(source_name)
                continue
            
            # Success! We have real data
            logger.info(f"Successfully fetched {len(data)} real crypto assets from {source_name}")
            LAST_CRYPTO_SOURCE = source_name
            
            # Update cache with real data
            _crypto_cache["data"] = data
            _crypto_cache["timestamp"] = time.time()
            
            return _filter_crypto_results(
                data,
                symbols=symbols,
                max_universe=max_universe,
                min_assets=min_assets
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch from {source_name}: {e}")
            FAILED_CRYPTO_SOURCES.append(source_name)
            continue
    
    # All sources failed - return empty list or minimal data
    logger.error("All crypto data sources failed to fetch real-time data")
    LAST_CRYPTO_SOURCE = "Failed - No Data Available"
    
    # Return empty list since we couldn't fetch any real data
    return []


def _filter_crypto_results(
    data: list[dict],
    symbols: list[str] | None = None,
    max_universe: int | None = None,
    min_assets: int | None = None
) -> list[dict]:
    """
    Filter and limit crypto results based on parameters.
    
    Args:
        data: Raw crypto data from real sources
        symbols: Specific symbols to include
        max_universe: Maximum number of results
        min_assets: Minimum number of results required
        
    Returns:
        Filtered list of real crypto data
    """
    if not data:
        return []
    
    # Filter by symbols if specified
    if symbols:
        symbols_upper = [s.upper() for s in symbols]
        filtered = []
        for item in data:
            asset_symbol = item.get("asset", "").upper()
            symbol = item.get("symbol", "").upper()
            # Check both asset and symbol fields for matches
            if asset_symbol in symbols_upper or symbol in symbols_upper:
                filtered.append(item)
        data = filtered
    
    # Sort by volume or market cap if available (highest first)
    if data and "volume" in data[0]:
        data = sorted(data, key=lambda x: x.get("volume", 0), reverse=True)
    
    # Limit to max_universe if specified
    if max_universe and len(data) > max_universe:
        data = data[:max_universe]
    
    # Log warning if we don't meet min_assets requirement
    if min_assets and len(data) < min_assets:
        logger.warning(f"Results ({len(data)}) below min_assets ({min_assets})")
    
    return data


def _add_real_crypto_history(
    data: list[dict],
    adapter: Any,
    symbols: list[str] | None = None
) -> list[dict]:
    """
    Add real historical price data to crypto assets from the same source.
    
    Args:
        data: List of crypto assets
        adapter: The adapter instance to fetch history with
        symbols: Specific symbols that were requested
        
    Returns:
        Data with real price_history added where available
    """
    # Limit history fetching to avoid rate limits
    max_history_fetch = 20 if not symbols else len(symbols)
    
    for i, asset in enumerate(data[:max_history_fetch]):
        try:
            symbol = asset.get("symbol", asset.get("asset", ""))
            
            # Check if adapter supports history fetching
            if hasattr(adapter, 'fetch_history'):
                # Fetch real historical data from the source
                history = adapter.fetch_history(symbol, days=7)
                if history and isinstance(history, list) and len(history) > 0:
                    asset["price_history"] = history
                    logger.debug(f"Added {len(history)} real history points for {symbol}")
            else:
                logger.debug(f"Adapter {adapter.__class__.__name__} doesn't support history fetching")
                
        except Exception as e:
            logger.debug(f"Could not fetch history for {asset.get('asset', 'unknown')}: {e}")
    
    return data


def fetch_crypto_historical_data(
    symbol: str,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    interval: str = "1d"
) -> dict:
    """
    Fetch real historical data for a specific cryptocurrency from internet sources.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC", "ETH")
        start_date: Start date for historical data
        end_date: End date for historical data
        interval: Data interval (1d, 1h, 5m, etc.)
        
    Returns:
        Dictionary with real historical data including OHLCV
    """
    # Default date range if not specified
    if not end_date:
        end_date = datetime.now()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Try to fetch from available sources that support OHLCV data
    adapters_with_history = [
        ("Binance", BinanceAdapter),
        ("CoinGecko", CoinGeckoAdapter),
        ("Kraken", KrakenAdapter),
        ("Coinbase", CoinbaseAdapter),
        ("KuCoin", KuCoinAdapter),
        ("CryptoCompare", CryptoCompareAdapter),
    ]
    
    for source_name, adapter_class in adapters_with_history:
        try:
            # Check if source is configured
            if not adapter_class.is_configured():
                logger.debug(f"{source_name} not configured for historical data")
                continue
                
            adapter = adapter_class()
            
            # Check if adapter supports OHLCV data fetching
            if hasattr(adapter, 'fetch_ohlcv'):
                logger.debug(f"Fetching OHLCV data from {source_name} for {symbol}")
                
                # Fetch real OHLCV data
                data = adapter.fetch_ohlcv(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )
                
                # Validate we got real data
                if data and isinstance(data, list) and len(data) > 0:
                    return {
                        "symbol": symbol,
                        "source": source_name,
                        "interval": interval,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "data": data,
                        "data_points": len(data)
                    }
                    
        except Exception as e:
            logger.debug(f"Could not fetch history from {source_name}: {e}")
            continue
    
    # If all sources fail, return empty result
    logger.warning(f"Could not fetch historical data for {symbol} from any source")
    return {
        "symbol": symbol,
        "source": "None",
        "interval": interval,
        "error": "No data available from any source",
        "data": []
    }


def get_available_crypto_sources() -> dict:
    """
    Get information about available cryptocurrency data sources and their status.
    
    Returns:
        Dictionary with source information and configuration status
    """
    sources_info = {}
    
    all_sources = [
        ("CoinGecko", CoinGeckoAdapter),
        ("CryptoCompare", CryptoCompareAdapter),
        ("Binance", BinanceAdapter),
        ("Coinbase", CoinbaseAdapter),
        ("Kraken", KrakenAdapter),
        ("KuCoin", KuCoinAdapter),
        ("OKX", OKXAdapter),
        ("Bybit", BybitAdapter),
        ("Gate.io", GateIOAdapter),
        ("Huobi", HuobiAdapter),
        ("Bitstamp", BitstampAdapter),
        ("Gemini", GeminiAdapter),
        ("CoinMarketCap", CoinMarketCapAdapter),
        ("Messari", MessariAdapter),
        ("Nomics", NomicsAdapter),
        ("CoinPaprika", CoinPaprikaAdapter),
        ("CoinLore", CoinLoreAdapter),
        ("CryptoRank", CryptoRankAdapter),
        ("CoinCodex", CoinCodexAdapter),
        ("LiveCoinWatch", LiveCoinWatchAdapter),
    ]
    
    for source_name, adapter_class in all_sources:
        try:
            sources_info[source_name] = {
                "configured": adapter_class.is_configured(),
                "requires_api_key": getattr(adapter_class, 'requires_api_key', False),
                "supports_history": hasattr(adapter_class, 'fetch_history') or hasattr(adapter_class, 'fetch_ohlcv'),
                "tier": _get_source_tier(source_name)
            }
        except Exception as e:
            sources_info[source_name] = {
                "configured": False,
                "error": str(e)
            }
    
    return sources_info


def _get_source_tier(source_name: str) -> int:
    """
    Get the tier/priority of a data source.
    
    Args:
        source_name: Name of the data source
        
    Returns:
        Tier number (1 = highest priority, 4 = lowest)
    """
    tier_map = {
        # Tier 1: Most reliable free APIs
        "CoinGecko": 1,
        "CryptoCompare": 1,
        "CoinPaprika": 1,
        
        # Tier 2: Major exchanges
        "Binance": 2,
        "Coinbase": 2,
        "Kraken": 2,
        "KuCoin": 2,
        
        # Tier 3: Other exchanges
        "OKX": 3,
        "Bybit": 3,
        "Gate.io": 3,
        "Huobi": 3,
        "Bitstamp": 3,
        "Gemini": 3,
        
        # Tier 4: Data aggregators
        "CoinMarketCap": 4,
        "Messari": 4,
        "CoinLore": 4,
        "CryptoRank": 4,
        "CoinCodex": 4,
        "LiveCoinWatch": 4,
        "Nomics": 4,
    }
    
    return tier_map.get(source_name, 5)


def test_crypto_sources() -> dict:
    """
    Test all configured crypto data sources and return their status.
    
    Returns:
        Dictionary with test results for each source
    """
    test_results = {}
    test_symbol = "BTC"  # Use Bitcoin as test symbol
    
    all_sources = [
        ("CoinGecko", CoinGeckoAdapter),
        ("CryptoCompare", CryptoCompareAdapter),
        ("Binance", BinanceAdapter),
        ("Coinbase", CoinbaseAdapter),
        ("Kraken", KrakenAdapter),
    ]
    
    for source_name, adapter_class in all_sources:
        try:
            if not adapter_class.is_configured():
                test_results[source_name] = {
                    "status": "not_configured",
                    "message": "API key or configuration missing"
                }
                continue
            
            # Try to fetch a single symbol
            adapter = adapter_class()
            start_time = time.time()
            
            if hasattr(adapter, 'fetch_symbols'):
                data = adapter.fetch_symbols([test_symbol])
            else:
                data = adapter.fetch_top_cryptos(limit=1)
            
            elapsed = time.time() - start_time
            
            if data and len(data) > 0:
                test_results[source_name] = {
                    "status": "success",
                    "response_time": f"{elapsed:.2f}s",
                    "sample_data": data[0] if data else None
                }
            else:
                test_results[source_name] = {
                    "status": "no_data",
                    "message": "Source returned no data"
                }
                
        except Exception as e:
            test_results[source_name] = {
                "status": "error",
                "message": str(e)
            }
    
    return test_results
