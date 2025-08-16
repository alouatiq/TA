"""
trading_core.data_fetcher.crypto
────────────────────────────────
Cryptocurrency data fetching with fallback chain.
All data is fetched from real-time internet sources.

Primary source: CoinGecko API
Fallback sources: Other available adapters
"""

from __future__ import annotations
import os
import time
import random
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

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

# Import available adapters with error handling
# We'll dynamically check which adapters are available
AVAILABLE_ADAPTERS = {}

# Try to import each adapter and track which ones are available
adapter_imports = [
    ("CoinGecko", "coingecko", "CoinGeckoAdapter"),
    ("CryptoCompare", "cryptocompare", "CryptoCompareAdapter"),
    ("Binance", "binance", "BinanceAdapter"),
    ("Coinbase", "coinbase", "CoinbaseAdapter"),
    ("Kraken", "kraken", "KrakenAdapter"),
    ("KuCoin", "kucoin", "KuCoinAdapter"),
    ("OKX", "okx", "OKXAdapter"),
    ("Bybit", "bybit", "BybitAdapter"),
    ("GateIO", "gateio", "GateIOAdapter"),
    ("Huobi", "huobi", "HuobiAdapter"),
    ("Bitstamp", "bitstamp", "BitstampAdapter"),
    ("Gemini", "gemini", "GeminiAdapter"),
    ("CoinMarketCap", "coinmarketcap", "CoinMarketCapAdapter"),
    ("Messari", "messari", "MessariAdapter"),
    ("Nomics", "nomics", "NomicsAdapter"),
    ("CoinPaprika", "coinpaprika", "CoinPaprikaAdapter"),
    ("CoinLore", "coinlore", "CoinLoreAdapter"),
    ("CryptoRank", "cryptorank", "CryptoRankAdapter"),
    ("CoinCodex", "coincodex", "CoinCodexAdapter"),
    ("LiveCoinWatch", "livecoinwatch", "LiveCoinWatchAdapter"),
]

for display_name, module_name, class_name in adapter_imports:
    try:
        # Dynamically import the adapter
        module = __import__(f"trading_core.data_fetcher.adapters.{module_name}", fromlist=[class_name])
        adapter_class = getattr(module, class_name, None)
        
        if adapter_class:
            AVAILABLE_ADAPTERS[display_name] = adapter_class
            logger.debug(f"Successfully imported {display_name} adapter")
        else:
            logger.debug(f"Class {class_name} not found in {module_name}")
            
    except ImportError as e:
        logger.debug(f"Could not import {display_name} adapter: {e}")
    except Exception as e:
        logger.debug(f"Error loading {display_name} adapter: {e}")

# Log available adapters
if AVAILABLE_ADAPTERS:
    logger.info(f"Available crypto adapters: {list(AVAILABLE_ADAPTERS.keys())}")
else:
    logger.warning("No crypto adapters could be loaded!")


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
    
    # Check if we have any adapters available
    if not AVAILABLE_ADAPTERS:
        logger.error("No crypto adapters are available")
        LAST_CRYPTO_SOURCE = "No Adapters Available"
        return _fetch_fallback_data(symbols, max_universe)
    
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
    
    # Build fallback chain from available adapters
    # Prioritize by reliability
    priority_order = [
        # Tier 1: Most reliable free APIs
        "CoinGecko", "CoinPaprika", 
        # Tier 2: Major exchanges
        "Binance", "Coinbase", "Kraken", "KuCoin",
        # Tier 3: Other exchanges
        "OKX", "Bybit", "GateIO", "Huobi", "Bitstamp", "Gemini",
        # Tier 4: Data aggregators
        "CoinMarketCap", "CryptoCompare", "Messari", "CoinLore", 
        "CryptoRank", "CoinCodex", "LiveCoinWatch", "Nomics",
    ]
    
    # Build fallback chain from available adapters in priority order
    fallback_chain = []
    for source_name in priority_order:
        if source_name in AVAILABLE_ADAPTERS:
            fallback_chain.append((source_name, AVAILABLE_ADAPTERS[source_name]))
    
    # Add any remaining adapters not in priority order
    for source_name, adapter_class in AVAILABLE_ADAPTERS.items():
        if source_name not in priority_order:
            fallback_chain.append((source_name, adapter_class))
    
    logger.debug(f"Fallback chain: {[name for name, _ in fallback_chain]}")
    
    # Try each source in the fallback chain
    for source_name, adapter_class in fallback_chain:
        try:
            logger.debug(f"Attempting to fetch real-time crypto data from {source_name}")
            
            # Check if API is configured (some may need API keys)
            if hasattr(adapter_class, 'is_configured'):
                if not adapter_class.is_configured():
                    logger.debug(f"{source_name} is not configured, skipping")
                    SKIPPED_CRYPTO_SOURCES.append(source_name)
                    continue
            
            # Initialize adapter for real-time data
            adapter = adapter_class()
            
            # Fetch data based on request type
            data = None
            if symbols:
                # Try to fetch specific symbols
                logger.debug(f"Fetching specific symbols: {symbols}")
                if hasattr(adapter, 'fetch_symbols'):
                    data = adapter.fetch_symbols(symbols)
                elif hasattr(adapter, 'get_quotes'):
                    data = adapter.get_quotes(symbols)
                elif hasattr(adapter, 'fetch'):
                    # Generic fetch method
                    data = adapter.fetch(symbols=symbols)
            else:
                # Fetch top cryptocurrencies by market cap
                limit = max_universe or min_assets or 100
                logger.debug(f"Fetching top {limit} cryptocurrencies")
                
                if hasattr(adapter, 'fetch_top_cryptos'):
                    data = adapter.fetch_top_cryptos(limit=limit)
                elif hasattr(adapter, 'get_top_coins'):
                    data = adapter.get_top_coins(limit=limit)
                elif hasattr(adapter, 'fetch'):
                    # Generic fetch method
                    data = adapter.fetch(limit=limit)
            
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
    
    # All sources failed - try basic fallback
    logger.error("All crypto data sources failed to fetch real-time data")
    LAST_CRYPTO_SOURCE = "Failed - Attempting Basic Fetch"
    
    return _fetch_fallback_data(symbols, max_universe)


def _fetch_fallback_data(symbols: list[str] | None = None, max_universe: int | None = None) -> list[dict]:
    """
    Attempt to fetch data using basic HTTP requests as a last resort.
    
    Args:
        symbols: Specific symbols to fetch
        max_universe: Maximum number of assets
        
    Returns:
        List of crypto data or empty list if all attempts fail
    """
    try:
        # Try to use requests directly for public APIs that don't need keys
        import requests
        
        # Try CoinGecko public API (no key required)
        try:
            if symbols:
                # For specific symbols, we need to map them to CoinGecko IDs
                symbol_map = {
                    "BTC": "bitcoin",
                    "ETH": "ethereum", 
                    "BNB": "binancecoin",
                    "SOL": "solana",
                    "XRP": "ripple",
                    "USDC": "usd-coin",
                    "ADA": "cardano",
                    "AVAX": "avalanche-2",
                    "DOGE": "dogecoin",
                    "DOT": "polkadot",
                }
                
                ids = []
                for symbol in symbols[:10]:  # Limit to 10 symbols
                    symbol_upper = symbol.upper().replace("USD", "").replace("USDT", "")
                    if symbol_upper in symbol_map:
                        ids.append(symbol_map[symbol_upper])
                
                if ids:
                    url = f"https://api.coingecko.com/api/v3/simple/price"
                    params = {
                        "ids": ",".join(ids),
                        "vs_currencies": "usd",
                        "include_24hr_vol": "true",
                        "include_24hr_change": "true"
                    }
                    
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        result = []
                        
                        for coin_id, coin_data in data.items():
                            # Map back to symbol
                            symbol = None
                            for sym, cid in symbol_map.items():
                                if cid == coin_id:
                                    symbol = sym
                                    break
                            
                            if symbol and "usd" in coin_data:
                                result.append({
                                    "asset": symbol,
                                    "symbol": f"{symbol}USD",
                                    "price": coin_data.get("usd", 0),
                                    "volume": coin_data.get("usd_24h_vol", 0),
                                    "day_range_pct": coin_data.get("usd_24h_change", 0),
                                    "currency": "USD",
                                    "exchange": "CoinGecko"
                                })
                        
                        if result:
                            logger.info(f"Fallback fetch successful: {len(result)} assets from CoinGecko")
                            return result
            else:
                # Fetch top coins by market cap
                limit = max_universe or 50
                url = f"https://api.coingecko.com/api/v3/coins/markets"
                params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": min(limit, 250),
                    "page": 1,
                    "sparkline": False
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    result = []
                    
                    for coin in data:
                        result.append({
                            "asset": coin.get("symbol", "").upper(),
                            "symbol": f"{coin.get('symbol', '').upper()}USD",
                            "price": coin.get("current_price", 0),
                            "volume": coin.get("total_volume", 0),
                            "day_range_pct": coin.get("price_change_percentage_24h", 0),
                            "currency": "USD",
                            "exchange": "CoinGecko",
                            "market_cap": coin.get("market_cap", 0),
                            "name": coin.get("name", "")
                        })
                    
                    if result:
                        logger.info(f"Fallback fetch successful: {len(result)} assets from CoinGecko")
                        return result
                        
        except Exception as e:
            logger.debug(f"CoinGecko fallback failed: {e}")
        
        # Try CoinPaprika public API (no key required) 
        try:
            url = "https://api.coinpaprika.com/v1/tickers"
            params = {"limit": max_universe or 50}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                result = []
                
                for coin in data[:max_universe or 50]:
                    result.append({
                        "asset": coin.get("symbol", ""),
                        "symbol": f"{coin.get('symbol', '')}USD",
                        "price": coin.get("quotes", {}).get("USD", {}).get("price", 0),
                        "volume": coin.get("quotes", {}).get("USD", {}).get("volume_24h", 0),
                        "day_range_pct": coin.get("quotes", {}).get("USD", {}).get("percent_change_24h", 0),
                        "currency": "USD",
                        "exchange": "CoinPaprika",
                        "name": coin.get("name", "")
                    })
                
                if result:
                    logger.info(f"Fallback fetch successful: {len(result)} assets from CoinPaprika")
                    return result
                    
        except Exception as e:
            logger.debug(f"CoinPaprika fallback failed: {e}")
            
    except ImportError:
        logger.error("requests library not available for fallback fetch")
    except Exception as e:
        logger.error(f"Fallback fetch failed: {e}")
    
    # If all fallback attempts fail, return empty list
    logger.error("All fallback attempts failed - no data available")
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
            # Also check if symbol without USD suffix matches
            for req_symbol in symbols_upper:
                if req_symbol.replace("USD", "") == asset_symbol or req_symbol.replace("USDT", "") == asset_symbol:
                    if item not in filtered:
                        filtered.append(item)
        data = filtered
    
    # Sort by volume or market cap if available (highest first)
    if data:
        if "market_cap" in data[0]:
            data = sorted(data, key=lambda x: x.get("market_cap", 0), reverse=True)
        elif "volume" in data[0]:
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
    
    # Try to fetch from available adapters that support OHLCV data
    history_capable_adapters = [
        "Binance", "CoinGecko", "Kraken", "Coinbase", 
        "KuCoin", "CryptoCompare", "OKX", "Bybit"
    ]
    
    for source_name in history_capable_adapters:
        if source_name not in AVAILABLE_ADAPTERS:
            continue
            
        adapter_class = AVAILABLE_ADAPTERS[source_name]
        
        try:
            # Check if source is configured
            if hasattr(adapter_class, 'is_configured'):
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
    
    for source_name, adapter_class in AVAILABLE_ADAPTERS.items():
        try:
            sources_info[source_name] = {
                "available": True,
                "configured": adapter_class.is_configured() if hasattr(adapter_class, 'is_configured') else True,
                "requires_api_key": getattr(adapter_class, 'requires_api_key', False),
                "supports_history": hasattr(adapter_class, 'fetch_history') or hasattr(adapter_class, 'fetch_ohlcv'),
                "tier": _get_source_tier(source_name)
            }
        except Exception as e:
            sources_info[source_name] = {
                "available": True,
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
        "CoinPaprika": 1,
        
        # Tier 2: Major exchanges
        "Binance": 2,
        "Coinbase": 2,
        "Kraken": 2,
        "KuCoin": 2,
        
        # Tier 3: Other exchanges
        "OKX": 3,
        "Bybit": 3,
        "GateIO": 3,
        "Huobi": 3,
        "Bitstamp": 3,
        "Gemini": 3,
        
        # Tier 4: Data aggregators
        "CoinMarketCap": 4,
        "CryptoCompare": 4,
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
    
    for source_name, adapter_class in AVAILABLE_ADAPTERS.items():
        try:
            if hasattr(adapter_class, 'is_configured'):
                if not adapter_class.is_configured():
                    test_results[source_name] = {
                        "status": "not_configured",
                        "message": "API key or configuration missing"
                    }
                    continue
            
            # Try to fetch a single symbol
            adapter = adapter_class()
            start_time = time.time()
            
            data = None
            if hasattr(adapter, 'fetch_symbols'):
                data = adapter.fetch_symbols([test_symbol])
            elif hasattr(adapter, 'get_quotes'):
                data = adapter.get_quotes([test_symbol])
            elif hasattr(adapter, 'fetch_top_cryptos'):
                data = adapter.fetch_top_cryptos(limit=1)
            elif hasattr(adapter, 'get_top_coins'):
                data = adapter.get_top_coins(limit=1)
            
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
