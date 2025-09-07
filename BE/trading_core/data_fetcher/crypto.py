# trading_core/data_fetcher/crypto.py
"""
crypto.py - Live Cryptocurrency Data Fetcher
═══════════════════════════════════════════════

This module fetches LIVE cryptocurrency data from real market APIs to identify
actual trading opportunities. NO hard-coded data, NO static seeds - everything
is based on real market conditions and trending/performing assets.

Strategy:
1. Fetch trending/gaining cryptocurrencies from multiple sources
2. Get real-time prices and 14-day historical data
3. Return only assets with sufficient data for technical analysis
4. Focus on volume and performance-based discovery

Fallback chain for discovery:
1. CoinGecko trending + markets (volume sorted)
2. CoinPaprika (volume sorted)
3. CoinCap (market cap sorted)
4. Manual discovery through multiple endpoints

NO static data - if APIs fail, return empty list to force user to check connection.
"""

from __future__ import annotations

import os
import time
import random
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta, timezone

import requests

# Import logging
try:
    from trading_core.utils.logging import get_logger
    log = get_logger(__name__)
except ImportError:
    import logging
    log = logging.getLogger(__name__)

# Configuration
REQUEST_TIMEOUT = 15
DEFAULT_LIMIT = 20
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.2

# Diagnostics
LAST_CRYPTO_SOURCE: str = "None"
FAILED_CRYPTO_SOURCES: List[str] = []
SKIPPED_CRYPTO_SOURCES: List[str] = []


# ════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ════════════════════════════════════════════════════════════════════════════

def _safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float, return None if not possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_request(url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
    """Make a safe HTTP request with retries and error handling."""
    if headers is None:
        headers = {"User-Agent": "Trading-Assistant/1.0"}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 429:  # Rate limited
                wait_time = RATE_LIMIT_DELAY * (2 ** attempt)
                log.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
                continue
            
            if response.ok:
                return response.json()
            else:
                log.warning(f"HTTP {response.status_code} for {url}")
                
        except requests.exceptions.Timeout:
            log.warning(f"Timeout on attempt {attempt + 1} for {url}")
        except requests.exceptions.RequestException as e:
            log.warning(f"Request error on attempt {attempt + 1}: {e}")
        except Exception as e:
            log.error(f"Unexpected error: {e}")
        
        if attempt < MAX_RETRIES - 1:
            time.sleep(RATE_LIMIT_DELAY)
    
    return None


# ════════════════════════════════════════════════════════════════════════════
# CoinGecko Implementation - Primary source for live trending data
# ════════════════════════════════════════════════════════════════════════════

def _fetch_coingecko_trending() -> List[str]:
    """Get trending cryptocurrencies from CoinGecko."""
    url = "https://api.coingecko.com/api/v3/search/trending"
    
    try:
        data = _safe_request(url)
        if not data:
            return []
        
        trending_coins = []
        for coin_data in data.get("coins", []):
            coin = coin_data.get("item", {})
            coin_id = coin.get("id")
            if coin_id:
                trending_coins.append(coin_id)
        
        log.info(f"CoinGecko trending returned {len(trending_coins)} coins")
        return trending_coins
        
    except Exception as e:
        log.error(f"CoinGecko trending failed: {e}")
        return []


def _fetch_coingecko_markets(limit: int) -> List[Dict[str, Any]]:
    """Get top cryptocurrencies by volume from CoinGecko markets."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    
    params = {
        "vs_currency": "usd",
        "order": "volume_desc",  # Sort by 24h volume
        "per_page": min(limit * 2, 100),  # Get more than needed for filtering
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    
    try:
        data = _safe_request(url, params)
        if not data:
            return []
        
        assets = []
        for coin in data:
            # Only include coins with significant volume and price data
            if (coin.get("total_volume", 0) > 1000000 and  # Min $1M volume
                coin.get("current_price", 0) > 0 and
                coin.get("market_cap", 0) > 10000000):  # Min $10M market cap
                
                assets.append({
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol", "").upper(),
                    "price": coin.get("current_price"),
                    "volume": coin.get("total_volume"),
                    "market_cap": coin.get("market_cap"),
                    "price_change_24h": coin.get("price_change_percentage_24h", 0),
                    "source": "coingecko_markets"
                })
        
        log.info(f"CoinGecko markets returned {len(assets)} valid assets")
        return assets
        
    except Exception as e:
        log.error(f"CoinGecko markets failed: {e}")
        return []


def _fetch_coingecko_history(coin_id: str, days: int) -> List[float]:
    """Fetch historical price data for a specific coin from CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    
    params = {
        "vs_currency": "usd",
        "days": days + 2,  # Get a bit more data for safety
        "interval": "daily"
    }
    
    try:
        data = _safe_request(url, params)
        if not data or "prices" not in data:
            return []
        
        # Extract closing prices (prices come as [timestamp, price] pairs)
        prices = []
        for price_point in data["prices"]:
            if len(price_point) >= 2 and price_point[1] is not None:
                prices.append(float(price_point[1]))
        
        # Return the last 'days' prices
        return prices[-days:] if len(prices) >= days else prices
        
    except Exception as e:
        log.warning(f"CoinGecko history failed for {coin_id}: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# CoinPaprika Implementation - Backup source
# ════════════════════════════════════════════════════════════════════════════

def _fetch_coinpaprika_markets(limit: int) -> List[Dict[str, Any]]:
    """Get top cryptocurrencies from CoinPaprika."""
    url = "https://api.coinpaprika.com/v1/tickers"
    
    try:
        data = _safe_request(url)
        if not data:
            return []
        
        # Filter and sort by volume
        valid_coins = []
        for coin in data:
            quotes = coin.get("quotes", {}).get("USD", {})
            volume_24h = quotes.get("volume_24h", 0)
            price = quotes.get("price", 0)
            market_cap = quotes.get("market_cap", 0)
            
            # Filter for active, liquid markets
            if (volume_24h and volume_24h > 1000000 and  # Min $1M volume
                price and price > 0 and
                market_cap and market_cap > 5000000):  # Min $5M market cap
                
                valid_coins.append({
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol", "").upper(),
                    "price": price,
                    "volume": volume_24h,
                    "market_cap": market_cap,
                    "price_change_24h": quotes.get("percent_change_24h", 0),
                    "source": "coinpaprika"
                })
        
        # Sort by volume and return top assets
        sorted_coins = sorted(valid_coins, key=lambda x: x["volume"], reverse=True)
        
        log.info(f"CoinPaprika returned {len(sorted_coins)} valid assets")
        return sorted_coins[:limit]
        
    except Exception as e:
        log.error(f"CoinPaprika failed: {e}")
        return []


def _fetch_coinpaprika_history(coin_id: str, days: int) -> List[float]:
    """Fetch historical data from CoinPaprika."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days + 2)
    
    url = f"https://api.coinpaprika.com/v1/tickers/{coin_id}/historical"
    params = {
        "start": start_date.strftime("%Y-%m-%d"),
        "interval": "1d"
    }
    
    try:
        data = _safe_request(url, params)
        if not data:
            return []
        
        prices = []
        for point in data:
            price = point.get("price")
            if price is not None:
                prices.append(float(price))
        
        return prices[-days:] if len(prices) >= days else prices
        
    except Exception as e:
        log.warning(f"CoinPaprika history failed for {coin_id}: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# CoinCap Implementation - Third fallback
# ════════════════════════════════════════════════════════════════════════════

def _fetch_coincap_markets(limit: int) -> List[Dict[str, Any]]:
    """Get top cryptocurrencies from CoinCap."""
    url = "https://api.coincap.io/v2/assets"
    
    params = {
        "limit": limit * 2  # Get more for filtering
    }
    
    try:
        data = _safe_request(url, params)
        if not data or "data" not in data:
            return []
        
        assets = []
        for coin in data["data"]:
            volume = _safe_float(coin.get("volumeUsd24Hr"))
            price = _safe_float(coin.get("priceUsd"))
            market_cap = _safe_float(coin.get("marketCapUsd"))
            
            # Filter for liquid markets
            if (volume and volume > 1000000 and
                price and price > 0 and
                market_cap and market_cap > 5000000):
                
                assets.append({
                    "id": coin.get("id"),
                    "symbol": coin.get("symbol", "").upper(),
                    "price": price,
                    "volume": volume,
                    "market_cap": market_cap,
                    "price_change_24h": _safe_float(coin.get("changePercent24Hr")) or 0,
                    "source": "coincap"
                })
        
        log.info(f"CoinCap returned {len(assets)} valid assets")
        return assets[:limit]
        
    except Exception as e:
        log.error(f"CoinCap failed: {e}")
        return []


def _fetch_coincap_history(asset_id: str, days: int) -> List[float]:
    """Fetch historical data from CoinCap."""
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_time = end_time - (days + 2) * 24 * 60 * 60 * 1000
    
    url = f"https://api.coincap.io/v2/assets/{asset_id}/history"
    params = {
        "interval": "d1",
        "start": start_time,
        "end": end_time
    }
    
    try:
        data = _safe_request(url, params)
        if not data or "data" not in data:
            return []
        
        prices = []
        for point in data["data"]:
            price = _safe_float(point.get("priceUsd"))
            if price:
                prices.append(price)
        
        return prices[-days:] if len(prices) >= days else prices
        
    except Exception as e:
        log.warning(f"CoinCap history failed for {asset_id}: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# Main Data Assembly Functions
# ════════════════════════════════════════════════════════════════════════════

def _enrich_with_history(assets: List[Dict[str, Any]], history_days: int) -> List[Dict[str, Any]]:
    """Add historical price data to assets using the same source that provided them."""
    enriched_assets = []
    
    for asset in assets:
        asset_id = asset.get("id")
        source = asset.get("source", "unknown")
        
        if not asset_id:
            continue
        
        # Fetch history from the same source
        price_history = []
        if source == "coingecko_markets":
            price_history = _fetch_coingecko_history(asset_id, history_days)
        elif source == "coinpaprika":
            price_history = _fetch_coinpaprika_history(asset_id, history_days)
        elif source == "coincap":
            price_history = _fetch_coincap_history(asset_id, history_days)
        
        # Only include assets with sufficient historical data
        if len(price_history) >= max(5, history_days // 2):  # At least 5 days or half requested
            asset_copy = asset.copy()
            asset_copy["price_history"] = price_history
            
            # Calculate day range from price history if not available
            if "day_range_pct" not in asset_copy and len(price_history) >= 2:
                recent_prices = price_history[-5:]  # Last 5 days
                if recent_prices:
                    price_min = min(recent_prices)
                    price_max = max(recent_prices)
                    current_price = asset.get("price", recent_prices[-1])
                    if current_price > 0:
                        day_range_pct = ((price_max - price_min) / current_price) * 100
                        asset_copy["day_range_pct"] = round(day_range_pct, 2)
            
            enriched_assets.append(asset_copy)
        else:
            log.warning(f"Insufficient history for {asset_id}: {len(price_history)} days")
    
    return enriched_assets


def _discover_trending_assets(limit: int) -> List[Dict[str, Any]]:
    """Discover trending and high-volume cryptocurrencies from multiple sources."""
    all_assets = []
    
    # Try CoinGecko first (best for trending + volume data)
    try:
        log.info("Attempting CoinGecko markets discovery...")
        coingecko_assets = _fetch_coingecko_markets(limit)
        if coingecko_assets:
            all_assets.extend(coingecko_assets)
            global LAST_CRYPTO_SOURCE
            LAST_CRYPTO_SOURCE = "CoinGecko"
            log.info(f"CoinGecko provided {len(coingecko_assets)} assets")
    except Exception as e:
        FAILED_CRYPTO_SOURCES.append(f"CoinGecko ({e})")
        log.error(f"CoinGecko discovery failed: {e}")
    
    # If we don't have enough assets, try CoinPaprika
    if len(all_assets) < limit:
        try:
            log.info("Attempting CoinPaprika discovery...")
            coinpaprika_assets = _fetch_coinpaprika_markets(limit - len(all_assets))
            if coinpaprika_assets:
                all_assets.extend(coinpaprika_assets)
                if LAST_CRYPTO_SOURCE == "None":
                    LAST_CRYPTO_SOURCE = "CoinPaprika"
                log.info(f"CoinPaprika provided {len(coinpaprika_assets)} additional assets")
        except Exception as e:
            FAILED_CRYPTO_SOURCES.append(f"CoinPaprika ({e})")
            log.error(f"CoinPaprika discovery failed: {e}")
    
    # If still not enough, try CoinCap
    if len(all_assets) < limit:
        try:
            log.info("Attempting CoinCap discovery...")
            coincap_assets = _fetch_coincap_markets(limit - len(all_assets))
            if coincap_assets:
                all_assets.extend(coincap_assets)
                if LAST_CRYPTO_SOURCE == "None":
                    LAST_CRYPTO_SOURCE = "CoinCap"
                log.info(f"CoinCap provided {len(coincap_assets)} additional assets")
        except Exception as e:
            FAILED_CRYPTO_SOURCES.append(f"CoinCap ({e})")
            log.error(f"CoinCap discovery failed: {e}")
    
    # Remove duplicates and sort by volume
    seen_symbols = set()
    unique_assets = []
    
    for asset in all_assets:
        symbol = asset.get("symbol", "")
        if symbol and symbol not in seen_symbols:
            seen_symbols.add(symbol)
            unique_assets.append(asset)
    
    # Sort by volume (highest first) and return top assets
    sorted_assets = sorted(unique_assets, key=lambda x: x.get("volume", 0), reverse=True)
    
    return sorted_assets[:limit]


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

def fetch_crypto_data(
    include_history: bool = False,
    *,
    limit: int = DEFAULT_LIMIT,
    history_days: int = 14
) -> List[Dict[str, Any]]:
    """
    Fetch live cryptocurrency data with NO hard-coded fallbacks.
    
    This function discovers trending and high-volume cryptocurrencies from live APIs
    and returns only assets with sufficient data for technical analysis.
    
    Args:
        include_history: Whether to include historical price data
        limit: Maximum number of assets to return
        history_days: Number of days of historical data to fetch
        
    Returns:
        List of cryptocurrency assets with live market data.
        Returns empty list if all APIs fail - forces user to check connection.
        
    Each asset dict contains:
        {
            "asset": "bitcoin",                    # API identifier
            "symbol": "BTC-USD",                   # Trading pair symbol
            "price": 65432.10,                    # Current USD price
            "volume": 28500000000.0,               # 24h volume in USD
            "day_range_pct": 3.45,                # Daily price volatility
            "price_history": [65000, 65200, ...], # Historical prices (if requested)
            "market_cap": 1280000000000.0,         # Market capitalization
            "price_change_24h": 2.1,              # 24h price change percentage
            "currency": "USD",                     # Quote currency
            "exchange": "Composite"                # Data source type
        }
    """
    global LAST_CRYPTO_SOURCE, FAILED_CRYPTO_SOURCES, SKIPPED_CRYPTO_SOURCES
    
    # Reset diagnostics
    LAST_CRYPTO_SOURCE = "None"
    FAILED_CRYPTO_SOURCES = []
    SKIPPED_CRYPTO_SOURCES = []
    
    log.info(f"Starting crypto data fetch: limit={limit}, history={include_history}, days={history_days}")
    
    try:
        # Step 1: Discover trending/high-volume assets
        print("Discovering trending cryptocurrencies from live APIs...")
        discovered_assets = _discover_trending_assets(limit)
        
        if not discovered_assets:
            print("No cryptocurrency data available from any source")
            print("Please check your internet connection and try again")
            return []
        
        print(f"Discovered {len(discovered_assets)} trending cryptocurrencies")
        
        # Step 2: Enrich with historical data if requested
        final_assets = []
        
        if include_history:
            print("Fetching historical price data for technical analysis...")
            enriched_assets = _enrich_with_history(discovered_assets, history_days)
            
            if not enriched_assets:
                print("No assets have sufficient historical data for analysis")
                print("This may be due to API rate limits or temporary issues")
                return []
            
            final_assets = enriched_assets
            print(f"Successfully enriched {len(final_assets)} assets with historical data")
        else:
            final_assets = discovered_assets
        
        # Step 3: Format for trading system
        formatted_assets = []
        for asset in final_assets:
            formatted_asset = {
                "asset": asset.get("id", asset.get("symbol", "unknown")),
                "symbol": f"{asset.get('symbol', 'UNKNOWN')}-USD",
                "price": asset.get("price", 0),
                "volume": asset.get("volume", 0),
                "day_range_pct": asset.get("day_range_pct", 0),
                "currency": "USD",
                "exchange": "Composite"
            }
            
            # Add optional fields
            if "market_cap" in asset:
                formatted_asset["market_cap"] = asset["market_cap"]
            if "price_change_24h" in asset:
                formatted_asset["price_change_24h"] = asset["price_change_24h"]
            if "price_history" in asset:
                formatted_asset["price_history"] = asset["price_history"]
            
            formatted_assets.append(formatted_asset)
        
        log.info(f"Successfully fetched {len(formatted_assets)} crypto assets from {LAST_CRYPTO_SOURCE}")
        return formatted_assets
        
    except Exception as e:
        log.error(f"Critical error in crypto data fetch: {e}")
        FAILED_CRYPTO_SOURCES.append(f"System ({e})")
        print(f"Critical error fetching cryptocurrency data: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# Module exports
# ════════════════════════════════════════════════════════════════════════════

__all__ = [
    "fetch_crypto_data",
    "LAST_CRYPTO_SOURCE",
    "FAILED_CRYPTO_SOURCES", 
    "SKIPPED_CRYPTO_SOURCES"
]
