# Trading Assistant – Backend (CLI)

AI‑assisted trading CLI that:

- supports 7 categories: crypto, forex, equities, commodities, futures, warrants, funds
- selects region → market with colored open/closed dots
- fetches live quotes + (optional) history
- computes multi‑indicator signals (technical, fundamentals, sentiment, microstructure)
- fuses signals into a Buy/Sell/Hold view with confidence
- offers a single‑asset analysis path where you pick the symbol and indicators
- persists sessions and prints a scorecard next time you run

This backend is organized to be reusable for a future web API (FE later).
It runs fully from the terminal today.

# 1) Quick start
```
cd TA/BE

# 1) Create virtual env + install deps
make venv
make deps        # or: pip install -r requirements.txt

# 2) (optional) copy env template and set keys
cp .env.example .env
# edit .env and add keys if you have them (TWELVEDATA_API_KEY, CRYPTOCOMPARE_API_KEY, OPENAI_API_KEY)

# 3) Run
make run
```

You’ll see:

- mode selection (category flow vs single‑asset)
- region/market menus with 🟢🟠🔴 status and local‑time sessions
- feature toggles (RSI, SMA, Sentiment)
- recommendations + fee table
- diagnostics about data sources used/failed/skipped

# 2) Project layout
```
TA/
└─ BE/
   ├─ app_cli/
   │  ├─ __init__.py
   │  ├─ main.py            # CLI entry; category flow + single-asset flow
   │  └─ terminal_ui.py     # All prompts/menus; region/market selection with dots + hours
   │
   ├─ trading_core/
   │  ├─ __init__.py
   │  ├─ config.py          # Loads markets.yml/seeds.yml, env helpers, market sessions & clocks
   │  ├─ data/
   │  │  ├─ markets.yml     # Market metadata (labels, tz, trading_days, sessions, region)
   │  │  └─ seeds.yml       # Fallback seed tickers per market (equities/funds)
   │  │
   │  ├─ data_fetcher/
   │  │  ├─ __init__.py     # Facade; diagnostics; single-symbol fetch helper
   │  │  ├─ equities.py     # Discovery + quotes/history (Yahoo→Stooq; TwelveData if key)
   │  │  ├─ crypto.py       # CoinGecko→(Paprika/Cap)→CryptoCompare fallback chain
   │  │  ├─ forex.py        # Region-aware FX universe + quotes
   │  │  ├─ commodities.py  # Metals + local ETF proxies
   │  │  ├─ futures.py      # Index cash proxies + index futures
   │  │  ├─ warrants.py     # Leveraged ETP proxy layer (region-aware)
   │  │  ├─ funds.py        # ETFs (local first, then US core)
   │  │  └─ adapters/
   │  │     ├─ yahoo.py         # yfinance + quote json + simple caches
   │  │     ├─ stooq.py         # CSV helper
   │  │     ├─ coingecko.py     # REST helper
   │  │     ├─ cryptocompare.py # REST (keyed)
   │  │     └─ twelvedata.py    # REST (keyed; equities/fx/crypto)
   │  │
   │  ├─ indicators/
   │  │  ├─ __init__.py
   │  │  ├─ price_history.py   # MTF resampling/alignment; simple OHLCV helpers
   │  │  ├─ technical.py       # SMA, EMA, MACD, ADX, RSI, Stoch, OBV, BBands, ATR, volume spikes
   │  │  ├─ fundamentals.py    # Stocks (P/E, growth, debt…) + Crypto proxies (tx count, dev activity…)
   │  │  ├─ sentiment.py       # Normalized sentiment features (news, social, Fear & Greed)
   │  │  └─ microstructure.py  # Order book imbalance, depth, whale trades (if data provided)
   │  │
   │  ├─ sentiment/
   │  │  ├─ __init__.py
   │  │  ├─ equities.py        # r/stocks, business news, Yahoo trending (region-aware)
   │  │  ├─ crypto.py          # CoinDesk/Telegraph, Fear & Greed, social scan
   │  │  └─ utils.py           # RSS helpers, dedup, scoring
   │  │
   │  ├─ scoring/
   │  │  ├─ __init__.py
   │  │  ├─ regime.py          # bull/bear/range/volatile (multi-timeframe)
   │  │  ├─ risk.py            # stop-loss, sizing, portfolio correlation checks
   │  │  ├─ weights.py         # combine indicators → single confidence score (regime-aware)
   │  │  └─ explainer.py       # “Key Reasons” from top weighted signals
   │  │
   │  ├─ strategy/
   │  │  ├─ __init__.py
   │  │  ├─ rules_engine.py    # deterministic logic (no LLM)
   │  │  └─ strategy_llm.py    # LLM wrapper (optional; JSON I/O only)
   │  │
   │  ├─ persistence/
   │  │  ├─ __init__.py
   │  │  ├─ history_tracker.py     # session logs
   │  │  ├─ performance_evaluator.py # next-day scorecard, hit/stop
   │  │  └─ cache.py               # simple file cache (quotes/headlines)
   │  │
   │  └─ utils/
   │     ├─ __init__.py
   │     ├─ timezones.py
   │     ├─ io.py
   │     └─ logging.py
   │
   ├─ Makefile
   ├─ requirements.txt
   ├─ pyproject.toml
   ├─ .env.example
   └─ README.md   # (this file)
```
# 3) Configuration
Environment variables (.env)
```
# Data providers
TWELVEDATA_API_KEY=your_key_here        # optional; becomes first-choice for quotes/history
CRYPTOCOMPARE_API_KEY=your_key_here     # optional fallback for crypto quotes

# Optional LLM
OPENAI_API_KEY=sk-...

# Tuning / misc
FORCE_TWELVEDATA=true                   # optional – force adapter preference
TA_DATA_CACHE_DIR=.cache                # optional – where to store small caches

```
TwelveData is optional. If present, equities/forex/crypto fetchers prefer it when helpful; otherwise they fall back to Yahoo / CoinGecko chains automatically.

Markets & seeds

- `trading_core/data/markets.yml` defines markets (label, region, timezone, `trading_days`, `sessions`).
The terminal UI uses this to print region → market menus with 🟢🟠🔴 and shows sessions in your local time.
- `trading_core/data/seeds.yml` provides fallback tickers per market for equities/funds when screeners are unavailable.

# 4) Using the CLI

## A) Category flow (original workflow)
```
make run
# choose: Category flow
# pick: equities (or others)
# choose: region (Americas / Europe / Middle East & Africa / Asia-Pacific)
# choose: market (NYSE, LSE, TADAWUL, etc.) – hours shown in YOUR local time
# toggle features (RSI/SMA/Sentiment)
# enter budget
```

You’ll get:

- Recommendations table (Asset, Qty, Buy, Cost, Target, Sell@, Profit)
- Fee & Net Profit table (venues vs net)
- Feature Check Summary (data source used / failed / skipped)
- Session persisted for the next scorecard

## B) Single‑asset analysis (new)
```
make run
# choose: Single-asset analysis
# enter: symbol (e.g., AAPL, BTC, EURUSD)
# select: asset class (equity | crypto | forex)
# (optional) choose region/market for exchange‑traded assets
# select indicators to run (EMA, MACD, ADX, RSI, Stoch, OBV, BBANDS, ATR, etc.)
# toggle sentiment if desired
# enter budget (for position sizing/stop suggestion)
```

You’ll get a one‑liner in the requested format:
```
[Action: Buy/Sell/Hold | Confidence: 73% | Key Reasons: EMA>price & rising; MACD cross↑; RSI 55 (neutral+); BB midline support]
```

Plus numeric fields for target/stop if risk rules are enabled.

# 5) Indicators & signal fusion

- Technical: SMA, EMA, MACD, ADX, RSI, Stochastic, OBV, Bollinger Bands, ATR, volume spikes.
- Fundamentals:
  - Stocks – P/E, earnings/revenue growth, debt ratios (via adapters when available).
  - Crypto – proxies for dev activity, network transactions, tokenomics.
- Sentiment: business/crypto news feeds, social, Fear & Greed.
- Microstructure: order book imbalance, liquidity depth, whale prints (if you provide data).
- Regime detection: bull / bear / range / volatile, multi‑timeframe.
- Weights: per‑regime weighting → single confidence score.
- Risk: stop‑loss placement, position sizing, simple portfolio correlation checks.

All lives in `trading_core/indicators/*` and `trading_core/scoring/*`.

# 6) Make targets
```
make venv        # create .venv
make deps        # install requirements into .venv
make run         # run CLI (python -m app_cli.main)
make test        # run tests (if/when added)
make fmt         # optional: black formatting
```
# 7) Troubleshooting

“No market data fetched”
Try another market/region, or ensure you have internet. TwelveData key can improve non‑US/EMEA coverage.

Some markets show 🔴 but are open
Our market clock uses `trading_core/data/markets.yml` + your local timezone. Check your system TZ and market sessions there.

RSI/SMA skipped
We require enough history (~15 bars). Some symbols on Stooq have limited history; TwelveData improves this.

LLM features
Only used if `OPENAI_API_KEY` is present. Otherwise, the deterministic `rules_engine` runs (default).

Region/market lists
Are driven by YAML. You can extend `markets.yml` and `seeds.yml` without touching code.

# 8) Roadmap (nice‑to‑have)

- REST API wrapper for this backend (so FE can call JSON endpoints)
- Better microstructure connectors (exchange websockets)
- Portfolio view + risk aggregation across recommendations
- Async fetch with caching + retries
- Strategy presets (swing/day/scalp) with different weights
