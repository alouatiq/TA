```
TA/
├─ FE/                                   # (Front-end placeholder – not building now)
│  └─ README.md                          # Notes for future web UI (framework, API endpoints)
│
└─ BE/                                   # Backend – focus here
   ├─ pyproject.toml                     # Build metadata (optional) or keep requirements.txt + Makefile
   ├─ requirements.txt                   # Pinned deps (yfinance, requests, feedparser, ta, pandas, numpy, tqdm, openai, etc.)
   ├─ Makefile                           # `make venv`, `make run`, `make test`
   ├─ .env.example                       # Sample env vars (API keys + runtime settings)
   ├─ README.md                          # How to run CLI; how to configure; architecture overview
   │
   ├─ app_cli/                           # CLI entrypoints & terminal UX (keeps current workflow)
   │  ├─ __init__.py
   │  ├─ main.py                         # CLI runner (keeps your current menu flow + new “single-asset” path)
   │  └─ terminal_ui.py                  # All user prompts/menus (now includes “analyze specific asset” flow)
   │
   ├─ trading_core/                      # Pure business logic; reusable for CLI and future web API
   │  ├─ __init__.py
   │  ├─ config.py                       # Market calendars, regions, sessions, key loading helpers (OpenAI, TwelveData, etc.)
   │  ├─ data/                           # Static data & mappings
   │  │  ├─ markets.yml                  # Market metadata (labels, tz, sessions, trading days, region)
   │  │  └─ seeds.yml                    # Seed tickers per market (equities, funds, etc.)
   │  │
   │  ├─ data_fetcher/                   # All data acquisition modules
   │  │  ├─ __init__.py
   │  │  ├─ equities.py                  # Equities discovery + quotes (Yahoo + fallback, TwelveData if configured)
   │  │  ├─ crypto.py                    # Crypto discovery + quotes (CoinGecko → Paprika → CoinCap → CryptoCompare)
   │  │  ├─ forex.py                     # FX pairs by market/region; quotes
   │  │  ├─ commodities.py               # Metals futures + optional local ETF proxies
   │  │  ├─ futures.py                   # Index futures/cash proxies by market/region
   │  │  ├─ warrants.py                  # Leveraged ETP “proxy” layer
   │  │  ├─ funds.py                     # ETFs (local first, then US core)
   │  │  └─ adapters/                    # Per‑provider adapters (clean separation)
   │  │     ├─ yahoo.py                  # yfinance + Yahoo Quote JSON + Stooq fallback
   │  │     ├─ stooq.py                  # CSV helper
   │  │     ├─ coingecko.py              # REST helper
   │  │     ├─ cryptocompare.py          # REST helper (keyed)
   │  │     └─ twelvedata.py             # REST helper (keyed; equities/fx/crypto price & history)
   │  │
   │  ├─ indicators/                     # All indicator calculations (vectorized, testable)
   │  │  ├─ __init__.py
   │  │  ├─ price_history.py             # Shared helpers for fetching/aligning OHLCV, resampling MTF
   │  │  ├─ technical.py                 # SMA, EMA, MACD, ADX, RSI, Stoch, OBV, BBands, ATR, volume spikes
   │  │  ├─ fundamentals.py              # Stocks: P/E, EPS growth, revenue, debt; Crypto: on-chain/dev proxies
   │  │  ├─ sentiment.py                 # Normalized sentiment features (news, social, F&G)
   │  │  └─ microstructure.py            # Order book imbalance, depth, whale trades (if source keys set)
   │  │
   │  ├─ sentiment/                      # Raw headline fetchers → normalized signals
   │  │  ├─ __init__.py
   │  │  ├─ equities.py                  # r/stocks, Yahoo trending, business news feeds
   │  │  ├─ crypto.py                    # CoinDesk, CoinTelegraph, Fear & Greed, social scan
   │  │  └─ utils.py                     # RSS helpers, dedup, parsing, scoring
   │  │
   │  ├─ scoring/                        # Signal fusion → recommendation
   │  │  ├─ __init__.py
   │  │  ├─ regime.py                    # Market regime detection (bull/bear/range/volatile; multi-timeframe)
   │  │  ├─ risk.py                      # Stop-loss placement, position sizing, portfolio correlation checks
   │  │  ├─ weights.py                   # Combine indicators → single confidence score (per regime)
   │  │  └─ explainer.py                 # Convert top weighted signals → human “Key Reasons”
   │  │
   │  ├─ strategy/                       # Strategy engines
   │  │  ├─ __init__.py
   │  │  ├─ rules_engine.py              # Deterministic rules (no LLM) for quick sims
   │  │  └─ strategy_llm.py              # LLM wrapper (market-time aware, JSON only, explains reasons)
   │  │
   │  ├─ persistence/                    # Storage & evaluation
   │  │  ├─ __init__.py
   │  │  ├─ history_tracker.py           # Log sessions, recommendations, realized results
   │  │  ├─ performance_evaluator.py     # Next-day scorecard; risk-adjusted metrics; hit/stop tracking
   │  │  └─ cache.py                     # Optional: simple local cache for quotes/headlines
   │  │
   │  └─ utils/                          # Cross-cutting helpers
   │     ├─ __init__.py
   │     ├─ timezones.py                 # TZ conversion, market clock utilities
   │     ├─ io.py                        # Safe file I/O, YAML/JSON helpers
   │     └─ logging.py                   # Unified logger
   │
   └─ tests/
      ├─ test_indicators.py              # Unit tests for SMA/EMA/MACD/… and volatility/volume
      ├─ test_fetchers.py                # Data fetch happy-path & fallback chain tests (mock HTTP)
      ├─ test_scoring.py                 # Regime detection, weighting, risk rules
      └─ test_cli.py                     # CLI paths (category flow + single-asset analysis)
```
