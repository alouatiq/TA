"""
CLI entrypoints for the Trading Assistant (TA).

This package wires the terminal UX (menus/prompts) to the reusable
business logic in `trading_core`. Nothing here should contain
provider-specific code; keep that inside trading_core.

Modules
-------
- main.py
    The executable entry-point for the CLI. Preserves your existing
    flow (category selection, budget, features) and adds an optional
    “analyze single asset” path.

- terminal_ui.py
    All input/output routines for the terminal: menus, choices,
    validation, pretty printing helpers. Keeping UI separate from
    logic makes it trivial to swap in a web API later.

Conventions
-----------
- All public functions in terminal_ui.py should be **pure UI**:
  they gather user input and return plain data structures
  (strings, numbers, dicts). They should not call network APIs.

- `main.py` is the only module that orchestrates data fetching,
  indicators, scoring, and strategy. It imports from:
    trading_core.data_fetcher
    trading_core.indicators
    trading_core.sentiment
    trading_core.scoring
    trading_core.strategy
    trading_core.persistence
    trading_core.utils

- Environment variables (e.g., API keys) are loaded by
  `trading_core.config`. The CLI should not read env vars directly.

Run
---
`python -m app_cli.main` or use `make run`.

"""
__all__ = ["main", "terminal_ui"]
