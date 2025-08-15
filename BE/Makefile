# Makefile – Trading Assistant Backend automation
# ================================================
#  Common targets
#  ─────────────
#   make setup                  ▸ create venv + install deps
#   make run                    ▸ start assistant  (ARGS="..." → app_cli.main)
#   make test                   ▸ run pytest
#   make clean                  ▸ delete logs & caches, keep venv
#   make nuke                   ▸ full wipe: venv + logs + caches
#   make freeze                 ▸ export locked requirements
#   make help                   ▸ list targets / docs

SHELL := /bin/bash

# Paths
VENV_DIR := venv
PY      := $(VENV_DIR)/bin/python
PIP     := $(VENV_DIR)/bin/pip
PYTEST  := $(VENV_DIR)/bin/pytest

# Allow passing CLI args: e.g. make run ARGS="--single-asset AAPL --asset-class equity"
ARGS ?=

# Detect OS-friendly python
PYTHON_BIN ?= python3

# Default target
.DEFAULT_GOAL := run

# ------------------------------------------------------------------
# Setup & dependencies (only when needed)
# ------------------------------------------------------------------

# Create venv directory if it doesn't exist
$(VENV_DIR):
	@echo "🐍 Creating virtual environment at $(VENV_DIR)"
	@$(PYTHON_BIN) -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created"

# Install marker file - only reinstall if requirements.txt is newer
$(VENV_DIR)/.installed: $(VENV_DIR) requirements.txt
	@echo "📦 Installing/updating requirements (requirements.txt changed)"
	@$(PIP) install --upgrade pip -q
	@$(PIP) install -r requirements.txt -q
	@touch $(VENV_DIR)/.installed
	@echo "✅ Packages installed successfully"

setup: $(VENV_DIR)/.installed  ## Create venv & install deps (only if needed)
	@echo "✅ Setup complete!"

install: $(VENV_DIR)/.installed  ## Install/upgrade Python packages (only if needed)

# Force reinstall (ignores timestamps)
force-install: $(VENV_DIR)      ## Force reinstall packages
	@echo "🔄 Force reinstalling packages from requirements.txt..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt --force-reinstall
	@touch $(VENV_DIR)/.installed

# ------------------------------------------------------------------
# Main entry points
# ------------------------------------------------------------------

run: $(VENV_DIR)/.installed     ## Launch assistant (ARGS pass to app_cli.main)
	@echo "🚀 Running the trading assistant..."
	@set -a; \
	if [ -f .env ]; then echo "🔧 Loading .env"; . ./.env; fi; \
	set +a; \
	export PYTHONPATH="$(PWD)"; \
	$(PY) -m app_cli.main $(ARGS)

test: $(VENV_DIR)/.installed    ## Run pytest
	@echo "🧪 Running tests"
	@export PYTHONPATH="$(PWD)"; \
	$(PYTEST) -q

# ------------------------------------------------------------------
# Development utilities
# ------------------------------------------------------------------

freeze: $(VENV_DIR)/.installed  ## Export locked requirements
	@echo "🔌 Writing locked requirements to requirements.lock"
	@$(PIP) freeze > requirements.lock

dev-install: $(VENV_DIR)       ## Install package in development mode
	@echo "🔧 Installing package in development mode..."
	@$(PIP) install -e .

lint: $(VENV_DIR)/.installed    ## Run code linting (if you have linting tools)
	@echo "🔍 Running linting..."
	@if $(PIP) show black >/dev/null 2>&1; then \
		echo "  → Running black..."; \
		$(PY) -m black --check .; \
	fi
	@if $(PIP) show flake8 >/dev/null 2>&1; then \
		echo "  → Running flake8..."; \
		$(PY) -m flake8 .; \
	fi

format: $(VENV_DIR)/.installed  ## Auto-format code (if you have black)
	@echo "✨ Formatting code..."
	@if $(PIP) show black >/dev/null 2>&1; then \
		$(PY) -m black .; \
	else \
		echo "⚠️  Black not installed. Run: $(PIP) install black"; \
	fi

# ------------------------------------------------------------------
# Utility / housekeeping
# ------------------------------------------------------------------

check-env:                      ## Check if .env file exists and show sample
	@if [ -f .env ]; then \
		echo "✅ .env file exists"; \
	else \
		echo "⚠️  .env file not found. Creating from .env.example..."; \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "📝 .env created from .env.example - please edit with your API keys"; \
		else \
			echo "❌ No .env.example found - please create .env manually"; \
		fi \
	fi

status:                         ## Show current environment status
	@echo "📊 Environment Status:"
	@echo "  Python: $(shell $(PY) --version 2>/dev/null || echo 'Not installed')"
	@echo "  Venv: $(shell [ -d $(VENV_DIR) ] && echo '✅ Exists' || echo '❌ Missing')"
	@echo "  Packages: $(shell [ -f $(VENV_DIR)/.installed ] && echo '✅ Installed' || echo '❌ Not installed')"
	@echo "  .env: $(shell [ -f .env ] && echo '✅ Exists' || echo '❌ Missing')"
	@echo "  Requirements: $(shell [ -f requirements.txt ] && echo '✅ Exists' || echo '❌ Missing')"

clean:                          ## Remove logs & caches (keeps venv)
	@echo "🧹 Cleaning workspace (keeping venv)..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@echo "✅ Cleanup complete."

clean-venv:                     ## Remove only the virtual environment
	@echo "🗑️  Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "✅ Virtual environment removed."

nuke: clean clean-venv          ## FULL wipe: venv + logs + caches
	@echo "💣 NUKE mode complete - environment fully reset."

reset-env:                      ## Reset .env file to example template
	@echo "♻️  Resetting .env file..."
	@if [ -f .env ]; then \
		cp .env .env.backup.$(shell date +%Y%m%d_%H%M%S); \
		echo "📋 Backed up existing .env"; \
	fi
	@if [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "✅ Reset .env from .env.example template"; \
		echo "⚠️  Please edit .env and add your real API keys"; \
	else \
		echo "❌ No .env.example found"; \
	fi

# ------------------------------------------------------------------
# Debugging & info
# ------------------------------------------------------------------

debug: $(VENV_DIR)/.installed   ## Run in debug mode with verbose output
	@echo "🐛 Running in debug mode..."
	@set -a; \
	if [ -f .env ]; then . ./.env; fi; \
	set +a; \
	export PYTHONPATH="$(PWD)"; \
	export DEBUG=1; \
	$(PY) -m app_cli.main $(ARGS)

info:                           ## Show system and project info
	@echo "📋 System Information:"
	@echo "  OS: $(shell uname -s)"
	@echo "  Architecture: $(shell uname -m)"
	@echo "  Python system: $(shell python3 --version 2>/dev/null || echo 'Not found')"
	@echo "  Make: $(shell make --version | head -1)"
	@echo ""
	@echo "📋 Project Information:"
	@echo "  Directory: $(PWD)"
	@echo "  Venv path: $(VENV_DIR)"
	@echo "  Python path: $(PY)"
	@echo "  Requirements: $(shell [ -f requirements.txt ] && wc -l < requirements.txt || echo '0') packages"

help:                           ## Show this help message
	@echo ""
	@echo "Trading Assistant Backend - Available Commands:"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	| sort | awk 'BEGIN{FS=":.*?##"}{printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make run                     # Start with default settings"
	@echo "  make run ARGS=\"--help\"       # Show CLI help"
	@echo "  make setup                   # First-time setup"
	@echo "  make clean && make setup     # Clean reinstall"
	@echo ""

# ------------------------------------------------------------------
# Special targets
# ------------------------------------------------------------------

.PHONY: setup install force-install run test freeze dev-install lint format
.PHONY: check-env status clean clean-venv nuke reset-env debug info help

# Prevent deletion of intermediate files
.PRECIOUS: $(VENV_DIR) $(VENV_DIR)/.installed
