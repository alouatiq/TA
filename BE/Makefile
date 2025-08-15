# TA/BE/Makefile
# Simple developer workflow:
#   make venv        # create virtualenv
#   make install     # install requirements into venv
#   make run         # run CLI (loads .env if present)
#   make test        # run pytest
#   make clean       # remove caches/pyc
#   make freeze      # write a resolved requirements lock snapshot

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
.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  make venv       - Create virtualenv in ./venv"
	@echo "  make install    - Install requirements into venv"
	@echo "  make run        - Run CLI (app_cli/main.py); pass ARGS=\"...\" for options"
	@echo "  make test       - Run pytest"
	@echo "  make freeze     - Export fully-resolved requirements to requirements.lock"
	@echo "  make clean      - Remove pyc caches and pytest caches"
	@echo "  make clean-venv - Remove the venv directory"
	@echo ""

$(VENV_DIR)/bin/activate: | $(VENV_DIR)
	@true

$(VENV_DIR):
	@echo "🐍 Creating virtualenv at $(VENV_DIR)"
	@$(PYTHON_BIN) -m venv $(VENV_DIR)

.PHONY: venv
venv: $(VENV_DIR)

.PHONY: install
install: venv
	@echo "📦 Installing requirements"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

.PHONY: run
run: install
	@echo "🚀  Running the trading assistant..."
	@set -a; \
	if [ -f .env ]; then echo "🔧 Loading .env"; . ./.env; fi; \
	set +a; \
	export PYTHONPATH="$(PWD)"; \
	$(PY) -m app_cli.main $(ARGS)

.PHONY: test
test: install
	@echo "🧪 Running tests"
	@export PYTHONPATH="$(PWD)"; \
	$(PYTEST) -q

.PHONY: freeze
freeze: install
	@echo "🔌 Writing locked requirements to requirements.lock"
	@$(PIP) freeze > requirements.lock

.PHONY: clean
clean:
	@echo "🧹 Cleaning caches"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	@rm -rf .pytest_cache || true

.PHONY: clean-venv
clean-venv:
	@echo "🗑️  Removing venv"
	@rm -rf $(VENV_DIR)
