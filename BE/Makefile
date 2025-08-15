# BE/Makefile
# Trading Assistant Backend - Development & Deployment Automation

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PYTHON := python3
VENV_DIR := venv
PIP := $(VENV_DIR)/bin/pip
PYTHON_VENV := $(VENV_DIR)/bin/python
PROJECT_NAME := trading-assistant-backend
SRC_DIR := .
REQUIREMENTS := requirements.txt
ENV_FILE := .env
ENV_EXAMPLE := .env.example

# Colors for pretty output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m # No Color

# ─────────────────────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────────────────────
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "$(CYAN)Trading Assistant Backend - Available Commands$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(GREEN)🏗️  Setup & Installation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(setup|install|deps|venv)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🔑 API Configuration:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(api|key)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🚀 Running & Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(run|dev|test|debug)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)🧹 Maintenance:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E "(clean|format|lint)" | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Setup & Installation
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: venv
venv: ## Create Python virtual environment
	@echo "$(BLUE)🏗️  Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo "$(GREEN)✅ Virtual environment created at $(VENV_DIR)$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment already exists at $(VENV_DIR)$(NC)"; \
	fi

.PHONY: deps
deps: venv ## Install Python dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS)" ]; then \
		$(PIP) install -r $(REQUIREMENTS); \
		echo "$(GREEN)✅ Dependencies installed from $(REQUIREMENTS)$(NC)"; \
	else \
		echo "$(RED)❌ $(REQUIREMENTS) not found$(NC)"; \
		exit 1; \
	fi

.PHONY: setup
setup: venv deps setup-env ## Complete setup (venv + deps + env file)
	@echo "$(GREEN)🎉 Setup complete! Next steps:$(NC)"
	@echo "  1. Configure API keys: $(YELLOW)make setup-api$(NC)"
	@echo "  2. Run the app: $(YELLOW)make run$(NC)"

.PHONY: setup-env
setup-env: ## Create .env file from template
	@echo "$(BLUE)⚙️  Setting up environment file...$(NC)"
	@if [ ! -f "$(ENV_FILE)" ]; then \
		if [ -f "$(ENV_EXAMPLE)" ]; then \
			cp $(ENV_EXAMPLE) $(ENV_FILE); \
			echo "$(GREEN)✅ Created $(ENV_FILE) from $(ENV_EXAMPLE)$(NC)"; \
			echo "$(YELLOW)⚠️  Please edit $(ENV_FILE) to add your API keys$(NC)"; \
		else \
			echo "$(YELLOW)⚠️  $(ENV_EXAMPLE) not found, creating basic $(ENV_FILE)$(NC)"; \
			echo "# Trading Assistant API Keys" > $(ENV_FILE); \
			echo "# Add your API keys below (optional but recommended)" >> $(ENV_FILE); \
			echo "" >> $(ENV_FILE); \
			echo "# Financial Data APIs" >> $(ENV_FILE); \
			echo "TWELVEDATA_API_KEY=" >> $(ENV_FILE); \
			echo "ALPHA_VANTAGE_API_KEY=" >> $(ENV_FILE); \
			echo "CRYPTOCOMPARE_API_KEY=" >> $(ENV_FILE); \
			echo "" >> $(ENV_FILE); \
			echo "# AI APIs (for advanced features)" >> $(ENV_FILE); \
			echo "OPENAI_API_KEY=" >> $(ENV_FILE); \
			echo "ANTHROPIC_API_KEY=" >> $(ENV_FILE); \
			echo "" >> $(ENV_FILE); \
			echo "# Optional: HTTP settings" >> $(ENV_FILE); \
			echo "HTTP_UA=Mozilla/5.0 (compatible; TradingAssistant/1.0)" >> $(ENV_FILE); \
			echo "STOOQ_TIMEOUT=12" >> $(ENV_FILE); \
			echo "$(GREEN)✅ Created basic $(ENV_FILE)$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW)⚠️  $(ENV_FILE) already exists$(NC)"; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# API Configuration
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: setup-api
setup-api: ## Interactive API key setup
	@echo "$(CYAN)🔑 API Key Configuration$(NC)"
	@echo "========================================"
	@echo ""
	@echo "$(BLUE)The Trading Assistant can work with several APIs to provide better data coverage:$(NC)"
	@echo ""
	@echo "$(GREEN)📊 Financial Data APIs (Recommended):$(NC)"
	@echo "  • $(YELLOW)TwelveData$(NC) - Free tier: 800 requests/day"
	@echo "    Sign up: https://twelvedata.com/"
	@echo "  • $(YELLOW)Alpha Vantage$(NC) - Free tier: 25 requests/day" 
	@echo "    Sign up: https://www.alphavantage.co/"
	@echo "  • $(YELLOW)CryptoCompare$(NC) - Free tier: 100,000 requests/month"
	@echo "    Sign up: https://www.cryptocompare.com/"
	@echo ""
	@echo "$(PURPLE)🤖 AI APIs (Optional - for advanced analysis):$(NC)"
	@echo "  • $(YELLOW)OpenAI$(NC) - For GPT-powered analysis"
	@echo "    Sign up: https://platform.openai.com/"
	@echo "  • $(YELLOW)Anthropic$(NC) - For Claude-powered analysis"
	@echo "    Sign up: https://console.anthropic.com/"
	@echo ""
	@echo "$(BLUE)📝 Note: All APIs are optional. The app works without them but with limited data sources.$(NC)"
	@echo ""
	@printf "$(CYAN)Press Enter to start configuration (or Ctrl+C to skip)...$(NC)"
	@read dummy
	@$(MAKE) _interactive_api_setup

.PHONY: _interactive_api_setup
_interactive_api_setup:
	@echo "$(BLUE)Setting up API keys interactively...$(NC)"
	@if [ ! -f "$(ENV_FILE)" ]; then \
		$(MAKE) setup-env; \
	fi
	@echo ""
	@echo "$(GREEN)📊 Financial Data APIs$(NC)"
	@echo "=========================="
	@printf "Enter TwelveData API key (or press Enter to skip): "; \
	read twelvedata_key; \
	if [ ! -z "$twelvedata_key" ]; then \
		if grep -q "TWELVEDATA_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/TWELVEDATA_API_KEY=.*/TWELVEDATA_API_KEY=$twelvedata_key/" $(ENV_FILE); \
		else \
			echo "TWELVEDATA_API_KEY=$twelvedata_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ TwelveData API key saved$(NC)"; \
	fi
	@printf "Enter Alpha Vantage API key (or press Enter to skip): "; \
	read alphavantage_key; \
	if [ ! -z "$alphavantage_key" ]; then \
		if grep -q "ALPHA_VANTAGE_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/ALPHA_VANTAGE_API_KEY=.*/ALPHA_VANTAGE_API_KEY=$alphavantage_key/" $(ENV_FILE); \
		else \
			echo "ALPHA_VANTAGE_API_KEY=$alphavantage_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ Alpha Vantage API key saved$(NC)"; \
	fi
	@printf "Enter CryptoCompare API key (or press Enter to skip): "; \
	read cryptocompare_key; \
	if [ ! -z "$cryptocompare_key" ]; then \
		if grep -q "CRYPTOCOMPARE_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/CRYPTOCOMPARE_API_KEY=.*/CRYPTOCOMPARE_API_KEY=$cryptocompare_key/" $(ENV_FILE); \
		else \
			echo "CRYPTOCOMPARE_API_KEY=$cryptocompare_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ CryptoCompare API key saved$(NC)"; \
	fi
	@echo ""
	@echo "$(PURPLE)🤖 AI APIs (Optional)$(NC)"
	@echo "======================"
	@printf "Enter OpenAI API key (or press Enter to skip): "; \
	read openai_key; \
	if [ ! -z "$openai_key" ]; then \
		if grep -q "OPENAI_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" $(ENV_FILE); \
		else \
			echo "OPENAI_API_KEY=$openai_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ OpenAI API key saved$(NC)"; \
	fi
	@printf "Enter Anthropic API key (or press Enter to skip): "; \
	read anthropic_key; \
	if [ ! -z "$anthropic_key" ]; then \
		if grep -q "ANTHROPIC_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$anthropic_key/" $(ENV_FILE); \
		else \
			echo "ANTHROPIC_API_KEY=$anthropic_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ Anthropic API key saved$(NC)"; \
	fi
	@if [ -f "$(ENV_FILE).bak" ]; then rm $(ENV_FILE).bak; fi
	@echo ""
	@echo "$(GREEN)🎉 API configuration complete!$(NC)"
	@echo "$(BLUE)💡 You can run 'make check-api' to verify your configuration$(NC)"

.PHONY: check-api
check-api: venv ## Check API key configuration status
	@echo "$(CYAN)🔍 Checking API Configuration$(NC)"
	@echo "================================="
	@$(PYTHON_VENV) -c "\
import os;\
from pathlib import Path;\
env_file = Path('.env');\
if env_file.exists():\
    with open(env_file) as f:\
        lines = f.readlines();\
    keys = {};\
    for line in lines:\
        if '=' in line and not line.strip().startswith('#'):\
            key, val = line.strip().split('=', 1);\
            keys[key] = bool(val.strip());\
    print('📊 Financial Data APIs:');\
    for api in ['TWELVEDATA_API_KEY', 'ALPHA_VANTAGE_API_KEY', 'CRYPTOCOMPARE_API_KEY']:\
        status = '✅ Configured' if keys.get(api, False) else '❌ Not configured';\
        print(f'   {api.replace(\"_API_KEY\", \"\")}: {status}');\
    print('');\
    print('🤖 AI APIs:');\
    for api in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:\
        status = '✅ Configured' if keys.get(api, False) else '❌ Not configured';\
        print(f'   {api.replace(\"_API_KEY\", \"\")}: {status}');\
    missing = [k.replace('_API_KEY', '') for k, v in keys.items() if not v and 'API_KEY' in k];\
    if missing:\
        print('');\
        print('🔧 To add missing keys: make setup-api');\
    else:\
        print('');\
        print('🎉 All API keys configured!');\
else:\
    print('❌ .env file not found. Run: make setup-env');\
"

.PHONY: show-api-info
show-api-info: ## Show information about supported APIs
	@echo "$(CYAN)📋 Supported APIs Information$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)📊 Financial Data APIs:$(NC)"
	@echo ""
	@echo "$(YELLOW)TwelveData$(NC) - Real-time and historical financial data"
	@echo "  • Free tier: 800 requests/day"
	@echo "  • Covers: Stocks, Forex, Crypto, Commodities"
	@echo "  • Sign up: https://twelvedata.com/"
	@echo ""
	@echo "$(YELLOW)Alpha Vantage$(NC) - Financial market data"
	@echo "  • Free tier: 25 requests/day (500 requests/day with registration)"
	@echo "  • Covers: Stocks, Forex, Crypto, Technical indicators"
	@echo "  • Sign up: https://www.alphavantage.co/"
	@echo ""
	@echo "$(YELLOW)CryptoCompare$(NC) - Cryptocurrency data"
	@echo "  • Free tier: 100,000 requests/month"
	@echo "  • Covers: All major cryptocurrencies, historical data"
	@echo "  • Sign up: https://www.cryptocompare.com/"
	@echo ""
	@echo "$(PURPLE)🤖 AI APIs (Optional):$(NC)"
	@echo ""
	@echo "$(YELLOW)OpenAI$(NC) - GPT models for analysis"
	@echo "  • Usage-based pricing"
	@echo "  • Used for: Advanced market analysis, strategy suggestions"
	@echo "  • Sign up: https://platform.openai.com/"
	@echo ""
	@echo "$(YELLOW)Anthropic$(NC) - Claude models for analysis"
	@echo "  • Usage-based pricing" 
	@echo "  • Used for: Market analysis, research assistance"
	@echo "  • Sign up: https://console.anthropic.com/"
	@echo ""
	@echo "$(BLUE)💡 Note: The app works without any API keys, using free data sources,$(NC)"
	@echo "$(BLUE)   but having API keys provides better data coverage and reliability.$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Running & Development
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: run
run: venv ## Run the trading assistant
	@echo "$(GREEN)🚀 Running the trading assistant...$(NC)"
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "$(YELLOW)⚠️  No .env file found. Creating one...$(NC)"; \
		$(MAKE) setup-env; \
	fi
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m app_cli.main

.PHONY: run-debug
run-debug: venv ## Run with debug logging
	@echo "$(GREEN)🐛 Running with debug logging...$(NC)"
	@cd $(SRC_DIR) && PYTHONPATH=. DEBUG=1 $(PYTHON_VENV) -m app_cli.main

.PHONY: dev
dev: venv ## Run in development mode with auto-reload
	@echo "$(GREEN)🔄 Running in development mode...$(NC)"
	@echo "$(YELLOW)Note: Auto-reload not implemented yet. Use Ctrl+C and 'make run' to restart.$(NC)"
	@$(MAKE) run

.PHONY: test
test: venv ## Run tests (when implemented)
	@echo "$(BLUE)🧪 Running tests...$(NC)"
	@echo "$(YELLOW)Tests not implemented yet.$(NC)"

.PHONY: debug-paths
debug-paths: venv ## Debug path resolution
	@echo "$(BLUE)🔍 Debugging path resolution...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -c "from trading_core.config import debug_paths; debug_paths()"

# ─────────────────────────────────────────────────────────────────────────────
# Maintenance
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Clean up generated files and caches
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)🧹 Deep cleaning (including venv)...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✅ Deep cleanup complete$(NC)"

.PHONY: format
format: venv ## Format code (when tools are configured)
	@echo "$(BLUE)📝 Code formatting...$(NC)"
	@echo "$(YELLOW)Code formatting tools not configured yet.$(NC)"

.PHONY: lint
lint: venv ## Lint code (when tools are configured)
	@echo "$(BLUE)🔍 Code linting...$(NC)"
	@echo "$(YELLOW)Code linting tools not configured yet.$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Information
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: status
status: ## Show project status
	@echo "$(CYAN)📊 Project Status$(NC)"
	@echo "=================="
	@echo "Project: $(GREEN)$(PROJECT_NAME)$(NC)"
	@echo "Python: $(shell $(PYTHON) --version 2>/dev/null || echo 'Not found')"
	@echo "Virtual env: $(if $(wildcard $(VENV_DIR)),$(GREEN)Exists$(NC),$(RED)Not created$(NC))"
	@echo "Dependencies: $(if $(wildcard $(VENV_DIR)/lib),$(GREEN)Installed$(NC),$(YELLOW)Not installed$(NC))"
	@echo "Environment: $(if $(wildcard $(ENV_FILE)),$(GREEN)Configured$(NC),$(YELLOW)Not configured$(NC))"

.PHONY: info
info: status show-api-info ## Show all project information

# ─────────────────────────────────────────────────────────────────────────────
# Quick start aliases
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: install
install: setup ## Alias for setup

.PHONY: start
start: run ## Alias for run

.PHONY: configure
configure: setup-api ## Alias for setup-api
