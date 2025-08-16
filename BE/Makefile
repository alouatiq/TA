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
			echo "# Core runtime" >> $(ENV_FILE); \
			echo "PYTHONUNBUFFERED=1" >> $(ENV_FILE); \
			echo "PYTHONIOENCODING=utf-8" >> $(ENV_FILE); \
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
			echo "# Logging / diagnostics" >> $(ENV_FILE); \
			echo "LOG_LEVEL=INFO" >> $(ENV_FILE); \
			echo "" >> $(ENV_FILE); \
			echo "# Regional defaults (optional)" >> $(ENV_FILE); \
			echo "DEFAULT_REGION=Europe" >> $(ENV_FILE); \
			echo "DEFAULT_MARKET=LSE" >> $(ENV_FILE); \
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
	@current_key=$$(grep "^TWELVEDATA_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	printf "Enter TwelveData API key"; \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "welvedata_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
		printf " (currently: ✅ configured, press Enter to keep): "; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		printf " (currently: ⚠️  placeholder, enter real key): "; \
	else \
		printf " (currently: ❌ not set, or press Enter to skip): "; \
	fi; \
	read twelvedata_key; \
	if [ ! -z "$$twelvedata_key" ]; then \
		if grep -q "TWELVEDATA_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/TWELVEDATA_API_KEY=.*/TWELVEDATA_API_KEY=$$twelvedata_key/" $(ENV_FILE); \
		else \
			echo "TWELVEDATA_API_KEY=$$twelvedata_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ TwelveData API key saved$(NC)"; \
	else \
		if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "welvedata_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
			echo "$(BLUE)🔄 TwelveData API key kept (existing)$(NC)"; \
		else \
			echo "$(YELLOW)⏭️  TwelveData API key skipped$(NC)"; \
		fi; \
	fi
	@current_key=$$(grep "^ALPHA_VANTAGE_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	printf "Enter Alpha Vantage API key"; \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "lphavantage_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
		printf " (currently: ✅ configured, press Enter to keep): "; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		printf " (currently: ⚠️  placeholder, enter real key): "; \
	else \
		printf " (currently: ❌ not set, or press Enter to skip): "; \
	fi; \
	read alphavantage_key; \
	if [ ! -z "$$alphavantage_key" ]; then \
		if grep -q "ALPHA_VANTAGE_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/ALPHA_VANTAGE_API_KEY=.*/ALPHA_VANTAGE_API_KEY=$$alphavantage_key/" $(ENV_FILE); \
		else \
			echo "ALPHA_VANTAGE_API_KEY=$$alphavantage_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ Alpha Vantage API key saved$(NC)"; \
	else \
		if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "lphavantage_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
			echo "$(BLUE)🔄 Alpha Vantage API key kept (existing)$(NC)"; \
		else \
			echo "$(YELLOW)⏭️  Alpha Vantage API key skipped$(NC)"; \
		fi; \
	fi
	@current_key=$$(grep "^CRYPTOCOMPARE_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	printf "Enter CryptoCompare API key"; \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "ryptocompare_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
		printf " (currently: ✅ configured, press Enter to keep): "; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		printf " (currently: ⚠️  placeholder, enter real key): "; \
	else \
		printf " (currently: ❌ not set, or press Enter to skip): "; \
	fi; \
	read cryptocompare_key; \
	if [ ! -z "$$cryptocompare_key" ]; then \
		if grep -q "CRYPTOCOMPARE_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/CRYPTOCOMPARE_API_KEY=.*/CRYPTOCOMPARE_API_KEY=$$cryptocompare_key/" $(ENV_FILE); \
		else \
			echo "CRYPTOCOMPARE_API_KEY=$$cryptocompare_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ CryptoCompare API key saved$(NC)"; \
	else \
		if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "ryptocompare_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
			echo "$(BLUE)🔄 CryptoCompare API key kept (existing)$(NC)"; \
		else \
			echo "$(YELLOW)⏭️  CryptoCompare API key skipped$(NC)"; \
		fi; \
	fi
	@echo ""
	@echo "$(PURPLE)🤖 AI APIs (Optional)$(NC)"
	@echo "======================"
	@current_key=$$(grep "^OPENAI_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	printf "Enter OpenAI API key"; \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "penai_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
		printf " (currently: ✅ configured, press Enter to keep): "; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		printf " (currently: ⚠️  placeholder, enter real key): "; \
	else \
		printf " (currently: ❌ not set, or press Enter to skip): "; \
	fi; \
	read openai_key; \
	if [ ! -z "$$openai_key" ]; then \
		if grep -q "OPENAI_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$$openai_key/" $(ENV_FILE); \
		else \
			echo "OPENAI_API_KEY=$$openai_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ OpenAI API key saved$(NC)"; \
	else \
		if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "penai_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
			echo "$(BLUE)🔄 OpenAI API key kept (existing)$(NC)"; \
		else \
			echo "$(YELLOW)⏭️  OpenAI API key skipped$(NC)"; \
		fi; \
	fi
	@current_key=$$(grep "^ANTHROPIC_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	printf "Enter Anthropic API key"; \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "nthropic_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
		printf " (currently: ✅ configured, press Enter to keep): "; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		printf " (currently: ⚠️  placeholder, enter real key): "; \
	else \
		printf " (currently: ❌ not set, or press Enter to skip): "; \
	fi; \
	read anthropic_key; \
	if [ ! -z "$$anthropic_key" ]; then \
		if grep -q "ANTHROPIC_API_KEY=" $(ENV_FILE); then \
			sed -i.bak "s/ANTHROPIC_API_KEY=.*/ANTHROPIC_API_KEY=$$anthropic_key/" $(ENV_FILE); \
		else \
			echo "ANTHROPIC_API_KEY=$$anthropic_key" >> $(ENV_FILE); \
		fi; \
		echo "$(GREEN)✅ Anthropic API key saved$(NC)"; \
	else \
		if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "nthropic_key" ] && [ "$$current_key" != "YOUR_API_KEY" ]; then \
			echo "$(BLUE)🔄 Anthropic API key kept (existing)$(NC)"; \
		else \
			echo "$(YELLOW)⏭️  Anthropic API key skipped$(NC)"; \
		fi; \
	fi
	@if [ -f "$(ENV_FILE).bak" ]; then rm $(ENV_FILE).bak; fi
	@echo ""
	@echo "$(GREEN)🎉 API configuration complete!$(NC)"
	@echo "$(BLUE)💡 You can run 'make check-api' to verify your configuration$(NC)"

.PHONY: check-api
check-api: venv ## Check API key configuration status
	@echo "$(CYAN)🔍 Checking API Configuration$(NC)"
	@echo "================================="
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "❌ .env file not found. Run: make setup-env"; \
		exit 1; \
	fi
	@echo "📊 Financial Data APIs:"
	@current_key=$$(grep "^TWELVEDATA_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "welvedata_key" ] && [ "$$current_key" != "YOUR_API_KEY" ] && [ "$${#current_key}" -gt 10 ]; then \
		echo "   TwelveData: ✅ Configured"; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		echo "   TwelveData: ⚠️  Placeholder detected"; \
	else \
		echo "   TwelveData: ❌ Not configured"; \
	fi
	@current_key=$$(grep "^ALPHA_VANTAGE_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "lphavantage_key" ] && [ "$$current_key" != "YOUR_API_KEY" ] && [ "$${#current_key}" -gt 10 ]; then \
		echo "   Alpha_Vantage: ✅ Configured"; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		echo "   Alpha_Vantage: ⚠️  Placeholder detected"; \
	else \
		echo "   Alpha_Vantage: ❌ Not configured"; \
	fi
	@current_key=$$(grep "^CRYPTOCOMPARE_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "ryptocompare_key" ] && [ "$$current_key" != "YOUR_API_KEY" ] && [ "$${#current_key}" -gt 10 ]; then \
		echo "   CryptoCompare: ✅ Configured"; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		echo "   CryptoCompare: ⚠️  Placeholder detected"; \
	else \
		echo "   CryptoCompare: ❌ Not configured"; \
	fi
	@echo ""
	@echo "🤖 AI APIs:"
	@current_key=$$(grep "^OPENAI_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "penai_key" ] && [ "$$current_key" != "YOUR_API_KEY" ] && [ "$${#current_key}" -gt 10 ]; then \
		echo "   OpenAI: ✅ Configured"; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		echo "   OpenAI: ⚠️  Placeholder detected"; \
	else \
		echo "   OpenAI: ❌ Not configured"; \
	fi
	@current_key=$$(grep "^ANTHROPIC_API_KEY=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$$//'); \
	if [ -n "$$current_key" ] && [ "$$current_key" != "" ] && [ "$$current_key" != "nthropic_key" ] && [ "$$current_key" != "YOUR_API_KEY" ] && [ "$${#current_key}" -gt 10 ]; then \
		echo "   Anthropic: ✅ Configured"; \
	elif [ -n "$$current_key" ] && [ "$$current_key" != "" ]; then \
		echo "   Anthropic: ⚠️  Placeholder detected"; \
	else \
		echo "   Anthropic: ❌ Not configured"; \
	fi
	@echo ""
	@echo "💡 To configure missing APIs: make setup-api"
	@echo "⚠️  Replace placeholder values with real API keys"

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
	@echo "$(YELLOW)Note: Auto-reload not implemented yet. Use run-debug for debugging.$(NC)"
	@$(MAKE) run-debug

.PHONY: test
test: venv ## Run tests
	@echo "$(GREEN)🧪 Running tests...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m pytest tests/ -v

.PHONY: test-coverage
test-coverage: venv ## Run tests with coverage report
	@echo "$(GREEN)📊 Running tests with coverage...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality & Maintenance
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: format
format: venv ## Format code with black and isort
	@echo "$(BLUE)🎨 Formatting code...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m black .
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m isort .
	@echo "$(GREEN)✅ Code formatted$(NC)"

.PHONY: lint
lint: venv ## Run linting with flake8
	@echo "$(BLUE)🔍 Running linter...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m flake8 .

.PHONY: type-check
type-check: venv ## Run type checking with mypy
	@echo "$(BLUE)🏷️  Running type checker...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON_VENV) -m mypy .

.PHONY: clean
clean: ## Clean up temporary files and caches
	@echo "$(BLUE)🧹 Cleaning up...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name ".coverage" -delete
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

.PHONY: clean-all
clean-all: clean ## Clean everything including virtual environment
	@echo "$(BLUE)🗑️  Removing virtual environment...$(NC)"
	@rm -rf $(VENV_DIR)
	@echo "$(GREEN)✅ Complete cleanup done$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Information & Status
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: status
status: ## Show project status and configuration
	@echo "$(CYAN)📋 Project Status$(NC)"
	@echo "=================="
	@echo ""
	@echo "$(BLUE)Environment:$(NC)"
	@echo "  Python: $($(PYTHON) --version 2>&1)"
	@echo "  Virtual Environment: $(if [ -d "$(VENV_DIR)" ]; then echo "✅ Present"; else echo "❌ Missing"; fi)"
	@echo "  Requirements: $(if [ -f "$(REQUIREMENTS)" ]; then echo "✅ Found"; else echo "❌ Missing"; fi)"
	@echo "  Environment File: $(if [ -f "$(ENV_FILE)" ]; then echo "✅ Found"; else echo "❌ Missing"; fi)"
	@echo ""
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "$(BLUE)API Configuration:$(NC)"; \
		configured_count=0; \
		total_count=5; \
		for api_key in "TWELVEDATA_API_KEY" "ALPHA_VANTAGE_API_KEY" "CRYPTOCOMPARE_API_KEY" "OPENAI_API_KEY" "ANTHROPIC_API_KEY"; do \
			current_key=$(grep "^$api_key=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//'); \
			api_display=$(echo $api_key | sed 's/_API_KEY//' | sed 's/_/ /g'); \
			if [ -n "$current_key" ] && [ "$current_key" != "" ] && [ "${#current_key}" -gt 10 ]; then \
				echo "  $api_display: ✅"; \
				configured_count=$((configured_count + 1)); \
			else \
				echo "  $api_display: ❌"; \
			fi; \
		done; \
		echo "  Total: $configured_count/$total_count APIs configured"; \
	fi
	@echo ""
	@echo "$(BLUE)Quick Commands:$(NC)"
	@echo "  Setup: make setup"
	@echo "  Configure APIs: make setup-api"
	@echo "  Run: make run"
	@echo "  Help: make help"

.PHONY: version
version: ## Show version information
	@echo "$(CYAN)Trading Assistant Backend$(NC)"
	@echo "=========================="
	@echo "Version: 1.0.0-beta"
	@echo "Python: $($(PYTHON) --version 2>&1)"
	@echo "Make: $(make --version | head -n1)"
	@echo ""
	@echo "$(BLUE)For more information, visit: https://github.com/your-repo/trading-assistant$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Documentation
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: docs
docs: venv ## Generate documentation
	@echo "$(BLUE)📚 Generating documentation...$(NC)"
	@echo "$(YELLOW)⚠️  Documentation generation not implemented yet$(NC)"
	@echo "$(BLUE)💡 For now, see README.md and inline code documentation$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Docker Support (if needed in the future)
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build Docker image
	@echo "$(BLUE)🐳 Building Docker image...$(NC)"
	@echo "$(YELLOW)⚠️  Docker support not implemented yet$(NC)"

.PHONY: docker-run
docker-run: ## Run in Docker container
	@echo "$(BLUE)🐳 Running in Docker...$(NC)"
	@echo "$(YELLOW)⚠️  Docker support not implemented yet$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Backup & Export
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: backup-config
backup-config: ## Backup configuration files
	@echo "$(BLUE)💾 Backing up configuration...$(NC)"
	@timestamp=$(date +%Y%m%d_%H%M%S); \
	backup_dir="backup_$timestamp"; \
	mkdir -p "$backup_dir"; \
	if [ -f "$(ENV_FILE)" ]; then cp $(ENV_FILE) "$backup_dir/"; fi; \
	if [ -f "$(REQUIREMENTS)" ]; then cp $(REQUIREMENTS) "$backup_dir/"; fi; \
	echo "$(GREEN)✅ Configuration backed up to $backup_dir$(NC)"

# ─────────────────────────────────────────────────────────────────────────────
# Utility Commands
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: clean-env
clean-env: ## Clean .env file of placeholder values
	@echo "$(BLUE)🧹 Cleaning placeholder values from .env file...$(NC)"
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "Backing up current .env to .env.backup..."; \
		cp $(ENV_FILE) $(ENV_FILE).backup; \
		echo "Cleaning placeholder values..."; \
		sed -i 's/TWELVEDATA_API_KEY=welvedata_key/TWELVEDATA_API_KEY=/g' $(ENV_FILE); \
		sed -i 's/ALPHA_VANTAGE_API_KEY=lphavantage_key/ALPHA_VANTAGE_API_KEY=/g' $(ENV_FILE); \
		sed -i 's/CRYPTOCOMPARE_API_KEY=ryptocompare_key/CRYPTOCOMPARE_API_KEY=/g' $(ENV_FILE); \
		sed -i 's/OPENAI_API_KEY=penai_key/OPENAI_API_KEY=/g' $(ENV_FILE); \
		sed -i 's/ANTHROPIC_API_KEY=nthropic_key/ANTHROPIC_API_KEY=/g' $(ENV_FILE); \
		sed -i 's/=YOUR_API_KEY/=/g' $(ENV_FILE); \
		sed -i 's/=your_key_here/=/g' $(ENV_FILE); \
		echo "$(GREEN)✅ Placeholder values cleaned$(NC)"; \
		echo "$(BLUE)💡 Backup saved as .env.backup$(NC)"; \
	else \
		echo "$(RED)❌ .env file not found$(NC)"; \
	fi

.PHONY: debug-env
debug-env: ## Debug .env file contents
	@echo "$(CYAN)🔍 Debugging .env Configuration$(NC)"
	@echo "=================================="
	@if [ -f "$(ENV_FILE)" ]; then \
		echo "📁 .env file found. Contents:"; \
		echo ""; \
		cat $(ENV_FILE); \
		echo ""; \
		echo "🔍 API Key Analysis:"; \
		for api_key in "TWELVEDATA_API_KEY" "ALPHA_VANTAGE_API_KEY" "CRYPTOCOMPARE_API_KEY" "OPENAI_API_KEY" "ANTHROPIC_API_KEY"; do \
			current_key=$(grep "^$api_key=" $(ENV_FILE) 2>/dev/null | cut -d'=' -f2- | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//'); \
			api_display=$(echo $api_key | sed 's/_API_KEY//'); \
			if [ -z "$current_key" ]; then \
				echo "  $api_display: ❌ Not found in file"; \
			elif [ "$current_key" = "" ]; then \
				echo "  $api_display: ⚪ Empty value"; \
			else \
				echo "  $api_display: 📝 Value present ($&#123;current_key&#125;) - ${#current_key} chars"; \
			fi; \
		done; \
	else \
		echo "❌ .env file not found"; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Help Functions
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: list-targets
list-targets: ## List all available make targets
	@echo "$(CYAN)Available Make Targets:$(NC)"
	@$(MAKE) -pRrq -f $(MAKEFILE_LIST) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($1 !~ "^[#.]") {print $1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$'
