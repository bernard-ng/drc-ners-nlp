.PHONY: default
default: help

.PHONY: help
help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

.PHONY: setup
setup: ## Setup virtual environment and install dependencies
	python -m venv .venv
	source .venv/bin/activate
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

.PHONY: install
install: ## Install/update dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

# =============================================================================
# DEVELOPMENT & CODE QUALITY
# =============================================================================

.PHONY: format
format: ## Format code with black
	black . --line-length 100

.PHONY: lint
lint: ## Lint code with flake8
	flake8 . --max-line-length=100 --ignore=E203,W503 --exclude=.venv

.PHONY: type-check
type-check: ## Type check with mypy
	mypy . --ignore-missing-imports

.PHONY: notebook
notebook: ## Start Jupyter notebook
	jupyter notebook notebooks/

# =============================================================================
# DEPLOYMENT & PRODUCTION
# =============================================================================

.PHONY: backup
backup: ## Backup datasets and results
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/data/
	@echo "Backup created in backups/$(shell date +%Y%m%d_%H%M%S)/"
