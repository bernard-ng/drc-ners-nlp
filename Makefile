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
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

.PHONY: install
install: ## Install/update dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: install-dev
install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install jupyter notebook ipykernel pytest black flake8 mypy

.PHONY: activate
activate: ## Show activation command
	@echo "Run: source .venv/bin/activate"

# =============================================================================
# MODEL TRAINING & ARTIFACTS
# =============================================================================

.PHONY: train-baseline
train-baseline: ## Train all baseline models and save artifacts
	python research/train.py --mode baseline

.PHONY: train-neural
train-neural: ## Train neural network models (LSTM, CNN, Transformer)
	python research/train.py --mode neural

.PHONY: train-model
train-model: ## Train specific model (use: make train-model MODEL=logistic_regression NAME=my_model)
	python research/train.py --model-type $(MODEL) --name $(NAME)

.PHONY: list-models
list-models: ## List all saved model artifacts
	python research/train.py --mode list

# =============================================================================
# RESEARCH & EXPERIMENTS
# =============================================================================

.PHONY: experiment
experiment: ## Create sample experiment configuration
	python research/cli.py run --name "sample_experiment" --features full_name --model-type logistic_regression

.PHONY: baseline
baseline: ## Run baseline experiments
	python research/cli.py baseline

.PHONY: ablation
ablation: ## Run feature ablation study
	python research/cli.py ablation

.PHONY: components
components: ## Run name component analysis
	python research/cli.py components

.PHONY: list-experiments
list-experiments: ## List all experiments
	python research/cli.py list

.PHONY: list-completed
list-completed: ## List completed experiments only
	python research/cli.py list --status completed

.PHONY: export-results
export-results: ## Export all experiment results to CSV
	python research/cli.py export --output results_$(shell date +%Y%m%d_%H%M%S).csv

.PHONY: best-model
best-model: ## Show best performing model
	python research/cli.py list --status completed | head -5

# =============================================================================
# WEB INTERFACE
# =============================================================================

.PHONY: web
web: ## Launch Streamlit web interface
	streamlit run web/app.py --server.runOnSave true --server.port 8501

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

.PHONY: lab
lab: ## Start Jupyter lab
	jupyter lab notebooks/

# =============================================================================
# DEPLOYMENT & PRODUCTION
# =============================================================================

.PHONY: backup
backup: ## Backup datasets and results
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@cp -r data/ backups/$(shell date +%Y%m%d_%H%M%S)/data/
	@echo "Backup created in backups/$(shell date +%Y%m%d_%H%M%S)/"
