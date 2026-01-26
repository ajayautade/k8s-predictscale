# ============================================
# K8s PredictScale - Makefile
# ============================================

.PHONY: help install dev test lint format clean docker-build docker-up docker-down

PYTHON := python3
PIP := pip
APP_NAME := k8s-predictscale
DOCKER_IMAGE := $(APP_NAME):latest

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Development ----

install: ## Install Python dependencies
	$(PIP) install -r requirements.txt

dev: ## Run the application in development mode
	$(PYTHON) -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# ---- Testing ----

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ -v

# ---- Code Quality ----

lint: ## Run linters
	flake8 src/ tests/
	mypy src/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

# ---- Docker ----

docker-build: ## Build Docker image
	docker build -t $(DOCKER_IMAGE) .

docker-up: ## Start local development stack
	docker-compose up -d

docker-down: ## Stop local development stack
	docker-compose down

docker-logs: ## View logs from all containers
	docker-compose logs -f

# ---- Infrastructure ----

tf-init: ## Initialize Terraform
	cd terraform/environments/dev && terraform init

tf-plan: ## Plan Terraform changes
	cd terraform/environments/dev && terraform plan

tf-apply: ## Apply Terraform changes
	cd terraform/environments/dev && terraform apply

tf-destroy: ## Destroy Terraform infrastructure
	cd terraform/environments/dev && terraform destroy

# ---- Cleanup ----

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
