SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV ?= uv
PYTHON_VERSION ?= 3.12
IMAGE ?= client-research-agent
TAG ?= local
RUN := $(UV) run --no-sync
PYTEST := $(RUN) pytest

.PHONY: help install lock lint format typecheck test test-unit test-integration test-e2e \
        test-security test-perf coverage security build docker-build bundle-validate \
        deploy-dev deploy-staging clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Create .venv with pinned dev dependencies and the package in editable mode
	$(UV) venv --python $(PYTHON_VERSION)
	$(UV) pip install -r requirements-dev.txt
	$(UV) pip install --no-deps -e .

lock: ## Recompile requirements.txt and requirements-dev.txt from pyproject.toml
	$(UV) pip compile pyproject.toml --universal --python-version 3.11 -o requirements.txt \
		--custom-compile-command "uv pip compile pyproject.toml --universal --python-version 3.11 -o requirements.txt"
	$(UV) pip compile pyproject.toml --universal --python-version 3.11 --extra dev --extra databricks --extra otlp \
		-o requirements-dev.txt \
		--custom-compile-command "uv pip compile pyproject.toml --universal --python-version 3.11 --extra dev --extra databricks --extra otlp -o requirements-dev.txt"

lint: ## ruff lint + format check, shellcheck, terraform fmt
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	@if command -v shellcheck >/dev/null; then shellcheck deployment/workflows/*.sh; else echo "shellcheck not installed; skipping"; fi
	@if command -v terraform >/dev/null; then terraform -chdir=deployment/terraform fmt -check -recursive; else echo "terraform not installed; skipping"; fi

format: ## Apply ruff fixes and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .
	@if command -v terraform >/dev/null; then terraform -chdir=deployment/terraform fmt -recursive; fi

typecheck: ## mypy --strict (configured in pyproject.toml)
	$(RUN) mypy

test: ## Full test suite
	$(PYTEST)

test-unit: ## Unit tests
	$(PYTEST) tests/unit

test-integration: ## Integration and contract tests
	$(PYTEST) -m "integration or contract"

test-e2e: ## End-to-end tests against local adapters
	$(PYTEST) -m e2e

test-security: ## Adversarial / policy tests
	$(PYTEST) -m security

test-perf: ## Latency and throughput budgets
	$(PYTEST) -m performance --benchmark-only

coverage: ## Unit tests with the 90% coverage gate and HTML report
	$(PYTEST) tests/unit --cov --cov-branch --cov-report=term-missing --cov-report=xml --cov-report=html --cov-fail-under=90

security: ## bandit + pip-audit
	$(RUN) bandit -c pyproject.toml -r src
	$(RUN) pip-audit --strict --disable-pip --no-deps -r requirements-dev.txt

build: ## Build wheel and sdist into dist/
	rm -rf dist
	$(UV) build --out-dir dist

docker-build: ## Build the container image
	docker build --build-arg VCS_REF=$$(git rev-parse --short HEAD 2>/dev/null || echo unknown) -t $(IMAGE):$(TAG) .

bundle-validate: ## Validate the Databricks Asset Bundle for every target
	databricks bundle validate --target dev
	databricks bundle validate --target staging
	databricks bundle validate --target prod

deploy-dev: ## Deploy the bundle to the dev target
	deployment/workflows/deploy.sh dev

deploy-staging: ## Deploy the bundle to staging and run the evaluation gate
	deployment/workflows/deploy.sh staging --evaluate

clean: ## Remove build, test and cache artefacts
	rm -rf dist build htmlcov .coverage coverage.xml junit-*.xml .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
