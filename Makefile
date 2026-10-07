.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv-serve
BIN    := $(VENV)/bin

.PHONY: help install lint format test train evaluate serve docker clean

help:  ## List available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-10s %s\n", $$1, $$2}'

install:  ## Create the serving venv and install pinned dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

lint:  ## Check style and common bugs
	$(BIN)/ruff check src tests

format:  ## Auto-format and fix lint issues
	$(BIN)/ruff format src tests
	$(BIN)/ruff check --fix src tests

test:  ## Run the test suite with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

train:  ## Train the model and write artifacts to models/  (Phase 1)
	$(BIN)/python -m rul.train

evaluate:  ## Score the trained model on the 100 test engines  (Phase 1)
	$(BIN)/python -m rul.evaluate

serve:  ## Run the API locally on port 8000  (Phase 3)
	$(BIN)/uvicorn rul.api.main:app --reload --port 8000

docker:  ## Build the Docker image  (Phase 4)
	docker build -t rul-api:local .

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
