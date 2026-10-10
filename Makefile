.DEFAULT_GOAL := help
PYTHON ?= python3
VENV   ?= .venv-serve
BIN    := $(VENV)/bin

# Docker settings. Azure runs linux/amd64, so images and training use it everywhere,
# also on Apple Silicon Macs, to get the same environment as production.
IMAGE    ?= rul-api
TAG      ?= local
PLATFORM ?= linux/amd64

.PHONY: help install lint format test test-all train evaluate serve docker-build docker-run docker-train clean

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

test:  ## Run the fast tests (synthetic data, seconds)
	$(BIN)/pytest -m "not slow" --cov --cov-report=term-missing

test-all:  ## Run every test, including the real-data checks (needs data/)
	$(BIN)/pytest --cov --cov-report=term-missing

train:  ## Train the model and write artifacts to models/  (Phase 1)
	$(BIN)/python -m rul.train

evaluate:  ## Score the trained model on the 100 test engines (fails if MAE > 9.5)
	$(BIN)/python -m rul.evaluate --max-mae 9.5

serve:  ## Run the API locally on http://127.0.0.1:8000 (docs at /docs)
	$(BIN)/uvicorn rul.api.main:app --reload --port 8000

docker-build:  ## Build the API image (the model in models/ is copied in)
	docker build --platform $(PLATFORM) -t $(IMAGE):$(TAG) .

docker-run:  ## Run the API container on http://127.0.0.1:8000 (Ctrl+C to stop)
	docker run --rm --platform $(PLATFORM) -p 8000:8000 --name rul-api $(IMAGE):$(TAG)

docker-train:  ## Train and evaluate inside a Linux container, writing models/
	docker build --platform $(PLATFORM) --target build -t $(IMAGE)-train:$(TAG) .
	docker run --rm --platform $(PLATFORM) --user $(shell id -u):$(shell id -g) \
		-v "$(CURDIR)/data:/data:ro" -v "$(CURDIR)/configs:/configs:ro" -v "$(CURDIR)/models:/models" \
		$(IMAGE)-train:$(TAG) python -m rul.train \
		--config /configs/train.yaml --train-file /data/train_FD001.txt --output /models
	docker run --rm --platform $(PLATFORM) --user $(shell id -u):$(shell id -g) \
		-v "$(CURDIR)/data:/data:ro" -v "$(CURDIR)/models:/models" \
		$(IMAGE)-train:$(TAG) python -m rul.evaluate --model-dir /models \
		--test-file /data/test_FD001.txt --rul-file /data/RUL_FD001.txt --max-mae 9.5

clean:  ## Remove caches
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
