# Shingan — developer task runner.
#
# GNU Make is not installed by default on Windows. If `make` is unavailable,
# every target below has a PowerShell equivalent in scripts/ (see scripts/README.md).
# macOS ships GNU Make 3.81, so this file sticks to portable constructs.

.DEFAULT_GOAL := help
PY ?= python
VENV ?= .venv
CONFIG ?= configs/default.yaml
# `poc_tech10` was renamed to `poc_largecap10`; the old name survived here and made every
# data target fail with "no such file". Keep this in step with DEFAULT_CONFIG_FILES in
# src/shingan/config.py, which is what the CLI falls back to.
DATA_CONFIG ?= configs/data/poc_largecap10.yaml
EVAL_CONFIG ?= configs/eval/default.yaml
TRAIN_CONFIG ?= configs/train/qlora_qwen3_14b.yaml
OUT ?= artifacts/demo

.PHONY: help bootstrap check install dev train-extras lint format typecheck test test-fast \
        coverage demo synth build sft train-structured train-lora eval report doctor \
        clean clean-all build-dist precommit

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- environment ------------------------------------------------------------

bootstrap: ## Create the virtualenv and install core + dev dependencies
	$(PY) -m venv $(VENV)
	$(VENV)/bin/$(PY) -m pip install --upgrade pip
	$(VENV)/bin/$(PY) -m pip install -e ".[dev]"
	@echo "Done. Activate with: source $(VENV)/bin/activate  (Windows: $(VENV)\\Scripts\\Activate.ps1)"

install: ## Install the package in editable mode (core deps only)
	$(PY) -m pip install -e .

dev: ## Install editable with the dev extra
	$(PY) -m pip install -e ".[dev]"

train-extras: ## Install the GPU training extra (needs torch cu128 to be present first)
	$(PY) -m pip install -e ".[train,parquet,plots]"

# --- quality ----------------------------------------------------------------

lint: ## Lint with ruff
	$(PY) -m ruff check .

format: ## Auto-fix and format with ruff
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

typecheck: ## Static type check with mypy
	$(PY) -m mypy src

test: ## Run the full test suite
	$(PY) -m pytest

test-fast: ## Run the test suite, skipping the slower end-to-end tests
	$(PY) -m pytest -m "not slow" -n auto

coverage: ## Run tests with a coverage report
	$(PY) -m pytest --cov=shingan --cov-report=term-missing --cov-report=xml

check: lint typecheck test ## Everything CI runs

precommit: ## Run all pre-commit hooks against the whole tree
	pre-commit run --all-files

doctor: ## Print an environment / GPU / Windows compatibility report
	$(PY) -m shingan doctor

# --- pipeline ---------------------------------------------------------------

synth: ## Generate the deterministic synthetic dataset (offline, no network)
	$(PY) -m shingan data synth --config $(CONFIG) --data-config $(DATA_CONFIG)

build: ## Assemble the processed panel dataset (as-of joins + features + labels)
	$(PY) -m shingan data build --config $(CONFIG) --data-config $(DATA_CONFIG)

sft: ## Emit the instruction-format JSONL used to train the text track
	$(PY) -m shingan data sft --config $(CONFIG) --data-config $(DATA_CONFIG)

train-structured: ## Fit the calibrated structured (GBDT) track on CPU
	$(PY) -m shingan train structured --config $(CONFIG) --data-config $(DATA_CONFIG)

train-lora: ## QLoRA fine-tune the text track (requires GPU and the train extra)
	$(PY) -m shingan train lora --config $(CONFIG) --train-config $(TRAIN_CONFIG)

eval: ## Evaluate a saved run and write the markdown report
	$(PY) -m shingan eval run --config $(CONFIG) --eval-config $(EVAL_CONFIG) --run-dir $(OUT)

demo: ## CPU-only end-to-end run on synthetic data; writes a full report
	$(PY) -m shingan demo --out $(OUT)

report: ## Re-render the report from a previous run directory
	$(PY) -m shingan eval report --run-dir $(OUT)

# --- packaging --------------------------------------------------------------

build-dist: ## Build sdist and wheel
	$(PY) -m build

# --- housekeeping -----------------------------------------------------------

clean: ## Remove caches, coverage data and build artefacts (keeps data/ and artifacts/)
	@rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type f -name '*.py[co]' -delete
	@rm -rf build dist *.egg-info src/*.egg-info

clean-all: clean ## Also delete data payloads and generated model artefacts
	@rm -rf artifacts data/raw/* data/interim/* data/processed/* data/external/*
	@find data -type d -empty -exec touch {}/.gitkeep \;
	@echo "Removed generated data and artifacts."
