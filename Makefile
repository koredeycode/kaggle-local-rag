# kaggle-local-rag — Makefile
#
# Two test targets, because this box cannot install pytest (§3: no network, no
# pip) while Kaggle can. `make test` is the local gate; `make test-kaggle` is
# the §22 gate that pytest satisfies there. Both run the same test classes.

PY      ?= python3
SRC     := src
export PYTHONPATH := $(SRC)

.DEFAULT_GOAL := help
.PHONY: help test test-kaggle lint fmt typecheck check measure measure-bm25 \
        measure-cache measure-precision preflight clean-eval

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

test: ## Run the full suite with stdlib unittest (works with no pip installed)
	$(PY) -m unittest discover -s tests -t . -v

test-kaggle: ## Run the suite with pytest (needs pip; the §22 gate)
	$(PY) -m pytest -q

lint: ## ruff check + format check
	ruff check .
	ruff format --check .

fmt: ## ruff format in place
	ruff format .

typecheck: ## mypy --strict
	mypy

check: test lint typecheck ## Everything

# --- Phase 0 measurements (§20) -------------------------------------------------
#
# measure-bm25 and measure-cache run anywhere: no GPU, no models, no network.
# measure-precision needs bge-m3 and realistically a GPU (Kaggle, §3).

measure: measure-bm25 measure-cache ## Run the Phase 0 measurements that fit on a laptop

measure-bm25: ## §6.7 BM25 build/query cost on a synthetic chunk set
	$(PY) scripts/measure_bm25.py --json eval/runs/measure_bm25.json

measure-cache: ## §20 embedding-cache hit rate across chunker settings
	$(PY) scripts/measure_cache_hits.py --json eval/runs/measure_cache_hits.json

measure-precision: ## §5.5 fp16 vs fp32 embedding agreement (Kaggle only)
	$(PY) scripts/measure_precision.py --json eval/runs/measure_precision.json

preflight: ## §19.1 environment, device and model report
	$(PY) scripts/preflight.py --json eval/runs/preflight.json

clean-eval: ## Remove measurement output (never the questions)
	rm -rf eval/runs/*
