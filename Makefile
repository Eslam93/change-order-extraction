# Change order extraction. Run `make help` for the targets.
#
# Every target is one `uv run` command, so it behaves the same under sh and cmd.exe.
# `make extract` prints exactly one JSON object to stdout; everything else goes to stderr.

.DEFAULT_GOAL := help
.PHONY: help setup extract explain test lint typecheck check samples samples-run eval eval-replay

help:
	@echo "make setup                 install dependencies into .venv (uv sync)"
	@echo "make extract FILE=path     extract one change order; JSON on stdout, logs on stderr"
	@echo "make explain FILE=path     the same, and write evidence and findings to out/"
	@echo "make test                  run the test suite (no network, no API key)"
	@echo "make lint                  ruff check and format check"
	@echo "make typecheck             mypy --strict"
	@echo "make check                 lint, typecheck and test"
	@echo "make samples               rebuild the sample PDFs from samples/build_samples.py"
	@echo "make samples-run           run the six samples against the live model (needs LLM_*)"
	@echo "make eval                  run the labelled evaluation set live (needs LLM_*)"
	@echo "make eval-replay           rebuild the evaluation report from recorded replies"

setup:
	uv sync

extract:
ifndef FILE
	$(error usage: make extract FILE=<path to a PDF or text file>)
endif
	@uv run --quiet python -m changeorder "$(FILE)"

explain:
ifndef FILE
	$(error usage: make explain FILE=<path to a PDF or text file>)
endif
	@uv run --quiet python -m changeorder "$(FILE)" --diagnostics out/diagnostics.json

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

check: lint typecheck test

samples:
	uv run python samples/build_samples.py

samples-run:
	uv run python -m changeorder.evaluation run samples

eval:
	uv run python -m changeorder.evaluation run eval

eval-replay:
	uv run python -m changeorder.evaluation replay eval
