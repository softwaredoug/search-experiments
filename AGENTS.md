# AGENTS.md

This file guides agentic coding tools working in this repository.
Keep changes small, testable, and aligned with existing patterns.

## Project Overview

- Project name: exps
- Purpose: lexical search benchmarks (BM25 + agentic search).
- Python: >= 3.11
- Package manager: uv
- Entrypoints:
  - `run` (benchmarks)
  - `query` (single query debug)

## Setup / Install

- Install runtime deps:
  - `uv sync`
- Install dev deps (flake8/pytest/ipython):
  - `uv sync --extra dev`

## Run Commands

- Run BM25 benchmark:
  - `uv run run --strategy configs/bm25_strong_title.yml --dataset wands`
- Run an agentic benchmark:
  - `uv run run --strategy configs/agentic_bm25_embeddings.yml --dataset wands`
- Run a single query:
  - `uv run query --strategy configs/bm25_strong_title.yml --query "salon chair"`
  - `uv run query --strategy configs/agentic_bm25_embeddings.yml --query "salon chair" --k 10`

## Lint / Format

- Lint with flake8:
  - `uv run flake8 .`
- Line length: 120 (see `.flake8`).
- No automatic formatter configured. Keep formatting consistent with existing files.

## Tests

- Run all tests:
  - `uv run pytest`
- Run a single test:
  - `uv run pytest tests/test_bm25.py::test_bm25_wands_ndcg_sanity`

## Test-Driven Flow

When a problem is reported, create a runner test first.

- Always use small data (few queries, WANDS dataset, 1 training round).
- Always run end-to-end; avoid mocking.
- Assume `OPENAI_API_KEY` is set; fail loudly if missing.
- If the issue cannot be reproduced under these constraints, prompt the user.
- Tests may take up to 5 minutes to run; that is acceptable.

## Code Style Guidelines

### Imports

- Use grouped imports with blank lines:
  1) standard library
  2) third-party
  3) local package
- Prefer explicit imports over wildcard imports.

### Formatting

- Keep line length <= 120.
- Use black-style hanging indents for multi-line function calls, but do not
  reformat entire files if only small changes are needed.

### Naming

- Modules: lowercase with underscores (e.g., `bm25.py`, `exps.py`).
- Classes: CapWords (e.g., `BM25Strategy`).
- Functions: snake_case (e.g., `run_strategy`).
- Constants: UPPER_SNAKE_CASE when appropriate.

### Types

- Use type hints when they improve clarity, especially in public entrypoints.
- Prefer simple `list`, `dict`, `tuple` annotations unless a specific type
  (e.g., `pd.Series`) helps readability.

### Error Handling

- Raise clear exceptions for invalid inputs (e.g., invalid strategy choices).
- Avoid silent failures; log or print in CLI runners when useful.

### Logging / Output

- CLI runners may print progress or results (e.g., NDCG summaries).
- Avoid excessive printing in library code unless required for debugging.

## Strategy Implementation Notes

- Strategies inherit from `cheat_at_search.strategy.SearchStrategy`.
- `search()` must return `(top_k_indices, scores)`.
- Indexing uses `SearchArray.index` with `snowball_tokenizer`.

## Data Notes

- WANDS data is loaded from `cheat_at_search.wands_data`.
- First run may clone datasets under `cheat_at_search` data path.
- Expect indexing cost on first use; subsequent runs reuse cached indices.

## Repository Rules

- No Cursor rules found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions found in `.github/copilot-instructions.md`.

## Contribution Practices

- Keep changes scoped to the request.
- Avoid modifying unrelated files.
- Update tests if behavior changes.
- Prefer `apply_patch` for small edits and new files.
- When creating git commits, include a `Co-authored-by:` trailer indicating the model used in the work.
