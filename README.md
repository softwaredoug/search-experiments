## Search experiments

BM25 and agentic benchmarks on datasets available in Cheat at Search.

## Setup

```bash
uv sync
```

Install dev dependencies (flake8/pytest/ipython):

```bash
uv sync --extra dev
```

## Scripts

Run benchmarks:

```bash
uv run run --strategy configs/bm25_strong_title.yml --dataset wands
uv run run --strategy configs/agentic_bm25_embeddings.yml --dataset wands
```

Run a single query:

```bash
uv run query --strategy configs/bm25_strong_title.yml --dataset wands --query "salon chair"
uv run query --strategy configs/agentic_bm25_embeddings.yml --dataset wands --query "salon chair" --k 10
```

Run diffing tools:

```bash
uv run diff --strategy-a configs/bm25_strong_title.yml --strategy-b configs/bm25_strong_title.yml --dataset wands
```

## Lint

```bash
uv run flake8 .
```

## Tests

```bash
uv run pytest
```
