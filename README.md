## Psuedorelevance Feedback benchmarks

BM25 + PRF benchmarks on datasets available in Cheat at Search.

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
uv run prf --strategy bm25 --dataset wands
uv run prf --strategy prf --dataset esci
```

Run a single query:

```bash
uv run prf-query --strategy bm25 --dataset wands --query "salon chair"
uv run prf-query --strategy prf --dataset msmarco --query "salon chair" --k 10
```

Run diffing tools:

```bash
uv run prf-diff --strategy bm25 --dataset wands
uv run prf-diff --strategy prf --dataset esci
```

Inspect PRF RM3 vectors:

```bash
uv run prf-vectors --dataset wands --query "salon chair" --k 10
uv run prf-vectors --dataset esci --query "salon chair" --fields title,description,category --k 10
uv run prf-vectors --dataset msmarco --query "salon chair" --fields title,description --debug-terms chair,stool --k 10
```

## Lint

```bash
uv run flake8 .
```

## Tests

```bash
uv run pytest
```
