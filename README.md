## Search experiments

Agentic search benchmarks on search datasets. How good can just an agent and a few tools do?

## E-commerce datasets

ESCI (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| bm25_strong_title | n/a | 0.2895 | 0.1707 |
| embedding_minilm | n/a | 0.2304 | 0.0854 |
| agentic_embeddings_ecommerce_gpt5_mini | gpt-5-mini | 0.2957 | 0.1863 |
| agentic_bm25_ecommerce_gpt5_mini | gpt-5-mini | 0.3807 | 0.3414 |
| agentic_bm25_ecommerce | gpt-5 | 0.3838 | 0.4268 |
| agentic_bm25_embeddings_ecommerce_gpt5_mini | gpt-5-mini | 0.3996 | 0.3510 |

WANDS (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| bm25_strong_title | n/a | 0.5408 | 0.4746 |
| embedding_minilm | n/a | 0.5060 | 0.4083 |
| agentic_embeddings_ecommerce_gpt5_mini | gpt-5-mini | 0.5367 | 0.4939 |
| agentic_bm25_ecommerce_gpt5_mini | gpt-5-mini | 0.5795 | 0.5609 |
| agentic_bm25_embeddings_ecommerce_gpt5_mini | gpt-5-mini | 0.5895 | 0.5609 |

## Benchmarks

### GPT-5


### GPT-5-mini



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

Datasets: esci, minimarco, msmarco, wands.

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
