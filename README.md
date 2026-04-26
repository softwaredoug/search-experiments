## Search experiments

Agentic search benchmarks on search datasets. How good can just an agent and a few tools do?

## E-commerce datasets

ESCI (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| bm25 | n/a | 0.2895 | 0.1707 |
| embedding_minilm | n/a | 0.2723 | 0.1552 |
| agentic_minilm_ecommerce_gpt5_mini | gpt-5-mini | 0.2952 | 0.1749 |
| agentic_e5_base_v2_ecommerce_gpt5_mini | gpt-5-mini | 0.3569 | 0.3399 |
| agentic_bm25_ecommerce_gpt5_mini | gpt-5-mini | 0.3777 | 0.3414 |
| agentic_bm25_minilm_ecommerce_gpt5_mini | gpt-5-mini | 0.3958 | 0.3414 |
| agentic_bm25_e5_base_v2_ecommerce_gpt5_mini | gpt-5-mini | 0.4152 | 0.3794 |

![ESCI NDCG plot](assets/esci_ndcg.png)

WANDS (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| bm25 | n/a | 0.5408 | 0.4746 |
| embedding_minilm | n/a | 0.5316 | 0.4779 |
| agentic_minilm_ecommerce_gpt5_mini | gpt-5-mini | 0.5330 | 0.4874 |
| agentic_e5_base_v2_ecommerce_gpt5_mini | gpt-5-mini | 0.5789 | 0.5609 |
| agentic_bm25_ecommerce_gpt5_mini | gpt-5-mini | 0.5795 | 0.5609 |
| agentic_bm25_minilm_ecommerce_gpt5_mini | gpt-5-mini | 0.5867 | 0.5609 |
| agentic_bm25_e5_base_v2_ecommerce_gpt5_mini | gpt-5-mini | 0.5970 | 0.5609 |

![WANDS NDCG plot](assets/wands_ndcg.png)

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
