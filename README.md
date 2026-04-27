## Search experiments

Agentic search benchmarks on search datasets. How good can just an agent and a few tools do?

## E-commerce datasets

ESCI (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| [bm25](configs/bm25.yml) | n/a | 0.2895 | 0.1707 |
| [embedding_minilm](configs/embedding_minilm.yml) | n/a | 0.2304 | 0.0854 |
| [agentic_minilm_ecommerce_gpt5_mini](configs/agentic_ecom_minilm_gpt5_mini.yml) | gpt-5-mini | 0.2952 | 0.1749 |
| [embedding_e5](configs/embedding_e5_base_v2.yml) | n/a | 0.3142 | 0.2250 |
| [agentic_e5_ecommerce_gpt5_mini](configs/agentic_ecom_e5_base_v2_gpt5_mini.yml) | gpt-5-mini | 0.3569 | 0.3399 |
| [agentic_bm25_ecommerce_gpt5_mini](configs/agentic_ecom_bm25_gpt5_mini.yml) | gpt-5-mini | 0.3777 | 0.3414 |
| [agentic_bm25_minilm_ecommerce_gpt5_mini](configs/agentic_ecom_2tools_gpt5_mini.yml) | gpt-5-mini | 0.3958 | 0.3414 |
| [agentic_bm25_e5_base_v2_ecommerce_gpt5_mini](configs/agentic_ecom_2tools_e5_gpt5_mini.yml) | gpt-5-mini | 0.4152 | 0.3794 |
| [agentic_bm25_e5_ecommerce_gpt5](configs/agentic_ecom_2tools_gpt5.yml) | gpt-5 | 0.4535 | 0.4417 |

![ESCI NDCG plot](assets/esci_ndcg.png)

WANDS (baselines first, agentic sorted by NDCG ascending):

| strategy | model | mean | median |
|---|---|---|---|
| [bm25](configs/bm25.yml) | n/a | 0.5408 | 0.4746 |
| [embedding_minilm](configs/embedding_minilm.yml) | n/a | 0.5060 | 0.4083 |
| [agentic_minilm_ecommerce_gpt5_mini](configs/agentic_ecom_minilm_gpt5_mini.yml) | gpt-5-mini | 0.5330 | 0.4874 |
| [embedding_e5](configs/embedding_e5_base_v2.yml) | n/a | 0.5571 | 0.5475 |
| [agentic_e5_ecommerce_gpt5_mini](configs/agentic_ecom_e5_base_v2_gpt5_mini.yml) | gpt-5-mini | 0.5789 | 0.5609 |
| [agentic_bm25_ecommerce_gpt5_mini](configs/agentic_ecom_bm25_gpt5_mini.yml) | gpt-5-mini | 0.5795 | 0.5609 |
| [agentic_bm25_minilm_ecommerce_gpt5_mini](configs/agentic_ecom_2tools_gpt5_mini.yml) | gpt-5-mini | 0.5867 | 0.5609 |
| [agentic_bm25_e5_base_v2_ecommerce_gpt5_mini](configs/agentic_ecom_2tools_e5_gpt5_mini.yml) | gpt-5-mini | 0.5970 | 0.5609 |
| [agentic_bm25_e5_ecommerce_gpt5](configs/agentic_ecom_2tools_gpt5.yml) | gpt-5 | 0.6171 | 0.6256 |

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
