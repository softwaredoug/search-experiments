## Search experiments

Agentic search benchmarks on search datasets.

How well can an agent search with just a few basic retrieval tools?

## E-commerce datasets

### The search tools

Retrieval using different models. With different sets of tools. Below you'll see some combination of:

- [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- [minilm](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- bm25 w/ std params on title and description fields.

All sharing basically the same prompt. Click each strategy to see prompt + tools used. A tool-calling loop will run through once, produce the best ranked results. We'll then evaluate those.

### Amazon ESCI 

Baselines first, agentic sorted by NDCG ascending. N=1000 queries.

ESCI is Amazon's Shopping Queries dataset for product search relevance with graded labels. [Source](https://github.com/amazon-science/esci-data)

| strategy | model | mean | median |
|---|---|---|---|
| [bm25](configs/ecom_base/bm25.yml) | n/a | 0.2895 | 0.1707 |
| [embedding_minilm](configs/ecom_base/embedding_minilm.yml) | n/a | 0.2304 | 0.0854 |
| [embedding_e5](configs/ecom_base/embedding_e5_base_v2.yml) | n/a | 0.3142 | 0.2250 |
| [agentic_minilm_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_minilm_gpt5_mini.yml) | gpt-5-mini | 0.2952 | 0.1749 |
| [agentic_e5_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_e5_base_v2_gpt5_mini.yml) | gpt-5-mini | 0.3569 | 0.3399 |
| [agentic_bm25_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_bm25_gpt5_mini.yml) | gpt-5-mini | 0.3777 | 0.3414 |
| [agentic_bm25_minilm_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_2tools_gpt5_mini.yml) | gpt-5-mini | 0.3958 | 0.3414 |
| [agentic_bm25_e5_base_v2_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_2tools_e5_gpt5_mini.yml) | gpt-5-mini | 0.4152 | 0.3794 |
| [agentic_bm25_e5_ecommerce_gpt5](configs/ecom_base/agentic_ecom_2tools_gpt5.yml) | gpt-5 | 0.4535 | 0.4417 |

![ESCI NDCG plot](assets/esci_ndcg.png)

![ESCI tool calls pareto](assets/esci_pareto_gpt5_mini.png)

### Wayfair WANDS

Baselines first, agentic sorted by NDCG ascending.

WANDS is Wayfair's product search relevance dataset with graded judgments. [Source](https://github.com/wayfair/WANDS)

| strategy | model | mean | median |
|---|---|---|---|
| [bm25](configs/ecom_base/bm25.yml) | n/a | 0.5408 | 0.4746 |
| [embedding_minilm](configs/ecom_base/embedding_minilm.yml) | n/a | 0.5060 | 0.4083 |
| [embedding_e5](configs/ecom_base/embedding_e5_base_v2.yml) | n/a | 0.5571 | 0.5475 |
| [agentic_minilm_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_minilm_gpt5_mini.yml) | gpt-5-mini | 0.5330 | 0.4874 |
| [agentic_e5_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_e5_base_v2_gpt5_mini.yml) | gpt-5-mini | 0.5789 | 0.5609 |
| [agentic_bm25_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_bm25_gpt5_mini.yml) | gpt-5-mini | 0.5795 | 0.5609 |
| [agentic_bm25_minilm_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_2tools_gpt5_mini.yml) | gpt-5-mini | 0.5867 | 0.5609 |
| [agentic_bm25_e5_base_v2_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_2tools_e5_gpt5_mini.yml) | gpt-5-mini | 0.5970 | 0.5609 |
| [agentic_bm25_e5_ecommerce_gpt5](configs/ecom_base/agentic_ecom_2tools_gpt5.yml) | gpt-5 | 0.6171 | 0.6256 |

![WANDS NDCG plot](assets/wands_ndcg.png)

![WANDS tool calls pareto](assets/wands_pareto_gpt5_mini.png)


### ESCI - Forcing more tool calls

Below we force the agent to make at least 4 calls to a retrieval backend, with two different enforcements on repeat queries: direct equivalence (after lowercasing, etc.) and semantic similarity.

| strategy | model | mean | median |
|---|---|---|---|
| [agentic_bm25_e5_ecommerce_gpt5_mini](configs/ecom_base/agentic_ecom_2tools_e5_gpt5_mini.yml) | gpt-5-mini | 0.4152 | 0.3794 |
| [agentic_bm25_e5_ecommerce_4calls_repeat_gpt5_mini](configs/ecom_mincalls/agentic_ecom_2tools_4calls_repeat_gpt5_mini.yml) | gpt-5-mini | 0.4292 | 0.3948 |
| [agentic_bm25_e5_ecommerce_4calls_sim0p9_gpt5_mini](configs/ecom_mincalls/agentic_ecom_2tools_4calls_sim0p9_gpt5_mini.yml) | gpt-5-mini | 0.4308 | 0.4258 |


## Run a strategy

```bash
uv run run --strategy configs/ecom_base/agentic_ecom_minilm_gpt5_mini.yml --dataset wands --num-queries 1000
```
