# This project

This project is focused on benchmarking search methods. This document specifies how I expect it to all work.

Its my labratory for search approaches - lexical, vector, agentic, etc retrieval. On open datasets like MSMarco, Wands, Amazon ESCI, etc.

This doc sets out the important requirements of this project.

## Documentation map

Start with this PRD for project-wide requirements, then follow the strategy guide for the
`strategy.type` being configured. The type names below are the registered values in
[`exps/strategy_config.py`](../exps/strategy_config.py); `strategy.name` is the experiment name,
not the strategy type.

| Strategy type | What it does | Strategy guide | Example config / implementation |
| --- | --- | --- | --- |
| `bm25` | Lexical retrieval baseline with configurable fields and boosts. | This PRD's [configurable strategies](#configurable-strategies) section; there is no separate BM25 PRD yet. | [E-commerce BM25 config](../configs/ecom_base/bm25.yml) · [implementation](../exps/strategies/bm25.py) |
| `embedding` | Dense retrieval using an embedding model. | There is no separate embedding PRD yet. | [MiniLM config](../configs/ecom_base/embedding_minilm.yml) · [E5 MS MARCO config](../configs/msmarco/embedding_e5_msmarco.yml) · [implementation](../exps/strategies/embedding.py) |
| `agentic` | Tool-using search agent with validators, stoppers, and optional agent plans. | [Agentic search guide](agentic/agentic.md). Related: [tools](agentic/tools.md), [conditionals](agentic/conditionals.md), [filesystem tools](agentic_filesystem_prd.md), [agent topology experiment](orchestrate_prd.md), and [agentic notebooks](notebooks_agentic.md). | [E-commerce agentic config](../configs/ecom_base/agentic_ecom_bm25_gpt5_mini.yml) · [implementation](../exps/agentic/strategy.py) |
| `scatter_gather_wands` | Select WANDS categories, search each category, then gather and rerank candidates. | [Scatter/gather guide](scatter_gather.md) (WANDS-specific). | [Scatter/gather config](../configs/cheat-at-search/scatter_gather_wands.yml) · [implementation](../exps/agentic/scatter_gather.py) |
| `bag_of_decisions` | Generate yes/no relevance decisions and use Jev probabilities to rerank candidates. | [Bag-of-decisions guide](bag_of_decisions.md). | [LLM-generated decisions config](../configs/ecom_decisions/bag_of_decisions.yml) · [implementation](../exps/bag_of_decisions/strategy.py) |
| `query_understanding` | Enrich a query (for example, classify its category) before retrieval. | [Query-understanding guide](query_understanding/query_understanding.md), [enrichment engines](query_understanding/enrichment_engines.md), and [engine evaluation](query_understanding/enrichment_engines_eval.md). | [E-commerce query-understanding config](../configs/ecom_class/category/openai/ecom_query_understanding.yml) · [implementation](../exps/query_understanding/strategy.py) |
| `rag` | Rewrite a query with an LLM, then retrieve documents with one search tool. | [RAG guide](rag.md). | [BM25 RAG config](../configs/rag_bm25.yml) · [implementation](../exps/strategies/rag.py) |
| `codegen` | Train generated retrieval/reranking code against judgments. | [Code-generation PRD](codegen_prd.md). | [E-commerce codegen config](../configs/codegen/codegen_ecom.yml) · [implementation](../exps/codegen/strategy.py) |

When adding a registered strategy type, add its guide or implementation/config entry here so agents
can discover the strategy-specific requirements from `AGENTS.md` → this PRD.

### Cross-cutting guides

- [Testing practices](tests.md) and [end-to-end test guidance](e2e_tests.md)
- [Runner tests PRD](runner_tests_prd.md)
- [Notebook generation PRD](notebooks_prd.md), plus strategy-specific notebook guides where present

## Python tooling

This project is managed by uv

## Python Dependencies

It uses two primary libraries

### 1. Cheat at Search - https://github.com/softwaredoug/cheat-at-search

A library originally written for my agentic search class, w/ OpenAI hooks and some agent helpers. Its where much of the reusable functionality comes from. Helpers for evals, running multiple queries, and datasets.

### 2. SearchArray for lexical search - https://github.com/softwaredoug/searcharray

Lexical search pandas extension array. Once indexed, it lets you call `score` on a term, then returns BM25 - or other similarity - on that term.

## Dataset Independence (mostly)

Each search strategy should be written from a standpoint of being dataset agnostic. Each dataset is a pandas dataframe fulfilling this contract:

1. An optional 'title' column - the title, product name, etc of the document
2. A 'description' field. This is always present. IE product description, etc

IE for MSMarco Passages, it ONLY hase a 'description' field, which is the passage text. For Amazon ESCI, it has both 'title' and 'description' fields, which are the product title and description respectively.

## Dataset Evaluation against strategies

From the cheat-at-search library.

Also part of the dataset will be a 'judgments' dataframe. It labels relevant results for a set of queries.

Then you use `run_strategy` as follows:

1. A strategy is setup, with a corpus and whatever other params
2. You call run_strategy w/ judgments. Producing a dataframe of every query's search results, concatt'd. Also labeled from the judgments. No label for a doc implies irrelevance.
3. A helper `ndcgs` that takes these results, and gives per-query NDCG
4. A helper `mrr` that takes these results, and gives per-query MRR

## Configurable strategies

Every strategy class should be able to be configurable via a yml config file. The yml file looks like:

```yaml
strategy:
  name: strategy_name # Name referred to in CLI
  type: strategy_type # Name of the strategy class to use, a value on the class itself
  params:
    param1: value1 # Params that configure the strategy
    param2: value2
```

This *name* of the strategy here is actually what's referred to in scripts. IE for the basic BM25 strategy, we might have:

```yaml
strategy:
  name: bm25_strong_title
  type: bm25
  params:
    k1: 1.5
    b: 0.75
    title_boost: 200.0
    description_boost: 1.0
```

This would correspond to a BM25 strategy:

```python
class BM25Strategy(SearchStrategy):
    _type = "bm25"

    def __init__(
        self,
        corpus,
        title_boost=...
        description_boost=...
        k1=...
        b=...

```

## Caching

When a strategy is run on a dataset, the results might cached to disk by run_strategy. This allows for faster iteration when making changes to strategy implementations, as you can bypass the actual search and just load the cached results.

You can force the cache to be bypassed with cache=False to run_strategy. The user controls with --no-cache (this only affects run_strategy results, not BM25 indices or embeddings).

## Run / train working folders

All runners use a shared utility to create working folders for each strategy run.

- Training (codegen, etc):
  `~/.search-experiments/<strategy_type>/<dataset>/<strategy_name>/<timestamp>`
- Agentic traces:
  `~/.search-experiments/agentic/<dataset>/<strategy_name>/<timestamp>`

## Strategy Agnostic Scripts

The different scripts here that compare strategies should take as "--strategy" argument a yml file. 

Where appropriate, we should expect these params:

--query         # Only run with this query. Bypass - but do not delete - caches. If query in judgments, show its ground truth info (ie show grades + a sample of the most relevant doc first)
--dataset       # The dataset being run on the strategy ie wands, msmarco, etc
--num-queries   # Number of queries to run as a subset (for faster analysis) 
--seed          # Random seed for query sampling when num-queries is set
--workers       # Number of workers to use for parallel processing when applicable
--no-cache      # Bypass run_strategy cache only (does not affect BM25 indices or embeddings)


Here's some example executions

### Compare two bm25 variants

#### bm25_1.yml
```yaml
strategy:
  name: bm25_strong_title
  type: bm25
  params:
    k1: 1.5
    b: 0.75
    title_boost: 200.0
    description_boost: 1.0
```

#### bm25_2.yml
```yaml
strategy:
  name: bm25_strong_title
  type: bm25
  params:
    k1: 0.1
    b: 0.75
    title_boost: 1..0
    description_boost: 1.0
```

```bash

uv run diff --strategy-a bm25_1.yml --strategy-b bm25_2.yml --dataset msmarco
```

### Run a single strategy on a dataset

```bash
uv run run --strategy-a bm25_1.yml --dataset msmarco
```

### Run a strategy on a dataset, but only on a subset of queries for faster iteration

```bash
uv run run --strategy bm25_1.yml --dataset msmarco --num-queries 100 
```


### Diff a single query

```bash
uv run query --strategy bm25_1.yml --query "salon chair" --dataset wands --k 10
```

## Generating Notebooks 

For when I ask you to turn an experiment into a notebook:

See notebooks_prd.md

## Runner tests

For when I ask you to audit and make better end-to-end tests.

See runner_tests_prd.md

## Codegen strategies

For when I ask you to make a strategy that iteratively edits code to improve search results.

See codegen_prd.md
