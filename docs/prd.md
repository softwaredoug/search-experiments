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
| `agentic` | Tool-using search agent with validators, stoppers, and optional agent plans. | [Agentic search guide](agentic/agentic.md). Related: [tools](agentic/tools.md), [conditionals](agentic/conditionals.md), [filesystem tools](agentic/agentic_filesystem_prd.md), [agent topology experiment](agentic/orchestrate_prd.md), and [agentic notebooks](agentic/notebooks_agentic.md). | [E-commerce agentic config](../configs/agentic/ecom_agentic/agentic_ecom_bm25_gpt5_mini.yml) · [implementation](../exps/agentic/strategy.py) |
| `scatter_gather_wands` | Select WANDS categories, search each category, then gather and rerank candidates. | [Scatter/gather guide](scatter_gather.md) (WANDS-specific). | [Scatter/gather config](../configs/agentic/cheat-at-search/scatter_gather_wands.yml) · [implementation](../exps/agentic/scatter_gather.py) |
| `bag_of_decisions` | Generate yes/no relevance decisions and use Jev probabilities to rerank candidates. | [Bag-of-decisions guide](bag_of_decisions.md). | [LLM-generated decisions config](../configs/ecom_decisions/bag_of_decisions.yml) · [implementation](../exps/bag_of_decisions/strategy.py) |
| `query_understanding` | Enrich a query (for example, classify its category) before retrieval. | [Query-understanding guide](query_understanding/query_understanding.md), [enrichment engines](query_understanding/enrichment_engines.md), and [engine evaluation](query_understanding/enrichment_engines_eval.md). | [E-commerce query-understanding config](../configs/ecom_class/category/openai/ecom_query_understanding.yml) · [implementation](../exps/query_understanding/strategy.py) |
| `rag` | Rewrite a query with an LLM, then retrieve documents with one search tool. | [RAG guide](rag.md). | [BM25 RAG config](../configs/rag_bm25.yml) · [implementation](../exps/strategies/rag.py) |
| `llm_pontwise_judge` | Sample judged query-document pairs and rank them by a configured pointwise relevance judge. | [LLM judge guide](judge/judge.md). | [WANDS GPT-5-mini config](../configs/ecom_llm_as_judge/wands_pointwise_judge_gpt5_mini.yml) · [implementation](../exps/strategies/judge.py) |
| `llm_pairwise_judge` | Compare sampled judged documents pairwise and rank them with the configured Copeland conversion. | [LLM judge guide](judge/judge.md). | [implementation](../exps/judge/pairwise_strategy.py) |
| `pairwise_trained_judge` | Train a binary pairwise outcome classifier from Jev/LLM judge features, then rank by predicted wins. | [Pairwise trained judge guide](judge/judge.md#pairwise-trained-judge). | [implementation](../exps/judge/pairwise_trained_strategy.py) · [training](../exps/judge/pairwise_train.py) |
| `codegen` | Train generated retrieval/reranking code against judgments. | [Code-generation PRD](codegen_prd.md). | [E-commerce codegen config](../configs/codegen/codegen_ecom.yml) · [implementation](../exps/codegen/strategy.py) |
| `random_search` | Tune BM25 retrieval parameters against judged queries. | [Random-search guide](random_search.md). | [Example config](../configs/random_search/ecom_bm25_random.yml) · [implementation](../exps/random_search/strategy.py) |

When adding a registered strategy type, add its guide or implementation/config entry here so agents
can discover the strategy-specific requirements from `AGENTS.md` → this PRD.

## Random search

`random_search` tunes BM25 parameters against judged queries and saves the selected configuration as a runnable artifact. See the [random-search guide](random_search.md) for its configuration, training, and artifact contract.

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
3. A helper `ndcgs` that takes these results, and gives per-query NDCG. The runner reports both
   `ndcg_m` (normalized by max-grade ideal DCG) and `ndcg_g` (normalized by judged-document ideal DCG).
4. A helper `mrr` that takes these results, and gives per-query MRR

Full benchmark runs automatically write `per_query.csv` and `full_results.parquet`
under `<artifact_path>/reports/<run_id>`. Strategies without an artifact path use a
temporary run directory. `RunResult.reports_path` and the optional `--summary-csv`
aggregate both identify the report directory so downstream analysis can locate the
per-query and document-level data.

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

Which would have a `search` or `search_batch` method, like

```
def search(self, query, k=10):
    """Dumb baseline lexical search"""
    tokenized = snowball_tokenizer(query)
    bm25_scores = np.zeros(len(self.index))
    for token in tokenized:
        bm25_scores += self.index['title_snowball'].array.score(token) * self.title_boost
        bm25_scores += self.index['description_snowball'].array.score(
            token) * self.description_boost
    top_k = np.argsort(-bm25_scores)[:k]
    scores = bm25_scores[top_k]
    return top_k, scores
```

## Resource reporting

The method `search` might return a 3rd return value: a dict of strings -> float | int, which report
some resource usage statistic. Basically resource cost for this query.

As convention, these should be prefixed `resource.` like `resource.latency`

Then running a strategy can report aggregate stats for any resource


## Caching

When a strategy is run on a dataset, the results might cached to disk by run_strategy. This allows for faster iteration when making changes to strategy implementations, as you can bypass the actual search and just load the cached results.

You can force the cache to be bypassed with cache=False to run_strategy. The user controls with --no-cache (this only affects run_strategy results, not BM25 indices or embeddings).

The query-sampling `--seed` does not affect the result-cache key. Use `--cache-salt` to separate repeated stochastic runs that use the same queries, such as replay rounds. Keep the seed fixed to sample the same query subset and change the cache salt for each replay.

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
--cache-salt    # Optional string to distinguish cached stochastic replays
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

## Research: Scripts, experiments, graphs, etc

A sequence of configs might be compared in an experiment. The general flow is

(a) We organize a set of configs in configs with some theme or experiment name, ie configs/ecom_decisions/*.yml
(b) We create a shell script that runs each config over some datasets, outputing results to a CSV to research/results
(c) We create a python script to graph the results of each variant to assets/ folder. The python script is
    run in the shell script as part of (b)
(d) A Markdown report summarizing what happened and giving how to reproduce may be requested to be built in in research/

### Config organization

configs/ folders organize yml files that configure experiments, incrementally varying some variables. Each intended to
be run vith `uv run ruv --strategy <config> --dataset <dataset>`

### Script 

We want to analyze the variants for retrieval quality. And also retrieval quality vs resource usage (ie to find the 
pareto frontier).

A script may then be requested to run these configs over a requested set of datasets (ie esci and wands are often 
used in e-commerce). This script outputs, for each variant config yml a CSV with the config name, the retrieval metrics
(ie NDCG, MRR, etc) and all resource.* metrics (ie tool calls, API calls, latency, token usage, etc). 

### Graphing

Graphs are intended to be some kind of line graph. And we should generate a couple types of graphs

- Retrieval: on LHS the worst variants, on RHS the best variants. Primarilly of NDCG unless otherwise requested
- Pareto graph: x-axis would be resources, y-axis would be retrieval - whatever is relevant to the experiments
  being run

#### Retrieval graph

A retrieval graph should be a single graph, X-axis being variants, Y-axis being retrieval metric (ie NDCG) and
if multiple datasets are present, each dataset should get its own graph.

A retrieval graph should have a dotted line on the Y-axis that takes the mean of `max_ndcg_m` of the query set
to show the maximum possible `ndcg_m` for this population of queries in this dataset. (`ndcg_m=1` is not
necessarily attainable because it is normalized by the max-grade ideal DCG.)

Graphs should add points and labels for each variant to name each one.


### Research Reports

Generate reports summarizing findings when requested.

Research reports should be brief explanation of how to run variants, graph, and table of results. It should
be written dryly and succinctly. See the research/ directory for existing reports, though many may not have
been written according to these guidelines

At the top there should be a disclaimer that reports were generated by an LLM.

## Generating Notebooks 

For when I ask you to turn an experiment into a notebook:

See notebooks_prd.md

## Runner tests

For when I ask you to audit and make better end-to-end tests.

See runner_tests_prd.md

## Codegen strategies

For when I ask you to make a strategy that iteratively edits code to improve search results.

See codegen_prd.md
