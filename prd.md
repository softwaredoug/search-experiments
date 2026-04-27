# This project

This project is focused on benchmarking search methods. This document specifies how I expect it to all work.

Its my labratory for search approaches - lexical, vector, agentic, etc retrieval. On open datasets like MSMarco, Wands, Amazon ESCI, etc.

This doc sets out the important requirements of this project.

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
