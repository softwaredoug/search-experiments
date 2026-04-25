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

You can force the cache to be bypassed with cache=False to run_strategy. The user controls with --no-cache. 

## Strategy Agnostic Scripts

The different scripts here that compare strategies should take as "--strategy" argument a yml file. 

Where appropriate, we should expect these params:

--query         # Only run with this query. Bypass - but do not delete - caches. If query in judgments, show its ground truth info (ie show grades + a sample of the most relevant doc first)
--dataset       # The dataset being run on the strategy ie wands, msmarco, etc
--num-queries   # Number of queries to run as a subset (for faster analysis) 
--seed          # Random seed for query sampling when num-queries is set
--workers       # Number of workers to use for parallel processing when applicable
--no-cache      # Whether to bypass caching of strategy results, forcing a fresh run (useful for testing changes to strategy implementations)


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

## Turning configs into jupyter notebooks

This task primarilly exists for you, the agent, to do on demand.

For educational purposes, you may need to turn experiment in configs into a jupyter notebook (ipynb). More specifically, you will be asked to take a command like

```
uv run run --strategy configs/bm25.yml --no-cache --dataset msmarco --workers 4 
```

And create a notebook that loads msmarco, reimplements the BM25 strategy in the notebook, runs it on the dataset, and evaluates the results with NDCG/MRR.

These notebooks work as following:

### Notebook audience

The audience will be semi-aware of general search concepts and python programming.

### Notebook dependencies

* They do not depend on any code in this repo
* They're intended for a Google colab environment. So you can assume standard python data libreries like pandas, numpy, matplotlib, etc are available. But not much else.

### Mounting gdrive

This code should exist near the top to mount gdrive

```
!pip install git+https://github.com/softwaredoug/cheat-at-search.git@<commit_hash>
from cheat_at_search.data_dir import mount
mount(use_gdrive=True)    # colab, share data across notebook runs on gdrive
# mount(use_gdrive=False) # <- colab without gdrive
# mount(use_gdrive=False, manual_path="/path/to/directory")  # <- force data path to specific directory, ie you're running locally.
```

### Importing dataset + loading keys

Here's a sample cell of loading WANDS dataset. Included is the text cell above the code cell, which explains what's going on. You can assume the dataset is already indexed, so loading it is fast and doesn't require the user to run any indexing code.

```
## Get an OpenAI Key + load corpus

This will prompt you for an OpenAI Key to interact with GPT-5

-- 

from cheat_at_search.data_dir import key_for_provider
from openai import OpenAI
from cheat_at_search.wands_data import corpus, judgments

OPENAI_KEY = key_for_provider("openai")

openai = OpenAI(api_key=OPENAI_KEY)
```


### No caching of BM25 index, results, embeddings, etc

In this repo, when run as a CLI, there's a great deal of caching to speed experimentation. Do not
attempt to do this as you'll run in a notebook environment.

The only acceptable caching is what's done when mounting above


### Top level comment - from config yaml

If the config has a `description` field, include that as a markdown cell at the top of the notebook. This gives context to the experiment being run. If not, you can write a brief description of the experiment in your own words based on the config.


### Take guidance from description as well

The intent, etc will also be in the description. Use that to document the cells you create.

### Where notebooks live

Notebooks should be placed in the `notebooks/` directory. They should be named according to the strategy and dataset they correspond to, for easy identification. For example, a notebook for the BM25 strategy on the MSMarco dataset could be named `bm25_ms_marco.ipynb`.


### Use SearchStrategy

Make sure what's central to the notebook is a SearchStrategy implementation that demonstrates teh experiment. Instantiate the strategy, call run_strategy to get results dataframes. Use cheat_at_search's ndcgs/mrrs helpers to give summary eval metrics. 


### Ignore these params

While I'll ask you to create a notebook given a config + command, you should ignore these command line params that help when running locally

--no-cache
--device


