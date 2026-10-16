# Agentic tools

This page documents the tools available when experimenting with agentic strategies.

See [Agentic Strategy Docs](agentic.md) for the complete details of agentic data and the agent loop.

## Tools

The search tools here are specific functions executing a type of retrieval. While they can share
backend indices with search strategies (such as BM25), their tool arguments and scoring code are
separate.

The runtime source of truth is [`TOOL_REGISTRY`](../../exps/tools/registry.py). Use the registry
key in `params.search_tools`; the Python/OpenAI function name may be different. `delegate_task` is
provided by the agent harness rather than the registry.

The following context gets passed to OpenAI from each tool:

- the tool name
- the tool description, including argument documentation and configured guard descriptions
- the parameter schema generated from the Python function signature and type annotations

Tools may be listed by name or as a single-key mapping when configuration is needed. For example:

```yaml
params:
  search_tools:
    - bm25:
        params:
          title_boost: 9.3
          description_boost: 4.1
        guards:
          - disallow_repeated_queries
    - e5_base_v2
```

Some tools accept `columns` configuration to append corpus fields to each returned document. The
`source` names a corpus column and `as` names the field exposed to the agent:

```yaml
    - bm25:
        columns:
          - source: category
            as: product_category
```

### BM25 and embedding search

- `bm25`: searches title and description with BM25. Its builder accepts `title_boost`,
  `description_boost`, `k1`, and `b` under the tool's `params` mapping.
- `fielded_bm25`: searches caller-selected, weighted fields. Only `title` and `description` are
  supported. See [Fielded BM25](#fielded-bm25) below.
- `minilm`: embedding search using the default embedding model.
- `embeddings`: alias for the default embedding search (`minilm`).
- `e5_base_v2`: E5-base-v2 embedding search with `query: ` and `passage: ` prefixes.

These tools normally accept a `top_k` argument, defaulting to 5, and enforce a maximum of 100
results per call. The agentic strategy guidance may recommend a smaller value to keep results and
descriptions manageable in the model context.

### Read by ID

`read` retrieves one corpus document by its stable ID and returns it in the standard document
result format. It returns an empty list when the ID is not present. SID agents require this tool;
include `read` in their configured `search_tools`.

### Query rewrite

`query_rewrite` asks an LLM for spelling and acronym variants of the input query. It always includes
the original query in its `rewriters` list. The tool mapping accepts `model`, `max_alternatives`,
`temperature`, `reasoning_effort`, and `verbosity`; the default maximum is 5. This tool does not
support guards.

```yaml
    - query_rewrite:
        model: gpt-5-mini
        max_alternatives: 3
```

### Composite tools

We need to create tools that combine multiple search tools together. Composite tools can take
a set of other tools, combine them with some strategy (ie RRF) and then optionally rerank the results.


```yaml
params:
  search_tools:
    - composite:
        retrieval_engine:
          type: rrf     # Weighted Reciprocal Rank Fusion
          tools:       # Any other tool can go here
            - bm25:
                params:
                  title_boost: 9.3
                  description_boost: 4.1
                guards:
                  - disallow_repeated_queries
            - e5_base_v2
          weights: [1, 1]   # Weights for each tool in the same order as above
        reranker_engine:    # Optional reranker engine to rerank the results of the retrieval engine
            type: jev
            k: 100  # Rerank the top 100 results from the retrieval engine
            params:
                decision_model: jev/jev-latest
                decision_weight: 10
                confidence_threshold: 0.7
                state_format: |
                {title}
                {description}
                prompt: "Is this document relevant to the {query}?"
```

#### Retrieval Engine

Retrieval engine simply defines the initial retrieval arms or arm that would produce top K results.

- Currently the only retrieval engine supported is `rrf`
- In `tools` any other search tool could be used

#### Reranker Engine

The reranker engine reranks up to `k` candidates from the retrieval engine. Two
types are supported: `jev` and `bag_of_decisions`.

`jev` asks one relevance question per candidate and adds its weighted relevance
probability to the retrieval score. Jev is a decision model from Typesafe; see
the [reranker cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe).

`bag_of_decisions` first uses an LLM to generate query-specific yes/no
questions, then asks Jev for the probability that each question is true for each
candidate. It adds `decision_weight` times the sum of those probabilities to the
RRF score. Noul probabilities have no separate confidence value; every valid
probability contributes to scoring, regardless of its annotation category.
By default, decision details are used only for scoring. Set
`params.expose_decisions: true` to also return decision evidence on each evaluated
candidate. The tool output then includes `requirements_met`,
`requirements_failed`, and `requirements_uncertain` string lists. Probabilities
at least `0.7` are marked met, probabilities at most `0.3` are marked failed,
and probabilities between those thresholds (or missing/invalid probabilities)
are marked uncertain. This does not change the reranking score.

```yaml
reranker_engine:
  type: bag_of_decisions
  k: 100
  params:
    decision_model: jev/jev-latest
    decision_weight: 10
    expose_decisions: true  # Optional; defaults to false.
    state_format: |
      Query: {query}
      {title}
      {description}
    generator:
      type: llm
      model: gpt-5-mini
      system_prompt: |
        Generate concise, distinct yes/no questions for search relevance.
        A yes answer should indicate that a candidate satisfies the query.
      prompt: |
        Generate varied yes/no questions for deciding whether a search result
        is relevant to this query: {query}
```

When enabled, a candidate's tool result includes fields such as:

```json
{
  "requirements_met": ["✅ Is this a desk?"],
  "requirements_failed": ["❌ Is this intended for outdoor use?"],
  "requirements_uncertain": ["🤷 Is this useful for studying?"]
}
```

Only candidates within the configured reranker `k` receive these fields; when no
questions are generated, their three lists are empty.
An e-commerce composite example with decision exposure enabled is available at
[`configs/ecom_comp/agentic_ecom_composite_rrf_bag_of_decisions_exposed_gpt5_mini.yml`](../../configs/ecom_comp/agentic_ecom_composite_rrf_bag_of_decisions_exposed_gpt5_mini.yml).

The LLM generator accepts `model`, `system_prompt`, `prompt`, and optional
`reasoning`, `temperature`, `verbosity`, and `no_cache` settings. Its prompt may
use `{query}`. The state format may reference corpus fields and `{query}`.

The `reranker_engine` can be omitted, in which case the results from the
retrieval engine are returned as-is.


### WANDS-specific tools

these tools use wands category or product-feature data. the wands-suffixed search and filesystem
tools require the `wands` dataset. `top_categories` uses its configured category column, and
`check_features_wands` requires a `features` column:

- `bm25_wands`: bm25 title/description search, optionally filtered by one or more categories.
- `bm25_wands_prefiltered`: bm25 search within the required category; used by scatter/gather.
- `minilm_wands`: default-model embedding search, optionally category-filtered.
- `e5_base_v2_wands`: e5-base-v2 search with query/passage prefixes, optionally category-filtered.
- `e5_base_v2_wands_prefiltered`: e5-base-v2 search within the required category; used by
  scatter/gather.
- `top_categories`: returns common wands categories for category selection.
- `check_features_wands`: finds wands product features similar to requested feature names for a
  given document id.

the `*_prefiltered` tools require a category argument. scatter/gather configurations commonly use
`top_categories` in the select step and the prefiltered bm25/embedding tools in the scatter step;
see the [scatter/gather guide](../scatter_gather.md).

### tool guards

guards reject a tool call based on its arguments or per-query agent state. A guard's description is
appended to the tool description. The registered guards are:

- `disallow_repeated_queries`: rejects a query already used with that tool during the run.
- `query_min_length`: rejects an embedding query with fewer than its configured `min_terms`.
- `disallow_similar_queries`: rejects a query whose embedding is too similar to a previous query;
  `threshold` defaults to `0.9`.

Example:

```yaml
    - e5_base_v2:
        guards:
          - disallow_similar_queries:
              threshold: 0.9
```

`query_rewrite` does not accept guards. If a tool call raises an exception, the agent receives a
string error response rather than the run crashing.

### Dataset-specific tools

Some tools require a particular dataset because they use its fields, category structure, or
filesystem layout. WANDS tools use the WANDS dataset; if such a tool is used for another dataset,
construction fails. The `_wands` and `_wands_prefiltered` names are conventions, while the tool
registry and builder checks determine actual availability.

### Fielded BM25

`fielded_bm25` accepts weighted fields and an operator:

```text
fields: ["title^9.3", "description^4.1"]
operator: and
```

Operators are `and`, `or`, and `phrase`. `phrase` scores the query token list as a single term;
`and` requires each query term to occur in at least one selected field. Only title and description
are supported.

### Filesystem tools

The generic virtual-filesystem tools are:

- `ls(path, glob, max_results=50)`: list matching paths; at most 50 results.
- `grep(pattern, glob, num_results=50)`: search matching files with a regular expression; at most
  50 results.
- `cat(path)`: read a file's contents.

The builder automatically adds `search_directory` when `ls`, `grep`, and `cat` are all configured.
It delegates a scoped search to a sub-agent and does not allow nested delegation. The WANDS
variants are `ls_wands`, `grep_wands`, `cat_wands`, and `search_directory_wands`; the WANDS search
directory helper is automatically added when all three WANDS file tools are configured.

See the [Agentic filesystem guide](agentic_filesystem_prd.md) for the virtual filesystem layout
and examples.

### Bash tools

- `bash`: execute a command in the sandboxed filesystem service for the active dataset.
- `bash_wands`: execute a command in the WANDS sandbox.

The service must be running in Docker. Commands run under `/corpus`, default to a 30-second timeout
(bounded by `EXPS_BASH_MAX_TIMEOUT`), and return at most 8,000 characters. See the filesystem guide
for setup instructions.

### Codegen tool

`codegen` exposes a generated reranker as a search tool. Its configuration requires a `path` to a
reranker file or run directory. It also accepts an optional tool `name`, `description`, extra
`return_fields`, and `dependencies` on other registered search tools. The reranker must provide a
compatible `rerank_<dataset>` or `rerank` function. See the [codegen PRD](../codegen_prd.md).

### TODO tools

`todo_write` and `todo_read` maintain an in-memory list in `agent_state["todos"]`:

- `todo_write(todo, status)` appends an entry.
- `todo_read()` returns the current entries.

### Delegate task tool

`delegate_task` gives work to a sub-agent and returns its result. The sub-agent receives its own
configured tools and prompt. This tool is supplied by the agent harness, not `TOOL_REGISTRY`; enable
it in an agent's `search_tools` when that agent is configured for delegation. See the
[agent plans section](agentic.md#plan-through-list-of-agents).

### Raw tools

`get_corpus` is registered as a raw tool that returns the corpus DataFrame, but raw tools are not
allowed in standard agentic strategies. Configuring it there raises an error.
