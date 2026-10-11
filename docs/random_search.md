# Random-search retrieval strategies

## Purpose

`random_search` is a trainable retrieval strategy that samples retrieval parameters,
evaluates them against judged queries, and saves the best configuration as a
run-ready artifact. The first version supports a BM25 retrieval engine only.

The strategy is configured as a retrieval engine in the same spirit as
`query_understanding`: a `base` selects the engine and `params` configure it. Unlike
`query_understanding`, the initial `random_search` strategy does not enrich queries;
it runs BM25 directly over the corpus using the configured fields and parameters.

Training and execution have different roles:

- `uv run train` performs the parameter search and writes an artifact.
- `uv run run` loads the artifact and executes its selected BM25 configuration.
- `strategy.path` identifies an existing artifact to continue training from or to
  run. Training without a path creates a timestamped artifact directory; running
  without a path loads the latest run for that strategy and dataset.

## Strategy configuration

The strategy parameters describe a retrieval engine. A parameter whose entire
value is a quoted Python-format placeholder, such as `"{k1}"`, is searchable.
Unmarked numeric parameters remain fixed. Each placeholder must have a matching
entry in `train.search_space`, which specifies the values from which random search
may sample.

```yaml
strategy:
  name: ecom_bm25_random
  type: random_search
  params:
    retrieval_engine:
      base: bm25
      params:
        k1: "{k1}"
        b: "{b}"
        title_boost: 9.3
        description_boost: 4.1
    train:
      rounds: 100
      search_seed: 42
      search_space:
        k1:
          distribution: uniform
          min: 0.2
          max: 2.0
        b:
          distribution: uniform
          min: 0.0
          max: 1.0
      objective: ndcg
      eval:
        train_fraction: 0.7
        validation_fraction: 0.15
        seed: 1234
```

`rounds` is the number of random candidate configurations to evaluate, and the
`--rounds` CLI option overrides it. The meaning of a round belongs to the
training method: for `random_search`, each round samples and evaluates a candidate
parameter assignment. It shares the outer-loop budget concept with codegen without
requiring the two methods to do the same work per round.

The initial search-space implementation should support numeric uniform ranges and
discrete choices. Distribution definitions are validated against the BM25
parameter's expected type and constraints. Search-space keys and template
placeholders must match exactly: an undeclared placeholder or an unused search
space entry is a configuration error. Placeholder values are parsed into the
expected numeric type before constructing the engine; templates are not arbitrary
executable Python.

Only `retrieval_engine.base: bm25` is supported initially. Its parameters follow
the existing BM25 strategy contract, including `k1`, `b`, `title_boost`, and
`description_boost`. Unknown parameter names and invalid/non-finite parameter
values fail with a configuration error before search begins. Corpus-field handling
follows the existing BM25 strategy.

## Query sampling and evaluation

The runner starts with the unique queries in the dataset's judgments.
`--num-queries`, when specified, samples a maximum-sized base query pool from
those queries using the CLI `--seed`; without it, all available judged queries
are used. The base pool is sampled once for the run.

The `train.eval` settings split that pool into disjoint query sets. The split uses
`train.eval.seed` and is fixed for all search rounds so every candidate is
compared on the same queries. `train_fraction` controls the training-query share;
`validation_fraction` controls the validation share; the remainder is the test
share. Fractions must be in range and leave non-empty sets when the selected pool
is large enough. Very small pools should produce a clear warning or error rather
than silently reporting misleading metrics.

For each candidate, the trainer records the objective metric on the training and
validation queries. It selects the winning candidate by validation objective,
using training objective as a diagnostic. The test set is held out from candidate
selection and is evaluated for the selected candidate at the end. The split
algorithm and all three query counts are written to metadata. This gives the
random-search strategy distinct validation and test sets; it should not inherit
codegen's current behavior of using the same remainder for both.

Random parameter sampling uses `train.search_seed`, independently of query-pool
sampling and split seeding. A run must be reproducible from its recorded seeds,
search-space configuration, and selected base query pool. Candidate sampling
should not depend on `rounds` changing the split.

The objective is a ranking metric over the configured top-k results and judgments.
The initial objective should be NDCG, consistent with the project's search
evaluation conventions. Candidate scores, sampled parameter values, and any
evaluation errors are recorded per round. Invalid candidates should be recorded
as failed trials rather than silently omitted, unless the failure makes the whole
configuration invalid.

## Training artifacts and continuation

A completed run directory contains at least:

- `metadata.json`: strategy type/name, relevant source configuration, dataset, seeds,
  split details, objective, winning parameters, completed rounds, and summary
  metrics.
- `trials.jsonl`: one record per candidate, including sampled parameters,
  training/validation metrics, status, and error details when applicable.
- `best_params.json`: a materialized winning configuration usable by the
  `random_search` runtime strategy without re-running the search.

The returned training result should use generic fields such as strategy name,
strategy parameters, artifact path, and metadata. Summary output should use
generic artifact/metrics names rather than codegen-specific column labels. The
artifact path is the durable output; logs and trial details remain in the run
directory.

When a run path with prior trials is supplied, continuation loads the recorded
search configuration and seeds, preserves existing trials, and adds the requested
number of additional candidate trials. The recorded round count is total completed
trials, while the invocation's `rounds` value is the number added. The previously
sampled base-query pool is stored in metadata and reused when continuing, including
when `--num-queries` is omitted on a later invocation.

## CLI example

```bash
uv run train \
  --strategy configs/random_search/ecom_bm25_random.yml \
  --dataset esci \
  --num-queries 500 \
  --seed 42 \
  --rounds 100
```

The runner prints the artifact path when training completes. That path can be
placed in `strategy.path` for `uv run run`; execution loads the winning BM25
parameters from the artifact and does not sample new candidates.

## Scope and extension points

The first implementation is deliberately limited to direct BM25 retrieval and
judgment-based ranking metrics. The retrieval-engine boundary should allow more
engine types later, while the training runner can keep common concerns such as
round budgets, seeds, run directories, and artifact reporting. Other training
methods can give `rounds` their own documented meaning and artifact format.

The initial implementation should not require changes to the ordinary BM25
strategy. It should reuse the same BM25 scoring behavior and ensure that a trained
artifact can be executed through the standard strategy `run` flow.
