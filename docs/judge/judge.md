# LLM Judge Strategies

LLM judge strategies are used to test how good LLMs are at evaluating search results.
They come in different flavors (ie pointwise, pairwise, blitzrank, etc). And each
is implemented as its own strategy.

Some strategies involve training a model out of aggregate judge / LLM  / jev.

## Judge evaluation

LLM Judge strategies work differently than most strategies in this repo.

Typically we'd get some organic search results and evaluate them using the
labeled results. IE compute NDCG and traditional metrics. But in this case,
we care more about pointwise and pairwise agreement.

To do this, and fit within existing APIs seamlessly, we sample 10 judged
query-document pairs from the dataset per query. We evaluate these, labeling
them with some score. We'll either evaluate this with pointwise agreement
(does score == grade) or pairwise ( doc1 > doc2 in score implies doc1 > doc2 in grade).

All this is handled for us in `run_strategy`

Batch runs request pointwise and pairwise agreement from `run_strategy` for all
strategy types. Both metrics are computed on the top 10 results. Pointwise
agreement is the fraction of sampled documents whose predicted grade matches
the judgment; pairwise agreement measures whether the predicted ordering
preserves the ordering of the judgments. The runner reports their per-query
means alongside its regular ranking metric.

## LLM Pointwise Judge

The `llm_pontwise_judge` strategy measures how well a pointwise judge reproduces a
dataset's numeric relevance grades. It samples judged query-document pairs,
asks the judge for a grade, then returns the sampled documents ranked by the
predicted grade. This makes the strategy runnable through the normal benchmark
path while keeping judge behavior and configuration inside the strategy.

### Configuration

The rubric keys are numeric grades supported by the dataset, and their values
are descriptions for the judge. Sampling is deterministic and divided as
evenly as possible across the query's available grades. `sample_size` may be
between 1 and 10; if a query has fewer judged documents, all available
documents are used.

`question` is an explicit required instruction for selecting a rubric grade.
Prompt and question placeholders include `{query}` and any corpus column, such
as `{title}` or `{description}`. The prompt can also include `{question}` and
`{rubric}`. The response uses a structured numeric grade constrained to the
rubric's grade keys.

```yaml
strategy:
  name: llm_judge_relevance
  type: llm_pontwise_judge
  params:
    model: gpt-5-mini
    system_prompt: You are a careful search relevance judge.
    prompt: |
      Judge how relevant this document is to the search query.

      Query: {query}
      Title: {title}
      Description: {description}
    question: |
      Which relevance label best describes the document's relevance to the query?
    rubric:
      2: relevant
      1: partially relevant
      0: irrelevant
    sample_size: 10
    sampling_seed: 42
```

The rubric is dataset-specific configuration, not an assumption in the
strategy implementation. For corpora without a title column, omit `{title}`;
other available corpus columns may be used in the prompt.

### Model selection / behavior

Models can be OpenAI or Jev models. OpenAI models either look like openai/gpt-5 
or just gpt-5 (ie no prefix implies OpenAI). Jev models always prefix jev/, ie
jev/jev-latest

When an OpenAI model is used, `LLMJudgeEngine` wraps cheat-at-search's
`AutoEnricher` for caching and retries. If `--no-cache` is passed to `uv run
run` (or a similar runner), the engine bypasses the enricher cache for that
call. The engine appends the explicit question and rubric independently when
the configured prompt does not contain the corresponding `{question}` or
`{rubric}` placeholder:

```
{question}
{rubric}
```

When a `jev/...` model is used, `JevJudgeEngine` calls Typesafe's API. The
rendered `prompt` becomes the Jev `state`; the explicit `question` becomes the
instructions for a Jev `Choice`; and the rubric's numeric grade keys and
descriptions become that choice's criteria. The selected choice maps back to a
numeric grade for ranking and agreement metrics. The engine also retains the
probability for the selected choice in its judgment result.

## LLM Pairwise Judge

LLM Pairwise Judge compares two documents. The first document is refered to as lhs
and the second as rhs (as in left-hand side and right-hand side).

We refer to the attributes as \*_lhs and \*_rhs.

```yaml
strategy:
  name: llm_judge_relevance
  type: llm_pairwise_judge
  params:
    model: gpt-5-mini
    system_prompt: You are a careful search relevance judge.
    prompt: |
      Query: {query}

      LHS Doc:
      Title: {title_lhs}
      Description: {description_lhs}

      RHS Doc:
      Title: {title_rhs}
      Description: {description_rhs}
    to_pointwise: copeland
    question: |
      Which document is more relevant to the query?
    sample_size: 10
    sampling_seed: 42
```

`sample_size` is the number of documents sampled per query (1 to 10), not the
number of comparisons. The strategy compares every unique pair from that sample.

Optional additional config:

```
confidence_threshold: 0.7
```

Only accept a Jev decision when the selected `lhs` or `rhs` choice probability is
strictly greater than `confidence_threshold`. Otherwise, return `Neither` for
this pair. OpenAI pairwise judges do not provide choice probabilities, so this
option is rejected for OpenAI models.

For Jev, the rendered `prompt` is the state being evaluated. The `question` is
sent separately as the `Choice` instructions, with `lhs` and `rhs` as the
available choices. A confidence-threshold abstention is applied by the strategy;
it does not require adding `neither` as a Jev choice.

```
explicit_neither: true
```
(defaults to false)

Jev or LLM explicitly allow the LLM to return "Neither" in the response_model / list of choices.

```
double_check: true
```
(defaults to true)

Swap LHS / RHS and if you result with a different pairwise evaluation, return "Neither" for this pair.

### Conversion to pointwise

`to_pointwise: copeland` converts pairwise outcomes to ranking scores. A win is
1 point, a loss is 0 points, and a tie or `Neither` is 0.5 points for each
document. Documents are sorted by total points, with corpus order breaking ties.


## LLM Pairwise Trained Judge

LLM Pairwise trained judge. Requires to be trained first via `uv run train`

1. Runs a set of judges, focused on different attributes (templated like pairwise judge above)
2. Trains a model to predict teh true pairwise outcome using these comparisons as features (model below)

(model can either be a decision tree or gradient boosted tree, from sklearn)

(See [this article](https://softwaredoug.com/blog/2025/01/21/llm-judge-decision-tree))

We refer to the attributes as \*_lhs and \*_rhs.

```yaml
strategy:
  name: llm_judge_relevance
  type: llm_pairwise_trained_judge
  params:
    train:
      - cross_validation: 5
    sample_size: 10
    sampling_seed: 42
    to_pointwise: copeland
    model: decision_tree
    features:
      - title:
            model: jev/jev-latest
            prompt: |
              Query: {query}

              LHS Doc:
              Title: {title_lhs}

              RHS Doc:
              Title: {title_rhs}
            question: |
              Which document is more relevant to the query?
      - description:
            model: jev/jev-latest
            prompt: |
              Query: {query}

              LHS Doc:
              Description: {description_lhs}

              RHS Doc:
              Description: {description_rhs}
            question: |
              Which document is more relevant to the query?
        ```
