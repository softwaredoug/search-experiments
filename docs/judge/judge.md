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

To do this, and fit within existing APIs seamlessly, strategies sample judged
query-document pairs according to their configured sampling policy. We evaluate
the selected documents, labeling them with some score. We'll either evaluate
this with pointwise agreement (does score == grade) or pairwise (doc1 > doc2 in
score implies doc1 > doc2 in grade).

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
are descriptions for the judge. Sampling uses the shared deterministic,
grade-stratified policy described in [Sample size](#sample-size). `sample_size`
may be between 1 and 10.

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

### Sample size

`sample_size` is a per-query target from 1 to 10, not a limit on the number of
documents or the number of comparisons. The shared sampler counts distinct
numeric grades across usable judgments (capped at 10), then sets a per-grade
quota to `max(round(sample_size / grade_count), 1)`. It samples up to that many
documents from each query/grade group, using `sampling_seed` for reproducibility.
This same sampling policy is used by the pointwise and pairwise-trained judges.

The quota is rounded to the nearest integer, with a minimum of one, so the
resulting sample can be smaller or larger than `sample_size`. For example,
`sample_size: 10` with three grades gives a quota of three per grade (up to nine
documents); `sample_size: 2` with four grades gives one per grade (up to four
documents). If a query/grade group has fewer documents than its quota, all of
its documents are used. The strategy compares every unique pair from the
resulting sample.

Optional additional config:

```
confidence_threshold: 0.7
```

Only accept a Jev decision when the selected `lhs` or `rhs` choice probability is
strictly greater than `confidence_threshold`. Otherwise, return `Neither` for
this pair. OpenAI pairwise judges do not provide choice probabilities, so this
option is rejected for OpenAI models.

For Jev, the rendered `prompt` is the state being evaluated. The pairwise engine
uses cheat-at-search's `AutoDecisionEnricher` to send that state, question, and
the configured `lhs`/`rhs` choices (plus optional `neither`) to Jev. Decision
probabilities are cached by request and used for pairwise confidence thresholds.
`uv run run --no-cache` clears the relevant decision cache before running.

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

## Blitzrank judge

Blitzrank asks the LLM to rank a list of documents in one call. We infer a preference
for each document pair from their relative positions in that ranking, then use those
preferences to measure pairwise agreement. Unlike the pairwise judge, Blitzrank does
not ask the LLM to evaluate each pair separately, so we do not need to sample pairs.

Blitzrank works in multiplle rounds, requiring a tournament style scheduler where the wins 
and losses are tracked in a directed graph. Then we choose N in each round to sort listwise 
from the scheduler.

The scheduler
- Constructs a DAG from the graph (eliminating cycles / strongly connected components)
- Counts losses (including transitive losses) - and accounts for the size of some SCCs
- Counts wins (including transitive wins) - and accounts for the size of some SCCs
- Tracks kappa (kappa = wins + losses + num_within_scc)

Then chooses the next N in the contest via

- Least losses
- Least wins on tie
- Highest kappa on tie of those two
- Doc ID for tie of those three

Only OpenAI models are supported for blitzrank.


```yaml
strategy:
  name: blitzrank_judge_relevance
  type: blitzrank_judge
  params:
    model: gpt-5-mini
    system_prompt: You are a careful search relevance judge, order the results given from best to worst
    prompt: |
      Rerank these search results from most to least relevant to
      the query:

      {query}
    result_prompt: |
      ## {title}

      {description}
    sample_size: 10
    sampling_seed: 42
    to_pointwise: copeland
    scheduler:
        type: tournament
        params:
            n_per_round: 3
            max_rounds: 5
```

Above, we set the system_prompt as usual.

prompt + sample_size * result_prompt for every initial search result we're evaluating for the query. With
the usual templating rules.

We run 5 rounds per query, with 3 documents per round. The scheduler will choose the next 3 documents to rank based on the previous rounds results.

Since we are outputing pairwise results, `to_pointwise: copeland` is used to convert the pairwise 
results to a pointwise score for each document.

Only LLMs are supported, not Jev

## Pairwise Trained Judge

The pairwise trained judge computes categorical pairwise-judge features and fits
a binary classifier to predict which document has the higher judged relevance.
Train it with `uv run train`; the final model is refit on all training pairs after
cross-validation. At runtime, the strategy computes the same features, predicts
each pair's winner, and converts wins to Copeland scores.

```yaml
strategy:
  name: pairwise_judge_relevance
  type: pairwise_trained_judge
  params:
    train:
      cross_validation: 5
    sample_size: 10
    sampling_seed: 42
    to_pointwise: copeland
    model:
      type: decision_tree
      params:
        prediction_threshold: 0.7
    features:
      - title:
          model: jev/jev-latest
          prompt: |
            Query: {query}

            LHS Doc:
            Title: {title_lhs}

            RHS Doc:
            Title: {title_rhs}
          question: Which document is more relevant to the query?
      - description:
          empty_value: No Description Provided
          model: gpt-5-mini
          system_prompt: You are a careful search relevance judge.
          prompt: |
            Query: {query}

            LHS Doc:
            Description: {description_lhs}

            RHS Doc:
            Description: {description_rhs}
          question: Which document is more relevant to the query?
```

`train.cross_validation` is the number of folds (at least 2). Folds are grouped
by query so pairs from one query cannot appear in both the fold's training and
validation sets. The reported accuracy is pairwise accuracy on each validation
fold. After cross-validation, the selected classifier is fit on all available
training pairs and saved with its feature definitions in the training artifact.
Cross-validation also reports precision, defined here as the fraction of
non-abstained pair predictions that are correct. `mean_precision` pools those
decisive predictions across folds, while `mean_prediction_coverage` reports the
fraction of validation pairs on which the classifier expressed a preference.
The same accuracy, precision, and coverage metrics are reported for each raw
pairwise feature in `cross_validation.feature_metrics`, allowing comparison
with the combined model. A feature's `neither` output remains a categorical
input to the combined model; for that feature's standalone metrics, it is an
abstention (incorrect for accuracy, excluded from precision).
The supported classifiers are `decision_tree` and `gradient_boosted_tree`.
The gradient-boosted classifier is sklearn's `GradientBoostingClassifier`; its
optional `model.params.n_estimators` sets the number of boosting stages (default
`100`), and `model.params.max_depth` sets the base trees' maximum depth (default
`3`; `null` allows unbounded depth). For example:

```yaml
model:
  type: gradient_boosted_tree
  params:
    n_estimators: 100
    max_depth: 3
    prediction_threshold: 0.9
```

Each classifier validates its own estimator parameters. `prediction_threshold`
is shared across classifiers and controls whether low-confidence predictions
abstain as `neither`.

The optional `model.params.prediction_threshold` uses the maximum class
probability from the classifier: predictions below the threshold become
`neither`. An abstention awards each document half a Copeland point and counts as
incorrect in cross-validation accuracy. If omitted, the classifier always
predicts its most likely class.

Pairs whose judged grades are equal are omitted from supervised training; the
target classes are only `lhs` and `rhs`. Feature judges may still return
`neither`, which remains a categorical feature value. Feature prompts use
`{query}` and corpus fields suffixed with `_lhs` or `_rhs`.
Each feature may set `empty_value` to replace empty prompt inputs. This covers
blank or whitespace-only strings, `None`, `NaN`/`pd.NA`, and empty containers. It
defaults to `Unknown`; for example, `empty_value: No Description Provided`
inserts that text wherever a feature's prompt input is empty.

Pass `--workers N` to `uv run train` to compute feature values for up to `N`
document pairs concurrently.

Training creates an artifact under the usual strategy training directory. Point
`strategy.path` at that artifact when running, or omit it to load the latest
trained run for the strategy and dataset.
