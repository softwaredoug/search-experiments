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

Prompt placeholders include `{query}`, `{rubric}`, and any corpus column, such
as `{title}` or `{description}`. The response uses a structured numeric grade
constrained to the rubric's grade keys.

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

When OpenAI is used, we use the `AutoEnricher` class from cheat-at-search for 
caching and retries. If `--no-cache` is passed to `uv run run` (or any 
similar runner) we will bypass the cache for the LLM call. If {question} and 
{rubric} are not in the prompt, we will concat them to the LLM prompt

```
{question}
{rubric}
```

When Jev is used, we use typesafe's API, with the rubric as the list of choices
to a Jev `choice` in its API. The `prompt` becomes the `state`. Then the `question`
becomes the `question` in the Jev API. The Jev model returns a choice and a 
probability for that choice.
