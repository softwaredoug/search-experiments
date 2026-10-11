# LLM Judge Strategy

The `llm_pontwise_judge` strategy measures how well a pointwise judge reproduces a
dataset's numeric relevance grades. It samples judged query-document pairs,
asks the judge for a grade, then returns the sampled documents ranked by the
predicted grade. This makes the strategy runnable through the normal benchmark
path while keeping judge behavior and configuration inside the strategy.

Batch runs request pointwise and pairwise agreement from `run_strategy` for all
strategy types. Both metrics are computed on the top 10 results. Pointwise
agreement is the fraction of sampled documents whose predicted grade matches
the judgment; pairwise agreement measures whether the predicted ordering
preserves the ordering of the judgments. The runner reports their per-query
means alongside its regular ranking metric.

## Configuration

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

      Use this rubric and return the matching numeric grade:
      {rubric}
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
