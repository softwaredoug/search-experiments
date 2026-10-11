# Bag of Decisions

Decision models, like jev, accurately answer yes / no questions with a given probability. That can be 
applied to search.

Imagine a query like

`wayfair tension rod`

An LLM can be used to generate questions like

* Is this a tension rod for a shower curtain?
* Is it wayfair branded?
...

Then a decision model can answer these questions

So this strategy has an LLM take a query, produce decisions (hopefully somewhat orthoginal), retrieve
results, then rank candidate documents based on the sum of the probability of the decisions that are true for that document.

Similar to docs/query_understanding.md, it has a retrieval engine that dictates the L0 retrieval. But we also have a decision engine
that generates decisions, and resolves them with decisions

## Example

Below gpt-5 is used to generate questions, then these are turned into `noul` yes/no questions to
Jev. The original BM25 score is then boosted by the sum of the probabilities of the decisions that are true for that document,
multiplied by `decision_weight` (below 10)

```yaml
strategy:
  name: bag_of_decisions_example
  type: bag_of_decisions
  params:
    decision_engine:
      generator:
        type: llm
        model: gpt-5
        system_prompt: |
          You are a search relevance decision engine. You will be given a query, and your job is to generate a list of yes/no questions that can be used to determine if a document is relevant to the query.
        prompt: |
          For the given query, please generate a list of yes/no questions where the affirmative
          indicates the document is relevant to the query
        
          Here's the query:
        
          {query}
      reranker:
        decision_model: jev/jev-latest
        decision_weight: 10
        k: 100
        state_format: |
          {title}
          {description}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]  # Baseline BM25
```

### LLM call

LLM calls are done with Cheat at Search's AutoEnricher class. If `--no-cache` is passed to `uv run run` (or any similar runner) we 
will bypass the cache for the LLM call

### Decision model call

A [`noul`][1] is created for each generated yes/no question. All questions are sent
in one request for each candidate document. Noul returns `P(yes)` and has no
separate confidence value or confidence threshold. The strategy sums the `P(yes)`
values for the candidate's decisions, multiplies the sum by `decision_weight`,
and adds it to the document's retrieval score.

The `state` of Jev is rendered from `state_format`, which can reference corpus
columns such as `{title}` and the current `{query}`.

We will rerank the top `decision_engine.reranker.k` documents from the retrieval engine.

### Acceptance threshold

The optional `decision_threshold` parameter is configured under
`decision_engine.reranker`. It is a minimum acceptance threshold for each
individual decision probability returned by Jev, not a threshold on the total
score for a candidate. For each candidate document, include a decision's
`P(yes)` in the sum only when it is strictly greater than `decision_threshold`.
Probabilities equal to or below the threshold are ignored; they contribute
neither a positive nor a negative amount to the sum. The strategy then applies
`decision_weight` to the sum of the accepted probabilities as usual.

For example, with `decision_threshold: 0.7`, a candidate whose decisions have
`P(yes)` values of `0.92`, `0.70`, and `0.31` has an accepted probability sum
of `0.92`: `0.70` is equal to (not greater than) the threshold, and `0.31` is
below it, so neither is included. With `decision_weight: 10`, the decisions add
`9.2` to the candidate's retrieval score.

```yaml
decision_engine:
  reranker:
    decision_model: jev/jev-latest
    decision_weight: 10
    decision_threshold: 0.7
    k: 100
    state_format: |
      {title}
      {description}
```

If `decision_threshold` is omitted, all returned `P(yes)` values are summed,
as described above. This only filters the probabilities used in the reranking
boost; it does not discard candidate documents or change which questions are
sent to Jev.

### Direct generator

Set `decision_engine.generator.type` to `direct` to use a single configured
question instead of calling an LLM to generate questions. The question is
formatted with `{query}` and then sent to Jev for each candidate document.
The optional `criteria` mapping describes what `true` and `false` mean for that
Noul; configured criteria are currently supported by the direct generator only.

```yaml
decision_engine:
  generator:
    type: direct
    question: |
      Does this candidate product satisfy the search query "{query}"?
    criteria:
      "true": >-
        The candidate is the requested product type and satisfies the query's
        key shopping intent and explicit constraints.
      "false": >-
        The candidate is a different product type, conflicts with an explicit
        constraint, or only overlaps with incidental query terms.
  reranker:
    decision_model: jev/jev-latest
    decision_weight: 10
    k: 100
    state_format: |
      Search query:
      {query}

      Candidate product:
      Title: {title}
      Description: {description}
```

1 - https://docs.typesafe.ai/primitives/noul
