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
        confidence_threshold: 0.7
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
in one request for each candidate document. The Noul probability is `P(yes)`; each
decision contributes its probability only when it is strictly greater than
`confidence_threshold`. The qualifying probabilities are summed and multiplied
by `decision_weight`, then added to the document's retrieval score.

The `state` of Jev is rendered from `state_format`, which can reference corpus
columns such as `{title}` and the current `{query}`.

We will rerank the top `decision_engine.reranker.k` documents from the retrieval engine.

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
    confidence_threshold: 0.7
    k: 100
    state_format: |
      Search query:
      {query}

      Candidate product:
      Title: {title}
      Description: {description}
```

1 - https://docs.typesafe.ai/primitives/noul
