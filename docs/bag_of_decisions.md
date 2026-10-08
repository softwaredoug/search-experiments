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
      system_prompt: |
        You are a search relevance decision engine. You will be given a query, and your job is to generate a list of yes/no questions that can be used to determine if a document is relevant to the query.
      prompt: |
        For the given query, please generate a list of yes/no questions where the affirmative 
        indicates the document is relevant to the query
        
        Here's the query:
        
        {query}
      model: gpt-5
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

The `state` of jev should be the document (see `state_format`). It uses a python format string, where {title} means to pull from the title column 
of the corpus dataframe, etc.

We will rerank the top `decision_engine.k` documents from the retrieval engine.

1 - https://docs.typesafe.ai/primitives/noul
