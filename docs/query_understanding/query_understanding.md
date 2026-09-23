# Query understanding strategy

In query understanding strategy, for a given dataset, we assume the corpus has a specific column that 
would be ideal to filter to. I'll generally refer to "category" here as that column, though this should be configured.

The goal is to use an LLM or other approach prior to searching, to identify the best category for the query, then return
the search results filtered or boosting results from that category (depending on configuration)

This strategy focuses on manageable vocabulary sizes, like dozens. Not 1000s.

Basic example

```yaml
strategy:
  name: query_understanding_category
  type: query_understanding
  params:
    reasoning: medium
    categorize:
      field: category
      enrichment_engine:
        type: llm_single
        model: gpt-5
        params:
          prompt: |
            For the given query, please generate the appropriate {field} to filter to zero-in on
            the most relevant results.

            If you're unsure, or the query is ambiguous, return "Unknown" as the category.

            Here's the query:

            {query}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]  # Baseline BM25
        boost_matches: 10  # How much to boost results that match the category returned by the enrichment engine
         
```

## Enrichment engine

The 'enrichment engine' decides how queries are resolved to categories.

We should assume all enrichers take the query and return a list. Even the "single" ones. This just makes it easier
to handle the output.

A list of enrichment engines can found documented at [enrichment_engines](./enrichment_engines.md)

## Enrichment ground truth and evaluation

A script exists to evaluate the enrichment engine against a corpus with judgments.

```
uv run query_classification --strategy configs/ecom_class/ecom_query_understanding.yml --dataset wands
```

With specific information on how eval works in [enrichment_engines_eval](./enrichment_engines_eval.md)

## Retrieval engine

The retrieval engine then decides what to do with the output of the enrichment engine. Below
details the options for BM25 enrichment

### bm25_boosted 

Run a BM25 search over the corpus (see the bm25 strategy for details) and boost results that match the category returned by the enrichment engine.

The ^ operator is used as a weight on the title / description fields.

Then, within these results, search the category of the query by phrase. A rough

```python
# ****
# If there's a category, boost that by a constant amount
for category in classified.categories:
    tokenized_category = snowball_tokenizer(category)
    category_match = np.ones(len(self.index))
    if tokenized_category:
        category_match = self.index['category_snowball'].array.score(tokenized_category) > 0
    bm25_scores[category_match] += self.category_boost
```


### bm25_hierarchy_boosted : Hierarchy boosted

Split the category into a hierarchy using the / separator. Then boost results that match the category at each level of the hierarchy, with a decaying boost for each level.

A search for 

"foo / bar / baz" would boost results that match "foo" the most, then "foo / bar" less, then "foo / bar / baz" the least.

Since we might have multiple classifications for the queries, we do this for each classification, and sum the boosts.

The configuration uses `boost_matches` for the level-zero boost and `decay` for
the multiplier applied at each subsequent level. For example, with
`boost_matches: 10` and `decay: 0.5`, the boosts for `foo / bar / baz` are
`10`, `5`, and `2.5`.

```yaml
retrieval_engine:
  base: bm25_hierarchy_boosted
  params:
    fields: [title^9.4, description^4]
    boost_matches: 10
    decay: 0.5
```
