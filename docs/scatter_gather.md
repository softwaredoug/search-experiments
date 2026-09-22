# The scatter gather strategy

Scatter gather is a search strategy. Currently it only supports the WANDS dataset.

Scatter gather utilizes a series of agents programatically. See docs/agentic_prd.md. All parameters
supported by agents, would be supported here (implementation hint, both rely on the same core Agent class). That should already be distinct from the agentic strategy. 

0. Select - An agent uses tools to gather the best N categories to search within
1. Scatter - A series of agents per category searches within those categories, gathers the best results within those
categories
2. Gather - An agent takes each candidate set and produces the top N most relevant results. IE we use an LLM to rerank the top results from each category and produce a final ranked list.

The config looks like this:

```
strategy:
  name: agentic_bm25_e5_ecommerce_gpt5
  type: scatter_gather_wands
  params:
    model: gpt-5
    reasoning: medium
    agents:
      select:
        system_prompt: |
          Use the provided tools to find the best categories and subcategories to search within.
        search_tools:
          - bm25_wands
          - top_categories
      scatter:
        system_prompt: |
          Use the provided tools to find the best results within the assigned category.
        search_tools:
          - bm25_wands_prefiltered
          - e5_base_v2_wands_prefiltered
      gather:
        system_prompt: |
          Given the candidate search results you've been provided, return the most relevant results to the users query
    plan:
      - select: "Find the best categories to search within for the query: {query}"
      - scatter: "For each category, find the best results within that category. Category: {category}"
      - gather: "Given the results from each category, return the most relevant results to the users query: {query}. Here are the results from each category: {results_by_category}"
```

Notice the `plan` - similar to agentic workflow plan. However, what's different is that the order must be select, scatter, gather.

Throw an error if these steps are not present, or in the wrong order.

Also throw an error if these steps are not configured.

Throw an error if gather has search tools

Throw an error if scatter uses a non "prefiltered" search tool - it should have a limited 
window into the corpus.

## New tools

Some new tools are required to support this strategy.

- top_categories - return the top N categories for the corpus
- bm25_wands_prefiltered - when created, a category filter is applied to the bm25 index, so that only results within that category are returned. The tool takes a category as input, and returns the top results within that category.
- e5_base_v2_wands_prefiltered - same as above, but with the e5 embedding index instead of bm25.

## Change the filtering column

The default for prefiltered is to filter by the 'category' column. However, you can change to another column in the corpus. IE

"cat_subcat" has both category and subcategory. We can rewrite the plan above using this:


    agents:
      select:
        system_prompt: |
          Use the provided tools to find the best categories + subcategories to search within.
        search_tools:
          - bm25_wands
          - top_categories
              column: cat_subcat
      scatter:
        system_prompt: |
          Use the provided tools to find the best results within the assigned category + subcategory.
        search_tools:
          - bm25_wands_prefiltered
              column: cat_subcat
          - e5_base_v2_wands_prefiltered
              column: cat_subcat
      gather:
        system_prompt: |
          Given the candidate search results you've been provided, return the most relevant results to the users query

Trying to create a scatter gather strategy for a column not in the corpus creates an error.
