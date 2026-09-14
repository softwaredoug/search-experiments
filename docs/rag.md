# RAG Search Strategy

The `rag` strategy uses an LLM to rewrite the user's request, then sends the
rewritten request to one existing retrieval tool. The retrieved documents are
returned to the normal benchmark evaluator.

This is the current one-turn benchmark form of RAG:

```text
user query -> query-generation LLM -> rewritten query -> retrieval tool -> results
```

It does not yet generate a natural-language answer. The output is still a
ranked list of documents, so it can be evaluated with the same NDCG or MRR
metrics as BM25 and embedding strategies.

## Configuration

```yaml
strategy:
  name: rag_bm25_ecommerce
  type: rag
  params:
    model: gpt-5-mini
    reasoning: medium
    system_prompt: |
      Given the user's search request, generate one concise query for the
      search tool. Return only a query that will retrieve relevant documents.
    search_tools:
      - bm25:
```

`search_tools` must contain exactly one retrieval tool. The tool is built by
the same `build_search_tools` registry used by agentic strategies, so RAG can
use the existing BM25, embedding, WANDS, and guarded tool implementations.

The difference from an agentic strategy is execution. An agentic strategy
gives tools to the LLM and lets it decide which calls to make. RAG makes one
structured LLM call to produce a query, then calls the configured retrieval
tool directly.

## Execution

For each dataset query:

1. The query is sent to the LLM with `system_prompt`.
2. The LLM returns a structured `{query: "..."}` response.
3. The generated query is passed to the configured tool with `top_k=k`.
4. Tool result IDs are mapped back to corpus rows.
5. The rows and retrieval scores are evaluated by the standard runner.

The configured tool must return the same result shape as the existing search
tools: a list of dictionaries containing at least `id`, and optionally
`score`, `title`, and `description`.

## Running

```bash
uv run run \
  --strategy configs/example/rag_bm25.yml \
  --dataset wands \
  --num-queries 100
```

The strategy can also be used with `uv run query` and `uv run diff` because it
implements the standard `SearchStrategy.search` interface.
