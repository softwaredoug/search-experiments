# SID agents

Cheat at Search implements an Agent backed by [SID](http://sid.ai). SID trains
a model specifically targeted at search, and we want to benchmark it.

SID documents aren't likely in your training data, refer to [SID docs](https://platform.sid.ai/docs/build).
to orient yourself when working with SID. You can also orient yourself to
Cheat at Search's [SIDAgent](https://github.com/softwaredoug/cheat-at-search/blob/main/cheat_at_search/agent/sid_agent.py)

SID Config example:


```yaml
strategy:
  name: agentic_bm25_ecommerce
  type: agentic
  params:
    model: sid/sid-1
    system_prompt: |
      You take user search queries and use a search tool to find products.

      Look at the search tools you have, their limitations, how they work, etc when forming your plan.

      Finally return results to the user per the SearchResults schema, ranked best to worst.

      Return exactly 10 results ranked best to worst.

      It's very important you consider carefully the correct ranking as you'll be evaluated on
      how close that is to the average shoppers ideal ranking.
    search_tools:
      - bm25:
      - minilm:
      - read:
```

- SID requires a `read` tool that accepts a stable document ID and returns that
  document. Add it to each SID strategy's `search_tools` list, as in the example.
- Use `typing.Annotated` for model-facing tool arguments. Its second value describes
  the argument; an optional third value provides an example used in SID's generated
  `<tool_call>` instructions. The BM25 and `read` tools include examples.
