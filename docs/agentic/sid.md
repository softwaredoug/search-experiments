# SID agents

Cheat at Search implements two SID-backed agents. SID-1 is provided by
`SIDLegacyAgent`; SID-2 is provided by `SIDAgent`. Both are configured through
the agentic strategy's `model` parameter.

SID's APIs evolve quickly, so refer to the [SID documentation](https://platform.sid.ai/docs/build)
when working with these agents. The pinned Cheat at Search implementations are
[SIDAgent](https://github.com/softwaredoug/cheat-at-search/blob/0aa58f8c61ca5c0184a29fadee1370fc22c993cf/cheat_at_search/agent/sid_agent.py)
and [SIDLegacyAgent](https://github.com/softwaredoug/cheat-at-search/blob/0aa58f8c61ca5c0184a29fadee1370fc22c993cf/cheat_at_search/agent/sid_legacy_agent.py)
implementations.

## SID-1

SID-1 uses `SIDLegacyAgent` and requires an explicit `read` tool configured
alongside one or more search tools:

```yaml
strategy:
  name: agentic_bm25_ecommerce
  type: agentic
  params:
    model: sid/sid-1 # SIDLegacyAgent
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

## SID-2

SID-2 uses `SIDAgent`. It automatically adds its cache-backed `read` tool and
the required `report_helpful_ids` tool, so do not include `read` in
`search_tools`. Configure at least one backend search tool:

```yaml
strategy:
  name: agentic_bm25_ecommerce_sid2
  type: agentic
  params:
    model: sid/sid-2 # SIDAgent
    system_prompt: |
      You take user search queries and use the available search tools to find products.
      Return the most relevant document IDs using report_helpful_ids.
    search_tools:
      - bm25:
      - minilm:
```

The SID-2 agent maps these tools' document descriptions and IDs into the format
its episode cache requires. For custom document-returning tools, return stable
IDs and non-empty document text so SID-2 can read and report the results.
BM25-family tools are exposed to SID-2 as `text_search`; embedding-family tools
are exposed as `search`. If multiple tools from one family are configured, the
first gets the canonical SID name and later tools retain their regular names so
each exposed tool name stays unique.

Both agents support the same loop/result interface, but use different SID API
protocols and have different `read` handling. Keep each model's tool list
appropriate to its version.

Use `typing.Annotated` for model-facing tool arguments. Its second value
describes the argument; an optional third value provides an example used in
SID's generated tool-call instructions. The BM25 and `read` tools include
examples.
