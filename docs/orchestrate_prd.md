# Orchestrate topology

We introduce here teh idea of a "topology" of agents that defines roughly a specifice way multiple
agents are organized together.

This is configured via the topology param, ie topology: orchestrate

This is in contrast to the default "direct" topology, which drives a single agent call / loop.

Instead of simply driving a single agent with search tools, the orchestrator has an outer planning agent (the orchestrator)
and inner search agents that do the search tools

Basic example:

```yaml
strategy:
  name: orchestrate_bm25_ecommerce
  type: agentic
  params:
    topology: orchestrate
    model: gpt-5
    reasoning: medium
    system_prompt: |
      You take user search queries and use a search tool to find products.

      Look at the search tools you have, their limitations, how they work, etc when forming your plan.

      Finally return results to the user per the SearchResults schema, ranked best to worst.

      Gather results until you have 10 best matches you can find. It's important to return at least 10.

      It's very important you consider carefully the correct ranking as you'll be evaluated on
      how close that is to the average shoppers ideal ranking.
    search_tools:
      - bm25:
      - minilm:
```

## Topology defined

Below we define the agent topoology here

### Orchestrator

The orchestrator agent creates a plan and calls subagents to execute that plan

It has access to an (internal only) TaskTool that it uses to delegate search tasks to the search subagents

It returns a set of ranked search results

### Subagents

The subagents perform similarly to the existing agentic agents. They have access to search_tools as configured

### Tool guards supported

Tool guards + agent state are supported, and passed from

orchestrator -> sub agent tool (TaskTool) -> search tools


### Stop / retry / validators

Here we still have stop / retry / validators at the outermost orchestrator layer.
