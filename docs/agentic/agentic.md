# Agentic search strategies

One key type of strategy here is the 'agentic' strategy. 

Agent as in an LLM that can call tools (in our case search tools) to satsify the query and return top 
N results

The basic idea is to execute a set of simple search tools. Then drive an agentic loop until exhuasted. Finally
producing a ranked list of search results.

Basic example:

```yaml
strategy:
  name: agentic_bm25_ecommerce
  type: agentic
  params:
    model: gpt-5
    reasoning: medium
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
```

This calls OpenAI with

- the system prompt here
- the query being searched for as the user prompt. 
- a set of simple search tools that the agent can call to gather info. In this case, BM25 and minilm embedding search. The agent can call these tools with different queries, etc to gather info. The agentic loop continues until the agent decides to stop (or max iterations is reached). Then the final ranked list of results is returned and evaluated.

The standard `agentic` strategy's `AgenticSearchResults.ranked_results` response field is constrained to
exactly 10 document IDs using Pydantic `min_length=10` and `max_length=10`. This is currently a
fixed response-schema contract; no `num_results` validator is needed to enforce the count. The
separate `scatter_gather_wands` strategy may use the unconstrained `SearchResults` model for its
intermediate outputs.

Implementation note: agentic strategies now use `OpenAIAgent` from cheat-at-search for the tool-calling loop. The harness applies validators + stop conditions around that agent loop.

### Agent state

Everytime we start a search, we initiate "agent_state". That's like a scratchpad for the agentic loop, harness, and tools to track state and prevent illegal operations. See more in "Tool guards" below. 

## Agentic trace folders

Agentic runs record tool calls and outputs under a working folder rooted at:

```
~/.search-experiments/agentic/<dataset>/<strategy_name>/<timestamp>
```

Each query has its own folder and log. Logs record timed model requests and retries, tool calls,
validator execution, and per-result Jev evaluation. Active queries emit a heartbeat every 30 seconds
with their current phase and elapsed time, plus a worker stack event after two minutes. Request and
result payload contents are not added to these diagnostic events.

## Few shop options

Options to add few-shot examples to the system prompt. These can be configured in the yml as well, and are added to the system prompt before the agentic loop starts.

It shows up as:

```
  params:
    model: gpt-5
    system_prompt: |
       ...
    few_shot:
       - sample_judgments: 10

```

### Few shot, random evaluated results

The option

```
    few_shot:
       - sample_judgments:
            num_rows: 10
``` 

Should sample the judgments to add 10 examples of queries, their products, and whether they're relevant or not.

When sampling, try to balance the number of relevant and non-relevant examples (ie exemplars of each relevance grade).

This should use the core columns expected on any data (title, description) and not other data.

To add custom columns

```
    few_shot:
       - sample_judgments:
           num_rows: 10
           columns:
            - category
            - price
```

And those would be inclulded in the prompt. If this corpus does not have this, then it should throw an error.

## Search Tools

Search Tools or just 'tools' are configured in the agentic strategy, as above the 'bm25' and 'minilm' tools are
specified

More details can be found in [Tools Documentation](tools.md)

### Top k

To not flood the agent's context, at most the agent can request 20 results.

### Tool guards

Tools can have guards that reject calls with an error. That can be based on the parameters themselves, or the agent state.

IE here's one that rejects repeat queries too similar to previous runs:

```
      - e5_base_v2:
          guards:
            - disallow_repeated_queries
```

### Implementation

Tools exist in a tools registry that produce a function suitable for use by the agent.

## Conditionals, Validators, Judges etc

Some params require wrapping the agentic loop itself to give feedback to the agent

IE a judge or validator or something else that sits outside, checks the results that came back and fails
if they don't meet the criteria. Then it can append a prompt to the context and call the agent again.

More details can be found in [Conditionals Documentation](conditionals.md)

## Plan through list of agents

A list of agents is possible

An agentic strategy might look like this:


```
  params:
    model: gpt-5
    reasoning: medium
    agents:
      planning:
        system_prompt: |
          You take user search queries and use a search tool to find products.

          Look at the search tools you have, their limitations, how they work, etc when forming your plan.

          Finally return results to the user per the SearchResults schema, ranked best to worst.

          Gather results until you have 10 best matches you can find. It's important to return at least 10.

          It's very important you consider carefully the correct ranking as you'll be evaluated on
          how close that is to the average shoppers ideal ranking.
        search_tools:
          - delegate_task:

      search:
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

      eval:
        system_prompt: |
          You take user search queries and use a search tool to find products.

          Look at the search tools you have, their limitations, how they work, etc when forming your plan.

          Finally return results to the user per the SearchResults schema, ranked best to worst.

          Gather results until you have 10 best matches you can find. It's important to return at least 10.

          It's very important you consider carefully the correct ranking as you'll be evaluated on
          how close that is to the average shoppers ideal ranking.
        search_tools:
          - delegate_task:
    plan:
      - planning: plan how to best search for {query}
      - search: find the most relevant results for {query}
      - eval: evaluate how relevant the results are for {query}
```

Throughout this whole process, the context is identical.

However, after one agent completes, the system prompt would be patched to the next agent's system prompt, and the tools would be patched to the next agent's tools.

This means that three agents would run, with different system prompts + tools

plan dictates the order of execution of the agents. The output of one agent does not get passed as input to the next agent, but the context (including agent state) is shared across all agents. So they can communicate implicitly through that.

This is like switching between Plan <-> Build mode in coding agents. Except we're doing it sequentially.

The user prompt is specified in plan, ie above planning agent gets "plan how to best search for {query}" as user prompt, and the search agent gets "find the most relevant results for {query}" as user prompt, etc. Replacing {query} with the actual query being searched for.

Params like retrying, stopping, etc would all occur WITHIN this, so logically this is 

```
for work_item in plan:
    system_prompt = agents[agent_name].system_prompt
    user_prompt = plan[agent_name].user_prompt
    tools = agents[agent_name].tools
    while not stop_condition:
        # Call OpenAIAgent + chat
        
        if stop_condition(output):
            break
        else:
            # append stopper/validator prompt before retrying
```
